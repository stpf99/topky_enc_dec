#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.2 — SYMULACJA OPTYCZNA + KOMPRESJA DCT/VLC      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v1.2 zmiany:                                                              ║
║  - Kompresja bloków DCT (zamiast surowego float16)                        ║
║  - Dead-zone quantization na osobnych kanałach Y i CbCr                   ║
║  - Zig-Zag scan dla lepszego upakowania wysokich częstotliwości           ║
║  - Adaptacyjne VLC: wybór między Sparse a RLE w zależności od zawartości   ║
║  - Zmiana formatu strumienia (Length-prefixed dla bloków) → NIEKOMPATYBILNY║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.fftpack import dct, idct
import zstandard as zstd
import struct
import argparse
import time
from functools import lru_cache
from contextlib import contextmanager

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

_MAGIC      = b'PH03'   # Zmieniony magic — niekompatybilny z v1.0/v1.1
_ZSTD_LEVEL = 19

# ═══════════════════════════════════════════════════════════════════════════════
# KONWERSJA KOLORÓW  RGB ↔ YCbCr
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_ycbcr(frame: np.ndarray) -> np.ndarray:
    f = frame[..., :3].astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B
    return np.stack([Y, Cb, Cr], axis=-1)

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    R = Y               + 1.402    * Cr
    G = Y - 0.344136   * Cb - 0.714136 * Cr
    B = Y + 1.772      * Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)

# ═══════════════════════════════════════════════════════════════════════════════
# NOWOŚĆ v1.2: SILNIK KOMPRESJI (Zig-Zag, DCT, Dead-Zone, VLC)
# ═══════════════════════════════════════════════════════════════════════════════

_zigzag_cache = {}

def get_zigzag_indices(n: int) -> np.ndarray:
    if n in _zigzag_cache: return _zigzag_cache[n]
    idx_mat = np.arange(n * n).reshape(n, n)
    out = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for r in range(min(s, n - 1), max(-1, s - n), -1):
                out.append(idx_mat[r, s - r])
        else:
            for r in range(max(0, s - n + 1), min(s + 1, n)):
                out.append(idx_mat[r, s - r])
    _zigzag_cache[n] = np.array(out, dtype=np.uint32)
    return _zigzag_cache[n]

def apply_dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.astype(np.float32).T, norm='ortho').T, norm='ortho')

def apply_idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.astype(np.float32).T, norm='ortho').T, norm='ortho')

def deadzone_quantize(dct_block: np.ndarray, Q: float, deadzone_ratio: float = 0.35) -> np.ndarray:
    if Q <= 0: return dct_block.astype(np.int16)
    delta = Q * deadzone_ratio
    abs_block = np.abs(dct_block)
    q_block = np.sign(dct_block) * np.floor((np.maximum(0.0, abs_block - delta) / Q) + 0.5)
    return q_block.astype(np.int16)

def encode_rle_int16(arr: np.ndarray) -> bytes:
    out, rle_data, flat, n, i = bytearray(), bytearray(), arr.flatten(), len(arr.flatten()), 0
    while i < n:
        val = int(flat[i])
        if val == 0:
            run_start = i
            while i < n and int(flat[i]) == 0: i += 1
            run_len = i - run_start
            while run_len > 0:
                chunk = min(run_len, 126); rle_data.append(0x80 + chunk - 1); run_len -= chunk
        else:
            if 0 <= val <= 127: rle_data.append(val)
            else: rle_data.append(0xFE); rle_data.extend(struct.pack('>h', val))
            i += 1
    out.extend(struct.pack('>H', len(rle_data))); out.extend(rle_data)
    return bytes(out)

def decode_rle_int16(data: bytes, expected_len: int, offset: int = 0) -> tuple:
    rle_len = struct.unpack_from('>H', data, offset)[0]; offset += 2
    rle_end = offset + rle_len; out = []; i = offset
    while len(out) < expected_len and i < rle_end:
        b = data[i]; i += 1
        if b == 0xFE: val = struct.unpack_from('>h', data, i)[0]; i += 2; out.append(val)
        elif b >= 0x80: out.extend([0] * ((b & 0x7F) + 1))
        else: out.append(b)
    while len(out) < expected_len: out.append(0)
    return np.array(out[:expected_len], dtype=np.int16), rle_end

def encode_sparse_int16(arr: np.ndarray) -> bytes:
    flat, n = arr.flatten(), len(arr.flatten())
    nz_mask = flat != 0; nz_count = int(np.sum(nz_mask))
    out = bytearray()
    if nz_count == 0: out.append(0x00); return bytes(out)
    if nz_count > n // 4: return None # Zbyt gęste na Sparse
    if nz_count < 255: out.append(nz_count)
    else: out.append(0xFF); out.extend(struct.pack('>H', nz_count))
    for pos in np.where(nz_mask)[0]:
        if n > 255: out.extend(struct.pack('>H', pos))
        else: out.append(pos)
        out.extend(struct.pack('>h', int(flat[pos])))
    return bytes(out)

def decode_sparse_int16(data: bytes, expected_len: int, offset: int = 0) -> tuple:
    out = np.zeros(expected_len, dtype=np.int16)
    nz_count = data[offset]; i = offset + 1
    if nz_count == 0xFF: nz_count = struct.unpack_from('>H', data, i)[0]; i += 2
    use_2b_pos = expected_len > 255
    for _ in range(nz_count):
        if use_2b_pos: pos = struct.unpack_from('>H', data, i)[0]; i += 2
        else: pos = data[i]; i += 1
        val = struct.unpack_from('>h', data, i)[0]; i += 2
        if pos < expected_len: out[pos] = val
    return out, i

def pack_photonic_block(block_3d: np.ndarray, bs: int, q_y: float, q_c: float, deadzone: float, is_intra: bool = False) -> bytes:
    zz = get_zigzag_indices(bs)
    out = bytearray()
    for c in range(3):
        Q = q_y if c == 0 else q_c
        ch = block_3d[:, :, c].astype(np.float32)
        
        # Dla Intra Y odliczamy 128, żeby wyśrodkować wokół zera
        if is_intra and c == 0: ch = ch - 128.0
        
        q_block = deadzone_quantize(apply_dct2(ch), Q, deadzone)
        flat_zz = q_block.flatten()[zz]
        
        # Wybór między Sparse a RLE
        sparse_data = encode_sparse_int16(flat_zz)
        if sparse_data is not None:
            out.append(0x02) # Marker Sparse
            out.extend(sparse_data)
        else:
            out.append(0x03) # Marker RLE
            out.extend(encode_rle_int16(flat_zz))
    return bytes(out)

def unpack_photonic_block(data: bytes, bs: int, q_y: float, q_c: float, is_intra: bool = False) -> tuple:
    zz = get_zigzag_indices(bs)
    n = bs * bs
    out_block = np.zeros((bs, bs, 3), dtype=np.float32)
    offset = 0
    
    for c in range(3):
        Q = q_y if c == 0 else q_c
        marker = data[offset]; offset += 1
        
        if marker == 0x02:
            flat_zz, offset = decode_sparse_int16(data, n, offset)
        else:
            flat_zz, offset = decode_rle_int16(data, n, offset)
            
        flat = np.zeros(n, dtype=np.int16)
        flat[zz] = flat_zz
        out_block[:, :, c] = apply_idct2(flat.reshape(bs, bs).astype(np.float32) * Q)
        
        if is_intra and c == 0: out_block[:, :, c] += 128.0

    return out_block, offset

# ═══════════════════════════════════════════════════════════════════════════════
# FIZYKA: INTERFERENCJA I KORELACJA OPTYCZNA
# ═══════════════════════════════════════════════════════════════════════════════

def shift_block(block: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(block)
    h, w = block.shape[:2]
    sy = slice(max(0, -dy), min(h, h - dy))
    sx = slice(max(0, -dx), min(w, w - dx))
    dy_ = slice(max(0,  dy), min(h, h + dy))
    dx_ = slice(max(0,  dx), min(w, w + dx))
    result[dy_, dx_] = block[sy, sx]
    return result

def optical_correlate(prev_block: np.ndarray, curr_block: np.ndarray) -> tuple:
    prev_y = prev_block[..., 0] if prev_block.ndim == 3 else prev_block
    curr_y = curr_block[..., 0] if curr_block.ndim == 3 else curr_block
    corr = np.fft.irfft2(np.fft.rfft2(prev_y) * np.conj(np.fft.rfft2(curr_y[::-1, ::-1], s=prev_y.shape)))
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = prev_y.shape
    dy = max_idx[0] if max_idx[0] < h // 2 else max_idx[0] - h
    dx = max_idx[1] if max_idx[1] < w // 2 else max_idx[1] - w
    e_prev = np.sqrt(np.sum(prev_y ** 2) + 1e-6)
    e_curr = np.sqrt(np.sum(curr_y ** 2) + 1e-6)
    ncc = float(np.clip(corr[max_idx] / (e_prev * e_curr), -1.0, 1.0))
    return dy, dx, ncc

def extract_interference_delta(prev_block: np.ndarray, curr_block: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return curr_block - shift_block(prev_block, dy, dx)

class PhotonicCodec:
    def __init__(self, brush_size: int = 64, energy_threshold: float = 0.85,
                 delta_threshold: float = 5.0, q_y: float = 8.0, q_c: float = 20.0, 
                 deadzone: float = 0.35, fast_skip_thr: float = 0.1):
        self.bs = brush_size
        self.energy_thr = energy_threshold
        self.delta_thr = delta_threshold
        self.q_y = q_y
        self.q_c = q_c
        self.deadzone = deadzone
        self.skip_thr = fast_skip_thr # Próg SAD na piksel do całkowitego pominięcia bloku
        self.prev_wave = None

    def encode_frame(self, frame_rgb: np.ndarray) -> list:
        paths = []
        bs = self.bs

        # Błyskawiczne wyodrębnienie kanału Y (luminancji) dla CAŁEGO obrazu
        curr_Y = np.dot(frame_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114])

        if self.prev_wave is None:
            # I-FRAME: Pełne pole falowe (musi być w całości)
            wave3 = rgb_to_ycbcr(frame_rgb)
            paths.append({'type': 0, 'data': wave3.tobytes()})
            self.prev_wave = wave3
            return paths

        # --- PRE-FILTER: Szybka mapa zmian (SAD Map) ---
        diff_Y = np.abs(curr_Y - self.prev_wave[..., 0])
        h, w = diff_Y.shape
        h_blocks = h // bs
        w_blocks = w // bs
        
        # Wektoryzowane sumowanie różnic w blokach (bez pythonowych pętli)
        valid_diff = diff_Y[:h_blocks*bs, :w_blocks*bs]
        sad_map = valid_diff.reshape(h_blocks, bs, w_blocks, bs).sum(axis=(1, 3))
        
        # Flaga bloków, które są absolutnie statyczne (np. zmiana < 0.1 na piksel)
        static_mask = sad_map < (bs * bs * self.skip_thr)

        # Optymalizacja: jeśli klatka jest w 100% statyczna (np. pauza w wideo)
        if np.all(static_mask):
            return [] 

        # Konwersja YCbCr wykonywana TYLKO jeśli wiemy, że coś się zmieniło
        wave3 = rgb_to_ycbcr(frame_rgb)

        # Główna pętla PO TYLKO ZMIENIAJĄCYCH SIĘ BLOKACH
        for row in range(h_blocks):
            for col in range(w_blocks):
                # SZYBKI SKIP: Zero pracy dla statycznej części pola fazowego!
                if static_mask[row, col]:
                    continue

                y, x = row * bs, col * bs
                bh, bw = bs, bs
                brush = wave3[y:y+bh, x:x+bw]
                prev_brush = self.prev_wave[y:y+bh, x:x+bw]

                dy, dx, ncc = optical_correlate(prev_brush, brush)

                if ncc >= self.energy_thr:
                    delta = extract_interference_delta(prev_brush, brush, dy, dx)
                    delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
                    
                    if np.any(delta_mask):
                        sparse_delta = delta * delta_mask[..., np.newaxis]
                        packed = pack_photonic_block(sparse_delta, bh, self.q_y, self.q_c, self.deadzone)
                        paths.append({
                            'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy, 'bh': bh, 'bw': bw,
                            'data': packed
                        })
                else:
                    packed = pack_photonic_block(brush, bh, self.q_y, self.q_c, self.deadzone, is_intra=True)
                    paths.append({
                        'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0, 'bh': bh, 'bw': bw,
                        'data': packed
                    })

        # Obsługa brzegów obrazu (resztki niekwadratowe na krawędziach)
        if h % bs != 0 or w % bs != 0:
            for y in range(0, h, bs):
                for x in range(0, w, bs):
                    if x < w_blocks * bs and y < h_blocks * bs: continue # Pomijaj kwadratowe
                    
                    bh, bw = min(bs, h - y), min(bs, w - x)
                    brush = wave3[y:y+bh, x:x+bw]
                    prev_brush = self.prev_wave[y:y+bh, x:x+bw]
                    
                    dy, dx, ncc = optical_correlate(prev_brush, brush)
                    if ncc >= self.energy_thr:
                        delta = extract_interference_delta(prev_brush, brush, dy, dx)
                        delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
                        if np.any(delta_mask):
                            sparse_delta = delta * delta_mask[..., np.newaxis]
                            packed = sparse_delta.astype(np.float16).tobytes()
                            paths.append({'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy, 'bh': bh, 'bw': bw, 'data': packed})
                    else:
                        packed = brush.astype(np.float16).tobytes()
                        paths.append({'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0, 'bh': bh, 'bw': bw, 'data': packed})

        self.prev_wave = wave3
        return paths

    def decode_frame(self, paths: list, h: int, w: int) -> np.ndarray:
        if self.prev_wave is None:
            self.prev_wave = (
                np.frombuffer(paths[0]['data'], dtype=np.float32).reshape(h, w, 3)
            )
            return ycbcr_to_rgb(self.prev_wave)

        # NOWOŚĆ: Jeśli lista ścieżek jest pusta, klatka była w 100% statyczna
        if not paths:
            return ycbcr_to_rgb(self.prev_wave)

        new_wave = self.prev_wave.copy()

        for path in paths:
            x, y, bh, bw = path['x'], path['y'], path['bh'], path['bw']
            is_square = (bh == bw and bh == self.bs)

            if path['type'] == 1:
                dy, dx = path['dy'], path['dx']
                if is_square:
                    delta, _ = unpack_photonic_block(path['data'], bh, self.q_y, self.q_c)
                else:
                    delta = np.frombuffer(path['data'], dtype=np.float16).reshape(bh, bw, 3).astype(np.float32)
                
                shifted = shift_block(new_wave[y:y+bh, x:x+bw], dy, dx)
                new_wave[y:y+bh, x:x+bw] = shifted + delta

            elif path['type'] == 2:
                if is_square:
                    brush, _ = unpack_photonic_block(path['data'], bh, self.q_y, self.q_c, is_intra=True)
                else:
                    brush = np.frombuffer(path['data'], dtype=np.float16).reshape(bh, bw, 3).astype(np.float32)
                    
                new_wave[y:y+bh, x:x+bw] = brush

        self.prev_wave = new_wave
        return ycbcr_to_rgb(new_wave)

# ═══════════════════════════════════════════════════════════════════════════════
# STRUMIENIE I/O (Zaktualizowane pod Variable-Length Payloads dla DCT/VLC)
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_paths(paths: list) -> bytes:
    out = bytearray()
    out.extend(struct.pack('>H', len(paths)))
    for p in paths:
        out.append(p['type'])
        if p['type'] == 0:
            # I-frame (cały obraz float32) - prefix długości
            out.extend(struct.pack('>I', len(p['data'])))
            out.extend(p['data'])
        else:
            out.extend(struct.pack('>HHhhHH', p['x'], p['y'], p['dx'], p['dy'], p['bh'], p['bw']))
            # Kluczowa zmiana: prefix z długością dla zmiennych danych VLC (zamiast sztywnego float16)
            out.extend(struct.pack('>I', len(p['data'])))
            out.extend(p['data'])
    return bytes(out)


def deserialize_paths(data: bytes) -> list:
    off = 0
    count = struct.unpack_from('>H', data, off)[0]; off += 2
    paths = []
    for _ in range(count):
        t = data[off]; off += 1
        if t == 0:
            size = struct.unpack_from('>I', data, off)[0]; off += 4
            paths.append({'type': 0, 'data': data[off:off+size]})
            off += size
        else:
            x, y, dx, dy, bh, bw = struct.unpack_from('>HHhhHH', data, off); off += 12
            # Odczyt zmiennej długości bloba
            size = struct.unpack_from('>I', data, off)[0]; off += 4
            paths.append({
                'type': t, 'x': x, 'y': y, 'dx': dx, 'dy': dy,
                'bh': bh, 'bw': bw, 'data': data[off:off+size]
            })
            off += size
    return paths

@contextmanager
def photonic_writer(path, w, h, fps):
    f = open(path, 'wb')
    f.write(_MAGIC)
    f.write(struct.pack('>HHf', w, h, fps))
    c = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
    try:
        class W:
            def __init__(self): self.f = f; self.c = c
            def write(self, paths):
                raw  = serialize_paths(paths)
                comp = self.c.compress(raw)
                # Czysty bytes I/O: Sklejamy nagłówek i payload w RAM w jeden ciągły blok
                # To gwarantuje dokładnie JEDNO wywołanie systemowe (syscall) na klatkę,
                # bez żadnego narzutu flush(). Blok >8KB automatycznie wymusi zapis na dysk.
                frame_blob = struct.pack('>I', len(comp)) + comp
                self.f.write(frame_blob)
            def close(self): self.f.close()
        yield W()
    finally:
        if not f.closed: f.close()


@contextmanager
def photonic_reader(path):
    f = open(path, 'rb')
    d = zstd.ZstdDecompressor()
    try:
        m = f.read(4)
        if m != _MAGIC:
            raise ValueError(f"Zły format pliku (magic={m!r}, oczekiwano {_MAGIC})")
        w, h, fps = struct.unpack('>HHf', f.read(8))
        class R:
            def __init__(self):
                self.f = f; self.d = d
                self.w = w; self.h = h; self.fps = fps
            def read(self):
                sz = self.f.read(4)
                if len(sz) < 4: return None
                comp = self.f.read(struct.unpack('>I', sz)[0])
                if not comp: return None
                return deserialize_paths(self.d.decompress(comp))
            def close(self): self.f.close()
        yield R()
    finally:
        if not f.closed: f.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
# (CLI pozostaje bez zmian)

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.2] Inicjalizacja z DCT/VLC...", flush=True)

    reader = iio.imiter(inp, plugin='pyav')
    first = next(reader)
    h, w = first.shape[:2]

    codec = PhotonicCodec(
        brush_size=args.brush_size,
        energy_threshold=args.ncc_threshold,
        delta_threshold=args.delta_threshold,
        q_y=args.q_y,
        q_c=args.q_c,
        deadzone=args.deadzone
    )

    with photonic_writer(out, w, h, args.fps) as writer:
        writer.write(codec.encode_frame(first[..., :3]))
        fi = 1; t = time.time()
        for frame in reader:
            paths = codec.encode_frame(frame[..., :3])
            writer.write(paths)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS"
                      f" | Ścieżek: {len(paths):3d}", end="", flush=True)
            fi += 1

    print(f"\n[Photonic] Zapisano {fi} klatek.", flush=True)

def decode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.2] Odtwarzanie z interferencji (DCT/VLC)...", flush=True)

    with photonic_reader(inp) as reader:
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)

        codec = PhotonicCodec(
            brush_size=args.brush_size, 
            q_y=args.q_y, 
            q_c=args.q_c,
            deadzone=args.deadzone
        )
        fi = 0; t = time.time()

        while True:
            paths = reader.read()
            if paths is None: break
            rgb = codec.decode_frame(paths, reader.h, reader.w)
            w_out.write_frame(rgb)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS",
                      end="", flush=True)
            fi += 1

        w_out.close()
    print(f"\n[Photonic] Odtworzono {fi} klatek.", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.2")
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')
    
    # Optyczne
    parser.add_argument('--brush-size',      type=int,   default=32)
    parser.add_argument('--fps',             type=float, default=25.0)
    parser.add_argument('--ncc-threshold',   type=float, default=0.85)
    parser.add_argument('--delta-threshold', type=float, default=5.0)
    
    # Kompresja (NOWOŚĆ)
    parser.add_argument('--q-y',             type=float, default=8.0,  help="Kwantyzacja Y (im mniej, tym lepiej)")
    parser.add_argument('--q-c',             type=float, default=20.0, help="Kwantyzacja CbCr")
    parser.add_argument('--deadzone',        type=float, default=0.35, help="Współczynnik martwej strefy (0.0-1.0)")
    
    args = parser.parse_args()

    if args.decode:
        decode_video(args.input, args.output, args)
    else:
        encode_video(args.input, args.output, args)
