#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.4 — ANTI-CHAOS ENTROPY REDUCTION ENGINE           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Maksymalna optymalizacja struktury danych pod kompresor strumieniowy:      ║
║  - Całkowite usunięcie koordynatów X,Y (Zastąpione przez Raster Skip Map)  ║
║  - Wymuszony Padding krawędzi (Brak wycieków surowych danych float16)       ║
║  - Kompresja DCT Planar RLE również dla klatek kluczowych I-Frame           ║
║  - Filtrowanie opłacalności ruchu na poziomie SAD (Anti-Noise MV Filter)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.fftpack import dct, idct
import zstandard as zstd
import struct
import argparse
import time
from contextlib import contextmanager

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

_MAGIC      = b'PH14'
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
# ZIGZAG & DCT MATHEMATICS
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

# ═══════════════════════════════════════════════════════════════════════════════
# NOWY STRUKTURALNY PLANAR RLE (Uporządkowany strumień bez overheadu indeksów)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_planar_channel(blocks_2d_list: list, bs: int, Q: float, deadzone: float,
                           is_luma_intra: bool = False) -> bytes:
    """Pakuje listę klocków sekwencyjnie w czysty strumień RLE bez narzutu koordynatów"""
    if not blocks_2d_list:
        return struct.pack('>H', 0)

    n_blocks = len(blocks_2d_list)
    zz = get_zigzag_indices(bs)
    out = bytearray()
    out.extend(struct.pack('>H', n_blocks))

    last_dc = 0
    for blk in blocks_2d_list:
        ch = blk.astype(np.float32)
        if is_luma_intra:
            ch = ch - 128.0
        q = deadzone_quantize(apply_dct2(ch), Q, deadzone).flatten()[zz]
        
        # DC Delta Coding (Likwidacja chaosu tła)
        dc_val = int(q[0])
        dc_delta = dc_val - last_dc
        last_dc = dc_val

        ac_coeffs = q[1:]
        nz_indices = np.where(ac_coeffs != 0)[0]
        nz_count = len(nz_indices)

        # Zapisujemy nagłówek klocka: DC delta oraz jawna liczba niezerowych AC
        out.extend(struct.pack('>hH', dc_delta, nz_count))

        # Zapis niezerowych AC za pomocą dystansu (run) bez indeksów globalnych
        last_idx = -1
        for idx in nz_indices:
            run = idx - last_idx - 1
            val = int(ac_coeffs[idx])
            out.extend(struct.pack('>Hh', run, val))
            last_idx = idx

    return bytes(out)


def decode_planar_channel(data: bytes, off: int, Q: float, bs: int, is_luma_intra: bool = False) -> tuple:
    n_blocks = struct.unpack_from('>H', data, off)[0]; off += 2
    if n_blocks == 0:
        return [], off

    zz = get_zigzag_indices(bs)
    zz_inv = np.argsort(zz)
    result = []
    last_dc = 0

    for _ in range(n_blocks):
        dc_delta, nz_count = struct.unpack_from('>hH', data, off); off += 4
        current_dc = last_dc + dc_delta
        last_dc = current_dc

        flat = np.zeros(bs * bs, dtype=np.float32)
        flat[0] = float(current_dc)

        curr_ac_idx = 0
        for _ in range(nz_count):
            run, val = struct.unpack_from('>Hh', data, off); off += 4
            curr_ac_idx += run + 1
            if curr_ac_idx < bs * bs:
                flat[curr_ac_idx] = float(val)

        block_coeffs = np.zeros(bs * bs, dtype=np.float32)
        block_coeffs[zz_inv] = flat * Q
        blk = apply_idct2(block_coeffs.reshape(bs, bs))
        if is_luma_intra:
            blk += 128.0
        result.append(blk)

    return result, off

# ═══════════════════════════════════════════════════════════════════════════════
# PACK/UNPACK FRAME PLANAR
# ═══════════════════════════════════════════════════════════════════════════════

def pack_frame_planar(blocks_ycbcr: list, bs: int, q_y: float, q_c: float,
                       deadzone: float, is_intra: bool = False) -> bytes:
    y_blocks  = [b[:, :, 0] for b in blocks_ycbcr]
    cb_blocks = [b[:, :, 1] for b in blocks_ycbcr]
    cr_blocks = [b[:, :, 2] for b in blocks_ycbcr]

    y_bytes  = encode_planar_channel(y_blocks,  bs, q_y, deadzone, is_luma_intra=is_intra)
    cb_bytes = encode_planar_channel(cb_blocks, bs, q_c, deadzone, is_luma_intra=False)
    cr_bytes = encode_planar_channel(cr_blocks, bs, q_c, deadzone, is_luma_intra=False)

    out = bytearray()
    for section in (y_bytes, cb_bytes, cr_bytes):
        out.extend(struct.pack('>I', len(section)))
        out.extend(section)
    return bytes(out)


def unpack_frame_planar(data: bytes, off: int, q_y: float, q_c: float, bs: int, is_intra: bool = False) -> tuple:
    sections_data = []
    for _ in range(3):
        sz = struct.unpack_from('>I', data, off)[0]; off += 4
        sections_data.append(data[off:off+sz])
        off += sz

    decoded_y, _  = decode_planar_channel(sections_data[0], 0, q_y, bs, is_luma_intra=is_intra)
    decoded_cb, _ = decode_planar_channel(sections_data[1], 0, q_c, bs, is_luma_intra=False)
    decoded_cr, _ = decode_planar_channel(sections_data[2], 0, q_c, bs, is_luma_intra=False)

    if not decoded_y:
        return [], off

    n = len(decoded_y)
    result = []
    for i in range(n):
        block = np.stack([decoded_y[i], decoded_cb[i], decoded_cr[i]], axis=-1)
        result.append(block)
    return result, off

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION COMPENSATORY ANALYSIS
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

def extract_interference_delta(prev_block: np.ndarray, curr_block: np.ndarray,
                                dy: int, dx: int) -> np.ndarray:
    return curr_block - shift_block(prev_block, dy, dx)

# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACJA STRUMIENIA KODERA (MVD + Skip Map bez koordynatów)
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_p_frame(block_modes: np.ndarray, mvs: list, inter_blocks: list,
                      intra_blocks: list, bs: int, q_y: float, q_c: float, deadzone: float) -> bytes:
    out = bytearray()
    out.append(1)  # Typ 1: P-Frame
    
    # 1. Skip Map / Block Modes — ciągły zapis typów, idealny dla Zstd
    out.extend(block_modes.astype(np.uint8).tobytes())

    # 2. Motion Vector Difference (MVD) — relatywne kodowanie wektorów
    last_dx, last_dy = 0, 0
    for dx, dy in mvs:
        mvd_x = dx - last_dx
        mvd_y = dy - last_dy
        last_dx, last_dy = dx, dy
        mx = max(-127, min(127, mvd_x))
        my = max(-127, min(127, mvd_y))
        out.extend(struct.pack('>bb', mx, my))

    # 3. Sekcja INTER Planar Residuals
    if inter_blocks:
        inter_bytes = pack_frame_planar(inter_blocks, bs, q_y, q_c, deadzone, is_intra=False)
        out.extend(struct.pack('>I', len(inter_bytes)) + inter_bytes)
    else:
        out.extend(struct.pack('>I', 0))

    # 4. Sekcja INTRA Planar Blocks
    if intra_blocks:
        intra_bytes = pack_frame_planar(intra_blocks, bs, q_y, q_c, deadzone, is_intra=True)
        out.extend(struct.pack('>I', len(intra_bytes)) + intra_bytes)
    else:
        out.extend(struct.pack('>I', 0))

    return bytes(out)


def deserialize_frame_v14(data: bytes, h_blocks: int, w_blocks: int, bs: int, q_y: float, q_c: float) -> dict:
    off = 0
    frame_type = data[off]; off += 1

    if frame_type == 2:  # Static Frame
        return {'type': 2}

    if frame_type == 0:  # I-Frame (Zdekodowany za pomocą potoku DCT Planar)
        all_blocks, _ = unpack_frame_planar(data, off, q_y, q_c, bs, is_intra=True)
        return {'type': 0, 'blocks': all_blocks}

    # P-Frame Parsing
    n_blocks = h_blocks * w_blocks
    block_modes = np.frombuffer(data, dtype=np.uint8, count=n_blocks, offset=off).copy()
    off += n_blocks

    n_inter = np.sum(block_modes == 1)

    # Rekonstrukcja wektorów z bazy MVD
    mvs = []
    last_dx, last_dy = 0, 0
    for _ in range(n_inter):
        mx, my = struct.unpack_from('>bb', data, off); off += 2
        real_dx = last_dx + mx
        real_dy = last_dy + my
        last_dx, last_dy = real_dx, real_dy
        mvs.append((real_dx, real_dy))

    inter_len = struct.unpack_from('>I', data, off)[0]; off += 4
    inter_blocks = []
    if inter_len > 0:
        inter_blocks, _ = unpack_frame_planar(data, off, q_y, q_c, bs, is_intra=False)
        off += inter_len

    intra_len = struct.unpack_from('>I', data, off)[0]; off += 4
    intra_blocks = []
    if intra_len > 0:
        intra_blocks, _ = unpack_frame_planar(data, off, q_y, q_c, bs, is_intra=True)
        off += intra_len

    return {
        'type': 1,
        'block_modes': block_modes.reshape(h_blocks, w_blocks),
        'mvs': mvs,
        'inter_blocks': inter_blocks,
        'intra_blocks': intra_blocks
    }

# ═══════════════════════════════════════════════════════════════════════════════
# KODEK GŁÓWNY (POTOK Z OPTYMALIZACJĄ SIATKI GRID I FILTREM HAŁASU SAD)
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicCodec:
    def __init__(self, brush_size: int = 32, energy_threshold: float = 0.85,
                 delta_threshold: float = 5.0, q_y: float = 8.0, q_c: float = 20.0,
                 deadzone: float = 0.35, fast_skip_thr: float = 0.1):
        self.bs = brush_size
        self.energy_thr = energy_threshold
        self.delta_thr = delta_threshold
        self.q_y = q_y
        self.q_c = q_c
        self.deadzone = deadzone
        self.skip_thr = fast_skip_thr
        self.prev_wave = None

    def encode_frame(self, frame_rgb: np.ndarray) -> tuple:
        bs = self.bs
        h_orig, w_orig = frame_rgb.shape[:2]

        # 1. IDEALNY PADDING (Likwidacja chaosu niesymetrycznych krawędzi)
        h_padded = ((h_orig + bs - 1) // bs) * bs
        w_padded = ((w_orig + bs - 1) // bs) * bs
        pad_y = h_padded - h_orig
        pad_x = w_padded - w_orig

        if pad_y > 0 or pad_x > 0:
            padded_rgb = np.pad(frame_rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        else:
            padded_rgb = frame_rgb

        wave3 = rgb_to_ycbcr(padded_rgb)
        h_blocks = h_padded // bs
        w_blocks = w_padded // bs

        # KLATKA KLUCZOWA (I-Frame) — Teraz w pełni kompresowana przez potok Planar DCT RLE
        if self.prev_wave is None:
            all_blocks = []
            for row in range(h_blocks):
                for col in range(w_blocks):
                    y, x = row * bs, col * bs
                    all_blocks.append(wave3[y:y+bs, x:x+bs])
            i_frame_bytes = pack_frame_planar(all_blocks, bs, self.q_y, self.q_c, self.deadzone, is_intra=True)
            self.prev_wave = wave3
            return 0, i_frame_bytes

        # KLATKI INTER (P-Frame) — Analiza logiczna z filtrem SAD opłacalności ruchu
        block_modes = np.zeros((h_blocks, w_blocks), dtype=np.uint8)
        mvs = []
        inter_blocks = []
        intra_blocks = []

        for row in range(h_blocks):
            for col in range(w_blocks):
                y, x = row * bs, col * bs
                brush = wave3[y:y+bs, x:x+bs]
                prev_brush = self.prev_wave[y:y+bs, x:x+bs]

                # Brutalny test SAD dla pozycji statycznej
                sad_static = np.sum(np.abs(brush[..., 0] - prev_brush[..., 0]))

                if sad_static < (bs * bs * self.skip_thr):
                    block_modes[row, col] = 0  # MODE 0: SKIP (Zero bajtów overheadu)
                    continue

                # Korelacja optyczna szuka przesunięcia
                dy, dx, ncc = optical_correlate(prev_brush, brush)
                shifted_prev = shift_block(prev_brush, dy, dx)
                sad_inter = np.sum(np.abs(brush[..., 0] - shifted_prev[..., 0]))

                # Filtr opłacalności: wektor akceptujemy TYLKO jeśli zysk błędu wynosi min. 25%
                if ncc >= self.energy_thr and sad_inter < 0.75 * sad_static:
                    block_modes[row, col] = 1  # MODE 1: INTER
                    mvs.append((dx, dy))
                    delta = brush - shifted_prev
                    delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
                    sparse_delta = delta * delta_mask[..., np.newaxis]
                    inter_blocks.append(sparse_delta)
                else:
                    block_modes[row, col] = 2  # MODE 2: INTRA
                    intra_blocks.append(brush)

        if np.all(block_modes == 0):
            self.prev_wave = wave3
            return 2, b''  # Czysty Static Frame

        raw_payload = serialize_p_frame(block_modes, mvs, inter_blocks, intra_blocks,
                                        bs, self.q_y, self.q_c, self.deadzone)
        self.prev_wave = wave3
        return 1, raw_payload

    def decode_frame(self, payload_info: dict, h_orig: int, w_orig: int) -> np.ndarray:
        bs = self.bs
        h_padded = ((h_orig + bs - 1) // bs) * bs
        w_padded = ((w_orig + bs - 1) // bs) * bs
        h_blocks = h_padded // bs
        w_blocks = w_padded // bs

        if payload_info['type'] == 2:  # Static
            return ycbcr_to_rgb(self.prev_wave)[:h_orig, :w_orig]

        if payload_info['type'] == 0:  # I-Frame Reconstruction
            all_blocks = payload_info['blocks']
            new_wave = np.zeros((h_padded, w_padded, 3), dtype=np.float32)
            idx = 0
            for row in range(h_blocks):
                for col in range(w_blocks):
                    y, x = row * bs, col * bs
                    new_wave[y:y+bs, x:x+bs] = all_blocks[idx]
                    idx += 1
            self.prev_wave = new_wave
            return ycbcr_to_rgb(new_wave)[:h_orig, :w_orig]

        # P-Frame Reconstruction via Raster Skip Map
        block_modes = payload_info['block_modes']
        mvs = payload_info['mvs']
        inter_blocks = payload_info['inter_blocks']
        intra_blocks = payload_info['intra_blocks']

        new_wave = self.prev_wave.copy()
        inter_idx = 0
        intra_idx = 0
        mv_idx = 0

        for row in range(h_blocks):
            for col in range(w_blocks):
                y, x = row * bs, col * bs
                mode = block_modes[row, col]

                if mode == 0:
                    continue
                elif mode == 1:
                    dx, dy = mvs[mv_idx]
                    mv_idx += 1
                    delta = inter_blocks[inter_idx]
                    inter_idx += 1
                    shifted = shift_block(self.prev_wave[y:y+bs, x:x+bs], dy, dx)
                    new_wave[y:y+bs, x:x+bs] = shifted + delta
                elif mode == 2:
                    new_wave[y:y+bs, x:x+bs] = intra_blocks[intra_idx]
                    intra_idx += 1

        self.prev_wave = new_wave
        return ycbcr_to_rgb(new_wave)[:h_orig, :w_orig]

# ═══════════════════════════════════════════════════════════════════════════════
# I/O STREAM INTERFACE (Optymalizacja Single-Pass Zstd na Klatkę)
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def photonic_writer(path, w, h, fps, bs, q_y, q_c, deadzone):
    f = open(path, 'wb')
    f.write(_MAGIC)
    f.write(struct.pack('>HHfHfff', w, h, fps, bs, q_y, q_c, deadzone))
    compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)

    try:
        class W:
            def __init__(self):
                self.f = f
                self.comp = compressor
                self.codec = PhotonicCodec(brush_size=bs, q_y=q_y, q_c=q_c, deadzone=deadzone)

            def write_frame(self, frame_rgb: np.ndarray):
                frame_type, raw = self.codec.encode_frame(frame_rgb)
                if frame_type == 2:
                    payload = bytes([2])
                elif frame_type == 0:
                    payload = bytes([0]) + raw
                else:
                    payload = raw

                compressed = self.comp.compress(payload)
                self.f.write(struct.pack('>I', len(compressed)) + compressed)

            def close(self): self.f.close()
        yield W()
    finally:
        if not f.closed: f.close()


@contextmanager
def photonic_reader(path):
    f = open(path, 'rb')
    try:
        m = f.read(4)
        if m != _MAGIC:
            raise ValueError(f"Zły identyfikator pliku (magic): {m!r}")
        
        # TUTAJ BYŁ BŁĄD: Zmieniono f.read(18) na f.read(22)
        w, h, fps, bs, q_y, q_c, deadzone = struct.unpack('>HHfHfff', f.read(22))
        
        decompressor = zstd.ZstdDecompressor()

        class R:
            def __init__(self):
                self.f = f
                self.w, self.h, self.fps = w, h, fps
                self.bs, self.q_y, self.q_c = bs, q_y, q_c
                self.decomp = decompressor
                self.h_blocks = ((h + bs - 1) // bs)
                self.w_blocks = ((w + bs - 1) // bs)

            def read(self) -> dict | None:
                sz_b = self.f.read(4)
                if len(sz_b) < 4: return None
                comp_len = struct.unpack('>I', sz_b)[0]
                comp_data = self.f.read(comp_len)
                if not comp_data: return None
                raw = self.decomp.decompress(comp_data)
                return deserialize_frame_v14(raw, self.h_blocks, self.w_blocks, self.bs, self.q_y, self.q_c)

            def close(self): self.f.close()
        yield R()
    finally:
        if not f.closed: f.close()
# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio na dysku. Zainstaluj imageio[pyav]")
    print("[Photonic v1.4] Encoding (Grid Padding + Raster Skip Map)...", flush=True)

    reader = iio.imiter(inp, plugin='pyav')
    first = next(reader)
    h, w = first.shape[:2]

    with photonic_writer(out, w, h, args.fps, args.brush_size, args.q_y, args.q_c, args.deadzone) as writer:
        writer.write_frame(first[..., :3])
        fi = 1; t = time.time()
        for frame in reader:
            writer.write_frame(frame[..., :3])
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS", end="", flush=True)
            fi += 1
    print(f"\n[Photonic] Proces zakończony sukcesem. Skompresowano {fi} klatek.", flush=True)


def decode_video(inp, out, args):
    if not iio: return print("Brak imageio na dysku. Zainstaluj imageio[pyav]")
    print("[Photonic v1.4] Decoding...", flush=True)

    with photonic_reader(inp) as reader:
        codec = PhotonicCodec(brush_size=reader.bs, q_y=reader.q_y, q_c=reader.q_c, deadzone=args.deadzone)
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)
        fi = 0; t = time.time()

        while True:
            payload_info = reader.read()
            if payload_info is None: break
            rgb = codec.decode_frame(payload_info, reader.h, reader.w)
            w_out.write_frame(rgb)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS", end="", flush=True)
            fi += 1
        w_out.close()
    print(f"\n[Photonic] Proces dekodowania zakończony. Wygenerowano {fi} klatek.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.4 Absolute Compact")
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')

    parser.add_argument('--brush-size',      type=int,   default=32)
    parser.add_argument('--fps',             type=float, default=25.0)
    parser.add_argument('--q-y',             type=float, default=8.0)
    parser.add_argument('--q-c',             type=float, default=20.0)
    parser.add_argument('--deadzone',        type=float, default=0.35)

    args = parser.parse_args()

    if args.decode:
        decode_video(args.input, args.output, args)
    else:
        encode_video(args.input, args.output, args)
