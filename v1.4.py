#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.4 — GRID PADDING + RASTER SKIP MAP                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v1.4 zmiany:                                                                ║
║  - Całkowite usunięcie FFT, wektorów ruchu i problematycznego shift_block    ║
║  - Wprowadzenie Grid Padding (automatyczne wyrównanie krawędzi obrazu)       ║
║  - Wprowadzenie Raster Skip Map (globalna maska bitowa spakowana packbits)   ║
║  - Ultra-czysta kompresja różnicowa (Temporal Delta) dla zmiennych bloków   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.fftpack import dct, idct
import zstandard as zstd
import struct
import argparse
import time
from collections import deque
from contextlib import contextmanager

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

_MAGIC      = b'PH05'     # Nowy identyfikator dla v1.4
_ZSTD_LEVEL = 19
_DICT_FRAMES = 8          
_DICT_MAX_BYTES = 112640  

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
# DCT / QUANTIZATION / ZIGZAG
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
# PLANAR DCT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def encode_planar_channel(blocks_2d_list: list, bs: int, Q: float, deadzone: float) -> bytes:
    if not blocks_2d_list:
        return struct.pack('>HH', 0, bs)

    n_blocks = len(blocks_2d_list)
    n_coeffs = bs * bs
    zz = get_zigzag_indices(bs)

    all_q = np.zeros((n_blocks, n_coeffs), dtype=np.int16)
    for i, blk in enumerate(blocks_2d_list):
        q = deadzone_quantize(apply_dct2(blk.astype(np.float32)), Q, deadzone)
        all_q[i] = q.flatten()[zz]

    dc = all_q[:, 0].astype(np.int32)
    dc_delta = np.diff(dc, prepend=0).astype(np.int16)

    ac = all_q[:, 1:]
    n_ac = n_coeffs - 1

    out = bytearray()
    out.extend(struct.pack('>HH', n_blocks, bs))
    out.extend(dc_delta.tobytes())

    for ac_pos in range(n_ac):
        col = ac[:, ac_pos]
        nz_idx = np.where(col != 0)[0].astype(np.uint16)
        nz_val = col[nz_idx].astype(np.int16)
        nz_count = len(nz_idx)
        out.extend(struct.pack('>H', nz_count))
        if nz_count > 0:
            out.extend(nz_idx.tobytes())
            out.extend(nz_val.tobytes())

    return bytes(out)

def decode_planar_channel(data: bytes, Q: float) -> list:
    off = 0
    n_blocks, bs = struct.unpack_from('>HH', data, off); off += 4
    if n_blocks == 0:
        return []

    n_coeffs = bs * bs
    n_ac = n_coeffs - 1
    zz = get_zigzag_indices(bs)
    zz_inv = np.argsort(zz)

    dc_delta = np.frombuffer(data, dtype=np.int16, count=n_blocks, offset=off).astype(np.int32)
    off += n_blocks * 2
    dc = np.cumsum(dc_delta).astype(np.int16)

    all_q = np.zeros((n_blocks, n_coeffs), dtype=np.int16)
    all_q[:, 0] = dc

    for ac_pos in range(n_ac):
        nz_count = struct.unpack_from('>H', data, off)[0]; off += 2
        if nz_count > 0:
            nz_idx = np.frombuffer(data, dtype=np.uint16, count=nz_count, offset=off); off += nz_count * 2
            nz_val = np.frombuffer(data, dtype=np.int16,  count=nz_count, offset=off); off += nz_count * 2
            all_q[nz_idx, ac_pos + 1] = nz_val

    result = []
    for i in range(n_blocks):
        zz_coeffs = all_q[i]
        flat = np.zeros(n_coeffs, dtype=np.float32)
        flat[zz_inv] = zz_coeffs.astype(np.float32)
        blk = apply_idct2(flat.reshape(bs, bs) * Q)
        result.append(blk)
    return result

def pack_frame_planar(blocks_ycbcr: list, bs: int, q_y: float, q_c: float, deadzone: float) -> bytes:
    y_blocks  = [b[:, :, 0] for b in blocks_ycbcr]
    cb_blocks = [b[:, :, 1] for b in blocks_ycbcr]
    cr_blocks = [b[:, :, 2] for b in blocks_ycbcr]

    y_bytes  = encode_planar_channel(y_blocks,  bs, q_y, deadzone)
    cb_bytes = encode_planar_channel(cb_blocks, bs, q_c, deadzone)
    cr_bytes = encode_planar_channel(cr_blocks, bs, q_c, deadzone)

    out = bytearray()
    for section in (y_bytes, cb_bytes, cr_bytes):
        out.extend(struct.pack('>I', len(section)))
        out.extend(section)
    return bytes(out)

def unpack_frame_planar(data: bytes, q_y: float, q_c: float) -> list:
    off = 0
    sections = []
    for _ in range(3):
        sz = struct.unpack_from('>I', data, off)[0]; off += 4
        sections.append(data[off:off+sz]); off += sz

    Q_list = [q_y, q_c, q_c]
    decoded = [decode_planar_channel(sec, Q_list[i]) for i, sec in enumerate(sections)]

    if not decoded[0]:
        return []

    n = len(decoded[0])
    result = []
    for i in range(n):
        block = np.stack([decoded[c][i] for c in range(3)], axis=-1)
        result.append(block)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACJA KLATKI (Nowy Format v1.4 z Raster Skip Map)
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_v14(frame_type: int, skip_mask: np.ndarray | None, active_blocks: list | None, 
                        raw_data: bytes | None, bs: int, q_y: float, q_c: float, deadzone: float) -> bytes:
    out = bytearray()
    out.append(frame_type)

    if frame_type == 0:  # I-frame
        out.extend(raw_data)
    elif frame_type == 1:  # P-frame (Raster Skip Map)
        h_b, w_b = skip_mask.shape
        out.extend(struct.pack('>HH', h_b, w_b))
        
        packed_mask = np.packbits(skip_mask)
        out.extend(struct.pack('>I', len(packed_mask)))
        out.extend(packed_mask.tobytes())
        
        if active_blocks:
            planar_data = pack_frame_planar(active_blocks, bs, q_y, q_c, deadzone)
            out.extend(planar_data)
    return bytes(out)

def deserialize_frame_v14(data: bytes, bs: int, q_y: float, q_c: float) -> tuple:
    off = 0
    frame_type = data[off]; off += 1

    if frame_type == 0:
        return frame_type, None, None, data[off:]
    elif frame_type == 2:
        return frame_type, None, None, None

    # P-frame
    h_b, w_b = struct.unpack_from('>HH', data, off); off += 4
    mask_len = struct.unpack_from('>I', data, off)[0]; off += 4
    
    packed_mask = np.frombuffer(data, dtype=np.uint8, count=mask_len, offset=off)
    off += mask_len
    
    skip_mask = np.unpackbits(packed_mask)[:h_b*w_b].reshape(h_b, w_b).astype(bool)
    
    n_active = np.count_nonzero(~skip_mask)
    active_blocks = []
    if n_active > 0:
        active_blocks = unpack_frame_planar(data[off:], q_y, q_c)
        
    return frame_type, skip_mask, active_blocks, None

# ═══════════════════════════════════════════════════════════════════════════════
# KODEK GŁÓWNY (v1.4 ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicCodec:
    def __init__(self, brush_size: int = 32, q_y: float = 8.0, q_c: float = 20.0,
                 deadzone: float = 0.35, fast_skip_thr: float = 0.1):
        self.bs = brush_size
        self.q_y = q_y
        self.q_c = q_c
        self.deadzone = deadzone
        self.skip_thr = fast_skip_thr
        self.prev_wave = None

    def encode_frame(self, frame_rgb: np.ndarray) -> tuple:
        bs = self.bs
        h_orig, w_orig = frame_rgb.shape[:2]
        
        # Grid Padding — wyrównanie krawędzi
        pad_h = (bs - (h_orig % bs)) % bs
        pad_w = (bs - (w_orig % bs)) % bs
        if pad_h > 0 or pad_w > 0:
            frame_rgb = np.pad(frame_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

        curr_wave = rgb_to_ycbcr(frame_rgb)

        if self.prev_wave is None:
            self.prev_wave = curr_wave
            return 0, None, None, curr_wave.tobytes()  # I-frame

        diff_Y = np.abs(curr_wave[..., 0] - self.prev_wave[..., 0])
        h, w = diff_Y.shape
        h_blocks = h // bs
        w_blocks = w // bs

        sad_map = diff_Y.reshape(h_blocks, bs, w_blocks, bs).sum(axis=(1, 3))
        skip_mask = sad_map < (bs * bs * self.skip_thr)

        if np.all(skip_mask):
            return 2, None, None, None  # Static frame

        active_blocks = []
        for row in range(h_blocks):
            for col in range(w_blocks):
                if skip_mask[row, col]:
                    continue
                y, x = row * bs, col * bs
                brush = curr_wave[y:y+bs, x:x+bs]
                prev_brush = self.prev_wave[y:y+bs, x:x+bs]
                
                # Czysta delta czasowa (Temporal Delta) bez estymacji ruchu!
                active_blocks.append(brush - prev_brush)

        self.prev_wave = curr_wave
        return 1, skip_mask, active_blocks, None  # P-frame

    def decode_frame(self, frame_info: tuple, h: int, w: int) -> np.ndarray:
        frame_type, skip_mask, active_blocks, raw_data = frame_info
        bs = self.bs
        
        pad_h = (bs - (h % bs)) % bs
        pad_w = (bs - (w % bs)) % bs
        h_padded, w_padded = h + pad_h, w + pad_w

        if self.prev_wave is None:
            self.prev_wave = np.frombuffer(raw_data, dtype=np.float32).reshape(h_padded, w_padded, 3)
            return ycbcr_to_rgb(self.prev_wave)[:h, :w]

        if frame_type == 2 or skip_mask is None:
            return ycbcr_to_rgb(self.prev_wave)[:h, :w]

        new_wave = self.prev_wave.copy()
        h_blocks, w_blocks = skip_mask.shape
        
        block_idx = 0
        for row in range(h_blocks):
            for col in range(w_blocks):
                if skip_mask[row, col]:
                    continue
                y, x = row * bs, col * bs
                delta = active_blocks[block_idx]
                block_idx += 1
                new_wave[y:y+bs, x:x+bs] += delta

        self.prev_wave = new_wave
        # Przycięcie Grid Padding z powrotem do oryginalnego h, w
        return ycbcr_to_rgb(new_wave)[:h, :w]

# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalDictBuilder:
    def __init__(self, frames_to_keep: int = _DICT_FRAMES, max_dict_bytes: int = _DICT_MAX_BYTES):
        self.samples = deque(maxlen=frames_to_keep)
        self.max_dict = max_dict_bytes
        self._dict_data: bytes | None = None
        self._frame_count = 0
        self._retrain_every = frames_to_keep

    def feed(self, raw_payload: bytes):
        self.samples.append(raw_payload)
        self._frame_count += 1
        if self._frame_count % self._retrain_every == 0 and len(self.samples) >= 2:
            self._retrain()

    def _retrain(self):
        try:
            self._dict_data = zstd.ZstdCompressionDict(
                zstd.train_dictionary(self.max_dict, list(self.samples))
            )
        except Exception:
            self._dict_data = None

    def get_compressor(self) -> zstd.ZstdCompressor:
        if self._dict_data:
            return zstd.ZstdCompressor(level=_ZSTD_LEVEL, dict_data=self._dict_data)
        return zstd.ZstdCompressor(level=_ZSTD_LEVEL)

    def get_decompressor(self) -> zstd.ZstdDecompressor:
        if self._dict_data:
            return zstd.ZstdDecompressor(dict_data=self._dict_data)
        return zstd.ZstdDecompressor()

    @property
    def dict_bytes(self) -> bytes:
        return self._dict_data.as_bytes() if self._dict_data else b''

# ═══════════════════════════════════════════════════════════════════════════════
# I/O INTERFACES
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def photonic_writer(path, w, h, fps, bs, q_y, q_c, deadzone):
    f = open(path, 'wb')
    f.write(_MAGIC)
    # Zapisujemy nagłówek o stałej długości 22 bajtów
    f.write(struct.pack('>HHfHfff', w, h, fps, bs, q_y, q_c, deadzone))
    dict_builder = TemporalDictBuilder()
    codec_enc = PhotonicCodec(brush_size=bs, q_y=q_y, q_c=q_c, deadzone=deadzone)

    try:
        class W:
            def __init__(self):
                self.f = f
                self.db = dict_builder
                self.codec = codec_enc
                self._frame_idx = 0

            def write_frame(self, frame_rgb: np.ndarray):
                frame_info = self.codec.encode_frame(frame_rgb)
                raw = serialize_frame_v14(*frame_info, bs, q_y, q_c, deadzone)
                self.db.feed(raw)
                comp = self.db.get_compressor().compress(raw)

                dict_payload = b''
                if self._frame_idx > 0 and self._frame_idx % _DICT_FRAMES == 0:
                    dict_payload = self.db.dict_bytes

                has_dict = 1 if dict_payload else 0
                self.f.write(struct.pack('>B', has_dict))
                if dict_payload:
                    self.f.write(struct.pack('>I', len(dict_payload)))
                    self.f.write(dict_payload)
                self.f.write(struct.pack('>I', len(comp)) + comp)
                self._frame_idx += 1

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
            raise ValueError(f"Zły format pliku (magic={m!r})")
        # Poprawne czytanie nagłówka 22-bajtowego
        w, h, fps, bs, q_y, q_c, deadzone = struct.unpack('>HHfHfff', f.read(22))
        current_dict: bytes | None = None

        class R:
            def __init__(self):
                self.f = f
                self.w = w; self.h = h; self.fps = fps
                self.bs = bs; self.q_y = q_y; self.q_c = q_c

            def read(self) -> tuple | None:
                hdr = self.f.read(1)
                if not hdr: return None
                has_dict = hdr[0]
                if has_dict:
                    dict_len = struct.unpack('>I', self.f.read(4))[0]
                    dict_bytes = self.f.read(dict_len)
                    self._dict_data = zstd.ZstdCompressionDict(dict_bytes)

                sz_b = self.f.read(4)
                if len(sz_b) < 4: return None
                comp = self.f.read(struct.unpack('>I', sz_b)[0])
                if not comp: return None

                d = zstd.ZstdDecompressor(dict_data=self._dict_data) if hasattr(self, '_dict_data') else zstd.ZstdDecompressor()
                raw = d.decompress(comp)
                return deserialize_frame_v14(raw, bs, q_y, q_c)

            def close(self): self.f.close()
        yield R()
    finally:
        if not f.closed: f.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.4] Encoding (Grid Padding + Raster Skip Map)...", flush=True)

    reader = iio.imiter(inp, plugin='pyav')
    first = next(reader)
    h, w = first.shape[:2]

    with photonic_writer(out, w, h, args.fps, args.brush_size,
                         args.q_y, args.q_c, args.deadzone) as writer:
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
    if not iio: return print("Brak imageio")
    print("[Photonic v1.4] Decoding...", flush=True)

    with photonic_reader(inp) as reader:
        codec = PhotonicCodec(brush_size=reader.bs, q_y=reader.q_y, q_c=reader.q_c, deadzone=args.deadzone)
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)
        fi = 0; t = time.time()

        while True:
            frame_info = reader.read()
            if frame_info is None: break
            rgb = codec.decode_frame(frame_info, reader.h, reader.w)
            w_out.write_frame(rgb)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS", end="", flush=True)
            fi += 1
        w_out.close()
    print(f"\n[Photonic] Proces dekodowania zakończony. Wygenerowano {fi} klatek.", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.4")
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