#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.3 — KOMPRESJA PLANARNO-TEMPORALNA (RLE + MVD)     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Poprawka wydajności strumienia:                                             ║
║  - Usunięto Sparse Column-Major overhead (zastąpiony uproszczonym RLE/EOB)  ║
║  - Dodano predykcję różnicową wektorów ruchu (Motion Vector Difference - MVD)║
║  - Poprawna obsługa bloków krawędziowych bez wycieku metadanych             ║
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

_MAGIC      = b'PH04'
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
# ZIGZAG & QUANTIZATION
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
# NOWY SILNIK: SEKWENCYJNY RLE + DC DELTA (Eliminacja overheadu Sparse Matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_planar_channel(blocks_2d_list: list, bs: int, Q: float, deadzone: float,
                           is_luma_intra: bool = False) -> bytes:
    """Zwraca spakowany kanał za pomocą uproszczonego RLE z flagą EOB marker"""
    if not blocks_2d_list:
        return struct.pack('>HH', 0, bs)

    n_blocks = len(blocks_2d_list)
    n_coeffs = bs * bs
    zz = get_zigzag_indices(bs)

    out = bytearray()
    out.extend(struct.pack('>HH', n_blocks, bs))

    last_dc = 0
    for blk in blocks_2d_list:
        ch = blk.astype(np.float32)
        if is_luma_intra:
            ch = ch - 128.0
        
        q = deadzone_quantize(apply_dct2(ch), Q, deadzone).flatten()[zz]
        
        # DC Delta Prediction
        dc_val = int(q[0])
        dc_delta = dc_val - last_dc
        last_dc = dc_val
        out.extend(struct.pack('>h', dc_delta))

        # AC RLE z odcinaniem końcowych zer (EOB)
        ac_coeffs = q[1:]
        nonzero_indices = np.where(ac_coeffs != 0)[0]
        
        if len(nonzero_indices) == 0:
            # Flaga EOB natychmiast na pozycji 0
            out.extend(struct.pack('>H', 0xFFFF))
        else:
            last_nz = nonzero_indices[-1]
            run = 0
            for i in range(last_nz + 1):
                val = int(ac_coeffs[i])
                if val == 0:
                    run += 1
                else:
                    while run >= 255:
                        out.extend(struct.pack('>BB', 255, 0)) # Sztuczny run-limit
                        run -= 255
                    out.extend(struct.pack('>Bbh', run, 1, val))
                    run = 0
            # Znak końca bloku po ostatnim istotnym współczynniku
            out.extend(struct.pack('>B', 0xFF)) 

    return bytes(out)


def decode_planar_channel(data: bytes, Q: float, is_luma_intra: bool = False) -> list:
    off = 0
    n_blocks, bs = struct.unpack_from('>HH', data, off); off += 4
    if n_blocks == 0:
        return []

    n_coeffs = bs * bs
    zz = get_zigzag_indices(bs)
    zz_inv = np.argsort(zz)

    result = []
    last_dc = 0

    for _ in range(n_blocks):
        flat = np.zeros(n_coeffs, dtype=np.float32)
        
        dc_delta = struct.unpack_from('>h', data, off)[0]; off += 2
        current_dc = last_dc + dc_delta
        last_dc = current_dc
        flat[0] = float(current_dc)

        # Dekodowanie strumienia AC RLE
        ac_idx = 1
        while ac_idx < n_coeffs:
            first_byte = data[off]; off += 1
            if first_byte == 0xFF: # EOB normalny
                break
            if first_byte == 0xFF and ac_idx == 1: # EOB dla pustego AC
                off += 1 # Wyrównanie struktury struktury 0xFFFF
                break
                
            run = first_byte
            # Sprawdzenie flagi obecności wartości
            has_val = data[off]; off += 1
            if has_val == 0:
                ac_idx += run
                continue
                
            val = struct.unpack_from('>h', data, off)[0]; off += 2
            ac_idx += run
            if ac_idx < n_coeffs:
                flat[ac_idx] = float(val)
            ac_idx += 1

        # Odwrócenie Zigzag i IDCT
        block_coeffs = np.zeros(n_coeffs, dtype=np.float32)
        block_coeffs[zz_inv] = flat * Q
        blk = apply_idct2(block_coeffs.reshape(bs, bs))
        if is_luma_intra:
            blk += 128.0
        result.append(blk)

    return result

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


def unpack_frame_planar(data: bytes, q_y: float, q_c: float, is_intra: bool = False) -> list:
    off = 0
    sections = []
    for _ in range(3):
        sz = struct.unpack_from('>I', data, off)[0]; off += 4
        sections.append(data[off:off+sz]); off += sz

    Q_list = [q_y, q_c, q_c]
    intra_flags = [is_intra, False, False]
    decoded = [decode_planar_channel(sec, Q_list[i], intra_flags[i])
               for i, sec in enumerate(sections)]

    if not decoded[0]:
        return []

    n = len(decoded[0])
    result = []
    for i in range(n):
        block = np.stack([decoded[c][i] for c in range(3)], axis=-1)
        result.append(block)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# FIZYKA: OPTICAL MOTION VECTOR ANALYSIS
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
# SERIALIZACJA KLATKI (MVD + Obsługa Krawędzi Edge)
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_v13(paths: list, bs: int, q_y: float, q_c: float, deadzone: float,
                         h: int, w: int) -> bytes:
    out = bytearray()
    if not paths:
        out.append(2)  # static
        return bytes(out)

    if len(paths) == 1 and paths[0]['type'] == 0:
        out.append(0)  # I-frame
        out.extend(paths[0]['data'])
        return bytes(out)

    out.append(1)  # P-frame

    intra_square = [p for p in paths if p['type'] == 2 and '_block' in p]
    intra_edge   = [p for p in paths if p['type'] == 2 and 'data' in p]
    inter_square = [p for p in paths if p['type'] == 1 and '_block' in p]
    inter_edge   = [p for p in paths if p['type'] == 1 and 'data' in p]

    out.extend(struct.pack('>HHHH', len(intra_square), len(intra_edge), len(inter_square), len(inter_edge)))

    # 1. INTRA Square (Planar RLE)
    if intra_square:
        for p in intra_square:
            out.extend(struct.pack('>HH', p['x'], p['y']))
        intra_blocks = [p['_block'] for p in intra_square]
        out.extend(pack_frame_planar(intra_blocks, bs, q_y, q_c, deadzone, is_intra=True))

    # 2. INTRA Edge (Fallback Raw)
    if intra_edge:
        for p in intra_edge:
            out.extend(struct.pack('>HHHHI', p['x'], p['y'], p['bh'], p['bw'], len(p['data'])))
            out.extend(p['data'])

    # 3. INTER Square (MVD + Planar Delta)
    if inter_square:
        last_dx, last_dy = 0, 0
        for p in inter_square:
            # Motion Vector Difference (MVD) predykcja wektora ruchu
            mvd_x = p['dx'] - last_dx
            mvd_y = p['dy'] - last_dy
            last_dx, last_dy = p['dx'], p['dy']
            
            # Pakowanie bezpieczne do signed byte
            mx = max(-127, min(127, mvd_x))
            my = max(-127, min(127, mvd_y))
            out.extend(struct.pack('>HHbb', p['x'], p['y'], mx, my))
            
        inter_blocks = [p['_block'] for p in inter_square]
        out.extend(pack_frame_planar(inter_blocks, bs, q_y, q_c, deadzone, is_intra=False))

    # 4. INTER Edge (Fallback Raw Delta)
    if inter_edge:
        for p in inter_edge:
            dx_c = max(-127, min(127, p['dx']))
            dy_c = max(-127, min(127, p['dy']))
            out.extend(struct.pack('>HHbbHHI', p['x'], p['y'], dx_c, dy_c, p['bh'], p['bw'], len(p['data'])))
            out.extend(p['data'])

    return bytes(out)


def deserialize_frame_v13(data: bytes, bs: int, q_y: float, q_c: float) -> list:
    off = 0
    frame_type = data[off]; off += 1

    if frame_type == 2:
        return []
    if frame_type == 0:
        return [{'type': 0, 'data': data[off:]}]

    n_intra_sq, n_intra_ed, n_inter_sq, n_inter_ed = struct.unpack_from('>HHHH', data, off); off += 8
    paths = []

    # 1. INTRA Square
    if n_intra_sq > 0:
        positions = []
        for _ in range(n_intra_sq):
            x, y = struct.unpack_from('>HH', data, off); off += 4
            positions.append((x, y))

        planar_start = off
        for _ in range(3):
            sz = struct.unpack_from('>I', data, off)[0]; off += 4 + sz

        intra_blocks = unpack_frame_planar(data[planar_start:off], q_y, q_c, is_intra=True)
        for i, (x, y) in enumerate(positions):
            paths.append({'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0, 'bh': bs, 'bw': bs, '_block': intra_blocks[i]})

    # 2. INTRA Edge
    for _ in range(n_intra_ed):
        x, y, bh, bw, d_len = struct.unpack_from('>HHHHI', data, off); off += 12
        raw_data = data[off:off+d_len]; off += d_len
        paths.append({'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0, 'bh': bh, 'bw': bw, 'data': raw_data})

    # 3. INTER Square
    if n_inter_sq > 0:
        mv_list = []
        last_dx, last_dy = 0, 0
        for _ in range(n_inter_sq):
            x, y, mx, my = struct.unpack_from('>HHbb', data, off); off += 6
            # Odbudowa struktury MVD
            real_dx = last_dx + mx
            real_dy = last_dy + my
            last_dx, last_dy = real_dx, real_dy
            mv_list.append((x, y, real_dx, real_dy))

        planar_start = off
        for _ in range(3):
            sz = struct.unpack_from('>I', data, off)[0]; off += 4 + sz

        inter_blocks = unpack_frame_planar(data[planar_start:off], q_y, q_c, is_intra=False)
        for i, (x, y, dx, dy) in enumerate(mv_list):
            paths.append({'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy, 'bh': bs, 'bw': bs, '_block': inter_blocks[i]})

    # 4. INTER Edge
    for _ in range(n_inter_ed):
        x, y, dx, dy, bh, bw, d_len = struct.unpack_from('>HHbbHHI', data, off); off += 16
        raw_data = data[off:off+d_len]; off += d_len
        paths.append({'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy, 'bh': bh, 'bw': bw, 'data': raw_data})

    return paths

# ═══════════════════════════════════════════════════════════════════════════════
# CODEC CORE
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

    def encode_frame(self, frame_rgb: np.ndarray) -> list:
        paths = []
        bs = self.bs
        curr_Y = np.dot(frame_rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114])

        if self.prev_wave is None:
            wave3 = rgb_to_ycbcr(frame_rgb)
            paths.append({'type': 0, 'data': wave3.tobytes()})
            self.prev_wave = wave3
            return paths

        diff_Y = np.abs(curr_Y - self.prev_wave[..., 0])
        h, w = diff_Y.shape
        h_blocks = h // bs
        w_blocks = w // bs

        valid_diff = diff_Y[:h_blocks*bs, :w_blocks*bs]
        sad_map = valid_diff.reshape(h_blocks, bs, w_blocks, bs).sum(axis=(1, 3))
        static_mask = sad_map < (bs * bs * self.skip_thr)

        if np.all(static_mask):
            return []

        wave3 = rgb_to_ycbcr(frame_rgb)

        for row in range(h_blocks):
            for col in range(w_blocks):
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
                        paths.append({
                            'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy,
                            'bh': bh, 'bw': bw, '_block': sparse_delta
                        })
                else:
                    paths.append({
                        'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0,
                        'bh': bh, 'bw': bw, '_block': brush
                    })

        # Krawędzie nienormatywne (float16 fallback)
        if h % bs != 0 or w % bs != 0:
            for ey in range(0, h, bs):
                for ex in range(0, w, bs):
                    if ex < w_blocks * bs and ey < h_blocks * bs:
                        continue
                    bh, bw = min(bs, h - ey), min(bs, w - ex)
                    brush = wave3[ey:ey+bh, ex:ex+bw]
                    prev_brush = self.prev_wave[ey:ey+bh, ex:ex+bw]
                    dy, dx, ncc = optical_correlate(prev_brush, brush)
                    if ncc >= self.energy_thr:
                        delta = extract_interference_delta(prev_brush, brush, dy, dx)
                        delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
                        if np.any(delta_mask):
                            paths.append({'type': 1, 'x': ex, 'y': ey, 'dx': dx, 'dy': dy, 'bh': bh, 'bw': bw,
                                          'data': (delta * delta_mask[..., np.newaxis]).astype(np.float16).tobytes()})
                    else:
                        paths.append({'type': 2, 'x': ex, 'y': ey, 'dx': 0, 'dy': 0, 'bh': bh, 'bw': bw,
                                      'data': brush.astype(np.float16).tobytes()})

        self.prev_wave = wave3
        return paths

    def decode_frame(self, paths: list, h: int, w: int) -> np.ndarray:
        if self.prev_wave is None:
            self.prev_wave = np.frombuffer(paths[0]['data'], dtype=np.float32).reshape(h, w, 3)
            return ycbcr_to_rgb(self.prev_wave)

        if not paths:
            return ycbcr_to_rgb(self.prev_wave)

        new_wave = self.prev_wave.copy()
        bs = self.bs

        for path in paths:
            x, y = path['x'], path['y']
            bh, bw = path['bh'], path['bw']
            is_square = (bh == bw and bh == bs)

            if path['type'] == 1:
                dy, dx = path['dy'], path['dx']
                if is_square and '_block' in path:
                    delta = path['_block']
                elif not is_square and 'data' in path:
                    delta = np.frombuffer(path['data'], dtype=np.float16).reshape(bh, bw, 3).astype(np.float32)
                else:
                    continue
                shifted = shift_block(new_wave[y:y+bh, x:x+bw], dy, dx)
                new_wave[y:y+bh, x:x+bw] = shifted + delta

            elif path['type'] == 2:
                if is_square and '_block' in path:
                    new_wave[y:y+bh, x:x+bw] = path['_block']
                elif not is_square and 'data' in path:
                    new_wave[y:y+bh, x:x+bw] = np.frombuffer(path['data'], dtype=np.float16).reshape(bh, bw, 3).astype(np.float32)

        self.prev_wave = new_wave
        return ycbcr_to_rgb(new_wave)

# ═══════════════════════════════════════════════════════════════════════════════
# I/O INTERFACE (Czysty jednoprzebiegowy Zstd na klatkę)
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
                paths = self.codec.encode_frame(frame_rgb)
                raw = serialize_frame_v13(paths, bs, q_y, q_c, deadzone, frame_rgb.shape[0], frame_rgb.shape[1])
                compressed = self.comp.compress(raw)
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
            raise ValueError(f"Zły magic pliku: {m!r}")
        w, h, fps, bs, q_y, q_c, deadzone = struct.unpack('>HHfHfff', f.read(18))
        decompressor = zstd.ZstdDecompressor()

        class R:
            def __init__(self):
                self.f = f
                self.w, self.h, self.fps = w, h, fps
                self.bs, self.q_y, self.q_c = bs, q_y, q_c
                self.decomp = decompressor

            def read(self) -> list | None:
                sz_b = self.f.read(4)
                if len(sz_b) < 4: return None
                comp_len = struct.unpack('>I', sz_b)[0]
                comp_data = self.f.read(comp_len)
                if not comp_data: return None
                raw = self.decomp.decompress(comp_data)
                return deserialize_frame_v13(raw, self.bs, self.q_y, self.q_c)

            def close(self): self.f.close()
        yield R()
    finally:
        if not f.closed: f.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.3] Encoding (RLE-Stream + MVD)...", flush=True)

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
    print(f"\n[Photonic] Gotowe. Zapisano {fi} klatek.", flush=True)


def decode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.3] Decoding...", flush=True)

    with photonic_reader(inp) as reader:
        codec = PhotonicCodec(brush_size=reader.bs, q_y=reader.q_y, q_c=reader.q_c, deadzone=args.deadzone)
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)
        fi = 0; t = time.time()

        while True:
            paths = reader.read()
            if paths is None: break
            rgb = codec.decode_frame(paths, reader.h, reader.w)
            w_out.write_frame(rgb)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS", end="", flush=True)
            fi += 1
        w_out.close()
    print(f"\n[Photonic] Odtworzono {fi} klatek.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.3 Optimized")
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