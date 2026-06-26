#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.3 — KOMPRESJA PLANARNO-TEMPORALNA               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v1.3 zmiany (inkompatybilny z v1.2):                                      ║
║  - Planar DCT layout: Y/Cb/Cr jako oddzielne ciągłe bufory int16           ║
║  - Global coefficient scan: DC → AC[1] → AC[2]... zamiast per-blok zigzag ║
║  - DC delta prediction (horizontal raster order) przed kompresją           ║
║  - Zstd dictionary z poprzednich klatek → kontekst temporalny              ║
║  - Frame-level single zstd pass zamiast per-frame niezależnych bloków      ║
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
_DICT_FRAMES = 8          # Liczba poprzednich klatek do budowy słownika
_DICT_MAX_BYTES = 112640  # Maks. rozmiar słownika zstd (~110KB)

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
# ZIGZAG (tylko per-blok dla AC reorder)
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
# NOWOŚĆ v1.3: PLANAR DCT ENGINE
# Wszystkie bloki jednego kanału → jedna tablica int16 z global coeff scan
# Layout: [DC_blok0, DC_blok1, ..., AC1_blok0, AC1_blok1, ..., AC(N²-1)_blokM]
# DC dodatkowo delta-coded w raster order.
# ═══════════════════════════════════════════════════════════════════════════════

def encode_planar_channel(blocks_2d_list: list, bs: int, Q: float, deadzone: float,
                           is_luma_intra: bool = False) -> bytes:
    """
    blocks_2d_list: lista tablic (bs, bs) float32 — wszystkie bloki kanału w raster order
    Zwraca: bytes gotowe do zstd
    
    Format:
      uint16 n_blocks
      uint16 bs
      int16[n_blocks] DC_delta  (DC[0] dosłowny, DC[i] = DC[i]-DC[i-1])
      uint32[n_coeffs] nonzero_count  (dla każdego AC coeff pozycji, ile bloków ma niezerową)
      -- dla każdej pozycji AC (zigzag order 1..N²-1):
         uint16 nz_count
         uint16[nz_count] block_indices
         int16[nz_count]  values
    """
    if not blocks_2d_list:
        return struct.pack('>HH', 0, bs)

    n_blocks = len(blocks_2d_list)
    n_coeffs = bs * bs
    zz = get_zigzag_indices(bs)

    # Kwantyzacja wszystkich bloków naraz
    all_q = np.zeros((n_blocks, n_coeffs), dtype=np.int16)
    for i, blk in enumerate(blocks_2d_list):
        ch = blk.astype(np.float32)
        if is_luma_intra:
            ch = ch - 128.0
        q = deadzone_quantize(apply_dct2(ch), Q, deadzone)
        all_q[i] = q.flatten()[zz]

    # DC delta coding
    dc = all_q[:, 0].astype(np.int32)
    dc_delta = np.diff(dc, prepend=0).astype(np.int16)  # DC[0] pozostaje, reszta to delta

    # AC: sparse column-major storage
    # all_q[:, 1:] — shape (n_blocks, n_coeffs-1)
    ac = all_q[:, 1:]  # (n_blocks, n_ac)
    n_ac = n_coeffs - 1

    out = bytearray()
    out.extend(struct.pack('>HH', n_blocks, bs))
    out.extend(dc_delta.tobytes())  # int16 * n_blocks

    # Dla każdej pozycji AC: sparse list (block_idx, value)
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


def decode_planar_channel(data: bytes, Q: float, is_luma_intra: bool = False) -> list:
    """Zwraca listę tablic (bs, bs) float32"""
    off = 0
    n_blocks, bs = struct.unpack_from('>HH', data, off); off += 4
    if n_blocks == 0:
        return []

    n_coeffs = bs * bs
    n_ac = n_coeffs - 1
    zz = get_zigzag_indices(bs)
    zz_inv = np.argsort(zz)

    # DC delta decode
    dc_delta = np.frombuffer(data, dtype=np.int16, count=n_blocks, offset=off).astype(np.int32)
    off += n_blocks * 2
    dc = np.cumsum(dc_delta).astype(np.int16)

    # Rekonstrukcja macierzy all_q
    all_q = np.zeros((n_blocks, n_coeffs), dtype=np.int16)
    all_q[:, 0] = dc

    for ac_pos in range(n_ac):
        nz_count = struct.unpack_from('>H', data, off)[0]; off += 2
        if nz_count > 0:
            nz_idx = np.frombuffer(data, dtype=np.uint16, count=nz_count, offset=off); off += nz_count * 2
            nz_val = np.frombuffer(data, dtype=np.int16,  count=nz_count, offset=off); off += nz_count * 2
            all_q[nz_idx, ac_pos + 1] = nz_val

    # IDCT dla każdego bloku
    result = []
    for i in range(n_blocks):
        zz_coeffs = all_q[i]
        # Odwrócenie zigzag
        flat = np.zeros(n_coeffs, dtype=np.float32)
        flat[zz_inv] = zz_coeffs.astype(np.float32)
        blk = apply_idct2(flat.reshape(bs, bs) * Q)
        if is_luma_intra:
            blk += 128.0
        result.append(blk)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PACK/UNPACK: Całe zdarzenie (3 kanały) → planar bytes + zstd
# ═══════════════════════════════════════════════════════════════════════════════

def pack_frame_planar(blocks_ycbcr: list, bs: int, q_y: float, q_c: float,
                       deadzone: float, is_intra: bool = False) -> bytes:
    """
    blocks_ycbcr: lista (bs, bs, 3) float32
    is_intra: True dla intra bloków (Y -= 128)
    Zwraca: raw bytes (3 sekcje planar, bez zstd — zstd odbywa się na poziomie klatki)
    """
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
    """Zwraca listę (bs, bs, 3) float32"""
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
# FIZYKA: INTERFERENCJA I KORELACJA OPTYCZNA (bez zmian)
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
# SERIALIZACJA KLATKI
# Nowy format: per-frame payload = [n_intra_blocks | n_inter_blocks | intra_planar | inter_planar | mv_table]
# MV table: per inter-block (x, y, dx, dy) uint16/int8
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_v13(paths: list, bs: int, q_y: float, q_c: float, deadzone: float,
                         h: int, w: int) -> bytes:
    """
    paths: lista dict jak w v1.2
    Zwraca surowy (nieskompresowany) payload klatki.
    
    Format:
      uint8  frame_type: 0=I-frame, 1=P-frame, 2=static (pusta)
    
    I-frame:
      float32[h*w*3] raw wave
    
    P-frame:
      uint16 n_intra
      uint16 n_inter
      -- jeśli n_intra > 0:
         uint16[n_intra*2] block_positions (x, y)  — raster sorted
         bytes planar_intra (3 sekcje)
      -- jeśli n_inter > 0:
         uint16[n_inter*4] mv_table (x, y, dx, dy packed)  
         bytes planar_inter (3 sekcje)
    
    static:
      (puste)
    """
    out = bytearray()

    if not paths:
        out.append(2)  # static
        return bytes(out)

    if len(paths) == 1 and paths[0]['type'] == 0:
        out.append(0)  # I-frame
        out.extend(paths[0]['data'])
        return bytes(out)

    out.append(1)  # P-frame

    intra_paths = [p for p in paths if p['type'] == 2]
    inter_paths = [p for p in paths if p['type'] == 1]

    out.extend(struct.pack('>HH', len(intra_paths), len(inter_paths)))

    # INTRA blocks (type 2)
    if intra_paths:
        for p in intra_paths:
            out.extend(struct.pack('>HH', p['x'], p['y']))
        intra_blocks = [p['_block'] for p in intra_paths]
        intra_planar = pack_frame_planar(intra_blocks, bs, q_y, q_c, deadzone, is_intra=True)
        out.extend(intra_planar)

    # INTER blocks (type 1): MV table + planar delta
    if inter_paths:
        for p in inter_paths:
            # x, y: uint16; dx, dy: int8 (clamp do ±127)
            dx_c = max(-127, min(127, p['dx']))
            dy_c = max(-127, min(127, p['dy']))
            out.extend(struct.pack('>HHbb', p['x'], p['y'], dx_c, dy_c))
        inter_blocks = [p['_block'] for p in inter_paths]
        inter_planar = pack_frame_planar(inter_blocks, bs, q_y, q_c, deadzone, is_intra=False)
        out.extend(inter_planar)

    return bytes(out)


def deserialize_frame_v13(data: bytes, bs: int, q_y: float, q_c: float) -> list:
    """Zwraca listę dict jak v1.2 paths, ale _block już zdekodowany"""
    off = 0
    frame_type = data[off]; off += 1

    if frame_type == 2:  # static
        return []

    if frame_type == 0:  # I-frame
        return [{'type': 0, 'data': data[off:]}]

    # P-frame
    n_intra, n_inter = struct.unpack_from('>HH', data, off); off += 4
    paths = []

    if n_intra > 0:
        positions = []
        for _ in range(n_intra):
            x, y = struct.unpack_from('>HH', data, off); off += 4
            positions.append((x, y))

        # Find end of planar_intra: musimy sparsować 3 sekcje
        planar_start = off
        for _ in range(3):
            sz = struct.unpack_from('>I', data, off)[0]; off += 4 + sz

        intra_blocks = unpack_frame_planar(data[planar_start:off], q_y, q_c, is_intra=True)
        for i, (x, y) in enumerate(positions):
            paths.append({'type': 2, 'x': x, 'y': y, 'dx': 0, 'dy': 0,
                          'bh': bs, 'bw': bs, '_block': intra_blocks[i]})

    if n_inter > 0:
        mv_list = []
        for _ in range(n_inter):
            x, y, dx, dy = struct.unpack_from('>HHbb', data, off); off += 6
            mv_list.append((x, y, dx, dy))

        planar_start = off
        for _ in range(3):
            sz = struct.unpack_from('>I', data, off)[0]; off += 4 + sz

        inter_blocks = unpack_frame_planar(data[planar_start:off], q_y, q_c, is_intra=False)
        for i, (x, y, dx, dy) in enumerate(mv_list):
            paths.append({'type': 1, 'x': x, 'y': y, 'dx': dx, 'dy': dy,
                          'bh': bs, 'bw': bs, '_block': inter_blocks[i]})

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# KODEK GŁÓWNY
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
        """Zwraca listę paths z dodatkowym kluczem '_block' (gotowy blok float32)"""
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

        # Brzegi (niekwadratowe) — float16 fallback jak w v1.2
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
                            paths.append({'type': 1, 'x': ex, 'y': ey, 'dx': dx, 'dy': dy,
                                          'bh': bh, 'bw': bw,
                                          'data': (delta * delta_mask[..., np.newaxis]).astype(np.float16).tobytes()})
                    else:
                        paths.append({'type': 2, 'x': ex, 'y': ey, 'dx': 0, 'dy': 0,
                                      'bh': bh, 'bw': bw,
                                      'data': brush.astype(np.float16).tobytes()})

        self.prev_wave = wave3
        return paths

    def decode_frame(self, paths: list, h: int, w: int) -> np.ndarray:
        if self.prev_wave is None:
            self.prev_wave = (
                np.frombuffer(paths[0]['data'], dtype=np.float32).reshape(h, w, 3)
            )
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
# NOWOŚĆ v1.3: ZSTD DICTIONARY — kontekst temporalny
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalDictBuilder:
    """
    Zbiera skompresowane payloady poprzednich klatek i trenuje słownik zstd.
    Nowy słownik co _DICT_FRAMES klatek.
    """
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
            self._dict_data = None  # Fallback: brak słownika

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
# I/O — Zaktualizowane pod v1.3
# Format pliku: MAGIC + (w, h, fps) + ramki
# Każda ramka: uint8 has_dict_update | [uint32 dict_len + dict_bytes] | uint32 payload_len + payload
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def photonic_writer(path, w, h, fps, bs, q_y, q_c, deadzone):
    f = open(path, 'wb')
    f.write(_MAGIC)
    f.write(struct.pack('>HHfHfff', w, h, fps, bs, q_y, q_c, deadzone))
    dict_builder = TemporalDictBuilder()
    codec_enc = PhotonicCodec.__new__(PhotonicCodec)
    codec_enc.__init__(brush_size=bs, q_y=q_y, q_c=q_c, deadzone=deadzone)

    try:
        class W:
            def __init__(self):
                self.f = f
                self.db = dict_builder
                self.codec = codec_enc
                self._frame_idx = 0

            def write_frame(self, frame_rgb: np.ndarray):
                paths = self.codec.encode_frame(frame_rgb)
                raw = serialize_frame_v13(paths, bs, q_y, q_c, deadzone,
                                          frame_rgb.shape[0], frame_rgb.shape[1])
                self.db.feed(raw)
                comp = self.db.get_compressor().compress(raw)

                # Czy emitujemy nowy słownik w tym miejscu?
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
            raise ValueError(f"Zły format pliku (magic={m!r}, oczekiwano {_MAGIC})")
        w, h, fps, bs, q_y, q_c, deadzone = struct.unpack('>HHfHfff', f.read(18))
        current_dict: bytes | None = None

        class R:
            def __init__(self):
                self.f = f
                self.w = w; self.h = h; self.fps = fps
                self.bs = bs; self.q_y = q_y; self.q_c = q_c
                self._dict_data = None

            def read(self) -> list | None:
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

                if self._dict_data:
                    d = zstd.ZstdDecompressor(dict_data=self._dict_data)
                else:
                    d = zstd.ZstdDecompressor()
                raw = d.decompress(comp)
                return deserialize_frame_v13(raw, bs, q_y, q_c)

            def close(self): self.f.close()

        yield R()
    finally:
        if not f.closed: f.close()



# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.3] Inicjalizacja z planarnym DCT + temporal dict...", flush=True)

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

    print(f"\n[Photonic] Zapisano {fi} klatek.", flush=True)


def decode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.3] Odtwarzanie...", flush=True)

    with photonic_reader(inp) as reader:
        codec = PhotonicCodec(brush_size=reader.bs, q_y=reader.q_y,
                              q_c=reader.q_c, deadzone=args.deadzone)
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
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.3")
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')

    parser.add_argument('--brush-size',      type=int,   default=32)
    parser.add_argument('--fps',             type=float, default=25.0)
    parser.add_argument('--ncc-threshold',   type=float, default=0.85)
    parser.add_argument('--delta-threshold', type=float, default=5.0)
    parser.add_argument('--q-y',             type=float, default=8.0)
    parser.add_argument('--q-c',             type=float, default=20.0)
    parser.add_argument('--deadzone',        type=float, default=0.35)

    args = parser.parse_args()

    if args.decode:
        decode_video(args.input, args.output, args)
    else:
        encode_video(args.input, args.output, args)
