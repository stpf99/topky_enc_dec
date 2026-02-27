#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   QDIFF CODEC v0.7 — UNIFIED VERSION                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Połączone ulepszenia z v0.2 + v0.5_strip + v0.6:                           ║
║                                                                              ║
║  Z v0.5_strip:                                                              ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  1. RLE z LENGTH-PREFIX (2B) — poprawna deserializacja                      ║
║  2. SPARSE ENCODING dla bloków <25% niezerowych                             ║
║  3. AUTO-MODE z per-frame parameter tuning                                  ║
║  4. SOURCE VIDEO PARAMETER EXTRACTION                                       ║
║  5. VFR z frame similarity detection                                        ║
║  6. DIAMOND SEARCH z early termination                                      ║
║  7. I-FRAME QUALITY BOOST                                                   ║
║  8. BACKWARD COMPATIBILITY (QDIF, QDF2, QDF3, QDF5, QDF6)                  ║
║                                                                              ║
║  Z v0.2 (nowe):                                                             ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  9. DELTA MV CODING z predykcją przestrzenną                                ║
║  10. ZIGZAG SCAN dla DCT coefficients                                       ║
║  11. ZSTD LEVEL 19 + THREADING                                              ║
║  12. ZWIĘKSZONE domyślne parametry (search=32, keyframe=100)               ║
║                                                                              ║
║  Format pliku: .qdiff (magic: QDF7)                                         ║
║  Uruchomienie:                                                               ║
║    python qdiff_codec_v0.7.py -i input.mp4 -o output.qdiff -a --vfr        ║
║    python qdiff_codec_v0.7.py -i output.qdiff -o decoded.mp4 -d            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.fftpack import dct, idct
import zstandard as zstd
import struct
import os
import sys
import time
import argparse
import concurrent.futures
import threading
import json
from collections import deque
from numpy.lib.stride_tricks import sliding_window_view

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA GLOBALNA
# ─────────────────────────────────────────────────────────────────────────────
_N_WORKERS   = int(os.environ.get('QDIFF_WORKERS', os.cpu_count() or 4))
_VLC_ENABLED = os.environ.get('QDIFF_VLC', '1') not in ('0', 'false', 'off', 'no')
_ZSTD_LEVEL  = int(os.environ.get('QDIFF_ZSTD_LEVEL', 19))

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE QDIFF
# ─────────────────────────────────────────────────────────────────────────────
QD_SKIP_TRUE   = 0b000
QD_SKIP_NOISE  = 0b001
QD_MV_ONLY     = 0b010
QD_DC_ONLY     = 0b011
QD_LOW_FREQ    = 0b100
QD_FULL_DCT    = 0b101
QD_INTRA_PATCH = 0b110
QD_SCENE_CUT   = 0b111

FAMILY_NAMES = {
    QD_SKIP_TRUE:   "SKIP_TRUE",
    QD_SKIP_NOISE:  "SKIP_NOISE",
    QD_MV_ONLY:     "MV_ONLY",
    QD_DC_ONLY:     "DC_ONLY",
    QD_LOW_FREQ:    "LOW_FREQ",
    QD_FULL_DCT:    "FULL_DCT",
    QD_INTRA_PATCH: "INTRA",
    QD_SCENE_CUT:   "SCENE_CUT",
}

_LOWFREQ_IDX = [(0,0), (0,1), (1,0), (2,0)]

# Zigzag scan order dla 8x8
_ZIGZAG_8x8 = [
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE PARAMETRÓW AUTO-MODE
# ─────────────────────────────────────────────────────────────────────────────
PARAM_Q_Y        = 0x01
PARAM_Q_C        = 0x02
PARAM_SR         = 0x03
PARAM_INTRA_F    = 0x04
PARAM_SUBPIXEL   = 0x05
PARAM_MAD_AVG    = 0x06
PARAM_COMPLEXITY = 0x07
PARAM_SCENE_DIFF = 0x08
PARAM_IS_IFRAME  = 0x09

PARAM_NAMES = {
    PARAM_Q_Y:        "Q_Y",
    PARAM_Q_C:        "Q_C",
    PARAM_SR:         "SEARCH_RANGE",
    PARAM_INTRA_F:    "INTRA_FACTOR",
    PARAM_SUBPIXEL:   "SUBPIXEL",
    PARAM_MAD_AVG:    "MAD_AVG",
    PARAM_COMPLEXITY: "COMPLEXITY",
    PARAM_SCENE_DIFF: "SCENE_DIFF",
    PARAM_IS_IFRAME:  "IS_IFRAME",
}

AUTO_Q_Y_MIN, AUTO_Q_Y_MAX = 12.0, 45.0
AUTO_Q_C_MIN, AUTO_Q_C_MAX = 22.0, 70.0
AUTO_SR_MIN, AUTO_SR_MAX = 12, 48
AUTO_INTRA_F_MIN, AUTO_INTRA_F_MAX = 2.5, 6.0

DEFAULT_DROP_THRESHOLD = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# MAGIC NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
_QDIFF_MAGIC    = b'QDIF'  # v0.1
_QDIFF_MAGIC_V2 = b'QDF2'  # v0.2 auto
_QDIFF_MAGIC_V3 = b'QDF3'  # v0.3 VFR
_QDIFF_MAGIC_V5 = b'QDF5'  # v0.5 STRIP v2
_QDIFF_MAGIC_V6 = b'QDF6'  # v0.6 STRIP v3 (poprawione ujemne)
_QDIFF_MAGIC_V7 = b'QDF7'  # v0.7 UNIFIED (obecna wersja)
_FRAME_I        = b'I'
_FRAME_P        = b'P'
_FRAME_DROPPED  = b'D'
_EOF_MARKER     = b'\xFF\xFF'


# ═══════════════════════════════════════════════════════════════════════════════
# RLE ENCODING (z v0.5_strip + poprawka v0.6 dla ujemnych)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_rle_int16(arr: np.ndarray) -> bytes:
    """
    Koduje tablicę int16 z RLE dla sekwencji zer.
    
    Format v0.7 (poprawiony):
      - Nagłówek (2B): długość zakodowanych danych
      - 0x00-0x7F: wartości 0-127 (bezpośrednio)
      - 0x80-0xFD: sekwencja zer, count = (byte & 0x7F) + 1 (1-126 zer)
      - 0xFE: duża/ujemna wartość następuje (3 bajty: 0xFE + int16 BE)
    
    OSZCZĘDNOŚĆ: 40-70% dla typowych bloków DCT po kwantyzacji.
    """
    out = bytearray()
    flat = arr.flatten()
    n = len(flat)
    i = 0
    
    rle_data = bytearray()
    
    while i < n:
        val = int(flat[i])
        
        if val == 0:
            # Policz sekwencję zer
            run_start = i
            while i < n and int(flat[i]) == 0:
                i += 1
            run_len = i - run_start
            
            # Koduj run
            while run_len > 0:
                chunk = min(run_len, 126)
                rle_data.append(0x80 + chunk - 1)
                run_len -= chunk
        else:
            # Wartość niezerowa
            if 0 <= val <= 127:
                rle_data.append(val)
            else:
                # Ujemna lub duża wartość - użyj markera 0xFE + int16 BE
                rle_data.append(0xFE)
                rle_data.extend(struct.pack('>h', val))
            i += 1
    
    # Nagłówek: długość danych RLE (2 bajty BE)
    out.extend(struct.pack('>H', len(rle_data)))
    out.extend(rle_data)
    
    return bytes(out)


def decode_rle_int16(data: bytes, expected_len: int, offset: int = 0) -> tuple:
    """
    Dekoduje RLE-encoded int16 array.
    
    Returns: (np.ndarray, new_offset)
    """
    rle_len = struct.unpack_from('>H', data, offset)[0]
    offset += 2
    rle_end = offset + rle_len
    
    out = []
    i = offset
    
    while len(out) < expected_len and i < rle_end:
        b = data[i]
        i += 1
        
        if b == 0xFE:
            # Duża/ujemna wartość
            val = struct.unpack_from('>h', data, i)[0]
            i += 2
            out.append(val)
        elif b >= 0x80:
            # Sekwencja zer
            count = (b & 0x7F) + 1
            out.extend([0] * count)
        else:
            # Mała wartość dodatnia
            out.append(b)
    
    # Pad zera
    while len(out) < expected_len:
        out.append(0)
    
    return np.array(out[:expected_len], dtype=np.int16), rle_end


# ═══════════════════════════════════════════════════════════════════════════════
# SPARSE ENCODING (z v0.5_strip)
# ═══════════════════════════════════════════════════════════════════════════════

def encode_sparse_int16(arr: np.ndarray) -> bytes:
    """Sparse encoding dla bloków z <25% niezerowych."""
    flat = arr.flatten()
    n = len(flat)
    nz_mask = flat != 0
    nz_count = int(np.sum(nz_mask))
    
    out = bytearray()
    
    if nz_count == 0:
        out.append(0x00)
        return bytes(out)
    
    if nz_count > n // 4:
        return None  # Sygnał do użycia RLE
    
    if nz_count < 255:
        out.append(nz_count)
    else:
        out.append(0xFF)
        out.extend(struct.pack('>H', nz_count))
    
    for pos in np.where(nz_mask)[0]:
        if n > 255:
            out.extend(struct.pack('>H', pos))
        else:
            out.append(pos)
        out.extend(struct.pack('>h', int(flat[pos])))
    
    return bytes(out)


def decode_sparse_int16(data: bytes, expected_len: int, offset: int = 0) -> tuple:
    out = np.zeros(expected_len, dtype=np.int16)
    i = offset
    
    nz_count = data[i]
    i += 1
    
    if nz_count == 0xFF:
        nz_count = struct.unpack_from('>H', data, i)[0]
        i += 2
    
    use_2b_pos = expected_len > 255
    
    for _ in range(nz_count):
        if use_2b_pos:
            pos = struct.unpack_from('>H', data, i)[0]
            i += 2
        else:
            pos = data[i]
            i += 1
        
        val = struct.unpack_from('>h', data, i)[0]
        i += 2
        
        if pos < expected_len:
            out[pos] = val
    
    return out, i


def encode_dct_block_optimized(q_dct: np.ndarray) -> bytes:
    """
    Inteligentne kodowanie bloku DCT:
      - 0x00: wszystkie zera (1B)
      - 0x01: tylko DC (3B)
      - 0x02: sparse encoding
      - 0x03: RLE encoding
    """
    flat = q_dct.flatten()
    n = len(flat)
    nz_count = int(np.count_nonzero(flat))
    
    if nz_count == 0:
        return b'\x00'
    
    if nz_count == 1 and flat[0] != 0:
        return b'\x01' + struct.pack('>h', int(flat[0]))
    
    if nz_count < n // 4:
        sparse_data = encode_sparse_int16(q_dct)
        if sparse_data is not None:
            return b'\x02' + sparse_data
    
    rle_data = encode_rle_int16(q_dct)
    return b'\x03' + rle_data


def decode_dct_block_optimized(data: bytes, block_size: int, offset: int = 0) -> tuple:
    n = block_size * block_size
    encoding_type = data[offset]
    offset += 1
    
    if encoding_type == 0x00:
        return np.zeros((block_size, block_size), dtype=np.int16), offset
    
    elif encoding_type == 0x01:
        dc_val = struct.unpack_from('>h', data, offset)[0]
        offset += 2
        out = np.zeros((block_size, block_size), dtype=np.int16)
        out[0, 0] = dc_val
        return out, offset
    
    elif encoding_type == 0x02:
        arr, offset = decode_sparse_int16(data, n, offset)
        return arr.reshape(block_size, block_size), offset
    
    else:  # 0x03 = RLE
        arr, offset = decode_rle_int16(data, n, offset)
        return arr.reshape(block_size, block_size), offset


# ═══════════════════════════════════════════════════════════════════════════════
# DELTA MV CODING (NOWOŚĆ v0.2)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_mv(mv_left: tuple, mv_top: tuple, mv_topleft: tuple) -> tuple:
    """Predyktor medianowy dla MV (podobnie jak H.264)."""
    if mv_left is None: mv_left = (0, 0)
    if mv_top is None: mv_top = (0, 0)
    if mv_topleft is None: mv_topleft = (0, 0)
    
    dx = int(np.median([mv_left[0], mv_top[0], mv_topleft[0]]))
    dy = int(np.median([mv_left[1], mv_top[1], mv_topleft[1]]))
    return (dx, dy)


def encode_mv_delta(mv: tuple, predictor: tuple) -> tuple:
    return (mv[0] - predictor[0], mv[1] - predictor[1])


def decode_mv_delta(delta: tuple, predictor: tuple) -> tuple:
    return (delta[0] + predictor[0], delta[1] + predictor[1])


def _pack_mv_delta(dx: int, dy: int) -> bytes:
    """
    Pakuje deltę MV (różnicę wektora ruchu po predykcji).
    
    Format:
      - Jeśli -126 <= dx,dy <= 126: 2 bajty (signed bytes)
      - W przeciwnym razie: 5 bajtów (escape 0x81 + 2×int16 BE)
    """
    if -126 <= dx <= 126 and -126 <= dy <= 126:
        return struct.pack('bb', dx, dy)
    else:
        # Escape: 0x81 + 2×int16 BE
        return struct.pack('B', 0x81) + struct.pack('>hh', dx, dy)


def _unpack_mv_delta(data: bytes, offset: int) -> tuple:
    """Rozpakowuje deltę MV."""
    b1 = data[offset]
    
    # Escape format (0x81 = 129)
    if b1 == 0x81:
        dx, dy = struct.unpack_from('>hh', data, offset + 1)
        return (dx, dy), offset + 5
    
    # Format 2-bajtowy (signed bytes)
    dx = struct.unpack_from('b', data, offset)[0]
    dy = struct.unpack_from('b', data, offset + 1)[0]
    return (dx, dy), offset + 2


# ═══════════════════════════════════════════════════════════════════════════════
# ZIGZAG SCAN (NOWOŚĆ v0.2)
# ═══════════════════════════════════════════════════════════════════════════════

def zigzag_encode_block(block: np.ndarray, block_size: int = 8) -> list:
    """Konwertuje blok na listę wartości w porządku zigzag."""
    if block_size == 8:
        flat = block.flatten()
        return [flat[_ZIGZAG_8x8[i]] for i in range(64)]
    elif block_size == 16:
        result = []
        for by in range(2):
            for bx in range(2):
                sub = block[by*8:(by+1)*8, bx*8:(bx+1)*8]
                result.extend(zigzag_encode_block(sub, 8))
        return result
    return block.flatten().tolist()


def zigzag_decode_block(values: list, block_size: int = 8) -> np.ndarray:
    """Odtwarza blok z wartości w porządku zigzag."""
    if block_size == 8:
        flat = np.zeros(64, dtype=np.int16)
        for i, v in enumerate(values[:64]):
            flat[_ZIGZAG_8x8[i]] = v
        return flat.reshape(8, 8)
    elif block_size == 16:
        block = np.zeros((16, 16), dtype=np.int16)
        idx = 0
        for by in range(2):
            for bx in range(2):
                sub_vals = values[idx:idx+64]
                sub = zigzag_decode_block(sub_vals, 8)
                block[by*8:(by+1)*8, bx*8:(bx+1)*8] = sub
                idx += 64
        return block
    return np.array(values[:block_size*block_size]).reshape(block_size, block_size)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH DCT
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_dct2_scipy(blocks):
    b = blocks.astype(np.float32)
    return dct(dct(b, type=2, norm='ortho', axis=-1), type=2, norm='ortho', axis=-2)

def _batch_idct2_scipy(blocks):
    b = blocks.astype(np.float32)
    return idct(idct(b, type=2, norm='ortho', axis=-1), type=2, norm='ortho', axis=-2)


def _try_setup_pyfftw():
    try:
        import pyfftw
        import pyfftw.interfaces.scipy_fftpack as fftwsp
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)
        pyfftw.config.NUM_THREADS = _N_WORKERS
        import inspect as _ins
        _has_workers = 'workers' in _ins.signature(fftwsp.dct).parameters
        if _has_workers:
            _dct_fn  = lambda x, ax: fftwsp.dct(x, type=2, norm='ortho', axis=ax, workers=_N_WORKERS)
            _idct_fn = lambda x, ax: fftwsp.idct(x, type=2, norm='ortho', axis=ax, workers=_N_WORKERS)
        else:
            _dct_fn  = lambda x, ax: fftwsp.dct(x, type=2, norm='ortho', axis=ax)
            _idct_fn = lambda x, ax: fftwsp.idct(x, type=2, norm='ortho', axis=ax)
        def _batch_dct2_fftw(blocks):
            return _dct_fn(_dct_fn(blocks.astype(np.float32), -1), -2)
        def _batch_idct2_fftw(blocks):
            return _idct_fn(_idct_fn(blocks.astype(np.float32), -1), -2)

        _test_blk = np.ones((4, 16, 16), dtype=np.float32) * 128.0
        _test_dct = _batch_dct2_fftw(_test_blk)
        _test_rec = _batch_idct2_fftw(_test_dct)
        _err = float(np.max(np.abs(_test_rec - _test_blk)))
        if _err > 1.0:
            return _batch_dct2_scipy, _batch_idct2_scipy, "scipy.fftpack (pyfftw IDCT error)"

        return _batch_dct2_fftw, _batch_idct2_fftw, f"pyfftw (workers={_N_WORKERS})"
    except ImportError:
        pass
    except Exception as e:
        print(f"[qdiff] pyfftw błąd: {e} → fallback scipy", flush=True)

    return _batch_dct2_scipy, _batch_idct2_scipy, "scipy.fftpack"


batch_dct2, batch_idct2, _DCT_BACKEND = _try_setup_pyfftw()


def plane_to_blocks(plane: np.ndarray, bs: int) -> np.ndarray:
    H, W = plane.shape
    Hb, Wb = H // bs, W // bs
    p = np.ascontiguousarray(plane[:Hb * bs, :Wb * bs], dtype=np.float32)
    return p.reshape(Hb, bs, Wb, bs).transpose(0, 2, 1, 3).reshape(-1, bs, bs)


def blocks_to_plane(blocks: np.ndarray, H: int, W: int, bs: int) -> np.ndarray:
    Hb, Wb = H // bs, W // bs
    return (blocks.reshape(Hb, Wb, bs, bs)
                  .transpose(0, 2, 1, 3)
                  .reshape(Hb * bs, Wb * bs))


def apply_dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.astype(np.float32).T, norm='ortho').T, norm='ortho')


def apply_idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.astype(np.float32).T, norm='ortho').T, norm='ortho')


# ═══════════════════════════════════════════════════════════════════════════════
# YUV 4:2:0
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_yuv420(rgb: np.ndarray):
    m = np.array([[ 0.299,  0.587,  0.114],
                  [-0.169, -0.331,  0.500],
                  [ 0.500, -0.419, -0.081]])
    yuv = np.dot(rgb, m.T)
    Y = yuv[:, :, 0]
    U = yuv[::2, ::2, 1]
    V = yuv[::2, ::2, 2]
    return Y, U, V


_YUV2RGB = np.array([[1.0,  0.0,    1.402],
                     [1.0, -0.344, -0.714],
                     [1.0,  1.772,  0.0  ]], dtype=np.float32)


def _upsample_plane(plane: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    H, W = h * 2, w * 2
    p = plane.astype(np.float32, copy=False)
    out = np.empty((H, W), dtype=np.float32)
    out[0::2, 0::2] = p
    out[0::2, 1:W-1:2] = (p[:, :-1] + p[:, 1:]) * 0.5
    out[0::2, W-1] = p[:, -1]
    top, bottom = out[0:H-2:2, :], out[2:H:2, :]
    out[1:H-1:2, :] = (top + bottom) * 0.5
    out[H-1, :] = out[H-2, :]
    yr_odd = np.arange(1, H-1, 2)
    mask = yr_odd >= 2
    if mask.any():
        yo = yr_odd[mask]
        corr = out[yo,:] + (out[yo,:] - (out[yo-2,:] + out[yo-1,:])*0.5) * 0.15
        out[yo, :] = np.clip(corr, -128.0, 127.0)
    return np.clip(out, -128.0, 127.0)


def yuv420_to_rgb(Y, U, V) -> np.ndarray:
    h, w = Y.shape
    U_f = _upsample_plane(U)[:h, :w]
    V_f = _upsample_plane(V)[:h, :w]
    yuv = np.stack((Y.astype(np.float32, copy=False), U_f, V_f), axis=-1)
    return np.clip(yuv @ _YUV2RGB.T, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-PIXEL INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

def _try_compile_numba():
    try:
        from numba import njit
        @njit(cache=False, fastmath=True, parallel=False)
        def _interp_numba(plane, y_fp, x_fp, block_size):
            h, w = plane.shape
            y_f = y_fp / 4.0; x_f = x_fp / 4.0
            y0 = int(y_f); x0 = int(x_f)
            y1 = min(y0+1, h-1); x1 = min(x0+1, w-1)
            y0 = max(0, min(y0, h-block_size-1)); x0 = max(0, min(x0, w-block_size-1))
            y1 = max(0, min(y1, h-block_size-1)); x1 = max(0, min(x1, w-block_size-1))
            fy = y_f - int(y_f); fx = x_f - int(x_f)
            result = np.empty((block_size, block_size), dtype=np.float32)
            for r in range(block_size):
                for c in range(block_size):
                    result[r, c] = (plane[y0+r, x0+c]*(1-fy)*(1-fx) +
                                    plane[y1+r, x0+c]*fy*(1-fx) +
                                    plane[y0+r, x1+c]*(1-fy)*fx +
                                    plane[y1+r, x1+c]*fy*fx)
            return result
        threading.Thread(target=lambda: _interp_numba(np.zeros((32,32),np.float32),0,0,8),
                         daemon=True).start()
        return _interp_numba, "numba JIT"
    except Exception:
        return None, "numpy (numba niedostępne)"


_NUMBA_INTERP, _NUMBA_INFO = _try_compile_numba()


def interpolate_subpixel(plane, y_fp, x_fp, block_size):
    if _NUMBA_INTERP is not None:
        return _NUMBA_INTERP(plane, y_fp, x_fp, block_size)
    h, w = plane.shape
    y_f = y_fp / 4.0; x_f = x_fp / 4.0
    y0 = max(0, min(int(np.floor(y_f)), h-block_size-1))
    x0 = max(0, min(int(np.floor(x_f)), w-block_size-1))
    y1 = max(0, min(y0+1, h-block_size-1))
    x1 = max(0, min(x0+1, w-block_size-1))
    fy = y_f - np.floor(y_f); fx = x_f - np.floor(x_f)
    b00 = plane[y0:y0+block_size, x0:x0+block_size].astype(np.float32)
    b10 = plane[y1:y1+block_size, x0:x0+block_size].astype(np.float32)
    b01 = plane[y0:y0+block_size, x1:x1+block_size].astype(np.float32)
    b11 = plane[y1:y1+block_size, x1:x1+block_size].astype(np.float32)
    return b00*(1-fy)*(1-fx) + b10*fy*(1-fx) + b01*(1-fy)*fx + b11*fy*fx


# ═══════════════════════════════════════════════════════════════════════════════
# DEBLOCK FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def deblock_filter(plane: np.ndarray, block_size: int = 16, threshold: float = 15.0) -> np.ndarray:
    h, w = plane.shape
    filtered = plane.astype(np.float32)
    for x in range(block_size, w, block_size):
        left, right = filtered[:, x-1], filtered[:, x]
        diff = np.abs(left - right); mask = diff < threshold
        avg = (left + right) * 0.5
        filtered[:, x-1] = np.where(mask, avg, left)
        filtered[:, x] = np.where(mask, avg, right)
    for y in range(block_size, h, block_size):
        top, bottom = filtered[y-1, :], filtered[y, :]
        diff = np.abs(top - bottom); mask = diff < threshold
        avg = (top + bottom) * 0.5
        filtered[y-1, :] = np.where(mask, avg, top)
        filtered[y, :] = np.where(mask, avg, bottom)
    return np.clip(filtered, 0, 255).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE VIDEO PARAMETER EXTRACTION (z v0.5_strip)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_source_video_params(input_path: str) -> dict:
    """Wyciąga parametry źródłowego wideo."""
    params = {
        'fps': 25.0, 'crf': 18, 'preset': 'medium', 'codec': 'libx264',
        'width': 0, 'height': 0, 'bitrate': None, 'duration': 0.0, 'n_frames': 0,
    }

    try:
        import imageio.v3 as _iio
        props = _iio.improps(input_path, plugin='pyav')
        if hasattr(props, 'fps') and props.fps:
            params['fps'] = float(props.fps)
        if hasattr(props, 'n_images'):
            params['n_frames'] = props.n_images
        if hasattr(props, 'duration'):
            params['duration'] = props.duration
        if hasattr(props, 'shape'):
            params['height'], params['width'] = props.shape[:2]

        try:
            import av
            container = av.open(input_path)
            stream = container.streams.video[0]
            params['codec'] = stream.codec_context.name if hasattr(stream.codec_context, 'name') else 'libx264'
            params['width'] = stream.width
            params['height'] = stream.height
            params['fps'] = float(stream.average_rate) if stream.average_rate else params['fps']
            params['n_frames'] = stream.frames if stream.frames else params['n_frames']
            if stream.bit_rate:
                params['bitrate'] = stream.bit_rate
            container.close()
        except: pass
    except: pass

    # Fallback: ffprobe
    if params['width'] == 0:
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', '-show_streams', input_path],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        params['width'] = stream.get('width', 0)
                        params['height'] = stream.get('height', 0)
                        params['codec'] = stream.get('codec_name', 'libx264')
                        fps_str = stream.get('r_frame_rate', '25/1')
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            params['fps'] = float(num) / float(den) if float(den) > 0 else 25.0
                        break
        except: pass

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# VFR FRAME SIMILARITY (z v0.5_strip)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1 is None or frame2 is None:
        return 0.0
    h = min(frame1.shape[0], frame2.shape[0])
    w = min(frame1.shape[1], frame2.shape[1])
    if len(frame1.shape) == 3:
        Y1 = np.dot(frame1[:h, :w, :3], [0.299, 0.587, 0.114])
    else:
        Y1 = frame1[:h, :w].astype(np.float32)
    if len(frame2.shape) == 3:
        Y2 = np.dot(frame2[:h, :w, :3], [0.299, 0.587, 0.114])
    else:
        Y2 = frame2[:h, :w].astype(np.float32)
    diff = np.abs(Y1 - Y2)
    similar_pixels = np.sum(diff <= 2.0)
    return similar_pixels / (h * w)


def should_drop_frame(frame: np.ndarray, prev_frame: np.ndarray,
                      drop_threshold: float = DEFAULT_DROP_THRESHOLD,
                      similarity_threshold: float = 0.999) -> tuple:
    if prev_frame is None:
        return False, 0.0, {'reason': 'first_frame'}
    similarity = compute_frame_similarity(frame, prev_frame)
    should_drop = similarity >= similarity_threshold
    return should_drop, similarity, {'similarity': similarity, 'reason': 'static' if should_drop else 'changed'}


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-TUNING PARAMETRÓW (z v0.5_strip)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_frame_for_auto_tuning(curr_Y: np.ndarray, prev_Y: np.ndarray = None) -> dict:
    h, w = curr_Y.shape
    variance = float(np.var(curr_Y))
    
    # Edge density
    gx = np.abs(curr_Y[1:-1, 2:] - curr_Y[1:-1, :-2])
    gy = np.abs(curr_Y[2:, 1:-1] - curr_Y[:-2, 1:-1])
    edge_strength = np.sqrt(gx**2 + gy**2)
    edge_density = np.sum(edge_strength > 30) / ((h-2) * (w-2))
    
    metrics = {
        'variance': variance,
        'edge_density': edge_density,
        'mad_avg': 0.0, 'mad_std': 0.0,
        'complexity': min(1.0, variance / 2500.0),
        'motion_ratio': 0.0, 'scene_diff': 0.0,
    }

    if prev_Y is not None:
        diff = np.abs(curr_Y[:prev_Y.shape[0], :prev_Y.shape[1]] - prev_Y)
        mad_map = diff.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))
        metrics['mad_avg'] = float(np.mean(mad_map))
        metrics['mad_std'] = float(np.std(mad_map))
        metrics['motion_ratio'] = np.sum(mad_map > 3.0) / mad_map.size
        metrics['scene_diff'] = metrics['mad_avg']
        metrics['complexity'] = min(1.0, metrics['complexity'] * 0.7 + metrics['motion_ratio'] * 0.3)

    return metrics


def auto_tune_params(metrics: dict, base_params: dict, is_iframe: bool = False,
                     i_frame_quality_boost: float = 0.7) -> dict:
    params = base_params.copy()
    params['auto_mode'] = True
    params['is_iframe'] = is_iframe
    params['mad_avg'] = metrics['mad_avg']
    params['complexity'] = metrics['complexity']
    params['scene_diff'] = metrics['scene_diff']

    q_y_factor = 1.0
    if metrics['mad_avg'] < 3.0:
        q_y_factor = 0.7 + 0.3 * (metrics['mad_avg'] / 3.0)
    elif metrics['mad_avg'] < 10.0:
        q_y_factor = 1.0 + 0.2 * ((metrics['mad_avg'] - 3.0) / 7.0)
    else:
        q_y_factor = 1.2 + 0.3 * min(1.0, (metrics['mad_avg'] - 10.0) / 20.0)

    if metrics['complexity'] > 0.7:
        q_y_factor *= 1.1
    if is_iframe:
        q_y_factor *= i_frame_quality_boost

    params['q_y'] = np.clip(base_params.get('q_y', 22.0) * q_y_factor, AUTO_Q_Y_MIN, AUTO_Q_Y_MAX)
    params['q_c'] = np.clip(base_params.get('q_c', 40.0) * q_y_factor * 1.1, AUTO_Q_C_MIN, AUTO_Q_C_MAX)

    if metrics['motion_ratio'] < 0.1:
        sr_factor = 0.6
    elif metrics['motion_ratio'] < 0.3:
        sr_factor = 1.0
    else:
        sr_factor = 1.0 + 0.5 * (metrics['motion_ratio'] - 0.3)
    params['search_range'] = int(np.clip(base_params.get('search_range', 32) * sr_factor, AUTO_SR_MIN, AUTO_SR_MAX))

    params['intra_factor'] = np.clip(
        base_params.get('intra_factor', 4.0) * (1.0 - 0.2 * metrics['motion_ratio']),
        AUTO_INTRA_F_MIN, AUTO_INTRA_F_MAX)
    params['use_subpixel'] = metrics['mad_avg'] > 2.0

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACJA PARAMETRÓW
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_params(params: dict) -> bytes:
    out = bytearray()
    flags = 0x01 if params.get('auto_mode', False) else 0x00
    flags |= 0x02 if params.get('is_iframe', False) else 0x00
    flags |= 0x04 if params.get('dropped', False) else 0x00
    out.append(flags)

    param_ids = [
        (PARAM_Q_Y, 'q_y'), (PARAM_Q_C, 'q_c'), (PARAM_SR, 'search_range'),
        (PARAM_INTRA_F, 'intra_factor'), (PARAM_SUBPIXEL, 'use_subpixel'),
        (PARAM_MAD_AVG, 'mad_avg'), (PARAM_COMPLEXITY, 'complexity'),
    ]

    params_to_write = [(pid, params[key]) for pid, key in param_ids if key in params]
    out.append(len(params_to_write))

    for pid, value in params_to_write:
        out.append(pid)
        val_f = float(value)
        if val_f == int(val_f) and -128 <= int(val_f) <= 127:
            out.append(0x00)
            out.append(int(val_f) & 0xFF)
        else:
            out.append(0x01)
            out.extend(struct.pack('>f', val_f))

    return bytes(out)


def deserialize_frame_params(data: bytes, offset: int = 0) -> tuple:
    params = {}
    flags = data[offset]
    params['auto_mode'] = bool(flags & 0x01)
    params['is_iframe'] = bool(flags & 0x02)
    params['dropped'] = bool(flags & 0x04)
    offset += 1

    param_count = data[offset]
    offset += 1

    for _ in range(param_count):
        pid = data[offset]; offset += 1
        val_marker = data[offset]; offset += 1
        if val_marker == 0x00:
            value = float(data[offset] if data[offset] < 128 else data[offset] - 256)
            offset += 1
        else:
            value = struct.unpack_from('>f', data, offset)[0]
            offset += 4

        if pid == PARAM_Q_Y: params['q_y'] = value
        elif pid == PARAM_Q_C: params['q_c'] = value
        elif pid == PARAM_SR: params['search_range'] = int(value)
        elif pid == PARAM_INTRA_F: params['intra_factor'] = value
        elif pid == PARAM_SUBPIXEL: params['use_subpixel'] = bool(value > 0.5)
        elif pid == PARAM_MAD_AVG: params['mad_avg'] = value
        elif pid == PARAM_COMPLEXITY: params['complexity'] = value

    return params, offset


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACJA QDIFF BLOKÓW (z RLE + Delta MV)
# ═══════════════════════════════════════════════════════════════════════════════

def _pack_mv(dx: int, dy: int) -> bytes:
    if -126 <= dx <= 126 and -126 <= dy <= 126:
        return struct.pack('bb', dx, dy)
    return struct.pack('b', -127) + struct.pack('>hh', dx, dy)


def _unpack_mv(data: bytes, offset: int) -> tuple:
    dx = struct.unpack_from('b', data, offset)[0]; offset += 1
    if dx == -127:
        dx, dy = struct.unpack_from('>hh', data, offset); offset += 4
    else:
        dy = struct.unpack_from('b', data, offset)[0]; offset += 1
    return (dx, dy), offset


def _pack_dct_lowfreq(q_dct: np.ndarray) -> bytes:
    vals = [int(q_dct[r, c]) for r, c in _LOWFREQ_IDX]
    out = bytearray()
    for v in vals:
        if -127 <= v <= 127:
            out.append(v & 0xFF)
        else:
            out.append(0x80)
            out.extend(struct.pack('>h', v))
    return bytes(out)


def _unpack_dct_lowfreq(data: bytes, offset: int, block_size: int) -> tuple:
    vals = []
    for _ in range(4):
        b = data[offset]; offset += 1
        if b == 0x80:
            v = struct.unpack_from('>h', data, offset)[0]; offset += 2
        else:
            v = b if b < 128 else b - 256
        vals.append(v)
    blk = np.zeros((block_size, block_size), dtype=np.int16)
    for i, (r, c) in enumerate(_LOWFREQ_IDX):
        blk[r, c] = vals[i]
    return blk, offset


def serialize_pframe_blocks_v7(block_list: list, h: int, w: int, params: dict = None,
                                use_delta_mv: bool = True) -> bytes:
    """Serializuje listę bloków QDiff v7 z RLE + Delta MV."""
    bs = 16
    cols = w // bs
    rows = h // bs
    n_blocks = rows * cols

    out = bytearray()
    if params is not None and params.get('auto_mode', False):
        out.extend(serialize_frame_params(params))

    families = np.zeros(n_blocks, dtype=np.uint8)
    block_map = {}
    for blk in block_list:
        x, y = blk['x'], blk['y']
        col, row = x // bs, y // bs
        idx = row * cols + col
        families[idx] = blk['family_id']
        block_map[idx] = blk

    out.extend(struct.pack('>HH', cols, rows))
    out.extend(families.tobytes())

    mv_grid = {}  # Dla delta MV

    for idx in range(n_blocks):
        fid = families[idx]
        if fid in (QD_SKIP_TRUE, QD_SCENE_CUT):
            continue
        blk = block_map.get(idx)
        if blk is None:
            continue
        bs_blk = blk['bs']
        bs_c = bs_blk // 2
        row, col = idx // cols, idx % cols

        if fid == QD_SKIP_NOISE:
            out.extend(struct.pack('B', blk.get('noise_mag', 0)))

        elif fid == QD_MV_ONLY:
            dx_qp, dy_qp = blk['mv_qp']
            mv = (dx_qp // 4, dy_qp // 4)
            if use_delta_mv:
                predictor = predict_mv(mv_grid.get((row, col-1)),
                                       mv_grid.get((row-1, col)),
                                       mv_grid.get((row-1, col-1)))
                delta = encode_mv_delta(mv, predictor)
                out.extend(_pack_mv_delta(delta[0], delta[1]))
                mv_grid[(row, col)] = mv
            else:
                out.extend(_pack_mv(mv[0], mv[1]))

        elif fid == QD_DC_ONLY:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            for plane in ['q_dct_y', 'q_dct_u', 'q_dct_v']:
                dc = int(blk[plane][0, 0])
                if -127 <= dc <= 127:
                    out.append(dc & 0xFF)
                else:
                    out.append(0x80)
                    out.extend(struct.pack('>h', dc))

        elif fid == QD_LOW_FREQ:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            out.extend(_pack_dct_lowfreq(blk['q_dct_y']))
            for plane in ['q_dct_u', 'q_dct_v']:
                dc = int(blk[plane][0, 0])
                if -127 <= dc <= 127:
                    out.append(dc & 0xFF)
                else:
                    out.append(0x80)
                    out.extend(struct.pack('>h', dc))

        elif fid == QD_FULL_DCT:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            out.extend(encode_dct_block_optimized(blk['q_dct_y']))
            out.extend(encode_dct_block_optimized(blk['q_dct_u']))
            out.extend(encode_dct_block_optimized(blk['q_dct_v']))

        elif fid == QD_INTRA_PATCH:
            out.extend(encode_dct_block_optimized(blk['q_dct_y']))
            out.extend(encode_dct_block_optimized(blk['q_dct_u']))
            out.extend(encode_dct_block_optimized(blk['q_dct_v']))

    return bytes(out)


def deserialize_pframe_blocks_v7(data: bytes, offset_in: int = 0, auto_mode: bool = False,
                                  use_delta_mv: bool = True) -> tuple:
    """Deserializuje strumień bajtów v7."""
    offset = offset_in
    params = {}

    if auto_mode:
        params, offset = deserialize_frame_params(data, offset)

    cols, rows = struct.unpack_from('>HH', data, offset); offset += 4
    if cols == 0 or rows == 0:
        return [], params, offset

    n_blocks = rows * cols
    bs = 16; bs_c = 8

    families = np.frombuffer(data[offset:offset + n_blocks], dtype=np.uint8).copy()
    offset += n_blocks

    block_list = []
    mv_grid = {}

    for idx in range(n_blocks):
        row, col = idx // cols, idx % cols
        x, y = col * bs, row * bs
        fid = int(families[idx])
        base = {'x': x, 'y': y, 'bs': bs, 'family_id': fid}

        if fid == QD_SKIP_TRUE:
            block_list.append({**base})
        elif fid == QD_SCENE_CUT:
            block_list.append({**base})
        elif fid == QD_SKIP_NOISE:
            noise_mag = data[offset]; offset += 1
            block_list.append({**base, 'noise_mag': noise_mag})
        elif fid == QD_MV_ONLY:
            if use_delta_mv:
                delta, offset = _unpack_mv_delta(data, offset)
                predictor = predict_mv(mv_grid.get((row, col-1)),
                                       mv_grid.get((row-1, col)),
                                       mv_grid.get((row-1, col-1)))
                mv = decode_mv_delta(delta, predictor)
                mv_grid[(row, col)] = mv
            else:
                mv, offset = _unpack_mv(data, offset)
            block_list.append({**base, 'mv_qp': (mv[0]*4, mv[1]*4)})
        elif fid == QD_DC_ONLY:
            (dx, dy), offset = _unpack_mv(data, offset)
            dc_vals = []
            for _ in range(3):
                b = data[offset]; offset += 1
                if b == 0x80:
                    v = struct.unpack_from('>h', data, offset)[0]; offset += 2
                else:
                    v = b if b < 128 else b - 256
                dc_vals.append(v)
            q_y = np.zeros((bs, bs), dtype=np.int16); q_y[0,0] = dc_vals[0]
            q_u = np.zeros((bs_c, bs_c), dtype=np.int16); q_u[0,0] = dc_vals[1]
            q_v = np.zeros((bs_c, bs_c), dtype=np.int16); q_v[0,0] = dc_vals[2]
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                              'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})
        elif fid == QD_LOW_FREQ:
            (dx, dy), offset = _unpack_mv(data, offset)
            q_y, offset = _unpack_dct_lowfreq(data, offset, bs)
            dc_vals = []
            for _ in range(2):
                b = data[offset]; offset += 1
                if b == 0x80:
                    v = struct.unpack_from('>h', data, offset)[0]; offset += 2
                else:
                    v = b if b < 128 else b - 256
                dc_vals.append(v)
            q_u = np.zeros((bs_c, bs_c), dtype=np.int16); q_u[0,0] = dc_vals[0]
            q_v = np.zeros((bs_c, bs_c), dtype=np.int16); q_v[0,0] = dc_vals[1]
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                              'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})
        elif fid == QD_FULL_DCT:
            (dx, dy), offset = _unpack_mv(data, offset)
            q_y, offset = decode_dct_block_optimized(data, bs, offset)
            q_u, offset = decode_dct_block_optimized(data, bs_c, offset)
            q_v, offset = decode_dct_block_optimized(data, bs_c, offset)
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                              'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})
        elif fid == QD_INTRA_PATCH:
            q_y, offset = decode_dct_block_optimized(data, bs, offset)
            q_u, offset = decode_dct_block_optimized(data, bs_c, offset)
            q_v, offset = decode_dct_block_optimized(data, bs_c, offset)
            block_list.append({**base, 'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})
        else:
            block_list.append({**base, 'family_id': QD_SKIP_TRUE})

    return block_list, params, offset


# ═══════════════════════════════════════════════════════════════════════════════
# KLASYFIKACJA BLOKU
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_block(q_dct_y, q_dct_u, q_dct_v, mv_qp, sad_mc, sad_static, intra_threshold):
    bs2 = q_dct_y.shape[0] * q_dct_y.shape[1]
    threshold_skip = bs2 * 1.5
    threshold_mv = bs2 * 1.0

    if sad_static < threshold_skip:
        return QD_SKIP_TRUE
    if sad_mc < threshold_mv:
        return QD_MV_ONLY

    nz_y = int(np.count_nonzero(q_dct_y))
    if nz_y == 0:
        return QD_MV_ONLY
    if nz_y == 1 and q_dct_y[0, 0] != 0:
        return QD_DC_ONLY
    if nz_y <= 4:
        lowfreq_positions = set(_LOWFREQ_IDX)
        nz_positions = set(zip(*np.nonzero(q_dct_y)))
        if nz_positions.issubset(lowfreq_positions):
            return QD_LOW_FREQ
    if sad_mc > intra_threshold:
        return QD_INTRA_PATCH
    return QD_FULL_DCT


# ═══════════════════════════════════════════════════════════════════════════════
# PRZETWARZANIE RZĘDU BLOKÓW P-FRAME (z Diamond Search + Early Termination)
# ═══════════════════════════════════════════════════════════════════════════════

def process_row_qdiff(args):
    (y, w_pad, h_pad, prev_Y, curr_Y, prev_U, curr_U, prev_V, curr_V,
     search_range, Q_Y, Q_C, use_subpixel, mad_row, intra_threshold) = args

    BS = 16; bs = BS; bs_c = bs // 2
    row_results = []
    col_idx = 0; x = 0
    bs2 = bs * bs
    threshold_skip = bs2 * 1.5
    threshold_mv = bs2 * 1.0
    threshold_intra_block = intra_threshold * 0.85

    while x < w_pad:
        curr_y_block = curr_Y[y:y+bs, x:x+bs].astype(np.float32)
        prev_y_static = prev_Y[y:y+bs, x:x+bs].astype(np.float32)
        sad_static = float(np.sum(np.abs(curr_y_block - prev_y_static)))

        if mad_row is not None and col_idx < len(mad_row):
            if mad_row[col_idx] < 1.0:
                row_results.append({'x': x, 'y': y, 'bs': bs, 'family_id': QD_SKIP_TRUE})
                x += bs; col_idx += 1; continue
        col_idx += 1

        if sad_static < threshold_skip:
            row_results.append({'x': x, 'y': y, 'bs': bs, 'family_id': QD_SKIP_TRUE})
            x += bs; continue

        # Early INTRA dla bardzo różnych bloków
        if sad_static > intra_threshold * 1.2:
            q_dct_y = np.round(apply_dct2(curr_y_block) / Q_Y).astype(np.int16)
            cy, cx = y // 2, x // 2
            curr_u_block = curr_U[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
            curr_v_block = curr_V[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
            q_dct_u = np.round(apply_dct2(curr_u_block) / Q_C).astype(np.int16)
            q_dct_v = np.round(apply_dct2(curr_v_block) / Q_C).astype(np.int16)
            row_results.append({
                'x': x, 'y': y, 'bs': bs, 'family_id': QD_INTRA_PATCH,
                'mv_qp': (0, 0), 'q_dct_y': q_dct_y, 'q_dct_u': q_dct_u, 'q_dct_v': q_dct_v,
            })
            x += bs; continue

        # Diamond Search z Early Termination
        best_dx_qp, best_dy_qp = 0, 0
        min_sad = sad_static
        early_exit = False
        step = max(1, search_range // 2)
        cy_tss, cx_tss = y, x

        while step >= 1 and not early_exit:
            diamond_offsets = [(0, 0), (-step, 0), (step, 0), (0, -step), (0, step)]
            for dy, dx in diamond_offsets:
                sy = max(0, min(cy_tss + dy, h_pad - bs))
                sx = max(0, min(cx_tss + dx, w_pad - bs))
                cand = prev_Y[sy:sy+bs, sx:sx+bs].astype(np.float32)
                sad = float(np.sum(np.abs(curr_y_block - cand)))
                if sad < min_sad:
                    min_sad = sad
                    best_dx_qp = (sx - x) * 4
                    best_dy_qp = (sy - y) * 4
                    cy_tss, cx_tss = sy, sx
                    if sad < threshold_mv:
                        early_exit = True
                        break
            step //= 2

        # Sub-pixel refinement (adaptacyjny)
        if use_subpixel and 0 < min_sad < threshold_intra_block:
            for qdy in range(-2, 3):
                for qdx in range(-2, 3):
                    if qdx == 0 and qdy == 0: continue
                    try_dy = best_dy_qp + qdy
                    try_dx = best_dx_qp + qdx
                    abs_y_qp = max(0, min(y*4 + try_dy, (h_pad - bs)*4))
                    abs_x_qp = max(0, min(x*4 + try_dx, (w_pad - bs)*4))
                    cand_sub = interpolate_subpixel(prev_Y, abs_y_qp, abs_x_qp, bs)
                    sad = float(np.sum(np.abs(curr_y_block - cand_sub)))
                    if sad < min_sad:
                        min_sad = sad; best_dy_qp = try_dy; best_dx_qp = try_dx
                        if sad < threshold_mv * 0.7:
                            break
                else:
                    continue
                break

        abs_y_qp = max(0, min(y*4 + best_dy_qp, (h_pad - bs)*4))
        abs_x_qp = max(0, min(x*4 + best_dx_qp, (w_pad - bs)*4))
        match_y = interpolate_subpixel(prev_Y, abs_y_qp, abs_x_qp, bs)

        diff_y = curr_y_block - match_y
        q_dct_y = np.round(apply_dct2(diff_y) / Q_Y).astype(np.int16)

        cy, cx = y // 2, x // 2
        uv_dy, uv_dx = best_dy_qp // 8, best_dx_qp // 8
        csy_c = max(0, min(cy + uv_dy, prev_U.shape[0] - bs_c))
        csx_c = max(0, min(cx + uv_dx, prev_U.shape[1] - bs_c))
        match_u = prev_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        match_v = prev_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        curr_u_block = curr_U[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
        curr_v_block = curr_V[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
        q_dct_u = np.round(apply_dct2(curr_u_block - match_u) / Q_C).astype(np.int16)
        q_dct_v = np.round(apply_dct2(curr_v_block - match_v) / Q_C).astype(np.int16)

        family_id = _classify_block(q_dct_y, q_dct_u, q_dct_v, (best_dx_qp, best_dy_qp),
                                    min_sad, sad_static, intra_threshold)

        row_results.append({
            'x': x, 'y': y, 'bs': bs, 'family_id': family_id,
            'mv_qp': (int(best_dx_qp), int(best_dy_qp)),
            'q_dct_y': q_dct_y, 'q_dct_u': q_dct_u, 'q_dct_v': q_dct_v,
        })
        x += bs

    return y, row_results


# ═══════════════════════════════════════════════════════════════════════════════
# GŁÓWNA KLASA KODEKA QDIFF v0.7
# ═══════════════════════════════════════════════════════════════════════════════

class QDiffCodec:
    def __init__(self, block_size=16, search_range=32, use_subpixel=True,
                 q_y=22.0, q_c=40.0, adaptive_q=False, intra_threshold_factor=4.0,
                 auto_mode=False, vfr_mode=False, drop_threshold=DEFAULT_DROP_THRESHOLD,
                 i_frame_quality_boost=0.7, use_delta_mv=True):
        self.block_size = block_size
        self.search_range = search_range
        self.use_subpixel = use_subpixel
        self.Q_Y = q_y; self.Q_Y_base = q_y
        self.Q_C = q_c; self.Q_C_base = q_c
        self.adaptive_q = adaptive_q
        self.intra_factor = intra_threshold_factor
        self.auto_mode = auto_mode
        self.vfr_mode = vfr_mode
        self.drop_threshold = drop_threshold
        self.i_frame_quality_boost = i_frame_quality_boost
        self.use_delta_mv = use_delta_mv
        self.prev_Y = self.prev_U = self.prev_V = None
        self.prev_frame = None
        self.base_params = {
            'q_y': q_y, 'q_c': q_c, 'search_range': search_range,
            'intra_factor': intra_threshold_factor, 'use_subpixel': use_subpixel,
        }

    def _get_frame_params(self, Y, h, w, is_iframe=False):
        if not self.auto_mode:
            params = self.base_params.copy()
            params['is_iframe'] = is_iframe
            if is_iframe:
                params['q_y'] = self.Q_Y_base * self.i_frame_quality_boost
                params['q_c'] = self.Q_C_base * self.i_frame_quality_boost
            return params
        metrics = analyze_frame_for_auto_tuning(Y, self.prev_Y)
        return auto_tune_params(metrics, self.base_params, is_iframe, self.i_frame_quality_boost)

    def _encode_iframe(self, Y, U, V, h, w, params=None):
        bs = self.block_size; bs_c = bs // 2
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C

        blks_Y = plane_to_blocks(Y[:h, :w], bs)
        q_blks_Y = np.round(batch_dct2(blks_Y) / q_y).astype(np.int16)
        rec_Y = blocks_to_plane(batch_idct2(q_blks_Y.astype(np.float32) * q_y), h, w, bs)

        blks_U = plane_to_blocks(U[:h//2, :w//2], bs_c)
        q_blks_U = np.round(batch_dct2(blks_U) / q_c).astype(np.int16)
        rec_U = blocks_to_plane(batch_idct2(q_blks_U.astype(np.float32) * q_c), h//2, w//2, bs_c)

        blks_V = plane_to_blocks(V[:h//2, :w//2], bs_c)
        q_blks_V = np.round(batch_dct2(blks_V) / q_c).astype(np.int16)
        rec_V = blocks_to_plane(batch_idct2(q_blks_V.astype(np.float32) * q_c), h//2, w//2, bs_c)

        q_Y_plane = blocks_to_plane(q_blks_Y.astype(np.float32), h, w, bs).astype(np.int16)
        q_U_plane = blocks_to_plane(q_blks_U.astype(np.float32), h//2, w//2, bs_c).astype(np.int16)
        q_V_plane = blocks_to_plane(q_blks_V.astype(np.float32), h//2, w//2, bs_c).astype(np.int16)

        self.prev_Y = deblock_filter(rec_Y, bs)
        self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)

        result = {'type': 'I', 'Y': q_Y_plane, 'U': q_U_plane, 'V': q_V_plane, 'h': h, 'w': w}
        if params: result['params'] = params
        return result

    def _encode_pframe(self, Y, U, V, h, w, params=None):
        bs = self.block_size
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C
        search_range = params.get('search_range', self.search_range) if params else self.search_range
        use_subpixel = params.get('use_subpixel', self.use_subpixel) if params else self.use_subpixel
        intra_factor = params.get('intra_factor', self.intra_factor) if params else self.intra_factor

        if self.adaptive_q and params is None:
            avg_mad = float(np.mean(np.abs(Y - self.prev_Y)))
            factor = 0.75 if avg_mad < 5.0 else (1.0 if avg_mad < 15.0 else min(1.0 + (avg_mad - 15.0) / 40.0, 1.5))
            q_y = self.Q_Y_base * factor
            q_c = self.Q_C_base * factor

        intra_threshold = intra_factor * q_y * bs * bs
        mad_map = self._compute_mad_map(Y, h, w, bs)

        tasks = []
        for row_idx, y in enumerate(range(0, h, bs)):
            mad_row = mad_map[row_idx] if row_idx < mad_map.shape[0] else None
            tasks.append((y, w, h, self.prev_Y, Y, self.prev_U, U, self.prev_V, V,
                          search_range, q_y, q_c, use_subpixel, mad_row, intra_threshold))

        results_by_row = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
            for y_out, row_res in ex.map(process_row_qdiff, tasks):
                results_by_row[y_out] = row_res

        ref_Y = self.prev_Y.copy(); ref_U = self.prev_U.copy(); ref_V = self.prev_V.copy()
        new_Y = ref_Y.copy(); new_U = ref_U.copy(); new_V = ref_V.copy()
        block_list = []
        family_counts = {k: 0 for k in range(8)}

        for y in range(0, h, bs):
            for blk in results_by_row[y]:
                block_list.append(blk)
                fid = blk['family_id']
                family_counts[fid] += 1
                x = blk['x']; bs_blk = blk['bs']; bs_c = bs_blk // 2

                if fid == QD_SKIP_TRUE:
                    continue
                elif fid == QD_SKIP_NOISE:
                    nm = blk.get('noise_mag', 0) / 255.0 * q_y * 0.5
                    if nm > 0.5:
                        from scipy.ndimage import uniform_filter as _uf
                        region = new_Y[y:y+bs_blk, x:x+bs_blk]
                        new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(_uf(region, size=3), 0, 255)
                elif fid == QD_MV_ONLY:
                    dx_qp, dy_qp = blk['mv_qp']
                    abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                    abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                    match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk)
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(match_y, 0, 255)
                    cy, cx = y//2, x//2
                    uv_dy, uv_dx = dy_qp//8, dx_qp//8
                    csy_c = max(0, min(cy+uv_dy, ref_U.shape[0]-bs_c))
                    csx_c = max(0, min(cx+uv_dx, ref_U.shape[1]-bs_c))
                    new_U[cy:cy+bs_c, cx:cx+bs_c] = ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                    new_V[cy:cy+bs_c, cx:cx+bs_c] = ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                elif fid in (QD_DC_ONLY, QD_LOW_FREQ, QD_FULL_DCT):
                    dx_qp, dy_qp = blk['mv_qp']
                    abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                    abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                    match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk)
                    res_y = apply_idct2(blk['q_dct_y'].astype(np.float32) * q_y)
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(match_y + res_y, 0, 255)
                    cy, cx = y//2, x//2
                    uv_dy, uv_dx = dy_qp//8, dx_qp//8
                    csy_c = max(0, min(cy+uv_dy, ref_U.shape[0]-bs_c))
                    csx_c = max(0, min(cx+uv_dx, ref_U.shape[1]-bs_c))
                    match_u = ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                    match_v = ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                    res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                    res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                    new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(match_u + res_u, -128, 127)
                    new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(match_v + res_v, -128, 127)
                elif fid == QD_INTRA_PATCH:
                    res_y = apply_idct2(blk['q_dct_y'].astype(np.float32) * q_y)
                    res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                    res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(res_y + 128, 0, 255)
                    cy, cx = y//2, x//2
                    new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_u, -128, 127)
                    new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_v, -128, 127)

        self.prev_Y = deblock_filter(new_Y, bs)
        self.prev_U = np.clip(new_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(new_V, -128, 127).astype(np.float32)

        total = len(block_list)
        stats = "  ".join(f"{FAMILY_NAMES[k]}:{v}({v/total*100:.0f}%)" for k, v in family_counts.items() if v > 0)
        print(f"   QDiff families [{total}]: {stats}", flush=True)

        result = {'type': 'P', 'blocks': block_list, 'h': h, 'w': w, 'family_counts': family_counts}
        if params: result['params'] = params
        return result

    def _compute_mad_map(self, Y, h, w, bs):
        Hb, Wb = h // bs, w // bs
        diff = np.abs(Y[:Hb*bs, :Wb*bs] - self.prev_Y[:Hb*bs, :Wb*bs])
        return diff.reshape(Hb, bs, Wb, bs).mean(axis=(1, 3))

    def encode_frame(self, frame, force_iframe=False):
        h, w = frame.shape[:2]
        h_pad, w_pad = h - (h % self.block_size), w - (w % self.block_size)

        if self.vfr_mode and self.prev_frame is not None and not force_iframe:
            should_drop, similarity, _ = should_drop_frame(frame, self.prev_frame, self.drop_threshold)
            if should_drop:
                self.prev_frame = frame.copy()
                return {'type': 'DROPPED', 'similarity': similarity, 'h': h, 'w': w}

        self.prev_frame = frame.copy()
        Y, U, V = rgb_to_yuv420(frame[:h_pad, :w_pad].astype(np.float32))
        is_iframe = force_iframe or (self.prev_Y is None)
        params = self._get_frame_params(Y, h_pad, w_pad, is_iframe)

        if is_iframe:
            return self._encode_iframe(Y, U, V, h_pad, w_pad, params)
        return self._encode_pframe(Y, U, V, h_pad, w_pad, params)

    def _decode_iframe(self, data, params=None):
        h, w = data['h'], data['w']
        bs = self.block_size; bs_c = bs // 2
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C

        q_blks_Y = plane_to_blocks(data['Y'].astype(np.float32), bs)
        rec_Y = blocks_to_plane(batch_idct2(q_blks_Y * q_y), h, w, bs)
        q_blks_U = plane_to_blocks(data['U'].astype(np.float32), bs_c)
        rec_U = blocks_to_plane(batch_idct2(q_blks_U * q_c), h//2, w//2, bs_c)
        q_blks_V = plane_to_blocks(data['V'].astype(np.float32), bs_c)
        rec_V = blocks_to_plane(batch_idct2(q_blks_V * q_c), h//2, w//2, bs_c)

        self.prev_Y = deblock_filter(rec_Y, bs)
        self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    def _decode_pframe(self, data, params=None):
        h, w = data['h'], data['w']
        bs = self.block_size; bs_c = bs // 2
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C

        ref_Y, ref_U, ref_V = self.prev_Y.copy(), self.prev_U.copy(), self.prev_V.copy()
        new_Y, new_U, new_V = ref_Y.copy(), ref_U.copy(), ref_V.copy()

        for blk in data['blocks']:
            x, y, fid = blk['x'], blk['y'], blk['family_id']
            bs_blk = blk['bs']

            if fid == QD_SKIP_TRUE:
                continue
            elif fid == QD_MV_ONLY:
                dx_qp, dy_qp = blk['mv_qp']
                abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk)
                new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(match_y, 0, 255)
                cy, cx = y//2, x//2
                uv_dy, uv_dx = dy_qp//8, dx_qp//8
                csy_c = max(0, min(cy+uv_dy, ref_U.shape[0]-bs_c))
                csx_c = max(0, min(cx+uv_dx, ref_U.shape[1]-bs_c))
                new_U[cy:cy+bs_c, cx:cx+bs_c] = ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                new_V[cy:cy+bs_c, cx:cx+bs_c] = ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
            elif fid in (QD_DC_ONLY, QD_LOW_FREQ, QD_FULL_DCT):
                dx_qp, dy_qp = blk['mv_qp']
                abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk)
                res_y = apply_idct2(blk['q_dct_y'].astype(np.float32) * q_y)
                new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(match_y + res_y, 0, 255)
                cy, cx = y//2, x//2
                uv_dy, uv_dx = dy_qp//8, dx_qp//8
                csy_c = max(0, min(cy+uv_dy, ref_U.shape[0]-bs_c))
                csx_c = max(0, min(cx+uv_dx, ref_U.shape[1]-bs_c))
                match_u = ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                match_v = ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]
                res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(match_u + res_u, -128, 127)
                new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(match_v + res_v, -128, 127)
            elif fid == QD_INTRA_PATCH:
                res_y = apply_idct2(blk['q_dct_y'].astype(np.float32) * q_y)
                res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(res_y + 128, 0, 255)
                cy, cx = y//2, x//2
                new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_u, -128, 127)
                new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_v, -128, 127)

        self.prev_Y = deblock_filter(new_Y, bs)
        self.prev_U = np.clip(new_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(new_V, -128, 127).astype(np.float32)
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    def decode_frame(self, data, params=None):
        ft = data['type']
        if ft == 'I':
            return self._decode_iframe(data, params)
        elif ft == 'P':
            return self._decode_pframe(data, params)
        elif ft == 'DROPPED':
            return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)
        raise ValueError(f"Nieznany typ klatki: {ft}")


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZACJA KLATEK DO PLIKU v0.7
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_iframe_v7(data, params=None, timestamp=0.0):
    h, w = data['h'], data['w']
    out = bytearray()
    out.extend(struct.pack('>d', timestamp))
    if params is not None and params.get('auto_mode', False):
        out.extend(serialize_frame_params(params))
    out.extend(struct.pack('>HH', h, w))
    out.extend(encode_rle_int16(data['Y']))
    out.extend(encode_rle_int16(data['U']))
    out.extend(encode_rle_int16(data['V']))
    return bytes(out)


def _deserialize_iframe_v7(raw, offset, auto_mode=False):
    params = {}
    timestamp = struct.unpack_from('>d', raw, offset)[0]; offset += 8
    if auto_mode:
        params, offset = deserialize_frame_params(raw, offset)
    h, w = struct.unpack_from('>HH', raw, offset); offset += 4
    hc, wc = h // 2, w // 2
    Y, offset = decode_rle_int16(raw, h * w, offset)
    U, offset = decode_rle_int16(raw, hc * wc, offset)
    V, offset = decode_rle_int16(raw, hc * wc, offset)
    return {'type': 'I', 'Y': Y.reshape(h, w), 'U': U.reshape(hc, wc), 'V': V.reshape(hc, wc),
            'h': h, 'w': w, 'timestamp': timestamp}, params, offset


def _serialize_pframe_v7(data, params=None, timestamp=0.0, use_delta_mv=True):
    h, w = data['h'], data['w']
    out = bytearray()
    out.extend(struct.pack('>d', timestamp))
    out.extend(serialize_pframe_blocks_v7(data['blocks'], h, w, params, use_delta_mv))
    return bytes(out)


def _deserialize_pframe_v7(raw, offset, auto_mode=False, use_delta_mv=True):
    timestamp = struct.unpack_from('>d', raw, offset)[0]; offset += 8
    blocks, params, offset = deserialize_pframe_blocks_v7(raw, offset, auto_mode, use_delta_mv)
    if blocks:
        h = max(b['y'] for b in blocks) + 16
        w = max(b['x'] for b in blocks) + 16
    else:
        h, w = 0, 0
    return {'type': 'P', 'blocks': blocks, 'h': h, 'w': w, 'timestamp': timestamp}, params, offset


def _serialize_dropped_frame(timestamp, similarity):
    return struct.pack('>df', timestamp, similarity)


def _deserialize_dropped_frame(raw, offset):
    timestamp = struct.unpack_from('>d', raw, offset)[0]; offset += 8
    similarity = struct.unpack_from('>f', raw, offset)[0]; offset += 4
    return {'type': 'DROPPED', 'timestamp': timestamp, 'similarity': similarity}, offset


# ═══════════════════════════════════════════════════════════════════════════════
# WIDEO I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _read_frames(input_path, max_frames, full=False):
    frames = []
    try:
        import imageio.v3 as _iio
        props = _iio.improps(input_path, plugin='pyav')
        n_total = props.n_images if hasattr(props, 'n_images') else 9999
        limit = n_total if full else min(max_frames, n_total)
        for i, frame in enumerate(_iio.imiter(input_path, plugin='pyav')):
            if i >= limit: break
            frames.append(np.asarray(frame)[:, :, :3])
            print(f"  Wczytano klatkę {i+1}/{limit}", flush=True)
        return frames
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać wideo: {e}")


def _write_frames(frames, output_path, fps=25.0):
    errors = []
    h, w = frames[0].shape[:2]
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            output_path, (w, h), fps=fps, codec='libx264',
            quality=None, output_params=['-crf', '18', '-preset', 'fast'])
        writer.send(None)
        for frame in frames:
            writer.send(np.asarray(frame, dtype=np.uint8).tobytes())
        writer.close()
        print(f"  [zapis] imageio_ffmpeg → {output_path}", flush=True)
        return
    except Exception as e:
        errors.append(f"imageio_ffmpeg: {e}")
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            raise RuntimeError("VideoWriter nie otworzył")
        for frame in frames:
            out.write(cv2.cvtColor(np.asarray(frame, np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  [zapis] cv2 → {output_path}", flush=True)
        return
    except Exception as e:
        errors.append(f"cv2: {e}")
    raise RuntimeError("Nie udało się zapisać wideo:\n" + "\n".join(f"  • {e}" for e in errors))


# ═══════════════════════════════════════════════════════════════════════════════
# ENKODOWANIE I DEKODOWANIE WIDEO
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(input_path, output_path, max_frames=30, full=False,
                 q_y=22.0, q_c=40.0, search_range=32,
                 use_subpixel=True, adaptive_q=False,
                 keyframe_interval=100, scene_cut_threshold=25.0,
                 intra_threshold_factor=4.0, auto_mode=False, vfr_mode=False,
                 drop_threshold=DEFAULT_DROP_THRESHOLD, i_frame_quality_boost=0.7,
                 zstd_level=19, use_delta_mv=True):

    print(f"\n╔══ QDIFF CODEC v0.7 — ENKODOWANIE ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Q_Y={q_y}  Q_C={q_c}  search={search_range}  keyframe_interval={keyframe_interval}")
    print(f"  Auto-mode: {auto_mode}  VFR: {vfr_mode}  DeltaMV: {use_delta_mv}")
    print(f"  Zstd level: {zstd_level}")
    print(f"╚{'═'*40}╝\n", flush=True)

    src_params = extract_source_video_params(input_path)
    fps = src_params.get('fps', 25.0)
    print(f"  Źródło: {src_params['width']}x{src_params['height']} @ {fps:.2f}fps", flush=True)

    frames = _read_frames(input_path, max_frames, full)
    n = len(frames)
    print(f"\n  Wczytano {n} klatek", flush=True)

    codec = QDiffCodec(
        search_range=search_range, use_subpixel=use_subpixel, q_y=q_y, q_c=q_c,
        adaptive_q=adaptive_q, intra_threshold_factor=intra_threshold_factor,
        auto_mode=auto_mode, vfr_mode=vfr_mode, drop_threshold=drop_threshold,
        i_frame_quality_boost=i_frame_quality_boost, use_delta_mv=use_delta_mv)

    encoded_frames = []
    vfr_total = 0
    total_bytes_before = 0
    total_bytes_after = 0

    for i, frame in enumerate(frames):
        t0 = time.time()
        h, w = frame.shape[:2]

        force_i = (i == 0) or (i % keyframe_interval == 0)
        if not force_i and codec.prev_Y is not None:
            Y_curr, _, _ = rgb_to_yuv420(frame[:h-(h%16), :w-(w%16)].astype(np.float32))
            scene_diff = float(np.mean(np.abs(Y_curr - codec.prev_Y[:Y_curr.shape[0], :Y_curr.shape[1]])))
            if scene_diff > scene_cut_threshold:
                print(f"  [SCENE CUT] klatka {i} — diff={scene_diff:.1f}", flush=True)
                force_i = True

        data = codec.encode_frame(frame, force_iframe=force_i)
        ft = data['type']

        timestamp = i / fps
        if ft == 'DROPPED':
            frame_bytes = _FRAME_DROPPED + _serialize_dropped_frame(timestamp, data['similarity'])
            vfr_total += 1
            print(f"  Klatka {i+1}/{n} [D] → VFR drop (sim={data['similarity']:.4f})", flush=True)
        elif ft == 'I':
            frame_bytes = _FRAME_I + _serialize_iframe_v7(data, data.get('params'), timestamp)
        else:
            frame_bytes = _FRAME_P + _serialize_pframe_v7(data, data.get('params'), timestamp, use_delta_mv)

        size_kb = len(frame_bytes) / 1024
        elapsed = time.time() - t0
        if ft not in ('DROPPED',):
            print(f"  Klatka {i+1}/{n} [{ft}] → {size_kb:.1f} KB ({elapsed:.2f}s)", flush=True)

        encoded_frames.append(frame_bytes)
        total_bytes_before += frame.shape[0] * frame.shape[1] * 3
        total_bytes_after += len(frame_bytes)

    print(f"\n  Kompresja zstd level {zstd_level}...", flush=True)
    header = (struct.pack('>4s', _QDIFF_MAGIC_V7) +
              struct.pack('>HHf', frames[0].shape[1], frames[0].shape[0], q_y) +
              struct.pack('>ffB', q_c, fps, 0x01 if auto_mode else 0x00))

    raw_stream = header + b''.join(encoded_frames) + _EOF_MARKER

    cctx = zstd.ZstdCompressor(level=zstd_level, threads=-1)
    compressed = cctx.compress(raw_stream)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    ratio = total_bytes_before / len(compressed)
    print(f"\n✓ SUKCES!")
    print(f"  Klatki: {n}  |  VFR dropped: {vfr_total}")
    print(f"  Raw: {total_bytes_before//1024} KB")
    print(f"  Pre-zstd: {total_bytes_after//1024} KB  |  Po zstd: {len(compressed)//1024} KB")
    print(f"  Kompresja: {ratio:.1f}×", flush=True)


def decode_video(input_path, output_path, fps=25.0):
    print(f"\n╔══ QDIFF CODEC v0.7 — DEKODOWANIE ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Wyjście: {output_path}", flush=True)

    with open(input_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.stream_reader(f).read()

    magic = raw[:4]
    
    # Backward compatibility
    if magic == _QDIFF_MAGIC_V7:
        print("  Format: v0.7 UNIFIED", flush=True)
    elif magic in (_QDIFF_MAGIC, _QDIFF_MAGIC_V2, _QDIFF_MAGIC_V3, _QDIFF_MAGIC_V5, _QDIFF_MAGIC_V6):
        print(f"  Format: {magic.decode()} — użyj odpowiedniej wersji kodeka", flush=True)
        raise NotImplementedError(f"Format {magic.decode()} wymaga dedykowanej wersji kodeka")
    else:
        raise ValueError(f"Nieznany format pliku (magic={magic})")

    W, H = struct.unpack_from('>HH', raw, 4)
    q_y = struct.unpack_from('>f', raw, 8)[0]
    q_c, fps = struct.unpack_from('>ff', raw, 12)
    auto_mode = raw[20] == 0x01
    offset = 21

    print(f"  Rozdzielczość: {W}x{H}  Q_Y={q_y:.1f}  Q_C={q_c:.1f}  FPS={fps:.2f}", flush=True)

    codec = QDiffCodec(q_y=q_y, q_c=q_c, auto_mode=auto_mode)
    decoded_frames = []

    while offset < len(raw) - 2:
        if raw[offset:offset+2] == _EOF_MARKER:
            break

        ft_byte = raw[offset:offset+1]; offset += 1

        if ft_byte == _FRAME_I:
            data, params, offset = _deserialize_iframe_v7(raw, offset, auto_mode)
        elif ft_byte == _FRAME_P:
            data, params, offset = _deserialize_pframe_v7(raw, offset, auto_mode)
        elif ft_byte == _FRAME_DROPPED:
            data, offset = _deserialize_dropped_frame(raw, offset)
            params = {}
        else:
            print(f"  [WARN] Nieznany typ klatki @ offset {offset-1}: {ft_byte}", flush=True)
            break

        img = codec.decode_frame(data, params if auto_mode else None)
        decoded_frames.append(img)
        print(f"  Zdekodowano klatkę {len(decoded_frames)} [{data['type']}]", flush=True)

    print(f"\n  Łącznie zdekodowano: {len(decoded_frames)} klatek", flush=True)
    _write_frames(decoded_frames, output_path, fps=fps)
    print(f"\n✓ SUKCES! → {output_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"[qdiff v0.7] DCT: {_DCT_BACKEND}", flush=True)
    print(f"[qdiff v0.7] Interpolacja: {_NUMBA_INFO}", flush=True)
    print(f"[qdiff v0.7] Wątki: {_N_WORKERS}", flush=True)
    print(f"[qdiff v0.7] Zstd level: {_ZSTD_LEVEL}", flush=True)

    parser = argparse.ArgumentParser(
        description="QDIFF CODEC v0.7 — UNIFIED (RLE + Sparse + DeltaMV + VFR + Zstd19)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ulepszenia v0.7 (z v0.5_strip + v0.2):
  1. RLE z length-prefix — poprawna serializacja
  2. Sparse encoding — tylko niezerowe współczynniki
  3. Auto-mode — per-frame parameter tuning
  4. Source parameter extraction — fps, rozdzielczość
  5. VFR — Variable Frame Rate (drop static frames)
  6. Diamond Search + Early Termination
  7. I-frame Quality Boost
  8. Delta MV Coding
  9. Zigzag Scan
  10. Zstd Level 19 + Threading
  11. Zwiększone domyślne parametry (search=32, keyframe=100)
        """)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-d', '--decode', action='store_true')
    parser.add_argument('-n', '--frames', type=int, default=30)
    parser.add_argument('-f', '--full', action='store_true')
    parser.add_argument('--q-y', type=float, default=22.0)
    parser.add_argument('--q-c', type=float, default=40.0)
    parser.add_argument('--search-range', type=int, default=32)
    parser.add_argument('--no-subpixel', action='store_true')
    parser.add_argument('--adaptive-q', action='store_true')
    parser.add_argument('--keyframe-interval', type=int, default=100)
    parser.add_argument('--scene-cut', type=float, default=25.0)
    parser.add_argument('--intra-factor', type=float, default=4.0)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('-a', '--auto', action='store_true', help='Auto-mode (per-frame tuning)')
    parser.add_argument('--vfr', action='store_true', help='Variable Frame Rate mode')
    parser.add_argument('--drop-threshold', type=float, default=DEFAULT_DROP_THRESHOLD)
    parser.add_argument('--i-frame-boost', type=float, default=0.7)
    parser.add_argument('--zstd-level', type=int, default=19, choices=range(1, 23))
    parser.add_argument('--no-delta-mv', action='store_true', help='Disable Delta MV coding')

    # Presety
    parser.add_argument('--preset-fast', action='store_true')
    parser.add_argument('--preset-quality', action='store_true')
    parser.add_argument('--preset-archive', action='store_true')
    args = parser.parse_args()

    if args.workers > 0:
        _N_WORKERS = args.workers

    q_y, q_c, sr = args.q_y, args.q_c, args.search_range
    zstd_lvl = args.zstd_level

    if args.preset_fast:
        q_y, q_c, sr, zstd_lvl = 32.0, 55.0, 24, 9
        print("▶ Preset FAST: Q_Y=32 Q_C=55 search=24 zstd=9")
    elif args.preset_quality:
        q_y, q_c, sr, zstd_lvl = 16.0, 28.0, 48, 22
        print("▶ Preset QUALITY: Q_Y=16 Q_C=28 search=48 zstd=22")
    elif args.preset_archive:
        q_y, q_c, zstd_lvl = 18.0, 32.0, 22
        print("▶ Preset ARCHIVE: Q_Y=18 Q_C=32 zstd=22 + VFR + Auto")

    if args.decode:
        decode_video(args.input, args.output, fps=args.fps)
    else:
        encode_video(
            args.input, args.output,
            max_frames=args.frames, full=args.full,
            q_y=q_y, q_c=q_c, search_range=sr,
            use_subpixel=not args.no_subpixel,
            adaptive_q=args.adaptive_q,
            keyframe_interval=args.keyframe_interval,
            scene_cut_threshold=args.scene_cut,
            intra_threshold_factor=args.intra_factor,
            auto_mode=args.auto or args.preset_archive,
            vfr_mode=args.vfr or args.preset_archive,
            drop_threshold=args.drop_threshold,
            i_frame_quality_boost=args.i_frame_boost,
            zstd_level=zstd_lvl,
            use_delta_mv=not args.no_delta_mv,
        )
