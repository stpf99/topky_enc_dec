#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   QDIFF CODEC v0.2 — Hierarchiczna rodzina bloków inspirowana mechaniką     ║
║   kwantową (QDiff) + TRYB AUTO (per-frame adaptive parameters).              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Zamiast binarnego SKIP/DETAIL — 8 klas (family_id, 3 bity):                ║
║                                                                              ║
║  000  SKIP-true      → 0 B       (blok identyczny, mv=(0,0))                ║
║  001  SKIP-noise     → 1 B       (kopiuj ref + wygładź szum kwantyzacji)    ║
║  010  MV-only        → 2–5 B     (ruch bez residualu)                       ║
║  011  DC-only        → mv + 3×i16  (tylko składowa DC Y+U+V)                ║
║  100  Low-freq       → mv + 4×i16Y + 2×i16UV (DCT coeffs 0-3)              ║
║  101  Full-DCT       → mv + 256×i16Y + 64×i16U + 64×i16V (= dawny DETAIL)  ║
║  110  Intra-patch    → 256×i16Y + 64×i16U + 64×i16V (bez mv, blok intra)   ║
║  111  Scene-cut      → 0 B marker (wymusza I-frame)                         ║
║                                                                              ║
║  NOWOŚĆ v0.2 — TRYB AUTO (-a/--auto):                                       ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Każda klatka ma zapisane swoje parametry encodowania (wagi) w formacie:     ║
║                                                                              ║
║  Frame Header (dla każdej klatki w trybie auto):                            ║
║    - flags: 1B (bit 0: auto_mode, bity 1-7: zarezerwowane)                  ║
║    - param_count: 1B (liczba parametrów)                                    ║
║    - params[]: param_count × (1B id + 4B float value)                       ║
║                                                                              ║
║  Parametry zapisywane per-frame w trybie auto:                              ║
║    0x01: Q_Y        — kwantyzacja luma                                      ║
║    0x02: Q_C        — kwantyzacja chroma                                    ║
║    0x03: SR         — search_range (int jako float)                         ║
║    0x04: INTRA_F    — intra_threshold_factor                                ║
║    0x05: SUBPIXEL   — use_subpixel (0.0/1.0)                                ║
║    0x06: MAD_AVG    — średnia MAD klatki (metryka ruchu)                    ║
║    0x07: COMPLEXITY — szacowana złożoność klatki (0.0-1.0)                  ║
║                                                                              ║
║  Format pliku: .qdiff                                                        ║
║  Uruchomienie:                                                               ║
║    python qdiff_codec.py -i input.mp4 -o output.qdiff                       ║
║    python qdiff_codec.py -i input.mp4 -o output.qdiff -a  # tryb auto       ║
║    python qdiff_codec.py -i output.qdiff -o decoded.mp4 -d                  ║
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

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE QDIFF
# ─────────────────────────────────────────────────────────────────────────────
QD_SKIP_TRUE   = 0b000  # Idealny skip — bez payloadu
QD_SKIP_NOISE  = 0b001  # Skip z widocznym szumem kwantyzacji
QD_MV_ONLY     = 0b010  # Ruch bez residualu
QD_DC_ONLY     = 0b011  # Tylko składowa DC (0,0) po DCT
QD_LOW_FREQ    = 0b100  # Pierwsze 4 współczynniki DCT
QD_FULL_DCT    = 0b101  # Pełne DCT
QD_INTRA_PATCH = 0b110  # Blok intra (bez referencji)
QD_SCENE_CUT   = 0b111  # Marker cięcia sceny

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

# Indeksy zigzag dla 4 najniższych częstotliwości (DC + 3 AC)
_LOWFREQ_IDX = [(0,0), (0,1), (1,0), (2,0)]  # DC + pierwsze AC

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE PARAMETRÓW AUTO-MODE
# ─────────────────────────────────────────────────────────────────────────────
# ID parametrów zapisywanych per-frame
PARAM_Q_Y        = 0x01  # Kwantyzacja Y
PARAM_Q_C        = 0x02  # Kwantyzacja Chroma
PARAM_SR         = 0x03  # Search Range
PARAM_INTRA_F    = 0x04  # Intra threshold factor
PARAM_SUBPIXEL   = 0x05  # Use subpixel (0.0/1.0)
PARAM_MAD_AVG    = 0x06  # Średnia MAD klatki
PARAM_COMPLEXITY = 0x07  # Złożoność klatki (0.0-1.0)
PARAM_SCENE_DIFF = 0x08  # Scene difference score

PARAM_NAMES = {
    PARAM_Q_Y:        "Q_Y",
    PARAM_Q_C:        "Q_C",
    PARAM_SR:         "SEARCH_RANGE",
    PARAM_INTRA_F:    "INTRA_FACTOR",
    PARAM_SUBPIXEL:   "SUBPIXEL",
    PARAM_MAD_AVG:    "MAD_AVG",
    PARAM_COMPLEXITY: "COMPLEXITY",
    PARAM_SCENE_DIFF: "SCENE_DIFF",
}

# Domyślne zakresy parametrów dla auto-tuningu
AUTO_Q_Y_MIN, AUTO_Q_Y_MAX = 12.0, 45.0
AUTO_Q_C_MIN, AUTO_Q_C_MAX = 22.0, 70.0
AUTO_SR_MIN, AUTO_SR_MAX = 12, 48
AUTO_INTRA_F_MIN, AUTO_INTRA_F_MAX = 2.5, 6.0


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 1 — BATCH DCT
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
            print(f"[qdiff] UWAGA: pyfftw IDCT round-trip błąd={_err:.2f} > 1.0 → fallback scipy",
                  flush=True)
            return _batch_dct2_scipy, _batch_idct2_scipy, "scipy.fftpack (pyfftw IDCT niepoprawny!)"

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
# CZĘŚĆ 2 — KOLORY YUV 4:2:0
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
    top    = out[0:H-2:2, :]
    bottom = out[2:H:2,   :]
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
# CZĘŚĆ 3 — INTERPOLACJA SUB-PIKSELOWA
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
# CZĘŚĆ 4 — FILTR DEBLOKUJĄCY
# ═══════════════════════════════════════════════════════════════════════════════

def deblock_filter(plane: np.ndarray, block_size: int = 16, threshold: float = 15.0) -> np.ndarray:
    h, w = plane.shape
    filtered = plane.astype(np.float32)
    for x in range(block_size, w, block_size):
        left  = filtered[:, x-1]; right = filtered[:, x]
        diff  = np.abs(left - right); mask = diff < threshold
        avg   = (left + right) * 0.5
        filtered[:, x-1] = np.where(mask, avg, left)
        filtered[:, x]   = np.where(mask, avg, right)
    for y in range(block_size, h, block_size):
        top    = filtered[y-1, :]; bottom = filtered[y, :]
        diff   = np.abs(top - bottom); mask = diff < threshold
        avg    = (top + bottom) * 0.5
        filtered[y-1, :] = np.where(mask, avg, top)
        filtered[y,   :] = np.where(mask, avg, bottom)
    return np.clip(filtered, 0, 255).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 5 — SERIALIZACJA PARAMETRÓW AUTO-MODE
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_params(params: dict) -> bytes:
    """
    Serializuje parametry klatki do formatu binarnego.

    Format:
      - flags: 1B (bit 0: auto_mode, bity 1-7: zarezerwowane)
      - param_count: 1B
      - params[]: param_count × (1B id + 4B float value)

    Używane parametry:
      0x01: Q_Y, 0x02: Q_C, 0x03: SR, 0x04: INTRA_F,
      0x05: SUBPIXEL, 0x06: MAD_AVG, 0x07: COMPLEXITY, 0x08: SCENE_DIFF
    """
    out = bytearray()

    # Flags: bit 0 = auto_mode enabled
    flags = 0x01 if params.get('auto_mode', False) else 0x00
    out.append(flags)

    # Parametry do zapisania (kolejność ważna dla spójności)
    param_ids = [
        (PARAM_Q_Y, 'q_y'),
        (PARAM_Q_C, 'q_c'),
        (PARAM_SR, 'search_range'),
        (PARAM_INTRA_F, 'intra_factor'),
        (PARAM_SUBPIXEL, 'use_subpixel'),
        (PARAM_MAD_AVG, 'mad_avg'),
        (PARAM_COMPLEXITY, 'complexity'),
        (PARAM_SCENE_DIFF, 'scene_diff'),
    ]

    # Zlicz parametry do zapisania
    params_to_write = []
    for pid, key in param_ids:
        if key in params:
            params_to_write.append((pid, params[key]))

    out.append(len(params_to_write))

    # Zapisz parametry
    for pid, value in params_to_write:
        out.append(pid)
        out.extend(struct.pack('>f', float(value)))

    return bytes(out)


def deserialize_frame_params(data: bytes, offset: int = 0) -> tuple:
    """
    Deserializuje parametry klatki z formatu binarnego.
    Zwraca (params_dict, new_offset).
    """
    params = {}

    flags = data[offset]
    params['auto_mode'] = bool(flags & 0x01)
    offset += 1

    param_count = data[offset]
    offset += 1

    for _ in range(param_count):
        pid = data[offset]
        offset += 1
        value = struct.unpack_from('>f', data, offset)[0]
        offset += 4

        if pid == PARAM_Q_Y:
            params['q_y'] = value
        elif pid == PARAM_Q_C:
            params['q_c'] = value
        elif pid == PARAM_SR:
            params['search_range'] = int(value)
        elif pid == PARAM_INTRA_F:
            params['intra_factor'] = value
        elif pid == PARAM_SUBPIXEL:
            params['use_subpixel'] = bool(value > 0.5)
        elif pid == PARAM_MAD_AVG:
            params['mad_avg'] = value
        elif pid == PARAM_COMPLEXITY:
            params['complexity'] = value
        elif pid == PARAM_SCENE_DIFF:
            params['scene_diff'] = value

    return params, offset


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 6 — AUTO-TUNING PARAMETRÓW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_frame_for_auto_tuning(curr_Y: np.ndarray, prev_Y: np.ndarray = None) -> dict:
    """
    Analizuje klatkę i zwraca metryki potrzebne do auto-tuningu parametrów.

    Metryki:
      - mad_avg: średnia Mean Absolute Difference od poprzedniej klatki
      - mad_std: odchylenie standardowe MAD
      - complexity: szacowana złożoność (0.0-1.0) na podstawie wariancji
      - motion_ratio: stosunek ruchu do obszaru (ile % obrazu się zmienia)
      - scene_diff: różnica scenowa (jeśli > threshold → scene cut)
    """
    h, w = curr_Y.shape

    # Podstawowe statystyki klatki
    variance = float(np.var(curr_Y))
    edge_density = _estimate_edge_density(curr_Y)

    metrics = {
        'variance': variance,
        'edge_density': edge_density,
        'mad_avg': 0.0,
        'mad_std': 0.0,
        'complexity': min(1.0, variance / 2500.0),  # Normalizowana wariancja
        'motion_ratio': 0.0,
        'scene_diff': 0.0,
    }

    if prev_Y is not None:
        # Oblicz MAD między klatkami
        diff = np.abs(curr_Y[:prev_Y.shape[0], :prev_Y.shape[1]] - prev_Y)
        mad_map = diff.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))

        metrics['mad_avg'] = float(np.mean(mad_map))
        metrics['mad_std'] = float(np.std(mad_map))

        # Motion ratio - ile % bloków ma znaczący ruch
        motion_threshold = 3.0
        motion_blocks = np.sum(mad_map > motion_threshold)
        total_blocks = mad_map.size
        metrics['motion_ratio'] = motion_blocks / total_blocks if total_blocks > 0 else 0.0

        # Scene difference score
        metrics['scene_diff'] = metrics['mad_avg']

        # Kompleksowość uwzględniająca ruch
        metrics['complexity'] = min(1.0, metrics['complexity'] * 0.7 + metrics['motion_ratio'] * 0.3)

    return metrics


def _estimate_edge_density(plane: np.ndarray) -> float:
    """Szacuje gęstość krawędzi w obrazie używając prostego operatora Sobel."""
    h, w = plane.shape
    if h < 4 or w < 4:
        return 0.0

    # Uproszczony gradient
    gx = np.abs(plane[1:-1, 2:] - plane[1:-1, :-2])
    gy = np.abs(plane[2:, 1:-1] - plane[:-2, 1:-1])

    edge_strength = np.sqrt(gx**2 + gy**2)
    edge_pixels = np.sum(edge_strength > 30)  # Próg krawędzi

    return edge_pixels / ((h-2) * (w-2))


def auto_tune_params(metrics: dict, base_params: dict) -> dict:
    """
    Automatycznie doiera parametry encodowania na podstawie metryk klatki.

    Strategia:
      1. Niska złożoność + mały ruch → niższe Q (lepsza jakość), mniejszy search_range
      2. Wysoka złożoność + duży ruch → wyższe Q (kompresja), większy search_range
      3. Scene cut → wymuś I-frame, resetuj parametry

    Wartości zwracane to wagi/konkretne parametry dla danej klatki.
    """
    params = base_params.copy()
    params['auto_mode'] = True

    complexity = metrics['complexity']
    motion_ratio = metrics['motion_ratio']
    mad_avg = metrics['mad_avg']
    scene_diff = metrics['scene_diff']

    # Zachowaj metryki w parametrach dla zapisu
    params['mad_avg'] = mad_avg
    params['complexity'] = complexity
    params['scene_diff'] = scene_diff

    # ─── Q_Y (kwantyzacja luma) ─────────────────────────────────────────────
    # Zakres: 12-45, base=22
    # - Mały ruch, niska złożoność → niższe Q (lepsza jakość)
    # - Duży ruch, wysoka złożoność → wyższe Q (więcej kompresji)

    q_y_factor = 1.0

    if mad_avg < 3.0:
        # Bardzo mały ruch - można użyć niższego Q dla lepszej jakości
        q_y_factor = 0.7 + 0.3 * (mad_avg / 3.0)
    elif mad_avg < 10.0:
        # Umiarkowany ruch - standardowe Q
        q_y_factor = 1.0 + 0.2 * ((mad_avg - 3.0) / 7.0)
    else:
        # Duży ruch - wyższe Q dla lepszej kompresji
        q_y_factor = 1.2 + 0.3 * min(1.0, (mad_avg - 10.0) / 20.0)

    # Korekta dla złożoności
    if complexity > 0.7:
        q_y_factor *= 1.1  # Bardzo złożone sceny potrzebują więcej kompresji

    params['q_y'] = np.clip(base_params.get('q_y', 22.0) * q_y_factor,
                            AUTO_Q_Y_MIN, AUTO_Q_Y_MAX)

    # ─── Q_C (kwantyzacja chroma) ───────────────────────────────────────────
    # Chroma jest mniej widoczna dla oka, może mieć wyższe Q
    q_c_factor = q_y_factor * 1.1  # Chroma zawsze nieco wyższe
    params['q_c'] = np.clip(base_params.get('q_c', 40.0) * q_c_factor,
                            AUTO_Q_C_MIN, AUTO_Q_C_MAX)

    # ─── Search Range ────────────────────────────────────────────────────────
    # Dostosuj do intensywności ruchu
    if motion_ratio < 0.1:
        # Mało ruchu - mniejszy search range
        sr_factor = 0.6
    elif motion_ratio < 0.3:
        # Umiarkowany ruch
        sr_factor = 1.0
    else:
        # Dużo ruchu - większy search range
        sr_factor = 1.0 + 0.5 * (motion_ratio - 0.3)

    base_sr = base_params.get('search_range', 24)
    params['search_range'] = int(np.clip(base_sr * sr_factor, AUTO_SR_MIN, AUTO_SR_MAX))

    # ─── Intra Threshold Factor ──────────────────────────────────────────────
    # Wyższy threshold oznacza rzadsze używanie intra-patch
    if motion_ratio > 0.5 and mad_avg > 15:
        # Dużo ruchu z dużym SAD - częściej używaj intra
        params['intra_factor'] = AUTO_INTRA_F_MIN
    else:
        params['intra_factor'] = np.clip(
            base_params.get('intra_factor', 4.0) * (1.0 - 0.2 * motion_ratio),
            AUTO_INTRA_F_MIN, AUTO_INTRA_F_MAX
        )

    # ─── Subpixel ────────────────────────────────────────────────────────────
    # Subpixel jest przydatny przy ruchu, ale kosztuje czas
    params['use_subpixel'] = mad_avg > 2.0  # Włącz tylko przy widocznym ruchu

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 7 — SERIALIZACJA QDIFF BLOKÓW
# ═══════════════════════════════════════════════════════════════════════════════

def _pack_mv(dx: int, dy: int) -> bytes:
    """Kompaktowy zapis motion vector."""
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
    """Pakuje pierwsze 4 współczynniki DCT (DC + 3 AC)."""
    vals = [int(q_dct[r, c]) for r, c in _LOWFREQ_IDX]
    return struct.pack('>4h', *vals)


def _unpack_dct_lowfreq(data: bytes, offset: int, block_size: int) -> tuple:
    vals = struct.unpack_from('>4h', data, offset)
    blk = np.zeros((block_size, block_size), dtype=np.int16)
    for i, (r, c) in enumerate(_LOWFREQ_IDX):
        blk[r, c] = vals[i]
    return blk, offset + 8


def _pack_block_coeffs(q_dct: np.ndarray) -> bytes:
    """Pakuje pełne współczynniki DCT."""
    return q_dct.astype(np.int16).flatten().tobytes()


def _unpack_block_coeffs(data: bytes, offset: int, bs: int) -> tuple:
    n = bs * bs
    arr = np.frombuffer(data[offset:offset + n*2], dtype=np.int16).reshape(bs, bs).copy()
    return arr, offset + n*2


def serialize_pframe_blocks(block_list: list, h: int, w: int, params: dict = None) -> bytes:
    """
    Serializuje listę bloków QDiff do strumienia bajtów.
    W trybie auto dodaje nagłówek z parametrami.

    Format (auto mode):
      - params_header: serialize_frame_params()
      - cols, rows: 4B (uint16 big-endian)
      - family_ids: N bajtów (1B per blok)
      - payloady bloków

    Format (normal mode):
      - cols, rows: 4B (uint16 big-endian)
      - family_ids: N bajtów (1B per blok)
      - payloady bloków
    """
    bs = 16
    cols = w // bs
    rows = h // bs
    n_blocks = rows * cols

    out = bytearray()

    # Dodaj nagłówek parametrów jeśli tryb auto
    if params is not None and params.get('auto_mode', False):
        out.extend(serialize_frame_params(params))

    # Mapa family_id per blok
    families = np.zeros(n_blocks, dtype=np.uint8)
    block_map = {}
    for blk in block_list:
        x = blk['x']; y = blk['y']
        col = x // bs; row = y // bs
        idx = row * cols + col
        fid = blk['family_id']
        families[idx] = fid
        block_map[idx] = blk

    out.extend(struct.pack('>HH', cols, rows))
    out.extend(families.tobytes())

    for idx in range(n_blocks):
        fid = families[idx]
        if fid in (QD_SKIP_TRUE, QD_SCENE_CUT):
            continue
        blk = block_map.get(idx)
        if blk is None:
            continue
        bs_blk = blk['bs']
        bs_c   = bs_blk // 2

        if fid == QD_SKIP_NOISE:
            out.extend(struct.pack('B', blk.get('noise_mag', 0)))

        elif fid == QD_MV_ONLY:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))

        elif fid == QD_DC_ONLY:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            out.extend(struct.pack('>3h',
                int(blk['q_dct_y'][0, 0]),
                int(blk['q_dct_u'][0, 0]),
                int(blk['q_dct_v'][0, 0])))

        elif fid == QD_LOW_FREQ:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            out.extend(_pack_dct_lowfreq(blk['q_dct_y']))
            out.extend(struct.pack('>2h',
                int(blk['q_dct_u'][0, 0]),
                int(blk['q_dct_v'][0, 0])))

        elif fid == QD_FULL_DCT:
            dx_qp, dy_qp = blk['mv_qp']
            out.extend(_pack_mv(dx_qp // 4, dy_qp // 4))
            out.extend(_pack_block_coeffs(blk['q_dct_y']))
            out.extend(_pack_block_coeffs(blk['q_dct_u']))
            out.extend(_pack_block_coeffs(blk['q_dct_v']))

        elif fid == QD_INTRA_PATCH:
            out.extend(_pack_block_coeffs(blk['q_dct_y']))
            out.extend(_pack_block_coeffs(blk['q_dct_u']))
            out.extend(_pack_block_coeffs(blk['q_dct_v']))

    return bytes(out)


def deserialize_pframe_blocks(data: bytes, offset_in: int = 0, auto_mode: bool = False) -> tuple:
    """
    Deserializuje strumień bajtów do listy bloków QDiff.
    W trybie auto odczytuje najpierw parametry klatki.

    Zwraca (block_list, params, consumed).
    """
    offset = offset_in
    params = {}

    # Odczytaj parametry jeśli tryb auto
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

    for idx in range(n_blocks):
        row = idx // cols; col = idx % cols
        x = col * bs; y = row * bs
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
            (dx, dy), offset = _unpack_mv(data, offset)
            block_list.append({**base, 'mv_qp': (dx*4, dy*4)})

        elif fid == QD_DC_ONLY:
            (dx, dy), offset = _unpack_mv(data, offset)
            dc_y, dc_u, dc_v = struct.unpack_from('>3h', data, offset); offset += 6
            q_y = np.zeros((bs, bs), dtype=np.int16); q_y[0,0] = dc_y
            q_u = np.zeros((bs_c, bs_c), dtype=np.int16); q_u[0,0] = dc_u
            q_v = np.zeros((bs_c, bs_c), dtype=np.int16); q_v[0,0] = dc_v
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                                'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})

        elif fid == QD_LOW_FREQ:
            (dx, dy), offset = _unpack_mv(data, offset)
            q_y, offset = _unpack_dct_lowfreq(data, offset, bs)
            dc_u, dc_v = struct.unpack_from('>2h', data, offset); offset += 4
            q_u = np.zeros((bs_c, bs_c), dtype=np.int16); q_u[0,0] = dc_u
            q_v = np.zeros((bs_c, bs_c), dtype=np.int16); q_v[0,0] = dc_v
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                                'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})

        elif fid == QD_FULL_DCT:
            (dx, dy), offset = _unpack_mv(data, offset)
            q_y, offset = _unpack_block_coeffs(data, offset, bs)
            q_u, offset = _unpack_block_coeffs(data, offset, bs_c)
            q_v, offset = _unpack_block_coeffs(data, offset, bs_c)
            block_list.append({**base, 'mv_qp': (dx*4, dy*4),
                                'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})

        elif fid == QD_INTRA_PATCH:
            q_y, offset = _unpack_block_coeffs(data, offset, bs)
            q_u, offset = _unpack_block_coeffs(data, offset, bs_c)
            q_v, offset = _unpack_block_coeffs(data, offset, bs_c)
            block_list.append({**base, 'q_dct_y': q_y, 'q_dct_u': q_u, 'q_dct_v': q_v})

        else:
            block_list.append({**base, 'family_id': QD_SKIP_TRUE})

    return block_list, params, offset


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 8 — KLASYFLKACJA BLOKU → FAMILY_ID
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_block(q_dct_y: np.ndarray,
                    q_dct_u: np.ndarray,
                    q_dct_v: np.ndarray,
                    mv_qp: tuple,
                    sad_mc: float,
                    sad_static: float,
                    intra_threshold: float) -> int:
    """
    Rzut bloku na bazę QDiff — wybór rodziny (analogia kolapsu funkcji falowej).
    """
    bs2 = q_dct_y.shape[0] * q_dct_y.shape[1]
    threshold_skip   = bs2 * 1.5
    threshold_mv     = bs2 * 1.0

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
# CZĘŚĆ 9 — PRZETWARZANIE RZĘDU BLOKÓW P-FRAME
# ═══════════════════════════════════════════════════════════════════════════════

def process_row_qdiff(args):
    (y, w_pad, h_pad,
     prev_Y, curr_Y,
     prev_U, curr_U,
     prev_V, curr_V,
     search_range, Q_Y, Q_C,
     use_subpixel, mad_row,
     intra_threshold) = args

    BS = 16; bs = BS; bs_c = bs // 2
    row_results = []
    col_idx = 0; x = 0

    while x < w_pad:
        curr_y_block = curr_Y[y:y+bs, x:x+bs].astype(np.float32)
        prev_y_static = prev_Y[y:y+bs, x:x+bs].astype(np.float32)
        sad_static = float(np.sum(np.abs(curr_y_block - prev_y_static)))

        if mad_row is not None and col_idx < len(mad_row):
            if mad_row[col_idx] < 1.0:
                row_results.append({'x': x, 'y': y, 'bs': bs, 'family_id': QD_SKIP_TRUE})
                x += bs; col_idx += 1; continue
        col_idx += 1

        if sad_static < bs * bs * 1.5:
            row_results.append({'x': x, 'y': y, 'bs': bs, 'family_id': QD_SKIP_TRUE})
            x += bs; continue

        # Motion Search (Three-Step Search)
        best_dx_qp, best_dy_qp = 0, 0
        min_sad = float('inf')
        step = max(1, search_range // 2)
        cy_tss, cx_tss = y, x

        while step >= 1:
            candidates = []
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    sy = max(0, min(cy_tss + dy, h_pad - bs))
                    sx = max(0, min(cx_tss + dx, w_pad - bs))
                    candidates.append((sy, sx))
            for sy, sx in candidates:
                cand = prev_Y[sy:sy+bs, sx:sx+bs].astype(np.float32)
                sad = float(np.sum(np.abs(curr_y_block - cand)))
                if sad < min_sad:
                    min_sad = sad
                    best_dx_qp = (sx - x) * 4
                    best_dy_qp = (sy - y) * 4
                    cy_tss, cx_tss = sy, sx
            step //= 2

        # Sub-pixel refinement
        if use_subpixel and min_sad > 0:
            for qdy in range(-3, 4):
                for qdx in range(-3, 4):
                    if qdx == 0 and qdy == 0: continue
                    try_dy = best_dy_qp + qdy
                    try_dx = best_dx_qp + qdx
                    abs_y_qp = max(0, min(y*4 + try_dy, (h_pad - bs)*4))
                    abs_x_qp = max(0, min(x*4 + try_dx, (w_pad - bs)*4))
                    cand_sub = interpolate_subpixel(prev_Y, abs_y_qp, abs_x_qp, bs)
                    sad = float(np.sum(np.abs(curr_y_block - cand_sub)))
                    if sad < min_sad:
                        min_sad = sad; best_dy_qp = try_dy; best_dx_qp = try_dx

        abs_y_qp = max(0, min(y*4 + best_dy_qp, (h_pad - bs)*4))
        abs_x_qp = max(0, min(x*4 + best_dx_qp, (w_pad - bs)*4))
        match_y = interpolate_subpixel(prev_Y, abs_y_qp, abs_x_qp, bs)

        # Residual DCT + kwantyzacja
        diff_y = curr_y_block - match_y
        q_dct_y = np.round(apply_dct2(diff_y) / Q_Y).astype(np.int16)

        cy, cx = y // 2, x // 2
        uv_dy = best_dy_qp // 8; uv_dx = best_dx_qp // 8
        csy_c = max(0, min(cy + uv_dy, prev_U.shape[0] - bs_c))
        csx_c = max(0, min(cx + uv_dx, prev_U.shape[1] - bs_c))

        match_u = prev_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        match_v = prev_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        curr_u_block = curr_U[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
        curr_v_block = curr_V[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
        q_dct_u = np.round(apply_dct2(curr_u_block - match_u) / Q_C).astype(np.int16)
        q_dct_v = np.round(apply_dct2(curr_v_block - match_v) / Q_C).astype(np.int16)

        # Klasyfikacja QDiff
        family_id = _classify_block(
            q_dct_y, q_dct_u, q_dct_v,
            (best_dx_qp, best_dy_qp),
            sad_mc=min_sad,
            sad_static=sad_static,
            intra_threshold=intra_threshold)

        blk = {
            'x': x, 'y': y, 'bs': bs, 'family_id': family_id,
            'mv_qp': (int(best_dx_qp), int(best_dy_qp)),
            'q_dct_y': q_dct_y, 'q_dct_u': q_dct_u, 'q_dct_v': q_dct_v,
        }
        row_results.append(blk)
        x += bs

    return y, row_results


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 10 — GŁÓWNA KLASA KODEKA QDIFF
# ═══════════════════════════════════════════════════════════════════════════════

class QDiffCodec:
    """
    Kodek oparty o hierarchiczną strukturę bloków QDiff.
    8 rodzin zamiast binarnego SKIP/DETAIL.
    Wspiera tryb auto z per-frame adaptacyjnymi parametrami.
    """

    def __init__(self, block_size: int = 16,
                 search_range: int = 24,
                 use_subpixel: bool = True,
                 q_y: float = 22.0,
                 q_c: float = 40.0,
                 adaptive_q: bool = False,
                 intra_threshold_factor: float = 4.0,
                 auto_mode: bool = False):
        self.block_size = block_size
        self.search_range = search_range
        self.use_subpixel = use_subpixel
        self.Q_Y = q_y; self.Q_Y_base = q_y
        self.Q_C = q_c; self.Q_C_base = q_c
        self.adaptive_q = adaptive_q
        self.intra_factor = intra_threshold_factor
        self.auto_mode = auto_mode
        self.prev_Y = self.prev_U = self.prev_V = None

        # Zapamiętaj bazowe parametry dla auto-tuningu
        self.base_params = {
            'q_y': q_y,
            'q_c': q_c,
            'search_range': search_range,
            'intra_factor': intra_threshold_factor,
            'use_subpixel': use_subpixel,
        }

    def _get_frame_params(self, Y: np.ndarray, h: int, w: int) -> dict:
        """
        Zwraca parametry dla bieżącej klatki.
        W trybie auto dokonuje analizy i auto-tuningu.
        W trybie normalnym zwraca bazowe parametry.
        """
        if not self.auto_mode:
            return self.base_params.copy()

        # Analiza klatki
        metrics = analyze_frame_for_auto_tuning(Y, self.prev_Y)

        # Auto-tuning parametrów
        params = auto_tune_params(metrics, self.base_params)

        return params

    # ─── I-Frame ──────────────────────────────────────────────────────────────

    def _encode_iframe(self, Y, U, V, h, w, params: dict = None) -> dict:
        bs = self.block_size; bs_c = bs // 2

        # Użyj parametrów z auto-tuningu jeśli dostępne
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

        result = {
            'type': 'I',
            'Y': q_Y_plane, 'U': q_U_plane, 'V': q_V_plane,
            'h': h, 'w': w,
        }

        # Dodaj parametry jeśli tryb auto
        if params:
            result['params'] = params

        return result

    # ─── P-Frame ──────────────────────────────────────────────────────────────

    def _encode_pframe(self, Y, U, V, h, w, params: dict = None) -> dict:
        bs = self.block_size

        # Pobierz parametry
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C
        search_range = params.get('search_range', self.search_range) if params else self.search_range
        use_subpixel = params.get('use_subpixel', self.use_subpixel) if params else self.use_subpixel
        intra_factor = params.get('intra_factor', self.intra_factor) if params else self.intra_factor

        if self.adaptive_q and params is None:
            avg_mad = float(np.mean(np.abs(Y - self.prev_Y)))
            factor = 0.75 if avg_mad < 5.0 else (1.0 if avg_mad < 15.0
                     else min(1.0 + (avg_mad - 15.0) / 40.0, 1.5))
            q_y = self.Q_Y_base * factor
            q_c = self.Q_C_base * factor

        intra_threshold = intra_factor * q_y * bs * bs

        # MAD pre-scan
        mad_map = self._compute_mad_map(Y, h, w, bs)

        tasks = []
        for row_idx, y in enumerate(range(0, h, bs)):
            mad_row = mad_map[row_idx] if row_idx < mad_map.shape[0] else None
            tasks.append((y, w, h,
                          self.prev_Y, Y, self.prev_U, U, self.prev_V, V,
                          search_range, q_y, q_c,
                          use_subpixel, mad_row, intra_threshold))

        results_by_row = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
            for y_out, row_res in ex.map(process_row_qdiff, tasks):
                results_by_row[y_out] = row_res

        # Rekonstrukcja
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
                    region = new_Y[y:y+bs_blk, x:x+bs_blk]
                    if nm > 0.5:
                        from scipy.ndimage import uniform_filter as _uf
                        new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(_uf(region, size=3), 0, 255)

                elif fid == QD_MV_ONLY:
                    dx_qp, dy_qp = blk['mv_qp']
                    abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                    abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                    match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk)
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(match_y, 0, 255)
                    cy, cx = y//2, x//2
                    uv_dy = dy_qp//8; uv_dx = dx_qp//8
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
                    uv_dy = dy_qp//8; uv_dx = dx_qp//8
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
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(res_y, 0, 255)
                    cy, cx = y//2, x//2
                    res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                    res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                    new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_u, -128, 127)
                    new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_v, -128, 127)

        self.prev_Y = deblock_filter(new_Y, bs)
        self.prev_U = new_U; self.prev_V = new_V

        total = len(block_list)
        stats = "  ".join(
            f"{FAMILY_NAMES[k]}:{v}({v/total*100:.0f}%)"
            for k, v in family_counts.items() if v > 0)
        print(f"   QDiff families [{total} bloków]: {stats}", flush=True)

        result = {'type': 'P', 'blocks': block_list, 'h': h, 'w': w}

        # Dodaj parametry jeśli tryb auto
        if params:
            result['params'] = params

        return result

    def _compute_mad_map(self, curr_Y, h, w, bs):
        rows = h // bs; cols = w // bs
        c = curr_Y[:rows*bs, :cols*bs].astype(np.float32)
        p = self.prev_Y[:rows*bs, :cols*bs].astype(np.float32)
        return np.abs(c - p).reshape(rows, bs, cols, bs).mean(axis=(1, 3))

    # ─── Dispatcher enkodowania ────────────────────────────────────────────────

    def encode_frame(self, img_np: np.ndarray, force_iframe: bool = False) -> dict:
        h, w, _ = img_np.shape
        h_pad = h - (h % 16); w_pad = w - (w % 16)
        Y, U, V = rgb_to_yuv420(img_np[:h_pad, :w_pad].astype(np.float32))

        # Pobierz parametry dla klatki (z auto-tuningu jeśli włączony)
        params = self._get_frame_params(Y, h_pad, w_pad)

        if self.prev_Y is None or force_iframe:
            return self._encode_iframe(Y, U, V, h_pad, w_pad, params)
        return self._encode_pframe(Y, U, V, h_pad, w_pad, params)

    # ─── Dekodowanie I-Frame ──────────────────────────────────────────────────

    def _decode_iframe(self, data: dict, params: dict = None) -> np.ndarray:
        bs = self.block_size; bs_c = bs // 2
        h, w = data['h'], data['w']

        # Użyj parametrów z klatki jeśli dostępne (tryb auto)
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C

        blks_Y = plane_to_blocks(data['Y'].astype(np.float32) * q_y, bs)
        rec_Y = blocks_to_plane(batch_idct2(blks_Y), h, w, bs)

        blks_U = plane_to_blocks(data['U'].astype(np.float32) * q_c, bs_c)
        rec_U = blocks_to_plane(batch_idct2(blks_U), h//2, w//2, bs_c)

        blks_V = plane_to_blocks(data['V'].astype(np.float32) * q_c, bs_c)
        rec_V = blocks_to_plane(batch_idct2(blks_V), h//2, w//2, bs_c)

        self.prev_Y = deblock_filter(rec_Y, bs)
        self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    # ─── Dekodowanie P-Frame ──────────────────────────────────────────────────

    def _decode_pframe(self, data: dict, params: dict = None) -> np.ndarray:
        h, w = data['h'], data['w']
        bs = self.block_size; bs_c = bs // 2

        # Użyj parametrów z klatki jeśli dostępne (tryb auto)
        q_y = params.get('q_y', self.Q_Y) if params else self.Q_Y
        q_c = params.get('q_c', self.Q_C) if params else self.Q_C

        new_Y = self.prev_Y.copy()
        new_U = self.prev_U.copy()
        new_V = self.prev_V.copy()
        ref_Y = self.prev_Y; ref_U = self.prev_U; ref_V = self.prev_V

        for blk in data['blocks']:
            x = blk['x']; y = blk['y']; bs_blk = blk['bs']; bs_c = bs_blk // 2
            fid = blk['family_id']

            if fid == QD_SKIP_TRUE or fid == QD_SCENE_CUT:
                pass

            elif fid == QD_SKIP_NOISE:
                nm = blk.get('noise_mag', 0)
                if nm > 64:
                    from scipy.ndimage import uniform_filter as _uf
                    region = new_Y[y:y+bs_blk, x:x+bs_blk].copy()
                    new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(_uf(region, size=3), 0, 255)

            elif fid == QD_MV_ONLY:
                dx_qp, dy_qp = blk['mv_qp']
                abs_y_qp = max(0, min(y*4 + dy_qp, (h - bs_blk)*4))
                abs_x_qp = max(0, min(x*4 + dx_qp, (w - bs_blk)*4))
                new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(
                    interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs_blk), 0, 255)
                cy, cx = y//2, x//2
                uv_dy = dy_qp//8; uv_dx = dx_qp//8
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
                uv_dy = dy_qp//8; uv_dx = dx_qp//8
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
                new_Y[y:y+bs_blk, x:x+bs_blk] = np.clip(res_y, 0, 255)
                cy, cx = y//2, x//2
                res_u = apply_idct2(blk['q_dct_u'].astype(np.float32) * q_c)
                res_v = apply_idct2(blk['q_dct_v'].astype(np.float32) * q_c)
                new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_u, -128, 127)
                new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(res_v, -128, 127)

        self.prev_Y = deblock_filter(new_Y, bs)
        self.prev_U = new_U; self.prev_V = new_V
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    def decode_frame(self, data: dict, params: dict = None) -> np.ndarray:
        ft = data['type']
        if ft == 'I':
            return self._decode_iframe(data, params)
        elif ft in ('P', 'B'):
            return self._decode_pframe(data, params)
        raise ValueError(f"Nieznany typ klatki: {ft}")


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 11 — SERIALIZACJA KLATEK DO PLIKU
# ═══════════════════════════════════════════════════════════════════════════════

_QDIFF_MAGIC = b'QDIF'
_QDIFF_MAGIC_V2 = b'QDF2'  # Wersja 0.2 z auto-mode
_FRAME_I     = b'I'
_FRAME_P     = b'P'
_EOF_MARKER  = b'\xFF\xFF'


def _serialize_iframe(data: dict, params: dict = None) -> bytes:
    """I-frame: rozkład DCT płaszczyzn → raw int16. Z params jeśli auto mode."""
    h, w = data['h'], data['w']
    out = bytearray()

    # Dodaj parametry jeśli tryb auto
    if params is not None and params.get('auto_mode', False):
        out.extend(serialize_frame_params(params))

    out.extend(struct.pack('>HH', h, w))
    out.extend(data['Y'].astype(np.int16).flatten().tobytes())
    out.extend(data['U'].astype(np.int16).flatten().tobytes())
    out.extend(data['V'].astype(np.int16).flatten().tobytes())
    return bytes(out)


def _deserialize_iframe(raw: bytes, offset: int, auto_mode: bool = False) -> tuple:
    """Deserializuje I-frame. Zwraca (data, params, new_offset)."""
    params = {}

    # Odczytaj parametry jeśli tryb auto
    if auto_mode:
        params, offset = deserialize_frame_params(raw, offset)

    h, w = struct.unpack_from('>HH', raw, offset); offset += 4
    hc, wc = h // 2, w // 2
    sz_Y = h * w
    sz_C = hc * wc

    Y = np.frombuffer(raw[offset:offset+sz_Y*2], dtype=np.int16).reshape(h, w).copy()
    offset += sz_Y * 2
    U = np.frombuffer(raw[offset:offset+sz_C*2], dtype=np.int16).reshape(hc, wc).copy()
    offset += sz_C * 2
    V = np.frombuffer(raw[offset:offset+sz_C*2], dtype=np.int16).reshape(hc, wc).copy()
    offset += sz_C * 2

    return {'type': 'I', 'Y': Y, 'U': U, 'V': V, 'h': h, 'w': w}, params, offset


def _serialize_pframe(data: dict, params: dict = None) -> bytes:
    """P-frame: serializacja bloków QDiff. Z params jeśli auto mode."""
    h, w = data['h'], data['w']
    return serialize_pframe_blocks(data['blocks'], h, w, params)


def _deserialize_pframe(raw: bytes, offset: int, auto_mode: bool = False) -> tuple:
    """Deserializuje P-frame. Zwraca (data, params, new_offset)."""
    blocks, params, offset = deserialize_pframe_blocks(raw, offset, auto_mode)

    # h, w są w blokach - musimy je odczytać
    h, w = struct.unpack_from('>HH', raw, offset - (len(blocks) + 4 + len(params) * 5 if auto_mode else len(blocks) + 4))
    # Powyższe jest błędne - musimy przeliczyć

    # Użyjmy h, w z pierwszego bloku lub obliczmy z liczby bloków
    if blocks:
        max_y = max(b['y'] for b in blocks) + 16
        max_x = max(b['x'] for b in blocks) + 16
        h, w = max_y, max_x
    else:
        h, w = 0, 0

    return {'type': 'P', 'blocks': blocks, 'h': h, 'w': w}, params, offset


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 12 — WIDEO I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _read_frames(input_path: str, max_frames: int, full: bool = False):
    """Wczytuje klatki z pliku wideo."""
    frames = []
    try:
        import imageio.v3 as _iio
        props = _iio.improps(input_path, plugin='pyav')
        n_total = props.n_images if hasattr(props, 'n_images') else 9999
        limit = n_total if full else min(max_frames, n_total)
        for i, frame in enumerate(_iio.imiter(input_path, plugin='pyav')):
            if i >= limit:
                break
            frames.append(np.asarray(frame)[:, :, :3])
            print(f"  Wczytano klatkę {i+1}/{limit}", flush=True)
        return frames
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać wideo: {e}\nZainstaluj: pip install imageio-ffmpeg")


def _write_frames(frames, output_path: str, fps: float = 25.0):
    """Zapisuje klatki do pliku wideo."""
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
# CZĘŚĆ 13 — ENKODOWANIE I DEKODOWANIE WIDEO
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(input_path: str, output_path: str,
                 max_frames: int = 30, full: bool = False,
                 q_y: float = 22.0, q_c: float = 40.0,
                 search_range: int = 24,
                 use_subpixel: bool = True,
                 adaptive_q: bool = False,
                 auto_mode: bool = False,  # NOWOŚĆ v0.2
                 keyframe_interval: int = 50,
                 scene_cut_threshold: float = 35.0,
                 intra_threshold_factor: float = 4.0):

    mode_str = "AUTO" if auto_mode else "MANUAL"
    print(f"\n╔══ QDIFF CODEC v0.2 — ENKODOWANIE [{mode_str}] ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Tryb: {'AUTO (per-frame adaptive params)' if auto_mode else 'MANUAL (fixed params)'}")
    print(f"  Q_Y={q_y}  Q_C={q_c}  search={search_range}  keyframe_interval={keyframe_interval}")
    print(f"  Architektura: 8-rodzinny QDiff (zamiast binarnego SKIP/DETAIL)")
    print(f"╚{'═'*50}╝\n", flush=True)

    frames = _read_frames(input_path, max_frames, full)
    n = len(frames)
    print(f"\n  Wczytano {n} klatek", flush=True)

    codec = QDiffCodec(
        q_y=q_y, q_c=q_c, search_range=search_range,
        use_subpixel=use_subpixel, adaptive_q=adaptive_q,
        intra_threshold_factor=intra_threshold_factor,
        auto_mode=auto_mode)

    encoded_frames = []
    frame_params_list = []  # Lista parametrów per-klatka
    family_stats_total = {k: 0 for k in range(8)}
    total_bytes_before = 0
    total_bytes_after = 0

    for i, frame in enumerate(frames):
        t0 = time.time()
        h, w = frame.shape[:2]

        # Detekcja cięcia sceny
        force_i = (i == 0) or (i % keyframe_interval == 0)
        if not force_i and codec.prev_Y is not None:
            Y_curr, _, _ = rgb_to_yuv420(frame[:h-(h%16), :w-(w%16)].astype(np.float32))
            scene_diff = float(np.mean(np.abs(Y_curr - codec.prev_Y[:Y_curr.shape[0], :Y_curr.shape[1]])))
            if scene_diff > scene_cut_threshold:
                print(f"  [SCENE CUT] klatka {i} — diff={scene_diff:.1f}", flush=True)
                force_i = True

        data = codec.encode_frame(frame, force_iframe=force_i)
        ft = data['type']
        params = data.get('params', None)

        # Zapisz parametry dla statystyk
        if params:
            frame_params_list.append(params)
            if i == 0 or len(frame_params_list) <= 3:
                param_str = "  ".join(f"{PARAM_NAMES.get(k, k)}={v:.2f}"
                                      for k, v in params.items()
                                      if k in ['q_y', 'q_c', 'search_range', 'mad_avg', 'complexity'])
                print(f"    [AUTO params] {param_str}", flush=True)

        # Serializacja
        if ft == 'I':
            frame_bytes = _FRAME_I + _serialize_iframe(data, params)
        else:
            frame_bytes = _FRAME_P + _serialize_pframe(data, params)

        size_kb = len(frame_bytes) / 1024
        elapsed = time.time() - t0
        print(f"  Klatka {i+1}/{n} [{ft}] → {size_kb:.1f} KB ({elapsed:.2f}s)", flush=True)

        encoded_frames.append(frame_bytes)
        total_bytes_before += frame.shape[0] * frame.shape[1] * 3
        total_bytes_after  += len(frame_bytes)

    # Zapis z kompresją zstd
    print(f"\n  Kompresja zstd...", flush=True)

    # Magic: QDF2 dla auto mode, QDIF dla normal
    magic = _QDIFF_MAGIC_V2 if auto_mode else _QDIFF_MAGIC

    header = (struct.pack('>4s', magic) +
              struct.pack('>HHf', frames[0].shape[1], frames[0].shape[0], q_y) +
              struct.pack('>ff', q_c, 25.0))  # q_c + fps placeholder

    raw_stream = header + b''.join(encoded_frames) + _EOF_MARKER

    cctx = zstd.ZstdCompressor(level=6)
    compressed = cctx.compress(raw_stream)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    ratio = total_bytes_before / len(compressed)
    print(f"\n✓ SUKCES!")
    print(f"  Klatki: {n}  |  Raw: {total_bytes_before//1024} KB")
    print(f"  Pre-zstd: {total_bytes_after//1024} KB  |  Po zstd: {len(compressed)//1024} KB")
    print(f"  Kompresja: {ratio:.1f}× ({len(compressed)/1024:.1f} KB)", flush=True)

    # Statystyki auto-tuningu jeśli włączony
    if auto_mode and frame_params_list:
        print(f"\n  Statystyki AUTO-MODE:", flush=True)
        avg_q_y = np.mean([p.get('q_y', q_y) for p in frame_params_list])
        avg_q_c = np.mean([p.get('q_c', q_c) for p in frame_params_list])
        avg_mad = np.mean([p.get('mad_avg', 0) for p in frame_params_list])
        print(f"    Średnie Q_Y: {avg_q_y:.1f} (zakres: {AUTO_Q_Y_MIN}-{AUTO_Q_Y_MAX})", flush=True)
        print(f"    Średnie Q_C: {avg_q_c:.1f} (zakres: {AUTO_Q_C_MIN}-{AUTO_Q_C_MAX})", flush=True)
        print(f"    Średnia MAD: {avg_mad:.2f}", flush=True)


def decode_video(input_path: str, output_path: str, fps: float = 25.0):
    print(f"\n╔══ QDIFF CODEC v0.2 — DEKODOWANIE ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Wyjście: {output_path}", flush=True)

    with open(input_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.stream_reader(f).read()

    magic = raw[:4]

    # Sprawdź wersję formatu
    if magic == _QDIFF_MAGIC_V2:
        auto_mode = True
        print(f"  Format: QDF2 (auto-mode enabled)", flush=True)
    elif magic == _QDIFF_MAGIC:
        auto_mode = False
        print(f"  Format: QDIF (legacy mode)", flush=True)
    else:
        raise ValueError(f"Nieznany format pliku (magic={magic})")

    # Nagłówek: 4B magic + 2B W + 2B H + 4B q_y + 4B q_c + 4B fps
    W, H = struct.unpack_from('>HH', raw, 4)
    q_y = struct.unpack_from('>f', raw, 8)[0]
    q_c, _ = struct.unpack_from('>ff', raw, 12)
    offset = 20

    codec = QDiffCodec(q_y=q_y, q_c=q_c, auto_mode=auto_mode)
    decoded_frames = []

    while offset < len(raw) - 2:
        if raw[offset:offset+2] == _EOF_MARKER:
            break

        ft_byte = raw[offset:offset+1]; offset += 1

        if ft_byte == _FRAME_I:
            data, params, offset = _deserialize_iframe(raw, offset, auto_mode)
        elif ft_byte == _FRAME_P:
            data, params, offset = _deserialize_pframe(raw, offset, auto_mode)
        else:
            print(f"  [WARN] Nieznany typ klatki @ offset {offset-1}: {ft_byte}", flush=True)
            break

        img = codec.decode_frame(data, params)
        decoded_frames.append(img)

        # Logowanie parametrów jeśli auto mode
        if auto_mode and params:
            param_str = "  ".join(f"{PARAM_NAMES.get(k, k)}={v:.2f}"
                                  for k, v in params.items()
                                  if k in ['q_y', 'q_c'])
            print(f"  Zdekodowano klatkę {len(decoded_frames)} [{data['type']}] {param_str}", flush=True)
        else:
            print(f"  Zdekodowano klatkę {len(decoded_frames)} [{data['type']}]", flush=True)

    print(f"\n  Łącznie zdekodowano: {len(decoded_frames)} klatek", flush=True)
    _write_frames(decoded_frames, output_path, fps=fps)
    print(f"\n✓ SUKCES! → {output_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 14 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"[qdiff] DCT backend: {_DCT_BACKEND}", flush=True)
    print(f"[qdiff] Interpolacja: {_NUMBA_INFO}", flush=True)
    print(f"[qdiff] VLC: {'ON' if _VLC_ENABLED else 'OFF'}", flush=True)
    print(f"[qdiff] Wątki: {_N_WORKERS}", flush=True)

    parser = argparse.ArgumentParser(
        description="QDIFF CODEC v0.2 — 8-rodzinny hierarchiczny kodek bloków z trybem AUTO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Rodziny QDiff:
  000 SKIP_TRUE    — identyczny blok, 0B payloadu
  001 SKIP_NOISE   — identyczny + wygładź ringing, 1B
  010 MV_ONLY      — ruch bez residualu, 2-5B
  011 DC_ONLY      — DC + MV, 8B
  100 LOW_FREQ     — 4 DCT + MV, 16B
  101 FULL_DCT     — pełny DCT + MV (dawny DETAIL)
  110 INTRA_PATCH  — blok intra (bez referencji)
  111 SCENE_CUT    — marker, 0B

Tryb AUTO (-a/--auto):
  Automatycznie doiera parametry Q_Y, Q_C, search_range, intra_factor
  dla każdej klatki na podstawie analizy ruchu i złożoności.
  Parametry są zapisywane w strumieniu i odczytywane przy dekodowaniu.

Parametry zapisywane per-frame w trybie auto:
  0x01 Q_Y        — kwantyzacja luma (12-45)
  0x02 Q_C        — kwantyzacja chroma (22-70)
  0x03 SR         — search range (12-48)
  0x04 INTRA_F    — intra threshold factor (2.5-6.0)
  0x05 SUBPIXEL   — use subpixel (0/1)
  0x06 MAD_AVG    — średnia MAD klatki (metryka)
  0x07 COMPLEXITY — złożoność klatki (0.0-1.0)
        """)
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')
    parser.add_argument('-n', '--frames',  type=int,   default=30)
    parser.add_argument('-f', '--full',    action='store_true')

    # NOWOŚĆ v0.2 - Tryb auto
    parser.add_argument('-a', '--auto',    action='store_true',
                        help='Tryb AUTO: automatyczny dobór parametrów per-klatka')

    parser.add_argument('--q-y',          type=float, default=22.0)
    parser.add_argument('--q-c',          type=float, default=40.0)
    parser.add_argument('--search-range', type=int,   default=24)
    parser.add_argument('--no-subpixel',  action='store_true')
    parser.add_argument('--adaptive-q',   action='store_true')
    parser.add_argument('--keyframe-interval', type=int, default=50)
    parser.add_argument('--scene-cut',    type=float, default=35.0)
    parser.add_argument('--intra-factor', type=float, default=4.0,
                        help='Próg intra-patch: SAD > factor*Q_Y*bs^2 → użyj intra')
    parser.add_argument('--fps',          type=float, default=25.0)
    parser.add_argument('--workers',      type=int,   default=0)

    # Presety
    parser.add_argument('--preset-fast',    action='store_true', help='Q_Y=32 Q_C=55')
    parser.add_argument('--preset-quality', action='store_true', help='Q_Y=16 Q_C=28 search=48')
    args = parser.parse_args()

    if args.workers > 0:
        _N_WORKERS = args.workers

    q_y, q_c, sr = args.q_y, args.q_c, args.search_range
    if args.preset_fast:
        q_y, q_c = 32.0, 55.0
        print("▶ Preset FAST: Q_Y=32 Q_C=55")
    elif args.preset_quality:
        q_y, q_c, sr = 16.0, 28.0, 48
        print("▶ Preset QUALITY: Q_Y=16 Q_C=28 search=48")

    if args.decode:
        decode_video(args.input, args.output, fps=args.fps)
    else:
        encode_video(
            args.input, args.output,
            max_frames=args.frames,
            full=args.full,
            q_y=q_y, q_c=q_c,
            search_range=sr,
            use_subpixel=not args.no_subpixel,
            adaptive_q=args.adaptive_q,
            auto_mode=args.auto,  # NOWOŚĆ v0.2
            keyframe_interval=args.keyframe_interval,
            scene_cut_threshold=args.scene_cut,
            intra_threshold_factor=args.intra_factor,
        )
