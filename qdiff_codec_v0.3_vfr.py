#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   QDIFF CODEC v0.3 — VFR + AUTO + Source Parameter Extraction                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NOWOŚCI v0.3:                                                              ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. VARIABLE FRAME RATE (VFR) --vfr                                         ║
║     • Detekcja klatek statycznych (visual margin dropout 99.9%)             ║
║     • Klatki z różnicą < 0.1% pikseli są dropowane                          ║
║     • Timestamps zapisywane dla każdej klatki                               ║
║                                                                              ║
║  2. KEYFRAME QUALITY BOOST (--i-frame-quality)                              ║
║     • I-frames (klatki kluczowe) mają wyższą jakość                         ║
║     • Domyślnie Q_Y i Q_C mnożone przez 0.7 dla I-frames                    ║
║                                                                              ║
║  3. SOURCE PARAMETER EXTRACTION (--match-source)                            ║
║     • Automatyczna ekstrakcja CRF, preset, FPS z wejściowego MP4            ║
║     • Użycie tych samych parametrów przy dekodowaniu do MP4                 ║
║                                                                              ║
║  4. VISUAL MARGIN DROPOUT (--drop-threshold)                                ║
║     • Procent różnicy pikseli poniżej którego klatka jest dropowana         ║
║     • Domyślnie 0.1% (99.9% podobieństwa = klatka statyczna)                ║
║                                                                              ║
║  Format pliku .qdiff v0.3:                                                  ║
║    Magic: QDF3                                                              ║
║    Header: W, H, fps, source_crf, source_preset, total_frames, kept_frames  ║
║    Per-frame: timestamp(8B) + dropped_flag(1B) + [frame_data if !dropped]   ║
║                                                                              ║
║  Uruchomienie:                                                               ║
║    python qdiff_codec.py -i input.mp4 -o output.qdiff -a --vfr              ║
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
PARAM_IS_IFRAME  = 0x09  # NOWOŚĆ v0.3 - czy to I-frame

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

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE VFR
# ─────────────────────────────────────────────────────────────────────────────
# Domyślny próg drop-out: 99.9% podobieństwa = klatka statyczna
DEFAULT_DROP_THRESHOLD = 0.001  # 0.1% różnicy

# ─────────────────────────────────────────────────────────────────────────────
# MAGIC NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
_QDIFF_MAGIC    = b'QDIF'  # v0.1
_QDIFF_MAGIC_V2 = b'QDF2'  # v0.2 auto
_QDIFF_MAGIC_V3 = b'QDF3'  # v0.3 VFR
_FRAME_I        = b'I'
_FRAME_P        = b'P'
_FRAME_DROPPED  = b'D'     # NOWOŚĆ v0.3 - marker dropped frame
_EOF_MARKER     = b'\xFF\xFF'


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
# CZĘŚĆ 5 — SOURCE VIDEO PARAMETER EXTRACTION (NOWOŚĆ v0.3)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_source_video_params(input_path: str) -> dict:
    """
    Ekstrahuje parametry kodowania z wejściowego pliku wideo.

    Zwraca:
      - fps: klatki na sekundę
      - crf: jakość (estymowana lub domyślna)
      - preset: preset kodowania
      - codec: kodek wideo
      - width, height: rozdzielczość
      - bitrate: bitrate (jeśli dostępny)
      - duration: czas trwania
      - n_frames: liczba klatek
    """
    params = {
        'fps': 25.0,
        'crf': 18,
        'preset': 'medium',
        'codec': 'libx264',
        'width': 0,
        'height': 0,
        'bitrate': None,
        'duration': 0.0,
        'n_frames': 0,
    }

    # Metoda 1: imageio
    try:
        import imageio.v3 as _iio
        props = _iio.improps(input_path, plugin='pyav')

        if hasattr(props, 'fps') and props.fps:
            params['fps'] = float(props.fps)
        elif hasattr(props, 'duration') and props.duration > 0:
            if hasattr(props, 'n_images'):
                params['fps'] = props.n_images / props.duration

        if hasattr(props, 'n_images'):
            params['n_frames'] = props.n_images
        if hasattr(props, 'duration'):
            params['duration'] = props.duration
        if hasattr(props, 'shape'):
            params['height'], params['width'] = props.shape[:2]

        # Próba ekstrakcji metadanych
        try:
            import av
            container = av.open(input_path)
            stream = container.streams.video[0]

            params['codec'] = stream.codec_context.name if hasattr(stream.codec_context, 'name') else 'libx264'
            params['width'] = stream.width
            params['height'] = stream.height
            params['fps'] = float(stream.average_rate) if stream.average_rate else params['fps']
            params['n_frames'] = stream.frames if stream.frames else params['n_frames']
            params['duration'] = float(stream.duration * stream.time_base) if stream.duration else params['duration']

            # Bitrate
            if stream.bit_rate:
                params['bitrate'] = stream.bit_rate

            # Metadane kodeka
            metadata = stream.metadata
            if 'encoder' in metadata:
                encoder = metadata['encoder'].lower()
                # Estymacja CRF na podstawie encodera
                if 'x264' in encoder or 'x265' in encoder:
                    # Próba wyciągniąć CRF z metadanych
                    pass

            container.close()
        except Exception as e:
            pass

    except Exception as e:
        print(f"[qdiff] Ostrzeżenie: nie można odczytać parametrów z {input_path}: {e}", flush=True)

    # Metoda 2: ffprobe (fallback)
    if params['width'] == 0:
        try:
            import subprocess
            import json

            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', '-show_streams', input_path],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        params['width'] = stream.get('width', params['width'])
                        params['height'] = stream.get('height', params['height'])
                        params['codec'] = stream.get('codec_name', params['codec'])

                        # FPS
                        fps_str = stream.get('r_frame_rate', '25/1')
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            params['fps'] = float(num) / float(den) if float(den) > 0 else 25.0

                        # Bitrate
                        if 'bit_rate' in stream:
                            params['bitrate'] = int(stream['bit_rate'])

                        # CRF estymacja z bitrate
                        if params['bitrate'] and params['width'] > 0:
                            pixels = params['width'] * params['height']
                            bpp = params['bitrate'] / (pixels * params['fps']) if params['fps'] > 0 else 0
                            # Estymacja CRF na podstawie bpp
                            # Niska jakość (CRF > 28): bpp < 0.05
                            # Średnia jakość (CRF 20-28): bpp 0.05-0.15
                            # Wysoka jakość (CRF < 20): bpp > 0.15
                            if bpp < 0.05:
                                params['crf'] = 28
                            elif bpp < 0.10:
                                params['crf'] = 23
                            elif bpp < 0.15:
                                params['crf'] = 20
                            else:
                                params['crf'] = 18

                        break

                if 'format' in data:
                    params['duration'] = float(data['format'].get('duration', 0))

        except Exception as e:
            pass

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 6 — DETEKCJA KLATEK STATYCZNYCH VFR (NOWOŚĆ v0.3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Oblicza podobieństwo między klatkami jako procent identycznych pikseli.

    Zwraca wartość 0.0-1.0 gdzie:
      - 1.0 = identyczne klatki
      - 0.0 = całkowicie różne

    Używa Y channel dla wydajności.
    """
    if frame1 is None or frame2 is None:
        return 0.0

    h = min(frame1.shape[0], frame2.shape[0])
    w = min(frame1.shape[1], frame2.shape[1])

    # Konwersja do Y jeśli RGB
    if len(frame1.shape) == 3:
        Y1 = np.dot(frame1[:h, :w, :3], [0.299, 0.587, 0.114])
    else:
        Y1 = frame1[:h, :w].astype(np.float32)

    if len(frame2.shape) == 3:
        Y2 = np.dot(frame2[:h, :w, :3], [0.299, 0.587, 0.114])
    else:
        Y2 = frame2[:h, :w].astype(np.float32)

    # Oblicz różnicę
    diff = np.abs(Y1 - Y2)

    # Procent pikseli z zerową lub minimalną różnicą (tolerancja 2 poziomy)
    threshold = 2.0
    similar_pixels = np.sum(diff <= threshold)
    total_pixels = h * w

    return similar_pixels / total_pixels


def should_drop_frame(frame: np.ndarray, prev_frame: np.ndarray,
                      drop_threshold: float = DEFAULT_DROP_THRESHOLD,
                      similarity_threshold: float = 0.999) -> tuple:
    """
    Decyduje czy klatka powinna być dropowana.

    Args:
        frame: Aktualna klatka
        prev_frame: Poprzednia klatka
        drop_threshold: Próg różnicy (0.001 = 0.1%)
        similarity_threshold: Próg podobieństwa (0.999 = 99.9%)

    Returns:
        (should_drop: bool, similarity: float, metrics: dict)
    """
    if prev_frame is None:
        return False, 0.0, {'reason': 'first_frame'}

    similarity = compute_frame_similarity(frame, prev_frame)

    # Klatka statyczna jeśli podobieństwo >= 99.9%
    should_drop = similarity >= similarity_threshold

    metrics = {
        'similarity': similarity,
        'diff_percent': 1.0 - similarity,
        'reason': 'static_frame' if should_drop else 'changed'
    }

    return should_drop, similarity, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 7 — SERIALIZACJA PARAMETRÓW AUTO-MODE
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_frame_params(params: dict) -> bytes:
    """Serializuje parametry klatki do formatu binarnego."""
    out = bytearray()

    flags = 0x01 if params.get('auto_mode', False) else 0x00
    flags |= 0x02 if params.get('is_iframe', False) else 0x00  # NOWOŚĆ v0.3
    flags |= 0x04 if params.get('dropped', False) else 0x00    # NOWOŚĆ v0.3
    out.append(flags)

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

    params_to_write = []
    for pid, key in param_ids:
        if key in params:
            params_to_write.append((pid, params[key]))

    out.append(len(params_to_write))

    for pid, value in params_to_write:
        out.append(pid)
        out.extend(struct.pack('>f', float(value)))

    return bytes(out)


def deserialize_frame_params(data: bytes, offset: int = 0) -> tuple:
    """Deserializuje parametry klatki z formatu binarnego."""
    params = {}

    flags = data[offset]
    params['auto_mode'] = bool(flags & 0x01)
    params['is_iframe'] = bool(flags & 0x02)  # NOWOŚĆ v0.3
    params['dropped'] = bool(flags & 0x04)    # NOWOŚĆ v0.3
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
# CZĘŚĆ 8 — AUTO-TUNING PARAMETRÓW
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_frame_for_auto_tuning(curr_Y: np.ndarray, prev_Y: np.ndarray = None) -> dict:
    """Analizuje klatkę i zwraca metryki potrzebne do auto-tuningu parametrów."""
    h, w = curr_Y.shape

    variance = float(np.var(curr_Y))
    edge_density = _estimate_edge_density(curr_Y)

    metrics = {
        'variance': variance,
        'edge_density': edge_density,
        'mad_avg': 0.0,
        'mad_std': 0.0,
        'complexity': min(1.0, variance / 2500.0),
        'motion_ratio': 0.0,
        'scene_diff': 0.0,
    }

    if prev_Y is not None:
        diff = np.abs(curr_Y[:prev_Y.shape[0], :prev_Y.shape[1]] - prev_Y)
        mad_map = diff.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))

        metrics['mad_avg'] = float(np.mean(mad_map))
        metrics['mad_std'] = float(np.std(mad_map))

        motion_threshold = 3.0
        motion_blocks = np.sum(mad_map > motion_threshold)
        total_blocks = mad_map.size
        metrics['motion_ratio'] = motion_blocks / total_blocks if total_blocks > 0 else 0.0
        metrics['scene_diff'] = metrics['mad_avg']
        metrics['complexity'] = min(1.0, metrics['complexity'] * 0.7 + metrics['motion_ratio'] * 0.3)

    return metrics


def _estimate_edge_density(plane: np.ndarray) -> float:
    """Szacuje gęstość krawędzi w obrazie."""
    h, w = plane.shape
    if h < 4 or w < 4:
        return 0.0

    gx = np.abs(plane[1:-1, 2:] - plane[1:-1, :-2])
    gy = np.abs(plane[2:, 1:-1] - plane[:-2, 1:-1])

    edge_strength = np.sqrt(gx**2 + gy**2)
    edge_pixels = np.sum(edge_strength > 30)

    return edge_pixels / ((h-2) * (w-2))


def auto_tune_params(metrics: dict, base_params: dict, is_iframe: bool = False,
                     i_frame_quality_boost: float = 0.7) -> dict:
    """
    Automatycznie doiera parametry encodowania.

    NOWOŚĆ v0.3:
      - is_iframe: czy to I-frame (klatka kluczowa)
      - i_frame_quality_boost: mnożnik jakości dla I-frames (0.7 = wyższa jakość)
    """
    params = base_params.copy()
    params['auto_mode'] = True
    params['is_iframe'] = is_iframe

    complexity = metrics['complexity']
    motion_ratio = metrics['motion_ratio']
    mad_avg = metrics['mad_avg']
    scene_diff = metrics['scene_diff']

    params['mad_avg'] = mad_avg
    params['complexity'] = complexity
    params['scene_diff'] = scene_diff

    # ─── Q_Y (kwantyzacja luma) ─────────────────────────────────────────────
    q_y_factor = 1.0

    if mad_avg < 3.0:
        q_y_factor = 0.7 + 0.3 * (mad_avg / 3.0)
    elif mad_avg < 10.0:
        q_y_factor = 1.0 + 0.2 * ((mad_avg - 3.0) / 7.0)
    else:
        q_y_factor = 1.2 + 0.3 * min(1.0, (mad_avg - 10.0) / 20.0)

    if complexity > 0.7:
        q_y_factor *= 1.1

    # NOWOŚĆ v0.3: Boost jakości dla I-frames
    if is_iframe:
        q_y_factor *= i_frame_quality_boost

    params['q_y'] = np.clip(base_params.get('q_y', 22.0) * q_y_factor,
                            AUTO_Q_Y_MIN, AUTO_Q_Y_MAX)

    # ─── Q_C (kwantyzacja chroma) ───────────────────────────────────────────
    q_c_factor = q_y_factor * 1.1

    # NOWOŚĆ v0.3: Boost jakości dla I-frames
    if is_iframe:
        q_c_factor *= i_frame_quality_boost

    params['q_c'] = np.clip(base_params.get('q_c', 40.0) * q_c_factor,
                            AUTO_Q_C_MIN, AUTO_Q_C_MAX)

    # ─── Search Range ────────────────────────────────────────────────────────
    if motion_ratio < 0.1:
        sr_factor = 0.6
    elif motion_ratio < 0.3:
        sr_factor = 1.0
    else:
        sr_factor = 1.0 + 0.5 * (motion_ratio - 0.3)

    base_sr = base_params.get('search_range', 24)
    params['search_range'] = int(np.clip(base_sr * sr_factor, AUTO_SR_MIN, AUTO_SR_MAX))

    # ─── Intra Threshold Factor ──────────────────────────────────────────────
    if motion_ratio > 0.5 and mad_avg > 15:
        params['intra_factor'] = AUTO_INTRA_F_MIN
    else:
        params['intra_factor'] = np.clip(
            base_params.get('intra_factor', 4.0) * (1.0 - 0.2 * motion_ratio),
            AUTO_INTRA_F_MIN, AUTO_INTRA_F_MAX
        )

    # ─── Subpixel ────────────────────────────────────────────────────────────
    params['use_subpixel'] = mad_avg > 2.0

    return params


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 9 — SERIALIZACJA QDIFF BLOKÓW
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
    return struct.pack('>4h', *vals)


def _unpack_dct_lowfreq(data: bytes, offset: int, block_size: int) -> tuple:
    vals = struct.unpack_from('>4h', data, offset)
    blk = np.zeros((block_size, block_size), dtype=np.int16)
    for i, (r, c) in enumerate(_LOWFREQ_IDX):
        blk[r, c] = vals[i]
    return blk, offset + 8


def _pack_block_coeffs(q_dct: np.ndarray) -> bytes:
    return q_dct.astype(np.int16).flatten().tobytes()


def _unpack_block_coeffs(data: bytes, offset: int, bs: int) -> tuple:
    n = bs * bs
    arr = np.frombuffer(data[offset:offset + n*2], dtype=np.int16).reshape(bs, bs).copy()
    return arr, offset + n*2


def serialize_pframe_blocks(block_list: list, h: int, w: int, params: dict = None) -> bytes:
    """Serializuje listę bloków QDiff do strumienia bajtów."""
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
    """Deserializuje strumień bajtów do listy bloków QDiff."""
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
# CZĘŚĆ 10 — KLASYFLKACJA BLOKU → FAMILY_ID
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_block(q_dct_y: np.ndarray,
                    q_dct_u: np.ndarray,
                    q_dct_v: np.ndarray,
                    mv_qp: tuple,
                    sad_mc: float,
                    sad_static: float,
                    intra_threshold: float) -> int:
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
# CZĘŚĆ 11 — PRZETWARZANIE RZĘDU BLOKÓW P-FRAME
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
        q_dct_v = np.round(apply_idct2(curr_v_block - match_v) / Q_C).astype(np.int16)

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
# CZĘŚĆ 12 — GŁÓWNA KLASA KODEKA QDIFF
# ═══════════════════════════════════════════════════════════════════════════════

class QDiffCodec:
    """
    Kodek QDiff v0.3 z obsługą:
    - Trybu auto (per-frame adaptive params)
    - VFR (Variable Frame Rate) z detekcją klatek statycznych
    - Boost jakości dla I-frames
    """

    def __init__(self, block_size: int = 16,
                 search_range: int = 24,
                 use_subpixel: bool = True,
                 q_y: float = 22.0,
                 q_c: float = 40.0,
                 adaptive_q: bool = False,
                 intra_threshold_factor: float = 4.0,
                 auto_mode: bool = False,
                 vfr_mode: bool = False,
                 drop_threshold: float = DEFAULT_DROP_THRESHOLD,
                 i_frame_quality_boost: float = 0.7):
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
        self.prev_Y = self.prev_U = self.prev_V = None
        self.prev_frame = None  # Do VFR detection

        self.base_params = {
            'q_y': q_y,
            'q_c': q_c,
            'search_range': search_range,
            'intra_factor': intra_threshold_factor,
            'use_subpixel': use_subpixel,
        }

    def _get_frame_params(self, Y: np.ndarray, h: int, w: int, is_iframe: bool = False) -> dict:
        """Zwraca parametry dla bieżącej klatki."""
        if not self.auto_mode:
            params = self.base_params.copy()
            params['is_iframe'] = is_iframe
            # Boost jakości dla I-frames nawet w trybie manual
            if is_iframe:
                params['q_y'] = self.Q_Y_base * self.i_frame_quality_boost
                params['q_c'] = self.Q_C_base * self.i_frame_quality_boost
            return params

        metrics = analyze_frame_for_auto_tuning(Y, self.prev_Y)
        return auto_tune_params(metrics, self.base_params, is_iframe, self.i_frame_quality_boost)

    # ─── I-Frame ──────────────────────────────────────────────────────────────

    def _encode_iframe(self, Y, U, V, h, w, params: dict = None) -> dict:
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

        result = {
            'type': 'I',
            'Y': q_Y_plane, 'U': q_U_plane, 'V': q_V_plane,
            'h': h, 'w': w,
        }

        if params:
            result['params'] = params

        return result

    # ─── P-Frame ──────────────────────────────────────────────────────────────

    def _encode_pframe(self, Y, U, V, h, w, params: dict = None) -> dict:
        bs = self.block_size

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

        if params:
            result['params'] = params

        return result

    def _compute_mad_map(self, curr_Y, h, w, bs):
        rows = h // bs; cols = w // bs
        c = curr_Y[:rows*bs, :cols*bs].astype(np.float32)
        p = self.prev_Y[:rows*bs, :cols*bs].astype(np.float32)
        return np.abs(c - p).reshape(rows, bs, cols, bs).mean(axis=(1, 3))

    # ─── Dispatcher enkodowania z VFR ────────────────────────────────────────────

    def encode_frame(self, img_np: np.ndarray, force_iframe: bool = False) -> dict:
        h, w, _ = img_np.shape
        h_pad = h - (h % 16); w_pad = w - (w % 16)
        Y, U, V = rgb_to_yuv420(img_np[:h_pad, :w_pad].astype(np.float32))

        # NOWOŚĆ v0.3: VFR - sprawdź czy klatka statyczna
        if self.vfr_mode and self.prev_frame is not None:
            should_drop, similarity, metrics = should_drop_frame(
                img_np, self.prev_frame,
                drop_threshold=self.drop_threshold,
                similarity_threshold=1.0 - self.drop_threshold
            )

            if should_drop:
                print(f"   [VFR DROP] podobieństwo={similarity*100:.2f}%", flush=True)
                return {
                    'type': 'DROPPED',
                    'similarity': similarity,
                    'h': h_pad, 'w': w_pad,
                }

        is_iframe = self.prev_Y is None or force_iframe
        params = self._get_frame_params(Y, h_pad, w_pad, is_iframe)

        # Zapisz poprzednią klatkę dla VFR
        if self.vfr_mode:
            self.prev_frame = img_np.copy()

        if is_iframe:
            return self._encode_iframe(Y, U, V, h_pad, w_pad, params)
        return self._encode_pframe(Y, U, V, h_pad, w_pad, params)

    # ─── Dekodowanie ────────────────────────────────────────────────────────────

    def _decode_iframe(self, data: dict, params: dict = None) -> np.ndarray:
        bs = self.block_size; bs_c = bs // 2
        h, w = data['h'], data['w']

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

    def _decode_pframe(self, data: dict, params: dict = None) -> np.ndarray:
        h, w = data['h'], data['w']
        bs = self.block_size; bs_c = bs // 2

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
        elif ft == 'DROPPED':
            # Zwróć poprzednią klatkę
            return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)
        raise ValueError(f"Nieznany typ klatki: {ft}")


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 13 — SERIALIZACJA KLATEK DO PLIKU
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_iframe(data: dict, params: dict = None, timestamp: float = 0.0) -> bytes:
    """I-frame: rozkład DCT płaszczyzn → raw int16."""
    h, w = data['h'], data['w']
    out = bytearray()

    # Timestamp (8B double)
    out.extend(struct.pack('>d', timestamp))

    if params is not None and params.get('auto_mode', False):
        out.extend(serialize_frame_params(params))

    out.extend(struct.pack('>HH', h, w))
    out.extend(data['Y'].astype(np.int16).flatten().tobytes())
    out.extend(data['U'].astype(np.int16).flatten().tobytes())
    out.extend(data['V'].astype(np.int16).flatten().tobytes())
    return bytes(out)


def _deserialize_iframe(raw: bytes, offset: int, auto_mode: bool = False) -> tuple:
    """Deserializuje I-frame."""
    params = {}

    # Timestamp
    timestamp = struct.unpack_from('>d', raw, offset)[0]
    offset += 8

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

    return {'type': 'I', 'Y': Y, 'U': U, 'V': V, 'h': h, 'w': w, 'timestamp': timestamp}, params, offset


def _serialize_pframe(data: dict, params: dict = None, timestamp: float = 0.0) -> bytes:
    """P-frame: serializacja bloków QDiff."""
    h, w = data['h'], data['w']
    out = bytearray()

    # Timestamp (8B double)
    out.extend(struct.pack('>d', timestamp))

    serialized = serialize_pframe_blocks(data['blocks'], h, w, params)
    out.extend(serialized)
    return bytes(out)


def _deserialize_pframe(raw: bytes, offset: int, auto_mode: bool = False) -> tuple:
    """Deserializuje P-frame."""
    # Timestamp
    timestamp = struct.unpack_from('>d', raw, offset)[0]
    offset += 8

    blocks, params, offset = deserialize_pframe_blocks(raw, offset, auto_mode)

    if blocks:
        max_y = max(b['y'] for b in blocks) + 16
        max_x = max(b['x'] for b in blocks) + 16
        h, w = max_y, max_x
    else:
        h, w = 0, 0

    return {'type': 'P', 'blocks': blocks, 'h': h, 'w': w, 'timestamp': timestamp}, params, offset


def _serialize_dropped_frame(timestamp: float, similarity: float) -> bytes:
    """Serializuje marker dropped frame."""
    out = bytearray()
    out.extend(struct.pack('>d', timestamp))
    out.extend(struct.pack('>f', similarity))
    return bytes(out)


def _deserialize_dropped_frame(raw: bytes, offset: int) -> tuple:
    """Deserializuje dropped frame."""
    timestamp = struct.unpack_from('>d', raw, offset)[0]
    offset += 8
    similarity = struct.unpack_from('>f', raw, offset)[0]
    offset += 4
    return {'type': 'DROPPED', 'timestamp': timestamp, 'similarity': similarity}, offset


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 14 — WIDEO I/O Z PARAMETRAMI ŹRÓDŁOWYMI
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


def _write_frames_with_source_params(frames, output_path: str, source_params: dict):
    """
    Zapisuje klatki do MP4 używając parametrów z wejściowego wideo.
    Wykorzystuje imageio-ffmpeg z tymi samymi parametrami kodowania.
    """
    errors = []
    h, w = frames[0].shape[:2]

    # Parametry z source
    fps = source_params.get('fps', 25.0)
    crf = source_params.get('crf', 18)
    preset = source_params.get('preset', 'medium')
    codec = source_params.get('codec', 'libx264')

    print(f"  [zapis] Parametry: fps={fps:.2f}, crf={crf}, preset={preset}, codec={codec}", flush=True)

    # Metoda 1: imageio_ffmpeg (preferowana - używa tych samych narzędzi co odczyt)
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(
            output_path, (w, h),
            fps=fps,
            codec=codec,
            quality=None,
            output_params=[
                '-crf', str(crf),
                '-preset', preset,
                '-pix_fmt', 'yuv420p',  # Standard dla kompatybilności
            ]
        )
        writer.send(None)
        for frame in frames:
            writer.send(np.asarray(frame, dtype=np.uint8).tobytes())
        writer.close()
        print(f"  [zapis] imageio_ffmpeg → {output_path}", flush=True)
        return
    except Exception as e:
        errors.append(f"imageio_ffmpeg: {e}")

    # Metoda 2: cv2 (fallback)
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            raise RuntimeError("VideoWriter nie otworzył")
        for frame in frames:
            out.write(cv2.cvtColor(np.asarray(frame, np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  [zapis] cv2 → {output_path} (UWAGA: parametry źródła nie zachowane)", flush=True)
        return
    except Exception as e:
        errors.append(f"cv2: {e}")

    raise RuntimeError("Nie udało się zapisać wideo:\n" + "\n".join(f"  • {e}" for e in errors))


def _write_frames(frames, output_path: str, fps: float = 25.0):
    """Zapisuje klatki do MP4 (legacy - bez parametrów źródła)."""
    source_params = {'fps': fps, 'crf': 18, 'preset': 'medium', 'codec': 'libx264'}
    _write_frames_with_source_params(frames, output_path, source_params)


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 15 — ENKODOWANIE I DEKODOWANIE WIDEO
# ═══════════════════════════════════════════════════════════════════════════════

def encode_video(input_path: str, output_path: str,
                 max_frames: int = 30, full: bool = False,
                 q_y: float = 22.0, q_c: float = 40.0,
                 search_range: int = 24,
                 use_subpixel: bool = True,
                 adaptive_q: bool = False,
                 auto_mode: bool = False,
                 vfr_mode: bool = False,          # NOWOŚĆ v0.3
                 drop_threshold: float = DEFAULT_DROP_THRESHOLD,  # NOWOŚĆ v0.3
                 i_frame_quality_boost: float = 0.7,  # NOWOŚĆ v0.3
                 keyframe_interval: int = 50,
                 scene_cut_threshold: float = 35.0,
                 intra_threshold_factor: float = 4.0,
                 match_source: bool = False):     # NOWOŚĆ v0.3

    # Ekstrakcja parametrów źródła
    source_params = {}
    if match_source:
        print(f"\n  [MATCH SOURCE] Ekstrakcja parametrów z wejściowego wideo...", flush=True)
        source_params = extract_source_video_params(input_path)
        print(f"    FPS: {source_params['fps']:.2f}", flush=True)
        print(f"    Rozdzielczość: {source_params['width']}x{source_params['height']}", flush=True)
        print(f"    CRF: {source_params['crf']}", flush=True)
        print(f"    Preset: {source_params['preset']}", flush=True)
        if source_params.get('bitrate'):
            print(f"    Bitrate: {source_params['bitrate']} bps", flush=True)

    mode_parts = []
    if auto_mode:
        mode_parts.append("AUTO")
    if vfr_mode:
        mode_parts.append("VFR")
    if match_source:
        mode_parts.append("MATCH_SRC")
    mode_str = "+".join(mode_parts) if mode_parts else "MANUAL"

    print(f"\n╔══ QDIFF CODEC v0.3 — ENKODOWANIE [{mode_str}] ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Q_Y={q_y}  Q_C={q_c}  search={search_range}")
    if vfr_mode:
        print(f"  VFR: ON (drop_threshold={drop_threshold*100:.2f}%)")
    if auto_mode:
        print(f"  AUTO: ON (i_frame_boost={i_frame_quality_boost})")
    print(f"  Architektura: 8-rodzinny QDiff")
    print(f"╚{'═'*50}╝\n", flush=True)

    frames = _read_frames(input_path, max_frames, full)
    n = len(frames)
    print(f"\n  Wczytano {n} klatek", flush=True)

    codec = QDiffCodec(
        q_y=q_y, q_c=q_c, search_range=search_range,
        use_subpixel=use_subpixel, adaptive_q=adaptive_q,
        intra_threshold_factor=intra_threshold_factor,
        auto_mode=auto_mode,
        vfr_mode=vfr_mode,
        drop_threshold=drop_threshold,
        i_frame_quality_boost=i_frame_quality_boost)

    encoded_frames = []
    frame_params_list = []
    timestamps = []  # NOWOŚĆ v0.3 - timestamps dla VFR
    dropped_count = 0

    total_bytes_before = 0
    total_bytes_after = 0

    # FPS dla timestampów
    fps = source_params.get('fps', 25.0) if match_source else 25.0
    frame_duration = 1.0 / fps

    for i, frame in enumerate(frames):
        t0 = time.time()
        h, w = frame.shape[:2]
        timestamp = i * frame_duration

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

        # NOWOŚĆ v0.3: Obsługa dropped frames
        if ft == 'DROPPED':
            frame_bytes = _FRAME_DROPPED + _serialize_dropped_frame(timestamp, data['similarity'])
            dropped_count += 1
            total_bytes_before += frame.shape[0] * frame.shape[1] * 3
            total_bytes_after += len(frame_bytes)
            encoded_frames.append(frame_bytes)
            print(f"  Klatka {i+1}/{n} [DROPPED] sim={data['similarity']*100:.1f}%", flush=True)
            continue

        if params:
            frame_params_list.append(params)
            if i == 0 or len(frame_params_list) <= 3:
                param_str = "  ".join(f"{PARAM_NAMES.get(k, k)}={v:.2f}"
                                      for k, v in params.items()
                                      if k in ['q_y', 'q_c', 'search_range', 'mad_avg'])
                extra = " [I-FRAME BOOST]" if params.get('is_iframe') else ""
                print(f"    [AUTO params]{extra} {param_str}", flush=True)

        # Serializacja
        if ft == 'I':
            frame_bytes = _FRAME_I + _serialize_iframe(data, params, timestamp)
        else:
            frame_bytes = _FRAME_P + _serialize_pframe(data, params, timestamp)

        size_kb = len(frame_bytes) / 1024
        elapsed = time.time() - t0
        print(f"  Klatka {i+1}/{n} [{ft}] → {size_kb:.1f} KB ({elapsed:.2f}s)", flush=True)

        encoded_frames.append(frame_bytes)
        total_bytes_before += frame.shape[0] * frame.shape[1] * 3
        total_bytes_after  += len(frame_bytes)

    # Zapis z kompresją zstd
    print(f"\n  Kompresja zstd...", flush=True)

    # Magic: QDF3 dla v0.3 (VFR)
    magic = _QDIFF_MAGIC_V3

    # Rozszerzony nagłówek v0.3
    header = bytearray()
    header.extend(struct.pack('>4s', magic))
    header.extend(struct.pack('>HH', frames[0].shape[1], frames[0].shape[0]))  # W, H
    header.extend(struct.pack('>f', fps))  # FPS
    header.extend(struct.pack('>I', n))    # Total frames
    header.extend(struct.pack('>I', n - dropped_count))  # Kept frames
    header.extend(struct.pack('>f', q_y))  # Base Q_Y
    header.extend(struct.pack('>f', q_c))  # Base Q_C

    # Source params (jeśli match_source)
    if match_source and source_params:
        header.extend(struct.pack('>B', 1))  # Flag: source params present
        header.extend(struct.pack('>B', source_params.get('crf', 18)))
        # Preset jako string (max 16 znaków)
        preset_bytes = source_params.get('preset', 'medium').encode('utf-8')[:16].ljust(16, b'\x00')
        header.extend(preset_bytes)
    else:
        header.extend(struct.pack('>B', 0))  # Flag: no source params

    raw_stream = bytes(header) + b''.join(encoded_frames) + _EOF_MARKER

    cctx = zstd.ZstdCompressor(level=6)
    compressed = cctx.compress(raw_stream)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    ratio = total_bytes_before / len(compressed)
    print(f"\n✓ SUKCES!")
    print(f"  Klatki: {n} (kept: {n-dropped_count}, dropped: {dropped_count})")
    print(f"  Raw: {total_bytes_before//1024} KB")
    print(f"  Pre-zstd: {total_bytes_after//1024} KB  |  Po zstd: {len(compressed)//1024} KB")
    print(f"  Kompresja: {ratio:.1f}× ({len(compressed)/1024:.1f} KB)", flush=True)

    # Statystyki VFR
    if vfr_mode and dropped_count > 0:
        drop_ratio = dropped_count / n * 100
        print(f"\n  Statystyki VFR:", flush=True)
        print(f"    Dropowane: {dropped_count}/{n} ({drop_ratio:.1f}%)", flush=True)
        print(f"    Oszczędność: ~{dropped_count * frame.shape[0] * frame.shape[1] * 3 // 1024} KB", flush=True)


def decode_video(input_path: str, output_path: str, fps: float = 25.0):
    print(f"\n╔══ QDIFF CODEC v0.3 — DEKODOWANIE ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Wyjście: {output_path}", flush=True)

    with open(input_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        raw = dctx.stream_reader(f).read()

    magic = raw[:4]

    # Sprawdź wersję formatu
    if magic == _QDIFF_MAGIC_V3:
        auto_mode = True
        vfr_mode = True
        print(f"  Format: QDF3 (v0.3 VFR + AUTO)", flush=True)
    elif magic == _QDIFF_MAGIC_V2:
        auto_mode = True
        vfr_mode = False
        print(f"  Format: QDF2 (v0.2 AUTO)", flush=True)
    elif magic == _QDIFF_MAGIC:
        auto_mode = False
        vfr_mode = False
        print(f"  Format: QDIF (legacy)", flush=True)
    else:
        raise ValueError(f"Nieznany format pliku (magic={magic})")

    # Nagłówek v0.3
    offset = 4
    W, H = struct.unpack_from('>HH', raw, offset); offset += 4
    source_fps = struct.unpack_from('>f', raw, offset)[0]; offset += 4
    total_frames, kept_frames = struct.unpack_from('>II', raw, offset); offset += 8
    q_y = struct.unpack_from('>f', raw, offset)[0]; offset += 4
    q_c = struct.unpack_from('>f', raw, offset)[0]; offset += 4

    # Source params
    source_params = {'fps': source_fps if source_fps > 0 else fps}
    has_source_params = raw[offset]; offset += 1
    if has_source_params:
        source_crf = raw[offset]; offset += 1
        preset_bytes = raw[offset:offset+16]; offset += 16
        source_params['crf'] = source_crf
        source_params['preset'] = preset_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
        print(f"  Source params: crf={source_crf}, preset={source_params['preset']}", flush=True)

    print(f"  W: {W}, H: {H}, FPS: {source_fps:.2f}", flush=True)
    print(f"  Total frames: {total_frames}, Kept: {kept_frames}", flush=True)

    codec = QDiffCodec(q_y=q_y, q_c=q_c, auto_mode=auto_mode, vfr_mode=vfr_mode)
    decoded_frames = []
    dropped_count = 0

    while offset < len(raw) - 2:
        if raw[offset:offset+2] == _EOF_MARKER:
            break

        ft_byte = raw[offset:offset+1]; offset += 1

        if ft_byte == _FRAME_I:
            data, params, offset = _deserialize_iframe(raw, offset, auto_mode)
        elif ft_byte == _FRAME_P:
            data, params, offset = _deserialize_pframe(raw, offset, auto_mode)
        elif ft_byte == _FRAME_DROPPED:
            data, offset = _deserialize_dropped_frame(raw, offset)
            dropped_count += 1
            # Dla dropped frame - zdekoduj jako powtórkę poprzedniej
            img = codec.decode_frame({'type': 'DROPPED'})
            decoded_frames.append(img)
            print(f"  Zdekodowano klatkę {len(decoded_frames)} [DROPPED] (powtórka)", flush=True)
            continue
        else:
            print(f"  [WARN] Nieznany typ klatki @ offset {offset-1}: {ft_byte}", flush=True)
            break

        img = codec.decode_frame(data, params)
        decoded_frames.append(img)

        if auto_mode and params:
            extra = " [I-FRAME]" if params.get('is_iframe') else ""
            print(f"  Zdekodowano klatkę {len(decoded_frames)} [{data['type']}]{extra}", flush=True)
        else:
            print(f"  Zdekodowano klatkę {len(decoded_frames)} [{data['type']}]", flush=True)

    print(f"\n  Łącznie zdekodowano: {len(decoded_frames)} klatek (dropped: {dropped_count})", flush=True)

    # Użyj parametrów źródła przy zapisie
    _write_frames_with_source_params(decoded_frames, output_path, source_params)
    print(f"\n✓ SUKCES! → {output_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CZĘŚĆ 16 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"[qdiff] DCT backend: {_DCT_BACKEND}", flush=True)
    print(f"[qdiff] Interpolacja: {_NUMBA_INFO}", flush=True)
    print(f"[qdiff] VLC: {'ON' if _VLC_ENABLED else 'OFF'}", flush=True)
    print(f"[qdiff] Wątki: {_N_WORKERS}", flush=True)

    parser = argparse.ArgumentParser(
        description="QDIFF CODEC v0.3 — VFR + AUTO + Source Parameter Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QDIFF CODEC v0.3 Features:
─────────────────────────────────────────────────────────────────────────────────

1. VARIABLE FRAME RATE (VFR) --vfr
   • Detekcja klatek statycznych (visual margin dropout 99.9%)
   • Klatki z różnicą < 0.1% pikseli są dropowane
   • Timestamps zapisywane dla każdej klatki
   • Dostosuj próg: --drop-threshold 0.002 (0.2%)

2. KEYFRAME QUALITY BOOST --i-frame-quality
   • I-frames (klatki kluczowe) mają wyższą jakość
   • Domyślnie Q_Y i Q_C mnożone przez 0.7 dla I-frames
   • Niższa wartość = wyższa jakość (0.5 = bardzo wysoka, 1.0 = brak boost)

3. SOURCE PARAMETER EXTRACTION --match-source
   • Automatyczna ekstrakcja CRF, preset, FPS z wejściowego MP4
   • Użycie tych samych parametrów przy dekodowaniu do MP4
   • Zachowanie jakości wyjściowej

4. AUTO MODE (-a/--auto)
   • Automatyczny dobór Q_Y, Q_C, search_range per klatka
   • Analiza ruchu i złożoności sceny

Rodziny QDiff:
  000 SKIP_TRUE    — identyczny blok, 0B payloadu
  001 SKIP_NOISE   — identyczny + wygładź ringing, 1B
  010 MV_ONLY      — ruch bez residualu, 2-5B
  011 DC_ONLY      — DC + MV, 8B
  100 LOW_FREQ     — 4 DCT + MV, 16B
  101 FULL_DCT     — pełny DCT + MV (dawny DETAIL)
  110 INTRA_PATCH  — blok intra (bez referencji)
  111 SCENE_CUT    — marker, 0B
        """)

    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')
    parser.add_argument('-n', '--frames',  type=int,   default=30)
    parser.add_argument('-f', '--full',    action='store_true')

    # Tryby
    parser.add_argument('-a', '--auto',    action='store_true',
                        help='Tryb AUTO: automatyczny dobór parametrów per-klatka')
    parser.add_argument('--vfr',           action='store_true',
                        help='Variable Frame Rate: detekcja i dropowanie klatek statycznych')
    parser.add_argument('--match-source',  action='store_true',
                        help='Użyj parametrów kodowania z wejściowego wideo')

    # Parametry VFR
    parser.add_argument('--drop-threshold', type=float, default=DEFAULT_DROP_THRESHOLD,
                        help='Próg różnicy dla dropowania klatek (domyślnie 0.001 = 0.1%%)')

    # Parametry jakości
    parser.add_argument('--i-frame-quality', type=float, default=0.7,
                        help='Mnożnik Q dla I-frames (0.5-1.0, niżej = wyższa jakość)')
    parser.add_argument('--q-y',          type=float, default=22.0)
    parser.add_argument('--q-c',          type=float, default=40.0)
    parser.add_argument('--search-range', type=int,   default=24)
    parser.add_argument('--no-subpixel',  action='store_true')
    parser.add_argument('--adaptive-q',   action='store_true')
    parser.add_argument('--keyframe-interval', type=int, default=50)
    parser.add_argument('--scene-cut',    type=float, default=35.0)
    parser.add_argument('--intra-factor', type=float, default=4.0)
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
            auto_mode=args.auto,
            vfr_mode=args.vfr,
            drop_threshold=args.drop_threshold,
            i_frame_quality_boost=args.i_frame_quality,
            keyframe_interval=args.keyframe_interval,
            scene_cut_threshold=args.scene_cut,
            intra_threshold_factor=args.intra_factor,
            match_source=args.match_source,
        )
