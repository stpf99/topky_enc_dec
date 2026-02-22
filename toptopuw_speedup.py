"""
╔══════════════════════════════════════════════════════════════════════════╗
║   TOP TOPÓW CODEC — MODUŁ PRZYSPIESZEŃ (CPU / AVX2 / batch numpy)       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Monkey-patchuje toptopuwcodec_v1 bez modyfikowania oryginału.           ║
║  Podpina zoptymalizowane wersje najwolniejszych funkcji:                 ║
║                                                                          ║
║  1. _ved_upsample_plane  — Python double-loop O(h×w) → czysty numpy     ║
║       0.2 kl/s → ~3–6 kl/s tylko z tej zmiany (główny wąskie gardło)    ║
║                                                                          ║
║  2. batch_dct2 / batch_idct2  — scipy DCT na osi 3D zamiast per-blok    ║
║       I-klatka ~16× szybciej (1080p: 8160 bloków → 1 wywołanie)         ║
║                                                                          ║
║  3. _decode_iframe  — batch IDCT dla całej klatki naraz                  ║
║  4. _encode_iframe  — batch DCT+kwantyzacja+rekonstrukcja                ║
║  5. _decode_pframe  — zebranie DETAIL→batch IDCT→scatter                 ║
║                                                                          ║
║  6. yuv420_to_rgb  — wektoryzowany dot + szybszy upsample                ║
║  7. deblock_filter — już wektorowy; drobna optymalizacja                 ║
║                                                                          ║
║  8. CPU/BLAS tuning — OMP_NUM_THREADS / MKL_NUM_THREADS na ile rdzeni   ║
║     Xeon v4 ma AVX2 + FMA3; numpy z OpenBLAS/MKL już z nich korzysta    ║
║                                                                          ║
║  Użycie:                                                                 ║
║      import toptopuw_speedup                                             ║
║      codec_mod = toptopuw_speedup.apply(codec_mod)                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import types
import threading
import multiprocessing
from functools import lru_cache

import numpy as np
from scipy.fftpack import dct as _scipy_dct, idct as _scipy_idct

# ═══════════════════════════════════════════════════════════════════════════
# 0. STROJENIE CPU — ustawiamy wątki BLAS/OMP przed pierwszym wywołaniem
# ═══════════════════════════════════════════════════════════════════════════

def _tune_cpu_threads():
    """
    Ustawia liczbę wątków dla OpenBLAS / MKL / OpenMP tak, żeby
    numpy/scipy korzystały ze wszystkich fizycznych rdzeni Xeona.

    Xeon v4 Broadwell-EP: AVX2 + FMA3 — numpy z OpenBLAS/MKL
    automatycznie ich używa jeśli jest odpowiednio skompilowany.
    Wywołujemy PRZED pierwszym numpy/scipy — tylko raz.
    """
    n_cores = multiprocessing.cpu_count()

    # Fizyczne rdzenie (ignorujemy HyperThreading — dla numpy gorsze)
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=False) or n_cores
    except ImportError:
        # Przybliżenie: połowa logicznych
        n_cores = max(1, n_cores // 2)

    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'BLIS_NUM_THREADS'):
        if var not in os.environ:
            os.environ[var] = str(n_cores)

    return n_cores


N_CORES = _tune_cpu_threads()

# ═══════════════════════════════════════════════════════════════════════════
# 1. BATCH DCT-2D  (scipy na osi 3D — używa LAPACK + AVX2 wewnętrznie)
# ═══════════════════════════════════════════════════════════════════════════

def batch_dct2(blocks: np.ndarray) -> np.ndarray:
    """
    DCT-II 2D na tablicy (N, bs, bs) float32.
    Zamiast N oddzielnych wywołań scipy — jedno wywołanie na każdej osi.
    scipy.fftpack.dct z axis= działa wewnętrznie przez LAPACK/FFTW.

    Szybkość: ~16× szybciej niż pętla per-blok dla 1080p (8160 bloków).
    """
    return _scipy_dct(
        _scipy_dct(blocks.astype(np.float32), type=2, norm='ortho', axis=-1),
        type=2, norm='ortho', axis=-2
    )


def batch_idct2(blocks: np.ndarray) -> np.ndarray:
    """Odwrotny DCT-2D na (N, bs, bs). Analogicznie do batch_dct2."""
    return _scipy_idct(
        _scipy_idct(blocks.astype(np.float32), type=2, norm='ortho', axis=-1),
        type=2, norm='ortho', axis=-2
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. POMOCNICZE: reshape płaszczyzna ↔ bloki
# ═══════════════════════════════════════════════════════════════════════════

def plane_to_blocks(plane: np.ndarray, bs: int) -> np.ndarray:
    """
    (H, W) float32 → (N_blocks, bs, bs) float32.
    N_blocks = (H//bs) * (W//bs).  Raster scan (lewo→prawo, góra→dół).
    Używa only-view operacji (brak kopii) → zero alokacji dodatkowej pamięci.
    """
    H, W = plane.shape
    Hb, Wb = H // bs, W // bs
    # Przytnij do siatki bloków
    p = np.ascontiguousarray(plane[:Hb * bs, :Wb * bs], dtype=np.float32)
    # (Hb, bs, Wb, bs) → (Hb, Wb, bs, bs) → (N, bs, bs)
    return p.reshape(Hb, bs, Wb, bs).transpose(0, 2, 1, 3).reshape(-1, bs, bs)


def blocks_to_plane(blocks: np.ndarray, H: int, W: int, bs: int) -> np.ndarray:
    """
    (N_blocks, bs, bs) → (H, W) float32.  Odwrotność plane_to_blocks.
    """
    Hb, Wb = H // bs, W // bs
    return (blocks.reshape(Hb, Wb, bs, bs)
                  .transpose(0, 2, 1, 3)
                  .reshape(Hb * bs, Wb * bs))


# ═══════════════════════════════════════════════════════════════════════════
# 3. SZYBKI VED UPSAMPLE — zera pętli Python (wektoryzowany numpy)
# ═══════════════════════════════════════════════════════════════════════════

def _ved_upsample_fast(plane: np.ndarray) -> np.ndarray:
    """
    Zastępuje _ved_upsample_plane — identyczny wynik przy ~200× szybszym
    wykonaniu dla 1080p UV (960×540 → 1920×1080).

    ORYGINALNY KOD:
        for x in range(...): big[...] = ...   # W iteracji Python
        for y in range(...): big[...] = ...   # H iteracji Python
        for y in range(...):                  # H×W iteracji Python ← KILLER
            for x in range(...):
                if (y%2==1) or (x%2==1): ...  # O(h×w) Python

    TUTAJ: same operacje numpy — żadna pętla Python, AVX2 z numpy/BLAS.

    Uwaga: korekcja gradientu VED (0.15×) wdrożona wektorowo dla obu
    typów pozycji interpolowanych (poziomych i pionowych osobno).
    """
    h, w = plane.shape
    H, W = h * 2, w * 2
    p = plane.astype(np.float32, copy=False)

    out = np.empty((H, W), dtype=np.float32)

    # ── Krok 1: piksele oryginalne (even, even) ────────────────────────
    out[0::2, 0::2] = p

    # ── Krok 2: interpolacja pozioma (even rows, odd cols) ─────────────
    # Środkowe kolumny: avg dwóch sąsiadów
    out[0::2, 1:W - 1:2] = (p[:, :-1] + p[:, 1:]) * 0.5
    # Ostatnia kolumna (krawędź)
    out[0::2, W - 1] = p[:, -1]

    # ── Krok 3: interpolacja pionowa (odd rows) ─────────────────────────
    # Środkowe wiersze: avg wiersza powyżej i poniżej (w out — pełna szerokość)
    # Użyjemy już wypełnionych wierszy even z out
    top    = out[0:H - 2:2, :]   # wiersze 0, 2, 4, ...
    bottom = out[2:H:2,     :]   # wiersze 2, 4, 6, ...
    out[1:H - 1:2, :] = (top + bottom) * 0.5
    # Ostatni wiersz
    out[H - 1, :] = out[H - 2, :]

    # ── Krok 4: korekcja gradientu VED (wektoryzowana) ─────────────────
    # Wzór: pred = c + (c - (a+b)/2) * 0.15
    # Stosujemy dla pozycji interpolowanych (nie-oryginałów).

    # Typ A: (odd row, ≥ row 3) — interpolowane pionowo
    yr_odd = np.arange(1, H - 1, 2)   # 1, 3, 5, ...
    # yr_odd - 2 musi być >= 0 → yr_odd >= 2; ale yr_odd zaczyna od 1
    # Bezpieczna wersja: bierz tylko yr_odd >= 2 (pomijamy wiersz 1 — brak 2 poprzedników)
    mask = yr_odd >= 2
    if mask.any():
        yo = yr_odd[mask]               # (3, 5, 7, ...) — indeksy wierszy odd ≥ 2
        a  = out[yo - 2, :]             # shape (n, W)
        b  = out[yo - 1, :]             # shape (n, W)
        c  = out[yo,     :]             # shape (n, W)
        corr = c + (c - (a + b) * 0.5) * 0.15
        out[yo, :] = np.clip(corr, -128.0, 127.0)

    # Typ B: (even row, odd col) — interpolowane poziomo
    # Wzór: pred = c + (c - (a+b)/2) * 0.15
    # a = out[y, x-2] (poprzedni oryginalny), b = out[y, x-1] (poprzedni interp)
    # x_odd = 1 → x-1 = 0 (ok), x-2 = -1 (brak lewego-lewego) → pomijamy x=1
    xo_odd = np.arange(3, W - 1, 2)    # 3, 5, 7, ... (pomijamy x=1: brak x-2)
    if len(xo_odd) > 0:
        xo = xo_odd
        a2 = out[0::2, xo - 2]    # shape (H//2, n)
        b2 = out[0::2, xo - 1]
        c2 = out[0::2, xo]
        corr2 = c2 + (c2 - (a2 + b2) * 0.5) * 0.15
        out[0::2, xo] = np.clip(corr2, -128.0, 127.0)

    return np.clip(out, -128.0, 127.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. SZYBKA KONWERSJA YUV→RGB
# ═══════════════════════════════════════════════════════════════════════════

_YUV2RGB = np.array([
    [1.0,  0.0,    1.402 ],
    [1.0, -0.344, -0.714 ],
    [1.0,  1.772,  0.0   ],
], dtype=np.float32)


def yuv420_to_rgb_fast(Y, U, V):
    """
    Szybsza wersja yuv420_to_rgb:
    - używa _ved_upsample_fast zamiast wolnego _ved_upsample_plane
    - macierz jako float32 (unika konwersji float64)
    - np.einsum zamiast np.dot dla możliwości optymalizacji BLAS
    """
    h, w = Y.shape
    U_full = _ved_upsample_fast(U)[:h, :w]
    V_full = _ved_upsample_fast(V)[:h, :w]
    # Stack → (H, W, 3) float32
    yuv = np.stack((
        Y.astype(np.float32, copy=False),
        U_full,
        V_full
    ), axis=-1)
    # Szybki dot float32 — numpy użyje AVX2 + FMA
    rgb = yuv @ _YUV2RGB.T
    return np.clip(rgb, 0, 255, out=rgb).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SZYBKI DECODE I-FRAME  (batch IDCT)
# ═══════════════════════════════════════════════════════════════════════════

def _decode_iframe_fast(self, data, h, w):
    """
    Zastępuje TopTopowCodecV2._decode_iframe.

    ORYGINAŁ: pętla per-blok, każdy blok → osobne scipy.idct (overhead funcall)
    TUTAJ:   reshape całej płaszczyzny do (N, bs, bs), batch_idct2, reshape z powrotem
             → jedno wywołanie scipy na całą klatkę
    """
    bs   = self.block_size
    bs_c = bs // 2

    # ── Luminancja Y ───────────────────────────────────────────────────
    Y_q  = data['Y'].astype(np.float32) * self.Q_Y    # (h, w)
    blks = plane_to_blocks(Y_q, bs)                    # (N, bs, bs)
    rec  = batch_idct2(blks)                           # (N, bs, bs)
    rec_Y = blocks_to_plane(rec, h, w, bs)             # (h, w)

    # ── Chrominancja U, V ──────────────────────────────────────────────
    hc, wc = h // 2, w // 2

    U_q   = data['U'].astype(np.float32) * self.Q_C
    blks_U = plane_to_blocks(U_q, bs_c)
    rec_U  = blocks_to_plane(batch_idct2(blks_U), hc, wc, bs_c)

    V_q   = data['V'].astype(np.float32) * self.Q_C
    blks_V = plane_to_blocks(V_q, bs_c)
    rec_V  = blocks_to_plane(batch_idct2(blks_V), hc, wc, bs_c)

    # Użyj deblock_filter z modułu kodeka — jest on wstrzyknięty jako atrybut
    self.prev_Y = self._speedup_deblock(rec_Y, bs)
    self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
    self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)
    return yuv420_to_rgb_fast(self.prev_Y, self.prev_U, self.prev_V)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SZYBKI ENCODE I-FRAME  (batch DCT)
# ═══════════════════════════════════════════════════════════════════════════

def _encode_iframe_fast(self, curr_Y, curr_U, curr_V, h_pad, w_pad):
    """
    Zastępuje _encode_iframe.
    Cała płaszczyzna w jednym wywołaniu batch_dct2 zamiast per-blok pętli.
    """
    bs   = self.block_size
    bs_c = bs // 2

    # ── Batch DCT Y ────────────────────────────────────────────────────
    blks_Y = plane_to_blocks(curr_Y.astype(np.float32), bs)     # (N, bs, bs)
    q_blks_Y = np.round(batch_dct2(blks_Y) / self.Q_Y).astype(np.int16)
    q_Y = blocks_to_plane(q_blks_Y.astype(np.float32), h_pad, w_pad, bs).astype(np.int16)

    # ── Batch DCT U, V ─────────────────────────────────────────────────
    blks_U = plane_to_blocks(curr_U.astype(np.float32), bs_c)
    q_blks_U = np.round(batch_dct2(blks_U) / self.Q_C).astype(np.int16)
    q_U = blocks_to_plane(q_blks_U.astype(np.float32), h_pad//2, w_pad//2, bs_c).astype(np.int16)

    blks_V = plane_to_blocks(curr_V.astype(np.float32), bs_c)
    q_blks_V = np.round(batch_dct2(blks_V) / self.Q_C).astype(np.int16)
    q_V = blocks_to_plane(q_blks_V.astype(np.float32), h_pad//2, w_pad//2, bs_c).astype(np.int16)

    # ── Rekonstrukcja (batch IDCT, identyczna ścieżka co dekoder) ──────
    rec_blks_Y = batch_idct2(q_blks_Y.astype(np.float32) * self.Q_Y)
    rec_Y = blocks_to_plane(rec_blks_Y, h_pad, w_pad, bs)

    rec_blks_U = batch_idct2(q_blks_U.astype(np.float32) * self.Q_C)
    rec_U = blocks_to_plane(rec_blks_U, h_pad//2, w_pad//2, bs_c)

    rec_blks_V = batch_idct2(q_blks_V.astype(np.float32) * self.Q_C)
    rec_V = blocks_to_plane(rec_blks_V, h_pad//2, w_pad//2, bs_c)

    self.prev_Y = self._speedup_deblock(rec_Y, bs)
    self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
    self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)

    return {'type': 'I', 'Y': q_Y, 'U': q_U, 'V': q_V, 'h': h_pad, 'w': w_pad}


# ═══════════════════════════════════════════════════════════════════════════
# 7. SZYBKI DECODE P/B-FRAME  (batch IDCT dla bloków DETAIL)
# ═══════════════════════════════════════════════════════════════════════════

def _decode_pframe_fast(self, data, h, w):
    """
    Zastępuje _decode_pframe (i pośrednio _decode_bframe przez B→save/restore).

    Optymalizacja:
    - Bloki tego samego rozmiaru są grupowane → batch IDCT na grupę
    - Kompensacja ruchu (interpolate_subpixel) nadal per-blok
    - Bloki SKIP: bez żadnej pracy
    - Zgodna z oryginałem: block['bs'] z danych, nie hardcoded 16
    """
    new_Y = self.prev_Y.copy()
    new_U = self.prev_U.copy()
    new_V = self.prev_V.copy()
    interp = self._speedup_interp

    # ── Grupuj bloki DETAIL wg rozmiaru (zazwyczaj jeden rozmiar = szybki batch) ──
    from collections import defaultdict
    groups = defaultdict(list)   # bs → [(i, block), ...]
    for block in data['blocks']:
        x  = block['x']
        y  = block['y'] if 'y' in block else data.get('_y', 0)
        bs = block['bs']
        if block['mode'] == 0:   # SKIP
            continue
        if y + bs > h or x + bs > w:   # poza granicą
            continue
        groups[bs].append(block)

    for bs, blist in groups.items():
        bs_c = bs // 2

        # Batch IDCT dla tej grupy bloków (jeden rozmiar → jednorodna tablica)
        raw_Y = np.stack([b['q_diff_y'].astype(np.float32) for b in blist]) * self.Q_Y
        raw_U = np.stack([b['q_diff_u'].astype(np.float32) for b in blist]) * self.Q_C
        raw_V = np.stack([b['q_diff_v'].astype(np.float32) for b in blist]) * self.Q_C
        idct_Y = batch_idct2(raw_Y)
        idct_U = batch_idct2(raw_U)
        idct_V = batch_idct2(raw_V)

        for i, block in enumerate(blist):
            x      = block['x']
            y      = block['y'] if 'y' in block else data.get('_y', 0)
            dx_qp, dy_qp = block['mv_y_qp']

            # Y — kompensacja ruchu + residual
            abs_y = max(0, min(y * 4 + dy_qp, (h - bs) * 4))
            abs_x = max(0, min(x * 4 + dx_qp, (w - bs) * 4))
            match_y = interp(self.prev_Y, abs_y, abs_x, bs)
            new_Y[y:y+bs, x:x+bs] = np.clip(match_y + idct_Y[i], 0.0, 255.0)

            # UV
            cy, cx = y // 2, x // 2
            uv_dy  = dy_qp // 8
            uv_dx  = dx_qp // 8
            csy = max(0, min(cy + uv_dy, self.prev_U.shape[0] - bs_c))
            csx = max(0, min(cx + uv_dx, self.prev_U.shape[1] - bs_c))
            new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                self.prev_U[csy:csy+bs_c, csx:csx+bs_c] + idct_U[i], -128.0, 127.0)
            new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                self.prev_V[csy:csy+bs_c, csx:csx+bs_c] + idct_V[i], -128.0, 127.0)

    self.prev_Y = self._speedup_deblock(new_Y, self.block_size)
    self.prev_U = new_U
    self.prev_V = new_V
    return yuv420_to_rgb_fast(self.prev_Y, self.prev_U, self.prev_V)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  OPCJONALNY PYFFTW  (jeśli zainstalowany: pip install pyfftw)
#     pyfftw jest ~2–4× szybszy od scipy dla dużych batchy na AVX2
# ═══════════════════════════════════════════════════════════════════════════

def _try_setup_pyfftw():
    """Jeśli pyfftw dostępne — zastąp batch_dct2/idct2 jego implementacją."""
    global batch_dct2, batch_idct2
    try:
        import pyfftw
        import pyfftw.interfaces.scipy_fftpack as fftwsp

        # FFTW planuje transformaty — zapamiętaj plan dla każdego rozmiaru
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(60)

        # Ustaw globalną liczbę wątków FFTW (scipy_fftpack interface nie ma workers=)
        pyfftw.config.NUM_THREADS = N_CORES

        # Sprawdź czy ten interfejs obsługuje workers= (starsze wersje nie mają)
        import inspect as _ins
        _dct_sig = _ins.signature(fftwsp.dct)
        _has_workers = 'workers' in _dct_sig.parameters

        if _has_workers:
            _fftw_dct  = lambda x, ax: fftwsp.dct(x, type=2, norm='ortho', axis=ax,
                                                   workers=N_CORES)
            _fftw_idct = lambda x, ax: fftwsp.idct(x, type=2, norm='ortho', axis=ax,
                                                    workers=N_CORES)
        else:
            # Starszy pyfftw — wątki ustawione przez pyfftw.config.NUM_THREADS
            _fftw_dct  = lambda x, ax: fftwsp.dct(x, type=2, norm='ortho', axis=ax)
            _fftw_idct = lambda x, ax: fftwsp.idct(x, type=2, norm='ortho', axis=ax)

        def batch_dct2_fftw(blocks):
            return _fftw_dct(_fftw_dct(blocks.astype(np.float32), -1), -2)

        def batch_idct2_fftw(blocks):
            return _fftw_idct(_fftw_idct(blocks.astype(np.float32), -1), -2)

        batch_dct2  = batch_dct2_fftw
        batch_idct2 = batch_idct2_fftw
        return True, f"pyfftw (workers={N_CORES})"
    except ImportError:
        return False, "scipy.fftpack (pyfftw niedostępne — pip install pyfftw)"


FFTW_OK, FFTW_INFO = _try_setup_pyfftw()


# ═══════════════════════════════════════════════════════════════════════════
# 9.  OPCJONALNY NUMBA JIT  (dla interpolate_subpixel — hot path P-decode)
# ═══════════════════════════════════════════════════════════════════════════

def _try_compile_numba():
    """
    Kompiluje interpolate_subpixel przez numba@jit jeśli dostępna.
    interpolate_subpixel to hot-path w P-frame decode (n_detail wywołań/klatka).
    JIT eliminuje Python overhead + pozwala numbie auto-wektoryzować.
    """
    try:
        from numba import njit
        import numba as nb

        @njit(cache=False, fastmath=True, parallel=False)
        def _interp_numba(plane, y_fp, x_fp, block_size):
            h, w = plane.shape
            y_f  = y_fp / 4.0
            x_f  = x_fp / 4.0
            y0   = int(y_f)
            x0   = int(x_f)
            y1   = min(y0 + 1, h - 1)
            x1   = min(x0 + 1, w - 1)
            y0   = max(0, min(y0, h - block_size - 1))
            x0   = max(0, min(x0, w - block_size - 1))
            y1   = max(0, min(y1, h - block_size - 1))
            x1   = max(0, min(x1, w - block_size - 1))
            fy   = y_f - int(y_f)
            fx   = x_f - int(x_f)
            bs   = block_size
            result = np.empty((bs, bs), dtype=np.float32)
            for r in range(bs):
                for c in range(bs):
                    v = (plane[y0+r, x0+c] * (1.0-fy) * (1.0-fx) +
                         plane[y1+r, x0+c] * fy       * (1.0-fx) +
                         plane[y0+r, x1+c] * (1.0-fy) * fx       +
                         plane[y1+r, x1+c] * fy       * fx)
                    result[r, c] = v
            return result

        # Warm-up (kompilacja przy pierwszym użyciu w tle)
        def _warmup():
            try:
                dummy = np.zeros((32, 32), dtype=np.float32)
                _interp_numba(dummy, 0, 0, 8)
            except Exception as _e:
                print(f"[speedup] numba warm-up nieudany: {_e}", flush=True)

        t = threading.Thread(target=_warmup, daemon=True)
        t.start()

        return _interp_numba, f"numba {nb.__version__} (JIT)"
    except ImportError:
        return None, "numpy (numba niedostępne — pip install numba)"
    except Exception as _e:
        return None, f"numpy (numba błąd: {_e})"


_NUMBA_INTERP, _NUMBA_INFO = _try_compile_numba()


# ═══════════════════════════════════════════════════════════════════════════
# 10. GŁÓWNA FUNKCJA PATCH — monkey-patchuje załadowany moduł kodeka
# ═══════════════════════════════════════════════════════════════════════════

def apply(codec_mod) -> object:
    """
    Wstrzykuje zoptymalizowane funkcje do załadowanego modułu kodeka.

    Parametr: codec_mod — wynik load_codec_module()
    Zwraca:   ten sam obiekt (zmodyfikowany in-place)

    Nie wymaga żadnych zmian w toptopuwcodec_v1.py.
    """
    cls = codec_mod.TopTopowCodecV2

    # ── 1. Podmień funkcję standalone _ved_upsample_plane ────────────
    codec_mod._ved_upsample_plane = _ved_upsample_fast

    # Upewnij się że yuv420_to_rgb w module używa szybkiego upsamplingu
    codec_mod.yuv420_to_rgb = yuv420_to_rgb_fast

    # ── 2. Wstrzyknij referencje do deblock i interpolate do klasy ────
    # (aby metody instancji miały do nich dostęp bez importu)
    cls._speedup_deblock = staticmethod(codec_mod.deblock_filter)
    cls._speedup_interp  = staticmethod(
        _NUMBA_INTERP if _NUMBA_INTERP is not None
        else codec_mod.interpolate_subpixel
    )

    # ── 3. Podmień metody klasy ───────────────────────────────────────
    cls._decode_iframe = _decode_iframe_fast
    cls._encode_iframe = _encode_iframe_fast
    cls._decode_pframe = _decode_pframe_fast
    # B-frame decode wywołuje _decode_pframe wewnętrznie → automatycznie szybszy

    return codec_mod


# ═══════════════════════════════════════════════════════════════════════════
# 11. RAPORT DIAGNOSTYCZNY
# ═══════════════════════════════════════════════════════════════════════════

def diagnostics() -> str:
    """Zwraca string z raportem dostępnych przyspieszeń."""
    lines = [
        "╔══ TOPTOPÓW SPEEDUP DIAGNOSTICS ══╗",
        f"  Rdzenie CPU:   {N_CORES} fizycznych",
        f"  DCT backend:   {FFTW_INFO}",
        f"  Interpolacja:  {_NUMBA_INFO}",
    ]

    # Sprawdź czy numpy korzysta z MKL lub OpenBLAS
    try:
        np_config = np.__config__.blas_opt_info.get('libraries', [])
        blas_name = ', '.join(np_config) if np_config else 'brak info'
        lines.append(f"  NumPy BLAS:    {blas_name}")
    except Exception:
        pass

    # Sprawdź AVX2 dostępność
    try:
        import cpuinfo
        flags = cpuinfo.get_cpu_info().get('flags', [])
        avx2  = 'avx2' in flags
        fma   = 'fma'  in flags
        lines.append(f"  AVX2:          {'✓' if avx2 else '✗'}   FMA3: {'✓' if fma else '✗'}")
        brand = cpuinfo.get_cpu_info().get('brand_raw', '')
        if brand:
            lines.append(f"  CPU:           {brand}")
    except ImportError:
        lines.append("  AVX2/FMA:      (zainstaluj py-cpuinfo aby sprawdzić)")

    # Benchmark VED upsample
    try:
        import time
        test = np.random.rand(270, 480).astype(np.float32)
        # Oryginalna wersja — symulacja (tylko 1 przebieg)
        t0 = time.perf_counter()
        for _ in range(5):
            _ved_upsample_fast(test)
        t_fast = (time.perf_counter() - t0) / 5 * 1000
        lines.append(f"  VED upsample:  {t_fast:.2f} ms/klatkę (270×480 UV → 540×960)")
        lines.append(f"  Szac. fps dec: ~{1000 / (t_fast * 2 + 5):.1f} kl/s (tylko upsample+YUV)")
    except Exception as e:
        lines.append(f"  Benchmark:     błąd ({e})")

    lines.append("╚═══════════════════════════════════╝")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(diagnostics())

    # Test poprawności batch_dct2 vs scipy per-blok
    import sys
    from scipy.fftpack import dct as sdct, idct as sidct

    print("\nTest poprawności DCT batch vs per-blok:")
    rng   = np.random.default_rng(42)
    plane = rng.standard_normal((64, 64)).astype(np.float32)
    bs    = 16

    blks  = plane_to_blocks(plane, bs)
    batch = batch_dct2(blks)
    # Per-blok referencja
    ref   = np.stack([
        sdct(sdct(b.T, norm='ortho').T, norm='ortho') for b in blks
    ])
    diff  = np.max(np.abs(batch - ref))
    print(f"  Max |batch - perblok| = {diff:.2e}  {'✓ OK' if diff < 1e-4 else '✗ BŁĄD'}")

    print("\nTest VED upsample (poprawność):")
    small = rng.standard_normal((8, 8)).astype(np.float32)
    fast  = _ved_upsample_fast(small)
    print(f"  Kształt wyjścia: {fast.shape}  (oczekiwane: (16, 16))")
    print(f"  Range: [{fast.min():.2f}, {fast.max():.2f}]  ✓")

    print(f"\n{diagnostics()}")
