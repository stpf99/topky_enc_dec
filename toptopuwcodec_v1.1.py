import numpy as np
from scipy.fftpack import dct, idct
import zstandard as zstd
import struct
import os
import imageio.v3 as iio
import time
import argparse
import concurrent.futures
from collections import deque

# =============================================================================
# KODEK TOP TOPÓW v2.6 — B-frame backward-only + uproszczona architektura
#
# FIX-5 (v2.6): Predykcja fwd/bi w B-klatkach była niemożliwa do poprawnego
#   zaimplementowania bez pełnego DPB (Decoded Picture Buffer).
#   Rozwiązanie: B-klatki używają WYŁĄCZNIE prev_Y (backward prediction).
#   future_Y używane TYLKO do dual-check SKIP (decyzja binarna SKIP/DETAIL)
#   — surowa klatka wystarczy, bo mismatch przy SKIP nie powoduje artefaktów.
#   Format B-klatek = identyczny z P-klatkami → jeden serializer/deserializer.
#   Efekt: zero artefaktów "drabinki", prostszy kod, ~140KB zamiast 175KB.
#
# FIX-4 (v2.5): Encoder-decoder reference loop (chwilowo zastąpiony przez v2.6)
# OPT-A/B (v2.4): Prescan MAD + Patch Delta Format
# FIX-3 (v2.3): Two-pass B-frame decode
# FIX-1/2 (v2.2): Brak 'y' w blokach P; błędny fallback decode_bframe
# OPT (v2.2): Q_Y 22, Q_C 40, bframe_interval 3, Zstd lvl 22
# VED (v2.1): Chroma upsampler, edge-aware deblock, spatial SKIP predictor
# =============================================================================
#
# FIX-4 (v2.5): GŁÓWNA PRZYCZYNA "drabinki" na B-klatkach:
#   Enkoder używał surowej (raw) klatki przyszłej jako future_Y.
#   Dekoder odtwarza klatkę przyszłą ZE STRATAMI kwantyzacji.
#   Różnica referencji → bloki B-frame trafiały w złe miejsca → artefakty.
#
#   Naprawa: encoder-decoder reference loop:
#     1. Zapisz stan kodeka (prev_Y/U/V)
#     2. Zakoduj i ZREKONSTRUUJ klatkę przyszłą (P lookahead — ta sama
#        ścieżka DCT→kwantyzacja→odwrotna co dekoder)
#     3. Użyj zrekonstruowanej wersji jako future_Y
#     4. Przywróć stan → zakoduj B-frame
#   Teraz encoder reference == decoder reference → czyste B-klatki.
#
# OPT-A/B (v2.4): Prescan MAD + Patch Delta Format
# FIX-3 (v2.3): Two-pass B-frame decode
# FIX-1/2 (v2.2): Brak 'y' w blokach P; błędny fallback decode_bframe
# OPT (v2.2): Q_Y 22, Q_C 40, bframe_interval 3, Zstd lvl 22
# VED (v2.1): Chroma upsampler, edge-aware deblock, spatial SKIP predictor
# =============================================================================
#
# NOWE w v2.4:
#   OPT-A: Prescan MAD Map — przed pętlą bloków P-frame, oblicz mapę różnic
#           całej klatki JEDNĄ operacją numpy (reshape + mean). Bloki z MAD < 2.5
#           są natychmiastowo SKIP bez liczenia SAD per-blok. Oszczędność: ~20%
#           czasu kodowania dla scen ze statycznym tłem.
#   OPT-B: Patch Delta Format — SKIP bloki nie trafiają do strumienia binarnego.
#           Stary format: N_bloków * (1B mode + 5B koord) = 6B per SKIP.
#           Nowy format: tylko N_DETAIL bloków z koordynatami.
#           Zysk: ~6B * N_SKIP. Przy 75% SKIP i 3200 blokach → ~14KB mniej (pre-Zstd).
#
# FIX-3 (v2.3): Two-pass B-frame decode — future_Y poprawnie ustawiane
# FIX-1/2 (v2.2): brak 'y' w blokach P; błędny fallback decode_bframe
# OPT (v2.2): Q_Y 22, Q_C 40, bframe_interval 3, Zstd lvl 22
# VED (v2.1): Chroma upsampler, edge-aware deblock, spatial SKIP predictor
# =============================================================================
# =============================================================================
# Nowe w v2.1 (relative do v2):
#   A. VED Chroma Upsampler — zastępuje np.repeat przy dekodowaniu UV
#      Predyktor: d = c + (c - (a+b)/2) * 0.15 (gradient chrominancji)
#      Efekt: lepsza jakość kolorów bez zmiany strumienia bitowego
#
#   B. Edge-Aware Deblock Filter — maska Sobela chroni prawdziwe krawędzie
#      Wygładza tylko granice bloków gdzie NIE MA ostrych detali
#      Efekt: mniej "mydlanego" obrazu, lepsze krawędzie
#
#   C. VED Spatial SKIP — przestrzenna predykcja sąsiadów dla bloków P
#      Jeśli blok ≈ predykcja z lewego+górnego sąsiada → SKIP bez MV
#      Efekt: 5-12% więcej bloków SKIP → mniejszy rozmiar pliku
#
# Nowe w v2 (oryginalne):
#   1. Wielowątkowość (ProcessPoolExecutor)
#   2. Sub-pikselowy ruch (1/4 piksela)
#   3. Klatki B (B-Frames)
#   4. Adaptacyjne rozmiary bloków (quad-tree: 8/16/32)
#   5. Osobna kwantyzacja Q_Y i Q_C
#   6. Estymacja ruchu UV
# =============================================================================

# ---------------------------------------------------------------------------
# 1. MATEMATYKA KOLORÓW (4:2:0)
# ---------------------------------------------------------------------------
def rgb_to_yuv420(rgb):
    m = np.array([[ 0.299,  0.587,  0.114],
                  [-0.169, -0.331,  0.500],
                  [ 0.500, -0.419, -0.081]])
    yuv = np.dot(rgb, m.T)
    Y = yuv[:, :, 0]
    U = yuv[::2, ::2, 1]
    V = yuv[::2, ::2, 2]
    return Y, U, V

def _ved_upsample_plane(plane):
    """
    Upsample 2x używając predyktora VED: d = c + (c - (a+b)/2).
    Lepszy od np.repeat — zachowuje gradienty chrominancji.
    Działa na jednym kanale UV (float32).
    """
    h, w = plane.shape
    # Najpierw pikselowe podwojenie (nearest-neighbor jako punkt startowy)
    big = np.zeros((h * 2, w * 2), dtype=np.float32)
    big[0::2, 0::2] = plane  # Oryginalne próbki

    # Interpolacja pozioma: wypełnij piksele pomiędzy (oś X)
    for x in range(0, w * 2 - 2, 2):
        big[0::2, x + 1] = (big[0::2, x] + big[0::2, x + 2]) / 2.0

    # Interpolacja pionowa: wypełnij piksele pomiędzy (oś Y)
    for y in range(0, h * 2 - 2, 2):
        big[y + 1, :] = (big[y, :] + big[y + 2, :]) / 2.0

    # VED predykcja gradientu — koryguj interpolowane piksele
    # d = c + (c - (a+b)/2)  — lokalny pęd zmiany
    for y in range(2, h * 2 - 1):
        for x in range(2, w * 2 - 1):
            if (y % 2 == 1) or (x % 2 == 1):  # Tylko interpolowane pozycje
                a = big[y - 2, x]
                b = big[y - 1, x] if x % 2 == 0 else big[y, x - 1]
                c = big[y, x]
                # Korekta pędem — ale ograniczona (nie overshooting)
                pred = c + (c - (a + b) / 2.0) * 0.15
                big[y, x] = np.clip(pred, -128.0, 127.0)

    return big


def yuv420_to_rgb(Y, U, V):
    h, w = Y.shape
    # VED upsample zamiast np.repeat — lepsze gradienty chrominancji
    U_full = _ved_upsample_plane(U)[:h, :w]
    V_full = _ved_upsample_plane(V)[:h, :w]
    yuv = np.stack((Y, U_full, V_full), axis=-1)
    m = np.array([[ 1.0,  0.0,  1.402],
                  [ 1.0, -0.344, -0.714],
                  [ 1.0,  1.772,  0.0]])
    return np.clip(np.dot(yuv, m.T), 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# 2. TRANSFORMATY DCT
# ---------------------------------------------------------------------------
def apply_dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ---------------------------------------------------------------------------
# 3. INTERPOLACJA SUB-PIKSELOWA (1/4 piksela — bilinearna)
# ---------------------------------------------------------------------------
def interpolate_subpixel(plane, y_fp, x_fp, block_size):
    """
    Pobiera blok z pozycji ułamkowej (quarter-pixel).
    y_fp, x_fp to pozycje w jednostkach 1/4 piksela (int).
    """
    h, w = plane.shape
    # Konwertujemy z quarter-pixel na float
    y_f = y_fp / 4.0
    x_f = x_fp / 4.0

    y0 = int(np.floor(y_f))
    x0 = int(np.floor(x_f))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)

    # Clamp
    y0 = max(0, min(y0, h - block_size - 1))
    x0 = max(0, min(x0, w - block_size - 1))
    y1 = max(0, min(y1, h - block_size - 1))
    x1 = max(0, min(x1, w - block_size - 1))

    fy = y_f - np.floor(y_f)
    fx = x_f - np.floor(x_f)

    # Bilinearna interpolacja czterech sąsiednich bloków
    b00 = plane[y0:y0+block_size, x0:x0+block_size].astype(np.float32)
    b10 = plane[y1:y1+block_size, x0:x0+block_size].astype(np.float32)
    b01 = plane[y0:y0+block_size, x1:x1+block_size].astype(np.float32)
    b11 = plane[y1:y1+block_size, x1:x1+block_size].astype(np.float32)

    result = (b00 * (1-fy) * (1-fx) +
              b10 * fy     * (1-fx) +
              b01 * (1-fy) * fx     +
              b11 * fy     * fx)
    return result

# ---------------------------------------------------------------------------
# 4. FILTR DEBLOKUJĄCY
# ---------------------------------------------------------------------------
def deblock_filter(plane, block_size=16, threshold=15):
    """
    Prosty deterministyczny deblocking — czysty numpy, zero dryftu enkoder/dekoder.
    Wygladza tylko granice blokow gdzie roznica sasiadow < threshold.
    Bez Sobela (ktory byl wolny i wprowadzal niezerowe roznice float32).
    """
    h, w = plane.shape
    filtered = plane.astype(np.float32)  # bez copy — bedziemy modyfikowac in-place

    # Pionowe granice blokow (wygladz poziomo)
    for x in range(block_size, w, block_size):
        left  = filtered[:, x-1]
        right = filtered[:, x]
        diff  = np.abs(left - right)
        mask  = diff < threshold
        avg   = (left + right) * 0.5
        filtered[:, x-1] = np.where(mask, avg, left)
        filtered[:, x]   = np.where(mask, avg, right)

    # Poziome granice blokow (wygladz pionowo)
    for y in range(block_size, h, block_size):
        top    = filtered[y-1, :]
        bottom = filtered[y,   :]
        diff   = np.abs(top - bottom)
        mask   = diff < threshold
        avg    = (top + bottom) * 0.5
        filtered[y-1, :] = np.where(mask, avg, top)
        filtered[y,   :] = np.where(mask, avg, bottom)

    return np.clip(filtered, 0, 255).astype(np.float32)

# ---------------------------------------------------------------------------
# 5. ADAPTACYJNY ROZMIAR BLOKU (quad-tree: 32→16→8)
# ---------------------------------------------------------------------------
def choose_block_size(plane_region):
    """
    Decyduje o rozmiarze bloku na podstawie wariancji regionu 32x32.
    Zwraca: 32 (płaski), 16 (średni detal), 8 (wysoki detal).
    """
    var = np.var(plane_region)
    if var < 15:
        return 32   # Płaskie tło — duży blok
    elif var < 80:
        return 16   # Normalny detal
    else:
        return 8    # Wysoki detal / krawędzie

# ---------------------------------------------------------------------------
# 6a. PRESCAN MAD — szybka mapa zmian całej klatki (numpy, bez pętli)
# ---------------------------------------------------------------------------
def compute_mad_map(curr_Y, prev_Y, block_size):
    """
    Oblicza MAD (Mean Absolute Difference) per blok dla całej klatki naraz.
    Bez żadnych pętli Python — czyste numpy reshaping + mean.
    Zwraca tablicę 2D float32: mad_map[row_idx][col_idx] = MAD bloku.
    """
    h, w = curr_Y.shape
    bs = block_size
    # Przytnij do siatki
    rows = h // bs
    cols = w // bs
    c = curr_Y[:rows*bs, :cols*bs].astype(np.float32)
    p = prev_Y[:rows*bs, :cols*bs].astype(np.float32)
    diff = np.abs(c - p)
    # Reshape: (rows, bs, cols, bs) → mean po osiach 1,3
    mad = diff.reshape(rows, bs, cols, bs).mean(axis=(1, 3))
    return mad  # shape: (rows, cols)

# ---------------------------------------------------------------------------
# 6b. FUNKCJA PRZETWARZAJĄCA JEDEN RZĄD BLOKÓW (dla wielowątkowości)
# ---------------------------------------------------------------------------
def process_row_pframe(args):
    """
    Przetwarza jeden rzad makroblokow klatki P z multi-reference DPB.
    dpb_refs: lista {'Y','U','V'} — wszystkie aktywne referencje.
    Kazdy blok wybiera referencje z najmniejszym SAD (MRF).
    """
    (y, w_pad, h_pad,
     dpb_refs, curr_Y,
     curr_U, curr_V,
     search_range, Q_Y, Q_C,
     use_subpixel, mad_row) = args

    # Dla kompatybilnosci: prev_Y/U/V = najnowsza referencja (ostatnia w DPB)
    prev_Y = dpb_refs[-1]['Y']
    prev_U = dpb_refs[-1]['U']
    prev_V = dpb_refs[-1]['V']

    BS = 16  # Staly rozmiar bloku (v2.7 wymaga bs=16, bez adaptacyjnego)
    bs = BS
    row_results = []
    col_idx = 0

    x = 0
    while x < w_pad:
        # --- PRESCAN FAST-SKIP: jezeli MAD bloku < prog -> od razu SKIP (bez SAD) ---
        if mad_row is not None and col_idx < len(mad_row):
            if mad_row[col_idx] < 1.0:  # tylko pikselowo identyczne bloki
                row_results.append({'x': x, 'bs': bs, 'mode': 0})
                x += bs
                col_idx += 1
                continue

        col_idx += 1
        curr_y_block = curr_Y[y:y+bs, x:x+bs]

        # --- Szybkie sprawdzenie czy SKIP (SAD vs poprzednia klatka) ---
        prev_y_static = prev_Y[y:y+bs, x:x+bs]
        sad_static = np.sum(np.abs(curr_y_block.astype(np.float32) -
                                    prev_y_static.astype(np.float32)))

        if sad_static < (bs * bs * 2):  # Surowy prog — tylko naprawde statyczne bloki
            row_results.append({
                'x': x, 'bs': bs, 'mode': 0,  # SKIP
            })
            x += bs
            continue

        # VED Spatial SKIP usunieto — powodowal ghosting przez fałszywe SKIPy

        # --- Estymacja ruchu Y (full-pixel) ---
        min_sad = float('inf')
        best_dx_qp, best_dy_qp = 0, 0  # quarter-pixel units

        # Three-Step Search: O(log n) zamiast O(n^2) — przy search_range=24
        # sprawdza ~25 kandydatow zamiast 2401, jakosc prawie identyczna.
        step = max(1, search_range // 2)
        cy_tss, cx_tss = y, x  # aktualny srodek przeszukiwania
        while step >= 1:
            candidates = []
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    sy = max(0, min(cy_tss + dy, h_pad - bs))
                    sx = max(0, min(cx_tss + dx, w_pad - bs))
                    candidates.append((sy, sx))
            for sy, sx in candidates:
                cand = prev_Y[sy:sy+bs, sx:sx+bs]
                sad = np.sum(np.abs(curr_y_block.astype(np.float32) -
                                     cand.astype(np.float32)))
                if sad < min_sad:
                    min_sad = sad
                    best_dx_qp = (sx - x) * 4
                    best_dy_qp = (sy - y) * 4
                    cy_tss, cx_tss = sy, sx
            step //= 2

        # --- Sub-pikselowe doprecyzowanie (±1 quarter-pixel wokół best) ---
        if use_subpixel and min_sad > 0:
            for qdy in range(-3, 4):
                for qdx in range(-3, 4):
                    if qdx == 0 and qdy == 0:
                        continue
                    try_dy = best_dy_qp + qdy
                    try_dx = best_dx_qp + qdx
                    # Pozycja absolutna w quarter-pixels
                    abs_y_qp = y * 4 + try_dy
                    abs_x_qp = x * 4 + try_dx
                    # Clamp
                    abs_y_qp = max(0, min(abs_y_qp, (h_pad - bs) * 4))
                    abs_x_qp = max(0, min(abs_x_qp, (w_pad - bs) * 4))
                    cand_sub = interpolate_subpixel(prev_Y, abs_y_qp, abs_x_qp, bs)
                    sad = np.sum(np.abs(curr_y_block.astype(np.float32) - cand_sub))
                    if sad < min_sad:
                        min_sad = sad
                        best_dy_qp = try_dy
                        best_dx_qp = try_dx

        # --- MRF: wybierz najlepsza referencje z DPB dla tego bloku ---
        best_ref_idx = len(dpb_refs) - 1  # domyslnie najnowsza
        best_ref_sad = min_sad
        best_ref_dx_qp, best_ref_dy_qp = best_dx_qp, best_dy_qp

        for ref_idx, ref in enumerate(dpb_refs[:-1]):  # sprawdz starsze referencje
            ref_Y = ref['Y']
            r_min_sad = float('inf')
            r_best_dx, r_best_dy = 0, 0
            r_cy, r_cx = y, x
            r_step = max(1, search_range // 2)
            while r_step >= 1:
                for dy in (-r_step, 0, r_step):
                    for dx in (-r_step, 0, r_step):
                        sy = max(0, min(r_cy + dy, h_pad - bs))
                        sx = max(0, min(r_cx + dx, h_pad - bs))
                        sx = max(0, min(r_cx + dx, w_pad - bs))
                        if sy + bs > ref_Y.shape[0] or sx + bs > ref_Y.shape[1]:
                            continue
                        cand = ref_Y[sy:sy+bs, sx:sx+bs]
                        sad = np.sum(np.abs(curr_y_block.astype(np.float32) - cand.astype(np.float32)))
                        if sad < r_min_sad:
                            r_min_sad = sad
                            r_best_dx = (sx - x) * 4
                            r_best_dy = (sy - y) * 4
                            r_cy, r_cx = sy, sx
                r_step //= 2
            if r_min_sad < best_ref_sad:
                best_ref_sad = r_min_sad
                best_ref_idx = ref_idx
                best_ref_dx_qp, best_ref_dy_qp = r_best_dx, r_best_dy

        # Uzyj najlepszej referencji
        use_ref = dpb_refs[best_ref_idx]
        use_prev_Y = use_ref['Y']
        use_prev_U = use_ref['U']
        use_prev_V = use_ref['V']
        best_dx_qp, best_dy_qp = best_ref_dx_qp, best_ref_dy_qp

        # --- Pobierz dopasowany blok Y z sub-pikselową precyzją ---
        abs_y_qp = y * 4 + best_dy_qp
        abs_x_qp = x * 4 + best_dx_qp
        abs_y_qp = max(0, min(abs_y_qp, (h_pad - bs) * 4))
        abs_x_qp = max(0, min(abs_x_qp, (w_pad - bs) * 4))
        match_y = interpolate_subpixel(use_prev_Y, abs_y_qp, abs_x_qp, bs)

        # --- DCT + Kwantyzacja resztek Y ---
        diff_y = curr_y_block.astype(np.float32) - match_y
        q_diff_y = np.round(apply_dct2(diff_y) / Q_Y).astype(np.int16)

        # --- Estymacja ruchu UV ---
        cy, cx = y // 2, x // 2
        bs_c = bs // 2
        uv_dy = best_dy_qp // 8
        uv_dx = best_dx_qp // 8
        csy_c = max(0, min(cy + uv_dy, use_prev_U.shape[0] - bs_c))
        csx_c = max(0, min(cx + uv_dx, use_prev_U.shape[1] - bs_c))
        match_u = use_prev_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        match_v = use_prev_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c].astype(np.float32)
        curr_u_block = curr_U[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)
        curr_v_block = curr_V[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32)

        q_diff_u = np.round(apply_dct2(curr_u_block - match_u) / Q_C).astype(np.int16)
        q_diff_v = np.round(apply_dct2(curr_v_block - match_v) / Q_C).astype(np.int16)

        row_results.append({
            'x': x, 'bs': bs, 'mode': 1,
            'ref_idx': best_ref_idx,   # ktora referencja z DPB
            'mv_y_qp': (int(best_dx_qp), int(best_dy_qp)),
            'q_diff_y': q_diff_y,
            'q_diff_u': q_diff_u,
            'q_diff_v': q_diff_v,
        })
        x += bs

    return y, row_results


# ---------------------------------------------------------------------------
# 7. GŁÓWNA KLASA KODEKA
# ---------------------------------------------------------------------------
class TopTopowCodecV2:
    def __init__(self, block_size=16, search_range=24, use_subpixel=True,
                 use_bframes=True, bframe_interval=3,
                 q_y=22.0, q_c=40.0, adaptive_q=False):
        self.block_size    = block_size
        self.search_range  = search_range
        self.use_subpixel  = use_subpixel
        self.use_bframes   = use_bframes
        self.bframe_interval = bframe_interval
        self.adaptive_q    = adaptive_q

        self.Q_Y = q_y
        self.Q_C = q_c
        self.Q_Y_base = q_y  # bazowa wartosc do adaptacji
        self.Q_C_base = q_c

        self.prev_Y = self.prev_U = self.prev_V = None
        self.future_Y = self.future_U = self.future_V = None  # Do klatek B

        # DPB — Decoded Picture Buffer: lista aktywnych referencji
        # Kazdy wpis: {'Y': ndarray, 'U': ndarray, 'V': ndarray, 'type': 'I'/'P'}
        # Rozmiar: az do I-frame (reset), max 8 wpisow
        self.dpb = []          # lista referencji w kolejnosci chronologicznej
        self.dpb_max = 8       # max rozmiar DPB

    # -----------------------------------------------------------------------
    # DPB — zarzadzanie buforem referencji
    # -----------------------------------------------------------------------
    def _dpb_push(self, Y, U, V, frame_type):
        """Dodaj zrekonstruowana klatke do DPB. I-frame resetuje bufor."""
        if frame_type == 'I':
            self.dpb = []  # reset przy I-frame
        entry = {'Y': Y.copy(), 'U': U.copy(), 'V': V.copy(), 'type': frame_type}
        self.dpb.append(entry)
        if len(self.dpb) > self.dpb_max:
            self.dpb.pop(0)  # usun najstarsza referencje

    def _dpb_find_best(self, block_Y, y, x, bs):
        """
        Znajdz referencje z DPB dajaca najmniejszy SAD dla bloku.
        Zwraca (ref_idx, best_sad) gdzie ref_idx to indeks w self.dpb
        (0 = najstarsza, len-1 = najnowsza).
        """
        if not self.dpb:
            return 0, float('inf')
        best_idx = len(self.dpb) - 1  # domyslnie najnowsza (prev)
        best_sad = float('inf')
        block_f = block_Y.astype(np.float32)
        for idx, ref in enumerate(self.dpb):
            h_ref, w_ref = ref['Y'].shape
            if y + bs > h_ref or x + bs > w_ref:
                continue
            sad = np.sum(np.abs(block_f - ref['Y'][y:y+bs, x:x+bs]))
            if sad < best_sad:
                best_sad = sad
                best_idx = idx
        return best_idx, best_sad

    # -----------------------------------------------------------------------
    # KLATKA I (Kluczowa)
    # -----------------------------------------------------------------------
    def _encode_iframe(self, curr_Y, curr_U, curr_V, h_pad, w_pad):
        """Koduje klatkę I — każdy blok przez DCT+kwantyzację."""
        q_Y = np.zeros_like(curr_Y, dtype=np.int16)
        q_U = np.zeros_like(curr_U, dtype=np.int16)
        q_V = np.zeros_like(curr_V, dtype=np.int16)

        for y in range(0, h_pad, self.block_size):
            for x in range(0, w_pad, self.block_size):
                bs = self.block_size
                q_Y[y:y+bs, x:x+bs] = np.round(
                    apply_dct2(curr_Y[y:y+bs, x:x+bs]) / self.Q_Y)

        bs_c = self.block_size // 2
        for y in range(0, h_pad // 2, bs_c):
            for x in range(0, w_pad // 2, bs_c):
                q_U[y:y+bs_c, x:x+bs_c] = np.round(
                    apply_dct2(curr_U[y:y+bs_c, x:x+bs_c]) / self.Q_C)
                q_V[y:y+bs_c, x:x+bs_c] = np.round(
                    apply_dct2(curr_V[y:y+bs_c, x:x+bs_c]) / self.Q_C)

        # Rekonstrukcja (co widzi dekoder)
        rec_Y = np.zeros_like(curr_Y, dtype=np.float32)
        rec_U = np.zeros_like(curr_U, dtype=np.float32)
        rec_V = np.zeros_like(curr_V, dtype=np.float32)
        for y in range(0, h_pad, self.block_size):
            for x in range(0, w_pad, self.block_size):
                bs = self.block_size
                rec_Y[y:y+bs, x:x+bs] = apply_idct2(
                    q_Y[y:y+bs, x:x+bs].astype(np.float32) * self.Q_Y)
        for y in range(0, h_pad // 2, bs_c):
            for x in range(0, w_pad // 2, bs_c):
                rec_U[y:y+bs_c, x:x+bs_c] = apply_idct2(
                    q_U[y:y+bs_c, x:x+bs_c].astype(np.float32) * self.Q_C)
                rec_V[y:y+bs_c, x:x+bs_c] = apply_idct2(
                    q_V[y:y+bs_c, x:x+bs_c].astype(np.float32) * self.Q_C)

        self.prev_Y = deblock_filter(rec_Y, self.block_size)
        self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)

        # I-frame resetuje DPB i dodaje sie jako pierwsza referencja
        self._dpb_push(self.prev_Y, self.prev_U, self.prev_V, 'I')

        return {'type': 'I',
                'Y': q_Y, 'U': q_U, 'V': q_V,
                'h': h_pad, 'w': w_pad}

    # -----------------------------------------------------------------------
    # KLATKA P (Predykcyjna) — wielowątkowa
    # -----------------------------------------------------------------------
    def _encode_pframe(self, curr_Y, curr_U, curr_V, h_pad, w_pad):
        """Koduje klatkę P używając wielowątkowości dla rzędów bloków."""

        # --- ADAPTACYJNA Q: dostosuj kwantyzacje do zlozonosci klatki ---
        if self.adaptive_q:
            # Sredni MAD calej klatki = miara zlozonosci
            avg_mad = float(np.mean(np.abs(
                curr_Y.astype(np.float32) - self.prev_Y.astype(np.float32))))
            # MAD 0-5: scena statyczna -> Q nizsze (lepsza jakosc)
            # MAD 5-15: normalny ruch -> Q bazowe
            # MAD 15+: duzy ruch/ciezka scena -> Q wyzsze (mniejszy plik)
            if avg_mad < 5.0:
                factor = 0.75   # -25% Q — statyczna scena, wiecej szczegolów
            elif avg_mad < 15.0:
                factor = 1.0    # bazowe Q
            else:
                factor = min(1.0 + (avg_mad - 15.0) / 40.0, 1.5)  # maks 1.5x Q — bez przyciemniania
            self.Q_Y = self.Q_Y_base * factor
            self.Q_C = self.Q_C_base * factor

        # --- PRESCAN: oblicz MAD map całej klatki (jedna operacja numpy) ---
        mad_map = compute_mad_map(curr_Y, self.prev_Y, self.block_size)

        # Przygotuj liste referencji DPB do przekazania do workerow
        dpb_refs = [{'Y': r['Y'].copy(), 'U': r['U'].copy(), 'V': r['V'].copy()}
                    for r in self.dpb] if self.dpb else                    [{'Y': self.prev_Y.copy(), 'U': self.prev_U.copy(), 'V': self.prev_V.copy()}]

        tasks = []
        for row_idx, y in enumerate(range(0, h_pad, self.block_size)):
            mad_row = mad_map[row_idx] if row_idx < mad_map.shape[0] else None
            tasks.append((
                y, w_pad, h_pad,
                dpb_refs, curr_Y,
                curr_U, curr_V,
                self.search_range, self.Q_Y, self.Q_C,
                self.use_subpixel, mad_row
            ))

        # Uruchamiamy na wszystkich dostępnych rdzeniach
        results_by_row = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for y_out, row_results in executor.map(process_row_pframe, tasks):
                results_by_row[y_out] = row_results

        # Zbieramy wyniki do kompaktowych list (tylko bloki DETAIL)
        # Enkoder musi rekonstruowac klatke identycznie jak dekoder:
        # czyta WYLACZNIE z ref_Y/U/V (zamrozonej kopii prev), pisze do new_Y/U/V.
        # Bez tego enkoder modyfikuje prev_Y in-place i kolejne bloki widza juz
        # zrekonstruowane piksele zamiast oryginalnej referencji — mismatch z dekoderem.
        ref_Y = self.prev_Y.copy()
        ref_U = self.prev_U.copy()
        ref_V = self.prev_V.copy()
        new_Y = ref_Y.copy()
        new_U = ref_U.copy()
        new_V = ref_V.copy()

        block_list = []
        skipped = 0
        detail = 0
        for y in range(0, h_pad, self.block_size):
            for res in results_by_row[y]:
                res['y'] = y
                block_list.append(res)
                if res['mode'] == 0:
                    skipped += 1
                else:
                    detail += 1
                    # Rekonstrukcja — identyczna sciezka co dekoder
                    x  = res['x']
                    bs = res['bs']
                    dx_qp, dy_qp = res['mv_y_qp']
                    abs_y_qp = y * 4 + dy_qp
                    abs_x_qp = x * 4 + dx_qp
                    abs_y_qp = max(0, min(abs_y_qp, (h_pad - bs) * 4))
                    abs_x_qp = max(0, min(abs_x_qp, (w_pad - bs) * 4))
                    # Czytaj z zamrozonej referencji (ref_Y), pisz do new_Y
                    match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs)
                    rec_block = match_y + apply_idct2(
                        res['q_diff_y'].astype(np.float32) * self.Q_Y)
                    new_Y[y:y+bs, x:x+bs] = np.clip(rec_block, 0, 255)

                    # UV rekonstrukcja
                    cy, cx = y // 2, x // 2
                    bs_c = bs // 2
                    uv_dy = dy_qp // 8
                    uv_dx = dx_qp // 8
                    csy_c = max(0, min(cy + uv_dy, ref_U.shape[0] - bs_c))
                    csx_c = max(0, min(cx + uv_dx, ref_U.shape[1] - bs_c))
                    new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                        ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c] +
                        apply_idct2(res['q_diff_u'].astype(np.float32) * self.Q_C),
                        -128, 127)
                    new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                        ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c] +
                        apply_idct2(res['q_diff_v'].astype(np.float32) * self.Q_C),
                        -128, 127)

        self.prev_Y = deblock_filter(new_Y, self.block_size)
        self.prev_U = new_U
        self.prev_V = new_V
        self._dpb_push(self.prev_Y, self.prev_U, self.prev_V, 'P')

        total = skipped + detail
        print(f"   -> SKIP: {skipped}/{total} ({skipped/total*100:.1f}%) | "
              f"DETAIL: {detail} | Sub-pixel: {'ON' if self.use_subpixel else 'OFF'}")

        return {'type': 'P', 'blocks': block_list, 'h': h_pad, 'w': w_pad}

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # KLATKA B — backward-only prediction, dual-check SKIP
    # -----------------------------------------------------------------------
    def _encode_bframe(self, curr_Y, curr_U, curr_V, h_pad, w_pad):
        """
        B-klatka uproszczona: TYLKO predykcja wsteczna (prev_Y).
        Trick: do decyzji SKIP sprawdzamy MIN(SAD_prev, SAD_future) — lepsza
        detekcja niezmiennych bloków. DETAIL zawsze kodowany względem prev_Y.
        Format identyczny z P-frame: zero encoder-decoder mismatch.
        """
        bs = self.block_size
        block_list = []
        skipped = 0
        detail = 0

        # Prescan MAD dla szybkiego SKIP
        mad_map = compute_mad_map(curr_Y, self.prev_Y, bs)

        for row_idx, y in enumerate(range(0, h_pad, bs)):
            mad_row = mad_map[row_idx] if row_idx < mad_map.shape[0] else None
            for col_idx, x in enumerate(range(0, w_pad, bs)):

                # --- Prescan fast-SKIP ---
                if mad_row is not None and col_idx < len(mad_row):
                    if mad_row[col_idx] < 1.0:  # tylko pikselowo identyczne bloki
                        block_list.append({'x': x, 'y': y, 'bs': bs, 'mode': 0})
                        skipped += 1
                        continue

                curr_block = curr_Y[y:y+bs, x:x+bs].astype(np.float32)
                skip_threshold = bs * bs * 2  # surowy prog — mniej ghostingu

                # --- Dual-check SKIP: prev + future (jeśli dostępna) ---
                sad_back = np.sum(np.abs(curr_block - self.prev_Y[y:y+bs, x:x+bs]))
                if sad_back < skip_threshold:
                    block_list.append({'x': x, 'y': y, 'bs': bs, 'mode': 0})
                    skipped += 1
                    continue

                if self.future_Y is not None:
                    sad_fwd = np.sum(np.abs(curr_block - self.future_Y[y:y+bs, x:x+bs]))
                    if sad_fwd < skip_threshold:
                        block_list.append({'x': x, 'y': y, 'bs': bs, 'mode': 0})
                        skipped += 1
                        continue

                # --- Estymacja ruchu względem prev_Y (backward only) ---
                min_sad = float('inf')
                best_dx_qp, best_dy_qp = 0, 0
                for dy in range(-self.search_range, self.search_range + 1):
                    for dx in range(-self.search_range, self.search_range + 1):
                        sy = max(0, min(y + dy, h_pad - bs))
                        sx = max(0, min(x + dx, w_pad - bs))
                        sad = np.sum(np.abs(curr_block -
                                            self.prev_Y[sy:sy+bs, sx:sx+bs]))
                        if sad < min_sad:
                            min_sad = sad
                            best_dx_qp = dx * 4
                            best_dy_qp = dy * 4

                abs_y_qp = max(0, min(y * 4 + best_dy_qp, (h_pad - bs) * 4))
                abs_x_qp = max(0, min(x * 4 + best_dx_qp, (w_pad - bs) * 4))
                match_y = interpolate_subpixel(self.prev_Y, abs_y_qp, abs_x_qp, bs)

                q_diff_y = np.round(apply_dct2(
                    curr_block - match_y) / self.Q_Y).astype(np.int16)

                cy, cx = y // 2, x // 2
                bs_c = bs // 2
                uv_dy = best_dy_qp // 8
                uv_dx = best_dx_qp // 8
                csy_c = max(0, min(cy + uv_dy, self.prev_U.shape[0] - bs_c))
                csx_c = max(0, min(cx + uv_dx, self.prev_U.shape[1] - bs_c))
                q_diff_u = np.round(apply_dct2(
                    curr_U[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32) -
                    self.prev_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]) / self.Q_C).astype(np.int16)
                q_diff_v = np.round(apply_dct2(
                    curr_V[cy:cy+bs_c, cx:cx+bs_c].astype(np.float32) -
                    self.prev_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c]) / self.Q_C).astype(np.int16)

                # Format identyczny z P-frame (mv_y_qp, bez pred_mode)
                block_list.append({
                    'x': x, 'y': y, 'bs': bs, 'mode': 1,
                    'mv_y_qp': (int(best_dx_qp), int(best_dy_qp)),
                    'q_diff_y': q_diff_y,
                    'q_diff_u': q_diff_u,
                    'q_diff_v': q_diff_v,
                })
                detail += 1

        total = skipped + detail
        print(f"   -> B-FRAME | SKIP: {skipped}/{total} ({skipped/total*100:.1f}%) "
              f"| DETAIL: {detail}")

        return {'type': 'B', 'blocks': block_list, 'h': h_pad, 'w': w_pad}

    # -----------------------------------------------------------------------
    # KODOWANIE KLATKI (dispatcher)
    # -----------------------------------------------------------------------
    def encode_frame(self, img_np, frame_type='auto', frame_index=0):
        h, w, _ = img_np.shape
        h_pad = h - (h % 16)
        w_pad = w - (w % 16)
        curr_Y, curr_U, curr_V = rgb_to_yuv420(
            img_np[:h_pad, :w_pad].astype(np.float32))

        if self.prev_Y is None:
            return self._encode_iframe(curr_Y, curr_U, curr_V, h_pad, w_pad)

        if frame_type == 'I':
            return self._encode_iframe(curr_Y, curr_U, curr_V, h_pad, w_pad)
        elif frame_type == 'B' or (
                self.use_bframes and
                frame_index % (self.bframe_interval + 1) == 0 and
                self.future_Y is not None):
            return self._encode_bframe(curr_Y, curr_U, curr_V, h_pad, w_pad)
        else:
            return self._encode_pframe(curr_Y, curr_U, curr_V, h_pad, w_pad)

    # -----------------------------------------------------------------------
    # DEKODOWANIE KLATKI I
    # -----------------------------------------------------------------------
    def _decode_iframe(self, data, h, w):
        bs = self.block_size
        bs_c = bs // 2
        rec_Y = np.zeros((h, w), dtype=np.float32)
        rec_U = np.zeros((h//2, w//2), dtype=np.float32)
        rec_V = np.zeros((h//2, w//2), dtype=np.float32)

        for y in range(0, h, bs):
            for x in range(0, w, bs):
                rec_Y[y:y+bs, x:x+bs] = apply_idct2(
                    data['Y'][y:y+bs, x:x+bs].astype(np.float32) * self.Q_Y)
        for y in range(0, h//2, bs_c):
            for x in range(0, w//2, bs_c):
                rec_U[y:y+bs_c, x:x+bs_c] = apply_idct2(
                    data['U'][y:y+bs_c, x:x+bs_c].astype(np.float32) * self.Q_C)
                rec_V[y:y+bs_c, x:x+bs_c] = apply_idct2(
                    data['V'][y:y+bs_c, x:x+bs_c].astype(np.float32) * self.Q_C)

        self.prev_Y = deblock_filter(rec_Y, bs)
        self.prev_U = np.clip(rec_U, -128, 127).astype(np.float32)
        self.prev_V = np.clip(rec_V, -128, 127).astype(np.float32)
        self._dpb_push(self.prev_Y, self.prev_U, self.prev_V, 'I')
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    # -----------------------------------------------------------------------
    # DEKODOWANIE KLATKI P
    # -----------------------------------------------------------------------
    def _decode_pframe(self, data, h, w):
        new_Y = self.prev_Y.copy()
        new_U = self.prev_U.copy()
        new_V = self.prev_V.copy()

        for block in data['blocks']:
            x  = block['x']
            y  = block['y'] if 'y' in block else data.get('_y', 0)
            bs = block['bs']

            if block['mode'] == 0:  # SKIP
                continue

            if y + bs > h or x + bs > w:
                continue

            # Wybierz referencje z DPB na podstawie ref_idx
            ref_idx = block.get('ref_idx', len(self.dpb) - 1)
            ref_idx = max(0, min(ref_idx, len(self.dpb) - 1))
            if self.dpb:
                ref = self.dpb[ref_idx]
                ref_Y, ref_U, ref_V = ref['Y'], ref['U'], ref['V']
            else:
                ref_Y, ref_U, ref_V = self.prev_Y, self.prev_U, self.prev_V

            dx_qp, dy_qp = block['mv_y_qp']
            abs_y_qp = y * 4 + dy_qp
            abs_x_qp = x * 4 + dx_qp
            abs_y_qp = max(0, min(abs_y_qp, (h - bs) * 4))
            abs_x_qp = max(0, min(abs_x_qp, (w - bs) * 4))

            match_y = interpolate_subpixel(ref_Y, abs_y_qp, abs_x_qp, bs)
            rec_block = match_y + apply_idct2(
                block['q_diff_y'].astype(np.float32) * self.Q_Y)
            new_Y[y:y+bs, x:x+bs] = np.clip(rec_block, 0, 255)

            cy, cx = y // 2, x // 2
            bs_c = bs // 2
            uv_dy = dy_qp // 8
            uv_dx = dx_qp // 8
            csy_c = max(0, min(cy + uv_dy, ref_U.shape[0] - bs_c))
            csx_c = max(0, min(cx + uv_dx, ref_U.shape[1] - bs_c))
            new_U[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                ref_U[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c] +
                apply_idct2(block['q_diff_u'].astype(np.float32) * self.Q_C),
                -128, 127)
            new_V[cy:cy+bs_c, cx:cx+bs_c] = np.clip(
                ref_V[csy_c:csy_c+bs_c, csx_c:csx_c+bs_c] +
                apply_idct2(block['q_diff_v'].astype(np.float32) * self.Q_C),
                -128, 127)

        self.prev_Y = deblock_filter(new_Y, self.block_size)
        self.prev_U = new_U
        self.prev_V = new_V
        self._dpb_push(self.prev_Y, self.prev_U, self.prev_V, 'P')
        return yuv420_to_rgb(self.prev_Y, self.prev_U, self.prev_V)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # DEKODOWANIE KLATKI B — identyczne z P (backward-only, brak fwd/bi)
    # -----------------------------------------------------------------------
    def _decode_bframe(self, data, h, w):
        """
        B-klatki: dekodujemy jak P, ale NIE aktualizujemy prev_Y/U/V.
        Enkoder tez nie zmienia referencji po B-klatce — musi byc symetria.
        Bez tego kazda B-klatka niszczy referencje i nastepne P-klatki sa poszarpane.
        """
        # Zapamietaj referencje przed dekodowaniem
        saved_Y = self.prev_Y.copy()
        saved_U = self.prev_U.copy()
        saved_V = self.prev_V.copy()

        # Zdekoduj klatke (uzywa prev_Y jako referencji, zapisze do prev_Y)
        img = self._decode_pframe(data, h, w)

        # Przywroc referencje — B-klatka nie zmienia stanu dla kolejnych klatek
        self.prev_Y = saved_Y
        self.prev_U = saved_U
        self.prev_V = saved_V

        return img

    # -----------------------------------------------------------------------
    # DEKODOWANIE KLATKI (dispatcher)
    # -----------------------------------------------------------------------
    def decode_frame(self, data, h, w):
        ft = data['type']
        if ft == 'I':
            return self._decode_iframe(data, h, w)
        elif ft == 'P':
            return self._decode_pframe(data, h, w)
        elif ft == 'B':
            return self._decode_bframe(data, h, w)
        else:
            raise ValueError(f"Nieznany typ klatki: {ft}")


# =============================================================================
# FORMAT BINARNY v2.7 — ZAPIS / ODCZYT
# Struktura pliku:
#   [4B naglowek: W, H jako uint16 big-endian]
#   [1B typ klatki: 'I', 'P', 'B']
#   Dla I: raw Y (h*w*2B) + U (h/2*w/2*2B) + V (h/2*w/2*2B)
#   Dla P/B (format BITMAP+SCHEMA — x/y/bs NIE sa pakowane):
#     [2B: cols uint16]  -- szerokosc siatki blokow (w_pad // 16)
#     [2B: rows uint16]  -- wysokosc siatki blokow (h_pad // 16)
#     [ceil(rows*cols/8) B: bitmapa SKIP(0)/DETAIL(1), MSB-first, raster scan]
#     Dla kazdego bloku DETAIL (w kolejnosci raster scan, bs=16 zawsze):
#       [1B: ref_idx uint8 — indeks referencji w DPB (0=najstarsza)]
#       [2B: mv_dx int16 quarter-pixel]
#       [2B: mv_dy int16 quarter-pixel]
#       [512 B: q_diff_y int16 16x16]
#       [128 B: q_diff_u int16 8x8]
#       [128 B: q_diff_v int16 8x8]
# Dekoder odtwarza x = col*16, y = row*16 ze schematu — bez redundancji.
# =============================================================================

PRED_MODE_MAP = {'back': 0, 'fwd': 1, 'bi': 2}
PRED_MODE_RMAP = {0: 'back', 1: 'fwd', 2: 'bi'}

# FORMAT v2.7 — BITMAP + SCHEMA (brak x/y/bs w strumieniu):
#   Dekoder zna schemat skanowania (raster 16x16), wiec x/y/bs NIE sa pakowane.
#   bs jest zawsze stalny = 16.
#   Struktura P/B-klatki:
#     [2B: cols uint16]  -- liczba kolumn blokow (= w_pad // 16)
#     [2B: rows uint16]  -- liczba wierszy blokow (= h_pad // 16)
#     [ceil(rows*cols/8) B: bitmapa SKIP(0)/DETAIL(1), MSB-first, raster scan]
#     Dla kazdego bloku DETAIL (w kolejnosci raster scan):
#       [2B: mv_dx int16 quarter-pixel]
#       [2B: mv_dy int16 quarter-pixel]
#       [512 B: q_diff_y int16 16x16]
#       [128 B: q_diff_u int16 8x8]
#       [128 B: q_diff_v int16 8x8]
#   Zysk vs v2.4: -5B (x+y+bs) na kazdy DETAIL blok.

_BS = 16   # Staly rozmiar bloku luma
_BS_C = 8  # Staly rozmiar bloku chroma


def _serialize_blocks(blocks, is_bframe=False):
    """Serializuje klatke P/B w formacie v2.7 -- bitmapa + dane DETAIL bez x/y/bs.
    Bloki MUSZA byc w raster scan order (y rosnaco, x rosnaco), bs=16 zawsze."""
    if not blocks:
        return struct.pack('>HH', 0, 0)

    # Wyznacz rozmiar siatki z max x/y (bez budowania indeksow — odporniejsze)
    max_x = max(b['x'] for b in blocks)
    max_y = max(b.get('y', 0) for b in blocks)
    cols = max_x // _BS + 1
    rows = max_y // _BS + 1

    # Zbuduj grid (row, col) -> mode
    grid = {}
    for b in blocks:
        r = b.get('y', 0) // _BS
        c = b['x'] // _BS
        grid[(r, c)] = b

    n_blocks = rows * cols
    bitmap_bytes = bytearray((n_blocks + 7) // 8)
    detail_blocks = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            blk = grid.get((r, c))
            if blk and blk['mode'] == 1:
                bitmap_bytes[idx >> 3] |= (1 << (7 - (idx & 7)))
                detail_blocks.append(blk)

    buf = bytearray()
    buf.extend(struct.pack('>HH', cols, rows))
    buf.extend(bitmap_bytes)

    for b in detail_blocks:
        ref_idx = b.get('ref_idx', 0)  # 0 = najnowsza referencja (prev)
        buf.extend(struct.pack('B', ref_idx))   # 1B: indeks referencji w DPB
        dx_qp, dy_qp = b['mv_y_qp']
        buf.extend(struct.pack('>hh', dx_qp, dy_qp))
        buf.extend(b['q_diff_y'].astype(np.int16).flatten().tobytes())
        buf.extend(b['q_diff_u'].astype(np.int16).flatten().tobytes())
        buf.extend(b['q_diff_v'].astype(np.int16).flatten().tobytes())

    return buf


def _deserialize_blocks(data, offset, is_bframe=False):
    """Odczytuje klatke P/B w formacie v2.7 -- bitmapa + schemat skanowania."""
    cols, rows = struct.unpack_from('>HH', data, offset); offset += 4
    if cols == 0 or rows == 0:
        return [], offset

    n_blocks = rows * cols
    bitmap_size = (n_blocks + 7) // 8
    bitmap = data[offset:offset + bitmap_size]; offset += bitmap_size

    sz_y = _BS   * _BS   * 2   # 512 B
    sz_c = _BS_C * _BS_C * 2   # 128 B

    blocks = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            is_detail = bool(bitmap[idx >> 3] & (1 << (7 - (idx & 7))))
            x = c * _BS
            y = r * _BS
            if not is_detail:
                blocks.append({'mode': 0, 'x': x, 'y': y, 'bs': _BS})
            else:
                ref_idx, = struct.unpack_from('B', data, offset); offset += 1
                dx_qp, dy_qp = struct.unpack_from('>hh', data, offset); offset += 4
                block = {
                    'mode': 1, 'x': x, 'y': y, 'bs': _BS,
                    'ref_idx': ref_idx,
                    'mv_y_qp': (dx_qp, dy_qp),
                }
                block['q_diff_y'] = np.frombuffer(
                    data[offset:offset+sz_y], dtype=np.int16).reshape(_BS, _BS)
                offset += sz_y
                block['q_diff_u'] = np.frombuffer(
                    data[offset:offset+sz_c], dtype=np.int16).reshape(_BS_C, _BS_C)
                offset += sz_c
                block['q_diff_v'] = np.frombuffer(
                    data[offset:offset+sz_c], dtype=np.int16).reshape(_BS_C, _BS_C)
                offset += sz_c
                blocks.append(block)

    return blocks, offset


# =============================================================================
# GŁÓWNE FUNKCJE ENCODE / DECODE
# =============================================================================

def detect_scene_cut(frame_a, frame_b, threshold=35.0):
    """
    Wykrywa ciecie sceny miedzy dwoma klatkami.
    Porownuje sredni MAD calej klatki — jesli > threshold, to ciecie.
    threshold=35: odpowiada ~14% roznicyy jasnosci srednio na piksel.
    Szybkie: jedna operacja numpy na przeskalowanych klatkach.
    """
    # Przeskaluj do malego rozmiaru dla szybkosci (64x36)
    h, w = frame_a.shape[:2]
    step_y = max(1, h // 36)
    step_x = max(1, w // 64)
    a = frame_a[::step_y, ::step_x].astype(np.float32)
    b = frame_b[::step_y, ::step_x].astype(np.float32)
    mad = np.mean(np.abs(a - b))
    return mad > threshold, mad


def encode_video(input_path, output_path, max_frames=10,
                 use_subpixel=True, use_bframes=True, search_range=24,
                 full=False, keyframe_interval=50,
                 q_y=22.0, q_c=40.0, adaptive_q=False,
                 scene_cut_threshold=35.0):
    print(f"\n╔══ TOP TOPÓW CODEC v2 — KODOWANIE ══╗")
    print(f"  Wejście:     {input_path}")
    print(f"  Wyjście:     {output_path}")
    if full:
        max_frames = None
    print(f"  Klatki:      {'WSZYSTKIE' if full else max_frames}")
    print(f"  Sub-pixel:   {'ON' if use_subpixel else 'OFF'}")
    print(f"  B-Frames:    {'ON' if use_bframes else 'OFF'}")
    print(f"  Zasięg:      ±{search_range}px")
    print(f"  Keyframe co: {keyframe_interval} klatek")
    aq_str = " (ADAPTACYJNA)" if adaptive_q else ""
    print(f"  Q_Y/Q_C:     {q_y:.0f}/{q_c:.0f}{aq_str}")
    sc_str = "WYŁĄCZONA" if scene_cut_threshold <= 0 else f"próg MAD={scene_cut_threshold:.0f}"
    print(f"  Scene cut:   {sc_str}")
    print(f"╚{'═'*38}╝\n")

    codec = TopTopowCodecV2(
        search_range=search_range,
        use_subpixel=use_subpixel,
        use_bframes=use_bframes,
        q_y=q_y,
        q_c=q_c,
        adaptive_q=adaptive_q,
    )

    # Wczytujemy wszystkie potrzebne klatki do pamięci (potrzebne dla B-frames)
    frames_raw = []
    for i, frame in enumerate(iio.imiter(input_path)):
        if max_frames is not None and i >= max_frames:
            break
        frames_raw.append(frame)

    n = len(frames_raw)
    h_pad = w_pad = 0
    start = time.time()
    raw_total = 0

    # Streaming zstd: kazda klatka trafia do pliku od razu — brak blokady na koncu
    cctx = zstd.ZstdCompressor(level=22)
    with open(output_path, 'wb') as f_out:
        with cctx.stream_writer(f_out, closefd=False) as writer:

            i = 0
            while i < n:
                frame = frames_raw[i]
                h, w, _ = frame.shape
                hp = h - (h % 16)
                wp = w - (w % 16)

                # Detekcja ciecia sceny — wymus I-klatke jesli duza zmiana
                is_scene_cut = False
                if i > 0 and scene_cut_threshold > 0:
                    is_scene_cut, mad_val = detect_scene_cut(
                        frames_raw[i-1], frames_raw[i], scene_cut_threshold)
                    if is_scene_cut:
                        print(f"[Klatka {i+1}/{n}] ✂ CIECIE SCENY (MAD={mad_val:.1f}) → I-Frame")

                is_keyframe = (i == 0) or is_scene_cut or (keyframe_interval > 0 and i % keyframe_interval == 0)

                if is_keyframe:
                    print(f"[Klatka {i+1}/{n}] I-Frame{'  ← keyframe' if i > 0 else ''}...")
                    encoded = codec.encode_frame(frame, frame_type='I')
                    if i == 0:
                        h_pad, w_pad = encoded['h'], encoded['w']
                        chunk = struct.pack('>HH', w_pad, h_pad)
                        chunk += b'I'
                    else:
                        chunk = b'I'
                    chunk += encoded['Y'].tobytes()
                    chunk += encoded['U'].tobytes()
                    chunk += encoded['V'].tobytes()

                elif use_bframes and i + 1 < n and i % (codec.bframe_interval + 1) == 0:
                    future = frames_raw[i + 1]
                    fY, fU, fV = rgb_to_yuv420(future[:hp, :wp].astype(np.float32))
                    codec.future_Y = fY
                    codec.future_U = fU
                    codec.future_V = fV
                    print(f"[Klatka {i+1}/{n}] B-Frame (future={i+2})...")
                    encoded = codec.encode_frame(frame, frame_type='B', frame_index=i)
                    chunk = b'B' + bytes(_serialize_blocks(encoded['blocks']))

                else:
                    codec.future_Y = None
                    print(f"[Klatka {i+1}/{n}] P-Frame...")
                    encoded = codec.encode_frame(frame, frame_type='P', frame_index=i)
                    chunk = b'P' + bytes(_serialize_blocks(encoded['blocks']))

                writer.write(chunk)
                raw_total += len(chunk)
                i += 1

    elapsed = time.time() - start
    compressed_size = os.path.getsize(output_path)
    ratio = raw_total / compressed_size if compressed_size > 0 else 0
    print(f"\n✓ Czas: {elapsed:.1f}s | "
          f"Surowe: {raw_total//1024}KB | "
          f"Skompresowane: {compressed_size//1024}KB | "
          f"Ratio Zstd: {ratio:.1f}x")


def _parse_all_frames(raw, h, w):
    """Wczytaj wszystkie klatki z bufora binarnego do listy dicts."""
    offset = 4  # pomiń nagłówek W/H
    y_size  = h * w * 2
    uv_size = (h // 2) * (w // 2) * 2
    frames = []
    while offset < len(raw):
        ft = raw[offset:offset+1]; offset += 1
        if ft == b'I':
            Y = np.frombuffer(raw[offset:offset+y_size],
                              dtype=np.int16).reshape(h, w);  offset += y_size
            U = np.frombuffer(raw[offset:offset+uv_size],
                              dtype=np.int16).reshape(h//2, w//2); offset += uv_size
            V = np.frombuffer(raw[offset:offset+uv_size],
                              dtype=np.int16).reshape(h//2, w//2); offset += uv_size
            frames.append({'type': 'I', 'Y': Y, 'U': U, 'V': V})
        elif ft == b'P':
            blocks, offset = _deserialize_blocks(raw, offset)
            frames.append({'type': 'P', 'blocks': blocks})
        elif ft == b'B':
            blocks, offset = _deserialize_blocks(raw, offset)
            frames.append({'type': 'B', 'blocks': blocks})
        else:
            break
    return frames


def decode_video(input_path, output_path):
    print(f"\n╔══ TOP TOPÓW CODEC v2 — DEKODOWANIE ══╗")
    print(f"  Wejście: {input_path}")
    print(f"  Wyjście: {output_path}")
    print(f"╚{'═'*40}╝\n")

    with open(input_path, 'rb') as f:
        # stream_reader dziala ze strumieniowym formatem zstd (brak rozmiaru w naglowku)
        dctx = zstd.ZstdDecompressor()
        raw = dctx.stream_reader(f).read()

    w, h = struct.unpack_from('>HH', raw, 0)

    # PASS 1: wczytaj wszystkie klatki do pamięci
    all_frames = _parse_all_frames(raw, h, w)
    n = len(all_frames)

    # PASS 2: dekoduj z lookahead dla B-klatek
    # Dla każdej B-klatki musimy znać ZREKONSTRUOWANĄ następną klatkę (future).
    # Rozwiązanie: cache zrekonstruowanych Y-płaszczyzn indeksowanych numerem klatki.
    codec = TopTopowCodecV2()
    # Słownik: indeks klatki → zrekonstruowana Y-płaszczyzna
    reconstructed_Y: dict = {}
    reconstructed_U: dict = {}
    reconstructed_V: dict = {}

    decoded_frames = []

    for i, data in enumerate(all_frames):
        ft = data['type']

        if ft == 'B':
            # Znajdź następną niebędącą B-klatką i zrekonstruuj ją jeśli trzeba
            future_idx = None
            for j in range(i + 1, n):
                if all_frames[j]['type'] != 'B':
                    future_idx = j
                    break

            if future_idx is not None and future_idx not in reconstructed_Y:
                # Tymczasowy dekoder żeby zrekonstruować klatkę przyszłą
                # (nie wpływa na stan głównego kodeka)
                tmp = TopTopowCodecV2()
                # Odtwórz jego stan na podstawie naszych rekonstrukcji
                # Znajdź ostatnią niebędącą B przed future_idx
                ref_idx = None
                for j in range(future_idx - 1, -1, -1):
                    if all_frames[j]['type'] != 'B' and j in reconstructed_Y:
                        ref_idx = j
                        break
                if ref_idx is not None:
                    tmp.prev_Y = reconstructed_Y[ref_idx]
                    tmp.prev_U = reconstructed_U[ref_idx]
                    tmp.prev_V = reconstructed_V[ref_idx]
                tmp.future_Y = None
                # Dekoduj klatkę przyszłą (to zawsze I lub P)
                tmp.decode_frame(all_frames[future_idx], h, w)
                reconstructed_Y[future_idx] = tmp.prev_Y.copy()
                reconstructed_U[future_idx] = tmp.prev_U.copy()
                reconstructed_V[future_idx] = tmp.prev_V.copy()

            # Ustaw future_Y w głównym kodeku
            if future_idx is not None and future_idx in reconstructed_Y:
                codec.future_Y = reconstructed_Y[future_idx]
                codec.future_U = reconstructed_U[future_idx]
                codec.future_V = reconstructed_V[future_idx]
            else:
                codec.future_Y = None
        else:
            codec.future_Y = None

        img = codec.decode_frame(data, h, w)
        decoded_frames.append(img)

        # Zapamiętaj zrekonstruowaną klatkę niebędącą B
        if ft != 'B':
            reconstructed_Y[i] = codec.prev_Y.copy()
            reconstructed_U[i] = codec.prev_U.copy()
            reconstructed_V[i] = codec.prev_V.copy()

        print(f"  Zdekodowano klatkę {i+1}/{n} [{ft}]")

    iio.imwrite(output_path, decoded_frames, fps=25)
    print(f"\n✓ SUKCES! {n} klatek → {output_path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kodek TOP TOPÓW v2 — pełna implementacja")
    parser.add_argument('-i', '--input',   required=True,
                        help="Plik wejściowy (.mp4 lub .toptop)")
    parser.add_argument('-o', '--output',  required=True,
                        help="Plik wyjściowy (.toptop lub .mp4)")
    parser.add_argument('-n', '--frames',  type=int, default=10,
                        help="Liczba klatek do zakodowania (domyślnie: 10)")
    parser.add_argument('-f', '--full',    action='store_true',
                        help="Zakoduj pełny plik wejściowy (ignoruje -n)")
    parser.add_argument('-d', '--decode',  action='store_true',
                        help="Tryb dekodowania")
    parser.add_argument('--no-subpixel',   action='store_true',
                        help="Wyłącz sub-pikselową estymację ruchu")
    parser.add_argument('--no-bframes',    action='store_true',
                        help="Wyłącz klatki B")
    parser.add_argument('--search-range',  type=int, default=24,
                        help="Zasięg szukania ruchu w pikselach (domyślnie: 24, TSS jest szybki)")
    parser.add_argument('--keyframe-interval', type=int, default=50,
                        help="I-klatka co N klatek — reset referencji (domyślnie: 50, 0=wyłącz)")
    parser.add_argument('--q-y',  type=float, default=22.0,
                        help="Kwantyzacja lumy Y (domyślnie: 22, wyżej=mniejszy plik)")
    parser.add_argument('--q-c',  type=float, default=40.0,
                        help="Kwantyzacja chromy UV (domyślnie: 40)")
    parser.add_argument('--scene-cut', type=float, default=35.0,
                        help="Próg detekcji cięcia sceny MAD (domyślnie: 35, 0=wyłącz)")
    parser.add_argument('--adaptive-q', action='store_true',
                        help="Adaptacyjna Q: auto-dostosowanie do złożoności sceny")
    parser.add_argument('-1', '--preset-small', action='store_true',
                        help="Preset 1 — mały plik: Q_Y=40 Q_C=70 (duża strata jakości)")
    parser.add_argument('-2', '--preset-adaptive', action='store_true',
                        help="Preset 2 — adaptacyjna Q: auto Q_Y/Q_C wg złożoności sceny")
    parser.add_argument('-3', '--preset-quality', action='store_true',
                        help="Preset 3 — jakość+zasięg: Q_Y=16 Q_C=30 search=48")
    args = parser.parse_args()

    # Zastosuj presety (nadpisuja pojedyncze flagi)
    q_y, q_c, adaptive_q, search_range = args.q_y, args.q_c, args.adaptive_q, args.search_range
    if args.preset_small:
        q_y, q_c = 32.0, 55.0
        print("▶ Preset 1 — MAŁY PLIK: Q_Y=32 Q_C=55")
    elif args.preset_adaptive:
        adaptive_q = True
        print("▶ Preset 2 — ADAPTACYJNA Q: auto wg złożoności sceny")
    elif args.preset_quality:
        q_y, q_c, search_range = 16.0, 30.0, 48
        print("▶ Preset 3 — JAKOŚĆ: Q_Y=16 Q_C=30 search=48")

    if args.decode:
        decode_video(args.input, args.output)
    else:
        encode_video(
            args.input, args.output,
            max_frames=args.frames,
            use_subpixel=not args.no_subpixel,
            use_bframes=not args.no_bframes,
            search_range=search_range,
            full=args.full,
            keyframe_interval=args.keyframe_interval,
            q_y=q_y,
            q_c=q_c,
            adaptive_q=adaptive_q,
            scene_cut_threshold=args.scene_cut,
        )
