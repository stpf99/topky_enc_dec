#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.1 — SYMULACJA OPTYCZNA                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v1.1 zmiany:                                                              ║
║  - kolor: pełne YCbCr (3 kanały) zamiast samego Y                         ║
║  - NCC poprawiona: peak / (||prev||₂ × ||curr||₂) → zakres [−1,1]         ║
║  - shift bez zawijania: zero-fill zamiast np.roll                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import zstandard as zstd
import struct
import argparse
import time
from contextlib import contextmanager

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

_MAGIC      = b'PH02'   # nowy magic — niekompatybilny z v1.0
_ZSTD_LEVEL = 19

# ═══════════════════════════════════════════════════════════════════════════════
# KONWERSJA KOLORÓW  RGB ↔ YCbCr
# ═══════════════════════════════════════════════════════════════════════════════

def rgb_to_ycbcr(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 → YCbCr float32.  Y ∈ [0,255]  Cb/Cr ∈ [−128,127]"""
    f = frame[..., :3].astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B
    return np.stack([Y, Cb, Cr], axis=-1)   # (H, W, 3) float32

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """YCbCr float32 → RGB uint8"""
    Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    R = Y               + 1.402    * Cr
    G = Y - 0.344136   * Cb - 0.714136 * Cr
    B = Y + 1.772      * Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)

# ═══════════════════════════════════════════════════════════════════════════════
# FIZYKA: INTERFERENCJA I KORELACJA OPTYCZNA
# ═══════════════════════════════════════════════════════════════════════════════

def shift_block(block: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Przesuwa blok (2D lub 3D HxWxC) bez zawijania.
    Krawędzie wypełniane zerami (zero-fill), nie cyklicznie.
    """
    result = np.zeros_like(block)
    h, w   = block.shape[:2]
    # wycinek źródłowy
    sy = slice(max(0, -dy), min(h, h - dy))
    sx = slice(max(0, -dx), min(w, w - dx))
    # wycinek docelowy
    dy_ = slice(max(0,  dy), min(h, h + dy))
    dx_ = slice(max(0,  dx), min(w, w + dx))
    result[dy_, dx_] = block[sy, sx]
    return result


def optical_correlate(prev_block: np.ndarray, curr_block: np.ndarray) -> tuple:
    """
    Cross-correlation na kanale Y (luminancja).
    Zwraca (dy, dx, NCC) gdzie NCC ∈ [−1, 1].

    NCC ≈ 1.0  → identyczna treść (tylko przesunięta) → ścieżka ruchu
    NCC << 1.0 → nowe zjawisko / cięcie sceny         → Intra blok
    """
    prev_y = prev_block[..., 0] if prev_block.ndim == 3 else prev_block
    curr_y = curr_block[..., 0] if curr_block.ndim == 3 else curr_block

    # Soczewka (FFT) → mnożenie fourierowskie → Detektor (IFFT)
    corr = np.fft.irfft2(
        np.fft.rfft2(prev_y) *
        np.conj(np.fft.rfft2(curr_y[::-1, ::-1], s=prev_y.shape))
    )

    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = prev_y.shape
    dy = max_idx[0] if max_idx[0] < h // 2 else max_idx[0] - h
    dx = max_idx[1] if max_idx[1] < w // 2 else max_idx[1] - w

    # Poprawna normalizacja NCC — Cauchy-Schwarz gwarantuje zakres [−1, 1]
    e_prev = np.sqrt(np.sum(prev_y ** 2) + 1e-6)
    e_curr = np.sqrt(np.sum(curr_y ** 2) + 1e-6)
    ncc    = float(np.clip(corr[max_idx] / (e_prev * e_curr), -1.0, 1.0))

    return dy, dx, ncc


def extract_interference_delta(prev_block: np.ndarray, curr_block: np.ndarray,
                               dy: int, dx: int) -> np.ndarray:
    """
    Delta = curr − shift(prev).
    Pracuje na blokach 3D (bh, bw, 3).  Shift bez zawijania.
    """
    return curr_block - shift_block(prev_block, dy, dx)

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITEKTURA KODEKA
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicCodec:
    def __init__(self, brush_size: int = 64, energy_threshold: float = 0.85,
                 delta_threshold: float = 5.0):
        self.bs         = brush_size
        self.energy_thr = energy_threshold   # NCC ≥ thr → ścieżka ruchu
        self.delta_thr  = delta_threshold    # |delta| ≤ thr → szum, ignoruj
        self.prev_wave  = None               # (H, W, 3) float32 YCbCr

    def _split_to_brushes(self, wave: np.ndarray):
        """Dzieli pole falowe (H, W, 3) na bloki robocze."""
        h, w = wave.shape[:2]
        for y in range(0, h, self.bs):
            for x in range(0, w, self.bs):
                yield x, y, wave[y:y+self.bs, x:x+self.bs]

    def encode_frame(self, frame_rgb: np.ndarray) -> list:
        """Zamiana klatki RGB uint8 na ścieżki optyczne."""
        wave3 = rgb_to_ycbcr(frame_rgb)   # (H, W, 3) float32
        paths = []

        if self.prev_wave is None:
            # Pierwsza klatka — pełne pole falowe float32
            paths.append({'type': 0, 'data': wave3.tobytes()})
            self.prev_wave = wave3
            return paths

        for x, y, brush in self._split_to_brushes(wave3):
            bh, bw     = brush.shape[:2]
            prev_brush = self.prev_wave[y:y+bh, x:x+bw]

            dy, dx, ncc = optical_correlate(prev_brush, brush)

            if ncc >= self.energy_thr:
                # Dobra korelacja → liczymy deltę interferencyjną
                delta = extract_interference_delta(prev_brush, brush, dy, dx)
                # Maska szumu: maksimum po kanałach
                delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
                if np.any(delta_mask):
                    sparse_delta = delta * delta_mask[..., np.newaxis]
                    paths.append({
                        'type': 1,
                        'x': x, 'y': y, 'dx': dx, 'dy': dy, 'bh': bh, 'bw': bw,
                        'data': sparse_delta.astype(np.float16).tobytes()
                    })
                # jeśli delta == 0 wszędzie → blok statyczny, nic nie piszemy
            else:
                # Słaba korelacja → nowe zjawisko (Intra blok)
                paths.append({
                    'type': 2,
                    'x': x, 'y': y, 'dx': 0, 'dy': 0, 'bh': bh, 'bw': bw,
                    'data': brush.astype(np.float16).tobytes()
                })

        self.prev_wave = wave3
        return paths

    def decode_frame(self, paths: list, h: int, w: int) -> np.ndarray:
        """Odwrotność optyczna — zwraca RGB uint8."""
        if self.prev_wave is None:
            # Pierwsza klatka (float32, 3 kanały)
            self.prev_wave = (
                np.frombuffer(paths[0]['data'], dtype=np.float32).reshape(h, w, 3)
            )
            return ycbcr_to_rgb(self.prev_wave)

        new_wave = self.prev_wave.copy()

        for path in paths:
            x, y   = path['x'], path['y']
            bh, bw = path['bh'], path['bw']

            if path['type'] == 1:   # Ścieżka ruchu
                dy, dx = path['dy'], path['dx']
                delta  = (np.frombuffer(path['data'], dtype=np.float16)
                           .reshape(bh, bw, 3).astype(np.float32))
                shifted = shift_block(new_wave[y:y+bh, x:x+bw], dy, dx)
                new_wave[y:y+bh, x:x+bw] = shifted + delta

            elif path['type'] == 2: # Intra blok
                new_wave[y:y+bh, x:x+bw] = (
                    np.frombuffer(path['data'], dtype=np.float16)
                    .reshape(bh, bw, 3).astype(np.float32)
                )

        self.prev_wave = new_wave
        return ycbcr_to_rgb(new_wave)

# ═══════════════════════════════════════════════════════════════════════════════
# STRUMIENIE I/O
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_paths(paths: list) -> bytes:
    out = bytearray()
    out.extend(struct.pack('>H', len(paths)))
    for p in paths:
        out.append(p['type'])
        if p['type'] == 0:
            out.extend(p['data'])
        else:
            out.extend(struct.pack('>HHhhHH',
                p['x'], p['y'], p['dx'], p['dy'], p['bh'], p['bw']))
            out.extend(p['data'])
    return bytes(out)


def deserialize_paths(data: bytes) -> list:
    off   = 0
    count = struct.unpack_from('>H', data, off)[0]; off += 2
    paths = []
    for _ in range(count):
        t = data[off]; off += 1
        if t == 0:
            paths.append({'type': 0, 'data': data[off:]})
            break   # klatka Intra to cała reszta bufora
        else:
            x, y, dx, dy, bh, bw = struct.unpack_from('>HHhhHH', data, off); off += 12
            size = bh * bw * 3 * 2   # float16, 3 kanały
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
                self.f.write(struct.pack('>I', len(comp)))
                self.f.write(comp)
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
            raise ValueError(f"Zły format pliku (magic={m!r}, oczekiwano PH02)")
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

def encode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic] Inicjalizacja pola falowego (YCbCr)...", flush=True)

    reader = iio.imiter(inp, plugin='pyav')
    first  = next(reader)
    h, w   = first.shape[:2]

    codec = PhotonicCodec(
        brush_size=args.brush_size,
        energy_threshold=args.ncc_threshold,
        delta_threshold=args.delta_threshold,
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
    print("[Photonic] Odtwarzanie z interferencji (kolor)...", flush=True)

    with photonic_reader(inp) as reader:
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)

        codec = PhotonicCodec(brush_size=args.brush_size)
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
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.1")
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')
    parser.add_argument('--brush-size',      type=int,   default=64,
                        help="Rozmiar bloku korelacji (domyślnie 64)")
    parser.add_argument('--fps',             type=float, default=25.0)
    parser.add_argument('--ncc-threshold',   type=float, default=0.85,
                        help="NCC ≥ thr → ścieżka ruchu (domyślnie 0.85)")
    parser.add_argument('--delta-threshold', type=float, default=5.0,
                        help="Próg szumu delty w YCbCr (domyślnie 5.0)")
    args = parser.parse_args()

    if args.decode:
        decode_video(args.input, args.output, args)
    else:
        encode_video(args.input, args.output, args)
