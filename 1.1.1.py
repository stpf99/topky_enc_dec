#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC VIDEO CODEC v1.1.1 — CZYSTA RÓŻNICA TEMPORALNA (BEZ FFT)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Modyfikacja:                                                                ║
║  - Całkowite usunięcie shift_block, optical_correlate, interference_delta   ║
║  - Zachowanie oryginalnego formatu float16 z v1.1 (brak błędów dekodowania)  ║
║  - Odchudzenie nagłówka bloku z 12 do 8 bajtów (brak dx, dy w strumieniu)   ║
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

_MAGIC      = b'PH22'   # Nowy identyfikator formatu bez wektorów ruchu
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
# ARCHITEKTURA KODEKA
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicCodec:
    def __init__(self, brush_size: int = 64, delta_threshold: float = 5.0):
        self.bs         = brush_size
        self.delta_thr  = delta_threshold    # |delta| ≤ thr → szum, ignoruj
        self.prev_wave  = None               # (H, W, 3) float32 YCbCr

    def _split_to_brushes(self, wave: np.ndarray):
        h, w = wave.shape[:2]
        for y in range(0, h, self.bs):
            for x in range(0, w, self.bs):
                yield x, y, wave[y:y+self.bs, x:x+self.bs]

    def encode_frame(self, frame_rgb: np.ndarray) -> list:
        wave3 = rgb_to_ycbcr(frame_rgb)
        paths = []

        if self.prev_wave is None:
            # Pierwska klatka kluczowa (Intra)
            paths.append({'type': 0, 'data': wave3.tobytes()})
            self.prev_wave = wave3
            return paths

        for x, y, brush in self._split_to_brushes(wave3):
            bh, bw     = brush.shape[:2]
            prev_brush = self.prev_wave[y:y+bh, x:x+bw]

            # Bezpośrednie porównanie klatka do klatki (czysta różnica)
            delta = brush - prev_brush
            delta_mask = np.max(np.abs(delta), axis=-1) > self.delta_thr
            
            if np.any(delta_mask):
                sparse_delta = delta * delta_mask[..., np.newaxis]
                paths.append({
                    'type': 1,
                    'x': x, 'y': y, 'bh': bh, 'bw': bw,
                    'data': sparse_delta.astype(np.float16).tobytes()
                })

        self.prev_wave = wave3
        return paths

    def decode_frame(self, paths: list, h: int, w: int) -> np.ndarray:
        if self.prev_wave is None:
            self.prev_wave = (
                np.frombuffer(paths[0]['data'], dtype=np.float32).reshape(h, w, 3)
            )
            return ycbcr_to_rgb(self.prev_wave)

        new_wave = self.prev_wave.copy()

        for path in paths:
            x, y   = path['x'], path['y']
            bh, bw = path['bh'], path['bw']

            if path['type'] == 1:   # Aplikacja różnicy temporalnej
                delta  = (np.frombuffer(path['data'], dtype=np.float16)
                           .reshape(bh, bw, 3).astype(np.float32))
                new_wave[y:y+bh, x:x+bw] = self.prev_wave[y:y+bh, x:x+bw] + delta

        self.prev_wave = new_wave
        return ycbcr_to_rgb(new_wave)

# ═══════════════════════════════════════════════════════════════════════════════
# STRUMIENIE I/O (Zsynchronizowane z oryginalnym float16 z v1.1)
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_paths(paths: list) -> bytes:
    out = bytearray()
    out.extend(struct.pack('>H', len(paths)))
    for p in paths:
        out.append(p['type'])
        if p['type'] == 0:
            out.extend(p['data'])
        else:
            # Usunięto dx, dy — nagłówek skrócony z 12 do 8 bajtów
            out.extend(struct.pack('>HHHH', p['x'], p['y'], p['bh'], p['bw']))
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
            break   
        else:
            # Czytamy tylko x, y, bh, bw (8 bajtów)
            x, y, bh, bw = struct.unpack_from('>HHHH', data, off); off += 8
            size = bh * bw * 3 * 2   # Stabilne wyliczenie rozmiaru float16 (3 kanały, 2 bajty)
            paths.append({
                'type': t, 'x': x, 'y': y,
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
            raise ValueError(f"Zły format pliku (magic={m!r}, oczekiwano {_MAGIC.decode()})")
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
    print("[Photonic v1.1.1] Inicjalizacja kompresji różnicowej (Bez FFT)...", flush=True)

    reader = iio.imiter(inp, plugin='pyav')
    first  = next(reader)
    h, w   = first.shape[:2]

    codec = PhotonicCodec(
        brush_size=args.brush_size,
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
                      f" | Zmienionych bloków: {len(paths):3d}", end="", flush=True)
            fi += 1

    print(f"\n[Photonic] Zapisano {fi} klatek.", flush=True)


def decode_video(inp, out, args):
    if not iio: return print("Brak imageio")
    print("[Photonic v1.1.1] Odtwarzanie klatek z różnic temporalnych...", flush=True)

    with photonic_reader(inp) as reader:
        w_out = iio.imopen(out, 'w', plugin='pyav')
        w_out.init_video_stream('libx264', fps=reader.fps)

        codec = PhotonicCodec(brush_size=args.brush_size)
        fi = 0; t = time.time()

        for paths in iter(reader.read, None):
            rgb = codec.decode_frame(paths, reader.h, reader.w)
            w_out.write_frame(rgb)
            if fi % 10 == 0:
                elapsed = time.time() - t
                print(f"\r[Photonic] Klatka: {fi:4d} | {fi/elapsed:4.1f} FPS", end="", flush=True)
            fi += 1

        w_out.close()
    print(f"\n[Photonic] Odtworzono {fi} klatek.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Photonic Video Codec v1.1.1 (No-FFT)")
    parser.add_argument('-i', '--input',   required=True)
    parser.add_argument('-o', '--output',  required=True)
    parser.add_argument('-d', '--decode',  action='store_true')
    parser.add_argument('--brush-size',      type=int,   default=64,
                        help="Rozmiar bloku (domyślnie 64)")
    parser.add_argument('--fps',             type=float, default=25.0)
    parser.add_argument('--delta-threshold', type=float, default=5.0,
                        help="Próg szumu delty w YCbCr (domyślnie 5.0)")
    args = parser.parse_args()

    if args.decode:
        decode_video(args.input, args.output, args)
    else:
        encode_video(args.input, args.output, args)