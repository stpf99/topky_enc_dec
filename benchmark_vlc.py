"""
Benchmark VLC + Zstd — uruchom: python benchmark_vlc.py
Wymaga: numpy, scipy, zstandard, toptopuw_vlc.py w tym samym katalogu.
"""
import sys, os, struct, time
import numpy as np
from scipy.fftpack import dct

# Szukaj toptopuw_vlc.py obok skryptu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toptopuw_vlc as vlc

def dct2(b): return dct(dct(b.T, norm='ortho').T, norm='ortho')

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("⚠  zstandard niedostępne — tylko benchmark VLC bez Zstd")

# ─── Parametry symulacji ──────────────────────────────────────────────────────
SEED       = 42
Q_Y, Q_C   = 22.0, 40.0
COLS, ROWS = 120, 68       # 1920×1080 w blokach 16×16
DETAIL_PCT = 0.30          # 30% bloków DETAIL (typowe P-frame)
REPS       = 10            # powtórzenia dla pomiaru czasu
ZSTD_LVL   = 22

rng = np.random.default_rng(SEED)
n_bl     = COLS * ROWS
n_detail = int(n_bl * DETAIL_PCT)

# ─── Buduj surowy frame P w formacie v2.7 ────────────────────────────────────
bm = bytearray((n_bl + 7) // 8)
det_idx = sorted(rng.choice(n_bl, n_detail, replace=False))
for idx in det_idx:
    bm[idx >> 3] |= (1 << (7 - (idx & 7)))

buf = bytearray()
buf.extend(struct.pack('>HH', COLS, ROWS))
buf.extend(bm)
for _ in range(n_detail):
    dx = int(rng.integers(-96, 97))
    dy = int(rng.integers(-96, 97))
    buf.extend(struct.pack('>hh', dx, dy))
    qy = np.round(dct2(rng.normal(0, 8, (16,16)).astype(np.float32)) / Q_Y).astype(np.int16)
    qu = np.round(dct2(rng.normal(0, 4,  (8, 8)).astype(np.float32)) / Q_C).astype(np.int16)
    qv = np.round(dct2(rng.normal(0, 4,  (8, 8)).astype(np.float32)) / Q_C).astype(np.int16)
    buf.extend(qy.flatten().tobytes())
    buf.extend(qu.flatten().tobytes())
    buf.extend(qv.flatten().tobytes())

raw = bytes(buf)

# ─── Warm-up ─────────────────────────────────────────────────────────────────
vlc_data = vlc.pack_pframe(raw)
assert vlc.unpack_pframe(vlc_data) == raw, "ROUNDTRIP FAIL — sprawdź toptopuw_vlc.py"

if HAS_ZSTD:
    cctx = zstd.ZstdCompressor(level=ZSTD_LVL)
    dctx = zstd.ZstdDecompressor()
    z_raw = cctx.compress(raw)
    z_vlc = cctx.compress(vlc_data)
    dctx.decompress(z_raw)
    dctx.decompress(z_vlc)

# ─── Pomiary ─────────────────────────────────────────────────────────────────
def bench(fn, reps=REPS):
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1000

t_pack   = bench(lambda: vlc.pack_pframe(raw))
t_unpack = bench(lambda: vlc.unpack_pframe(vlc_data))

if HAS_ZSTD:
    t_zc_raw = bench(lambda: cctx.compress(raw))
    t_zc_vlc = bench(lambda: cctx.compress(vlc_data))
    t_zd_raw = bench(lambda: dctx.decompress(z_raw))
    t_zd_vlc = bench(lambda: dctx.decompress(z_vlc))

# ─── Rozkład symboli ─────────────────────────────────────────────────────────
from collections import Counter
sym_cnt   = Counter(vlc_data)
total_sym = len(vlc_data)

# ─── Raport ──────────────────────────────────────────────────────────────────
W = 68
print(f"╔{'═'*W}╗")
print(f"║  BENCHMARK VLC + ZSTD — klatka P 1080p{' '*28}║")
print(f"║  {COLS}×{ROWS} bloków = {n_bl} total, {n_detail} DETAIL ({DETAIL_PCT*100:.0f}%), Q_Y={Q_Y:.0f} Q_C={Q_C:.0f}{' '*6}║")
print(f"╠{'═'*W}╣")

def row(label, val, pct=None, extra=""):
    s = f"  {label:<30s} {val:>12s}"
    if pct: s += f"  {pct}"
    if extra: s += f"  {extra}"
    print(f"║{s:<{W}}║")

print(f"║  {'─── Rozmiary ───':<{W-2}}║")
row("Raw int16 (v2.7):",      f"{len(raw):>10,} B",  "  100%")
row("VLC:",                   f"{len(vlc_data):>10,} B",
    f"  {100*len(vlc_data)/len(raw):4.1f}%",
    f"→ {len(raw)/len(vlc_data):.1f}× mniej")
if HAS_ZSTD:
    row("Raw + Zstd(22):",    f"{len(z_raw):>10,} B",
        f"  {100*len(z_raw)/len(raw):4.1f}%",
        f"→ {len(raw)/len(z_raw):.1f}× mniej")
    row("VLC + Zstd(22):",    f"{len(z_vlc):>10,} B",
        f"  {100*len(z_vlc)/len(raw):4.1f}%",
        f"→ {len(raw)/len(z_vlc):.1f}× mniej")
    row("VLC+Zstd vs Zstd:",  f"{len(z_raw)/len(z_vlc):.2f}×",
        extra=f"({100*(1-len(z_vlc)/len(z_raw)):.0f}% mniej niż samo Zstd)")

print(f"║  {'─── Czasy (ms, średnia {REPS} powtórzeń) ─── ':<{W-2}}║")
row("VLC pack:",              f"{t_pack:>7.2f} ms")
row("VLC unpack:",            f"{t_unpack:>7.2f} ms")
if HAS_ZSTD:
    row("Zstd compress raw:", f"{t_zc_raw:>7.2f} ms")
    row("Zstd compress VLC:", f"{t_zc_vlc:>7.2f} ms")
    row("Zstd decomp raw:",   f"{t_zd_raw:>7.2f} ms")
    row("Zstd decomp VLC:",   f"{t_zd_vlc:>7.2f} ms")

if HAS_ZSTD:
    print(f"║  {'─── Łączny pipeline ───':<{W-2}}║")
    row("Enkodowanie  raw→Zstd:",
        f"{t_zc_raw:>7.2f} ms")
    row("Enkodowanie  raw→VLC→Zstd:",
        f"{t_pack+t_zc_vlc:>7.2f} ms",
        extra=f"(VLC {t_pack:.2f} + Zstd {t_zc_vlc:.2f})")
    row("Dekodowanie  Zstd→raw:",
        f"{t_zd_raw:>7.2f} ms")
    row("Dekodowanie  Zstd→VLC→raw:",
        f"{t_zd_vlc+t_unpack:>7.2f} ms",
        extra=f"(Zstd {t_zd_vlc:.2f} + VLC {t_unpack:.2f})")

print(f"║  {'─── Rozkład symboli VLC ───':<{W-2}}║")
row("Unikalne bajty:", f"{len(sym_cnt)} / 256")
for b, c in sym_cnt.most_common(10):
    sym = vlc._B2SYM[b]
    if sym == vlc._EOB:          lbl = "EOB"
    elif sym == vlc._ESC_RUN:    lbl = "ESC_RUN"
    elif isinstance(sym, tuple): lbl = f"({sym[0]},{sym[1]:+d})"
    elif b == vlc.ESCAPE_LIT:    lbl = "ESC_LIT"
    else:                        lbl = f"0x{b:02x}"
    row(f"  {b:#04x} {lbl}", f"{100*c/total_sym:5.1f}%")

print(f"╚{'═'*W}╝")
