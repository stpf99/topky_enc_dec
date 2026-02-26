
```shell
tom@tom-solus ~ $ python benchmark_vlc.py 
╔════════════════════════════════════════════════════════════════════╗
║  BENCHMARK VLC + ZSTD — klatka P 1080p                            ║
║  120×68 bloków = 8160 total, 2448 DETAIL (30%), Q_Y=22 Q_C=40      ║
╠════════════════════════════════════════════════════════════════════╣
║  ─── Rozmiary ───                                                  ║
║  Raw int16 (v2.7):               1,890,880 B    100%               ║
║  VLC:                              125,301 B     6.6%  → 15.1× mniej║
║  Raw + Zstd(22):                   105,236 B     5.6%  → 18.0× mniej║
║  VLC + Zstd(22):                    77,656 B     4.1%  → 24.3× mniej║
║  VLC+Zstd vs Zstd:                     1.36×  (26% mniej niż samo Zstd)║
║  ─── Czasy (ms, średnia {REPS} powtórzeń) ───                      ║
║  VLC pack:                         295.72 ms                       ║
║  VLC unpack:                        64.68 ms                       ║
║  Zstd compress raw:               1037.10 ms                       ║
║  Zstd compress VLC:                 31.03 ms                       ║
║  Zstd decomp raw:                    1.05 ms                       ║
║  Zstd decomp VLC:                    0.15 ms                       ║
║  ─── Łączny pipeline ───                                           ║
║  Enkodowanie  raw→Zstd:           1037.10 ms                       ║
║  Enkodowanie  raw→VLC→Zstd:        326.75 ms  (VLC 295.72 + Zstd 31.03)║
║  Dekodowanie  Zstd→raw:              1.05 ms                       ║
║  Dekodowanie  Zstd→VLC→raw:         64.83 ms  (Zstd 0.15 + VLC 64.68)║
║  ─── Rozkład symboli VLC ───                                       ║
║  Unikalne bajty:                   242 / 256                       ║
║    0x02 (0,+1)                          7.8%                       ║
║    0x03 (0,-1)                          7.7%                       ║
║    0x09 (1,-1)                          6.5%                       ║
║    0x08 (1,+1)                          6.5%                       ║
║    0x00 EOB                             5.6%                       ║
║    0x0f (2,-1)                          5.3%                       ║
║    0x0e (2,+1)                          5.3%                       ║
║    0x01 ESC_RUN                         5.3%                       ║
║    0x15 (3,-1)                          4.4%                       ║
║    0x14 (3,+1)                          4.4%                       ║



qdiff_codec.py "qbit style" 192 klatki

  Kompresja zstd...

✓ SUKCES!
  Klatki: 192  |  Raw: 518400 KB
  Pre-zstd: 162437 KB  |  Po zstd: 5911 KB
  Kompresja: 87.7× (5911.2 KB)


qdiff_codec_v0.2_auto.py 30 klatek

  Kompresja zstd...

✓ SUKCES!
  Klatki: 30  |  Raw: 81000 KB
  Pre-zstd: 18677 KB  |  Po zstd: 285 KB
  Kompresja: 284.1× (285.1 KB)

  Statystyki AUTO-MODE:
    Średnie Q_Y: 20.1 (zakres: 12.0-45.0)


qdiff_codec_v0.3_vfr.py 30 klatek
  Kompresja zstd...

✓ SUKCES!
  Klatki: 30 (kept: 30, dropped: 0)
  Raw: 81000 KB
  Pre-zstd: 18242 KB  |  Po zstd: 283 KB
  Kompresja: 285.5× (283.7 KB)

    Średnie Q_C: 40.2 (zakres: 22.0-70.0)
    Średnia MAD: 2.36

