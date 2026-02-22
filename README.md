'tom@tom-solus ~/Pobrane $ python toptopuwcodec_v1.py -i video.mp4 -o video_top1-p1-f.test -1 -f
▶ Preset 1 — MAŁY PLIK: Q_Y=32 Q_C=55

╔══ TOP TOPÓW CODEC v2 — KODOWANIE ══╗
  Wejście:     video.mp4
  Wyjście:     video_top1-p1-f.test
  Klatki:      WSZYSTKIE
  Sub-pixel:   ON
  B-Frames:    ON
  Zasięg:      ±24px
  Keyframe co: 50 klatek
  Q_Y/Q_C:     32/55
  Scene cut:   próg MAD=35
╚══════════════════════════════════════╝

[Klatka 1/145] I-Frame...
[Klatka 2/145] P-Frame...
   -> SKIP: 1188/1225 (97.0%) | DETAIL: 37 | Sub-pixel: ON
[Klatka 3/145] P-Frame...
   -> SKIP: 1167/1225 (95.3%) | DETAIL: 58 | Sub-pixel: ON
[Klatka 4/145] P-Frame...
   -> SKIP: 1156/1225 (94.4%) | DETAIL: 69 | Sub-pixel: ON
[Klatka 5/145] B-Frame (future=6)...

............................................................

[Klatka 141/145] B-Frame (future=142)...
   -> B-FRAME | SKIP: 1209/1225 (98.7%) | DETAIL: 16
[Klatka 142/145] P-Frame...
   -> SKIP: 1204/1225 (98.3%) | DETAIL: 21 | Sub-pixel: ON
[Klatka 143/145] P-Frame...
   -> SKIP: 1205/1225 (98.4%) | DETAIL: 20 | Sub-pixel: ON
[Klatka 144/145] P-Frame...
   -> SKIP: 1207/1225 (98.5%) | DETAIL: 18 | Sub-pixel: ON
[Klatka 145/145] P-Frame...
   -> SKIP: 1205/1225 (98.4%) | DETAIL: 20 | Sub-pixel: ON

✓ Czas: 200.8s | Surowe: 16852KB | Skompresowane: 139KB | Ratio Zstd: 120.4x
tom@tom-solus ~/Pobrane $ python toptopuwcodec_v1.py -o 1-video.mp4 -i video_top1-p1-f.test -d

╔══ TOP TOPÓW CODEC v2 — DEKODOWANIE ══╗
  Wejście: video_top1-p1-f.test
  Wyjście: 1-video.mp4
╚════════════════════════════════════════╝

  Zdekodowano klatkę 1/145 [I]
  Zdekodowano klatkę 2/145 [P]
  Zdekodowano klatkę 3/145 [P]
  Zdekodowano klatkę 4/145 [P]
......
  Zdekodowano klatkę 145/145 [P]

✓ SUKCES! 145 klatek → 1-video.mp4
tom@tom-solus ~/Pobrane $ 
'
