"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TOP TOPÓW CODEC — MODUŁ VLC (Variable-Length Coding)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Kompaktuje bitstream PRZED kompresją Zstd przez zastąpienie int16 (2B)     ║
║  na współczynniku DCT pojedynczym bajtem VLC.                               ║
║                                                                              ║
║  Idea: jak zapis RLE na taśmie, ale zamiast (count, value) pary             ║
║  kodujemy (run_of_zeros, level) jako jeden bajt z tablicy 92 symboli.       ║
║                                                                              ║
║  DLACZEGO TO DZIAŁA:                                                         ║
║  • Q_Y=22, Q_C=40 → po kwantyzacji 83% współczynników = 0                  ║
║  • 94% niezerowych ma |level| = 1                                            ║
║  • Alfabet (run=0..14, level=±1..±3) + EOB + escape = 92 symbole            ║
║  • Wartości poza zakresem → bajt 0xFE + raw int16 (2B)                      ║
║                                                                              ║
║  WYNIKI (symulacja Q_Y=22, residual N(0,8)):                                 ║
║  • Przed: 512B / blok (256× int16)                                           ║
║  • Po VLC: ~27B / blok (19× mniej)                                           ║
║  • Entropia symbolowa: 4.6 bits/sym → potencjał ~18× przed Zstd             ║
║  • Po Zstd na VLC: ещё ~1.2-1.5× (Zstd kocha małe alfabet)                 ║
║                                                                              ║
║  WEKTORY RUCHU (mv_dx, mv_dy):                                               ║
║  • search_range=24, quarter-pixel → zakres ±96 → mieści się w int8          ║
║  • 4B int16×2 → 2B int8×2 (2× oszczędność)                                 ║
║                                                                              ║
║  UŻYCIE:                                                                     ║
║      import toptopuw_vlc as vlc                                              ║
║      packed   = vlc.pack_frame(raw_frame_bytes)                              ║
║      unpacked = vlc.unpack_frame(packed)                                     ║
║      # assert unpacked == raw_frame_bytes                                    ║
║                                                                              ║
║  INTEGRACJA z kodekiem (monkey-patch):                                       ║
║      vlc.apply(codec_mod)  # podmienia _serialize_blocks/_deserialize_blocks ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import struct
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ZIGZAG SCAN 16×16  i  8×8
# ═══════════════════════════════════════════════════════════════════════════════

def _build_zigzag(n: int) -> list:
    """Zwraca listę (row, col) w kolejności zigzag dla bloku n×n."""
    idx = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = min(s, n - 1);  c = s - r
            while r >= 0 and c < n:
                idx.append((r, c));  r -= 1;  c += 1
        else:
            c = min(s, n - 1);  r = s - c
            while c >= 0 and r < n:
                idx.append((r, c));  r += 1;  c -= 1
    return idx


_ZZ16 = _build_zigzag(16)   # 256 par (row, col)
_ZZ8  = _build_zigzag(8)    # 64  par (row, col)

# Odwrócony zigzag: pozycja sekwencyjna → (row, col)
_ZZ16_R = [(r, c) for r, c in _ZZ16]
_ZZ8_R  = [(r, c) for r, c in _ZZ8]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABLICA VLC  (symbol → bajt,  bajt → symbol)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Symbol = (run, level) gdzie:
#   run   = 0..14  (liczba zer przed tym współczynnikiem)
#   level = ±1, ±2, ±3  (wartość niezerowa; pokrywa 99%+ przypadków)
#
# Specjalne symbole:
#   EOB         = (-1, 0)   bajt 0x00 — koniec bloku (trailing zeros pominięte)
#   ESCAPE_RUN  = (15, 0)   bajt 0x01 — ciąg 15 zer bez wartości (kontynuuj run)
#   ESCAPE_LIT  = 0xFE      — następne 2B to raw int16 (|level| > 3 lub run > 14)
#
# Łącznie 92 symbole (bajty 0x00..0x5B), 0x5C..0xFD wolne (przyszłe użycie),
# 0xFE = escape literal, 0xFF zarezerwowany.
# ───────────────────────────────────────────────────────────────────────────────

_RUNS   = list(range(15))          # 0..14
_LEVELS = [1, -1, 2, -2, 3, -3]   # posortowane po częstości

_SYM2B: dict  = {}   # (run, level) → int bajt
_B2SYM: list  = [None] * 256  # int bajt → (run, level) | None

_EOB        = (-1, 0)
_ESC_RUN    = (15, 0)
ESCAPE_LIT  = 0xFE   # następne 2B = raw int16

_SYM2B[_EOB]     = 0x00;  _B2SYM[0x00] = _EOB
_SYM2B[_ESC_RUN] = 0x01;  _B2SYM[0x01] = _ESC_RUN

_idx = 2
for _r in _RUNS:
    for _lv in _LEVELS:
        _SYM2B[(_r, _lv)] = _idx
        _B2SYM[_idx] = (_r, _lv)
        _idx += 1

# Sanity check
assert _idx == 92, f"Oczekiwano 92 symboli, mamy {_idx}"
assert _B2SYM[ESCAPE_LIT] is None, "0xFE musi być wolny"

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENKODER BLOKU  (int16 array → bytes VLC)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_block(block_int16: np.ndarray, zz: list) -> bytes:
    """
    Koduje jeden blok (16×16 lub 8×8) int16 jako strumień bajtów VLC.

    Algorytm:
    1. Zigzag scan → sekwencja współczynników
    2. Odetnij trailing zeros (EOB)
    3. Dla każdego współczynnika:
       a. Jeśli 0 → inkrement run (gdy run==14 → emituj ESCAPE_RUN)
       b. Jeśli |level|≤3 i (run,level) ∈ tablicy → emituj 1B VLC
       c. Inaczej → emituj ESCAPE_LIT + 2B raw int16 (+ opcjonalnie
          najpierw jeśli run>0: emituj run jako ciąg ESCAPE_RUN + reszta)
    4. Emituj EOB (0x00)
    """
    coeffs = [int(block_int16[r, c]) for r, c in zz]

    # Odetnij trailing zeros — nie ma sensu ich kodować
    last_nz = len(coeffs) - 1
    while last_nz > 0 and coeffs[last_nz] == 0:
        last_nz -= 1
    coeffs = coeffs[:last_nz + 1]

    out = bytearray()
    run = 0

    for v in coeffs:
        if v == 0:
            run += 1
            if run == 15:
                # Maksymalny run → emituj (15,0) i zresetuj
                out.append(_SYM2B[_ESC_RUN])
                run = 0
        else:
            # Mamy wartość niezerową — najpierw obsłuż run
            level = v

            sym = (run, level)
            if sym in _SYM2B:
                # Szczęśliwy przypadek: 1 bajt
                out.append(_SYM2B[sym])
            else:
                # Escape: emituj pending run jako ciąg ESCAPE_RUN (po 15 zer)
                # i pojedyncze zera jako (run % 15) bloków + jeden ESCAPE_LIT
                full_runs = run // 15
                rem_run   = run % 15
                for _ in range(full_runs):
                    out.append(_SYM2B[_ESC_RUN])
                # Teraz emitujemy (rem_run, level) przez escape
                out.append(ESCAPE_LIT)
                # Pakujemy run (4 bity) + sign (1 bit) + mag (11 bitów) w 2B
                # Prościej: 1B dla rem_run, 2B dla raw int16 level
                out.append(rem_run & 0xFF)
                out.extend(struct.pack('>h', level))

            run = 0

    # EOB tylko jeśli niezerowe nie kończą się dokładnie na ostatniej pozycji zigzag.
    # Gdy ostatni niezerowy = pozycja len(zz)-1, po jego odczycie seq_idx = len(zz)
    # → while seq_idx < total kończy się samo → dekoder NIE przeczyta EOB.
    # W każdym innym przypadku (trailing zeros lub seq_idx < total) EOB jest potrzebny.
    #
    # Uproszczone: zawsze emituj EOB jeśli seq_idx < len(zz) przed EOB.
    # "seq_idx przed EOB" = total_emitted (pozycja tuż po ostatnim niezerowym + run)
    # Ale łatwiej: po pętli `run` może być > 0 tylko jeśli trailing zeros...
    # ale już je odcięliśmy. Po odcięciu trailing zeros: seq_idx po ostatnim
    # niezerowym = last_nz_pos + 1. Jeśli last_nz_pos + 1 < len(zz) → emituj EOB.
    # Jeśli last_nz_pos + 1 == len(zz) → nie emituj (dekoder skończy pętlę sam).
    #
    # Implementacja: oblicz pozycję ostatniego niezerowego w zigzag
    if coeffs:  # coeffs już przycięte do last_nz+1
        last_nz_in_zz = len(coeffs) - 1   # indeks w zigzag ostatniego niezerowego
        if last_nz_in_zz + 1 < len(zz):
            out.append(_SYM2B[_EOB])   # EOB potrzebny
        # else: dekoder wyjdzie z pętli gdy seq_idx == total — EOB zbędny
    else:
        # Pusty blok (same zera) — emituj tylko EOB
        out.append(_SYM2B[_EOB])
    return bytes(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEKODER BLOKU  (bytes VLC → int16 array)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_block(data: bytes, offset: int, zz: list, n: int) -> tuple:
    """
    Dekoduje jeden blok VLC z data[offset:] → (ndarray int16 n×n, nowy_offset).
    """
    block = np.zeros((n, n), dtype=np.int16)
    seq_idx = 0   # pozycja w ciągu zigzag
    total = n * n

    while seq_idx < total:
        b = data[offset]; offset += 1

        if b == _SYM2B[_EOB]:
            # EOB — reszta bloku to zera
            break

        sym = _B2SYM[b]

        if sym is not None:
            run, level = sym
            if sym == _ESC_RUN:
                # 15 zer — przesuń seq_idx
                seq_idx += 15
            else:
                seq_idx += run
                if seq_idx < total:
                    r, c = zz[seq_idx]
                    block[r, c] = level
                    seq_idx += 1
        elif b == ESCAPE_LIT:
            rem_run = data[offset]; offset += 1
            level,  = struct.unpack_from('>h', data, offset); offset += 2
            seq_idx += rem_run
            if seq_idx < total:
                r, c = zz[seq_idx]
                block[r, c] = level
                seq_idx += 1
        else:
            raise ValueError(f"Nieznany bajt VLC: {b:#04x} @ offset {offset-1}")

    return block, offset


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PAKOWANIE / ODPAKOWANIE WEKTORÓW RUCHU
# ═══════════════════════════════════════════════════════════════════════════════
#
# Oryginał: 2× int16 = 4B  (zakres ±32767)
# search_range=24, quarter-pixel → faktyczny zakres ±96 → int8 ±127
# Nowy format: 2× int8 = 2B (oszczędność 2B/blok)
#
# Zabezpieczenie: jeśli |mv| > 127 → emituj flag bajt 0xFF + 4B int16×2

_MV_ESCAPE = 0x7F  # Sentinel: mv_dx lub mv_dy = ±127 oznacza wartość poza zakresem

def _pack_mv(dx: int, dy: int) -> bytes:
    """Pakuje (dx, dy) quarter-pixel do 2B (int8×2) lub 5B (flag + 4B raw)."""
    if -126 <= dx <= 126 and -126 <= dy <= 126:
        return struct.pack('bb', dx, dy)
    else:
        # Rzadki przypadek: large motion vector → 1B flag + 4B raw
        return struct.pack('b', -127) + struct.pack('>hh', dx, dy)


def _unpack_mv(data: bytes, offset: int) -> tuple:
    """Zwraca ((dx, dy), nowy_offset)."""
    dx = struct.unpack_from('b', data, offset)[0]; offset += 1
    if dx == -127:
        # Escape: następne 4B to raw int16×2
        dx, dy = struct.unpack_from('>hh', data, offset); offset += 4
    else:
        dy = struct.unpack_from('b', data, offset)[0]; offset += 1
    return (dx, dy), offset


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PAKOWANIE CAŁEJ KLATKI P/B
# ═══════════════════════════════════════════════════════════════════════════════
#
# Format VLC klatki P/B (zastępuje oryginalny format v2.7 dla danych DETAIL):
#   [2B: cols uint16]
#   [2B: rows uint16]
#   [ceil(rows*cols/8) B: bitmapa SKIP/DETAIL — bez zmian]
#   Dla każdego bloku DETAIL:
#     [2B lub 5B: mv pakowany]
#     [N1 B: VLC blok Y  (16×16)]
#     [N2 B: VLC blok U  (8×8)]
#     [N3 B: VLC blok V  (8×8)]
#   [2B: 0xABCD magic end-of-frame]
#
# Magic na końcu umożliwia walidację i łatwy skip do następnej klatki.

_EOF_MAGIC = struct.pack('>H', 0xABCD)
_BS   = 16
_BS_C = 8


def pack_pframe(raw_bytes: bytes) -> bytes:
    """
    Kompaktuje surowy format v2.7 klatki P/B do formatu VLC.
    raw_bytes to wyjście _serialize_blocks().
    Zwraca skompaktowany bufor.
    """
    if len(raw_bytes) < 4:
        return raw_bytes

    offset = 0
    cols, rows = struct.unpack_from('>HH', raw_bytes, offset); offset += 4

    if cols == 0 or rows == 0:
        return raw_bytes   # Pusta klatka — nie ma co pakować

    n_blocks   = rows * cols
    bitmap_sz  = (n_blocks + 7) // 8
    bitmap     = raw_bytes[offset:offset + bitmap_sz]; offset += bitmap_sz

    sz_y = _BS   * _BS   * 2   # 512 B
    sz_c = _BS_C * _BS_C * 2   # 128 B

    out = bytearray()
    out.extend(struct.pack('>HH', cols, rows))
    out.extend(bitmap)

    for bit_idx in range(n_blocks):
        is_detail = bool(bitmap[bit_idx >> 3] & (1 << (7 - (bit_idx & 7))))
        if not is_detail:
            continue

        # Wektor ruchu
        dx, dy = struct.unpack_from('>hh', raw_bytes, offset); offset += 4
        out.extend(_pack_mv(dx, dy))

        # Blok Y (16×16 int16)
        blk_y = np.frombuffer(raw_bytes[offset:offset+sz_y],
                              dtype=np.int16).reshape(_BS, _BS).copy()
        offset += sz_y
        out.extend(_encode_block(blk_y, _ZZ16))

        # Blok U (8×8 int16)
        blk_u = np.frombuffer(raw_bytes[offset:offset+sz_c],
                              dtype=np.int16).reshape(_BS_C, _BS_C).copy()
        offset += sz_c
        out.extend(_encode_block(blk_u, _ZZ8))

        # Blok V (8×8 int16)
        blk_v = np.frombuffer(raw_bytes[offset:offset+sz_c],
                              dtype=np.int16).reshape(_BS_C, _BS_C).copy()
        offset += sz_c
        out.extend(_encode_block(blk_v, _ZZ8))

    out.extend(_EOF_MAGIC)
    return bytes(out)


def unpack_pframe(vlc_bytes: bytes) -> bytes:
    """
    Odwraca pack_pframe() → oryginalny format v2.7 (_deserialize_blocks kompatybilny).
    """
    if len(vlc_bytes) < 4:
        return vlc_bytes

    offset = 0
    cols, rows = struct.unpack_from('>HH', vlc_bytes, offset); offset += 4

    if cols == 0 or rows == 0:
        return vlc_bytes

    n_blocks  = rows * cols
    bitmap_sz = (n_blocks + 7) // 8
    bitmap    = vlc_bytes[offset:offset + bitmap_sz]; offset += bitmap_sz

    out = bytearray()
    out.extend(struct.pack('>HH', cols, rows))
    out.extend(bitmap)

    for bit_idx in range(n_blocks):
        is_detail = bool(bitmap[bit_idx >> 3] & (1 << (7 - (bit_idx & 7))))
        if not is_detail:
            continue

        # Wektor ruchu
        (dx, dy), offset = _unpack_mv(vlc_bytes, offset)
        out.extend(struct.pack('>hh', dx, dy))

        # Blok Y
        blk_y, offset = _decode_block(vlc_bytes, offset, _ZZ16R := _ZZ16_R, _BS)
        out.extend(blk_y.astype(np.int16).flatten().tobytes())

        # Blok U
        blk_u, offset = _decode_block(vlc_bytes, offset, _ZZ8_R, _BS_C)
        out.extend(blk_u.astype(np.int16).flatten().tobytes())

        # Blok V
        blk_v, offset = _decode_block(vlc_bytes, offset, _ZZ8_R, _BS_C)
        out.extend(blk_v.astype(np.int16).flatten().tobytes())

    # Sprawdź magic (opcjonalne — nie rzucaj wyjątku, tylko ostrzeż)
    if vlc_bytes[offset:offset+2] != _EOF_MAGIC:
        import sys
        print(f"[vlc] OSTRZEŻENIE: brak magic 0xABCD @ offset {offset}", file=sys.stderr)

    return bytes(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MONKEY-PATCH KODEKA
# ═══════════════════════════════════════════════════════════════════════════════

def apply(codec_mod) -> object:
    """
    Wstrzykuje VLC do załadowanego modułu kodeka.
    Podmienia _serialize_blocks i _deserialize_blocks.

    Nowy format pliku .toptop = identyczny z v2.7 z wyjątkiem tego że
    dane klatki P/B są pakowane przez VLC zamiast raw int16.
    Klatki I nie są zmieniane (już są int16 całej płaszczyzny, Zstd
    świetnie je kompresuje — mało sensu VLC dla DC/AC całego obrazu).
    """
    _orig_serialize   = codec_mod._serialize_blocks
    _orig_deserialize = codec_mod._deserialize_blocks

    def _vlc_serialize(blocks, is_bframe=False):
        raw = _orig_serialize(blocks, is_bframe)
        return pack_pframe(raw)

    def _vlc_deserialize(data, offset, is_bframe=False):
        # Musimmy znaleźć koniec VLC klatki — czytamy do magic 0xABCD
        # Najpierw sprawdzamy czy to format VLC (czy jest magic na końcu)
        # przez próbę odczytu i weryfikację
        #
        # Problem: nie znamy rozmiaru VLC z góry — szukamy magic.
        # Rozwiązanie: odczytujemy VLC strumieniowo, magic jest na końcu.
        cols, rows = struct.unpack_from('>HH', data, offset)
        if cols == 0 or rows == 0:
            return _orig_deserialize(data, offset, is_bframe)

        # Zdekoduj VLC do raw i przekaż do oryginalnego deserializera
        vlc_chunk = bytes(data[offset:])  # slice od offset
        raw = unpack_pframe(vlc_chunk)

        # Oblicz ile bajtów zajął VLC chunk (potrzebne do aktualizacji offset)
        # Robimy to przez śledzenie w unpack_pframe — dodajemy pomocniczą funkcję
        _, vlc_size = _unpack_pframe_with_size(vlc_chunk)

        blocks, _ = _orig_deserialize(raw, 0, is_bframe)
        return blocks, offset + vlc_size

    codec_mod._serialize_blocks   = _vlc_serialize
    codec_mod._deserialize_blocks = _vlc_deserialize
    return codec_mod


def _unpack_pframe_with_size(vlc_bytes: bytes) -> tuple:
    """
    Jak unpack_pframe() ale zwraca też rozmiar skonsumowanych bajtów VLC.
    Zwraca (raw_bytes, consumed_bytes).
    """
    if len(vlc_bytes) < 4:
        return vlc_bytes, len(vlc_bytes)

    offset = 0
    cols, rows = struct.unpack_from('>HH', vlc_bytes, offset); offset += 4

    if cols == 0 or rows == 0:
        return vlc_bytes, 4

    n_blocks  = rows * cols
    bitmap_sz = (n_blocks + 7) // 8
    bitmap    = vlc_bytes[offset:offset + bitmap_sz]; offset += bitmap_sz

    out = bytearray()
    out.extend(struct.pack('>HH', cols, rows))
    out.extend(bitmap)

    for bit_idx in range(n_blocks):
        is_detail = bool(bitmap[bit_idx >> 3] & (1 << (7 - (bit_idx & 7))))
        if not is_detail:
            continue

        (dx, dy), offset = _unpack_mv(vlc_bytes, offset)
        out.extend(struct.pack('>hh', dx, dy))

        blk_y, offset = _decode_block(vlc_bytes, offset, _ZZ16_R, _BS)
        out.extend(blk_y.astype(np.int16).flatten().tobytes())

        blk_u, offset = _decode_block(vlc_bytes, offset, _ZZ8_R, _BS_C)
        out.extend(blk_u.astype(np.int16).flatten().tobytes())

        blk_v, offset = _decode_block(vlc_bytes, offset, _ZZ8_R, _BS_C)
        out.extend(blk_v.astype(np.int16).flatten().tobytes())

    # Przesuń za magic
    if vlc_bytes[offset:offset+2] == _EOF_MAGIC:
        offset += 2

    return bytes(out), offset


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TESTY I BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest():
    """Weryfikacja roundtrip i pomiar kompresji."""
    import time
    from scipy.fftpack import dct

    def dct2(b): return dct(dct(b.T, norm='ortho').T, norm='ortho')

    rng = np.random.default_rng(42)
    Q_Y, Q_C = 22.0, 40.0

    print("═══ TEST ROUNDTRIP ═══")

    # Generuj bloki residual (jak w P-frame)
    n_blocks_y = 100
    residual_y = rng.normal(0, 8, (n_blocks_y, 16, 16)).astype(np.float32)
    q_blocks_y = [np.round(dct2(b) / Q_Y).astype(np.int16) for b in residual_y]

    residual_c = rng.normal(0, 4, (n_blocks_y, 8, 8)).astype(np.float32)
    q_blocks_c = [np.round(dct2(b) / Q_C).astype(np.int16) for b in residual_c]

    # Test encode/decode pojedynczego bloku
    for i in range(5):
        blk = q_blocks_y[i]
        enc = _encode_block(blk, _ZZ16)
        dec, _ = _decode_block(enc, 0, _ZZ16_R, 16)
        assert np.array_equal(blk, dec), f"Roundtrip blok {i} FAILED!\n{blk}\n{dec}"
    print("✓ Roundtrip bloku Y (5 bloków)")

    for i in range(5):
        blk = q_blocks_c[i]
        enc = _encode_block(blk, _ZZ8)
        dec, _ = _decode_block(enc, 0, _ZZ8_R, 8)
        assert np.array_equal(blk, dec), f"Roundtrip blok chroma {i} FAILED!"
    print("✓ Roundtrip bloku C (5 bloków)")

    # Test MV
    for dx, dy in [(0,0), (10,-5), (-96,96), (100,-130), (200,300)]:
        packed = _pack_mv(dx, dy)
        (rx, ry), _ = _unpack_mv(packed, 0)
        assert (rx, ry) == (dx, dy), f"MV roundtrip failed: ({dx},{dy}) → ({rx},{ry})"
    print("✓ Roundtrip wektorów ruchu (5 przypadków, w tym escape)")

    print("\n═══ BENCHMARK KOMPRESJI ═══")

    # Symuluj pełną klatkę P (1080p = 120×68 bloków = 8160 bloków, ~30% DETAIL)
    n_y = 120 * 68
    detail_ratio = 0.3
    n_detail = int(n_y * detail_ratio)

    # Zbuduj fałszywą klatkę P w formacie v2.7
    import io
    buf = bytearray()
    cols, rows = 120, 68
    n_blocks = cols * rows
    bitmap = bytearray((n_blocks + 7) // 8)

    # Zaznacz pierwsze n_detail bloków jako DETAIL
    detail_indices = sorted(rng.choice(n_blocks, n_detail, replace=False))
    for idx in detail_indices:
        bitmap[idx >> 3] |= (1 << (7 - (idx & 7)))

    buf.extend(struct.pack('>HH', cols, rows))
    buf.extend(bitmap)

    raw_qdiffs = []
    for _ in range(n_detail):
        dx = int(rng.integers(-96, 97))
        dy = int(rng.integers(-96, 97))
        buf.extend(struct.pack('>hh', dx, dy))

        qy = np.round(dct2(rng.normal(0,8,(16,16)).astype(np.float32)) / Q_Y).astype(np.int16)
        qu = np.round(dct2(rng.normal(0,4,(8,8)).astype(np.float32))  / Q_C).astype(np.int16)
        qv = np.round(dct2(rng.normal(0,4,(8,8)).astype(np.float32))  / Q_C).astype(np.int16)
        buf.extend(qy.flatten().tobytes())
        buf.extend(qu.flatten().tobytes())
        buf.extend(qv.flatten().tobytes())
        raw_qdiffs.append((dx, dy, qy, qu, qv))

    raw_frame = bytes(buf)

    # VLC pack
    t0 = time.perf_counter()
    vlc_frame = pack_pframe(raw_frame)
    t_pack = (time.perf_counter() - t0) * 1000

    # Roundtrip
    t0 = time.perf_counter()
    raw_recovered = unpack_pframe(vlc_frame)
    t_unpack = (time.perf_counter() - t0) * 1000

    # Weryfikacja
    assert raw_recovered == raw_frame, \
        f"Pframe roundtrip FAILED! raw={len(raw_frame)}B, rec={len(raw_recovered)}B"
    print("✓ Pełny roundtrip klatki P (1080p, 30% DETAIL)")

    ratio = len(raw_frame) / len(vlc_frame)
    print(f"\n  Rozmiar raw v2.7:  {len(raw_frame):>8,} B")
    print(f"  Rozmiar VLC:       {len(vlc_frame):>8,} B")
    print(f"  Kompresja VLC:     {ratio:.2f}× ({100*(1-1/ratio):.0f}% mniej)")
    print(f"  Czas pack:         {t_pack:.1f} ms")
    print(f"  Czas unpack:       {t_unpack:.1f} ms")

    # Ile da Zstd po VLC vs bez VLC?
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=22)
        z_raw = cctx.compress(raw_frame)
        z_vlc = cctx.compress(vlc_frame)
        print(f"\n  Po Zstd(22) raw:   {len(z_raw):>8,} B  ({len(raw_frame)/len(z_raw):.1f}× vs raw)")
        print(f"  Po Zstd(22) VLC:   {len(z_vlc):>8,} B  ({len(raw_frame)/len(z_vlc):.1f}× vs raw)")
        print(f"  VLC+Zstd vs Zstd:  {len(z_raw)/len(z_vlc):.2f}×")
        print(f"  VLC+Zstd total:    {len(raw_frame)/len(z_vlc):.1f}× vs raw int16")
    except ImportError:
        print("  (zstandard niedostępne — pomiń porównanie Zstd)")

    print("\n═══ ROZMIAR SYMBOLÓW ═══")
    all_bytes = []
    for _, _, qy, qu, qv in raw_qdiffs[:50]:
        all_bytes.extend(_encode_block(qy, _ZZ16))
        all_bytes.extend(_encode_block(qu, _ZZ8))
        all_bytes.extend(_encode_block(qv, _ZZ8))
    from collections import Counter
    sym_cnt = Counter(all_bytes)
    print(f"  Unikalne bajty VLC: {len(sym_cnt)} (z 256 możliwych)")
    total_syms = sum(sym_cnt.values())
    top10 = sym_cnt.most_common(10)
    print(f"  Top 10 symboli:")
    for b, cnt in top10:
        sym = _B2SYM[b] if b < len(_B2SYM) and _B2SYM[b] else ('ESC_LIT' if b==ESCAPE_LIT else '?')
        label = "EOB" if sym == _EOB else ("ESC_RUN" if sym == _ESC_RUN else
                f"({sym[0]},{sym[1]:+d})" if isinstance(sym, tuple) else str(sym))
        print(f"    {b:#04x} {label:12s}  {100*cnt/total_syms:5.1f}%")


if __name__ == '__main__':
    _selftest()
