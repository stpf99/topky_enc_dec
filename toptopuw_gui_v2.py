"""
╔══════════════════════════════════════════════════════════════════════════╗
║   TOP TOPÓW CODEC v2 — ANALIZATOR GUI v2 (PyQt6)                        ║
║   Player .toptop | Auto-dobór parametrów | Log | Statystyki | Analiza   ║
╚══════════════════════════════════════════════════════════════════════════╝

Uruchomienie:
    pip install PyQt6 imageio[pyav] zstandard numpy scipy --break-system-packages
    python toptopuw_gui_v2.py
"""

import sys, os, time, threading, subprocess, struct, traceback, importlib.util
from pathlib import Path
from collections import deque

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QGroupBox, QTabWidget, QSplitter, QFrame, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QStatusBar, QSizePolicy, QSlider,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QTimer, QElapsedTimer, QObject, QSize
from PyQt6.QtGui   import (
    QFont, QColor, QPalette, QTextCursor, QPixmap, QImage,
    QSyntaxHighlighter, QTextCharFormat, QPainter, QBrush, QPen,
)

try:
    import imageio.v3 as iio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False

# Moduł przyspieszeń — ładowany lazily po wskazaniu pliku kodeka
_speedup = None
_speedup_ok = False
_speedup_info = "nie załadowano"

def _load_speedup():
    """Ładuje toptopuw_speedup.py z tego samego katalogu co GUI."""
    global _speedup, _speedup_ok, _speedup_info
    if _speedup is not None:
        return _speedup

    # Zbierz kandydatów — bez duplikatów na podstawie str(absolute)
    _seen = set()
    candidates = []

    # Buduj listę katalogów do przeszukania
    _dirs = []
    # 1. Katalog tego pliku GUI (najważniejszy)
    try:
        _dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    # 2. sys.argv[0] może być ścieżką bezwzględną lub względną — bierzemy parent
    try:
        _p = Path(sys.argv[0]).resolve()
        if _p.is_file():
            _dirs.append(_p.parent)
        else:
            _dirs.append(_p)
    except Exception:
        pass
    # 3. CWD
    try:
        _dirs.append(Path.cwd())
    except Exception:
        pass
    # 4. HOME użytkownika
    try:
        _dirs.append(Path.home())
    except Exception:
        pass

    for _d in _dirs:
        p = _d / "toptopuw_speedup.py"
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in _seen:
            _seen.add(key)
            candidates.append(p)

    print(f"[speedup] szukam w: {[str(c) for c in candidates]}", flush=True)
    errors = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location("toptopuw_speedup", str(p))
            mod  = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _speedup     = mod
            _speedup_ok  = True
            _speedup_info = f"✓ załadowano z {p}"
            return mod
        except Exception as e:
            errors.append(f"{p}: {e}")
            continue  # próbuj następny kandydat

    if errors:
        _speedup_info = "✗ błąd importu: " + " | ".join(errors)
    else:
        _speedup_info = "✗ nie znaleziono toptopuw_speedup.py"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# PALETA KOLORÓW
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":       "#12121f",
    "bg2":      "#1a1a2e",
    "bg3":      "#0f3460",
    "accent":   "#e94560",
    "accent2":  "#533483",
    "text":     "#eaeaea",
    "dim":      "#888899",
    "green":    "#00c896",
    "yellow":   "#f5c518",
    "red":      "#ff4444",
    "blue":     "#4fc3f7",
    "orange":   "#ff8c42",
    "purple":   "#bb86fc",
    "border":   "#2a2a4a",
    "player_bg":"#080810",
}

STYLESHEET = f"""
QMainWindow, QWidget {{ background-color:{C['bg']}; color:{C['text']};
    font-family:'Segoe UI','Ubuntu',sans-serif; font-size:13px; }}
QGroupBox {{ border:1px solid {C['border']}; border-radius:6px; margin-top:10px;
    padding-top:8px; font-weight:bold; color:{C['blue']}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 6px; }}
QPushButton {{ background-color:{C['bg3']}; border:1px solid {C['accent']};
    border-radius:5px; padding:6px 14px; color:{C['text']}; font-weight:bold; }}
QPushButton:hover {{ background-color:{C['accent']}; color:white; }}
QPushButton:disabled {{ background-color:{C['bg2']}; border-color:{C['border']};
    color:{C['dim']}; }}
QPushButton#start_btn {{ background-color:{C['green']}; border-color:{C['green']};
    color:#000; font-size:14px; padding:10px 24px; border-radius:6px; }}
QPushButton#start_btn:hover {{ background-color:#00ffb3; }}
QPushButton#start_btn:disabled {{ background-color:{C['border']}; color:{C['dim']}; }}
QPushButton#stop_btn {{ background-color:{C['red']}; border-color:{C['red']};
    color:white; font-size:14px; padding:10px 24px; border-radius:6px; }}
QPushButton#stop_btn:disabled {{ background-color:{C['border']}; color:{C['dim']}; }}
QPushButton#play_btn {{ background-color:{C['bg3']}; border:2px solid {C['green']};
    border-radius:20px; min-width:40px; min-height:40px; font-size:18px; padding:0; }}
QPushButton#play_btn:hover {{ background-color:{C['green']}; color:#000; }}
QPushButton#auto_btn {{ background-color:{C['accent2']}; border-color:{C['purple']};
    color:white; font-weight:bold; padding:7px 16px; }}
QPushButton#auto_btn:hover {{ background-color:{C['purple']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color:{C['bg2']}; border:1px solid {C['border']};
    border-radius:4px; padding:4px 8px; color:{C['text']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color:{C['accent']}; }}
QTextEdit {{ background-color:{C['bg2']}; border:1px solid {C['border']};
    border-radius:4px; color:{C['text']};
    font-family:'Cascadia Code','Consolas','Courier New',monospace; font-size:12px; }}
QProgressBar {{ background-color:{C['bg2']}; border:1px solid {C['border']};
    border-radius:4px; text-align:center; color:{C['text']}; height:22px; }}
QProgressBar::chunk {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 {C['bg3']},stop:1 {C['accent']}); border-radius:3px; }}
QSlider::groove:horizontal {{ background:{C['bg2']}; height:6px; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{C['accent']}; width:14px; height:14px;
    margin:-4px 0; border-radius:7px; }}
QSlider::sub-page:horizontal {{ background:{C['accent']}; border-radius:3px; }}
QTabWidget::pane {{ border:1px solid {C['border']}; border-radius:4px; }}
QTabBar::tab {{ background-color:{C['bg2']}; border:1px solid {C['border']};
    padding:6px 16px; margin-right:2px; border-radius:4px 4px 0 0;
    color:{C['dim']}; }}
QTabBar::tab:selected {{ background-color:{C['bg3']}; color:{C['text']};
    border-bottom-color:{C['bg3']}; }}
QTabBar::tab:hover {{ background-color:{C['bg3']}; color:{C['text']}; }}
QTableWidget {{ background-color:{C['bg2']}; border:1px solid {C['border']};
    gridline-color:{C['border']}; alternate-background-color:{C['bg']};
    color:{C['text']}; }}
QTableWidget::item:selected {{ background-color:{C['bg3']}; }}
QHeaderView::section {{ background-color:{C['bg3']}; border:none;
    padding:4px 8px; font-weight:bold; color:{C['blue']}; }}
QCheckBox::indicator {{ width:16px; height:16px; border:1px solid {C['border']};
    border-radius:3px; background-color:{C['bg2']}; }}
QCheckBox::indicator:checked {{ background-color:{C['green']};
    border-color:{C['green']}; }}
QScrollBar:vertical {{ background-color:{C['bg2']}; width:10px; border-radius:5px; }}
QScrollBar::handle:vertical {{ background-color:{C['border']}; border-radius:5px;
    min-height:30px; }}
QScrollBar::handle:vertical:hover {{ background-color:{C['accent']}; }}
QStatusBar {{ background-color:{C['bg2']}; border-top:1px solid {C['border']};
    color:{C['dim']}; }}
"""

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMICZNY IMPORT KODEKA
# ══════════════════════════════════════════════════════════════════════════════

_codec_module = None
_codec_module_path = ""

def load_codec_module(path: str):
    """Dynamicznie importuje toptopuwcodec_v1.py i aplikuje speedup."""
    global _codec_module, _codec_module_path
    if path == _codec_module_path and _codec_module is not None:
        return _codec_module
    spec = importlib.util.spec_from_file_location("toptopuwcodec", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Aplikuj przyspieszenia jeśli speedup.py dostępny
    sp = _load_speedup()
    if sp is not None:
        try:
            sp.apply(mod)
        except Exception as e:
            print(f"[speedup] błąd aplikowania patchy: {e}")
    _codec_module      = mod
    _codec_module_path = path
    return mod


def _fmt_size(b: int) -> str:
    if b < 1024:    return f"{b} B"
    if b < 1<<20:   return f"{b/1024:.1f} KB"
    if b < 1<<30:   return f"{b/1<<20:.2f} MB"
    return                  f"{b/1<<30:.3f} GB"

# ══════════════════════════════════════════════════════════════════════════════
# DEKODER W TLE (dla playera)
# ══════════════════════════════════════════════════════════════════════════════

class DecoderSignals(QObject):
    frame_ready   = pyqtSignal(int, object, str)   # (idx, np_rgb, frame_type)
    progress      = pyqtSignal(int, int)            # (current, total)
    finished      = pyqtSignal(int)                 # total frames decoded
    error         = pyqtSignal(str)


class ToptopDecoderThread(threading.Thread):
    """
    Dekoduje plik .toptop w tle bezpośrednio przez importowany moduł kodeka.
    Emituje kolejne klatki jako numpy uint8 RGB przez sygnały Qt.
    """

    def __init__(self, signals: DecoderSignals, toptop_path: str, codec_path: str):
        super().__init__(daemon=True)
        self.signals      = signals
        self.toptop_path  = toptop_path
        self.codec_path   = codec_path
        self._cancel      = threading.Event()
        # Przechowujemy zdekodowane klatki
        self.frames: list = []          # list[np.ndarray uint8 RGB]
        self.frame_types: list = []     # list[str] 'I'/'P'/'B'

    def stop(self):
        self._cancel.set()

    def run(self):
        try:
            self._decode()
        except Exception:
            self.signals.error.emit(traceback.format_exc())

    def _decode(self):
        import zstandard as zstd

        mod = load_codec_module(self.codec_path)

        with open(self.toptop_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            raw  = dctx.stream_reader(f).read()

        w, h = struct.unpack_from('>HH', raw, 0)
        all_frames = mod._parse_all_frames(raw, h, w)
        n = len(all_frames)

        # Pełna logika dekodowania (z DPB dla B-klatek) – skopiowana z decode_video
        codec = mod.TopTopowCodecV2()
        rec_Y, rec_U, rec_V = {}, {}, {}
        decoded_frames = []
        frame_types    = []

        for i, data in enumerate(all_frames):
            if self._cancel.is_set():
                break
            ft = data['type']

            if ft == 'B':
                future_idx = None
                for j in range(i + 1, n):
                    if all_frames[j]['type'] != 'B':
                        future_idx = j; break
                if future_idx is not None and future_idx not in rec_Y:
                    tmp = mod.TopTopowCodecV2()
                    ref_idx = None
                    for j in range(future_idx - 1, -1, -1):
                        if all_frames[j]['type'] != 'B' and j in rec_Y:
                            ref_idx = j; break
                    if ref_idx is not None:
                        tmp.prev_Y, tmp.prev_U, tmp.prev_V = (
                            rec_Y[ref_idx], rec_U[ref_idx], rec_V[ref_idx])
                    tmp.future_Y = None
                    tmp.decode_frame(all_frames[future_idx], h, w)
                    rec_Y[future_idx] = tmp.prev_Y.copy()
                    rec_U[future_idx] = tmp.prev_U.copy()
                    rec_V[future_idx] = tmp.prev_V.copy()
                if future_idx is not None and future_idx in rec_Y:
                    codec.future_Y = rec_Y[future_idx]
                    codec.future_U = rec_U[future_idx]
                    codec.future_V = rec_V[future_idx]
                else:
                    codec.future_Y = None
            else:
                codec.future_Y = None

            try:
                img = codec.decode_frame(data, h, w)
            except Exception as _exc:
                import traceback as _tb
                print(f"[DECODE ERROR] klatka {i}, typ={ft}, h={h}, w={w}", flush=True)
                print(_tb.format_exc(), flush=True)
                raise
            decoded_frames.append(img)
            frame_types.append(ft)

            if ft != 'B':
                rec_Y[i] = codec.prev_Y.copy()
                rec_U[i] = codec.prev_U.copy()
                rec_V[i] = codec.prev_V.copy()

            self.signals.frame_ready.emit(i, img, ft)
            self.signals.progress.emit(i + 1, n)

        self.frames      = decoded_frames
        self.frame_types = frame_types
        self.signals.finished.emit(len(decoded_frames))


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET PLAYERA
# ══════════════════════════════════════════════════════════════════════════════

FRAME_TYPE_COLOR = {'I': '#f5c518', 'P': '#00c896', 'B': '#ff8c42'}

class VideoCanvas(QLabel):
    """QLabel z czarnym tłem i wyśrodkowanym wideo (keep aspect ratio)."""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background:{C['player_bg']}; border:1px solid {C['border']};")
        self.setMinimumSize(480, 270)
        self._pixmap_raw: QPixmap | None = None
        self._overlay_text = ""

    def set_frame(self, rgb: np.ndarray, overlay: str = ""):
        h, w = rgb.shape[:2]
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_raw  = QPixmap.fromImage(qimg)
        self._overlay_text = overlay
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap_raw is None:
            return
        # Scaled keep-aspect
        scaled = self._pixmap_raw.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(self)
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

        # Overlay badge
        if self._overlay_text:
            painter.setPen(QColor("#000000"))
            painter.setBrush(QBrush(QColor(
                FRAME_TYPE_COLOR.get(self._overlay_text[0], '#ffffff') + "cc")))
            painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            text = self._overlay_text
            tw = fm.horizontalAdvance(text) + 12
            th = fm.height() + 6
            painter.drawRoundedRect(x + 8, y + 8, tw, th, 4, 4)
            painter.setPen(QColor("#000000"))
            painter.drawText(x + 14, y + 8 + fm.ascent() + 2, text)
        painter.end()


class ToptopPlayerWidget(QWidget):
    """
    Kompletny player do plików .toptop.
    Dekoduje w tle, przechowuje klatki w pamięci, odtwarza przez QTimer.
    """

    def __init__(self, codec_finder_fn, parent=None):
        super().__init__(parent)
        self._get_codec = codec_finder_fn   # callable → str ścieżki kodeka
        self._frames: list[np.ndarray] = []
        self._types:  list[str]        = []
        self._current_frame = 0
        self._playing       = False
        self._loaded_path   = ""

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._fps = 25.0

        self._decoder: ToptopDecoderThread | None = None
        self._dec_signals: DecoderSignals | None  = None

        self._build_ui()

    # ── Budowa UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Selektor pliku ────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Wybierz plik .toptop do odtworzenia…")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_file)
        self._btn_load = QPushButton("⬇ Wczytaj & Dekoduj")
        self._btn_load.clicked.connect(self._load_file)
        file_row.addWidget(self._path_edit)
        file_row.addWidget(btn_browse)
        file_row.addWidget(self._btn_load)
        lay.addLayout(file_row)

        # ── Pasek postępu dekodowania ─────────────────────────────────────
        self._decode_bar = QProgressBar()
        self._decode_bar.setFormat("Dekodowanie: %v / %m klatek (%p%)")
        self._decode_bar.hide()
        lay.addWidget(self._decode_bar)

        # ── Canvas wideo ──────────────────────────────────────────────────
        self._canvas = VideoCanvas()
        self._show_placeholder()
        lay.addWidget(self._canvas, stretch=1)

        # ── Slider pozycji ────────────────────────────────────────────────
        seek_row = QHBoxLayout()
        self._pos_label   = QLabel("0:00.000")
        self._pos_label.setStyleSheet(f"color:{C['dim']}; font-size:11px; font-family:Consolas;")
        self._pos_label.setFixedWidth(70)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(0)
        self._seek_slider.sliderMoved.connect(self._seek_to)
        self._dur_label   = QLabel("0:00.000")
        self._dur_label.setStyleSheet(f"color:{C['dim']}; font-size:11px; font-family:Consolas;")
        self._dur_label.setFixedWidth(70)
        self._dur_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        seek_row.addWidget(self._pos_label)
        seek_row.addWidget(self._seek_slider)
        seek_row.addWidget(self._dur_label)
        lay.addLayout(seek_row)

        # ── Kontrolki playbacku ───────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._btn_prev = QPushButton("⏮")
        self._btn_prev.setFixedSize(38, 38)
        self._btn_prev.clicked.connect(self._step_back)

        self._btn_play = QPushButton("▶")
        self._btn_play.setObjectName("play_btn")
        self._btn_play.setFixedSize(48, 48)
        self._btn_play.clicked.connect(self._toggle_play)

        self._btn_next = QPushButton("⏭")
        self._btn_next.setFixedSize(38, 38)
        self._btn_next.clicked.connect(self._step_fwd)

        self._btn_stop_play = QPushButton("⏹")
        self._btn_stop_play.setFixedSize(38, 38)
        self._btn_stop_play.clicked.connect(self._stop_play)

        # FPS
        fps_lbl = QLabel("FPS:")
        fps_lbl.setStyleSheet(f"color:{C['dim']}")
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 120.0)
        self._fps_spin.setValue(25.0)
        self._fps_spin.setFixedWidth(72)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)

        # Info klatki
        self._frame_info = QLabel("Klatka: — / —   Typ: —")
        self._frame_info.setStyleSheet(f"color:{C['dim']}; font-size:11px; font-family:Consolas;")

        ctrl_row.addStretch()
        ctrl_row.addWidget(self._btn_prev)
        ctrl_row.addWidget(self._btn_play)
        ctrl_row.addWidget(self._btn_next)
        ctrl_row.addWidget(self._btn_stop_play)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(fps_lbl)
        ctrl_row.addWidget(self._fps_spin)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._frame_info)
        lay.addLayout(ctrl_row)

        # ── Statystyki pliku ──────────────────────────────────────────────
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(
            f"color:{C['dim']}; font-size:11px; padding:2px 6px; "
            f"background:{C['bg2']}; border-radius:3px;")
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._stats_label)

        self._set_controls_enabled(False)

    def _show_placeholder(self):
        self._canvas.setText(
            "🎞  Wczytaj plik .toptop aby odtworzyć\n\n"
            "Plik zostanie zdekodowany w tle — bezpośrednio z danych kodeka"
        )
        self._canvas.setStyleSheet(
            f"background:{C['player_bg']}; color:{C['dim']}; "
            f"border:1px solid {C['border']}; font-size:14px;")

    # ── Obsługa pliku ──────────────────────────────────────────────────────
    def load_file(self, path: str):
        """Publiczna metoda: wczytaj wskazany plik."""
        self._path_edit.setText(path)
        self._load_file()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Otwórz plik .toptop", "",
            "TopTop (*.toptop);;Wszystkie (*)")
        if path:
            self._path_edit.setText(path)

    def _load_file(self):
        path   = self._path_edit.text().strip()
        codec  = self._get_codec()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Błąd", "Podaj istniejący plik .toptop")
            return
        if not codec or not os.path.exists(codec):
            QMessageBox.warning(self, "Błąd", f"Nie znaleziono kodeka:\n{codec}")
            return

        # Zatrzymaj poprzedni decoder
        if self._decoder and self._decoder.is_alive():
            self._decoder.stop()
        self._stop_play()
        self._frames.clear()
        self._types.clear()
        self._current_frame = 0
        self._loaded_path = path

        # Przygotuj UI
        self._decode_bar.setValue(0)
        self._decode_bar.show()
        self._btn_load.setEnabled(False)
        self._set_controls_enabled(False)
        self._canvas.setText(f"⏳  Dekodowanie: {os.path.basename(path)} …")
        self._canvas.setStyleSheet(
            f"background:{C['player_bg']}; color:{C['yellow']}; "
            f"border:1px solid {C['border']}; font-size:14px;")
        self._seek_slider.setMaximum(0)

        # Uruchom decoder
        sigs = DecoderSignals()
        sigs.frame_ready.connect(self._on_frame_ready)
        sigs.progress.connect(self._on_decode_progress)
        sigs.finished.connect(self._on_decode_finished)
        sigs.error.connect(self._on_decode_error)
        self._dec_signals = sigs

        self._decoder = ToptopDecoderThread(sigs, path, codec)
        self._decoder.start()

    def _on_frame_ready(self, idx: int, rgb: np.ndarray, ft: str):
        self._frames.append(rgb)
        self._types.append(ft)
        # Pokaż pierwszą klatkę od razu
        if idx == 0:
            self._canvas.setStyleSheet(
                f"background:{C['player_bg']}; border:1px solid {C['border']};")
            self._canvas.setText("")
            self._show_frame(0)

    def _on_decode_progress(self, cur: int, total: int):
        self._decode_bar.setMaximum(total)
        self._decode_bar.setValue(cur)
        self._seek_slider.setMaximum(max(0, len(self._frames) - 1))

    def _on_decode_finished(self, total: int):
        self._decode_bar.hide()
        self._btn_load.setEnabled(True)
        self._set_controls_enabled(True)
        self._seek_slider.setMaximum(max(0, total - 1))

        dur = total / self._fps
        self._dur_label.setText(self._fmt_time(dur))

        # Policz typy
        ci = self._types.count('I')
        cp = self._types.count('P')
        cb = self._types.count('B')
        sz = os.path.getsize(self._loaded_path) if os.path.exists(self._loaded_path) else 0
        bpf = sz / total if total > 0 else 0
        self._stats_label.setText(
            f"Klatki: {total}   I:{ci}  P:{cp}  B:{cb}   "
            f"FPS: {self._fps:.1f}   Czas: {self._fmt_time(dur)}   "
            f"Plik: {_fmt_size(sz)}   Śr./klatkę: {_fmt_size(int(bpf))}"
        )

    def _on_decode_error(self, msg: str):
        self._decode_bar.hide()
        self._btn_load.setEnabled(True)
        QMessageBox.critical(self, "Błąd dekodowania", msg[:400])

    # ── Playback ───────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self._frames:
            return
        if self._current_frame >= len(self._frames) - 1:
            self._current_frame = 0
        self._playing = True
        self._btn_play.setText("⏸")
        interval = max(1, int(1000 / self._fps))
        self._play_timer.start(interval)

    def _pause(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setText("▶")

    def _stop_play(self):
        self._pause()
        self._current_frame = 0
        if self._frames:
            self._show_frame(0)

    def _advance_frame(self):
        if not self._frames:
            return
        if self._current_frame < len(self._frames) - 1:
            self._current_frame += 1
            self._show_frame(self._current_frame)
        else:
            # Koniec — pętla
            self._current_frame = 0
            self._show_frame(0)

    def _step_back(self):
        self._pause()
        if self._frames and self._current_frame > 0:
            self._current_frame -= 1
            self._show_frame(self._current_frame)

    def _step_fwd(self):
        self._pause()
        if self._frames and self._current_frame < len(self._frames) - 1:
            self._current_frame += 1
            self._show_frame(self._current_frame)

    def _seek_to(self, pos: int):
        self._pause()
        self._current_frame = max(0, min(pos, len(self._frames) - 1))
        self._show_frame(self._current_frame)

    def _show_frame(self, idx: int):
        if idx >= len(self._frames):
            return
        rgb = self._frames[idx]
        ft  = self._types[idx] if idx < len(self._types) else '?'
        overlay = f"{ft}  #{idx+1}"
        self._canvas.set_frame(rgb, overlay)
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(idx)
        self._seek_slider.blockSignals(False)
        ts = self._fmt_time(idx / self._fps)
        self._pos_label.setText(ts)
        total = len(self._frames)
        self._frame_info.setText(f"Klatka: {idx+1} / {total}   Typ: {ft}")

    def _on_fps_changed(self, v: float):
        self._fps = v
        if self._playing:
            self._play_timer.setInterval(max(1, int(1000 / v)))
        total = len(self._frames)
        if total > 0:
            self._dur_label.setText(self._fmt_time(total / v))

    def _set_controls_enabled(self, en: bool):
        for w in (self._btn_play, self._btn_prev, self._btn_next,
                  self._btn_stop_play, self._seek_slider, self._fps_spin):
            w.setEnabled(en)

    @staticmethod
    def _fmt_time(s: float) -> str:
        m = int(s // 60)
        return f"{m}:{s % 60:06.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# POMOCNIK FPS — odporna ekstrakcja z ImageProperties (imageio v2 / v3)
# ══════════════════════════════════════════════════════════════════════════════

def _get_fps(props, path: str = "") -> float:
    """
    Zwraca FPS z obiektu ImageProperties lub z iio.immeta().
    Obsługuje: props.fps (stare imageio), props.metadata dict (nowe imageio),
    immeta() jako fallback, ułamki "30/1" i floaty.
    """
    def _parse(val) -> float | None:
        if not val:
            return None
        try:
            if isinstance(val, str) and '/' in val:
                a, b = val.split('/')
                v = float(a) / float(b)
            else:
                v = float(val)
            return v if v > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    # 1) atrybut .fps (imageio < 2.28)
    v = _parse(getattr(props, 'fps', None))
    if v: return v

    # 2) słownik .metadata (imageio >= 2.28)
    meta = getattr(props, 'metadata', None) or {}
    for key in ('fps', 'average_rate', 'r_frame_rate', 'framerate', 'tbr'):
        v = _parse(meta.get(key))
        if v: return v

    # 3) iio.immeta() jako ostateczny fallback
    if path and IMAGEIO_OK:
        try:
            m2 = iio.immeta(path, plugin="pyav")
            for key in ('fps', 'average_rate', 'r_frame_rate', 'framerate', 'tbr'):
                v = _parse(m2.get(key))
                if v: return v
        except Exception:
            pass

    return 25.0   # ostateczny default


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-DOBÓR PARAMETRÓW
# ══════════════════════════════════════════════════════════════════════════════

class AutoParamSignals(QObject):
    progress = pyqtSignal(str)          # komunikat statusu
    finished = pyqtSignal(dict)         # wynik: słownik rekomendacji
    error    = pyqtSignal(str)


class AutoParamAnalyzer(threading.Thread):
    """
    Analizuje wideo wejściowe próbkując N klatek i obliczając:
      - rozdzielczość & liczba bloków
      - ruch (średni MAD między kolejnymi klatkami)
      - złożoność tekstury (wariancja)
      - FPS wejściowy
      - długość wideo
    Zwraca słownik z gotowymi wartościami parametrów (do wstawienia w UI).
    """
    N_SAMPLE = 16   # liczba klatek do próbkowania

    def __init__(self, signals: AutoParamSignals, video_path: str):
        super().__init__(daemon=True)
        self.signals = signals
        self.path    = video_path

    def run(self):
        try:
            result = self._analyze()
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit(traceback.format_exc())

    def _analyze(self) -> dict:
        sig = self.signals

        if not IMAGEIO_OK:
            raise RuntimeError("Zainstaluj imageio[pyav] aby używać auto-analizy")

        sig.progress.emit("⏳ Wczytywanie metadanych wideo…")
        props = iio.improps(self.path, plugin="pyav")

        sh  = props.shape
        fps = _get_fps(props, self.path)

        # ── Rozdzielczość & liczba klatek ─────────────────────────────────
        if len(sh) == 4:
            n_total, h, w, _ = sh
        elif len(sh) == 3:
            h, w, _ = sh
            n_total = 1
        else:
            h, w, n_total = 360, 640, 100

        # Zbierz próbki klatek (równomiernie rozłożone)
        sig.progress.emit(f"📐 Analiza rozdzielczości: {w}×{h}, ~{n_total} klatek, {fps:.2f} FPS")
        step = max(1, n_total // self.N_SAMPLE)
        frames = []
        for i, frame in enumerate(iio.imiter(self.path, plugin="pyav")):
            if i % step == 0:
                frames.append(frame)
                sig.progress.emit(f"   Próbka {len(frames)}/{self.N_SAMPLE} (klatka {i+1})")
            if len(frames) >= self.N_SAMPLE:
                break

        if len(frames) < 2:
            raise RuntimeError("Za krótkie wideo (< 2 klatek) — nie można analizować")

        # ── Metryki ────────────────────────────────────────────────────────
        sig.progress.emit("🔍 Obliczanie metryk ruchu i złożoności…")

        # Konwersja do float luma (Y = 0.299R + 0.587G + 0.114B)
        lumas = [
            (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]).astype(np.float32)
            for f in frames
        ]

        mads, variances = [], []
        for i in range(1, len(lumas)):
            mad = float(np.mean(np.abs(lumas[i] - lumas[i-1])))
            mads.append(mad)
        for luma in lumas:
            variances.append(float(np.var(luma)))

        avg_mad = float(np.mean(mads))
        avg_var = float(np.mean(variances))
        max_mad = float(np.max(mads)) if mads else avg_mad
        scene_cuts_est = sum(1 for m in mads if m > 35)

        n_blocks = (w // 16) * (h // 16)
        dur_s    = n_total / fps if fps > 0 else 0

        # ── Heurystyka doboru parametrów ───────────────────────────────────
        sig.progress.emit("🧮 Dobieranie optymalnych parametrów…")

        # --- Q_Y ---
        # Baza wg rozdzielczości (więcej pikseli → więcej szczegółów do zachowania)
        if n_blocks > 8000:       # 4K
            q_y_base = 18.0
        elif n_blocks > 3000:     # FullHD
            q_y_base = 20.0
        elif n_blocks > 1200:     # 720p
            q_y_base = 22.0
        else:                     # SD
            q_y_base = 20.0

        # Korekta wg ruchu (dużo ruchu → więcej kompresji inter, można Q podwyższyć)
        if avg_mad < 3.0:     motion_factor = -4.0   # statyczna scena → popraw jakość
        elif avg_mad < 8.0:   motion_factor =  0.0
        elif avg_mad < 20.0:  motion_factor = +3.0
        else:                 motion_factor = +6.0   # bardzo dynamiczna → mniejszy plik

        # Korekta wg tekstury (wysoka wariancja → sceny z detalami → mniej agresywnie)
        if avg_var > 1500:    texture_factor = -2.0
        elif avg_var > 500:   texture_factor =  0.0
        else:                 texture_factor = +2.0  # płaskie tło → mocniejsza kompresja

        q_y = round(max(12.0, min(50.0, q_y_base + motion_factor + texture_factor)), 1)
        q_c = round(max(20.0, min(90.0, q_y * 1.8)), 1)

        # --- Search range ---
        if avg_mad > 25:    search = 48
        elif avg_mad > 12:  search = 32
        elif avg_mad > 5:   search = 24
        else:               search = 16

        # --- Keyframe interval ---
        if fps <= 0:
            kf = 50
        else:
            # 2–4 sekundy GOP (group of pictures)
            kf = min(200, max(20, int(fps * 3.0)))

        # --- Scene cut ---
        scene_cut = 35.0
        if max_mad > 80:   scene_cut = 25.0   # gwałtowne cięcia → czulejszy detektor
        elif max_mad < 15: scene_cut = 0.0    # brak cięć → wyłącz (oszczędność CPU)

        # --- B-frames ---
        use_bframes = avg_mad < 30   # przy bardzo dużym ruchu B-klatki są mało efektywne

        # --- Subpixel ---
        use_subpixel = avg_mad > 2.0   # przy statycznej scenie subpixel nic nie daje

        # --- Adaptive Q ---
        adaptive_q = (max_mad / (avg_mad + 0.1)) > 2.5   # duże wahania → adaptive

        # ── Szacunki wynikowe ──────────────────────────────────────────────
        est_skip_ratio = max(0, min(0.95, (1 - avg_mad / 40.0)))
        est_compression = 3.0 + est_skip_ratio * 15 + (q_y / 22.0 - 1) * 5

        result = {
            # Parametry do UI
            "q_y":              q_y,
            "q_c":              q_c,
            "search_range":     search,
            "keyframe_interval":kf,
            "scene_cut":        scene_cut,
            "subpixel":         use_subpixel,
            "bframes":          use_bframes,
            "adaptive_q":       adaptive_q,
            # Metryki diagnostyczne
            "_avg_mad":         avg_mad,
            "_max_mad":         max_mad,
            "_avg_var":         avg_var,
            "_n_blocks":        n_blocks,
            "_resolution":      f"{w}×{h}",
            "_fps":             fps,
            "_duration":        dur_s,
            "_n_total":         n_total,
            "_scene_cuts_est":  scene_cuts_est,
            "_est_skip_ratio":  est_skip_ratio,
            "_est_compression": est_compression,
        }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PANEL STATYSTYK
# ══════════════════════════════════════════════════════════════════════════════

class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.meta_group   = QGroupBox("📹  Analiza wejścia")
        self.meta_table   = self._tbl(["Parametr", "Wartość"])
        self.meta_group.setLayout(self._wrap(self.meta_table))
        lay.addWidget(self.meta_group)

        self.auto_group   = QGroupBox("🧮  Rekomendacje auto-analizy")
        self.auto_table   = self._tbl(["Metryka", "Wartość", "Opis"])
        self.auto_group.setLayout(self._wrap(self.auto_table))
        lay.addWidget(self.auto_group)

        self.codec_group  = QGroupBox("⚙  Statystyki kodeka")
        self.codec_table  = self._tbl(["Parametr", "Wartość"])
        self.codec_group.setLayout(self._wrap(self.codec_table))
        lay.addWidget(self.codec_group)

        self.frames_group = QGroupBox("🎞  Dystrybucja klatek")
        self.frames_table = self._tbl(["Typ", "Liczba", "%"])
        self.frames_group.setLayout(self._wrap(self.frames_table))
        lay.addWidget(self.frames_group)

        lay.addStretch()

    def _wrap(self, w):
        l = QVBoxLayout(); l.addWidget(w); return l

    def _tbl(self, headers):
        t = QTableWidget()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        t.setMinimumHeight(60)
        t.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        return t

    def _fill(self, table, rows, colors=None):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter |
                    (Qt.AlignmentFlag.AlignLeft if c == 0
                     else Qt.AlignmentFlag.AlignRight if c == len(row)-1
                     else Qt.AlignmentFlag.AlignCenter))
                if colors and r < len(colors) and colors[r]:
                    item.setForeground(QColor(colors[r]))
                elif c == 1 and len(row) == 2:
                    item.setForeground(QColor(C['yellow']))
                table.setItem(r, c, item)
        table.resizeRowsToContents()

    def set_metadata(self, meta: dict):
        rows = [[k.replace('_',' ').title(), v]
                for k, v in meta.items() if not (k=='błąd' and v is None)]
        self._fill(self.meta_table, rows)

    def set_auto_analysis(self, res: dict):
        def mot(m):
            if m < 3:   return "🟢 Statyczna"
            if m < 10:  return "🟡 Mały ruch"
            if m < 25:  return "🟠 Średni ruch"
            return              "🔴 Duży ruch"
        def tex(v):
            if v > 1500: return "🔷 Bogata (dużo krawędzi)"
            if v > 500:  return "🔹 Normalna"
            return               "◽ Płaska (tło)"

        avg_mad = res.get("_avg_mad", 0)
        avg_var = res.get("_avg_var", 0)
        rows = [
            ["Rozdzielczość",     res.get("_resolution","?"),       "piksele"],
            ["Klatek wejścia",    res.get("_n_total","?"),          "szt."],
            ["FPS",               f"{res.get('_fps',0):.2f}",       "klatki/s"],
            ["Czas trwania",      f"{res.get('_duration',0):.1f}",  "s"],
            ["Bloków 16×16/klatkę", f"{res.get('_n_blocks',0):,}",  "szt."],
            ["Ruch (avg MAD)",    f"{avg_mad:.2f}",                  mot(avg_mad)],
            ["Ruch (maks MAD)",   f"{res.get('_max_mad',0):.2f}",   "peak"],
            ["Tekstura (wariancja)", f"{avg_var:.0f}",              tex(avg_var)],
            ["Est. scen cuts",    res.get("_scene_cuts_est","?"),   "szt."],
            ["Est. skip ratio",   f"{res.get('_est_skip_ratio',0)*100:.0f}%", "bloków SKIP"],
            ["Est. kompresja Zstd", f"~{res.get('_est_compression',0):.1f}×", "szacunek"],
        ]
        colors = [None]*len(rows)
        self._fill(self.auto_table, rows, colors)

    def set_codec_stats(self, stats: dict):
        mode = "Kodowanie" if stats.get('mode') == 'encode' else "Dekodowanie"
        n    = max(1, stats.get('klatki_total', 1))
        rows = [
            ["Tryb",                mode],
            ["Czas przetwarzania",  f"{stats.get('czas',0):.2f} s"],
            ["Prędkość",            f"{stats.get('fps_proc',0):.1f} kl/s"],
            ["Klatek łącznie",      str(stats.get('klatki_total',0))],
            ["Rozmiar wejścia",     _fmt_size(stats.get('rozmiar_we',0))],
            ["Rozmiar wyjścia",     _fmt_size(stats.get('rozmiar_wy',0))],
            ["Kompresja",           f"{stats.get('ratio',0):.2f}×"],
            ["Oszczędność",         f"{stats.get('oszcz_proc',0):.1f}%"],
            ["Bloków SKIP",         f"{stats.get('skip_total',0):,}"],
            ["Bloków DETAIL",       f"{stats.get('detail_total',0):,}"],
            ["Cięć sceny",          str(stats.get('scene_cuts',0))],
        ]
        if stats.get('skip_total',0)+stats.get('detail_total',0) > 0:
            t = stats['skip_total'] + stats['detail_total']
            rows.append(["Efektywność SKIP", f"{stats['skip_total']/t*100:.1f}%"])
        self._fill(self.codec_table, rows)

        ni, np_, nb = stats.get('I',0), stats.get('P',0), stats.get('B',0)
        frame_rows = [
            ["I (kluczowe)",   ni,  f"{ni/n*100:.1f}%"],
            ["P (predykcja)",  np_, f"{np_/n*100:.1f}%"],
            ["B (wsteczne)",   nb,  f"{nb/n*100:.1f}%"],
            ["Łącznie",        n,   "100%"],
        ]
        col = [C['yellow'], C['green'], C['orange'], C['blue']]
        self._fill(self.frames_table, frame_rows, col)


# ══════════════════════════════════════════════════════════════════════════════
# LOG HIGHLIGHTER
# ══════════════════════════════════════════════════════════════════════════════

class LogHighlighter(QSyntaxHighlighter):
    RULES = [
        (r"✓.*|✅.*",              C['green'],  True),
        (r"✂.*|CIECIE.*",          C['yellow'], True),
        (r"✗.*|ERROR.*|Błąd.*",   C['red'],    True),
        (r"⚠.*|WARN.*",           C['orange'], False),
        (r"\[Klatka.*?\]",         C['blue'],   False),
        (r"I-Frame.*",             C['yellow'], False),
        (r"B-Frame.*",             C['orange'], False),
        (r"P-Frame.*",             C['green'],  False),
        (r"SKIP:.*",               C['dim'],    False),
        (r"╔.*╗|╚.*╝|║.*",        C['accent2'],False),
        (r"AUTO.*|🧮.*|📊.*",      C['purple'], False),
        (r"\d+\.\d+[xs]|\d+KB|\d+MB", C['accent'], False),
    ]
    def __init__(self, parent):
        super().__init__(parent)
        import re
        self._compiled = []
        for pat, col, bold in self.RULES:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(col))
            if bold: fmt.setFontWeight(QFont.Weight.Bold)
            self._compiled.append((re.compile(pat), fmt))
    def highlightBlock(self, text):
        for rx, fmt in self._compiled:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end()-m.start(), fmt)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER KODEKA (subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class CodecSignals(QObject):
    log_line    = pyqtSignal(str)
    progress    = pyqtSignal(int, int)
    finished    = pyqtSignal(dict)
    error       = pyqtSignal(str)


class CodecWorker(threading.Thread):
    def __init__(self, signals, codec_path, mode, params):
        super().__init__(daemon=True)
        self.signals    = signals
        self.codec_path = codec_path
        self.mode       = mode
        self.params     = params
        self._stop_flag = threading.Event()

    def stop(self): self._stop_flag.set()

    def run(self):
        try:    self._run_codec()
        except Exception:
            self.signals.error.emit(traceback.format_exc())

    def _run_codec(self):
        p    = self.params
        # Flaga -u wymusza tryb niebuforowany stdout/stderr w procesie potomnym.
        # Bez niej Python trzyma cały output w buforze pipe'a aż do końca procesu
        # — GUI nie widzi żadnych linii postępu podczas dekodowania/kodowania.
        args = [sys.executable, '-u', self.codec_path]
        if self.mode == 'encode':
            args += ['-i', p['input'], '-o', p['output'],
                     '--q-y', str(p['q_y']), '--q-c', str(p['q_c']),
                     '--search-range', str(p['search_range']),
                     '--keyframe-interval', str(p['keyframe_interval']),
                     '--scene-cut', str(p['scene_cut'])]
            if p.get('full'):             args.append('-f')
            else:                         args += ['-n', str(p['max_frames'])]
            if not p.get('subpixel',True):  args.append('--no-subpixel')
            if not p.get('bframes',True):   args.append('--no-bframes')
            if p.get('adaptive_q'):         args.append('--adaptive-q')
            preset = p.get('preset', 0)
            if preset == 1:   args.append('-1')
            elif preset == 2: args.append('-2')
            elif preset == 3: args.append('-3')
        else:
            args += ['-i', p['input'], '-o', p['output'], '-d']

        self.signals.log_line.emit(f"► {' '.join(args)}\n")
        start    = time.time()
        fstats   = {'I':0,'P':0,'B':0,'skip_total':0,'detail_total':0,'scene_cuts':0}
        total_f  = p.get('max_frames',0) if not p.get('full') else 0
        cur_f    = 0
        import re

        # PYTHONUNBUFFERED=1 jako dodatkowe zabezpieczenie
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding='utf-8', errors='replace', env=env)
            for line in proc.stdout:
                if self._stop_flag.is_set():
                    proc.terminate()
                    self.signals.log_line.emit("\n⚠ Zatrzymano.\n")
                    break
                line = line.rstrip('\n')
                self.signals.log_line.emit(line+'\n')

                # Postęp KODOWANIA: [Klatka X/Y]
                m = re.search(r'\[Klatka\s+(\d+)/(\d+)\]', line)
                if m:
                    cur_f, total_f = int(m.group(1)), int(m.group(2))
                    self.signals.progress.emit(cur_f, total_f)

                # Postęp DEKODOWANIA: "  Zdekodowano klatkę X/Y [I]"
                # (inny format niż kodowanie — osobna gałąź)
                md = re.search(r'Zdekodowano\s+klatkę\s+(\d+)/(\d+)', line)
                if md:
                    cur_f, total_f = int(md.group(1)), int(md.group(2))
                    self.signals.progress.emit(cur_f, total_f)

                # Typy klatek — enkoder drukuje "P-Frame...", dekoder "[P]"
                if   'I-Frame' in line:                     fstats['I'] += 1
                elif 'P-Frame' in line:                     fstats['P'] += 1
                elif 'B-Frame' in line:                     fstats['B'] += 1
                elif re.search(r'Zdekodowano.*\[I\]', line): fstats['I'] += 1
                elif re.search(r'Zdekodowano.*\[P\]', line): fstats['P'] += 1
                elif re.search(r'Zdekodowano.*\[B\]', line): fstats['B'] += 1

                ms = re.search(r'SKIP:\s*(\d+)/(\d+).*DETAIL:\s*(\d+)', line)
                if ms:
                    fstats['skip_total']   += int(ms.group(1))
                    fstats['detail_total'] += int(ms.group(3))
                if 'CIECIE SCENY' in line: fstats['scene_cuts'] += 1
            proc.wait()
            elapsed = time.time()-start
            in_sz  = os.path.getsize(p['input'])  if os.path.exists(p['input'])  else 0
            out_sz = os.path.getsize(p['output']) if os.path.exists(p['output']) else 0
            self.signals.finished.emit({
                **fstats,
                'czas': elapsed, 'fps_proc': cur_f/elapsed if elapsed>0 else 0,
                'klatki_total': cur_f,
                'rozmiar_we': in_sz, 'rozmiar_wy': out_sz,
                'ratio': in_sz/out_sz if out_sz>0 else 0,
                'oszcz_proc': (1-out_sz/in_sz)*100 if in_sz>0 else 0,
                'mode': self.mode, 'retcode': proc.returncode,
            })
        except FileNotFoundError as e:
            self.signals.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GŁÓWNE OKNO
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOP TOPÓW CODEC v2 — GUI Analizator & Player")
        self.setMinimumSize(1200, 780)
        self.resize(1450, 900)

        self._worker        = None
        self._auto_worker   = None
        self._elapsed_timer = QElapsedTimer()
        self._tick_timer    = QTimer(self)
        self._tick_timer.timeout.connect(self._tick_elapsed)
        self._codec_path    = self._find_codec()

        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        self._update_output_placeholder()
        # Załaduj speedup w tle — żeby był gotowy zanim użytkownik otworzy plik
        _load_speedup()

    # ── Lokalizacja kodeka ─────────────────────────────────────────────────
    def _find_codec(self) -> str:
        for c in [
            Path(__file__).parent / "toptopuwcodec_v1.py",
            Path("/mnt/user-data/uploads/toptopuwcodec_v1.py"),
            Path("toptopuwcodec_v1.py"),
        ]:
            if c.exists(): return str(c)
        return ""

    # ── Budowa UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([420, 1000])
        root.addWidget(splitter, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._set_status("Gotowy", C['dim'])
        perm = QLabel(f"  Kodek: {self._codec_path or 'BRAK — wybierz ręcznie'}")
        perm.setStyleSheet(f"color:{C['dim']};")
        self.status_bar.addPermanentWidget(perm)

    def _build_header(self):
        f = QFrame()
        f.setStyleSheet(f"""QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 {C['bg3']},stop:1 {C['bg2']});
            border:1px solid {C['accent']};border-radius:8px;padding:4px;}}""")
        lay = QHBoxLayout(f); lay.setContentsMargins(14, 8, 14, 8)
        t = QLabel("⬡ TOP TOPÓW CODEC v2 — Analizator · Player · Auto-Parametry")
        t.setStyleSheet(f"font-size:17px;font-weight:bold;color:{C['accent']};border:none;")
        lay.addWidget(t); lay.addStretch()
        s = QLabel("DCT · B-Frames · Sub-pixel · Zstd-22 · PyQt6")
        s.setStyleSheet(f"color:{C['dim']};font-size:11px;border:none;")
        lay.addWidget(s)
        return f

    # ── LEWA KOLUMNA ───────────────────────────────────────────────────────
    def _build_left(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(6)
        lay.addWidget(self._build_io_group())
        lay.addWidget(self._build_mode_group())
        lay.addWidget(self._build_params_group())
        lay.addWidget(self._build_control_group())
        lay.addStretch()
        return w

    def _build_io_group(self):
        g = QGroupBox("📂  Pliki")
        lay = QGridLayout(g); lay.setSpacing(5)

        lay.addWidget(QLabel("Wejście:"), 0, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Plik .mp4 / .toptop …")
        self.input_edit.textChanged.connect(self._on_input_changed)
        lay.addWidget(self.input_edit, 0, 1)
        b_in = QPushButton("…"); b_in.setFixedWidth(30); b_in.clicked.connect(self._browse_input)
        lay.addWidget(b_in, 0, 2)

        # Przycisk auto-analizy obok pola wejścia
        self.auto_btn = QPushButton("🧮 Auto-Parametry")
        self.auto_btn.setObjectName("auto_btn")
        self.auto_btn.setToolTip(
            "Analizuje wideo i automatycznie dobiera optymalne\n"
            "Q_Y, Q_C, zasięg ruchu, B-frames, keyframe interval…")
        self.auto_btn.clicked.connect(self._run_auto_params)
        lay.addWidget(self.auto_btn, 0, 3)

        lay.addWidget(QLabel("Wyjście:"), 1, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Auto lub wybierz …")
        lay.addWidget(self.output_edit, 1, 1)
        b_out = QPushButton("…"); b_out.setFixedWidth(30); b_out.clicked.connect(self._browse_output)
        lay.addWidget(b_out, 1, 2)

        lay.addWidget(QLabel("Kodek:"), 2, 0)
        self.codec_edit = QLineEdit(self._codec_path)
        self.codec_edit.setPlaceholderText("toptopuwcodec_v1.py …")
        self.codec_edit.textChanged.connect(lambda t: setattr(self,'_codec_path',t))
        lay.addWidget(self.codec_edit, 2, 1)
        b_c = QPushButton("…"); b_c.setFixedWidth(30); b_c.clicked.connect(self._browse_codec)
        lay.addWidget(b_c, 2, 2)

        lay.setColumnStretch(1, 1)
        return g

    def _build_mode_group(self):
        g = QGroupBox("🔄  Tryb")
        lay = QHBoxLayout(g)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Kodowanie (encode)", "Dekodowanie (decode)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        lay.addWidget(self.mode_combo)
        lay.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Brak (ręczne)", "1 — Mały plik", "2 — Adaptive Q", "3 — Jakość"])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        lay.addWidget(self.preset_combo)
        return g

    def _build_params_group(self):
        g = QGroupBox("⚙  Parametry kodowania")
        self.params_widget = g
        lay = QGridLayout(g); lay.setSpacing(5)
        row = 0

        def row_w(label, widget, tooltip=""):
            lay.addWidget(QLabel(label), row, 0)
            if tooltip: widget.setToolTip(tooltip)
            lay.addWidget(widget, row, 1)

        self.q_y_spin = QDoubleSpinBox(); self.q_y_spin.setRange(1,100); self.q_y_spin.setValue(22); self.q_y_spin.setSingleStep(1)
        row_w("Q_Y (luma):", self.q_y_spin, "Kwantyzacja luminancji. Wyżej → mniejszy plik, niższa jakość."); row+=1

        self.q_c_spin = QDoubleSpinBox(); self.q_c_spin.setRange(1,150); self.q_c_spin.setValue(40); self.q_c_spin.setSingleStep(1)
        row_w("Q_C (chroma):", self.q_c_spin, "Kwantyzacja chrominancji (kolor)."); row+=1

        self.search_spin = QSpinBox(); self.search_spin.setRange(4,128); self.search_spin.setValue(24)
        row_w("Zasięg ruchu (px):", self.search_spin, "Zasięg TSS w estymacji ruchu."); row+=1

        self.kf_spin = QSpinBox(); self.kf_spin.setRange(0,9999); self.kf_spin.setValue(50)
        row_w("I-klatka co N:", self.kf_spin, "GOP length. 0 = tylko pierwsza klatka."); row+=1

        self.scene_spin = QDoubleSpinBox(); self.scene_spin.setRange(0,200); self.scene_spin.setValue(35)
        row_w("Scene cut MAD:", self.scene_spin, "Próg cięcia sceny. 0 = wyłączone."); row+=1

        self.frames_spin = QSpinBox(); self.frames_spin.setRange(1,999999); self.frames_spin.setValue(30)
        row_w("Maks. klatek:", self.frames_spin); row+=1

        self.full_check     = QCheckBox("Koduj cały plik (-f)")
        self.subpixel_check = QCheckBox("Sub-pixel ¼px");  self.subpixel_check.setChecked(True)
        self.bframes_check  = QCheckBox("B-Frames");        self.bframes_check.setChecked(True)
        self.adaptive_check = QCheckBox("Adaptacyjna Q")
        lay.addWidget(self.full_check,     row,0,1,2); row+=1
        lay.addWidget(self.subpixel_check, row,0); lay.addWidget(self.bframes_check, row,1); row+=1
        lay.addWidget(self.adaptive_check, row,0,1,2); row+=1

        # Wskaźnik auto-analizy
        self._auto_indicator = QLabel("")
        self._auto_indicator.setStyleSheet(f"color:{C['purple']};font-size:11px;")
        lay.addWidget(self._auto_indicator, row, 0, 1, 2); row+=1

        lay.setColumnStretch(1,1)
        return g

    def _build_control_group(self):
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(5)

        prog_g = QGroupBox("📊  Postęp")
        pg_lay = QVBoxLayout(prog_g)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Klatka %v / %m  (%p%)")
        pg_lay.addWidget(self.progress_bar)
        self.time_label = QLabel("Czas: — | Prędkość: — | Pozostało: —")
        self.time_label.setStyleSheet(f"color:{C['dim']};font-size:11px;")
        pg_lay.addWidget(self.time_label)
        lay.addWidget(prog_g)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  START"); self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn  = QPushButton("■  STOP");  self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop_processing)
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.stop_btn)
        lay.addLayout(btn_row)
        return w

    # ── PRAWA KOLUMNA (tabs) ───────────────────────────────────────────────
    def _build_right(self):
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(4,0,0,0); lay.setSpacing(0)
        self._tabs = QTabWidget()

        # ── Tab 1: Log ────────────────────────────────────────────────────
        log_w = QWidget(); ll = QVBoxLayout(log_w); ll.setContentsMargins(4,4,4,4)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        LogHighlighter(self.log_edit.document())
        log_btn = QHBoxLayout()
        b_clr = QPushButton("🗑 Wyczyść"); b_clr.clicked.connect(self.log_edit.clear)
        b_sav = QPushButton("💾 Zapisz");  b_sav.clicked.connect(self._save_log)
        log_btn.addWidget(b_clr); log_btn.addWidget(b_sav); log_btn.addStretch()
        ll.addWidget(self.log_edit); ll.addLayout(log_btn)
        self._tabs.addTab(log_w, "📋  Log")

        # ── Tab 2: Player ─────────────────────────────────────────────────
        self.player = ToptopPlayerWidget(lambda: self._codec_path)
        self._tabs.addTab(self.player, "▶  Player")

        # ── Tab 3: Statystyki ─────────────────────────────────────────────
        self.stats_panel = StatsPanel()
        sc = QScrollArea(); sc.setWidget(self.stats_panel)
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._tabs.addTab(sc, "📊  Statystyki")

        # ── Tab 4: Analiza pliku ──────────────────────────────────────────
        fi_w = QWidget(); fi_l = QVBoxLayout(fi_w); fi_l.setContentsMargins(4,4,4,4)
        b_ana = QPushButton("🔍 Analizuj plik wejściowy")
        b_ana.clicked.connect(self._run_file_analysis)
        self.file_info_text = QTextEdit(); self.file_info_text.setReadOnly(True)
        self.file_info_text.setFont(QFont("Consolas",12))
        fi_l.addWidget(b_ana); fi_l.addWidget(self.file_info_text)
        self._tabs.addTab(fi_w, "🔍  Analiza pliku")

        # ── Tab 5: Pomoc ──────────────────────────────────────────────────
        help_w = QTextEdit(); help_w.setReadOnly(True)
        help_w.setMarkdown(HELP_MD)
        self._tabs.addTab(help_w, "❓  Pomoc")

        # ── Tab 6: CPU & Speedup ──────────────────────────────────────────
        cpu_w = QWidget(); cpu_l = QVBoxLayout(cpu_w); cpu_l.setContentsMargins(4,4,4,4)
        self._cpu_text = QTextEdit(); self._cpu_text.setReadOnly(True)
        self._cpu_text.setFont(QFont("Consolas", 12))
        btn_cpu_row = QHBoxLayout()
        btn_diag = QPushButton("🔬 Diagnostyka CPU & Speedup")
        btn_diag.clicked.connect(self._run_cpu_diag)
        btn_install = QPushButton("📦 Pokaż komendy instalacji")
        btn_install.clicked.connect(self._show_install_cmds)
        btn_cpu_row.addWidget(btn_diag); btn_cpu_row.addWidget(btn_install); btn_cpu_row.addStretch()
        cpu_l.addLayout(btn_cpu_row); cpu_l.addWidget(self._cpu_text)
        self._tabs.addTab(cpu_w, "⚡  CPU & Speedup")

        lay.addWidget(self._tabs)
        return w

    # ══════════════════════════════════════════════════════════════════════
    # AUTO-PARAMETRY
    # ══════════════════════════════════════════════════════════════════════

    def _run_auto_params(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Auto-analiza", "Podaj istniejący plik wejściowy.")
            return
        if Path(path).suffix.lower() == '.toptop':
            QMessageBox.information(self, "Auto-analiza",
                "Auto-analiza działa na plikach wideo (mp4, avi…),\n"
                "nie na plikach .toptop.")
            return

        self.auto_btn.setEnabled(False)
        self.auto_btn.setText("⏳ Analizuję…")
        self._auto_indicator.setText("🧮 Trwa analiza wideo…")
        self._log("═══ AUTO-ANALIZA PARAMETRÓW ═══\n")
        self._tabs.setCurrentIndex(0)  # pokaż log

        sigs = AutoParamSignals()
        sigs.progress.connect(lambda msg: self._log(msg+'\n'))
        sigs.finished.connect(self._on_auto_finished)
        sigs.error.connect(self._on_auto_error)
        self._auto_signals = sigs

        self._auto_worker = AutoParamAnalyzer(sigs, path)
        self._auto_worker.start()

    def _on_auto_finished(self, res: dict):
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("🧮 Auto-Parametry")

        # Wstaw parametry do UI
        self.q_y_spin.setValue(res['q_y'])
        self.q_c_spin.setValue(res['q_c'])
        self.search_spin.setValue(res['search_range'])
        self.kf_spin.setValue(res['keyframe_interval'])
        self.scene_spin.setValue(res['scene_cut'])
        self.subpixel_check.setChecked(res['subpixel'])
        self.bframes_check.setChecked(res['bframes'])
        self.adaptive_check.setChecked(res['adaptive_q'])
        self.preset_combo.setCurrentIndex(0)  # "Brak" – ręczne

        # Aktualizuj wskaźnik
        mad = res.get('_avg_mad', 0)
        mot = "statyczny" if mad < 3 else "mały" if mad < 10 else "średni" if mad < 25 else "duży"
        self._auto_indicator.setText(
            f"✅ Auto: Q_Y={res['q_y']} Q_C={res['q_c']} "
            f"SR={res['search_range']} | Ruch: {mot} (MAD={mad:.1f})")

        # Log podsumowanie
        self._log(
            f"\n✓ REKOMENDACJE AUTO-ANALIZY:\n"
            f"   Q_Y={res['q_y']}  Q_C={res['q_c']}\n"
            f"   Zasięg={res['search_range']}px  "
            f"Keyframe co {res['keyframe_interval']}  "
            f"Scene-cut={res['scene_cut']}\n"
            f"   B-frames={'ON' if res['bframes'] else 'OFF'}  "
            f"Sub-pixel={'ON' if res['subpixel'] else 'OFF'}  "
            f"Adaptive={'ON' if res['adaptive_q'] else 'OFF'}\n"
            f"   Ruch avg={res.get('_avg_mad',0):.2f} max={res.get('_max_mad',0):.2f}  "
            f"Var={res.get('_avg_var',0):.0f}  "
            f"Est.kompresja≈{res.get('_est_compression',0):.1f}×\n"
        )

        # Wstaw do statystyk
        self.stats_panel.set_auto_analysis(res)
        self._tabs.setCurrentIndex(2)  # pokaż statystyki
        self._set_status("✓ Auto-parametry dobrane i wstawione do formularza", C['green'])

    def _on_auto_error(self, msg: str):
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("🧮 Auto-Parametry")
        self._auto_indicator.setText(f"✗ Błąd auto-analizy")
        self._log(f"\n✗ BŁĄD AUTO-ANALIZY:\n{msg}\n")
        self._set_status("✗ Błąd auto-analizy", C['red'])

    # ══════════════════════════════════════════════════════════════════════
    # KODOWANIE / DEKODOWANIE
    # ══════════════════════════════════════════════════════════════════════

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Plik wejściowy","",
            "Wideo / toptop (*.mp4 *.avi *.mkv *.mov *.webm *.toptop);;Wszystkie (*)")
        if path: self.input_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Plik wyjściowy","",
            "TopTop (*.toptop);;MP4 (*.mp4);;Wszystkie (*)")
        if path: self.output_edit.setText(path)

    def _browse_codec(self):
        path, _ = QFileDialog.getOpenFileName(self, "Plik kodeka","","Python (*.py)")
        if path: self.codec_edit.setText(path); self._codec_path = path

    def _on_input_changed(self, text):
        self._update_output_placeholder()

    def _update_output_placeholder(self):
        inp = self.input_edit.text().strip()
        if not inp: return
        p = Path(inp)
        ext = '.toptop' if self.mode_combo.currentIndex()==0 else '.mp4'
        if not self.output_edit.text().strip():
            self.output_edit.setPlaceholderText(str(p.with_suffix(ext)))

    def _on_mode_changed(self, idx):
        self.params_widget.setEnabled(idx==0)
        self._update_output_placeholder()

    def _apply_preset(self, idx):
        if idx==1: self.q_y_spin.setValue(32); self.q_c_spin.setValue(55)
        elif idx==2: self.adaptive_check.setChecked(True)
        elif idx==3: self.q_y_spin.setValue(16); self.q_c_spin.setValue(30); self.search_spin.setValue(48)

    def _start_processing(self):
        inp   = self.input_edit.text().strip()
        out   = self.output_edit.text().strip() or self.output_edit.placeholderText()
        codec = self._codec_path

        for label, val in [("Wejście",inp),("Wyjście",out),("Kodek",codec)]:
            if not val or (label!="Wyjście" and not os.path.exists(val)):
                QMessageBox.warning(self,"Błąd",f"{label}: {val or 'brak'}"); return

        mode   = 'encode' if self.mode_combo.currentIndex()==0 else 'decode'
        params = dict(
            input=inp, output=out,
            q_y=self.q_y_spin.value(), q_c=self.q_c_spin.value(),
            search_range=self.search_spin.value(),
            keyframe_interval=self.kf_spin.value(),
            scene_cut=self.scene_spin.value(),
            max_frames=self.frames_spin.value(),
            full=self.full_check.isChecked(),
            subpixel=self.subpixel_check.isChecked(),
            bframes=self.bframes_check.isChecked(),
            adaptive_q=self.adaptive_check.isChecked(),
            preset=self.preset_combo.currentIndex(),
        )

        self.progress_bar.setMaximum(params['max_frames'] if not params['full'] else 0)
        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self._log(f"═══ {mode.upper()} | {os.path.basename(inp)} ═══\n")

        sigs = CodecSignals()
        sigs.log_line.connect(self._log)
        sigs.progress.connect(self._on_progress)
        sigs.finished.connect(self._on_finished)
        sigs.error.connect(self._on_error)
        self._codec_signals = sigs

        self._worker = CodecWorker(sigs, codec, mode, params)
        self._worker.start()

        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._elapsed_timer.start(); self._tick_timer.start(500)
        self._set_status(f"⏳ {mode}…", C['yellow'])
        self._tabs.setCurrentIndex(0)

    def _stop_processing(self):
        if self._worker: self._worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_progress(self, cur, tot):
        if tot>0: self.progress_bar.setMaximum(tot); self.progress_bar.setValue(cur)

    def _tick_elapsed(self):
        s   = self._elapsed_timer.elapsed()/1000.0
        cur = self.progress_bar.value()
        tot = self.progress_bar.maximum() or 1
        fps = cur/s if s>0 and cur>0 else 0
        rem = (tot-cur)/fps if fps>0 else 0
        self.time_label.setText(
            f"Czas: {s:.1f}s  |  {fps:.1f} kl/s  |  Pozostało: {rem:.0f}s")

    def _on_finished(self, stats):
        self._tick_timer.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(self.progress_bar.maximum())
        ok = stats.get('retcode',0)==0
        self._log(
            f"\n{'✓' if ok else '✗'} ZAKOŃCZONO │ "
            f"Czas: {stats.get('czas',0):.2f}s │ "
            f"{stats.get('fps_proc',0):.1f} kl/s │ "
            f"{_fmt_size(stats.get('rozmiar_we',0))} → "
            f"{_fmt_size(stats.get('rozmiar_wy',0))} │ "
            f"Ratio: {stats.get('ratio',0):.2f}×\n"
        )
        self._set_status(
            f"{'✓' if ok else '✗'} "
            f"{_fmt_size(stats.get('rozmiar_wy',0))} "
            f"({stats.get('oszcz_proc',0):.1f}%)",
            C['green'] if ok else C['red'])
        self.stats_panel.set_codec_stats(stats)

        # Jeśli po dekodowaniu → zaproponuj otwarcie w playerze
        if stats.get('mode') == 'decode':
            out = self.output_edit.text().strip() or self.output_edit.placeholderText()
            if out.endswith('.toptop') and os.path.exists(out):
                self._offer_player(out)

        # Jeśli po kodowaniu → zaproponuj otwarcie wyjścia w playerze
        if stats.get('mode') == 'encode':
            out = self.output_edit.text().strip() or self.output_edit.placeholderText()
            if out.endswith('.toptop') and os.path.exists(out):
                self._offer_player(out)

    def _offer_player(self, path: str):
        ans = QMessageBox.question(self, "Otwórz w playerze",
            f"Zakodowany plik jest gotowy:\n{path}\n\nOtworzyć w playerze?")
        if ans == QMessageBox.StandardButton.Yes:
            self._tabs.setCurrentIndex(1)   # Player tab
            self.player.load_file(path)

    def _on_error(self, msg):
        self._tick_timer.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._log(f"\n✗ BŁĄD:\n{msg}\n")
        self._set_status(f"✗ {msg[:60]}", C['red'])

    def _log(self, text):
        cur = self.log_edit.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text)
        self.log_edit.setTextCursor(cur)
        self.log_edit.ensureCursorVisible()

    def _set_status(self, msg, color=None):
        self.status_bar.showMessage(msg)
        if color:
            self.status_bar.setStyleSheet(
                f"QStatusBar{{color:{color};background:{C['bg2']};}}")

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(self,"Zapisz log","codec_log.txt","Tekst (*.txt)")
        if path:
            with open(path,'w',encoding='utf-8') as f: f.write(self.log_edit.toPlainText())
            self._set_status(f"Log: {path}", C['green'])

    def _run_cpu_diag(self):
        """Uruchamia diagnostykę speedup i wyświetla wyniki."""
        self._cpu_text.setPlainText("⏳ Uruchamianie diagnostyki (benchmark VED upsample…)")
        QApplication.processEvents()

        # Upewnij się że próbowaliśmy załadować speedup (może nie być załadowany
        # jeśli użytkownik nie otwierał jeszcze żadnego pliku .toptop)
        _load_speedup()

        lines = [
            "╔══ DIAGNOSTYKA CPU & PRZYSPIESZEŃ ══╗", "",
            f"  Speedup moduł: {_speedup_info}", "",
        ]

        # Podstawowe info CPU
        import multiprocessing
        n_log = multiprocessing.cpu_count()
        lines += [f"  Wątki logiczne: {n_log}"]
        try:
            import psutil
            n_phys = psutil.cpu_count(logical=False)
            freq   = psutil.cpu_freq()
            lines += [
                f"  Rdzenie fizyczne: {n_phys}",
                f"  Taktowanie: {freq.current:.0f} MHz  (max: {freq.max:.0f} MHz)",
            ]
        except ImportError:
            lines += ["  (zainstaluj psutil dla pełnych info: pip install psutil)"]

        # numpy config
        try:
            np_info = np.__config__.blas_opt_info
            libs = np_info.get('libraries', ['?'])
            lines += [f"  NumPy BLAS: {libs}"]
        except Exception:
            pass

        try:
            np_ver = np.__version__
            import scipy; sc_ver = scipy.__version__
            lines += [f"  NumPy {np_ver}  SciPy {sc_ver}"]
        except Exception:
            pass

        # AVX2 check
        try:
            import cpuinfo
            info  = cpuinfo.get_cpu_info()
            flags = info.get("flags", [])
            brand = info.get("brand_raw", "?")
            lines += [
                f"  CPU: {brand}",
                f"  AVX2: {'✓' if 'avx2' in flags else '✗'}  "
                f"FMA3: {'✓' if 'fma' in flags else '✗'}  "
                f"SSE4.2: {'✓' if 'sse4_2' in flags else '✗'}",
            ]
        except ImportError:
            lines += ["  (zainstaluj py-cpuinfo: pip install py-cpuinfo)"]

        lines += [""]

        # Speedup diagnostics
        if _speedup is not None:
            try:
                diag = _speedup.diagnostics()
                lines += [diag, ""]
            except Exception as e:
                lines += [f"  Błąd diagnostyki: {e}", ""]
        else:
            lines += [
                "  ⚠ Moduł toptopuw_speedup.py nie jest załadowany.",
                f"  Przyczyna: {_speedup_info}",
                "  Upewnij się, że toptopuw_speedup.py jest w tym samym katalogu co GUI.",
                "",
            ]

        # Benchmark VED upsample
        lines += ["─── Benchmark VED upsample ───", ""]
        try:
            import time

            test_uv = np.random.rand(270, 480).astype(np.float32)

            # Pobierz funkcję z załadowanego modułu albo przeszukaj dysk
            _fast = None
            if _speedup is not None:
                _fast = getattr(_speedup, '_ved_upsample_fast', None)
            if _fast is None:
                import importlib.util as _ilu2
                for _sp_path in [
                    Path("toptopuw_speedup.py"),
                    Path(__file__).parent / "toptopuw_speedup.py",
                    Path(sys.argv[0]).parent / "toptopuw_speedup.py",
                ]:
                    if _sp_path.exists():
                        try:
                            _spec2 = _ilu2.spec_from_file_location("_sp_tmp", str(_sp_path))
                            _mod2 = _ilu2.module_from_spec(_spec2)
                            _spec2.loader.exec_module(_mod2)
                            _fast = _mod2._ved_upsample_fast
                            break
                        except Exception:
                            pass

            if _fast is None:
                raise ImportError("_ved_upsample_fast niedostępne — załaduj toptopuw_speedup.py")

            REPS = 20
            t0 = time.perf_counter()
            for _ in range(REPS):
                _fast(test_uv)
            t_fast = (time.perf_counter() - t0) / REPS * 1000

            lines += [
                f"  VED fast (numpy):  {t_fast:.2f} ms  →  ~{1000/t_fast:.0f} upsamp/s",
                f"  Na klatkę (U+V):   {t_fast*2:.2f} ms",
                f"  Szacunkowy FPS:    ~{1000/(t_fast*2 + 8):.1f} kl/s (samo dekodowanie UV+YUV)",
            ]
        except Exception as e:
            lines += [f"  Błąd benchmarku: {e}"]

        # Benchmark batch DCT
        dct_backend = getattr(_speedup, 'FFTW_INFO', 'scipy.fftpack') if _speedup else 'scipy.fftpack'
        lines += ["", f"─── Benchmark batch IDCT (16×16, 1080p ~8000 bloków) [{dct_backend}] ───", ""]
        try:
            import time
            # Warm-up (pierwsze wywołanie FFTW planuje transform — nie liczymy go)
            dummy = np.random.rand(8000, 16, 16).astype(np.float32)

            if _speedup is not None:
                _speedup.batch_idct2(dummy)  # warm-up
                REPS = 5
                t0 = time.perf_counter()
                for _ in range(REPS):
                    _speedup.batch_idct2(dummy)
                t_batch = (time.perf_counter() - t0) / REPS * 1000

                # Per-blok — czyste scipy (nie monkey-patchowane)
                import importlib, scipy.fftpack as _sfp
                _sidct = _sfp.idct
                t1 = time.perf_counter()
                for b in dummy[:200]:
                    _sidct(_sidct(b, type=2, norm="ortho"), type=2, norm="ortho")
                t_per = (time.perf_counter() - t1) / 200 * 8000

                lines += [
                    f"  Per-blok (scipy):  {t_per:.0f} ms / klatkę  (ekstrapolacja z 200 bloków)",
                    f"  Batch ({dct_backend}): {t_batch:.1f} ms / klatkę",
                    f"  Przyspieszenie:    {t_per/t_batch:.1f}×",
                ]
            else:
                lines += ["  (speedup niedostępny)"]
        except Exception as e:
            lines += [f"  Błąd: {e}"]

        lines += ["", "─── Zmienne środowiskowe BLAS ───", ""]
        for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                  "PYTHONUNBUFFERED","NUMEXPR_NUM_THREADS"):
            val = os.environ.get(v, "(nieustawiona)")
            lines += [f"  {v} = {val}"]

        self._cpu_text.setPlainText("\n".join(lines))
        self._tabs.setCurrentWidget(self._tabs.widget(self._tabs.count()-1))

    def _show_install_cmds(self):
        cmds = """# ── Instalacja opcjonalnych przyspieszeń ──────────────────────────────

# pyfftw — FFTW3 binding, ~2-4× szybszy DCT niż scipy na AVX2
# Wymaga libfftw3 w systemie:
#   Ubuntu/Debian:  sudo apt install libfftw3-dev
#   Fedora/RHEL:    sudo dnf install fftw-devel
pip install pyfftw --break-system-packages

# numba — JIT compiler dla interpolate_subpixel (hot-path P-frame)
pip install numba --break-system-packages

# psutil — informacje o CPU (rdzenie fizyczne, taktowanie)
pip install psutil --break-system-packages

# py-cpuinfo — detekcja AVX2/FMA/SSE flagszacku CPU
pip install py-cpuinfo --break-system-packages

# numpy z MKL (Intel — najlepszy dla Xeonów):
pip install numpy --extra-index-url https://pypi.anaconda.org/intel/simple --break-system-packages
# lub przez conda:
# conda install numpy mkl

# Sprawdź czy numpy widzi AVX2:
python -c "import numpy as np; np.show_config()"

# ── Zmienne środowiskowe (ustaw przed uruchomieniem GUI) ──────────────
# Zastąp N liczbą fizycznych rdzeni Xeona (np. 20 dla E5-2698 v4)
export OMP_NUM_THREADS=N
export MKL_NUM_THREADS=N
export OPENBLAS_NUM_THREADS=N
export PYTHONUNBUFFERED=1
python toptopuw_gui_v2.py"""
        self._cpu_text.setPlainText(cmds)

    def _run_file_analysis(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            self.file_info_text.setPlainText("Wybierz plik wejściowy."); return
        self.file_info_text.setPlainText("⏳ Analizowanie…")
        QApplication.processEvents()
        lines = [f"{'═'*62}",f"ANALIZA: {os.path.basename(path)}",f"{'═'*62}",""]
        ext = Path(path).suffix.lower()

        if ext=='.toptop':
            try:
                import zstandard as zstd
                sz = os.path.getsize(path)
                lines.append(f"Typ: Strumień .toptop  ({_fmt_size(sz)})")
                lines.append("")
                with open(path,'rb') as f:
                    raw = zstd.ZstdDecompressor().stream_reader(f).read()
                w_r,h_r = struct.unpack_from('>HH',raw,0)
                lines.append(f"  Rozdzielczość:  {w_r}×{h_r}")
                # Policz klatki
                offset=4; ci=cp=cb=0
                y_size=h_r*w_r*2; uv_size=(h_r//2)*(w_r//2)*2
                while offset < len(raw):
                    ft=raw[offset:offset+1]; offset+=1
                    if ft==b'I':
                        offset+=y_size+uv_size*2; ci+=1
                    elif ft in (b'P',b'B'):
                        cols2,rows2=struct.unpack_from('>HH',raw,offset); offset+=4
                        if cols2>0 and rows2>0:
                            nb2=rows2*cols2; bsz=(nb2+7)//8
                            bmp=raw[offset:offset+bsz]; offset+=bsz
                            sz_y=16*16*2; sz_c=8*8*2
                            for ri in range(rows2):
                                for ci2 in range(cols2):
                                    idx=ri*cols2+ci2
                                    if bmp[idx>>3]&(1<<(7-(idx&7))):
                                        offset+=4+sz_y+sz_c*2
                        if ft==b'P': cp+=1
                        else: cb+=1
                    else: break
                total=ci+cp+cb
                lines+=[f"  Klatki I:       {ci}",f"  Klatki P:       {cp}",
                         f"  Klatki B:       {cb}",f"  Łącznie:        {total}",
                         f"  Kompresja Zstd: {len(raw)//1024} KB surowych → {sz//1024} KB na dysku"]
            except Exception as e:
                lines.append(f"Błąd: {e}")
        else:
            if IMAGEIO_OK:
                try:
                    props = iio.improps(path, plugin="pyav")
                    sh = props.shape
                    if len(sh)==4: n_f,h_v,w_v,_=sh
                    elif len(sh)==3: h_v,w_v,_=sh; n_f=1
                    else: h_v=w_v=n_f=0
                    fps_v = _get_fps(props, path)
                    lines+=[
                        f"  Rozdzielczość:  {w_v}×{h_v}",
                        f"  Klatek:         {n_f}",
                        f"  FPS:            {fps_v:.3f}",
                        f"  Czas:           {n_f/fps_v:.2f} s" if fps_v>0 else "  Czas:           ?",
                        f"  Rozmiar pliku:  {_fmt_size(os.path.getsize(path))}",
                        f"  Bloków 16×16:   {(w_v//16)*(h_v//16):,} / klatkę",
                    ]
                    lines+=["","─── Szacunki dla kodowania ───",
                        f"  RAM / klatkę ≈  {(w_v//16)*(h_v//16)*512*3//1024//1024:.1f} MB"]
                except Exception as e:
                    lines.append(f"Błąd imageio: {e}")
            else:
                lines.append("imageio niedostępne — zainstaluj imageio[pyav]")

        self.file_info_text.setPlainText('\n'.join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# TEKST POMOCY
# ══════════════════════════════════════════════════════════════════════════════

HELP_MD = """
# TOP TOPÓW CODEC v2 — Pomoc

## Player .toptop

Zakładka **▶ Player** dekoduje plik `.toptop` bezpośrednio w pamięci
(importując moduł kodeka, bez pliku pośredniego).

**Kontrolki:**
- `⬇ Wczytaj & Dekoduj` — dekoduje w tle, klatki pojawiają się na żywo
- `▶ / ⏸` — play/pause, `⏮/⏭` — poprzednia/następna klatka
- `⏹` — stop i powrót do klatki 0
- Slider — scrubbing do dowolnej klatki
- **FPS** — regulacja prędkości odtwarzania (1–120)
- Kolorowy badge (żółty=I, zielony=P, pomarańczowy=B) nakładany na obraz

---

## Auto-Parametry 🧮

Przycisk **Auto-Parametry** analizuje wideo wejściowe:

1. Próbkuje ~16 klatek równomiernie rozłożonych
2. Oblicza:
   - **MAD** (motion) — średni i maksymalny ruch między klatkami
   - **Wariancja** — bogactwo tekstury / detali
   - Rozdzielczość, FPS, liczbę bloków
3. Na tej podstawie dobiera:

| Metryka | Wynik |
|---|---|
| Mały ruch (MAD < 3) | Q_Y niższe, SR mniejszy, B-frames ON |
| Duży ruch (MAD > 20) | Q_Y wyższe, SR większy, Adaptive ON |
| Bogata tekstura | Q_Y niższe (-2) |
| Płaskie tło | Q_Y wyższe (+2) |
| Duże wahania ruchu | Adaptive Q ON |
| Brak cięć sceny | Scene-cut wyłączony |

Wyniki trafiają bezpośrednio do formularza — można je ręcznie korygować.

---

## Parametry

| Param | Opis |
|---|---|
| **Q_Y** | Kwantyzacja luminancji (12–50). Wyżej = mniejszy plik. |
| **Q_C** | Kwantyzacja chrominancji. Oko mniej czułe → może być ~1.8×Q_Y. |
| **Zasięg ruchu** | Zasięg TSS w pikselach (16–48). Więcej = lepsza kompresja ruchu. |
| **I co N** | GOP length. 2–4 sekundy typowo. |
| **Scene cut** | MAD próg cięcia sceny. 0 = wyłączone. |
| **B-Frames** | Klatki wsteczne — ~10-20% zysk. Wyłącz przy bardzo dużym ruchu. |
| **Sub-pixel** | Precyzja ¼px — lepsza jakość ruchomych obiektów. |
| **Adaptive Q** | Auto-dostosowanie Q do złożoności każdej klatki. |

---

## Format .toptop (v2.7)

```
[4B: W uint16, H uint16 big-endian]  ← nagłówek
+ zstd-22 compressed {
  Per klatka: [1B: typ 'I'|'P'|'B']
  I: raw Y(h×w×2B) + U + V (int16 DCT)
  P/B: [2B cols][2B rows][bitmap SKIP/DETAIL][dane DETAIL...]
}
```
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TopTopow Analyzer v2")
    pal = QPalette()
    for role, col in [
        (QPalette.ColorRole.Window,     C['bg']),
        (QPalette.ColorRole.WindowText, C['text']),
        (QPalette.ColorRole.Base,       C['bg2']),
        (QPalette.ColorRole.Text,       C['text']),
        (QPalette.ColorRole.Button,     C['bg3']),
        (QPalette.ColorRole.ButtonText, C['text']),
        (QPalette.ColorRole.Highlight,  C['accent']),
    ]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
