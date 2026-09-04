import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QLineEdit, QPushButton, QFrame, QSizePolicy,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor
import html as _html
from md_render import md_to_qthtml as _md_to_qthtml

# ---- Palette: minimalist dark, professional ----
BG         = "#0d0d0f"   # app background
BG_ELEV    = "#16161a"   # elevated surfaces (header, input)
BG_USER    = "#2563eb"   # user bubble (clean blue)
BG_AI      = "#17171c"   # assistant bubble
BG_INPUT   = "#1d1d22"
BG_STATUS  = "#141419"   # status card
TEXT       = "#ececf0"
TEXT_DIM   = "#8a8a94"
TEXT_FAINT = "#5a5a64"
TEXT_USER  = "#ffffff"
ACCENT     = "#3b82f6"
GREEN      = "#10b981"
AMBER      = "#f59e0b"
BORDER     = "#26262d"
BORDER_LT  = "#33333d"

APP_NAME   = "ALEXANDRIA"
APP_TAG    = ""

FONT = "Noto Sans"


class LoadModelThread(QThread):
    done  = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, engine):
        super().__init__(); self.engine = engine
    def run(self):
        try:
            self.engine.load(); self.done.emit()
        except Exception as e:
            self.error.emit(str(e))


class InferenceThread(QThread):
    token  = pyqtSignal(str)
    status = pyqtSignal(str, str)
    done   = pyqtSignal(str)
    error  = pyqtSignal(str)
    def __init__(self, engine, query):
        super().__init__(); self.engine = engine; self.query = query
    def run(self):
        try:
            full = self.engine.generate_stream(
                self.query,
                token_callback  = lambda t: self.token.emit(t),
                status_callback = lambda m, c: self.status.emit(m, c),
            )
            self.done.emit(full)
        except Exception as e:
            self.error.emit(str(e))


class Bubble(QFrame):
    def __init__(self, text, role):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 3, 0, 3); row.setSpacing(0)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Preferred)
        self.label.setFont(QFont(FONT, 13))
        if role == "user":
            self.label.setStyleSheet(f"""
                color:{TEXT_USER};background:{BG_USER};
                border-radius:16px;border-bottom-right-radius:4px;
                padding:11px 16px;""")
            self.label.setLineHeight = None
            row.addStretch(3)
            row.addWidget(self.label, 7)
        elif role == "assistant":
            self.label.setStyleSheet(f"""
                color:{TEXT};background:{BG_AI};
                border-radius:16px;border-bottom-left-radius:4px;
                padding:13px 18px;border:1px solid {BORDER};""")
            row.addWidget(self.label, 9)
            row.addStretch(1)
        else:
            self.label.setFont(QFont(FONT, 10))
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setStyleSheet(f"color:{TEXT_FAINT};padding:4px 0;")
            row.addStretch(); row.addWidget(self.label); row.addStretch()

    def append_text(self, piece):
        self.label.setText(self.label.text() + piece)

    def stream_piece(self, piece):
        # Live formatting that KEEPS streaming visible: completed lines render
        # as formatted HTML; the in-progress last line shows as plain text and
        # updates on EVERY token (this is what was missing before).
        if not hasattr(self, "_raw"):
            self._raw = ""
            self._fmt_cache = ""
        self._raw += piece
        if "\n" in piece:
            nl = self._raw.rfind("\n")
            self._fmt_cache = _md_to_qthtml(self._raw[:nl])
        nl = self._raw.rfind("\n")
        tail = self._raw[nl + 1:] if nl >= 0 else self._raw
        sep = "<br>" if self._fmt_cache else ""
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setText(self._fmt_cache + sep + _html.escape(tail))

    def render_formatted(self, full=None):
        if full is not None:
            self._raw = full
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setText(_md_to_qthtml(getattr(self, "_raw", "")))


class StatusCard(QFrame):
    """Prominent live-status card shown while the system works.
    Displays the current pipeline stage with an animated pulse dot."""
    SEARCH_STAGES = ["Searching the knowledge base",
                     "Looking across the archives",
                     "Gathering sources",
                     "Reading through the material"]
    WRITE_STAGES  = ["Composing the answer"]

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 3, 0, 3); row.setSpacing(0)
        card = QFrame()
        card.setStyleSheet(f"""
            background:{BG_STATUS};border:1px solid {BORDER};
            border-radius:16px;border-bottom-left-radius:4px;""")
        cl = QHBoxLayout(card)
        cl.setContentsMargins(16, 12, 20, 12); cl.setSpacing(12)
        self.dot = QLabel("●")
        self.dot.setFont(QFont(FONT, 13))
        self.dot.setStyleSheet(f"color:{ACCENT};")
        self._stages = self.SEARCH_STAGES
        self.msg = QLabel("Searching the knowledge base")
        self.msg.setFont(QFont(FONT, 12, QFont.Weight.Medium))
        self.msg.setStyleSheet(f"color:{TEXT};")
        self.sub = QLabel("")
        self.sub.setFont(QFont(FONT, 10))
        self.sub.setStyleSheet(f"color:{TEXT_FAINT};")
        cl.addWidget(self.dot); cl.addWidget(self.msg); cl.addSpacing(4)
        cl.addWidget(self.sub)
        row.addWidget(card)
        row.addStretch(2)
        # pulse animation on the dot
        self._eff = QGraphicsOpacityEffect(self.dot)
        self.dot.setGraphicsEffect(self._eff)
        self._anim = QPropertyAnimation(self._eff, b"opacity")
        self._anim.setStartValue(1.0); self._anim.setEndValue(0.25)
        self._anim.setDuration(700)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setLoopCount(-1)
        # auto-advance through stages if the engine is quiet
        self._auto = QTimer(self); self._auto.timeout.connect(self._tick)
        self._stage = 0; self._auto.start(1500)
        self._anim.start()

    def set_status(self, msg, color):
        low = (msg or "").lower()
        col = {"green": GREEN, "amber": AMBER, "blue": ACCENT}.get(color, ACCENT)
        self.dot.setStyleSheet(f"color:{col};")
        if "generat" in low or "writ" in low or "compos" in low:
            self._stages = self.WRITE_STAGES
            self._stage = 0; self.msg.setText(self._stages[0])
            self._auto.start(1800)
        else:
            self._stages = self.SEARCH_STAGES
            self._stage = 0; self.msg.setText(self._stages[0])
            self._auto.start(1500)

    def _tick(self):
        self._stage = (self._stage + 1) % len(self._stages)
        self.msg.setText(self._stages[self._stage])

    def stop(self):
        self._anim.stop(); self._auto.stop()


class HistoryDialog(QDialog):
    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Past Conversations")
        self.setMinimumSize(520, 440)
        self.setStyleSheet(f"background:{BG_ELEV};color:{TEXT};")
        self.selected_id = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(12)
        lbl = QLabel("Resume a conversation")
        lbl.setFont(QFont(FONT, 13, QFont.Weight.Medium))
        lbl.setStyleSheet(f"color:{TEXT};")
        layout.addWidget(lbl)
        self.lw = QListWidget()
        self.lw.setFont(QFont(FONT, 12))
        self.lw.setStyleSheet(f"""
            QListWidget{{background:{BG};color:{TEXT};
                border:1px solid {BORDER};border-radius:10px;padding:4px;}}
            QListWidget::item:selected{{background:{ACCENT};color:white;
                border-radius:6px;}}
            QListWidget::item{{padding:11px;border-radius:6px;}}""")
        for s in sessions:
            it = QListWidgetItem(s["title"] + "…")
            it.setData(Qt.ItemDataRole.UserRole, s["id"])
            self.lw.addItem(it)
        self.lw.itemDoubleClicked.connect(lambda _: self._open())
        layout.addWidget(self.lw)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open |
            QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet(f"""
            QPushButton{{background:{BG_INPUT};color:{TEXT};border:1px solid {BORDER};
                border-radius:8px;padding:7px 18px;}}
            QPushButton:hover{{border-color:{BORDER_LT};}}""")
        btns.accepted.connect(self._open)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _open(self):
        it = self.lw.currentItem()
        if it:
            self.selected_id = it.data(Qt.ItemDataRole.UserRole)
            self.accept()


class ChatWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.inf_thread = None
        self.status_card = None
        self.stream_bubble = None
        self._build_window()
        self._build_ui()
        self._load_model()

    def _build_window(self):
        self.setWindowTitle(APP_NAME)
        self.showMaximized()
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};}}
            QScrollBar:vertical{{background:transparent;width:8px;margin:2px;}}
            QScrollBar::handle:vertical{{background:{BORDER_LT};
                border-radius:4px;min-height:30px;}}
            QScrollBar::handle:vertical:hover{{background:#44444e;}}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}""")

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0); vbox.setSpacing(0)

        # ---- Header ----
        top = QFrame(); top.setFixedHeight(60)
        top.setStyleSheet(f"background:{BG_ELEV};border-bottom:1px solid {BORDER};")
        tl = QHBoxLayout(top); tl.setContentsMargins(32, 0, 32, 0); tl.setSpacing(12)
        name = QLabel(APP_NAME)
        name.setFont(QFont(FONT, 16, QFont.Weight.DemiBold))
        name.setStyleSheet(f"color:{TEXT};letter-spacing:1px;")
        self.status_lbl = QLabel("● Starting")
        self.status_lbl.setFont(QFont(FONT, 11, QFont.Weight.Medium))
        self.status_lbl.setStyleSheet(f"color:{AMBER};")
        hist = self._chip("History"); hist.clicked.connect(self._open_history)
        newb = self._chip("New chat"); newb.clicked.connect(self._new_chat)
        tl.addWidget(name); tl.addStretch()
        tl.addWidget(self.status_lbl); tl.addSpacing(8)
        tl.addWidget(hist); tl.addWidget(newb)
        vbox.addWidget(top)

        # ---- Conversation (centered column) ----
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(f"QScrollArea{{background:{BG};border:none;}}")
        outer = QWidget(); outer.setStyleSheet(f"background:{BG};")
        ol = QHBoxLayout(outer); ol.setContentsMargins(0, 0, 0, 0)
        ol.addStretch(1)
        self.chat_w = QWidget(); self.chat_w.setStyleSheet(f"background:{BG};")
        self.chat_w.setMaximumWidth(760); self.chat_w.setMinimumWidth(420)
        self.chat_l = QVBoxLayout(self.chat_w)
        self.chat_l.setContentsMargins(8, 28, 8, 28); self.chat_l.setSpacing(12)
        self.chat_l.addStretch()
        ol.addWidget(self.chat_w, 6)
        ol.addStretch(1)
        self.scroll.setWidget(outer)
        vbox.addWidget(self.scroll, stretch=1)

        # ---- Input ----
        inp = QFrame(); inp.setFixedHeight(86)
        inp.setStyleSheet(f"background:{BG_ELEV};border-top:1px solid {BORDER};")
        iouter = QHBoxLayout(inp); iouter.setContentsMargins(0, 16, 0, 16)
        iouter.addStretch(1)
        ibox = QWidget(); ibox.setMaximumWidth(760)
        il = QHBoxLayout(ibox); il.setContentsMargins(8, 0, 8, 0); il.setSpacing(10)
        self.field = QLineEdit()
        self.field.setPlaceholderText("Ask anything — any topic…")
        self.field.setFont(QFont(FONT, 13)); self.field.setEnabled(False)
        self.field.setStyleSheet(f"""
            QLineEdit{{background:{BG_INPUT};color:{TEXT};border:1px solid {BORDER};
                border-radius:24px;padding:11px 22px;}}
            QLineEdit:focus{{border-color:{ACCENT};}}
            QLineEdit:disabled{{color:#444;}}""")
        self.field.returnPressed.connect(self._send)
        self.send = QPushButton("Send"); self.send.setFixedSize(88, 46)
        self.send.setFont(QFont(FONT, 12, QFont.Weight.Medium)); self.send.setEnabled(False)
        self.send.setStyleSheet(f"""
            QPushButton{{background:{ACCENT};color:white;border-radius:23px;border:none;}}
            QPushButton:hover{{background:#5a95f7;}}
            QPushButton:pressed{{background:#2f74e0;}}
            QPushButton:disabled{{background:#202028;color:#4a4a52;}}""")
        self.send.clicked.connect(self._send)
        il.addWidget(self.field); il.addWidget(self.send)
        iouter.addWidget(ibox, 6); iouter.addStretch(1)
        vbox.addWidget(inp)

    def _chip(self, label):
        b = QPushButton(label); b.setFixedHeight(32); b.setFont(QFont(FONT, 10))
        b.setStyleSheet(f"""
            QPushButton{{background:transparent;color:{TEXT_DIM};
                border:1px solid {BORDER};border-radius:16px;padding:0 16px;}}
            QPushButton:hover{{color:{TEXT};border-color:{BORDER_LT};}}""")
        return b

    # ---- model load ----
    def _load_model(self):
        self._add("Initializing the system, please wait a few seconds…", "system")
        t = LoadModelThread(self.engine)
        t.done.connect(self._ready)
        t.error.connect(lambda e: self._add(f"Load error: {e}", "system"))
        t.start(); self._lt = t

    def _ready(self):
        self.status_lbl.setText("● Ready")
        self.status_lbl.setStyleSheet(f"color:{GREEN};")
        self.field.setEnabled(True); self.send.setEnabled(True); self.field.setFocus()
        self._add("All set. Ask me anything.", "system")

    # ---- helpers ----
    def _add(self, text, role):
        b = Bubble(text, role)
        self.chat_l.insertWidget(self.chat_l.count() - 1, b)
        QTimer.singleShot(40, self._bottom)
        return b

    def _bottom(self):
        sb = self.scroll.verticalScrollBar(); sb.setValue(sb.maximum())

    # ---- send / inference ----
    def _send(self):
        q = self.field.text().strip()
        if not q or self.inf_thread:
            return
        self.field.clear(); self.field.setEnabled(False); self.send.setEnabled(False)
        self._add(q, "user")
        # show the prominent animated status card
        self.status_card = StatusCard()
        self.chat_l.insertWidget(self.chat_l.count() - 1, self.status_card)
        QTimer.singleShot(40, self._bottom)
        self.stream_bubble = None
        self.status_lbl.setText("● Working")
        self.status_lbl.setStyleSheet(f"color:{AMBER};")
        self.inf_thread = InferenceThread(self.engine, q)
        self.inf_thread.token.connect(self._on_token)
        self.inf_thread.status.connect(self._on_status)
        self.inf_thread.done.connect(self._on_done)
        self.inf_thread.error.connect(self._on_error)
        self.inf_thread.finished.connect(self._cleanup)
        self.inf_thread.start()

    def _on_status(self, msg, color):
        # real engine stage updates feed the status card
        if self.status_card is not None:
            self.status_card.set_status(msg, color)

    def _on_token(self, piece):
        if self.stream_bubble is None:
            if self.status_card is not None:
                self.status_card.stop()
                self.status_card.setParent(None)
                self.status_card.deleteLater()
                self.status_card = None
            self.stream_bubble = self._add("", "assistant")
            self.status_lbl.setText("● Writing")
            self.status_lbl.setStyleSheet(f"color:{GREEN};")
        self.stream_bubble.stream_piece(piece)
        self._bottom()

    def _on_done(self, full):
        if self.stream_bubble is not None:
            self.stream_bubble.render_formatted(full)
            self._bottom()
        self.stream_bubble = None

    def _on_error(self, err):
        if self.status_card is not None:
            self.status_card.stop()
            self.status_card.setParent(None); self.status_card.deleteLater()
            self.status_card = None
        self._add(f"Error: {err}", "system")

    def _cleanup(self):
        self.inf_thread = None
        self.field.setEnabled(True); self.send.setEnabled(True); self.field.setFocus()
        self.status_lbl.setText("● Ready")
        self.status_lbl.setStyleSheet(f"color:{GREEN};")

    # ---- chat management ----
    def _new_chat(self):
        if self.inf_thread:
            return
        self.engine.new_conversation()
        while self.chat_l.count() > 1:
            it = self.chat_l.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._add("New conversation started.", "system")

    def _open_history(self):
        sessions = self.engine.list_sessions()
        if not sessions:
            self._add("No saved conversations yet.", "system"); return
        dlg = HistoryDialog(sessions, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_id:
            hist = self.engine.load_session(dlg.selected_id)
            while self.chat_l.count() > 1:
                it = self.chat_l.takeAt(0)
                if it.widget(): it.widget().deleteLater()
            for m in hist:
                self._add(m["content"], m["role"])
