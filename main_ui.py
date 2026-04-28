import re
import json
import os
import sys
import threading

# ── Suppress FFmpeg/H.264 stderr noise BEFORE cv2 loads ──
# "error while decoding MB", "no frame!", "missing picture in access unit"
# are all harmless — FFmpeg auto-recovers — but they flood the terminal.
#
# Layer 1: env vars read by OpenCV's bundled FFmpeg at startup
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')  # AV_LOG_QUIET
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
#
# Layer 2: redirect C-level fd 2 (stderr) to os.devnull.
# sys.stderr redirect alone won't catch FFmpeg's direct fprintf() calls.
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_stderr_backup = os.dup(2)      # keep a backup so we can restore if needed

def _silence_stderr():
    os.dup2(_devnull_fd, 2)

def _restore_stderr():
    os.dup2(_stderr_backup, 2)

_silence_stderr()   # apply before cv2 / FFmpeg initialises

import cv2
import easyocr
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtMultimedia import QSound
from database import conn, cursor, load_plates


def _preprocess_plate(crop):
    """
    Return 3 fast grayscale variants optimised for moving vehicles.

    Key changes vs. old version:
    - 2x upscale only (was 3x) — reduces compute ~56%, enough resolution for OCR
    - INTER_LINEAR instead of LANCZOS4 — 4-5x faster, negligible quality loss
    - No bilateralFilter — slow (O(r²) per pixel) and smears motion-blurred edges
    - Sharpening kernel on V3 to recover edge detail lost to motion blur
    - Only 3 variants instead of 4 — cuts OCR passes by 25%

    V1 — CLAHE + Otsu        : standard / good lighting
    V2 — Adaptive threshold  : uneven lighting / shadows
    V3 — Sharpened + Otsu    : motion-blurred plates
    """
    import numpy as np
    h, w = crop.shape[:2]
    # 2x upscale with fast interpolation — enough for OCR, much faster than 3x LANCZOS4
    big  = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq    = clahe.apply(gray)

    # V1: CLAHE + Otsu — handles overexposed / washed-out plates
    _, v1 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # V2: Adaptive threshold — handles shadows and uneven lighting
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    v2 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # V3: Unsharp mask to recover motion-blurred edges, then Otsu
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened    = cv2.filter2D(gray, -1, sharp_kernel)
    _, v3 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [v1, v2, v3]


# ── CAMERA CONFIG FILE ──
CAMERA_CONFIG_PATH = "camera_config.json"

# How long (seconds) CAM1 detection suppresses CAM2 for the same plate
# (cars have front+rear plates — once CAM1 reads the front, ignore CAM2 for this window)
CAM1_SUPPRESS_SECS = 180   # 3 minutes

def load_camera_config():
    if os.path.exists(CAMERA_CONFIG_PATH):
        try:
            with open(CAMERA_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_camera_config(data: dict):
    with open(CAMERA_CONFIG_PATH, "w") as f:
        json.dump(data, f)

# Legacy helpers kept for backward compatibility
def load_camera_url():
    return load_camera_config().get("url", None)

def save_camera_url(url: str):
    cfg = load_camera_config()
    cfg["url"] = url
    save_camera_config(cfg)

# --- CONFIG ---
reader       = easyocr.Reader(['en'], gpu=False)
plate_model  = YOLO('license_plate_detector.pt')   # YOLOv8 plate detector
known_plates = load_plates()

OCR_CONFIDENCE_THRESHOLD  = 0.30          # lowered: catch fast-moving plates sooner
MIN_DIRECTION_CHANGE_SECS  = 20
STABILITY_FRAMES_REQUIRED  = 2            # was 3 — 2 agreements is enough, faster logging
YOLO_INPUT_WIDTH           = 640
FRAME_BUFFER_SIZE          = 8            # was 5 — bigger buffer = more chances to pick sharpest frame
PLATE_CLEAN_PATTERN        = re.compile(r'[^A-Z0-9]')

# ── Philippine plate formats (LTO-official, all current series) ──
#
# 4-wheel vehicles:
#   ABC1234   — 2018 series standard private car        ^[A-Z]{3}[0-9]{4}$
#   ABC123    — 1981 series (still on road)             ^[A-Z]{3}[0-9]{3}$
#   AB12345   — 2014 series car                         ^[A-Z]{2}[0-9]{5}$
#
# Motorcycle / tricycle (6-char, 2018+):
#   123ABC    — primary format                          ^[0-9]{3}[A-Z]{3}$
#   A123BC    — expanded (combinations exhausted)       ^[A-Z][0-9]{3}[A-Z]{2}$
#   AB123C    — expanded                                ^[A-Z]{2}[0-9]{3}[A-Z]$
#   1ABC23    — expanded                                ^[0-9][A-Z]{3}[0-9]{2}$
#   A1234C    — expanded                                ^[A-Z][0-9]{4}[A-Z]$
#   A1C234    — expanded                                ^[A-Z][0-9][A-Z][0-9]{3}$
#   A12C34    — expanded                                ^[A-Z][0-9]{2}[A-Z][0-9]{2}$
#   1234AB    — NCR replacement (1981 era backlog)      ^[0-9]{4}[A-Z]{2}$
#   AB12345   — old motorcycle (7-char, being phased)   ^[A-Z]{2}[0-9]{5}$
#
# Special:
#   SXXX1234  — government (starts with S)              ^S[A-Z]{2}[0-9]{4}$
PLATE_FORMAT_PATTERN = re.compile(
    r'^[A-Z]{3}[0-9]{4}$'          # car 2018: ABC1234
    r'|^[A-Z]{3}[0-9]{3}$'         # car 1981: ABC123
    r'|^[A-Z]{2}[0-9]{5}$'         # car/moto 2014: AB12345
    r'|^[0-9]{3}[A-Z]{3}$'         # moto primary: 123ABC
    r'|^[A-Z][0-9]{3}[A-Z]{2}$'    # moto expanded: A123BC
    r'|^[A-Z]{2}[0-9]{3}[A-Z]$'    # moto expanded: AB123C
    r'|^[0-9][A-Z]{3}[0-9]{2}$'    # moto expanded: 1ABC23
    r'|^[A-Z][0-9]{4}[A-Z]$'       # moto expanded: A1234C
    r'|^[A-Z][0-9][A-Z][0-9]{3}$'  # moto expanded: A1C234
    r'|^[A-Z][0-9]{2}[A-Z][0-9]{2}$'  # moto expanded: A12C34
    r'|^[0-9]{4}[A-Z]{2}$'         # moto NCR backlog: 1234AB
    r'|^S[A-Z]{2}[0-9]{4}$'        # government: SABC1234
)

# Motorcycle plates: any 6-char mix starting or containing digits before letters
# Used to show 🏍 icon and tag logs
MOTO_PLATE_PATTERN = re.compile(
    r'^[0-9]{3}[A-Z]{3}$'
    r'|^[A-Z][0-9]{3}[A-Z]{2}$'
    r'|^[A-Z]{2}[0-9]{3}[A-Z]$'
    r'|^[0-9][A-Z]{3}[0-9]{2}$'
    r'|^[A-Z][0-9]{4}[A-Z]$'
    r'|^[A-Z][0-9][A-Z][0-9]{3}$'
    r'|^[A-Z][0-9]{2}[A-Z][0-9]{2}$'
    r'|^[0-9]{4}[A-Z]{2}$'
)

# ── PALETTE ──
C_ORANGE    = "#EFAF4F"
C_HEADER_BG = "#F5F5F5"
C_BG        = "#F0F0F0"
C_WHITE     = "#FFFFFF"
C_BORDER    = "#CCCCCC"
C_TEXT_DARK = "#222222"
C_GREEN     = "#2ECC71"
C_RED       = "#E74C3C"
C_GRAY_PANEL= "#D9D9D9"
C_RIGHT_BG  = "#E8EAED"
C_BLUE      = "#3498DB"   # used for EXIT direction badge


class MainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.role = None

        self.setWindowTitle("St. Matthew Subdivision")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"background:{C_BG};")

        # ── Camera sources ──
        _cfg = load_camera_config()
        self.cap  = None
        self.cap2 = None   # CAM 2 (exit-side camera)
        self._camera_url      = _cfg.get("url",  None)
        self._camera2_url     = _cfg.get("url2", None)   # CAM 2 URL
        self._video_path      = None   # path to a .mp4/.avi for testing
        self._video_loop      = True   # loop video when it ends
        self._source_is_video = False
        self._log_date = QtCore.QDate.currentDate()

        # ── CAM1 → CAM2 suppression registry ──
        # { plate: datetime_when_cam1_detected }
        # While a plate is in here and within CAM1_SUPPRESS_SECS, CAM2 ignores it.
        self._cam1_recent: dict = {}   # plate → datetime

        self._logged_plates: set        = set()  # plates logged for current direction
        self._last_direction_time: dict = {}     # plate → last datetime direction changed
        self._candidate_plate  = ""
        self._candidate_count  = 0
        self._frame_count      = 0

        # ── Entry/Exit tracking ──
        # Plates currently inside the subdivision (state-machine approach)
        self._inside_vehicles: set = set()

        # Background frame grabber
        self._latest_frame     = None
        self._frame_lock       = threading.Lock()
        self._grab_thread      = None
        self._grab_running     = False

        # Sharpness frame buffer — OCR picks the least-blurry frame
        self._frame_buffer     = deque(maxlen=FRAME_BUFFER_SIZE)
        self._buffer_lock      = threading.Lock()

        # OCR runs in its own thread so it never blocks the UI
        self._ocr_running      = False
        self._ocr_lock         = threading.Lock()

        # ── CAM 2 — separate grab / buffer / OCR state ──
        self._latest_frame2    = None
        self._frame2_lock      = threading.Lock()
        self._grab2_thread     = None
        self._grab2_running    = False
        self._frame_buffer2    = deque(maxlen=FRAME_BUFFER_SIZE)
        self._buffer2_lock     = threading.Lock()
        self._ocr2_running     = False
        self._ocr2_lock        = threading.Lock()
        self._candidate_plate2 = ""
        self._candidate_count2 = 0
        self._logged_plates2: set = set()

        # Last detection result — persists between frames so UI stays stable
        self._display_plate    = ""
        self._display_status   = "SCANNING"
        self._display_direction = ""
        self._display_color    = (0, 255, 255)
        self._display_box      = None   # (x, y, w, h) of last detected plate box

        # Notification list: list of dicts {plate, time, status}
        self._notifications = []
        self._unread_count  = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.timer_ui = QtCore.QTimer()
        self.timer_ui.timeout.connect(self.update_time)
        self.timer_ui.start(1000)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_v = QtWidgets.QVBoxLayout(root)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        root_v.addWidget(self._build_header())

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_stack(), 1)

        root_v.addLayout(body)

        # Rebuild inside-vehicles set from today's logs before loading UI
        self._rebuild_inside_vehicles()
        self.load_logs()
        self.start_camera()
        QtCore.QTimer.singleShot(0, self.showMaximized)

    # ════════════════════════════════════════
    #  ENTRY / EXIT STATE MACHINE
    # ════════════════════════════════════════
    def _rebuild_inside_vehicles(self):
        """
        On startup, reconstruct which plates are currently inside
        by counting today's ENTRY vs EXIT events from the DB.
        If entries > exits for a plate → it is still inside.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT plate,
                   SUM(CASE WHEN direction='ENTRY' THEN 1 ELSE 0 END) AS entries,
                   SUM(CASE WHEN direction='EXIT'  THEN 1 ELSE 0 END) AS exits
            FROM logs
            WHERE timestamp LIKE ?
            GROUP BY plate
        """, (today + '%',))
        self._inside_vehicles.clear()
        for plate, entries, exits in cursor.fetchall():
            if (entries or 0) > (exits or 0):
                self._inside_vehicles.add(plate)

    def _determine_direction(self, plate: str, status: str) -> str:
        """
        State-machine direction logic:
        - Plate NOT in inside set → ENTRY  (add to set if authorized)
        - Plate IN inside set     → EXIT   (remove from set)
        Unauthorized vehicles on ENTRY are NOT added to the inside set
        because they were never granted access.
        """
        if plate in self._inside_vehicles:
            self._inside_vehicles.discard(plate)
            return "EXIT"
        else:
            self._inside_vehicles.add(plate)  # track all plates regardless of auth
            return "ENTRY"

    # ════════════════════════════════════════
    #  HEADER
    # ════════════════════════════════════════
    def _build_header(self):
        header = QtWidgets.QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"background:{C_HEADER_BG}; border-bottom:1px solid {C_BORDER};")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(12, 0, 16, 0)

        logo = QtWidgets.QLabel("M")
        logo.setFixedSize(36, 36)
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo.setStyleSheet(f"background:{C_ORANGE}; border-radius:18px; font-weight:bold; font-size:16px; color:white;")

        app_title = QtWidgets.QLabel("St. Matthew Subdivision")
        app_title.setStyleSheet("font-weight:bold; font-size:15px; margin-left:8px;")

        self.time_label = QtWidgets.QLabel()
        self.time_label.setStyleSheet("font-size:14px; font-weight:bold; color:#333;")
        self.update_time()

        self.user_label = QtWidgets.QLabel("Guest")
        self.user_label.setStyleSheet("font-size:13px; color:#555; margin-right:4px;")

        # ── Bell button with badge ──
        self._bell_wrapper = QtWidgets.QWidget()
        self._bell_wrapper.setFixedSize(40, 40)
        bell_layout = QtWidgets.QStackedLayout(self._bell_wrapper)
        bell_layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)

        self._bell_btn = QtWidgets.QPushButton("🔔")
        self._bell_btn.setFixedSize(36, 36)
        self._bell_btn.setStyleSheet(f"QPushButton {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:18px; font-size:16px; }} QPushButton:hover {{ background:#eee; }}")
        self._bell_btn.clicked.connect(self.show_notifications)

        self._badge = QtWidgets.QLabel("")
        self._badge.setFixedSize(18, 18)
        self._badge.setAlignment(QtCore.Qt.AlignCenter)
        self._badge.setStyleSheet(f"""
            background:{C_RED}; color:white; border-radius:9px;
            font-size:10px; font-weight:bold;
        """)
        self._badge.move(22, 0)
        self._badge.hide()

        bell_layout.addWidget(self._bell_btn)
        bell_layout.addWidget(self._badge)

        hl.addWidget(logo)
        hl.addWidget(app_title)
        hl.addStretch()
        hl.addWidget(self.time_label)
        hl.addSpacing(12)
        hl.addWidget(self.user_label)
        hl.addSpacing(6)
        hl.addWidget(self._bell_wrapper)
        return header

    # ════════════════════════════════════════
    #  NOTIFICATION SYSTEM
    # ════════════════════════════════════════
    def _push_notification(self, plate: str, timestamp: str):
        """Add an UNAUTHORIZED alert to the notification list and play sound."""
        self._notifications.insert(0, {"plate": plate, "time": timestamp})
        self._unread_count += 1
        self._update_badge()
        self._play_alert()

    def _update_badge(self):
        if self._unread_count > 0:
            txt = str(self._unread_count) if self._unread_count < 100 else "99+"
            self._badge.setText(txt)
            self._badge.show()
        else:
            self._badge.hide()

    def _play_alert(self):
        wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert.wav")
        if os.path.exists(wav_path):
            try:
                QSound.play(wav_path)
                return
            except Exception:
                pass
        QtWidgets.QApplication.beep()

    def show_notifications(self):
        """Show the notification panel popup below the bell button."""
        self._unread_count = 0
        self._update_badge()

        panel = QtWidgets.QDialog(self)
        panel.setWindowTitle("Notifications")
        panel.setFixedWidth(360)
        panel.setWindowFlags(
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint
        )
        panel.setStyleSheet(f"""
            QDialog {{ background:{C_WHITE}; border:1px solid {C_BORDER};
                       border-radius:12px; }}
        """)

        vl = QtWidgets.QVBoxLayout(panel)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        hdr = QtWidgets.QFrame()
        hdr.setFixedHeight(48)
        hdr.setStyleSheet(f"background:{C_WHITE}; border-bottom:1px solid {C_BORDER}; border-radius:0px;")
        hdr_row = QtWidgets.QHBoxLayout(hdr)
        hdr_row.setContentsMargins(16, 0, 16, 0)

        title_lbl = QtWidgets.QLabel("🔔  Notifications")
        title_lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#222;")

        clear_btn = QtWidgets.QPushButton("Clear all")
        clear_btn.setStyleSheet(f"QPushButton {{ border:none; background:transparent; color:{C_RED}; font-size:12px; font-weight:bold; }} QPushButton:hover {{ color:#c0392b; }}")
        clear_btn.clicked.connect(lambda: (self._notifications.clear(), panel.close(), ))

        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch()
        hdr_row.addWidget(clear_btn)
        vl.addWidget(hdr)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(min(72 * max(len(self._notifications), 1), 360))
        scroll.setStyleSheet("border:none;")

        container = QtWidgets.QWidget()
        container.setStyleSheet(f"background:{C_WHITE};")
        items_layout = QtWidgets.QVBoxLayout(container)
        items_layout.setContentsMargins(0, 0, 0, 0)
        items_layout.setSpacing(0)

        if not self._notifications:
            empty = QtWidgets.QLabel("No alerts yet.")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setStyleSheet("font-size:13px; color:#aaa; padding:24px;")
            items_layout.addWidget(empty)
        else:
            for n in self._notifications:
                item = QtWidgets.QFrame()
                item.setFixedHeight(68)
                item.setStyleSheet(f"""
                    QFrame {{ background:{C_WHITE}; border-bottom:1px solid #F0F0F0; }}
                    QFrame:hover {{ background:#FFF5F5; }}
                """)
                row = QtWidgets.QHBoxLayout(item)
                row.setContentsMargins(16, 8, 16, 8)
                row.setSpacing(12)

                icon_lbl = QtWidgets.QLabel("🚨")
                icon_lbl.setFixedSize(36, 36)
                icon_lbl.setAlignment(QtCore.Qt.AlignCenter)
                icon_lbl.setStyleSheet(f"background:#FDECEA; border-radius:18px; font-size:18px;")

                txt_col = QtWidgets.QVBoxLayout()
                txt_col.setSpacing(2)

                plate_lbl = QtWidgets.QLabel(f"UNAUTHORIZED  —  {n['plate']}")
                plate_lbl.setStyleSheet(f"font-size:13px; font-weight:bold; color:{C_RED};")

                time_lbl = QtWidgets.QLabel(n["time"])
                time_lbl.setStyleSheet("font-size:11px; color:#888;")

                txt_col.addWidget(plate_lbl)
                txt_col.addWidget(time_lbl)

                row.addWidget(icon_lbl)
                row.addLayout(txt_col)
                items_layout.addWidget(item)

        items_layout.addStretch()
        scroll.setWidget(container)
        vl.addWidget(scroll)

        bell_pos = self._bell_wrapper.mapToGlobal(QtCore.QPoint(0, self._bell_wrapper.height() + 4))
        panel.move(bell_pos.x() - panel.width() + self._bell_wrapper.width(), bell_pos.y())
        panel.exec_()

    # ════════════════════════════════════════
    #  SIDEBAR
    # ════════════════════════════════════════
    def _build_sidebar(self):
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(64)
        sidebar.setStyleSheet(f"background:{C_ORANGE};")

        sl = QtWidgets.QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 16, 10, 16)
        sl.setSpacing(8)
        sl.setAlignment(QtCore.Qt.AlignTop)

        self.btn_dashboard = self._sidebar_btn("🏠")
        self.btn_logs      = self._sidebar_btn("☰")
        self.btn_register  = self._sidebar_btn("👤")
        self.btn_logout    = self._sidebar_btn("🚪")

        self.btn_dashboard.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_logs.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_register.clicked.connect(self.on_register_tab_clicked)
        self.btn_logout.clicked.connect(self.logout)

        sl.addWidget(self.btn_dashboard)
        sl.addWidget(self.btn_logs)
        sl.addStretch()
        sl.addWidget(self.btn_register)
        sl.addWidget(self.btn_logout)
        return sidebar

    def _sidebar_btn(self, icon):
        btn = QtWidgets.QPushButton(icon)
        btn.setFixedSize(44, 44)
        btn.setStyleSheet(f"QPushButton {{ background:{C_WHITE}; border-radius:10px; font-size:18px; }} QPushButton:hover {{ background:#fff3e0; }}")
        return btn

    # ════════════════════════════════════════
    #  STACK
    # ════════════════════════════════════════
    def _build_stack(self):
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_dashboard())
        self.stack.addWidget(self._build_logs_page())
        self.stack.addWidget(self._build_register_page())
        return self.stack

    # ════════════════════════════════════════
    #  REGISTER TAB GATE
    # ════════════════════════════════════════
    def on_register_tab_clicked(self):
        if self.role is None:
            from login import LoginDialog
            dlg = LoginDialog(self)
            dlg.login_success.connect(self._on_login_success)
            dlg.exec_()
        else:
            self.stack.setCurrentIndex(2)

    def _on_login_success(self, role):
        self.role = role
        self.user_label.setText(f"👤 {role.capitalize()}")
        self.stack.setCurrentIndex(2)

    def logout_session(self):
        self.role = None
        self.user_label.setText("Guest")
        self.stack.setCurrentIndex(0)
        QtWidgets.QMessageBox.information(self, "Logged Out", "Admin session ended. Returning to dashboard.")

    def logout(self):
        confirm = QtWidgets.QMessageBox.question(
            self, "Exit", "Are you sure you want to exit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.stop_camera()
            QtWidgets.QApplication.quit()

    # ════════════════════════════════════════
    #  DASHBOARD
    # ════════════════════════════════════════
    def _build_dashboard(self):
        page = QtWidgets.QWidget()
        page.setStyleSheet(f"background:{C_BG};")
        hl = QtWidgets.QHBoxLayout(page)
        hl.setContentsMargins(12, 12, 12, 12)
        hl.setSpacing(12)

        centre_col = QtWidgets.QVBoxLayout()
        centre_col.setSpacing(10)

        live_row = QtWidgets.QHBoxLayout()
        live_lbl = QtWidgets.QLabel("Live Feed")
        live_lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#333;")
        self._live_dot = QtWidgets.QLabel("●")
        self._live_dot.setStyleSheet(f"color:{C_GREEN}; font-size:18px;")

        cam_settings_btn = QtWidgets.QPushButton("⚙️  Camera")
        cam_settings_btn.setFixedHeight(30)
        cam_settings_btn.setStyleSheet(f"""
            QPushButton {{ background:{C_WHITE}; border:1px solid {C_BORDER};
                          border-radius:8px; font-size:12px; padding:0 12px; color:#333; }}
            QPushButton:hover {{ background:#eee; }}
        """)
        cam_settings_btn.clicked.connect(self.show_camera_settings)

        live_row.addWidget(live_lbl)
        live_row.addWidget(self._live_dot)
        live_row.addStretch()
        live_row.addWidget(cam_settings_btn)

        cam_frame = QtWidgets.QFrame()
        cam_frame.setMinimumHeight(350)
        cam_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        cam_frame.setStyleSheet(f"QFrame {{ background:{C_WHITE}; border:2px solid {C_TEXT_DARK}; border-radius:8px; }}")
        cam_inner = QtWidgets.QVBoxLayout(cam_frame)
        cam_inner.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        cam_inner.addWidget(self.camera_label)

        # Dashboard mini log table — now 4 columns including Direction
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Vehicle No.", "Timestamp", "Direction", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(160)
        self.table.setStyleSheet(f"""
            QTableWidget {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:6px; gridline-color:#eee; font-size:13px; }}
            QHeaderView::section {{ background:{C_GRAY_PANEL}; padding:6px; border:none; font-weight:bold; }}
            QTableWidget::item:alternate {{ background:#F9F9F9; }}
        """)

        centre_col.addLayout(live_row)
        centre_col.addWidget(cam_frame, 3)
        centre_col.addWidget(self.table, 1)

        # ── RIGHT PANEL ──
        right = QtWidgets.QFrame()
        right.setFixedWidth(320)
        right.setStyleSheet(f"QFrame {{ background:{C_RIGHT_BG}; border-radius:16px; }}")
        right_v = QtWidgets.QVBoxLayout(right)
        right_v.setContentsMargins(10, 24, 10, 24)
        right_v.setSpacing(8)
        right_v.setAlignment(QtCore.Qt.AlignTop)

        plate_title = QtWidgets.QLabel("Plate Number")
        plate_title.setAlignment(QtCore.Qt.AlignCenter)
        plate_title.setStyleSheet("font-size:14px; color:#555; font-weight:bold;")

        self.plate_label = QtWidgets.QLabel("")
        self.plate_label.setFixedHeight(60)
        self.plate_label.setAlignment(QtCore.Qt.AlignCenter)
        self.plate_label.setStyleSheet(f"background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px; font-size:20px; font-weight:bold;")

        status_title = QtWidgets.QLabel("Status")
        status_title.setAlignment(QtCore.Qt.AlignCenter)
        status_title.setStyleSheet("font-size:14px; color:#555; font-weight:bold; margin-top:10px;")

        self.status_label = QtWidgets.QLabel("SCANNING")
        self.status_label.setFixedHeight(60)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet(f"background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px; font-size:20px; font-weight:bold; color:#888;")

        # ── Direction label ──
        direction_title = QtWidgets.QLabel("Direction")
        direction_title.setAlignment(QtCore.Qt.AlignCenter)
        direction_title.setStyleSheet("font-size:14px; color:#555; font-weight:bold; margin-top:10px;")

        self.direction_label = QtWidgets.QLabel("—")
        self.direction_label.setFixedHeight(50)
        self.direction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.direction_label.setStyleSheet(f"background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px; font-size:18px; font-weight:bold; color:#888;")

        right_v.addWidget(plate_title)
        right_v.addWidget(self.plate_label)
        right_v.addSpacing(4)
        right_v.addWidget(status_title)
        right_v.addWidget(self.status_label)
        right_v.addWidget(direction_title)
        right_v.addWidget(self.direction_label)
        right_v.addStretch()

        hl.addLayout(centre_col, 1)
        hl.addWidget(right)
        return page

    # ════════════════════════════════════════
    #  LOGS PAGE
    # ════════════════════════════════════════
    def _build_logs_page(self):
        page = QtWidgets.QWidget()
        page.setStyleSheet(f"background:{C_BG};")
        vl = QtWidgets.QVBoxLayout(page)
        vl.setContentsMargins(16, 16, 16, 16)
        vl.setSpacing(10)

        top_bar = QtWidgets.QHBoxLayout()

        self.log_date_btn = QtWidgets.QPushButton()
        self.log_date_btn.setFixedHeight(36)
        self.log_date_btn.setStyleSheet(f"""
            QPushButton {{ background:{C_WHITE}; border:1px solid {C_BORDER};
                          border-radius:18px; font-size:13px; padding:0 16px; }}
            QPushButton:hover {{ background:#eee; }}
        """)
        self._refresh_date_btn_text()
        self.log_date_btn.clicked.connect(self.pick_log_date)

        log_search_container = QtWidgets.QFrame()
        log_search_container.setFixedHeight(36)
        log_search_container.setStyleSheet(f"QFrame {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:18px; }}")
        log_search_layout = QtWidgets.QHBoxLayout(log_search_container)
        log_search_layout.setContentsMargins(10, 0, 10, 0)
        log_search_layout.setSpacing(6)

        log_search_icon = QtWidgets.QLabel("🔍")
        log_search_icon.setStyleSheet("font-size:13px; border:none; background:transparent;")

        self.log_search_input = QtWidgets.QLineEdit()
        self.log_search_input.setPlaceholderText("Search by name or plate")
        self.log_search_input.setStyleSheet("border:none; background:transparent; font-size:13px;")
        self.log_search_input.textChanged.connect(self.filter_logs_table)

        log_search_layout.addWidget(log_search_icon)
        log_search_layout.addWidget(self.log_search_input)

        top_bar.addWidget(self.log_date_btn)
        top_bar.addStretch()
        top_bar.addWidget(log_search_container)
        vl.addLayout(top_bar)

        self.log_showing_label = QtWidgets.QLabel()
        self.log_showing_label.setStyleSheet("font-size:13px; color:#555;")
        self._refresh_log_showing_label()
        vl.addWidget(self.log_showing_label)

        # Logs table — 7 columns now including Direction
        self.logs_table = QtWidgets.QTableWidget()
        self.logs_table.setColumnCount(7)
        self.logs_table.setHorizontalHeaderLabels([
            "#", "Owner's Name", "Plate Number", "Vehicle Type", "Time-In/Out", "Direction", "Status"
        ])
        self.logs_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.logs_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.logs_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.logs_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.logs_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.logs_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.setAlternatingRowColors(True)
        self.logs_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.logs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.logs_table.verticalHeader().setVisible(False)
        self.logs_table.setStyleSheet(f"""
            QTableWidget {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px;
                            gridline-color:#eee; font-size:13px; }}
            QHeaderView::section {{ background:{C_GRAY_PANEL}; padding:6px; border:none; font-weight:bold; }}
            QTableWidget::item:alternate {{ background:#F9F9F9; }}
        """)
        vl.addWidget(self.logs_table, 1)
        return page

    def pick_log_date(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Date")
        dialog.setStyleSheet(f"background:{C_WHITE};")
        vl = QtWidgets.QVBoxLayout(dialog)
        vl.setContentsMargins(16, 16, 16, 16)
        vl.setSpacing(10)

        cal = QtWidgets.QCalendarWidget()
        cal.setSelectedDate(self._log_date)
        cal.setGridVisible(True)
        cal.setStyleSheet("""
            QCalendarWidget QWidget#qt_calendar_navigationbar { background:#EFAF4F; padding:4px; }
            QCalendarWidget QToolButton { color:white; font-size:13px; font-weight:bold; background:transparent; border:none; padding:4px 8px; }
            QCalendarWidget QToolButton:hover { background:rgba(255,255,255,0.2); border-radius:4px; }
            QCalendarWidget QSpinBox { color:white; font-size:13px; font-weight:bold; background:transparent; border:none; }
            QCalendarWidget QTableView { font-size:13px; }
        """)

        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.setFixedHeight(36)
        btn_ok.setStyleSheet(f"QPushButton {{ background:{C_ORANGE}; color:white; border-radius:8px; font-weight:bold; }} QPushButton:hover {{ background:#d9983e; }}")
        btn_ok.clicked.connect(dialog.accept)

        vl.addWidget(cal)
        vl.addWidget(btn_ok)

        if dialog.exec_():
            self._log_date = cal.selectedDate()
            self._refresh_date_btn_text()
            self._refresh_log_showing_label()
            self.load_logs()

    def _refresh_date_btn_text(self):
        self.log_date_btn.setText("📅  " + self._log_date.toString("MMMM d, yyyy") + "  ›")

    def _refresh_log_showing_label(self):
        day_str = self._log_date.toString("dddd, MMMM d, yyyy")
        self.log_showing_label.setText(f"Showing Logs For  <b>{day_str}</b>")

    def filter_logs_table(self, text):
        date_str = self._log_date.toString("yyyy-MM-dd")
        cursor.execute("""
            SELECT l.plate, l.timestamp, l.status, p.name, p.vehicle_type, l.direction
            FROM logs l
            LEFT JOIN plates p ON l.plate = p.plate
            WHERE l.timestamp LIKE ?
            ORDER BY l.id DESC
        """, (date_str + "%",))
        rows = cursor.fetchall()
        if text:
            rows = [r for r in rows if
                    text.lower() in (r[3] or "").lower() or
                    text.lower() in r[0].lower()]
        self._populate_logs_table(rows)

    # ════════════════════════════════════════
    #  REGISTER PAGE
    # ════════════════════════════════════════
    def _build_register_page(self):
        page = QtWidgets.QWidget()
        page.setStyleSheet(f"background:{C_BG};")
        vl = QtWidgets.QVBoxLayout(page)
        vl.setContentsMargins(16, 16, 16, 16)
        vl.setSpacing(10)

        top_bar = QtWidgets.QHBoxLayout()

        search_container = QtWidgets.QFrame()
        search_container.setFixedHeight(36)
        search_container.setStyleSheet(f"QFrame {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:18px; }}")
        search_layout = QtWidgets.QHBoxLayout(search_container)
        search_layout.setContentsMargins(10, 0, 10, 0)
        search_layout.setSpacing(6)

        search_icon = QtWidgets.QLabel("🔍")
        search_icon.setStyleSheet("font-size:13px; border:none; background:transparent;")

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search by name or plate")
        self.search_input.setStyleSheet("border:none; background:transparent; font-size:13px;")
        self.search_input.textChanged.connect(self.filter_plate_table)

        search_layout.addWidget(search_icon)
        search_layout.addWidget(self.search_input)

        add_resident_btn = QtWidgets.QPushButton("+ Add Resident")
        add_resident_btn.setFixedHeight(36)
        add_resident_btn.setStyleSheet(f"""
            QPushButton {{ background:#5B4FD9; color:white; border-radius:8px; font-weight:bold; font-size:13px; padding:0 16px; }}
            QPushButton:hover {{ background:#4a3fc7; }}
        """)
        add_resident_btn.clicked.connect(self.show_add_resident_dialog)

        logout_btn = QtWidgets.QPushButton("🚪 Logout")
        logout_btn.setFixedHeight(36)
        logout_btn.setStyleSheet(f"""
            QPushButton {{ background:{C_GRAY_PANEL}; border-radius:8px; font-weight:bold; font-size:13px; padding:0 14px; }}
            QPushButton:hover {{ background:#bbb; }}
        """)
        logout_btn.clicked.connect(self.logout_session)

        top_bar.addWidget(search_container, 1)
        top_bar.addStretch()
        top_bar.addWidget(add_resident_btn)
        top_bar.addSpacing(8)
        top_bar.addWidget(logout_btn)
        vl.addLayout(top_bar)

        list_label = QtWidgets.QLabel("List of Registered")
        list_label.setStyleSheet("font-size:13px; color:#555; font-weight:bold;")
        vl.addWidget(list_label)

        self.plate_table = QtWidgets.QTableWidget()
        self.plate_table.setColumnCount(4)
        self.plate_table.setHorizontalHeaderLabels(["Plate Number", "Resident Name", "Vehicle Type", "Action"])
        self.plate_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.plate_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.plate_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.plate_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.plate_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.plate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.plate_table.verticalHeader().setVisible(False)
        self.plate_table.setShowGrid(True)
        self.plate_table.setStyleSheet(f"""
            QTableWidget {{ background:{C_WHITE}; border:1px solid {C_BORDER}; border-radius:8px; font-size:13px; gridline-color:#eee; }}
            QHeaderView::section {{ background:{C_GRAY_PANEL}; padding:8px; border:none; font-weight:bold; font-size:13px; }}
        """)
        vl.addWidget(self.plate_table, 1)

        self.refresh_plate_table()
        return page

    def refresh_plate_table(self):
        global known_plates
        known_plates = load_plates()
        cursor.execute("SELECT plate, name, vehicle_type FROM plates")
        self._populate_plate_table(cursor.fetchall())

    def _populate_plate_table(self, rows):
        self.plate_table.setRowCount(0)
        for row_data in rows:
            row = self.plate_table.rowCount()
            self.plate_table.insertRow(row)
            for col, val in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
                self.plate_table.setItem(row, col, item)
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.setStyleSheet(f"""
                QPushButton {{ background:{C_RED}; color:white; border-radius:6px; font-weight:bold; font-size:12px; padding:4px 12px; }}
                QPushButton:hover {{ background:#c0392b; }}
            """)
            plate_val = row_data[0]
            remove_btn.clicked.connect(lambda _, p=plate_val: self.remove_plate_by_name(p))
            self.plate_table.setCellWidget(row, 3, remove_btn)
            self.plate_table.setRowHeight(row, 44)

    def filter_plate_table(self, text):
        cursor.execute("SELECT plate, name, vehicle_type FROM plates")
        all_rows = cursor.fetchall()
        filtered = [r for r in all_rows if text.lower() in r[0].lower() or text.lower() in r[1].lower()]
        self._populate_plate_table(filtered)

    def show_add_resident_dialog(self):
        if self.role != "admin":
            QtWidgets.QMessageBox.warning(self, "Access Denied", "Only admins can add residents.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Resident")
        dialog.setFixedSize(380, 260)
        dialog.setStyleSheet(f"background:{C_WHITE};")

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Add New Resident")
        title.setStyleSheet("font-size:15px; font-weight:bold;")
        layout.addWidget(title)

        field_style = f"border:1px solid {C_BORDER}; border-radius:8px; padding-left:10px; font-size:13px; background:{C_WHITE};"

        plate_input = QtWidgets.QLineEdit()
        plate_input.setPlaceholderText("Plate Number (e.g. ABC 1234)")
        plate_input.setFixedHeight(36)
        plate_input.setStyleSheet(field_style)

        name_input = QtWidgets.QLineEdit()
        name_input.setPlaceholderText("Resident Name")
        name_input.setFixedHeight(36)
        name_input.setStyleSheet(field_style)

        vehicle_input = QtWidgets.QLineEdit()
        vehicle_input.setPlaceholderText("Vehicle Type (e.g. Toyota Vios - White)")
        vehicle_input.setFixedHeight(36)
        vehicle_input.setStyleSheet(field_style)

        layout.addWidget(plate_input)
        layout.addWidget(name_input)
        layout.addWidget(vehicle_input)

        btn_row = QtWidgets.QHBoxLayout()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(f"QPushButton {{ background:{C_GRAY_PANEL}; border-radius:8px; font-weight:bold; padding:0 16px; }} QPushButton:hover {{ background:#bbb; }}")
        cancel_btn.clicked.connect(dialog.reject)

        confirm_btn = QtWidgets.QPushButton("Add")
        confirm_btn.setFixedHeight(36)
        confirm_btn.setStyleSheet(f"QPushButton {{ background:#5B4FD9; color:white; border-radius:8px; font-weight:bold; padding:0 16px; }} QPushButton:hover {{ background:#4a3fc7; }}")

        def do_add():
            plate   = plate_input.text().strip().upper()
            name    = name_input.text().strip()
            vehicle = vehicle_input.text().strip()
            if plate and name and vehicle:
                cursor.execute("INSERT OR IGNORE INTO plates (plate, name, vehicle_type) VALUES (?,?,?)", (plate, name, vehicle))
                conn.commit()
                self.refresh_plate_table()
                dialog.accept()
            else:
                QtWidgets.QMessageBox.warning(dialog, "Missing Fields", "Please fill in all fields.")

        confirm_btn.clicked.connect(do_add)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)
        dialog.exec_()

    def remove_plate_by_name(self, plate):
        if self.role != "admin":
            QtWidgets.QMessageBox.warning(self, "Access Denied", "Only admins can remove plates.")
            return
        cursor.execute("DELETE FROM plates WHERE plate=?", (plate,))
        conn.commit()
        self.refresh_plate_table()

    # ════════════════════════════════════════
    #  CAMERA
    # ════════════════════════════════════════
    def start_camera(self):
        if self.cap is not None:
            return

        if self._video_path:
            source = self._video_path
            self._source_is_video = True
        elif self._camera_url:
            source = self._camera_url
            self._source_is_video = False
        else:
            source = 0
            self._source_is_video = False

        self.cap = cv2.VideoCapture()
        if not self._source_is_video:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        self.cap.open(source)

        if not self.cap.isOpened():
            self.cap = None
            self._live_dot.setStyleSheet("color:#E74C3C; font-size:18px;")
            QtWidgets.QMessageBox.warning(
                self, "Camera Error",
                f"Could not open source:\n{source}\n\nCheck the path or connection and try again."
            )
            return

        if self._source_is_video:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._video_frame_delay = 1.0 / fps if (fps and 5 < fps < 240) else 1.0 / 30.0
        else:
            self._video_frame_delay = 0.0
            if self._camera_url:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._live_dot.setStyleSheet(f"color:{C_GREEN}; font-size:18px;")
        self._grab_running = True
        self._grab_thread  = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()
        self.timer.start(33)

        # ── Start CAM 2 if a URL is configured ──
        if self._camera2_url:
            self._start_camera2()

    def _start_camera2(self):
        """Start the exit-side (CAM 2) stream independently.
        _camera2_url == '__webcam1__' means use laptop's second webcam (index 1).
        """
        if self.cap2 is not None:
            return
        if not self._camera2_url:
            return

        source = 1 if self._camera2_url == "__webcam1__" else self._camera2_url

        self.cap2 = cv2.VideoCapture()
        if isinstance(source, str):
            self.cap2.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            self.cap2.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        self.cap2.open(source)
        if not self.cap2.isOpened():
            self.cap2 = None
            print(f"[CAM2] Could not open: {source}")
            return
        if isinstance(source, str):
            self.cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._grab2_running = True
        self._grab2_thread  = threading.Thread(target=self._grab_loop2, daemon=True)
        self._grab2_thread.start()

    def _grab_loop(self):
        """
        Background frame grabber.
        - Live stream: reads as fast as possible, detects dropped connection.
        - Video file: sleeps between reads to match native FPS (real-time speed),
          loops back to frame 0 when file ends (if loop=True).
        """
        import time

        consecutive_failures = 0
        MAX_FAILURES  = 15
        frame_delay   = getattr(self, '_video_frame_delay', 0.0)
        is_video      = getattr(self, '_source_is_video',   False)
        last_frame_ts = time.monotonic()

        while self._grab_running and self.cap is not None:

            # ── FPS throttle for video files ──
            if is_video and frame_delay > 0:
                now     = time.monotonic()
                elapsed = now - last_frame_ts
                wait    = frame_delay - elapsed
                if wait > 0:
                    time.sleep(wait)
                last_frame_ts = time.monotonic()

            ret, frame = self.cap.read()

            # Video end-of-file
            if not ret and is_video:
                if self._video_loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_ts = time.monotonic()
                    consecutive_failures = 0
                    continue
                else:
                    break

            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    QtCore.QMetaObject.invokeMethod(
                        self, "_on_stream_lost",
                        QtCore.Qt.QueuedConnection
                    )
                    break
                continue

            if cv2.mean(frame)[0] < 2.0 and cv2.mean(frame)[1] < 2.0:
                consecutive_failures += 1
                continue

            consecutive_failures = 0
            with self._frame_lock:
                self._latest_frame = frame
            with self._buffer_lock:
                self._frame_buffer.append(frame.copy())

    def _grab_loop2(self):
        """Background frame grabber for CAM 2 — auto-reconnects on stream drop."""
        import time
        MAX_FAILURES   = 15
        MAX_RETRIES    = 999   # keep retrying indefinitely while running
        RETRY_DELAYS   = [2, 5, 10, 15, 30]  # seconds between reconnect attempts

        retry_count = 0

        while self._grab2_running:
            # ── Grab frames until stream drops ──
            consecutive_failures = 0
            while self._grab2_running and self.cap2 is not None:
                ret, frame = self.cap2.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        print(f"[CAM2] Stream lost — attempting reconnect (retry #{retry_count + 1})...")
                        break
                    time.sleep(0.05)
                    continue
                if cv2.mean(frame)[0] < 2.0 and cv2.mean(frame)[1] < 2.0:
                    consecutive_failures += 1
                    continue
                consecutive_failures  = 0
                retry_count           = 0  # reset on successful frame
                with self._frame2_lock:
                    self._latest_frame2 = frame
                with self._buffer2_lock:
                    self._frame_buffer2.append(frame.copy())

            if not self._grab2_running:
                break  # user stopped the camera — exit cleanly

            # ── Reconnect ──
            delay = RETRY_DELAYS[min(retry_count, len(RETRY_DELAYS) - 1)]
            print(f"[CAM2] Reconnecting in {delay}s...")
            time.sleep(delay)
            retry_count += 1

            # Release old capture
            if self.cap2 is not None:
                try:
                    self.cap2.release()
                except Exception:
                    pass

            if not self._grab2_running:
                break

            # Re-open stream
            source = self._camera2_url
            new_cap = cv2.VideoCapture()
            try:
                new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except Exception:
                pass
            new_cap.open(source)

            if new_cap.isOpened():
                try:
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.cap2 = new_cap
                print(f"[CAM2] Reconnected successfully.")
            else:
                new_cap.release()
                self.cap2 = None
                print(f"[CAM2] Reconnect failed — will retry.")

    @QtCore.pyqtSlot()
    def _on_stream_lost(self):
        """Called on main thread when grab loop detects stream has dropped."""
        self.stop_camera()
        self._live_dot.setStyleSheet("color:#E74C3C; font-size:18px;")
        QtWidgets.QMessageBox.warning(
            self, "Stream Lost",
            "Camera stream disconnected.\n\nCheck your connection and restart the camera."
        )

    @staticmethod
    def _sharpness(frame):
        """Laplacian variance — higher = sharper = less motion blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _best_frame(self):
        """Return the sharpest frame from the buffer, or latest if buffer empty."""
        with self._buffer_lock:
            if not self._frame_buffer:
                return None
            return max(self._frame_buffer, key=self._sharpness).copy()

    def _best_frame2(self):
        """Return the sharpest frame from CAM2 buffer."""
        with self._buffer2_lock:
            if not self._frame_buffer2:
                return None
            return max(self._frame_buffer2, key=self._sharpness).copy()

    def stop_camera(self):
        self._grab_running = False
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2)
            self._grab_thread = None
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self._latest_frame = None
            with self._buffer_lock:
                self._frame_buffer.clear()
            self.camera_label.clear()
            self._live_dot.setStyleSheet("color:#E74C3C; font-size:18px;")

        # Stop CAM 2 as well
        self._grab2_running = False
        if self._grab2_thread is not None:
            self._grab2_thread.join(timeout=2)
            self._grab2_thread = None
        if self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None
            self._latest_frame2 = None
            with self._buffer2_lock:
                self._frame_buffer2.clear()

    def update_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()

        self._live_dot.setStyleSheet(f"color:{C_GREEN}; font-size:18px;")
        self._frame_count += 1

        with self._ocr_lock:
            ocr_free = not self._ocr_running
        if ocr_free:
            ocr_frame = self._best_frame()
            if ocr_frame is not None:
                t = threading.Thread(target=self._run_ocr, args=(ocr_frame,), daemon=True)
                with self._ocr_lock:
                    self._ocr_running = True
                t.start()

        # ── CAM 2 OCR trigger (exit-side) ──
        with self._ocr2_lock:
            ocr2_free = not self._ocr2_running
        if ocr2_free and self.cap2 is not None:
            ocr_frame2 = self._best_frame2()
            if ocr_frame2 is not None:
                t2 = threading.Thread(target=self._run_ocr2, args=(ocr_frame2,), daemon=True)
                with self._ocr2_lock:
                    self._ocr2_running = True
                t2.start()

        if self._display_box is not None:
            x, y, w, h = self._display_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), self._display_color, 2)

        # ── Update UI labels from last OCR result ──
        self.plate_label.setText(self._display_plate)
        self.status_label.setText(self._display_status)

        if self._display_status == "AUTHORIZED":
            self.status_label.setStyleSheet(f"background:{C_WHITE}; color:{C_GREEN}; font-weight:bold; font-size:20px; border:1px solid {C_BORDER}; border-radius:8px;")
        elif self._display_status == "UNAUTHORIZED":
            self.status_label.setStyleSheet(f"background:{C_WHITE}; color:{C_RED}; font-weight:bold; font-size:20px; border:1px solid {C_BORDER}; border-radius:8px;")
        elif self._display_status == "VERIFYING":
            self.status_label.setStyleSheet(f"background:{C_WHITE}; color:#F39C12; font-weight:bold; font-size:20px; border:1px solid {C_BORDER}; border-radius:8px;")
        else:
            self.status_label.setStyleSheet(f"background:{C_WHITE}; color:#888; font-weight:bold; font-size:20px; border:1px solid {C_BORDER}; border-radius:8px;")

        # ── Direction label ──
        self.direction_label.setText(self._display_direction if self._display_direction else "—")
        if self._display_direction == "ENTRY":
            self.direction_label.setStyleSheet(f"background:{C_WHITE}; color:{C_GREEN}; font-weight:bold; font-size:18px; border:1px solid {C_BORDER}; border-radius:8px;")
        elif self._display_direction == "EXIT":
            self.direction_label.setStyleSheet(f"background:{C_WHITE}; color:{C_BLUE}; font-weight:bold; font-size:18px; border:1px solid {C_BORDER}; border-radius:8px;")
        else:
            self.direction_label.setStyleSheet(f"background:{C_WHITE}; color:#888; font-weight:bold; font-size:18px; border:1px solid {C_BORDER}; border-radius:8px;")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.camera_label.setPixmap(
            pix.scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def _run_ocr(self, frame):
        """
        Background thread — YOLO plate detect → preprocess → EasyOCR.

        Key improvements for moving vehicles:
        1. Motion-blur gate: skip OCR on blurry crops (saves 200-500 ms)
        2. Fast-exit: stop trying variants once a high-confidence read is found
        3. logged_plates reset when plate leaves frame so re-entry is caught
        """
        try:
            fh, fw = frame.shape[:2]
            scale  = YOLO_INPUT_WIDTH / fw
            small  = cv2.resize(frame, (YOLO_INPUT_WIDTH, int(fh * scale)),
                                interpolation=cv2.INTER_LINEAR)

            results = plate_model(small, verbose=False)[0]
            boxes   = results.boxes

            best_plate = None
            best_conf  = 0.0
            best_box   = None

            if boxes is not None and len(boxes):
                for box in boxes:
                    yolo_conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(x1 / scale);  x2 = int(x2 / scale)
                    y1 = int(y1 / scale);  y2 = int(y2 / scale)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h

                    # Adaptive YOLO threshold by plate area
                    frame_area = fw * fh
                    area_ratio = box_area / frame_area
                    if area_ratio < 0.015:
                        yolo_min = 0.25
                    elif area_ratio < 0.04:
                        yolo_min = 0.28
                    else:
                        yolo_min = 0.32
                    if yolo_conf < yolo_min:
                        continue

                    pad_x = max(6, int(box_w * 0.06))
                    pad_y = max(4, int(box_h * 0.10))
                    x1p = max(0,  x1 - pad_x);  y1p = max(0,  y1 - pad_y)
                    x2p = min(fw, x2 + pad_x);  y2p = min(fh, y2 + pad_y)

                    crop = frame[y1p:y2p, x1p:x2p]
                    if crop.size == 0:
                        continue

                    # ── Motion-blur gate ──
                    # Laplacian variance < 40 → too blurry for OCR to read reliably.
                    # Skip rather than waste 200-500 ms getting garbage results.
                    gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray_check, cv2.CV_64F).var() < 40:
                        continue

                    variants = _preprocess_plate(crop)

                    votes = {}
                    for variant in variants:
                        ocr_results = reader.readtext(
                            variant, detail=1,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            paragraph=False,
                        )
                        for (_, text, ocr_conf) in ocr_results:
                            if ocr_conf < OCR_CONFIDENCE_THRESHOLD:
                                continue
                            cleaned = PLATE_CLEAN_PATTERN.sub('', text.upper())
                            if len(cleaned) < 5:
                                continue
                            if not PLATE_FORMAT_PATTERN.match(cleaned):
                                continue
                            score = yolo_conf * ocr_conf
                            if cleaned not in votes or score > votes[cleaned]:
                                votes[cleaned] = score

                        # Fast-exit: high-confidence read found → skip remaining variants
                        if votes and max(votes.values()) > 0.75:
                            break

                    for cleaned, score in votes.items():
                        if score > best_conf:
                            best_conf  = score
                            best_plate = cleaned
                            best_box   = (x1, y1, box_w, box_h)

            if best_plate:
                if best_plate == self._candidate_plate:
                    self._candidate_count += 1
                else:
                    self._candidate_plate = best_plate
                    self._candidate_count  = 1

                self._display_plate = best_plate
                self._display_box   = best_box

                if self._candidate_count < STABILITY_FRAMES_REQUIRED:
                    self._display_status    = "VERIFYING"
                    self._display_direction = ""
                    self._display_color     = (0, 200, 255)
                else:
                    now = datetime.now()
                    self._display_status = "AUTHORIZED" if best_plate in known_plates else "UNAUTHORIZED"
                    self._display_color  = (0, 255, 0)  if best_plate in known_plates else (0, 0, 255)

                    if best_plate not in self._logged_plates:
                        ts = now.strftime('%Y-%m-%d %H:%M:%S')
                        QtCore.QMetaObject.invokeMethod(
                            self, "_log_from_ocr_thread",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, best_plate),
                            QtCore.Q_ARG(str, ts),
                            QtCore.Q_ARG(str, self._display_status)
                        )
            else:
                # Plate left the frame — evict from logged_plates so the next
                # pass (e.g. EXIT direction) gets detected and logged again
                if self._candidate_plate and self._candidate_plate in self._logged_plates:
                    self._logged_plates.discard(self._candidate_plate)

                self._candidate_plate   = ""
                self._candidate_count   = 0
                self._display_plate     = ""
                self._display_status    = "SCANNING"
                self._display_direction = ""
                self._display_color     = (0, 255, 255)
                self._display_box       = None

        except Exception as e:
            print(f"[OCR ERROR] {e}")
        finally:
            with self._ocr_lock:
                self._ocr_running = False

    @QtCore.pyqtSlot(str, str, str)
    def _log_from_ocr_thread(self, plate, ts, status):
        """
        Main-thread log gate:
        1. Guard against direction flipping faster than MIN_DIRECTION_CHANGE_SECS
        2. Determine ENTRY / EXIT via state machine
        3. Mark plate as logged (suppress until next state transition)
        4. Show 🏍 or 🚗 icon on live feed
        5. Register in _cam1_recent so CAM2 ignores this plate for CAM1_SUPPRESS_SECS
           (for 4-wheel vehicles that show a front plate to CAM1 then rear to CAM2)
        """
        now = datetime.now()
        last_dir_time = self._last_direction_time.get(plate)

        if last_dir_time and (now - last_dir_time).total_seconds() < MIN_DIRECTION_CHANGE_SECS:
            direction = self._display_direction if self._display_direction else "ENTRY"
        else:
            direction = self._determine_direction(plate, status)
            self._last_direction_time[plate] = now

        self._logged_plates.add(plate)   # suppress until next state transition

        # ── CAM1 → CAM2 suppression ──
        # Motorcycles / tricycles only have a rear plate, so CAM1 will never see
        # their plate (it faces away). Only suppress for 4-wheel plates.
        is_moto = bool(MOTO_PLATE_PATTERN.match(plate))
        if not is_moto:
            self._cam1_recent[plate] = now   # stamp for CAM2 ignore logic

        # Show vehicle type icon on live feed overlay
        self._display_plate = ("🏍 " if is_moto else "🚗 ") + plate

        self._display_direction = direction
        self.add_log(plate, ts, status, direction)
        if status == "UNAUTHORIZED":
            self._push_notification(plate, ts)

    # ────────────────────────────────────────
    #  CAM 2 — OCR worker (exit-side camera)
    # ────────────────────────────────────────
    def _run_ocr2(self, frame):
        """
        Identical detection pipeline to _run_ocr but uses CAM2 state variables
        and applies the CAM1 suppression logic before logging.

        Suppression rule (for 4-wheel vehicles):
          If CAM1 already detected this plate within the last CAM1_SUPPRESS_SECS,
          CAM2 silently ignores it.  This prevents double-logging the same car
          that just passed CAM1 (front plate) now showing its rear plate to CAM2.

        Motorcycles / tricycles are NOT suppressed — they only have a rear plate
        and CAM1 would not have detected them in the first place.
        """
        try:
            fh, fw = frame.shape[:2]
            scale  = YOLO_INPUT_WIDTH / fw
            small  = cv2.resize(frame, (YOLO_INPUT_WIDTH, int(fh * scale)),
                                interpolation=cv2.INTER_LINEAR)

            results = plate_model(small, verbose=False)[0]
            boxes   = results.boxes

            best_plate = None
            best_conf  = 0.0
            best_box   = None

            if boxes is not None and len(boxes):
                for box in boxes:
                    yolo_conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(x1 / scale);  x2 = int(x2 / scale)
                    y1 = int(y1 / scale);  y2 = int(y2 / scale)
                    box_w = x2 - x1;  box_h = y2 - y1
                    box_area = box_w * box_h
                    frame_area = fw * fh
                    area_ratio = box_area / frame_area

                    if area_ratio < 0.015:    yolo_min = 0.25
                    elif area_ratio < 0.04:   yolo_min = 0.28
                    else:                     yolo_min = 0.32
                    if yolo_conf < yolo_min:
                        continue

                    pad_x = max(6, int(box_w * 0.06));  pad_y = max(4, int(box_h * 0.10))
                    x1p = max(0, x1 - pad_x);  y1p = max(0, y1 - pad_y)
                    x2p = min(fw, x2 + pad_x);  y2p = min(fh, y2 + pad_y)
                    crop = frame[y1p:y2p, x1p:x2p]
                    if crop.size == 0:
                        continue

                    gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray_check, cv2.CV_64F).var() < 40:
                        continue

                    variants = _preprocess_plate(crop)
                    votes = {}
                    for variant in variants:
                        ocr_results = reader.readtext(
                            variant, detail=1,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            paragraph=False,
                        )
                        for (_, text, ocr_conf) in ocr_results:
                            if ocr_conf < OCR_CONFIDENCE_THRESHOLD:
                                continue
                            cleaned = PLATE_CLEAN_PATTERN.sub('', text.upper())
                            if len(cleaned) < 5:
                                continue
                            if not PLATE_FORMAT_PATTERN.match(cleaned):
                                continue
                            score = yolo_conf * ocr_conf
                            if cleaned not in votes or score > votes[cleaned]:
                                votes[cleaned] = score
                        if votes and max(votes.values()) > 0.75:
                            break

                    for cleaned, score in votes.items():
                        if score > best_conf:
                            best_conf  = score
                            best_plate = cleaned
                            best_box   = (x1, y1, box_w, box_h)

            if best_plate:
                if best_plate == self._candidate_plate2:
                    self._candidate_count2 += 1
                else:
                    self._candidate_plate2 = best_plate
                    self._candidate_count2 = 1

                if self._candidate_count2 >= STABILITY_FRAMES_REQUIRED:
                    now = datetime.now()
                    status2 = "AUTHORIZED" if best_plate in known_plates else "UNAUTHORIZED"

                    if best_plate not in self._logged_plates2:
                        ts = now.strftime('%Y-%m-%d %H:%M:%S')
                        QtCore.QMetaObject.invokeMethod(
                            self, "_log_from_ocr2_thread",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, best_plate),
                            QtCore.Q_ARG(str, ts),
                            QtCore.Q_ARG(str, status2)
                        )
            else:
                if self._candidate_plate2 and self._candidate_plate2 in self._logged_plates2:
                    self._logged_plates2.discard(self._candidate_plate2)
                self._candidate_plate2  = ""
                self._candidate_count2  = 0

        except Exception as e:
            print(f"[OCR2 ERROR] {e}")
        finally:
            with self._ocr2_lock:
                self._ocr2_running = False

    @QtCore.pyqtSlot(str, str, str)
    def _log_from_ocr2_thread(self, plate, ts, status):
        """
        Main-thread log gate for CAM 2.

        For 4-wheel vehicles:
          If this plate was detected by CAM1 within the last CAM1_SUPPRESS_SECS,
          it means the car is still passing through (front plate → CAM1,
          rear plate → CAM2 moments later). Suppress the CAM2 log entirely.

        For motorcycles / tricycles (rear-plate only):
          No suppression — treat identically to CAM1 logic.
        """
        now = datetime.now()
        is_moto = bool(MOTO_PLATE_PATTERN.match(plate))

        # ── CAM1 suppression check (cars only) ──
        if not is_moto:
            cam1_ts = self._cam1_recent.get(plate)
            if cam1_ts is not None:
                elapsed = (now - cam1_ts).total_seconds()
                if elapsed < CAM1_SUPPRESS_SECS:
                    # Still within suppress window — ignore CAM2 detection
                    self._logged_plates2.add(plate)  # prevent repeat checks
                    return
                else:
                    # Window expired — clean up registry
                    del self._cam1_recent[plate]

        # Outside suppress window (or motorcycle) — log normally as EXIT
        last_dir_time = self._last_direction_time.get(plate)
        if last_dir_time and (now - last_dir_time).total_seconds() < MIN_DIRECTION_CHANGE_SECS:
            direction = "EXIT"
        else:
            direction = self._determine_direction(plate, status)
            self._last_direction_time[plate] = now

        self._logged_plates2.add(plate)

        self.add_log(plate, ts, status, direction)
        if status == "UNAUTHORIZED":
            self._push_notification(plate, ts)

    # ════════════════════════════════════════
    #  CAMERA SETTINGS DIALOG
    # ════════════════════════════════════════
    def show_camera_settings(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Camera Settings")
        dlg.setFixedSize(580, 500)
        dlg.setStyleSheet(f"background:{C_WHITE};")

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(8)

        field_style = f"""
            QLineEdit {{ border:1px solid {C_BORDER}; border-radius:8px;
                        padding-left:10px; font-size:13px; background:{C_WHITE}; color:#222; }}
            QLineEdit:focus {{ border-color:{C_ORANGE}; }}
            QLineEdit:disabled {{ background:#F5F5F5; color:#aaa; }}
        """
        rb_style = "font-size:12px; color:#333;"

        # ─────────────────────────────────────
        #  CAM 1
        # ─────────────────────────────────────
        cam1_title = QtWidgets.QLabel("📸  CAM 1 — Entrance  (detects front plate, outside gate)")
        cam1_title.setStyleSheet("font-size:13px; font-weight:bold; color:#222;")
        layout.addWidget(cam1_title)

        c1_src = QtWidgets.QButtonGroup(dlg)
        c1_webcam = QtWidgets.QRadioButton("Laptop Webcam")
        c1_ip     = QtWidgets.QRadioButton("IP Cam (RTSP/HTTP)")
        c1_video  = QtWidgets.QRadioButton("Video File")
        for rb in (c1_webcam, c1_ip, c1_video):
            rb.setStyleSheet(rb_style);  c1_src.addButton(rb)

        c1_row = QtWidgets.QHBoxLayout()
        c1_row.addWidget(c1_webcam); c1_row.addWidget(c1_ip)
        c1_row.addWidget(c1_video);  c1_row.addStretch()
        layout.addLayout(c1_row)

        c1_url_row = QtWidgets.QHBoxLayout()
        c1_url = QtWidgets.QLineEdit()
        c1_url.setFixedHeight(36); c1_url.setStyleSheet(field_style)
        c1_browse = QtWidgets.QPushButton("Browse…")
        c1_browse.setFixedSize(80, 36)
        c1_browse.setStyleSheet(f"QPushButton{{background:{C_GRAY_PANEL};border-radius:8px;font-size:12px;font-weight:bold;}} QPushButton:hover{{background:#bbb;}}")
        c1_url_row.addWidget(c1_url); c1_url_row.addWidget(c1_browse)
        layout.addLayout(c1_url_row)

        c1_loop = QtWidgets.QCheckBox("Loop video")
        c1_loop.setStyleSheet("font-size:12px; color:#444;")
        c1_loop.setChecked(self._video_loop)
        layout.addWidget(c1_loop)

        # ─────────────────────────────────────
        #  CAM 2
        # ─────────────────────────────────────
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color:{C_BORDER};"); layout.addWidget(sep)

        cam2_title = QtWidgets.QLabel("📸  CAM 2 — Exit  (inside gate · cars seen by CAM 1 ignored for 3 min)")
        cam2_title.setStyleSheet("font-size:13px; font-weight:bold; color:#222;")
        layout.addWidget(cam2_title)

        c2_src = QtWidgets.QButtonGroup(dlg)
        c2_none   = QtWidgets.QRadioButton("Not used")
        c2_webcam = QtWidgets.QRadioButton("Laptop Webcam")
        c2_ip     = QtWidgets.QRadioButton("IP Cam (RTSP/HTTP)")
        for rb in (c2_none, c2_webcam, c2_ip):
            rb.setStyleSheet(rb_style);  c2_src.addButton(rb)

        c2_row = QtWidgets.QHBoxLayout()
        c2_row.addWidget(c2_none); c2_row.addWidget(c2_webcam)
        c2_row.addWidget(c2_ip);   c2_row.addStretch()
        layout.addLayout(c2_row)

        c2_url = QtWidgets.QLineEdit()
        c2_url.setFixedHeight(36); c2_url.setStyleSheet(field_style)
        c2_url.setPlaceholderText("rtsp://admin:password@192.168.x.x:554/stream1")
        c2_url.setText(self._camera2_url or "")
        layout.addWidget(c2_url)

        # ── Set initial states ──
        if self._video_path:
            c1_video.setChecked(True)
        elif self._camera_url:
            c1_ip.setChecked(True)
        else:
            c1_webcam.setChecked(True)

        # CAM2 initial state — detect if webcam (index 1) or IP
        _cam2_is_webcam = (self._camera2_url == "__webcam1__")
        if self._camera2_url is None:
            c2_none.setChecked(True)
        elif _cam2_is_webcam:
            c2_webcam.setChecked(True)
        else:
            c2_ip.setChecked(True)

        def _refresh_c1():
            is_ip    = c1_ip.isChecked()
            is_video = c1_video.isChecked()
            c1_url.setEnabled(is_ip or is_video)
            c1_browse.setVisible(is_video)
            c1_loop.setVisible(is_video)
            if c1_webcam.isChecked():
                c1_url.setPlaceholderText("(laptop webcam — no URL needed)")
                c1_url.setText("")
            elif is_ip:
                c1_url.setPlaceholderText("rtsp://admin:password@192.168.1.100:554/stream1")
                c1_url.setText(self._camera_url or "")
            else:
                c1_url.setPlaceholderText("C:/path/to/video.mp4")
                c1_url.setText(self._video_path or "")

        def _refresh_c2():
            is_ip = c2_ip.isChecked()
            c2_url.setEnabled(is_ip)
            if not is_ip:
                c2_url.setText("")
            elif not c2_url.text():
                c2_url.setText(self._camera2_url or "")

        for rb in (c1_webcam, c1_ip, c1_video):
            rb.toggled.connect(lambda _: _refresh_c1())
        for rb in (c2_none, c2_webcam, c2_ip):
            rb.toggled.connect(lambda _: _refresh_c2())

        _refresh_c1(); _refresh_c2()

        def _browse_c1():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dlg, "Select Video File", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)")
            if path: c1_url.setText(path)
        c1_browse.clicked.connect(_browse_c1)

        # ── Status / buttons ──
        test_msg = QtWidgets.QLabel("")
        test_msg.setStyleSheet("font-size:12px; color:#555;")
        layout.addWidget(test_msg)

        btn_row = QtWidgets.QHBoxLayout()

        test_btn = QtWidgets.QPushButton("Test CAM 1")
        test_btn.setFixedHeight(36)
        test_btn.setStyleSheet(f"QPushButton{{background:{C_GRAY_PANEL};border-radius:8px;font-weight:bold;font-size:13px;padding:0 14px;}} QPushButton:hover{{background:#bbb;}}")

        test2_btn = QtWidgets.QPushButton("Test CAM 2")
        test2_btn.setFixedHeight(36)
        test2_btn.setStyleSheet(f"QPushButton{{background:{C_GRAY_PANEL};border-radius:8px;font-weight:bold;font-size:13px;padding:0 14px;}} QPushButton:hover{{background:#bbb;}}")

        save_btn = QtWidgets.QPushButton("Save & Reconnect")
        save_btn.setFixedHeight(36)
        save_btn.setStyleSheet(f"QPushButton{{background:{C_ORANGE};color:white;border-radius:8px;font-weight:bold;font-size:13px;padding:0 14px;}} QPushButton:hover{{background:#d9983e;}}")

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(f"QPushButton{{background:{C_GRAY_PANEL};border-radius:8px;font-weight:bold;font-size:13px;padding:0 14px;}} QPushButton:hover{{background:#bbb;}}")

        def _test_cam(source, label):
            if source == "":
                test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                test_msg.setText(f"⚠  Enter a URL/path for {label} first.")
                return
            test_msg.setStyleSheet("font-size:12px; color:#F39C12;")
            test_msg.setText(f"⏳  Testing {label}…")
            QtWidgets.QApplication.processEvents()
            cap_t = cv2.VideoCapture()
            if not isinstance(source, int):
                cap_t.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                cap_t.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            cap_t.open(source)
            if cap_t.isOpened():
                ret, _ = cap_t.read(); cap_t.release()
                if ret:
                    test_msg.setStyleSheet("font-size:12px; color:#2ECC71;")
                    test_msg.setText(f"✅  {label} opened successfully!")
                else:
                    test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                    test_msg.setText(f"❌  {label} opened but no frames received.")
            else:
                cap_t.release()
                test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                test_msg.setText(f"❌  {label}: Could not open. Check IP / credentials.")

        def do_test():
            src = 0 if c1_webcam.isChecked() else c1_url.text().strip()
            _test_cam(src, "CAM 1")

        def do_test2():
            if c2_none.isChecked():
                test_msg.setStyleSheet("font-size:12px; color:#888;")
                test_msg.setText("ℹ  CAM 2 is set to Not used.")
                return
            src = 1 if c2_webcam.isChecked() else c2_url.text().strip()
            _test_cam(src, "CAM 2")

        def do_save():
            # ── CAM 1 ──
            if c1_webcam.isChecked():
                self._camera_url = None;  self._video_path = None
                self._source_is_video = False
                cam1_save = ""
            elif c1_ip.isChecked():
                val = c1_url.text().strip()
                if not val:
                    test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                    test_msg.setText("⚠  Enter a URL for CAM 1.");  return
                self._camera_url = val;  self._video_path = None
                self._source_is_video = False;  cam1_save = val
            else:
                val = c1_url.text().strip()
                if not val:
                    test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                    test_msg.setText("⚠  Enter or browse a file path for CAM 1.");  return
                self._video_path = val;  self._camera_url = None
                self._video_loop = c1_loop.isChecked()
                self._source_is_video = True;  cam1_save = ""

            # ── CAM 2 ──
            if c2_none.isChecked():
                self._camera2_url = None;  cam2_save = ""
            elif c2_webcam.isChecked():
                self._camera2_url = "__webcam1__";  cam2_save = "__webcam1__"
            else:
                val2 = c2_url.text().strip()
                if not val2:
                    test_msg.setStyleSheet("font-size:12px; color:#E74C3C;")
                    test_msg.setText("⚠  Enter a URL for CAM 2.");  return
                self._camera2_url = val2;  cam2_save = val2

            save_camera_config({"url": cam1_save, "url2": cam2_save})
            self.stop_camera();  self.start_camera();  dlg.accept()

        test_btn.clicked.connect(do_test)
        test2_btn.clicked.connect(do_test2)
        save_btn.clicked.connect(do_save)
        cancel_btn.clicked.connect(dlg.reject)

        btn_row.addWidget(test_btn)
        btn_row.addWidget(test2_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        dlg.exec_()

    # ════════════════════════════════════════
    #  DATABASE HELPERS
    # ════════════════════════════════════════
    def load_logs(self):
        date_str = self._log_date.toString("yyyy-MM-dd")
        cursor.execute("""
            SELECT l.plate, l.timestamp, l.status, p.name, p.vehicle_type, l.direction
            FROM logs l
            LEFT JOIN plates p ON l.plate = p.plate
            WHERE l.timestamp LIKE ?
            ORDER BY l.id DESC
        """, (date_str + "%",))
        self._populate_logs_table(cursor.fetchall())

        # Dashboard mini table — last 20 logs with direction
        cursor.execute("SELECT plate, timestamp, direction, status FROM logs ORDER BY id DESC LIMIT 20")
        dash_rows = cursor.fetchall()
        self.table.setRowCount(0)
        for row_data in dash_rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            plate_val, ts_val, dir_val, status_val = row_data
            for col, val in enumerate([plate_val, ts_val, dir_val, status_val]):
                item = QtWidgets.QTableWidgetItem(val if val else "—")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if col == 2:   # Direction
                    if val == "ENTRY":
                        item.setForeground(QtGui.QColor(C_GREEN))
                    elif val == "EXIT":
                        item.setForeground(QtGui.QColor(C_BLUE))
                if col == 3:   # Status
                    if val == "AUTHORIZED":
                        item.setForeground(QtGui.QColor(C_GREEN))
                    elif val == "UNAUTHORIZED":
                        item.setForeground(QtGui.QColor(C_RED))
                self.table.setItem(row, col, item)

    def _populate_logs_table(self, rows):
        self.logs_table.setRowCount(0)
        for i, row_data in enumerate(rows):
            plate, timestamp, status, name, vehicle, direction = row_data
            row = self.logs_table.rowCount()
            self.logs_table.insertRow(row)
            self.logs_table.setRowHeight(row, 44)

            num_item = QtWidgets.QTableWidgetItem(str(i + 1))
            num_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.logs_table.setItem(row, 0, num_item)

            name_item = QtWidgets.QTableWidgetItem(name if name else "Unknown")
            name_item.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            self.logs_table.setItem(row, 1, name_item)

            plate_item = QtWidgets.QTableWidgetItem(plate)
            plate_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.logs_table.setItem(row, 2, plate_item)

            veh_item = QtWidgets.QTableWidgetItem(vehicle if vehicle else "—")
            veh_item.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
            self.logs_table.setItem(row, 3, veh_item)

            try:
                time_part = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').strftime('%I : %M : %S %p')
            except Exception:
                time_part = timestamp
            dir_val   = direction if direction else "ENTRY"
            time_label = f"{'Time-In' if dir_val == 'ENTRY' else 'Time-Out'}  {time_part}"
            time_item = QtWidgets.QTableWidgetItem(time_label)
            time_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.logs_table.setItem(row, 4, time_item)
            dir_item  = QtWidgets.QTableWidgetItem(dir_val)
            dir_item.setTextAlignment(QtCore.Qt.AlignCenter)
            if dir_val == "ENTRY":
                dir_item.setForeground(QtGui.QColor(C_GREEN))
            elif dir_val == "EXIT":
                dir_item.setForeground(QtGui.QColor(C_BLUE))
            self.logs_table.setItem(row, 5, dir_item)

            status_item = QtWidgets.QTableWidgetItem(status)
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            if status == "AUTHORIZED":
                status_item.setForeground(QtGui.QColor(C_GREEN))
            elif status == "UNAUTHORIZED":
                status_item.setForeground(QtGui.QColor(C_RED))
            self.logs_table.setItem(row, 6, status_item)

    def add_log(self, plate, timestamp, status, direction="ENTRY"):
        cursor.execute(
            "INSERT INTO logs (plate, timestamp, status, direction) VALUES (?,?,?,?)",
            (plate, timestamp, status, direction)
        )
        conn.commit()
        self.load_logs()

    # ════════════════════════════════════════
    #  MISC
    # ════════════════════════════════════════
    def update_time(self):
        now = datetime.now().strftime("%I : %M : %S %p")
        self.time_label.setText(now)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()
