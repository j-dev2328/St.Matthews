from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon
import os

USERS = {
    "admin": "admin123",
    "guard": "guard123"
}


class LoginDialog(QtWidgets.QDialog):
    """
    Modal login dialog shown inside the Register tab.
    On success, emits login_success(role: str).
    """
    login_success = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("St. Matthew Subdivision")
        self.setFixedSize(980, 580)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.setStyleSheet("background:#EFAF4F;")

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── HEADER ──
        header = QtWidgets.QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet("background:#FFFFFF;")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(12)

        logo_circle = QtWidgets.QLabel("M")
        logo_circle.setFixedSize(40, 40)
        logo_circle.setAlignment(QtCore.Qt.AlignCenter)
        logo_circle.setStyleSheet(
            "background:#EFAF4F; border:2px solid #333; border-radius:20px;"
            "font-weight:bold; font-size:18px; color:#333;"
        )

        header_title = QtWidgets.QLabel("St. Matthew Subdivision")
        header_title.setStyleSheet("font-size:18px; font-weight:bold; color:#111;")

        hl.addWidget(logo_circle)
        hl.addWidget(header_title)
        hl.addStretch()
        root.addWidget(header)

        # ── BODY (orange) ──
        body = QtWidgets.QWidget()
        body.setStyleSheet("background:#EFAF4F;")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setAlignment(QtCore.Qt.AlignCenter)
        body_layout.setContentsMargins(0, 0, 0, 0)

        # ── CARD ──
        card = QtWidgets.QFrame()
        card.setFixedSize(420, 420)
        card.setStyleSheet("""
            QFrame {
                background:#FFFFFF;
                border-radius:24px;
            }
        """)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(44, 32, 44, 28)
        card_layout.setSpacing(0)

        # Avatar circle
        avatar = QtWidgets.QLabel()
        avatar.setFixedSize(64, 64)
        avatar.setAlignment(QtCore.Qt.AlignCenter)
        avatar.setText("👤")
        avatar.setStyleSheet(
            "background:#EFAF4F; border-radius:32px; font-size:28px;"
            "border: 2px solid #d99b3a;"
        )
        avatar_row = QtWidgets.QHBoxLayout()
        avatar_row.addStretch()
        avatar_row.addWidget(avatar)
        avatar_row.addStretch()
        card_layout.addLayout(avatar_row)
        card_layout.addSpacing(10)

        # "Admin" label
        role_lbl = QtWidgets.QLabel("Admin")
        role_lbl.setAlignment(QtCore.Qt.AlignCenter)
        role_lbl.setStyleSheet("font-size:16px; font-weight:bold; color:#111; background:transparent;")
        card_layout.addWidget(role_lbl)
        card_layout.addSpacing(6)

        # Divider
        div = QtWidgets.QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background:#DDDDDD;")
        card_layout.addWidget(div)
        card_layout.addSpacing(16)

        # ── Username field ──
        user_frame = QtWidgets.QFrame()
        user_frame.setFixedHeight(44)
        user_frame.setStyleSheet(
            "QFrame { background:#F0F0F0; border-radius:10px; border:1px solid #E0E0E0; }"
        )
        user_row = QtWidgets.QHBoxLayout(user_frame)
        user_row.setContentsMargins(12, 0, 12, 0)
        user_row.setSpacing(8)

        user_icon = QtWidgets.QLabel("👤")
        user_icon.setStyleSheet("font-size:15px; background:transparent; border:none;")
        user_icon.setFixedWidth(20)

        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText("Username or Email")
        self.username.setStyleSheet(
            "border:none; background:transparent; font-size:13px; color:#333;"
        )

        user_row.addWidget(user_icon)
        user_row.addWidget(self.username)
        card_layout.addWidget(user_frame)
        card_layout.addSpacing(10)

        # ── Password field ──
        pass_frame = QtWidgets.QFrame()
        pass_frame.setFixedHeight(44)
        pass_frame.setStyleSheet(
            "QFrame { background:#F0F0F0; border-radius:10px; border:1px solid #E0E0E0; }"
        )
        pass_row = QtWidgets.QHBoxLayout(pass_frame)
        pass_row.setContentsMargins(12, 0, 12, 0)
        pass_row.setSpacing(8)

        lock_icon = QtWidgets.QLabel("🔒")
        lock_icon.setStyleSheet("font-size:15px; background:transparent; border:none;")
        lock_icon.setFixedWidth(20)

        self.password = QtWidgets.QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setStyleSheet(
            "border:none; background:transparent; font-size:13px; color:#333;"
        )
        self.password.returnPressed.connect(self.do_login)

        # ── Toggle button with PNG icons ──
        self.toggle_btn = QtWidgets.QPushButton()
        self.toggle_btn.setFixedSize(24, 24)
        self.toggle_btn.setStyleSheet(
            "QPushButton { border:none; background:transparent; }"
            "QPushButton:hover { background:transparent; }"
        )
        self.toggle_btn.setCursor(QtCore.Qt.PointingHandCursor)

        # Load eye_closed icon as default (password is hidden initially)
        self._icon_eye_open   = self._load_icon("eye_open.png")
        self._icon_eye_closed = self._load_icon("eye_closed.png")
        self.toggle_btn.setIcon(self._icon_eye_closed)
        self.toggle_btn.setIconSize(QtCore.QSize(18, 18))

        self.toggle_btn.clicked.connect(self.toggle_password)

        pass_row.addWidget(lock_icon)
        pass_row.addWidget(self.password)
        pass_row.addWidget(self.toggle_btn)
        card_layout.addWidget(pass_frame)
        card_layout.addSpacing(8)

        # ── Remember me ──
        remember = QtWidgets.QCheckBox("Remember me")
        remember.setStyleSheet(
            "QCheckBox { font-size:13px; color:#444; background:transparent; }"
            "QCheckBox::indicator { width:16px; height:16px; border:1px solid #aaa; border-radius:3px; }"
            "QCheckBox::indicator:checked { background:#EFAF4F; border-color:#EFAF4F; }"
        )
        card_layout.addWidget(remember)
        card_layout.addSpacing(12)

        # ── Error message ──
        self.msg = QtWidgets.QLabel("")
        self.msg.setAlignment(QtCore.Qt.AlignCenter)
        self.msg.setStyleSheet("color:#E74C3C; font-size:12px; background:transparent;")
        card_layout.addWidget(self.msg)
        card_layout.addSpacing(4)

        # ── Log In button ──
        login_btn = QtWidgets.QPushButton("Log In")
        login_btn.setFixedHeight(44)
        login_btn.setStyleSheet("""
            QPushButton {
                background:#5A5A5A; color:white;
                border-radius:10px; font-weight:bold; font-size:14px;
            }
            QPushButton:hover { background:#444444; }
            QPushButton:pressed { background:#333333; }
        """)
        login_btn.clicked.connect(self.do_login)
        card_layout.addWidget(login_btn)
        card_layout.addSpacing(10)

        # ── Forgot Password ──
        forgot = QtWidgets.QLabel('<a href="#" style="color:#555; font-size:12px;">Forgot Password?</a>')
        forgot.setAlignment(QtCore.Qt.AlignCenter)
        forgot.setStyleSheet("background:transparent;")
        forgot.setOpenExternalLinks(False)
        card_layout.addWidget(forgot)

        body_layout.addWidget(card)
        root.addWidget(body)

    def _load_icon(self, filename: str) -> QtGui.QIcon:
        """
        Load a PNG icon from the same directory as this script.
        Falls back to an empty icon if the file is not found.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return QtGui.QIcon(path)
        # Fallback: return a blank icon so the app still runs
        return QtGui.QIcon()

    def toggle_password(self):
        if self.password.echoMode() == QtWidgets.QLineEdit.Password:
            self.password.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.toggle_btn.setIcon(self._icon_eye_open)     # show "eye open" — password visible
        else:
            self.password.setEchoMode(QtWidgets.QLineEdit.Password)
            self.toggle_btn.setIcon(self._icon_eye_closed)   # show "eye closed" — password hidden

    def do_login(self):
        u = self.username.text().strip()
        p = self.password.text()
        if u in USERS and USERS[u] == p:
            self.login_success.emit(u)
            self.accept()
        else:
            self.msg.setText("Invalid username or password.")
            self.password.clear()
            self.password.setFocus()
