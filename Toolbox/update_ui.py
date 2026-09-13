"""Compact update entry and an on-demand release dialog."""
from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QTextBrowser, QToolButton, QVBoxLayout,
)

from Toolbox.updater import UpdateChecker


class UpdateButton(QToolButton):
    def __init__(self, checker: UpdateChecker, parent=None):
        super().__init__(parent)
        self.checker = checker
        self.setObjectName("updateButton")
        self.setFixedSize(36, 36)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        checker.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        if self.checker.has_update:
            label = f"发现新版本 {self.checker.release.version}，点击查看更新"
        elif self.checker.checking:
            label = "正在检查更新…"
        elif self.checker.state == "current":
            label = f"当前已是最新版本 {self.checker.current_version}，点击重新检查"
        elif self.checker.error:
            label = "检查更新失败，点击重试"
        else:
            label = "检查更新"
        self.setToolTip(label)
        self.setAccessibleName(label)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.save()
        painter.translate(8, 8)
        painter.scale(20 / 24, 20 / 24)
        painter.setPen(QPen(QColor("#DCE6F1"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        path = QPainterPath()
        path.moveTo(3, 11)
        path.cubicTo(3, 6.6, 6.6, 3, 11, 3)
        path.cubicTo(15, 3, 17, 5, 21, 9)
        path.moveTo(21, 3)
        path.lineTo(21, 9)
        path.lineTo(15, 9)
        path.moveTo(21, 13)
        path.cubicTo(21, 17.4, 17.4, 21, 13, 21)
        path.cubicTo(9, 21, 7, 19, 3, 15)
        path.moveTo(9, 15)
        path.lineTo(3, 15)
        path.lineTo(3, 21)
        painter.drawPath(path)
        painter.restore()
        if self.checker.has_update:
            painter.setPen(QPen(QColor("#17202A"), 2))
            painter.setBrush(QColor("#EF4444"))
            painter.drawEllipse(QPointF(29, 7), 4.5, 4.5)


class UpdateDialog(QDialog):
    def __init__(self, checker: UpdateChecker, parent=None):
        super().__init__(parent)
        self.checker = checker
        self.setObjectName("updateDialog")
        self.setWindowTitle("检查更新")
        self.resize(500, 380)
        self.setMinimumSize(420, 300)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        self.heading = QLabel()
        self.heading.setObjectName("updateHeading")
        self.heading.setTextFormat(Qt.PlainText)
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.versions = QLabel()
        self.versions.setTextFormat(Qt.PlainText)
        self.versions.setWordWrap(True)
        layout.addWidget(self.versions)
        self.message = QLabel()
        self.message.setTextFormat(Qt.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.notes = QTextBrowser()
        self.notes.setObjectName("releaseNotes")
        self.notes.setAccessibleName("更新说明")
        layout.addWidget(self.notes, 1)
        layout.addStretch()

        actions = QHBoxLayout()
        close_button = QPushButton("关闭")
        close_button.setObjectName("secondaryButton")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        actions.addStretch()
        self.check_button = QPushButton("重新检查")
        self.check_button.setObjectName("secondaryButton")
        self.check_button.clicked.connect(checker.check)
        actions.addWidget(self.check_button)
        self.download_button = QPushButton()
        self.download_button.clicked.connect(self.open_download)
        actions.addWidget(self.download_button)
        layout.addLayout(actions)
        checker.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        checker = self.checker
        if checker.checking:
            heading, message = "正在检查更新…", "正在获取最新版本信息，请稍候。"
        elif checker.has_update:
            heading = "发现新版本"
            message = checker.error or "新版已发布，可前往下载并安装。"
        elif checker.error:
            heading, message = "暂时无法检查更新", checker.error
        elif checker.state == "current":
            heading, message = "当前已是最新版本", "暂时没有可用更新。"
        else:
            heading, message = "检查更新", "点击“重新检查”获取最新版本信息。"
        self.heading.setText(heading)
        self.message.setText(message)
        versions = f"当前版本  {checker.current_version}"
        if checker.release is not None:
            versions += f"    最新版本  {checker.release.version}"
        self.versions.setText(versions)
        self.progress.setVisible(checker.checking)
        self.check_button.setEnabled(not checker.checking)
        self.check_button.setText("检查中…" if checker.checking else "重新检查")
        self.notes.setVisible(checker.has_update)
        self.notes.setPlainText((checker.release.notes or "此版本暂无更新说明。") if checker.has_update else "")
        self.download_button.setText("前往下载" if checker.has_update else "发布页面")

    def open_download(self) -> None:
        url = self.checker.release.url if self.checker.has_update else self.checker.releases_url
        if not QDesktopServices.openUrl(QUrl(url)):
            self.message.setText(f"无法打开浏览器，请复制此地址访问：\n{url}")
            self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
