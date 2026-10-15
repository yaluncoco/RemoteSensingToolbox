"""Compact update entry and an on-demand release dialog."""
from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QTextBrowser, QToolButton, QVBoxLayout,
)

from Toolbox.updater import UpdateChecker
from Toolbox.update_installer import UpdateInstaller


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
    install_requested = pyqtSignal()

    def __init__(self, checker: UpdateChecker, parent=None, installer=None):
        super().__init__(parent)
        self.checker = checker
        self.installer = installer or UpdateInstaller(self)
        self.setObjectName("updateDialog")
        self.setWindowTitle("检查更新")
        self.resize(480, 360)
        self.setMinimumSize(420, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

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
        self.progress.setTextVisible(True)
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
        self.cancel_button = QPushButton("取消下载")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.clicked.connect(self.installer.cancel)
        actions.addWidget(self.cancel_button)
        self.download_button.clicked.connect(self.download_or_install)
        actions.addWidget(self.download_button)
        layout.addLayout(actions)
        checker.changed.connect(self.refresh)
        self.installer.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        checker, installer = self.checker, self.installer
        asset = checker.release.installer if checker.has_update else None
        matches = asset is not None and asset == installer.asset
        if installer.busy:
            if installer.state == "verifying":
                heading, message = "正在校验安装包…", "正在验证文件完整性，请稍候。"
            else:
                heading = "正在下载更新…"
                message = (f"已下载 {installer.received / 1048576:.1f} / {installer.total / 1048576:.1f} MB。"
                           "关闭此窗口后仍会继续下载。")
        elif matches and installer.state == "ready":
            heading = "更新已准备好"
            message = installer.error or "安装包已通过校验。安装时将退出工具箱，完成后自动启动新版。便携版将安装为当前用户版本。"
        elif matches and installer.state in ("error", "cancelled"):
            heading = "下载失败" if installer.state == "error" else "下载已取消"
            message = installer.error or "可随时重新下载，不影响当前版本使用。"
        elif checker.checking:
            heading, message = "正在检查更新…", "正在获取最新版本信息，请稍候。"
        elif checker.has_update:
            heading = "发现新版本"
            message = (checker.error or (f"可在此下载并安装新版（{asset.size / 1048576:.1f} MB）。"
                       if asset else "此版本暂无可校验的 Windows 安装包，请稍后重新检查。"))
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
        self.progress.setVisible(checker.checking or installer.busy)
        if installer.state == "downloading" and installer.busy and installer.total:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, int(installer.received * 100 / installer.total)))
        else:
            self.progress.setRange(0, 0)
        self.check_button.setEnabled(not checker.checking and not installer.busy)
        self.check_button.setText("检查中…" if checker.checking else "重新检查")
        self.cancel_button.setVisible(installer.busy)
        self.cancel_button.setText("取消校验" if installer.state == "verifying" else "取消下载")
        self.notes.setVisible(checker.has_update)
        notes = (checker.release.notes or "此版本暂无更新说明。") if checker.has_update else ""
        if self.notes.toPlainText() != notes:
            self.notes.setPlainText(notes)
        self.download_button.setVisible(not installer.busy and checker.has_update)
        self.download_button.setEnabled(asset is not None and not checker.checking)
        self.download_button.setText("安装并重启" if matches and installer.state == "ready" else
                                     "重新下载" if matches and installer.state in ("error", "cancelled") else "下载更新")

    def download_or_install(self) -> None:
        if not self.checker.has_update or self.checker.checking or self.installer.busy:
            return
        asset = self.checker.release.installer
        if asset is None:
            return
        if self.installer.asset == asset and self.installer.state == "ready":
            self.install_requested.emit()
        else:
            self.installer.start(asset)
