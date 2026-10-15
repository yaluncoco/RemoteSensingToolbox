from __future__ import annotations

import os
import sys
import traceback
import logging
import argparse
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QShortcut,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

try:
    from .tool_registry import TOOLS, ToolSpec, load_tool_class
except ImportError:
    from tool_registry import TOOLS, ToolSpec, load_tool_class

from Toolbox.task_support import bind_task
from Toolbox.updater import UpdateChecker
from Toolbox.update_ui import UpdateButton, UpdateDialog
from Toolbox.update_installer import UpdateInstaller, launch_installer
from Toolbox.version import APP_ID, APP_TITLE, APP_VERSION


CATEGORIES = ("全部", "图像处理", "点云处理", "数据集")
WINDOW_LAYOUT_VERSION = 2
CARD_MIN_WIDTH = 244
_STANDARD_ICON_CACHE: dict[str, QIcon] = {}


class ToolLoader(QThread):
    finished_signal = pyqtSignal(object, object)

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def run(self):
        try:
            self.finished_signal.emit(load_tool_class(self.spec), None)
        except Exception as exc:
            logging.exception("Failed to load tool %s", self.spec.id)
            self.finished_signal.emit(None, exc)


def resource_path(relative_path: str) -> str:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_path / relative_path)


def get_standard_icon(style: QStyle, icon_name: str) -> QIcon:
    cached = _STANDARD_ICON_CACHE.get(icon_name)
    if cached is not None:
        return cached
    icon_id = getattr(QStyle, icon_name, QStyle.SP_FileIcon)
    icon = style.standardIcon(icon_id)
    _STANDARD_ICON_CACHE[icon_name] = icon
    return icon


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class ToolCard(QFrame):
    open_requested = pyqtSignal(object)

    def __init__(self, spec: ToolSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("toolCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(144)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(spec.summary)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        icon_label = QLabel()
        icon_label.setObjectName("toolIcon")
        icon_label.setFixedSize(32, 32)
        icon_label.setAlignment(Qt.AlignCenter)
        icon = get_standard_icon(self.style(), spec.icon_name)
        icon_label.setPixmap(icon.pixmap(20, 20))
        header.addWidget(icon_label)

        title = QLabel(spec.title)
        title.setObjectName("toolTitle")
        title.setWordWrap(True)
        header.addWidget(title, 1)
        layout.addLayout(header)

        summary = QLabel(spec.summary)
        summary.setObjectName("toolSummary")
        summary.setWordWrap(True)
        summary.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        summary.setMinimumHeight(36)
        layout.addWidget(summary, 1)

        action_row = QHBoxLayout()
        category = QLabel(spec.category)
        category.setObjectName("toolCategory")
        action_row.addWidget(category)
        action_row.addStretch(1)
        open_button = QPushButton("打开")
        open_button.setObjectName("openToolButton")
        open_button.setAccessibleName(f"打开{spec.title}")
        open_button.setCursor(Qt.PointingHandCursor)
        open_button.clicked.connect(lambda: self.open_requested.emit(self.spec))
        action_row.addWidget(open_button, 0)
        layout.addLayout(action_row)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_requested.emit(self.spec)
        super().mouseDoubleClickEvent(event)


class ToolboxWindow(QMainWindow):
    def __init__(self, persist_settings=True, check_updates=True):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(640, 420)
        self.resize(960, 600)
        self.settings = QSettings(APP_ID, APP_ID) if persist_settings else None
        self._restored_geometry = False
        self.update_checker = UpdateChecker(self, settings=self.settings)
        self.update_installer = UpdateInstaller(self)
        self.update_installer.install_ready.connect(self._launch_update)
        self.update_dialog = None
        self._loader = None
        self._closing_children = False
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(120)
        self.render_timer.timeout.connect(self.render_tools)
        self.open_windows: Dict[str, QWidget] = {}
        self._detached_visualizations = []
        self.active_category = "全部"
        self.category_buttons: Dict[str, QPushButton] = {}
        self.search_index = {
            spec.id: " ".join((spec.id, spec.title, spec.category, spec.summary, *spec.tags)).casefold()
            for spec in TOOLS
        }

        self._init_ui()
        self.tool_cards = {}
        for spec in TOOLS:
            card = ToolCard(spec, self.tools_view)
            card.open_requested.connect(self.open_tool)
            self.tool_cards[spec.id] = card
        self.empty_label = QLabel("没有找到匹配的工具，请尝试其他关键词。", self.tools_view)
        self.empty_label.setObjectName("emptyLabel")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.hide()
        self.search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.search_shortcut.activated.connect(self.search_edit.setFocus)
        if self.settings:
            geometry = self.settings.value("geometry")
            if geometry and self.settings.value("layout_version", 0, type=int) == WINDOW_LAYOUT_VERSION:
                self._restored_geometry = self.restoreGeometry(geometry)
            category = self.settings.value("category", "全部")
            if category in CATEGORIES:
                self.active_category = category
                self.category_buttons[category].setChecked(True)
        self._center_on_screen()
        self.render_tools()
        if check_updates:
            self.update_checker.start_automatic_checks()

    def _init_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shell_layout.addWidget(self._create_sidebar(), 0)
        shell_layout.addWidget(self._create_content(), 1)
        self.setCentralWidget(shell)
        self.statusBar().showMessage("就绪")
        self.loading_indicator = QProgressBar()
        self.loading_indicator.setRange(0, 0)
        self.loading_indicator.setMaximumWidth(120)
        self.loading_indicator.hide()
        self.statusBar().addPermanentWidget(self.loading_indicator)

    def _create_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(160)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 12)
        layout.setSpacing(6)

        title = QLabel("综合工具箱")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        subtitle = QLabel("影像 · 点云 · 数据集")
        subtitle.setObjectName("sidebarSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for category in CATEGORIES:
            button = QPushButton(category)
            button.setObjectName("categoryButton")
            button.setCheckable(True)
            button.setMinimumHeight(34)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, name=category: self.set_category(name))
            layout.addWidget(button)
            group.addButton(button)
            self.category_buttons[category] = button

        self.category_buttons["全部"].setChecked(True)

        layout.addStretch(1)
        self.update_button = UpdateButton(self.update_checker, sidebar)
        self.update_button.clicked.connect(self.open_updates)
        layout.addWidget(self.update_button, 0, Qt.AlignLeft)
        return sidebar

    def _create_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        self.page_title = QLabel("全部工具")
        self.page_title.setObjectName("pageTitle")
        title_block.addWidget(self.page_title)
        self.summary_label = QLabel()
        self.summary_label.setObjectName("summaryLabel")
        title_block.addWidget(self.summary_label)
        top_row.addLayout(title_block, 1)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("searchEdit")
        self.search_edit.setPlaceholderText("搜索工具、格式或关键词")
        self.search_edit.setToolTip("搜索工具、格式或关键词（Ctrl+K）")
        self.search_edit.setAccessibleName("搜索工具")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(180)
        self.search_edit.setMaximumWidth(300)
        self.search_edit.textChanged.connect(lambda _text: self.render_timer.start())
        top_row.addWidget(self.search_edit, 1)
        layout.addLayout(top_row)

        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        self.tools_view = QWidget()
        self.tools_view.setObjectName("toolsView")
        self.tools_layout = QGridLayout(self.tools_view)
        self.tools_layout.setContentsMargins(0, 0, 0, 0)
        self.tools_layout.setHorizontalSpacing(10)
        self.tools_layout.setVerticalSpacing(10)
        scroll.setWidget(self.tools_view)
        self.tool_scroll = scroll
        layout.addWidget(scroll, 1)

        return content

    def _center_on_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            return
        available = screen.availableGeometry()
        max_width = max(1, available.width() - 24)
        max_height = max(1, available.height() - 48)
        self.setMinimumSize(min(640, max_width), min(420, max_height))
        self.resize(min(self.width(), max_width), min(self.height(), max_height))
        if self._restored_geometry:
            frame = self.frameGeometry()
            if frame.intersects(available):
                self.move(max(available.left(), min(self.x(), available.right() - frame.width() + 1)),
                          max(available.top(), min(self.y(), available.bottom() - frame.height() + 1)))
                return
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def set_category(self, category: str) -> None:
        self.active_category = category
        for name, button in self.category_buttons.items():
            button.setProperty("active", name == category)
            repolish(button)
        self.render_tools()

    def filtered_tools(self) -> Iterable[ToolSpec]:
        words = self.search_edit.text().casefold().split()
        for spec in TOOLS:
            if self.active_category != "全部" and spec.category != self.active_category:
                continue
            haystack = self.search_index[spec.id]
            if not all(word in haystack for word in words):
                continue
            yield spec

    def render_tools(self) -> None:
        while self.tools_layout.count():
            item = self.tools_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()

        for row in range(len(TOOLS) + 1):
            self.tools_layout.setRowStretch(row, 0)
        for column in range(3):
            self.tools_layout.setColumnStretch(column, 0)

        tools = list(self.filtered_tools())
        total = len(TOOLS)
        self.page_title.setText("全部工具" if self.active_category == "全部" else self.active_category)
        self.summary_label.setText(f"{len(tools)} / {total} 个工具")

        if not tools:
            self.tools_layout.addWidget(self.empty_label, 0, 0)
            self.empty_label.show()
            self.tools_layout.setRowStretch(1, 1)
            return

        # Reserve scrollbar space so filtering cannot make the grid alternate
        # between column counts when it is close to a breakpoint.
        width = self.tool_scroll.contentsRect().width() - self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        spacing = self.tools_layout.horizontalSpacing()
        columns = max(1, min(3, (width + spacing) // (CARD_MIN_WIDTH + spacing)))
        for index, spec in enumerate(tools):
            card = self.tool_cards[spec.id]
            row, col = divmod(index, columns)
            self.tools_layout.addWidget(card, row, col)
            card.show()

        row_count = (len(tools) + columns - 1) // columns
        self.tools_layout.setRowStretch(row_count, 1)
        for column in range(columns):
            self.tools_layout.setColumnStretch(column, 1)

    def open_tool(self, spec: ToolSpec) -> None:
        existing = self.open_windows.get(spec.id)
        if existing is not None:
            existing.show()
            existing.activateWindow()
            existing.raise_()
            self.set_status(f"已切换到：{spec.title}")
            return

        if self._loader is not None:
            self.set_status("正在加载工具，请稍候…")
            return

        self.set_status(f"正在打开：{spec.title}，首次加载可能需要几秒")
        self.loading_indicator.show()
        self.tools_view.setEnabled(False)
        self._loader = ToolLoader(spec)
        bind_task(self, "_loader", lambda window_class, error: self._tool_loaded(spec, window_class, error))
        self._loader.start()

    def _tool_loaded(self, spec, window_class, error):
        self.loading_indicator.hide()
        self.tools_view.setEnabled(True)
        try:
            if error is not None:
                raise error
            window = window_class()
            window.setAttribute(Qt.WA_DeleteOnClose, True)
            # Capture the Python state while the widget is alive. Looking up an
            # attribute on a destroyed Qt wrapper can abort the Windows process.
            window.destroyed.connect(lambda _=None, tool_id=spec.id, state=vars(window):
                                     self._tool_closed(tool_id, state.get("visualization_thread")))
            self.open_windows[spec.id] = window
            window.show()
            window.activateWindow()
            window.raise_()
            self.set_status(f"已打开：{spec.title}")
        except Exception as exc:
            logging.exception("Failed to open tool %s", spec.id)
            self.set_status(f"打开失败：{spec.title}")
            QMessageBox.critical(
                self,
                "工具启动失败",
                f"无法打开「{spec.title}」。\n\n错误信息：{exc}",
            )

    def _tool_closed(self, tool_id, visualization=None):
        self.open_windows.pop(tool_id, None)
        if visualization is not None and visualization.is_alive():
            self._detached_visualizations.append(visualization)
        if self._closing_children:
            QTimer.singleShot(0, self.close)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "tool_cards"):
            self.render_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.render_tools)

    def closeEvent(self, event):
        refused = []
        for window in list(self.open_windows.values()):
            if not window.close():
                refused.append(window)
        if refused:
            self._closing_children = all(bool(window.property("closing_task")) for window in refused)
            event.ignore()
            return
        if self.settings:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("layout_version", WINDOW_LAYOUT_VERSION)
            self.settings.setValue("category", self.active_category)
        self.update_checker.stop()
        self.update_installer.stop()
        if self.update_dialog is not None:
            self.update_dialog.close()
        event.accept()

    def open_updates(self):
        if self.update_dialog is None:
            self.update_dialog = UpdateDialog(self.update_checker, self, self.update_installer)
            self.update_dialog.install_requested.connect(self.install_update)
        self.update_dialog.show()
        self.update_dialog.raise_()
        self.update_dialog.activateWindow()
        if not self.update_checker.has_update:
            self.update_checker.check()

    def _update_tasks_running(self):
        self._detached_visualizations = [thread for thread in self._detached_visualizations if thread.is_alive()]
        if self._detached_visualizations:
            return True
        for window in [self, *self.open_windows.values()]:
            if any(isinstance(value, QThread) and value.isRunning() for value in vars(window).values()):
                return True
            visualization = getattr(window, "visualization_thread", None)
            if visualization is not None and visualization.is_alive():
                return True
        return False

    def install_update(self):
        if self._update_tasks_running():
            self.update_installer.fail("仍有任务或三维窗口在运行，请完成任务并关闭三维窗口后再安装。")
            return
        self.update_installer.prepare_install()

    def _launch_update(self):
        # Re-check after the asynchronous checksum pass, before any process is started.
        if self._update_tasks_running():
            self.update_installer.fail("仍有任务在运行，请完成后再次点击“安装并重启”。")
            return
        try:
            launch_installer(self.update_installer.path)
        except (OSError, RuntimeError) as exc:
            self.update_installer.fail(f"无法启动安装：{exc}。当前版本仍可继续使用。")
            return
        # The installer waits for this PID to exit, so no running files are replaced.
        self.close()

    def set_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)


def apply_application_style(app: QApplication) -> None:
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")

    style_path = resource_path("style.qss")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as handle:
            app.setStyleSheet(handle.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.add_argument("--self-test", action="store_true", help="Run offline checks and exit")
    parser.add_argument("--report", type=Path, help="Self-test JSON report path")
    parser.add_argument("--screenshots", type=Path, help="Self-test screenshots directory")
    parser.add_argument("--check-renderer", action="store_true", help="Include native OpenGL rendering in the self-test")
    args = parser.parse_args()

    from Toolbox.tool_registry import configure_runtime_environment
    configure_runtime_environment()
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "state"))) / APP_ID / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler(log_dir / "toolbox.log", maxBytes=2_000_000,
                                                      backupCount=2, encoding="utf-8")])
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([sys.argv[0]])
    app.setOrganizationName(APP_ID)
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)
    app.setFont(QFont("Microsoft YaHei", 10))
    apply_application_style(app)

    if args.self_test:
        from Toolbox.self_test import run_self_test
        report = args.report or log_dir / "self-test.json"
        return run_self_test(app, report, args.screenshots, args.check_renderer)

    def handle_exception(exc_type, value, tb):
        logging.error("Unhandled exception", exc_info=(exc_type, value, tb))
        QMessageBox.critical(None, "操作异常", f"{value}\n\n详细信息已保存到：{log_dir / 'toolbox.log'}")
    sys.excepthook = handle_exception

    window = ToolboxWindow()
    window.setWindowIcon(window.style().standardIcon(QStyle.SP_ComputerIcon))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
