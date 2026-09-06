from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
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
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

try:
    from .tool_registry import TOOLS, ToolSpec, load_tool_class
except ImportError:
    from tool_registry import TOOLS, ToolSpec, load_tool_class


APP_TITLE = "遥感与点云综合工具箱"
CATEGORIES = ("全部", "图像处理", "点云处理", "数据集")
_STANDARD_ICON_CACHE: dict[str, QIcon] = {}


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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(178)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(spec.summary)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setObjectName("toolIcon")
        icon = get_standard_icon(self.style(), spec.icon_name)
        icon_label.setPixmap(icon.pixmap(26, 26))
        header.addWidget(icon_label, 0, Qt.AlignTop)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel(spec.title)
        title.setObjectName("toolTitle")
        title.setWordWrap(True)
        title_block.addWidget(title)

        category = QLabel(spec.category)
        category.setObjectName("toolCategory")
        title_block.addWidget(category)
        header.addLayout(title_block, 1)
        layout.addLayout(header)

        summary = QLabel(spec.summary)
        summary.setObjectName("toolSummary")
        summary.setWordWrap(True)
        summary.setMinimumHeight(44)
        layout.addWidget(summary)

        tag_row = QHBoxLayout()
        tag_row.setSpacing(6)
        for tag in spec.tags[:4]:
            pill = QLabel(tag)
            pill.setObjectName("tagPill")
            tag_row.addWidget(pill, 0)
        tag_row.addStretch(1)
        layout.addLayout(tag_row)

        layout.addStretch(1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        open_button = QPushButton("打开")
        open_button.setObjectName("primaryButton")
        open_button.setIcon(get_standard_icon(self.style(), "SP_ArrowForward"))
        open_button.clicked.connect(lambda: self.open_requested.emit(self.spec))
        action_row.addWidget(open_button, 0)
        layout.addLayout(action_row)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_requested.emit(self.spec)
        super().mouseDoubleClickEvent(event)


class ToolboxWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1080, 720)
        self.resize(1180, 760)
        self.open_windows: Dict[str, QWidget] = {}
        self.active_category = "全部"
        self.category_buttons: Dict[str, QPushButton] = {}
        self.search_index = {
            spec.id: " ".join((spec.title, spec.category, spec.summary, *spec.tags)).lower()
            for spec in TOOLS
        }

        self._init_ui()
        self._center_on_screen()
        self.render_tools()

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

    def _create_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(248)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(12)

        title = QLabel("综合工具箱")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)

        subtitle = QLabel("图像处理 · 数据集 · 点云")
        subtitle.setObjectName("sidebarSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(14)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for category in CATEGORIES:
            button = QPushButton(category)
            button.setObjectName("categoryButton")
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, name=category: self.set_category(name))
            layout.addWidget(button)
            group.addButton(button)
            self.category_buttons[category] = button

        self.category_buttons["全部"].setChecked(True)

        layout.addStretch(1)

        hint = QLabel("新增工具：在 Toolbox/tool_registry.py 中追加 ToolSpec 即可接入。")
        hint.setObjectName("sidebarHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.sidebar_status = QLabel("状态：就绪")
        self.sidebar_status.setObjectName("sidebarStatus")
        self.sidebar_status.setWordWrap(True)
        layout.addWidget(self.sidebar_status)
        return sidebar

    def _create_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(16)

        top_row = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title = QLabel(APP_TITLE)
        title.setObjectName("pageTitle")
        title_block.addWidget(title)
        subtitle = QLabel("统一入口保留原有功能，按场景快速打开需要的工具。")
        subtitle.setObjectName("pageSubtitle")
        title_block.addWidget(subtitle)
        top_row.addLayout(title_block, 1)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("searchEdit")
        self.search_edit.setPlaceholderText("搜索工具、格式或关键词")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(300)
        self.search_edit.textChanged.connect(self.render_tools)
        top_row.addWidget(self.search_edit, 0, Qt.AlignTop)
        layout.addLayout(top_row)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("summaryLabel")
        layout.addWidget(self.summary_label)

        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        self.tools_view = QWidget()
        self.tools_layout = QGridLayout(self.tools_view)
        self.tools_layout.setContentsMargins(0, 0, 0, 0)
        self.tools_layout.setHorizontalSpacing(14)
        self.tools_layout.setVerticalSpacing(14)
        scroll.setWidget(self.tools_view)
        layout.addWidget(scroll, 1)

        return content

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        available = screen.availableGeometry()
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
        query = self.search_edit.text().strip().lower()
        for spec in TOOLS:
            if self.active_category != "全部" and spec.category != self.active_category:
                continue
            haystack = self.search_index[spec.id]
            if query and query not in haystack:
                continue
            yield spec

    def render_tools(self) -> None:
        while self.tools_layout.count():
            item = self.tools_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        tools = list(self.filtered_tools())
        total = len(TOOLS)
        self.summary_label.setText(f"当前显示 {len(tools)} / {total} 个工具")

        if not tools:
            empty = QLabel("没有找到匹配的工具。")
            empty.setObjectName("emptyLabel")
            empty.setAlignment(Qt.AlignCenter)
            self.tools_layout.addWidget(empty, 0, 0, 1, 2)
            self.tools_layout.setRowStretch(1, 1)
            return

        columns = 2
        for index, spec in enumerate(tools):
            card = ToolCard(spec)
            card.open_requested.connect(self.open_tool)
            row, col = divmod(index, columns)
            self.tools_layout.addWidget(card, row, col)

        row_count = (len(tools) + columns - 1) // columns
        self.tools_layout.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding), row_count, 0)
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

        self.set_status(f"正在打开：{spec.title}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            window_class = load_tool_class(spec)
            window = window_class()
            window.setAttribute(Qt.WA_DeleteOnClose, False)
            window.destroyed.connect(lambda _=None, tool_id=spec.id: self.open_windows.pop(tool_id, None))
            self.open_windows[spec.id] = window
            window.show()
            window.activateWindow()
            window.raise_()
            self.set_status(f"已打开：{spec.title}")
        except Exception as exc:
            traceback.print_exc()
            self.set_status(f"打开失败：{spec.title}")
            QMessageBox.critical(
                self,
                "工具启动失败",
                f"无法打开「{spec.title}」。\n\n错误信息：{exc}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def set_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)
        self.sidebar_status.setText(f"状态：{message}")


def apply_application_style(app: QApplication) -> None:
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")

    style_path = resource_path("style.qss")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as handle:
            app.setStyleSheet(handle.read())


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setFont(QFont("Microsoft YaHei", 10))
    apply_application_style(app)

    window = ToolboxWindow()
    window.setWindowIcon(window.style().standardIcon(QStyle.SP_ComputerIcon))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
