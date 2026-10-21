"""Compact, explicit format controls for prediction visualization."""
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QWheelEvent
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QCheckBox, QPlainTextEdit, QPushButton, QColorDialog, QSizePolicy,
    QApplication, QAbstractScrollArea,
)

from PointCloud.visualization_data import (
    POINT_LAYOUTS, VisualizationOptions, read_visualization_text, extract_visualization_data,
)


def _scroll_enclosing_page(widget, event):
    # Qt 5 does not reliably bubble an ignored wheel event to the viewport.
    # Preserve wheel/touchpad deltas and phases when forwarding it explicitly.
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            forwarded = QWheelEvent(event)
            QApplication.sendEvent(parent.viewport(), forwarded)
            event.setAccepted(forwarded.isAccepted())
            return
        parent = parent.parentWidget()
    event.ignore()


class _NoWheelComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        _scroll_enclosing_page(self, event)


class _NoWheelSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel and watched is self.lineEdit():
            _scroll_enclosing_page(self, event)
            return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event):
        _scroll_enclosing_page(self, event)


class VisualizationFormatWidget(QWidget):
    color_mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = None
        self.uniform_color = (0.5, 0.5, 0.5)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setSpacing(8)

        self.layout_combo = _NoWheelComboBox()
        self.layout_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.layout_combo.setMinimumContentsLength(12)
        for key, layout in POINT_LAYOUTS.items():
            self.layout_combo.addItem(layout.title, key)
        self.layout_combo.setCurrentIndex(self.layout_combo.findData("xyz_rgb_label"))
        form.addRow("数据排列", self.layout_combo)
        self.mode_combo = _NoWheelComboBox()
        form.addRow("着色方式", self.mode_combo)

        self.custom_widget = QWidget()
        custom_form = QFormLayout(self.custom_widget)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.xyz_spins, xyz_row = self._column_row(("X", "Y", "Z"), (1, 2, 3))
        custom_form.addRow("坐标列", xyz_row)
        self.label_spin = self._spin(7)
        self.label_row = QWidget()
        label_layout = QHBoxLayout(self.label_row)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.addWidget(QLabel("Label"))
        label_layout.addWidget(self.label_spin)
        label_layout.addWidget(QLabel("列号从 1 开始"))
        label_layout.addStretch()
        custom_form.addRow(self.label_row)
        self.rgb_spins, self.rgb_row = self._column_row(("R", "G", "B"), (4, 5, 6))
        custom_form.addRow(self.rgb_row)
        form.addRow(self.custom_widget)

        self.rgb_options = QWidget()
        rgb_form = QFormLayout(self.rgb_options)
        rgb_form.setContentsMargins(0, 0, 0, 0)
        self.range_combo = _NoWheelComboBox()
        for title, key in (("自动（按整份数据判断）", "auto"), ("0–255", "0-255"), ("0–1", "0-1")):
            self.range_combo.addItem(title, key)
        rgb_form.addRow("RGB 范围", self.range_combo)
        form.addRow(self.rgb_options)

        self.uniform_button = QPushButton("选择单色 · #808080")
        self.uniform_button.clicked.connect(self._choose_color)
        form.addRow(self.uniform_button)

        self.mapping_label = QLabel()
        self.mapping_label.setWordWrap(True)
        self.mapping_label.setStyleSheet("color: #64748B;")
        self.mapping_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        form.addRow(self.mapping_label)

        self.details_toggle = QCheckBox("读取选项与数据预览")
        form.addRow(self.details_toggle)
        self.details = QWidget()
        detail_form = QFormLayout(self.details)
        detail_form.setContentsMargins(0, 0, 0, 0)
        detail_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.delimiter_combo = _NoWheelComboBox()
        for title, key in (("自动（空白 / 逗号 / 分号）", "auto"), ("空格 / Tab", "whitespace"),
                           ("逗号 ,", "comma"), ("分号 ;", "semicolon"), ("Tab", "tab")):
            self.delimiter_combo.addItem(title, key)
        detail_form.addRow("分隔符", self.delimiter_combo)
        self.skip_spin = _NoWheelSpinBox()
        self.skip_spin.setRange(0, 1000000)
        self.skip_spin.setSuffix(" 行")
        self.skip_spin.setToolTip("按文件物理行数跳过表头；空行和 # 注释会自动忽略。")
        detail_form.addRow("跳过表头", self.skip_spin)
        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setFixedHeight(82)
        self.preview_edit.setPlaceholderText("选择文件后显示前 5 行数值；列号从 1 开始。")
        detail_form.addRow(self.preview_edit)
        form.addRow(self.details)
        self.details.hide()
        self.details_toggle.toggled.connect(self.details.setVisible)

        self.preview_label = QLabel("支持 TXT / XYZ / CSV 数值文本；请按实际列顺序选择数据排列。")
        self.preview_label.setWordWrap(True)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        form.addRow(self.preview_label)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(180)
        self.preview_timer.timeout.connect(self.refresh_preview)

        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        for spin in (*self.xyz_spins, self.label_spin, *self.rgb_spins, self.skip_spin):
            spin.valueChanged.connect(self._options_changed)
        self.delimiter_combo.currentIndexChanged.connect(self._options_changed)
        self.range_combo.currentIndexChanged.connect(self._options_changed)
        self._layout_changed()

    @staticmethod
    def _spin(value):
        spin = _NoWheelSpinBox()
        spin.setRange(1, 1000000)
        spin.setValue(value)
        spin.setMinimumWidth(64)
        return spin

    def _column_row(self, names, values):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        spins = []
        for name, value in zip(names, values):
            layout.addWidget(QLabel(name))
            spin = self._spin(value)
            layout.addWidget(spin)
            spins.append(spin)
        layout.addStretch()
        return spins, row

    def _layout_changed(self):
        key = self.layout_combo.currentData()
        layout = POINT_LAYOUTS[key]
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        if layout.label is not None or key == "custom":
            self.mode_combo.addItem("按标签映射颜色", "label")
        if layout.rgb is not None or key == "custom":
            self.mode_combo.addItem("原始 RGB 颜色", "rgb")
        self.mode_combo.addItem("统一单色", "uniform")
        self.mode_combo.blockSignals(False)
        self.custom_widget.setVisible(key == "custom")
        self._mode_changed()

    def _mode_changed(self):
        mode = self.mode_combo.currentData()
        self.label_row.setVisible(mode == "label")
        self.rgb_row.setVisible(mode == "rgb")
        self.rgb_options.setVisible(mode == "rgb")
        self.uniform_button.setVisible(mode == "uniform")
        self.color_mode_changed.emit(mode)
        self._options_changed()

    def _options_changed(self):
        options = self.options()
        layout = POINT_LAYOUTS[options.layout]
        custom = options.layout == "custom"
        xyz = options.xyz_columns if custom else (1, 2, 3)
        text = f"XYZ → 第 {', '.join(map(str, xyz))} 列"
        if options.color_mode == "label":
            text += f"；Label → 第 {options.label_column if custom else layout.label} 列；标签保持原编号。"
        elif options.color_mode == "rgb":
            rgb = options.rgb_columns if custom else layout.rgb
            text += f"；RGB → 第 {', '.join(map(str, rgb))} 列。"
        else:
            text += "；使用统一单色。"
        self.mapping_label.setText(text)
        if self.path:
            self.preview_timer.start()

    def _choose_color(self):
        selected = QColorDialog.getColor(QColor.fromRgbF(*self.uniform_color), self, "选择点云单色")
        if selected.isValid():
            self.uniform_color = (selected.redF(), selected.greenF(), selected.blueF())
            self.uniform_button.setText(f"选择单色 · {selected.name().upper()}")

    def options(self):
        return VisualizationOptions(
            layout=self.layout_combo.currentData(), color_mode=self.mode_combo.currentData(),
            xyz_columns=tuple(spin.value() for spin in self.xyz_spins), label_column=self.label_spin.value(),
            rgb_columns=tuple(spin.value() for spin in self.rgb_spins),
            delimiter=self.delimiter_combo.currentData(), skip_rows=self.skip_spin.value(),
            rgb_range=self.range_combo.currentData(), uniform_color=self.uniform_color,
        )

    def set_path(self, path, merged=False):
        self.path = path
        if merged:
            # Merge output is known to be whitespace-separated, with label in column 7.
            self.delimiter_combo.setCurrentIndex(0)
            self.skip_spin.setValue(0)
            self.layout_combo.setCurrentIndex(self.layout_combo.findData("xyz_rgb_label"))
            self.mode_combo.setCurrentIndex(self.mode_combo.findData("label"))
        self.refresh_preview()

    def refresh_preview(self):
        self.preview_timer.stop()
        if not self.path:
            return
        self.preview_edit.clear()
        try:
            data = read_visualization_text(self.path, self.options(), preview=True)
            # Keep wide inputs bounded as well; unused columns remain in the source file.
            visible = data[:, :12]
            heading = "  |  ".join(f"第 {i + 1} 列" for i in range(visible.shape[1]))
            lines = ["  |  ".join(f"{value:g}" for value in row) for row in visible]
            self.preview_edit.setPlainText("\n".join([heading, *lines]))
            extract_visualization_data(data, self.options())
            self.preview_label.setStyleSheet("color: #475569;")
            self.preview_label.setText(f"样本检查通过：{data.shape[1]} 列，已读前 {len(data)} 行；开始时检查完整文件。")
        except ValueError as exc:
            self.preview_label.setStyleSheet("color: #B45309;")
            self.preview_label.setText(str(exc))
