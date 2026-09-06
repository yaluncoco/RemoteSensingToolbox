from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.Image import Resampling
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from Toolbox.task_support import bind_task
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDesktopWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


Image.MAX_IMAGE_PIXELS = None
IMAGE_FILTER = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
MAX_ANALYSIS_SIDE = 2200
_NDIMAGE = None
_NDIMAGE_CHECKED = False


@dataclass
class ColorBlock:
    index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    fill_ratio: float
    rgb: tuple[int, int, int]
    category: str = ""

    @property
    def hex_color(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def load_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > MAX_ANALYSIS_SIDE:
            image.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Resampling.LANCZOS)
        return image.copy()


def qimage_to_pil_rgb(qimage: QImage) -> Image.Image:
    qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, qimage.bytesPerLine()))
    arr = arr[:, : width * 4].reshape((height, width, 4)).copy()
    return Image.fromarray(arr, mode="RGBA").convert("RGB")


def build_candidate_mask(rgb: np.ndarray, color_delta: int, include_neutral: bool) -> np.ndarray:
    data = rgb.astype(np.int16, copy=False)
    max_channel = data.max(axis=2)
    min_channel = data.min(axis=2)
    delta = max_channel - min_channel

    near_white = (max_channel >= 245) & (delta <= 12)
    colorful = (delta >= color_delta) & (max_channel >= 35)
    if include_neutral:
        dark_or_gray = ((max_channel <= 88) | ((delta <= color_delta) & (max_channel <= 210))) & ~near_white
        return (colorful | dark_or_gray) & ~near_white
    return colorful & ~near_white


def get_ndimage():
    global _NDIMAGE, _NDIMAGE_CHECKED
    if _NDIMAGE_CHECKED:
        return _NDIMAGE
    try:
        from scipy import ndimage

        _NDIMAGE = ndimage
    except Exception:
        _NDIMAGE = None
    _NDIMAGE_CHECKED = True
    return _NDIMAGE


def iter_fast_components(mask: np.ndarray, ndimage):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        local_mask = labels[slices] == label_id
        ys, xs = np.nonzero(local_mask)
        if ys.size == 0:
            continue
        x_min = int(x_slice.start)
        y_min = int(y_slice.start)
        x_max = int(x_slice.stop - 1)
        y_max = int(y_slice.stop - 1)
        yield ys + y_min, xs + x_min, (x_min, y_min, x_max, y_max)


def iter_connected_components(mask: np.ndarray):
    height, width = mask.shape
    flat = mask.ravel()
    visited = np.zeros(flat.shape[0], dtype=bool)
    starts = np.flatnonzero(flat)

    for start in starts:
        if visited[start]:
            continue

        stack = [int(start)]
        visited[start] = True
        pixels: list[int] = []
        x_min = width
        x_max = -1
        y_min = height
        y_max = -1

        while stack:
            pos = stack.pop()
            pixels.append(pos)
            y, x = divmod(pos, width)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= height:
                    continue
                row_offset = ny * width
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= width or (ny == y and nx == x):
                        continue
                    npos = row_offset + nx
                    if flat[npos] and not visited[npos]:
                        visited[npos] = True
                        stack.append(int(npos))

        ys, xs = np.divmod(np.asarray(pixels, dtype=np.int64), width)
        yield ys, xs, (x_min, y_min, x_max, y_max)


def sort_blocks(blocks: list[ColorBlock], mode: str) -> list[ColorBlock]:
    if mode == "按行从左到右":
        return sorted(blocks, key=lambda block: (block.y + block.height / 2, block.x))

    if not blocks:
        return []

    median_width = float(np.median([block.width for block in blocks]))
    gap_threshold = max(median_width * 2.5, 60.0)
    by_x = sorted(blocks, key=lambda block: block.x + block.width / 2)
    columns: list[list[ColorBlock]] = [[by_x[0]]]
    previous_center = by_x[0].x + by_x[0].width / 2
    for block in by_x[1:]:
        center = block.x + block.width / 2
        if center - previous_center > gap_threshold:
            columns.append([block])
        else:
            columns[-1].append(block)
        previous_center = center

    ordered: list[ColorBlock] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda block: (block.y, block.x)))
    return ordered


def detect_color_blocks(
    image: Image.Image,
    min_area: int,
    min_width: int,
    min_height: int,
    color_delta: int,
    include_neutral: bool,
    sort_mode: str,
) -> list[ColorBlock]:
    rgb = np.asarray(image.convert("RGB"))
    mask = build_candidate_mask(rgb, color_delta=color_delta, include_neutral=include_neutral)

    blocks: list[ColorBlock] = []
    image_area = rgb.shape[0] * rgb.shape[1]
    ndimage = get_ndimage() if mask.size >= 450_000 else None
    if ndimage is None:
        components = iter_connected_components(mask)
    else:
        components = iter_fast_components(mask, ndimage)

    for ys, xs, (x_min, y_min, x_max, y_max) in components:
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = int(ys.size)
        if width < min_width or height < min_height or area < min_area:
            continue

        bbox_area = width * height
        fill_ratio = area / bbox_area if bbox_area else 0.0
        aspect = width / height if height else 0.0
        if fill_ratio < 0.55 or aspect < 0.35 or aspect > 8.5:
            continue
        if bbox_area > image_area * 0.20:
            continue

        sample = rgb[ys, xs, :]
        median_rgb = tuple(int(np.median(sample[:, channel])) for channel in range(3))
        blocks.append(
            ColorBlock(
                index=0,
                x=x_min,
                y=y_min,
                width=width,
                height=height,
                area=area,
                fill_ratio=fill_ratio,
                rgb=median_rgb,
            )
        )

    ordered = sort_blocks(blocks, sort_mode)
    for index, block in enumerate(ordered, start=1):
        block.index = index
    return ordered


class ColorExtractThread(QThread):
    finished_signal = pyqtSignal(object, object)

    def __init__(self, image, params):
        super().__init__()
        self.image = image.copy()
        self.params = params

    def run(self):
        try:
            self.finished_signal.emit(detect_color_blocks(self.image, **self.params), None)
        except Exception as exc:
            self.finished_signal.emit(None, exc)


class RgbColorExtractorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB 颜色提取工具")
        self.setGeometry(0, 0, 1060, 680)
        self._center_window()
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))

        self.image_path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self.blocks: list[ColorBlock] = []
        self.extract_thread = None

        self._init_ui()
        self._update_actions()

    def _center_window(self):
        try:
            qr = self.frameGeometry()
            cp = QDesktopWidget().availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())
        except Exception:
            pass

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 10)
        main_layout.setSpacing(8)

        toolbar = QFrame()
        toolbar.setObjectName("extractorToolbar")
        toolbar.setFrameShape(QFrame.StyledPanel)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(8)

        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.select_image_btn.clicked.connect(self.select_image)
        toolbar_layout.addWidget(self.select_image_btn)

        self.clipboard_btn = QPushButton("剪贴板")
        self.clipboard_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.clipboard_btn.clicked.connect(self.load_from_clipboard)
        toolbar_layout.addWidget(self.clipboard_btn)

        self.extract_btn = QPushButton("提取色块")
        self.extract_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.extract_btn.clicked.connect(self.extract_blocks)
        toolbar_layout.addWidget(self.extract_btn)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(5, 100000)
        self.min_area_spin.setValue(60)
        self.min_area_spin.setMaximumWidth(82)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(QLabel("面积"))
        toolbar_layout.addWidget(self.min_area_spin)

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(2, 500)
        self.min_width_spin.setValue(8)
        self.min_width_spin.setMaximumWidth(68)
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(2, 500)
        self.min_height_spin.setValue(6)
        self.min_height_spin.setMaximumWidth(68)
        toolbar_layout.addWidget(QLabel("宽"))
        toolbar_layout.addWidget(self.min_width_spin)
        toolbar_layout.addWidget(QLabel("高"))
        toolbar_layout.addWidget(self.min_height_spin)

        self.color_delta_spin = QSpinBox()
        self.color_delta_spin.setRange(5, 180)
        self.color_delta_spin.setValue(24)
        self.color_delta_spin.setToolTip("RGB 三通道最大差值阈值；数值越低，越容易识别浅色块。")
        self.color_delta_spin.setMaximumWidth(68)
        toolbar_layout.addWidget(QLabel("色差"))
        toolbar_layout.addWidget(self.color_delta_spin)

        self.sort_mode_combo = QComboBox()
        self.sort_mode_combo.addItems(["按列从上到下", "按行从左到右"])
        self.sort_mode_combo.setToolTip("图例为多列排布时，通常选择按列排序。")
        self.sort_mode_combo.setMaximumWidth(140)
        toolbar_layout.addWidget(self.sort_mode_combo)

        self.include_neutral_checkbox = QCheckBox("包含黑色/灰色色块")
        self.include_neutral_checkbox.setToolTip("开启后可检测低饱和色块，但文字也更容易被误识别。")
        toolbar_layout.addWidget(self.include_neutral_checkbox)
        toolbar_layout.addStretch(1)
        main_layout.addWidget(toolbar)

        content_splitter = QSplitter(Qt.Horizontal)

        preview_panel = QFrame()
        preview_panel.setObjectName("extractorPanel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)
        preview_title = QLabel("预览")
        preview_title.setObjectName("sectionTitle")
        preview_layout.addWidget(preview_title)
        self.preview_label = QLabel("选择图例截图后显示预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 300)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("background-color: white; border: 1px solid #cccccc;")
        preview_layout.addWidget(self.preview_label)
        content_splitter.addWidget(preview_panel)

        result_panel = QFrame()
        result_panel.setObjectName("extractorPanel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(10, 10, 10, 10)
        result_layout.setSpacing(8)
        result_title = QLabel("提取结果")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)
        self.result_table = QTableWidget(0, 9)
        self.result_table.setHorizontalHeaderLabels(["序号", "类别", "色块", "R", "G", "B", "HEX", "位置", "尺寸"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setAlternatingRowColors(True)
        result_layout.addWidget(self.result_table, 1)

        category_row = QHBoxLayout()
        category_row.setSpacing(8)
        self.category_text_edit = QTextEdit()
        self.category_text_edit.setPlaceholderText("可选：每行输入一个类别名称，顺序与表格色块一致。")
        self.category_text_edit.setFixedHeight(72)
        category_row.addWidget(self.category_text_edit, 1)
        self.apply_categories_btn = QPushButton("按行填充类别")
        self.apply_categories_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.apply_categories_btn.clicked.connect(self.apply_categories)
        category_row.addWidget(self.apply_categories_btn, 0, Qt.AlignBottom)
        result_layout.addLayout(category_row)
        content_splitter.addWidget(result_panel)
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 6)
        main_layout.addWidget(content_splitter, 1)

        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        self.copy_dict_btn = QPushButton("复制 Python 字典")
        self.copy_dict_btn.clicked.connect(self.copy_python_dicts)
        export_row.addWidget(self.copy_dict_btn)

        self.export_csv_btn = QPushButton("导出 CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_row.addWidget(self.export_csv_btn)

        self.export_json_btn = QPushButton("导出 JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        export_row.addWidget(self.export_json_btn)
        export_row.addStretch(1)

        self.status_label = QLabel("就绪")
        export_row.addWidget(self.status_label)
        main_layout.addLayout(export_row)

    def _set_button_path(self, text: str, tooltip: str = ""):
        display = os.path.basename(text) if os.path.exists(text) else text
        self.select_image_btn.setText(f"图片: {display}")
        self.select_image_btn.setToolTip(tooltip or text)

    def _update_actions(self):
        busy = self.extract_thread is not None
        has_image = self.image is not None and not busy
        has_blocks = bool(self.blocks) and not busy
        self.select_image_btn.setEnabled(not busy)
        self.clipboard_btn.setEnabled(not busy)
        self.extract_btn.setEnabled(has_image)
        self.apply_categories_btn.setEnabled(has_blocks)
        self.copy_dict_btn.setEnabled(has_blocks)
        self.export_csv_btn.setEnabled(has_blocks)
        self.export_json_btn.setEnabled(has_blocks)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图例截图", "", IMAGE_FILTER)
        if not path:
            return
        try:
            self.image = load_image_rgb(path)
            self.image_path = path
            self.blocks = []
            self.result_table.setRowCount(0)
            self._set_button_path(path, path)
            self.status_label.setText("已载入图片")
            self.render_preview()
            self._update_actions()
        except UnidentifiedImageError:
            QMessageBox.critical(self, "图像错误", "无法识别该图片文件。")
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", f"读取图片失败：{exc}")

    def load_from_clipboard(self):
        clipboard = QApplication.clipboard()
        qimage = clipboard.image()
        if qimage.isNull():
            QMessageBox.warning(self, "剪贴板为空", "剪贴板中没有可读取的截图。")
            return

        self.image = qimage_to_pil_rgb(qimage)
        if max(self.image.size) > MAX_ANALYSIS_SIDE:
            self.image.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Resampling.LANCZOS)
        self.image_path = "clipboard"
        self.blocks = []
        self.result_table.setRowCount(0)
        self._set_button_path("剪贴板截图")
        self.status_label.setText("已从剪贴板载入截图")
        self.render_preview()
        self._update_actions()

    def extract_blocks(self):
        if self.extract_thread is not None:
            return
        if self.image is None:
            QMessageBox.warning(self, "未选择图片", "请先选择图片或从剪贴板读取截图。")
            return

        params = dict(
            min_area=self.min_area_spin.value(),
            min_width=self.min_width_spin.value(),
            min_height=self.min_height_spin.value(),
            color_delta=self.color_delta_spin.value(),
            include_neutral=self.include_neutral_checkbox.isChecked(),
            sort_mode=self.sort_mode_combo.currentText(),
        )

        self.extract_thread = ColorExtractThread(self.image, params)
        bind_task(self, "extract_thread", self.on_extraction_finished)
        self.status_label.setText("正在提取色块…")
        self._update_actions()
        self.extract_thread.start()

    def on_extraction_finished(self, blocks, error):
        self.extract_thread = None
        if error is not None:
            self.status_label.setText("色块提取失败")
            self._update_actions()
            QMessageBox.critical(self, "提取失败", str(error))
            return
        self.blocks = blocks

        self.populate_table()
        self.render_preview()
        self.status_label.setText(f"已提取 {len(self.blocks)} 个色块")
        if not self.blocks:
            QMessageBox.information(self, "未检测到色块", "没有检测到符合参数的色块，可降低面积/尺寸阈值或颜色差阈值后重试。")
        self._update_actions()

    def render_preview(self):
        if self.image is None:
            self.preview_label.setText("选择图例截图后显示预览")
            self.preview_label.setPixmap(QPixmap())
            return

        rgb = np.ascontiguousarray(np.asarray(self.image.convert("RGB")))
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        max_width = max(360, self.preview_label.width() - 16)
        max_height = max(280, self.preview_label.height() - 16)
        scaled = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.blocks:
            sx = scaled.width() / width
            sy = scaled.height() / height
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            font = QFont()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            for block in self.blocks:
                rect_x = int(round(block.x * sx))
                rect_y = int(round(block.y * sy))
                rect_w = max(2, int(round(block.width * sx)))
                rect_h = max(2, int(round(block.height * sy)))
                painter.setPen(QPen(QColor(255, 255, 255), 4))
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                painter.setPen(QPen(QColor(220, 30, 30), 2))
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)
                painter.setBrush(QBrush(QColor(220, 30, 30)))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(rect_x, max(12, rect_y - 4), str(block.index))
            painter.end()

        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")

    def populate_table(self):
        self.result_table.setRowCount(len(self.blocks))
        for row, block in enumerate(self.blocks):
            self._set_item(row, 0, str(block.index), editable=False)
            self._set_item(row, 1, block.category, editable=True)

            color_item = QTableWidgetItem(block.hex_color)
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            color_item.setBackground(QColor(*block.rgb))
            color_item.setForeground(QColor(255, 255, 255) if sum(block.rgb) < 384 else QColor(0, 0, 0))
            self.result_table.setItem(row, 2, color_item)

            for offset, value in enumerate(block.rgb, start=3):
                self._set_item(row, offset, str(value), editable=False)
            self._set_item(row, 6, block.hex_color, editable=False)
            self._set_item(row, 7, f"{block.x}, {block.y}", editable=False)
            self._set_item(row, 8, f"{block.width} x {block.height}", editable=False)
        self.result_table.resizeRowsToContents()

    def _set_item(self, row: int, column: int, text: str, editable: bool):
        item = QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.result_table.setItem(row, column, item)

    def apply_categories(self):
        lines = [line.strip() for line in self.category_text_edit.toPlainText().splitlines() if line.strip()]
        if not lines:
            QMessageBox.warning(self, "类别为空", "请先在类别文本框中输入类别名称。")
            return
        for row, category in enumerate(lines[: self.result_table.rowCount()]):
            self.result_table.item(row, 1).setText(category)
        self.status_label.setText(f"已填充 {min(len(lines), self.result_table.rowCount())} 个类别")

    def table_rows(self) -> list[dict]:
        rows = []
        for row in range(self.result_table.rowCount()):
            category_item = self.result_table.item(row, 1)
            index = int(self.result_table.item(row, 0).text())
            r = int(self.result_table.item(row, 3).text())
            g = int(self.result_table.item(row, 4).text())
            b = int(self.result_table.item(row, 5).text())
            block = self.blocks[row]
            rows.append(
                {
                    "index": index,
                    "category": category_item.text().strip() if category_item else "",
                    "rgb": [r, g, b],
                    "hex": self.result_table.item(row, 6).text(),
                    "x": block.x,
                    "y": block.y,
                    "width": block.width,
                    "height": block.height,
                    "area": block.area,
                    "fill_ratio": round(block.fill_ratio, 4),
                }
            )
        return rows

    def copy_python_dicts(self):
        rows = self.table_rows()
        label_names = {row["index"]: row["category"] or f"class_{row['index']}" for row in rows}
        colors_by_id = {row["index"]: row["rgb"] for row in rows}
        colors_by_name = {label_names[row["index"]]: row["rgb"] for row in rows}
        text = (
            f"label_names = {repr(label_names)}\n\n"
            f"colors = {repr(colors_by_id)}\n\n"
            f"fixed_color_map = {repr(colors_by_name)}"
        )
        QApplication.clipboard().setText(text)
        self.status_label.setText("已复制 Python 字典")
        QMessageBox.information(self, "复制完成", "已复制 label_names、colors 和 fixed_color_map。")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", self._default_output_path(".csv"), "CSV Files (*.csv)")
        if not path:
            return
        rows = self.table_rows()
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "category", "r", "g", "b", "hex", "x", "y", "width", "height", "area", "fill_ratio"])
                for row in rows:
                    writer.writerow(
                        [
                            row["index"],
                            row["category"],
                            *row["rgb"],
                            row["hex"],
                            row["x"],
                            row["y"],
                            row["width"],
                            row["height"],
                            row["area"],
                            row["fill_ratio"],
                        ]
                    )
            self.status_label.setText(f"已导出 CSV: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"导出 CSV 失败：{exc}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", self._default_output_path(".json"), "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.table_rows(), handle, ensure_ascii=False, indent=2)
            self.status_label.setText(f"已导出 JSON: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"导出 JSON 失败：{exc}")

    def _default_output_path(self, suffix: str) -> str:
        if self.image_path and self.image_path != "clipboard":
            source = Path(self.image_path)
            return str(source.with_name(f"{source.stem}_rgb_colors{suffix}"))
        return str(Path.cwd() / f"rgb_colors{suffix}")
