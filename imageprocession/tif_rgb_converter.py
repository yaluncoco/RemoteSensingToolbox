import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDesktopWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


try:
    import rasterio

    RASTERIO_AVAILABLE = True
    RASTERIO_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    rasterio = None
    RASTERIO_AVAILABLE = False
    RASTERIO_IMPORT_ERROR = exc


TIF_FILTER = "TIFF Files (*.tif *.tiff);;All Files (*)"
OUTPUT_FORMATS = ("PNG", "JPG", "RGB TIF")


def normalize_band_to_uint8(band: np.ndarray, stretch_percent: float) -> np.ndarray:
    valid = np.asarray(band)
    finite_mask = np.isfinite(valid)
    if not finite_mask.any():
        return np.zeros(valid.shape, dtype=np.uint8)

    finite_values = valid[finite_mask].astype(np.float32, copy=False)
    if stretch_percent > 0:
        low = np.percentile(finite_values, stretch_percent)
        high = np.percentile(finite_values, 100.0 - stretch_percent)
    else:
        low = float(np.nanmin(finite_values))
        high = float(np.nanmax(finite_values))

    if high <= low:
        return np.zeros(valid.shape, dtype=np.uint8)

    scaled = (valid.astype(np.float32) - low) * (255.0 / (high - low))
    scaled = np.clip(scaled, 0, 255)
    scaled[~finite_mask] = 0
    return scaled.astype(np.uint8)


def build_rgb_array(src, red_band: int, green_band: int, blue_band: int, stretch_percent: float) -> np.ndarray:
    if src.count == 1:
        gray = normalize_band_to_uint8(src.read(1), stretch_percent)
        return np.dstack([gray, gray, gray])

    band_indexes = [red_band, green_band, blue_band]
    if any(index < 1 or index > src.count for index in band_indexes):
        raise ValueError(f"波段编号超出范围。当前 TIF 共有 {src.count} 个波段。")

    channels = [normalize_band_to_uint8(src.read(index), stretch_percent) for index in band_indexes]
    return np.dstack(channels)


def default_output_path(input_path: str, output_dir: str, output_format: str) -> str:
    stem = Path(input_path).stem
    suffix = {
        "PNG": ".png",
        "JPG": ".jpg",
        "RGB TIF": ".tif",
    }[output_format]
    return str(Path(output_dir) / f"{stem}_rgb{suffix}")


def convert_tif_to_rgb(
    input_path: str,
    output_path: str,
    output_format: str,
    red_band: int = 1,
    green_band: int = 2,
    blue_band: int = 3,
    stretch_percent: float = 2.0,
    preserve_georef: bool = True,
) -> None:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(f"rasterio 依赖不可用：{RASTERIO_IMPORT_ERROR}")

    output_format = output_format.upper()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with rasterio.open(input_path) as src:
        rgb = build_rgb_array(src, red_band, green_band, blue_band, stretch_percent)

        if output_format == "RGB TIF":
            meta = src.meta.copy() if preserve_georef else {
                "driver": "GTiff",
                "height": src.height,
                "width": src.width,
                "count": 3,
                "dtype": "uint8",
            }
            meta.update(driver="GTiff", count=3, dtype="uint8", compress="lzw")
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(np.moveaxis(rgb, -1, 0))
            return

    image = Image.fromarray(rgb, mode="RGB")
    if output_format == "JPG":
        image.save(output_path, "JPEG", quality=95)
    elif output_format == "PNG":
        image.save(output_path, "PNG")
    else:
        raise ValueError(f"不支持的输出格式：{output_format}")


class TifRgbConvertThread(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str, str)

    def __init__(
        self,
        inputs: list[str],
        output_dir: str,
        output_format: str,
        red_band: int,
        green_band: int,
        blue_band: int,
        stretch_percent: float,
        preserve_georef: bool,
    ):
        super().__init__()
        self.inputs = inputs
        self.output_dir = output_dir
        self.output_format = output_format
        self.red_band = red_band
        self.green_band = green_band
        self.blue_band = blue_band
        self.stretch_percent = stretch_percent
        self.preserve_georef = preserve_georef

    def run(self):
        try:
            if not self.inputs:
                self.finished_signal.emit("没有可转换的 TIF 文件。", "error")
                return

            total = len(self.inputs)
            converted = 0
            for index, input_path in enumerate(self.inputs, start=1):
                if self.isInterruptionRequested():
                    self.finished_signal.emit("操作已取消。", "info")
                    return

                output_path = default_output_path(input_path, self.output_dir, self.output_format)
                self.progress_updated.emit(
                    int((index - 1) / total * 100),
                    f"正在转换：{os.path.basename(input_path)}",
                )
                convert_tif_to_rgb(
                    input_path=input_path,
                    output_path=output_path,
                    output_format=self.output_format,
                    red_band=self.red_band,
                    green_band=self.green_band,
                    blue_band=self.blue_band,
                    stretch_percent=self.stretch_percent,
                    preserve_georef=self.preserve_georef,
                )
                converted += 1
                self.progress_updated.emit(
                    int(index / total * 100),
                    f"已输出：{output_path}",
                )

            self.finished_signal.emit(f"TIF 转 RGB 完成，共输出 {converted} 个文件。", "info")
        except Exception as exc:
            self.finished_signal.emit(f"转换失败：{exc}", "error")


class TifRgbConverterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIF 转 RGB 工具")
        self.setGeometry(0, 0, 760, 620)
        self._center_window()
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))

        self.single_input_path: Optional[str] = None
        self.single_output_dir: Optional[str] = None
        self.batch_input_dir: Optional[str] = None
        self.batch_output_dir: Optional[str] = None
        self.convert_thread: Optional[TifRgbConvertThread] = None

        self._init_ui()
        self._update_buttons()

        if not RASTERIO_AVAILABLE:
            QMessageBox.critical(
                self,
                "依赖缺失",
                "TIF 转 RGB 功能需要 rasterio 及其 GDAL 运行库。\n"
                f"当前环境导入失败：{RASTERIO_IMPORT_ERROR}",
            )
            for child in self.findChildren(QWidget):
                child.setEnabled(False)

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
        main_layout.setSpacing(10)

        tab_widget = QTabWidget()
        tab_widget.addTab(self._create_single_tab(), "单文件转换")
        tab_widget.addTab(self._create_batch_tab(), "批量转换")
        main_layout.addWidget(tab_widget)

        params_group = QGroupBox("转换参数")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(OUTPUT_FORMATS)
        self.output_format_combo.setToolTip("PNG/JPG 适合普通查看；RGB TIF 会尽量保留原图地理信息。")
        params_layout.addRow("输出格式:", self.output_format_combo)

        self.red_band_spin = QSpinBox()
        self.red_band_spin.setRange(1, 999)
        self.red_band_spin.setValue(1)
        self.green_band_spin = QSpinBox()
        self.green_band_spin.setRange(1, 999)
        self.green_band_spin.setValue(2)
        self.blue_band_spin = QSpinBox()
        self.blue_band_spin.setRange(1, 999)
        self.blue_band_spin.setValue(3)

        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("R"))
        band_row.addWidget(self.red_band_spin)
        band_row.addWidget(QLabel("G"))
        band_row.addWidget(self.green_band_spin)
        band_row.addWidget(QLabel("B"))
        band_row.addWidget(self.blue_band_spin)
        band_row.addStretch(1)
        params_layout.addRow("RGB 波段:", band_row)

        self.stretch_edit = QLineEdit("2")
        self.stretch_edit.setToolTip("按百分位拉伸到 8 位 RGB。常用值 1-2；填 0 表示按最小/最大值拉伸。")
        params_layout.addRow("百分位拉伸 (%):", self.stretch_edit)

        self.preserve_georef_checkbox = QCheckBox("输出 RGB TIF 时保留地理参考")
        self.preserve_georef_checkbox.setChecked(True)
        params_layout.addRow(self.preserve_georef_checkbox)
        main_layout.addWidget(params_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("转换日志将显示在此处...")
        self.log_edit.setFixedHeight(150)
        main_layout.addWidget(self.log_edit)

    def _create_single_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        paths_group = QGroupBox("输入与输出")
        form = QFormLayout(paths_group)
        form.setSpacing(10)

        self.single_input_btn = QPushButton("选择 TIF 文件")
        self.single_input_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.single_input_btn.clicked.connect(self.select_single_input)
        form.addRow(self.single_input_btn)

        self.single_output_btn = QPushButton("选择输出目录")
        self.single_output_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.single_output_btn.clicked.connect(self.select_single_output_dir)
        form.addRow(self.single_output_btn)
        layout.addWidget(paths_group)

        self.single_start_btn = QPushButton("开始转换单文件")
        self.single_start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.single_start_btn.clicked.connect(lambda: self.start_conversion(is_batch=False))
        layout.addWidget(self.single_start_btn, alignment=Qt.AlignCenter)
        layout.addStretch(1)
        return tab

    def _create_batch_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        paths_group = QGroupBox("输入与输出")
        form = QFormLayout(paths_group)
        form.setSpacing(10)

        self.batch_input_btn = QPushButton("选择 TIF 文件夹")
        self.batch_input_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.batch_input_btn.clicked.connect(self.select_batch_input_dir)
        form.addRow(self.batch_input_btn)

        self.batch_output_btn = QPushButton("选择输出目录")
        self.batch_output_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.batch_output_btn.clicked.connect(self.select_batch_output_dir)
        form.addRow(self.batch_output_btn)
        layout.addWidget(paths_group)

        self.batch_start_btn = QPushButton("开始批量转换")
        self.batch_start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.batch_start_btn.clicked.connect(lambda: self.start_conversion(is_batch=True))
        layout.addWidget(self.batch_start_btn, alignment=Qt.AlignCenter)
        layout.addStretch(1)
        return tab

    def _set_button_path(self, button: QPushButton, base_text: str, path: Optional[str]) -> None:
        if path:
            button.setText(f"{base_text}: {os.path.basename(path)}")
            button.setToolTip(path)
        else:
            button.setText(base_text)
            button.setToolTip("")

    def _update_buttons(self):
        enabled = RASTERIO_AVAILABLE and self.convert_thread is None
        self.single_start_btn.setEnabled(enabled and bool(self.single_input_path and self.single_output_dir))
        self.batch_start_btn.setEnabled(enabled and bool(self.batch_input_dir and self.batch_output_dir))

    def select_single_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 TIF 文件", "", TIF_FILTER)
        if path:
            self.single_input_path = path
            self._set_button_path(self.single_input_btn, "TIF 文件", path)
            if not self.single_output_dir:
                self.single_output_dir = os.path.dirname(path)
                self._set_button_path(self.single_output_btn, "输出目录", self.single_output_dir)
            self._update_buttons()

    def select_single_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.single_output_dir = path
            self._set_button_path(self.single_output_btn, "输出目录", path)
            self._update_buttons()

    def select_batch_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择 TIF 文件夹")
        if path:
            self.batch_input_dir = path
            self._set_button_path(self.batch_input_btn, "TIF 文件夹", path)
            if not self.batch_output_dir:
                self.batch_output_dir = path
                self._set_button_path(self.batch_output_btn, "输出目录", self.batch_output_dir)
            self._update_buttons()

    def select_batch_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.batch_output_dir = path
            self._set_button_path(self.batch_output_btn, "输出目录", path)
            self._update_buttons()

    def _read_stretch_percent(self) -> Optional[float]:
        try:
            value = float(self.stretch_edit.text().strip())
            if not (0.0 <= value < 50.0):
                raise ValueError("百分位拉伸必须在 0 到 50 之间。")
            return value
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", f"无效的百分位拉伸参数：{exc}")
            return None

    def _collect_inputs(self, is_batch: bool) -> tuple[list[str], Optional[str]]:
        if is_batch:
            if not self.batch_input_dir or not self.batch_output_dir:
                QMessageBox.warning(self, "路径缺失", "请选择输入文件夹和输出目录。")
                return [], None
            inputs = sorted(
                str(path)
                for path in Path(self.batch_input_dir).iterdir()
                if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
            )
            if not inputs:
                QMessageBox.warning(self, "没有文件", "输入文件夹中没有找到 .tif 或 .tiff 文件。")
                return [], None
            return inputs, self.batch_output_dir

        if not self.single_input_path or not self.single_output_dir:
            QMessageBox.warning(self, "路径缺失", "请选择 TIF 文件和输出目录。")
            return [], None
        return [self.single_input_path], self.single_output_dir

    def start_conversion(self, is_batch: bool):
        if self.convert_thread is not None:
            QMessageBox.warning(self, "正在处理", "当前转换任务尚未完成。")
            return
        if not RASTERIO_AVAILABLE:
            QMessageBox.critical(self, "依赖缺失", "rasterio 依赖不可用，无法执行 TIF 转 RGB。")
            return

        stretch_percent = self._read_stretch_percent()
        if stretch_percent is None:
            return

        inputs, output_dir = self._collect_inputs(is_batch)
        if not inputs or not output_dir:
            return

        self.progress_bar.setValue(0)
        self.log_edit.clear()
        self.log_edit.append("开始转换...")
        QApplication.processEvents()

        self.convert_thread = TifRgbConvertThread(
            inputs=inputs,
            output_dir=output_dir,
            output_format=self.output_format_combo.currentText(),
            red_band=self.red_band_spin.value(),
            green_band=self.green_band_spin.value(),
            blue_band=self.blue_band_spin.value(),
            stretch_percent=stretch_percent,
            preserve_georef=self.preserve_georef_checkbox.isChecked(),
        )
        self.convert_thread.progress_updated.connect(self.update_progress)
        self.convert_thread.finished_signal.connect(self.on_conversion_finished)
        self._update_buttons()
        self.convert_thread.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.log_edit.append(message)
        self.log_edit.ensureCursorVisible()
        QApplication.processEvents()

    def on_conversion_finished(self, message: str, msg_type: str):
        self.log_edit.append(f"--------------------\n{msg_type.upper()}: {message}")
        self.log_edit.ensureCursorVisible()
        self.convert_thread = None
        self._update_buttons()

        if msg_type == "info":
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "转换完成", message)
        else:
            QMessageBox.critical(self, "转换失败", message)

    def closeEvent(self, event):
        if self.convert_thread and self.convert_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "退出确认",
                "TIF 转 RGB 正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.convert_thread.requestInterruption()
                self.convert_thread.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
