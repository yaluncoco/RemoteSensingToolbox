import os
import math
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from Toolbox.file_io import atomic_output
from Toolbox.task_support import bind_task
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


RGB_BLOCK_SIZE = 512
STRETCH_SAMPLE_PIXELS = 1_048_576


class ConversionCancelled(Exception):
    pass


def _valid_pixels(band):
    data = np.ma.getdata(band)
    return ~np.ma.getmaskarray(band) & np.isfinite(data)


def _band_limits(band, stretch_percent):
    values = np.ma.getdata(band)[_valid_pixels(band)]
    if not values.size:
        return 0.0, 0.0
    if stretch_percent:
        return tuple(np.percentile(values, [stretch_percent, 100.0 - stretch_percent]))
    return float(values.min()), float(values.max())


def normalize_band_to_uint8(band: np.ndarray, stretch_percent: float, limits=None) -> np.ndarray:
    if not math.isfinite(stretch_percent) or not 0 <= stretch_percent < 50:
        raise ValueError("百分位拉伸必须在 0 到 50 之间。")
    low, high = limits if limits is not None else _band_limits(band, stretch_percent)
    result = np.zeros(band.shape, dtype=np.uint8)
    if high <= low:
        return result
    valid = _valid_pixels(band)
    scaled = (np.ma.getdata(band)[valid].astype(np.float64) - low) * (255.0 / (high - low))
    result[valid] = np.clip(scaled, 0, 255).astype(np.uint8)
    return result


def build_rgb_array(src, red_band: int, green_band: int, blue_band: int, stretch_percent: float) -> np.ndarray:
    if src.count == 1:
        gray = normalize_band_to_uint8(src.read(1, masked=True), stretch_percent)
        return np.dstack([gray, gray, gray])

    band_indexes = [red_band, green_band, blue_band]
    if any(index < 1 or index > src.count for index in band_indexes):
        raise ValueError(f"波段编号超出范围。当前 TIF 共有 {src.count} 个波段。")

    channels = [normalize_band_to_uint8(src.read(index, masked=True), stretch_percent) for index in band_indexes]
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
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(f"rasterio 依赖不可用：{RASTERIO_IMPORT_ERROR}")

    from rasterio.enums import ColorInterp, Resampling
    from rasterio.shutil import copy as copy_raster
    from rasterio.windows import Window

    output_format = output_format.upper()
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"不支持的输出格式：{output_format}")
    if not math.isfinite(stretch_percent) or not 0 <= stretch_percent < 50:
        raise ValueError("百分位拉伸必须在 0 到 50 之间。")
    if Path(input_path).resolve() == Path(output_path).resolve() or (
        Path(output_path).exists() and os.path.samefile(input_path, output_path)
    ):
        raise ValueError("输出路径不能与输入文件相同。")

    def check_cancel():
        if should_cancel and should_cancel():
            raise ConversionCancelled()

    def emit(value, message):
        if progress_callback:
            progress_callback(value, message)

    check_cancel()
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_PAM_ENABLED=False, GDAL_CACHEMAX=64_000_000):
        with rasterio.open(input_path) as src, atomic_output(output_path) as output_tmp:
            bands = [1, 1, 1] if src.count == 1 else [red_band, green_band, blue_band]
            if any(index < 1 or index > src.count for index in bands):
                raise ValueError(f"波段编号超出范围。当前 TIF 共有 {src.count} 个波段。")
            unique_bands = list(dict.fromkeys(bands))
            total = math.ceil(src.width / RGB_BLOCK_SIZE) * math.ceil(src.height / RGB_BLOCK_SIZE)

            def windows():
                for y in range(0, src.height, RGB_BLOCK_SIZE):
                    for x in range(0, src.width, RGB_BLOCK_SIZE):
                        yield Window(x, y, min(RGB_BLOCK_SIZE, src.width - x), min(RGB_BLOCK_SIZE, src.height - y))

            limits = {}
            if stretch_percent:
                # A bounded, deterministic overview gives one global stretch, without tile seams.
                scale = min(1.0, math.sqrt(STRETCH_SAMPLE_PIXELS / (src.width * src.height)))
                shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
                for index in unique_bands:
                    check_cancel()
                    sample = src.read(index, out_shape=shape, masked=True, resampling=Resampling.nearest)
                    limits[index] = _band_limits(sample, stretch_percent)
                emit(25, "已计算全图百分位拉伸范围（有效像素采样）")
            else:
                limits = {index: (math.inf, -math.inf) for index in unique_bands}
                last_percent = -1
                for number, window in enumerate(windows(), 1):
                    check_cancel()
                    for index in unique_bands:
                        band = src.read(index, window=window, masked=True)
                        if _valid_pixels(band).any():
                            low, high = _band_limits(band, 0)
                            limits[index] = (min(limits[index][0], low), max(limits[index][1], high))
                    percent = int(number / total * 25)
                    if percent != last_percent:
                        emit(percent, "正在分块统计有效像素的最小/最大值")
                        last_percent = percent

            meta = dict(driver="GTiff", height=src.height, width=src.width,
                        count=3, dtype="uint8", compress="lzw", tiled=True,
                        blockxsize=RGB_BLOCK_SIZE, blockysize=RGB_BLOCK_SIZE, BIGTIFF="IF_SAFER")
            if output_format == "RGB TIF" and preserve_georef:
                meta.update(crs=src.crs, transform=src.transform)

            with tempfile.TemporaryDirectory(prefix=".rgb-", dir=output_tmp.parent) as scratch:
                raster_path = output_tmp if output_format == "RGB TIF" else Path(scratch) / "rgb.tif"
                with rasterio.open(raster_path, "w", **meta) as dst:
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                    last_percent = -1
                    for number, window in enumerate(windows(), 1):
                        check_cancel()
                        tile = src.read(bands, window=window, masked=True)
                        valid = np.all(_valid_pixels(tile), axis=0)
                        rgb = np.stack([normalize_band_to_uint8(tile[channel], stretch_percent, limits[index])
                                        for channel, index in enumerate(bands)])
                        rgb[:, ~valid] = 0
                        dst.write(rgb, window=window)
                        if output_format == "RGB TIF":
                            dst.write_mask(valid.astype(np.uint8) * 255, window=window)
                        percent = 25 + int(number / total * 65)
                        if percent != last_percent:
                            emit(percent, f"正在转换分块 {number}/{total}")
                            last_percent = percent
                check_cancel()
                if output_format != "RGB TIF":
                    emit(92, f"正在编码 {output_format} 文件")
                    options = {"QUALITY": 95} if output_format == "JPG" else {}
                    copy_raster(raster_path, output_tmp, driver="JPEG" if output_format == "JPG" else "PNG", **options)
                check_cancel()
    emit(100, "转换完成")


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
            outputs = [default_output_path(path, self.output_dir, self.output_format) for path in self.inputs]
            normalized = [os.path.normcase(str(Path(path).resolve())) for path in outputs]
            input_paths = {os.path.normcase(str(Path(path).resolve())) for path in self.inputs}
            if len(set(normalized)) != len(normalized) or input_paths.intersection(normalized):
                raise ValueError("批量文件的输出路径存在冲突，请使用独立输出目录并检查同名 TIF/TIFF。")
            converted = 0
            for index, input_path in enumerate(self.inputs, start=1):
                if self.isInterruptionRequested():
                    self.finished_signal.emit("操作已取消。", "warning")
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
                    should_cancel=self.isInterruptionRequested,
                    progress_callback=lambda percent, message, offset=index - 1: self.progress_updated.emit(
                        int((offset + percent / 100) / total * 100), message),
                )
                converted += 1
                self.progress_updated.emit(
                    int(index / total * 100),
                    f"已输出：{output_path}",
                )

            self.finished_signal.emit(f"TIF 转 RGB 完成，共输出 {converted} 个文件。", "info")
        except ConversionCancelled:
            self.finished_signal.emit("操作已取消，未完成的输出已清理。", "warning")
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

        self.cancel_btn = QPushButton("取消转换")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        main_layout.addWidget(self.cancel_btn, alignment=Qt.AlignRight)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.document().setMaximumBlockCount(1000)
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
        self.cancel_btn.setEnabled(self.convert_thread is not None and not self.convert_thread.isInterruptionRequested())

    def cancel_conversion(self):
        if self.convert_thread:
            self.convert_thread.requestInterruption()
            self.cancel_btn.setEnabled(False)
            self.log_edit.append("正在取消，当前文件将在安全结束后清理…")

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
        bind_task(self, "convert_thread", self.on_conversion_finished)
        self._update_buttons()
        self.convert_thread.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.log_edit.append(message)
        self.log_edit.ensureCursorVisible()

    def on_conversion_finished(self, message: str, msg_type: str):
        self.log_edit.append(f"--------------------\n{msg_type.upper()}: {message}")
        self.log_edit.ensureCursorVisible()
        self.convert_thread = None
        self._update_buttons()

        if msg_type == "info":
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "转换完成", message)
        elif msg_type == "warning":
            QMessageBox.information(self, "转换已取消", message)
        else:
            QMessageBox.critical(self, "转换失败", message)

    def closeEvent(self, event):
        event.accept()
