from Toolbox.task_support import bind_task
from Toolbox.file_io import atomic_output
import os
from contextlib import ExitStack
from dataclasses import dataclass
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
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


try:
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.windows import Window

    RASTERIO_AVAILABLE = True
    RASTERIO_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    rasterio = None
    RasterioIOError = Exception
    Window = object
    RASTERIO_AVAILABLE = False
    RASTERIO_IMPORT_ERROR = exc


IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
IMAGE_FILE_FILTER = "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
BLACK_PIXEL_MAX = 3
WHITE_PIXEL_MIN_UINT8 = 250
WHITE_PERCENT_OF_DTYPE_MAX = 0.98
EDGE_MODE_PAD = "pad"
EDGE_MODE_SKIP = "skip"


class CuttingCancelled(Exception):
    """Raised when the user cancels an active cutting task."""


@dataclass
class CutStats:
    total_tiles: int
    processed_tiles: int = 0
    saved_image_tiles: int = 0
    saved_label_tiles: int = 0
    skipped_small_tiles: int = 0
    skipped_invalid_tiles: int = 0
    skipped_background_tiles: int = 0
    skipped_band_tiles: int = 0
    padded_edge_tiles: int = 0


@dataclass(frozen=True)
class CutJob:
    image_path: str
    label_path: Optional[str]
    output_prefix: Optional[str] = None


def merge_cut_stats(target: CutStats, source: CutStats) -> None:
    target.total_tiles += source.total_tiles
    target.processed_tiles += source.processed_tiles
    target.saved_image_tiles += source.saved_image_tiles
    target.saved_label_tiles += source.saved_label_tiles
    target.skipped_small_tiles += source.skipped_small_tiles
    target.skipped_invalid_tiles += source.skipped_invalid_tiles
    target.skipped_background_tiles += source.skipped_background_tiles
    target.skipped_band_tiles += source.skipped_band_tiles
    target.padded_edge_tiles += source.padded_edge_tiles


def get_supported_image_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_label_lookup(folder_path: str) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    extension_priority = {ext: index for index, ext in enumerate(IMAGE_EXTENSIONS)}
    for path in get_supported_image_files(folder_path):
        current = lookup.get(path.stem)
        if current is None:
            lookup[path.stem] = path
            continue

        current_priority = extension_priority.get(current.suffix.lower(), len(IMAGE_EXTENSIONS))
        new_priority = extension_priority.get(path.suffix.lower(), len(IMAGE_EXTENSIONS))
        if new_priority < current_priority:
            lookup[path.stem] = path
    return lookup


def build_cut_jobs(
    image_path: str,
    label_path: Optional[str],
    batch_mode: bool,
    single_image_mode: bool,
) -> tuple[list[CutJob], list[str]]:
    if not batch_mode:
        return [CutJob(image_path=image_path, label_path=None if single_image_mode else label_path)], []

    image_files = get_supported_image_files(image_path)
    if single_image_mode:
        return [
            CutJob(image_path=str(image_file), label_path=None, output_prefix=image_file.stem)
            for image_file in image_files
        ], []

    label_lookup = build_label_lookup(label_path or "")
    jobs: list[CutJob] = []
    missing_labels: list[str] = []
    for image_file in image_files:
        matched_label = label_lookup.get(image_file.stem)
        if matched_label is None:
            missing_labels.append(image_file.name)
            continue
        jobs.append(
            CutJob(
                image_path=str(image_file),
                label_path=str(matched_label),
                output_prefix=image_file.stem,
            )
        )
    return jobs, missing_labels


def count_tiles(image_path: str, tile_size: int) -> int:
    with rasterio.open(image_path) as src:
        num_h = (src.height + tile_size - 1) // tile_size
        num_w = (src.width + tile_size - 1) // tile_size
        return num_h * num_w


def get_white_threshold(tile_array: np.ndarray) -> Optional[float]:
    """Return a conservative near-white threshold for no-data backgrounds."""
    dtype = tile_array.dtype
    if np.issubdtype(dtype, np.integer):
        dtype_max = np.iinfo(dtype).max
        if dtype_max <= 255:
            return WHITE_PIXEL_MIN_UINT8
        return dtype_max * WHITE_PERCENT_OF_DTYPE_MAX

    finite_values = tile_array[np.isfinite(tile_array)]
    if finite_values.size == 0:
        return None

    max_value = float(np.nanmax(finite_values))
    if max_value <= 1.5:
        return 0.98
    if max_value <= 255:
        return WHITE_PIXEL_MIN_UINT8
    return max_value * WHITE_PERCENT_OF_DTYPE_MAX


def get_display_bands(tile_array: np.ndarray) -> np.ndarray:
    """Use visual bands for black/white background checks."""
    if tile_array.shape[0] >= 3:
        return tile_array[:3, :, :]
    return tile_array


def get_alpha_band_indexes(colorinterp, band_count: int) -> list[int]:
    if not colorinterp:
        return []

    alpha_indexes = []
    for index, interp in enumerate(colorinterp[:band_count]):
        name = getattr(interp, "name", str(interp)).lower()
        if name.endswith("alpha") or name == "alpha":
            alpha_indexes.append(index)
    return alpha_indexes


def get_alpha_transparent_mask(raw: np.ndarray, colorinterp=None) -> Optional[np.ndarray]:
    alpha_indexes = get_alpha_band_indexes(colorinterp, raw.shape[0])
    if not alpha_indexes:
        return None

    alpha_bands = raw[alpha_indexes, :, :]
    return np.any(alpha_bands <= BLACK_PIXEL_MAX, axis=0)


def calculate_valid_pixel_ratio(tile_array: np.ndarray, nodata=None, colorinterp=None) -> float:
    """
    Calculate valid-pixel ratio from the original raster values.

    A pixel is valid when at least one band contains meaningful data and it is
    not an all-black/all-white no-data background pixel.
    """
    tile = np.ma.asarray(tile_array)
    raw = np.asarray(np.ma.getdata(tile))
    mask = np.ma.getmaskarray(tile)

    if raw.ndim != 3 or raw.shape[1] == 0 or raw.shape[2] == 0:
        return 0.0

    display_raw = get_display_bands(raw)

    if nodata is not None:
        valid_mask = np.any(raw != nodata, axis=0)
    else:
        valid_mask = np.any(raw != 0, axis=0)

    if mask.shape == raw.shape and mask.any():
        display_mask = get_display_bands(mask)
        valid_mask &= np.any(~display_mask, axis=0)

    if np.issubdtype(raw.dtype, np.floating):
        valid_mask &= np.any(np.isfinite(display_raw), axis=0)

    transparent_mask = get_alpha_transparent_mask(raw, colorinterp=colorinterp)
    if transparent_mask is not None:
        valid_mask &= ~transparent_mask

    black_mask = np.all(display_raw <= BLACK_PIXEL_MAX, axis=0)
    valid_mask &= ~black_mask

    white_threshold = get_white_threshold(display_raw)
    if white_threshold is not None:
        white_mask = np.all(display_raw >= white_threshold, axis=0)
        valid_mask &= ~white_mask

    total_pixels = valid_mask.shape[0] * valid_mask.shape[1]
    if total_pixels == 0:
        return 0.0
    return np.count_nonzero(valid_mask) / total_pixels


def to_uint8_rgb(tile_array: np.ndarray, nodata=None, colorinterp=None) -> Optional[np.ndarray]:
    """
    Convert a rasterio tile array (bands, height, width) to displayable RGB uint8.

    Non-uint8 rasters are converted with 2/98 percentile stretching, matching the
    behavior used by common remote-sensing previews.
    """
    tile = np.ma.asarray(tile_array)
    data = np.asarray(np.ma.filled(tile, 0))

    if data.ndim != 3:
        return None

    bands, _height, _width = data.shape
    if bands == 1:
        rgb_bands = np.repeat(data, 3, axis=0)
    elif bands >= 3:
        rgb_bands = data[:3, :, :]
    else:
        return None

    rgb = np.transpose(rgb_bands, (1, 2, 0))
    transparent_mask = get_alpha_transparent_mask(data, colorinterp=colorinterp)

    if rgb.dtype == np.uint8:
        if transparent_mask is not None and transparent_mask.any():
            rgb = rgb.copy()
            rgb[transparent_mask] = 0
        return rgb

    rgb_float = rgb.astype(np.float32, copy=False)
    output = np.zeros(rgb_float.shape, dtype=np.uint8)
    white_threshold = get_white_threshold(rgb_bands)

    for channel_index in range(rgb_float.shape[2]):
        band = rgb_float[:, :, channel_index]
        valid = np.isfinite(band) & (band > 0)
        if nodata is not None:
            valid &= band != nodata
        if white_threshold is not None:
            valid &= band < white_threshold

        valid_values = band[valid]
        if valid_values.size == 0:
            continue

        p2, p98 = np.percentile(valid_values, (2, 98))
        if p98 <= p2:
            continue

        stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
        stretched[~np.isfinite(stretched)] = 0
        output[:, :, channel_index] = stretched.astype(np.uint8)

    if transparent_mask is not None and transparent_mask.any():
        output[transparent_mask] = 0

    return output


def cut_image_tiles(
    image_path: str,
    label_path: Optional[str],
    output_dir: str,
    tile_size: int,
    valid_pixel_threshold: float,
    bg_thresh: float,
    edge_mode: str = EDGE_MODE_PAD,
    output_prefix: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> CutStats:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError(f"rasterio 依赖不可用：{RASTERIO_IMPORT_ERROR}")

    if edge_mode not in (EDGE_MODE_PAD, EDGE_MODE_SKIP):
        edge_mode = EDGE_MODE_PAD

    img_output_dir = os.path.join(output_dir, "images")
    os.makedirs(img_output_dir, exist_ok=True)

    label_output_dir = None
    if label_path:
        label_output_dir = os.path.join(output_dir, "labels")
        os.makedirs(label_output_dir, exist_ok=True)

    with rasterio.open(image_path) as src, ExitStack() as stack:
        width, height = src.width, src.height
        num_h = (height + tile_size - 1) // tile_size
        num_w = (width + tile_size - 1) // tile_size
        stats = CutStats(total_tiles=num_h * num_w)

        label_src = None
        if label_path:
            label_src = stack.enter_context(rasterio.open(label_path))
            if label_src.width != width or label_src.height != height:
                raise ValueError("错误：原始图像和标签图像尺寸不匹配。")

        def mark_processed():
            stats.processed_tiles += 1
            if progress_callback:
                progress_callback(stats.processed_tiles, stats.total_tiles)

        for i in range(num_h):
            for j in range(num_w):
                if should_cancel and should_cancel():
                    raise CuttingCancelled()

                x, y = j * tile_size, i * tile_size
                actual_width = min(tile_size, width - x)
                actual_height = min(tile_size, height - y)
                is_partial_edge = actual_width < tile_size or actual_height < tile_size

                if is_partial_edge and edge_mode == EDGE_MODE_SKIP:
                    stats.skipped_small_tiles += 1
                    mark_processed()
                    continue

                window_width = tile_size if edge_mode == EDGE_MODE_PAD else actual_width
                window_height = tile_size if edge_mode == EDGE_MODE_PAD else actual_height
                window = Window(x, y, window_width, window_height)

                if edge_mode == EDGE_MODE_PAD:
                    fill_value = src.nodata if src.nodata is not None else 0
                    tile_data = src.read(
                        window=window,
                        masked=True,
                        boundless=True,
                        fill_value=fill_value,
                    )
                    if is_partial_edge:
                        stats.padded_edge_tiles += 1
                else:
                    tile_data = src.read(window=window, masked=True)

                valid_ratio = calculate_valid_pixel_ratio(
                    tile_data,
                    nodata=src.nodata,
                    colorinterp=src.colorinterp,
                )
                if valid_ratio < valid_pixel_threshold:
                    stats.skipped_invalid_tiles += 1
                    mark_processed()
                    continue

                image_rgb = to_uint8_rgb(tile_data, nodata=src.nodata, colorinterp=src.colorinterp)
                if image_rgb is None:
                    print(f"Skipping tile {i}_{j} due to unexpected image band count: {src.count}")
                    stats.skipped_band_tiles += 1
                    mark_processed()
                    continue

                label_data = None
                if label_src is not None:
                    if edge_mode == EDGE_MODE_PAD:
                        label_data = label_src.read(
                            1,
                            window=window,
                            boundless=True,
                            fill_value=0,
                            masked=True,
                        )
                    else:
                        label_data = label_src.read(1, window=window, masked=True)
                    label_data = np.ma.filled(label_data, 0)
                    bg_ratio = np.count_nonzero(label_data == 0) / label_data.size
                    if bg_ratio >= bg_thresh:
                        stats.skipped_background_tiles += 1
                        mark_processed()
                        continue
                    if (not np.isfinite(label_data).all() or label_data.min() < 0 or
                            label_data.max() > 65535 or not np.equal(label_data, np.floor(label_data)).all()):
                        raise ValueError("PNG 标签切片需要 0 到 65535 的整数类别值。")
                    label_data = label_data.astype(np.uint8 if label_data.max() <= 255 else np.uint16)

                name_prefix = f"{output_prefix}_" if output_prefix else ""
                img_filename = f"{name_prefix}tile_{i}_{j}.jpg"
                with atomic_output(os.path.join(img_output_dir, img_filename)) as temporary:
                    Image.fromarray(image_rgb).save(temporary, quality=95)
                stats.saved_image_tiles += 1

                if label_data is not None and label_output_dir:
                    label_filename = f"{name_prefix}tile_{i}_{j}.png"
                    with atomic_output(os.path.join(label_output_dir, label_filename)) as temporary:
                        Image.fromarray(label_data).save(temporary)
                    stats.saved_label_tiles += 1

                mark_processed()

    return stats


class CutterThread(QThread):
    """Worker thread for image cutting to prevent GUI freezing."""

    progress_updated = pyqtSignal(int)
    finished_signal = pyqtSignal(str, str)

    def __init__(
        self,
        jobs: list[CutJob],
        output_dir: str,
        tile_size: int,
        valid_pixel_threshold: float,
        bg_thresh: float,
        edge_mode: str,
    ):
        super().__init__()
        self.jobs = jobs
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.valid_pixel_threshold = valid_pixel_threshold
        self.bg_thresh = bg_thresh
        self.edge_mode = edge_mode

    def _emit_progress(self, processed_tiles: int, total_tiles: int):
        progress = int((processed_tiles / total_tiles) * 100) if total_tiles > 0 else 0
        self.progress_updated.emit(progress)

    def run(self):
        try:
            total_tiles = sum(count_tiles(job.image_path, self.tile_size) for job in self.jobs)
            processed_offset = 0
            merged_stats = CutStats(total_tiles=0)

            for job in self.jobs:
                if self.isInterruptionRequested():
                    raise CuttingCancelled()

                def emit_job_progress(processed_tiles: int, _job_total: int):
                    self._emit_progress(processed_offset + processed_tiles, total_tiles)

                stats = cut_image_tiles(
                    job.image_path,
                    job.label_path,
                    self.output_dir,
                    self.tile_size,
                    self.valid_pixel_threshold,
                    self.bg_thresh,
                    edge_mode=self.edge_mode,
                    output_prefix=job.output_prefix,
                    progress_callback=emit_job_progress,
                    should_cancel=self.isInterruptionRequested,
                )
                processed_offset += stats.total_tiles
                merge_cut_stats(merged_stats, stats)

            has_labels = any(job.label_path for job in self.jobs)
            label_part = f"，标签 {merged_stats.saved_label_tiles} 张" if has_labels else ""
            file_part = f"，处理文件 {len(self.jobs)} 个" if len(self.jobs) > 1 else ""
            message = (
                f"图像切片完成！输出图像 {merged_stats.saved_image_tiles} 张{label_part}{file_part}。"
                f"\n过滤无效/空白切片 {merged_stats.skipped_invalid_tiles} 张，"
                f"过滤标签背景切片 {merged_stats.skipped_background_tiles} 张。"
                f"\n补齐边缘切片 {merged_stats.padded_edge_tiles} 张，"
                f"跳过尺寸不足边缘 {merged_stats.skipped_small_tiles} 张。"
            )
            self.finished_signal.emit(message, "info")

        except CuttingCancelled:
            self.finished_signal.emit("操作已取消。", "info")
        except FileNotFoundError as exc:
            self.finished_signal.emit(f"文件未找到错误: {exc}", "error")
        except RasterioIOError as exc:
            self.finished_signal.emit(f"Rasterio 读取错误: {exc}", "error")
        except ValueError as exc:
            self.finished_signal.emit(str(exc), "error")
        except Exception as exc:
            self.finished_signal.emit(f"切片过程中发生未知错误: {exc}", "error")


class ImageCutterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像切片工具")
        self.setGeometry(0, 0, 720, 600)
        self._center_window()
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))

        self.image_path: Optional[str] = None
        self.label_path: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.cutter_thread: Optional[CutterThread] = None

        self._init_ui()
        self._update_cut_button_state()

        if not RASTERIO_AVAILABLE:
            QMessageBox.critical(
                self,
                "依赖缺失",
                "图像切片功能需要 rasterio 及其 GDAL 运行库。\n"
                f"当前环境导入失败：{RASTERIO_IMPORT_ERROR}",
            )
            for child in self.findChildren(QWidget):
                child.setEnabled(False)

    def _center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        paths_group = QGroupBox("输入文件/目录选择")
        paths_layout = QFormLayout(paths_group)
        paths_layout.setSpacing(10)

        self.batch_mode_checkbox = QCheckBox("批量处理文件夹")
        self.batch_mode_checkbox.setToolTip("开启后选择图像文件夹和标签文件夹，按相同文件名自动一一配对切片。")
        self.batch_mode_checkbox.stateChanged.connect(self.on_batch_mode_changed)
        paths_layout.addRow(self.batch_mode_checkbox)

        self.image_path_btn = QPushButton("选择原始图像")
        self.image_path_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.image_path_btn.setToolTip("选择原始遥感影像文件。")
        self.image_path_btn.clicked.connect(self.select_image)
        paths_layout.addRow(self.image_path_btn)

        self.single_image_checkbox = QCheckBox("仅切原始图像（不选择标签）")
        self.single_image_checkbox.setToolTip("开启后只切原始图像，不读取标签，也不生成 labels 目录。")
        self.single_image_checkbox.stateChanged.connect(self.on_single_image_mode_changed)
        paths_layout.addRow(self.single_image_checkbox)

        self.label_path_btn = QPushButton("选择标签图像")
        self.label_path_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.label_path_btn.setToolTip("选择单波段标签影像文件，像素值代表类别。")
        self.label_path_btn.clicked.connect(self.select_label)
        paths_layout.addRow(self.label_path_btn)

        self.output_dir_btn = QPushButton("选择输出目录")
        self.output_dir_btn.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.output_dir_btn.setToolTip("选择切片后图像和标签的存放目录。")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        paths_layout.addRow(self.output_dir_btn)
        main_layout.addWidget(paths_group)

        params_group = QGroupBox("切片参数设置")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        self.tile_size_edit = QLineEdit("512")
        self.tile_size_edit.setToolTip("切片后每个小图像的边长（像素）。")
        params_layout.addRow(QLabel("切片大小 (像素):"), self.tile_size_edit)

        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItem("补齐边缘，固定输出尺寸", EDGE_MODE_PAD)
        self.edge_mode_combo.addItem("跳过边缘不足尺寸", EDGE_MODE_SKIP)
        self.edge_mode_combo.setToolTip(
            "默认补齐边缘，保证输出图像和标签都是切片大小；补齐区域标签值为 0。"
        )
        params_layout.addRow(QLabel("边缘处理:"), self.edge_mode_combo)

        self.valid_threshold_edit = QLineEdit("0.80")
        self.valid_threshold_edit.setToolTip(
            "有效像素占比阈值。低于此值的切片会被跳过；正射影像建议 0.8。"
        )
        params_layout.addRow(QLabel("有效像素阈值 (0-1):"), self.valid_threshold_edit)

        self.bg_threshold_label = QLabel("标签背景占比阈值 (0-1):")
        self.bg_threshold_edit = QLineEdit("0.90")
        self.bg_threshold_edit.setToolTip("标签中背景类别占比大于此值则跳过。")
        params_layout.addRow(self.bg_threshold_label, self.bg_threshold_edit)
        main_layout.addWidget(params_group)

        self.cut_btn = QPushButton("开始切片")
        self.cut_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.cut_btn.setToolTip("根据以上设置开始执行切片操作。")
        self.cut_btn.clicked.connect(self.start_cutting_process)
        main_layout.addWidget(self.cut_btn, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("显示当前切片进度。")
        main_layout.addWidget(self.progress_bar)

    def _update_button_text(self, button: QPushButton, base_text: str, path: Optional[str]):
        if path:
            button.setText(f"{base_text}: {os.path.basename(path)}")
            button.setToolTip(path)
        else:
            button.setText(base_text)
            button.setToolTip("")

    def _image_button_text(self) -> str:
        return "原始图像文件夹" if self.batch_mode_checkbox.isChecked() else "原始图像"

    def _label_button_text(self) -> str:
        return "标签文件夹" if self.batch_mode_checkbox.isChecked() else "标签图像"

    def _update_cut_button_state(self):
        single_image_mode = self.single_image_checkbox.isChecked()
        batch_mode = self.batch_mode_checkbox.isChecked()
        image_ready = bool(self.image_path)
        label_ready = bool(self.label_path)
        if image_ready:
            image_path = Path(self.image_path)
            image_ready = image_path.is_dir() if batch_mode else image_path.is_file()
        if label_ready:
            label_path = Path(self.label_path)
            label_ready = label_path.is_dir() if batch_mode else label_path.is_file()
        enabled = bool(
            RASTERIO_AVAILABLE
            and image_ready
            and self.output_dir
            and (single_image_mode or label_ready)
        )
        self.cut_btn.setEnabled(enabled)

    def on_batch_mode_changed(self, _state=None):
        batch_mode = self.batch_mode_checkbox.isChecked()
        self.image_path_btn.setText("选择原始图像文件夹" if batch_mode else "选择原始图像")
        self.image_path_btn.setToolTip(
            "选择包含待切片遥感影像的文件夹。" if batch_mode else "选择原始遥感影像文件。"
        )
        if not self.single_image_checkbox.isChecked():
            self.label_path_btn.setText("选择标签文件夹" if batch_mode else "选择标签图像")
            self.label_path_btn.setToolTip(
                "选择标签文件夹；默认按与原图相同的文件名配对。"
                if batch_mode
                else "选择单波段标签影像文件，像素值代表类别。"
            )
        self._update_cut_button_state()

    def on_single_image_mode_changed(self, _state=None):
        single_image_mode = self.single_image_checkbox.isChecked()
        self.label_path_btn.setEnabled(not single_image_mode)
        self.bg_threshold_label.setEnabled(not single_image_mode)
        self.bg_threshold_edit.setEnabled(not single_image_mode)
        if single_image_mode:
            self.label_path_btn.setText("标签（当前不使用）")
            self.label_path_btn.setToolTip("仅切原始图像时不会读取标签。")
            self.output_dir_btn.setToolTip("选择切片后图像的存放目录。")
        else:
            self._update_button_text(self.label_path_btn, self._label_button_text(), self.label_path)
            self.label_path_btn.setToolTip(
                "选择标签文件夹；默认按与原图相同的文件名配对。"
                if self.batch_mode_checkbox.isChecked()
                else "选择单波段标签影像文件，像素值代表类别。"
            )
            self.output_dir_btn.setToolTip("选择切片后图像和标签的存放目录。")
        self._update_cut_button_state()

    def select_image(self):
        if self.batch_mode_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择原始图像文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择原始图像", "", IMAGE_FILE_FILTER)
        if path:
            self.image_path = path
            self._update_button_text(self.image_path_btn, self._image_button_text(), path)
            self._update_cut_button_state()

    def select_label(self):
        if self.batch_mode_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择标签图像", "", IMAGE_FILE_FILTER)
        if path:
            self.label_path = path
            self._update_button_text(self.label_path_btn, self._label_button_text(), path)
            self._update_cut_button_state()

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir = path
            self._update_button_text(self.output_dir_btn, "输出目录", path)
            self._update_cut_button_state()

    def start_cutting_process(self):
        if not RASTERIO_AVAILABLE:
            QMessageBox.critical(
                self,
                "依赖缺失",
                "图像切片功能需要 rasterio 及其 GDAL 运行库，请修复环境后重试。",
            )
            return

        single_image_mode = self.single_image_checkbox.isChecked()
        batch_mode = self.batch_mode_checkbox.isChecked()
        if not self.image_path or not self.output_dir or (not single_image_mode and not self.label_path):
            if batch_mode:
                required = "原始图像文件夹和输出目录" if single_image_mode else "原始图像文件夹、标签文件夹和输出目录"
            else:
                required = "原始图像和输出目录" if single_image_mode else "原始图像、标签图像和输出目录"
            QMessageBox.warning(self, "输入错误", f"请确保已选择{required}。")
            return

        image_path_obj = Path(self.image_path)
        if batch_mode and not image_path_obj.is_dir():
            QMessageBox.warning(self, "输入错误", "批量模式下请选择原始图像文件夹。")
            return
        if not batch_mode and not image_path_obj.is_file():
            QMessageBox.warning(self, "输入错误", "单张模式下请选择原始图像文件。")
            return

        if not single_image_mode and self.label_path:
            label_path_obj = Path(self.label_path)
            if batch_mode and not label_path_obj.is_dir():
                QMessageBox.warning(self, "输入错误", "批量模式下请选择标签文件夹。")
                return
            if not batch_mode and not label_path_obj.is_file():
                QMessageBox.warning(self, "输入错误", "单张模式下请选择标签图像文件。")
                return

        try:
            tile_size = int(self.tile_size_edit.text())
            if tile_size <= 0:
                raise ValueError("切片大小必须为正整数。")
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", f"无效的切片大小: {exc}")
            return

        try:
            valid_pixel_threshold = float(self.valid_threshold_edit.text())
            if not (0.0 <= valid_pixel_threshold <= 1.0):
                raise ValueError("有效像素阈值必须在 0.0 和 1.0 之间。")
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", f"无效的有效像素阈值: {exc}")
            return

        bg_thresh = 1.0
        if not single_image_mode:
            try:
                bg_thresh = float(self.bg_threshold_edit.text())
                if not (0.0 <= bg_thresh <= 1.0):
                    raise ValueError("标签背景占比阈值必须在 0.0 和 1.0 之间。")
            except ValueError as exc:
                QMessageBox.warning(self, "参数错误", f"无效的标签背景占比阈值: {exc}")
                return

        label_path = None if single_image_mode else self.label_path
        jobs, missing_labels = build_cut_jobs(
            self.image_path,
            label_path,
            batch_mode=batch_mode,
            single_image_mode=single_image_mode,
        )
        if not jobs:
            QMessageBox.warning(self, "输入错误", "没有找到可处理的图像文件。")
            return
        if missing_labels:
            preview = "\n".join(missing_labels[:10])
            suffix = "\n..." if len(missing_labels) > 10 else ""
            QMessageBox.warning(
                self,
                "标签缺失",
                f"以下图像没有找到同名标签文件，已停止批量切片：\n{preview}{suffix}",
            )
            return

        edge_mode = self.edge_mode_combo.currentData() or EDGE_MODE_PAD

        self.cut_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.cutter_thread = CutterThread(
            jobs,
            self.output_dir,
            tile_size,
            valid_pixel_threshold,
            bg_thresh,
            edge_mode,
        )
        self.cutter_thread.progress_updated.connect(self.progress_bar.setValue)
        bind_task(self, "cutter_thread", self.on_cutting_finished)
        self.cutter_thread.start()

    def on_cutting_finished(self, message: str, msg_type: str):
        self._update_cut_button_state()
        if msg_type == "info":
            QMessageBox.information(self, "操作完成", message)
            self.progress_bar.setValue(100)
        elif msg_type == "error":
            QMessageBox.critical(self, "操作失败", message)

    def closeEvent(self, event):
        # TaskBinding defers close while the worker is still running.
        event.accept()
