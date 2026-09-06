from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import fiona
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform
    from rasterio.warp import transform_bounds, transform_geom

    GIS_AVAILABLE = True
    GIS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    fiona = None
    rasterio = None
    CRS = None
    rasterize = None
    Window = None
    window_bounds = None
    window_transform = None
    transform_bounds = None
    transform_geom = None
    GIS_AVAILABLE = False
    GIS_IMPORT_ERROR = exc


DEFAULT_LABELS = """# 推荐写法：像素值=类别
# 0 固定为背景，不要填写
# 示例：
1=房子
2=树木
"""
RASTERIZE_BLOCK_SIZE = 2048
COMMON_LABEL_FIELDS = (
    "label",
    "class",
    "category",
    "type",
    "name",
    "cultivo",
    "crop",
    "类别",
    "类型",
)


class RasterizeCancelled(Exception):
    """Raised when the user cancels an active rasterization task."""


@dataclass(frozen=True)
class ShpRasterizeResult:
    output_tif: str
    label_to_id: dict[str, int]
    burned_features: int
    skipped_unlabeled_features: int
    skipped_unknown_features: int
    coordinate_note: str = ""


@dataclass(frozen=True)
class ShpRasterizeJob:
    shp_path: str
    reference_tif_path: str
    output_tif_path: str


@dataclass(frozen=True)
class RasterShape:
    geometry: dict
    class_id: int
    bounds: tuple[float, float, float, float]


def parse_class_mapping(text: str) -> dict[str, int]:
    label_to_id: dict[str, int] = {}
    used_ids: set[int] = {0}
    next_auto_id = 1

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        separator = "=" if "=" in line else ":" if ":" in line else None
        if separator:
            left, right = [part.strip() for part in line.split(separator, 1)]
            if not left or not right:
                raise ValueError(f"类别映射格式错误：{raw_line}")

            if left.isdigit() and not right.isdigit():
                class_id = int(left)
                label = right
            elif right.isdigit():
                label = left
                class_id = int(right)
            else:
                raise ValueError(f"类别映射需要包含一个整数像素值：{raw_line}")
        else:
            label = line
            while next_auto_id in used_ids:
                next_auto_id += 1
            class_id = next_auto_id

        if class_id <= 0:
            raise ValueError("像素值 0 固定保留为背景，类别像素值必须大于 0。")
        if label in label_to_id:
            raise ValueError(f"类别重复：{label}")
        if class_id in used_ids:
            raise ValueError(f"像素值重复：{class_id}")

        label_to_id[label] = class_id
        used_ids.add(class_id)

    if not label_to_id:
        raise ValueError("类别映射不能为空。")
    return label_to_id


def normalize_class_mapping(class_mapping: dict[str, int] | list[str]) -> dict[str, int]:
    if isinstance(class_mapping, dict):
        return dict(class_mapping)
    return {label: index + 1 for index, label in enumerate(class_mapping)}


def resolve_label_field(properties_schema: dict, requested_field: str) -> Optional[str]:
    if requested_field in properties_schema:
        return requested_field

    requested_lower = requested_field.lower()
    for field_name in properties_schema:
        if field_name.lower() == requested_lower:
            return field_name
    return None


def infer_label_field(properties_schema: dict, requested_field: str) -> Optional[str]:
    resolved = resolve_label_field(properties_schema, requested_field)
    if resolved is not None:
        return resolved

    for candidate in COMMON_LABEL_FIELDS:
        resolved = resolve_label_field(properties_schema, candidate)
        if resolved is not None:
            return resolved
    return None


def dtype_for_max_class_id(max_class_id: int) -> str:
    if max_class_id <= 255:
        return "uint8"
    if max_class_id <= 65535:
        return "uint16"
    return "uint32"


def get_supported_tif_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
    )


def get_supported_shp_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".shp")


def build_tif_lookup(folder_path: str) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    priority = {".tif": 0, ".tiff": 1}
    for path in get_supported_tif_files(folder_path):
        current = lookup.get(path.stem)
        if current is None or priority.get(path.suffix.lower(), 99) < priority.get(current.suffix.lower(), 99):
            lookup[path.stem] = path
    return lookup


def build_batch_jobs(shp_folder: str, reference_tif_folder: str, output_folder: str) -> tuple[list[ShpRasterizeJob], list[str]]:
    tif_lookup = build_tif_lookup(reference_tif_folder)
    jobs: list[ShpRasterizeJob] = []
    missing: list[str] = []
    for shp_path in get_supported_shp_files(shp_folder):
        reference_tif = tif_lookup.get(shp_path.stem)
        if reference_tif is None:
            missing.append(shp_path.name)
            continue
        jobs.append(
            ShpRasterizeJob(
                shp_path=str(shp_path),
                reference_tif_path=str(reference_tif),
                output_tif_path=str(Path(output_folder) / f"{shp_path.stem}_label_id.tif"),
            )
        )
    return jobs, missing


def crs_equal(left, right) -> bool:
    if not left or not right:
        return True
    try:
        return CRS.from_user_input(left) == CRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)


def maybe_reproject_geometry(geometry: dict, source_crs, target_crs) -> dict:
    if not source_crs or not target_crs or crs_equal(source_crs, target_crs):
        return geometry
    return transform_geom(source_crs, target_crs, geometry)


def bounds_to_tuple(bounds) -> Optional[tuple[float, float, float, float]]:
    if bounds is None:
        return None
    try:
        left, bottom, right, top = bounds
        return float(left), float(bottom), float(right), float(top)
    except Exception:
        return None


def bounds_area(bounds: tuple[float, float, float, float]) -> float:
    left, bottom, right, top = bounds
    return max(0.0, right - left) * max(0.0, top - bottom)


def bounds_intersection_area(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
) -> float:
    left = max(left_bounds[0], right_bounds[0])
    bottom = max(left_bounds[1], right_bounds[1])
    right = min(left_bounds[2], right_bounds[2])
    top = min(left_bounds[3], right_bounds[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def bounds_overlap_ratio(
    left_bounds: Optional[tuple[float, float, float, float]],
    right_bounds: Optional[tuple[float, float, float, float]],
) -> float:
    if left_bounds is None or right_bounds is None:
        return 0.0
    smaller_area = min(bounds_area(left_bounds), bounds_area(right_bounds))
    if smaller_area <= 0:
        return 0.0
    return bounds_intersection_area(left_bounds, right_bounds) / smaller_area


def format_bounds(bounds: Optional[tuple[float, float, float, float]]) -> str:
    if bounds is None:
        return "未知"
    return f"({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f})"


def crs_name(crs) -> str:
    if not crs:
        return "未知 CRS"
    try:
        parsed = CRS.from_user_input(crs)
        epsg = parsed.to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
        return parsed.name or str(crs)
    except Exception:
        return str(crs)


def decide_coordinate_mode(
    source_crs,
    target_crs,
    source_bounds: Optional[tuple[float, float, float, float]],
    target_bounds: Optional[tuple[float, float, float, float]],
) -> tuple[str, str]:
    """
    Return either "direct" or "reproject".

    Some orthomosaic TIF files carry an EngineeringCRS that describes the same
    projected plane as the SHP, but PROJ cannot build a datum operation. In that
    case, direct rasterization is correct if the numeric bounds already overlap.
    """
    if not source_crs:
        return "direct", "SHP 没有 CRS 信息，按参考影像坐标直接栅格化。"
    if not target_crs:
        return "direct", "参考 TIF 没有 CRS 信息，按 SHP 原始坐标直接栅格化。"
    if crs_equal(source_crs, target_crs):
        return "direct", "SHP 与参考 TIF 坐标系一致，直接栅格化。"

    direct_overlap = bounds_overlap_ratio(source_bounds, target_bounds)

    try:
        transformed_bounds = bounds_to_tuple(
            transform_bounds(source_crs, target_crs, *source_bounds, densify_pts=21)
            if source_bounds is not None
            else None
        )
    except Exception as exc:
        if direct_overlap > 0:
            return (
                "direct",
                "SHP 与参考 TIF 的 CRS 描述不同，且 PROJ 无法自动建立转换；"
                "但两者坐标范围在当前坐标值下有重叠，已按参考影像坐标直接栅格化。"
                f"转换失败原因：{exc}",
            )
        raise ValueError(
            "SHP 坐标系与参考 TIF 坐标系不同，且无法自动重投影；两者当前坐标范围也不重叠，"
            "为避免输出全 0 标签，已停止处理。\n"
            f"SHP CRS：{crs_name(source_crs)}\n"
            f"参考 TIF CRS：{crs_name(target_crs)}\n"
            f"SHP 范围：{format_bounds(source_bounds)}\n"
            f"参考 TIF 范围：{format_bounds(target_bounds)}\n"
            "请检查 SHP 的 .prj 或参考 TIF 的投影信息，必要时先在 GIS 中将二者重投影到同一坐标系。"
        ) from exc

    transformed_overlap = bounds_overlap_ratio(transformed_bounds, target_bounds)
    if transformed_overlap > 0:
        return (
            "reproject",
            f"SHP CRS 与参考 TIF CRS 不同，已自动重投影到参考 TIF 坐标系。"
            f"重投影后范围重叠比例约 {transformed_overlap:.3f}。",
        )

    if direct_overlap > 0:
        return (
            "direct",
            "SHP CRS 与参考 TIF CRS 描述不同，但重投影后范围不重叠；"
            "原始坐标范围与参考 TIF 有重叠，判断为 CRS 元数据不一致，已直接按参考影像坐标栅格化。",
        )

    raise ValueError(
        "SHP 和参考 TIF 坐标范围不重叠，转换后也不重叠，输出会全为背景，已停止处理。\n"
        f"SHP CRS：{crs_name(source_crs)}\n"
        f"参考 TIF CRS：{crs_name(target_crs)}\n"
        f"SHP 原始范围：{format_bounds(source_bounds)}\n"
        f"SHP 重投影后范围：{format_bounds(transformed_bounds)}\n"
        f"参考 TIF 范围：{format_bounds(target_bounds)}"
    )


def geometry_bounds(geometry: dict) -> Optional[tuple[float, float, float, float]]:
    coordinates = geometry.get("coordinates") if geometry else None
    if coordinates is None:
        return None

    xs: list[float] = []
    ys: list[float] = []

    def collect(values) -> None:
        if not values:
            return
        first = values[0]
        if isinstance(first, (int, float)) and len(values) >= 2:
            xs.append(float(values[0]))
            ys.append(float(values[1]))
            return
        for item in values:
            collect(item)

    collect(coordinates)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bounds_intersect(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
) -> bool:
    return not (
        left_bounds[2] <= right_bounds[0]
        or left_bounds[0] >= right_bounds[2]
        or left_bounds[3] <= right_bounds[1]
        or left_bounds[1] >= right_bounds[3]
    )


def rasterize_to_tif_by_windows(
    shapes: list[RasterShape],
    output_path: Path,
    meta: dict,
    height: int,
    width: int,
    transform,
    dtype: str,
    all_touched: bool,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=0,
        compress="lzw",
        BIGTIFF="IF_SAFER",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    x_offsets = list(range(0, width, RASTERIZE_BLOCK_SIZE))
    y_offsets = list(range(0, height, RASTERIZE_BLOCK_SIZE))
    total_windows = len(x_offsets) * len(y_offsets)
    processed_windows = 0
    last_percent = 54
    shape_bounds = np.asarray([item.bounds for item in shapes], dtype=np.float64)
    shape_geometries = [item.geometry for item in shapes]
    shape_class_ids = [item.class_id for item in shapes]

    with rasterio.open(output_path, "w", **meta) as dst:
        for y in y_offsets:
            window_height = min(RASTERIZE_BLOCK_SIZE, height - y)
            for x in x_offsets:
                if should_cancel and should_cancel():
                    raise RasterizeCancelled()

                window_width = min(RASTERIZE_BLOCK_SIZE, width - x)
                window = Window(x, y, window_width, window_height)
                current_bounds = window_bounds(window, transform)
                intersects = ~(
                    (shape_bounds[:, 2] <= current_bounds[0])
                    | (shape_bounds[:, 0] >= current_bounds[2])
                    | (shape_bounds[:, 3] <= current_bounds[1])
                    | (shape_bounds[:, 1] >= current_bounds[3])
                )
                indexes = np.flatnonzero(intersects)
                window_shapes = [(shape_geometries[index], shape_class_ids[index]) for index in indexes]
                if window_shapes:
                    tile = rasterize(
                        shapes=window_shapes,
                        out_shape=(window_height, window_width),
                        transform=window_transform(window, transform),
                        fill=0,
                        dtype=dtype,
                        all_touched=all_touched,
                    )
                else:
                    tile = np.zeros((window_height, window_width), dtype=dtype)

                dst.write(tile, 1, window=window)
                processed_windows += 1
                percent = 55 + int(processed_windows / total_windows * 25)
                if progress_callback and (percent != last_percent or processed_windows == total_windows):
                    progress_callback(percent, f"分块栅格化 {processed_windows}/{total_windows}")
                    last_percent = percent


def rasterize_shp_labels(
    shp_path: str,
    reference_tif_path: str,
    output_tif_path: str,
    label_field: str,
    class_mapping: dict[str, int] | list[str],
    all_touched: bool = False,
    encoding: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> ShpRasterizeResult:
    if not GIS_AVAILABLE:
        raise RuntimeError(f"缺少 fiona/rasterio 依赖：{GIS_IMPORT_ERROR}")

    def emit(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    def check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise RasterizeCancelled()

    label_to_id = normalize_class_mapping(class_mapping)
    if not label_to_id:
        raise ValueError("类别映射不能为空。")
    if any(class_id <= 0 for class_id in label_to_id.values()):
        raise ValueError("像素值 0 固定保留为背景，类别像素值必须大于 0。")
    if len(set(label_to_id.values())) != len(label_to_id):
        raise ValueError("类别像素值不能重复。")

    dtype = dtype_for_max_class_id(max(label_to_id.values()))
    emit(5, f"类别映射：0=背景，{', '.join(f'{value}={key}' for key, value in label_to_id.items())}")

    with rasterio.open(reference_tif_path) as ref:
        meta = ref.meta.copy()
        height, width = ref.height, ref.width
        transform = ref.transform
        reference_crs = ref.crs
        reference_bounds = bounds_to_tuple(ref.bounds)
    emit(15, f"已读取参考 TIF：{width} x {height}")

    shapes: list[RasterShape] = []
    skipped_unlabeled = 0
    skipped_unknown = 0
    coordinate_note = ""
    open_kwargs = {"encoding": encoding} if encoding else {}
    with fiona.open(shp_path, "r", **open_kwargs) as shp:
        properties_schema = shp.schema.get("properties", {})
        actual_label_field = infer_label_field(properties_schema, label_field)
        if actual_label_field is None:
            available = ", ".join(properties_schema.keys()) or "无"
            raise ValueError(f"SHP 中没有类别字段 '{label_field}'。可用字段：{available}")
        if actual_label_field != label_field:
            emit(18, f"未找到字段 '{label_field}'，已自动使用类别字段 '{actual_label_field}'。")

        source_crs = shp.crs_wkt or shp.crs
        source_bounds = bounds_to_tuple(shp.bounds)
        coordinate_mode, coordinate_note = decide_coordinate_mode(
            source_crs,
            reference_crs,
            source_bounds,
            reference_bounds,
        )
        emit(20, coordinate_note)

        feature_count = len(shp)
        last_read_percent = 24
        for index, feature in enumerate(shp, start=1):
            check_cancelled()
            value = feature["properties"].get(actual_label_field)
            if value is None or str(value).strip() == "":
                skipped_unlabeled += 1
                continue

            label = str(value).strip()
            class_id = label_to_id.get(label)
            if class_id is None:
                skipped_unknown += 1
                continue

            geometry = feature.get("geometry")
            if not geometry:
                skipped_unlabeled += 1
                continue

            if coordinate_mode == "reproject":
                try:
                    geometry = transform_geom(source_crs, reference_crs, geometry)
                except Exception as exc:
                    raise ValueError(
                        f"第 {index} 个 SHP 要素重投影失败：{exc}\n"
                        "请检查 SHP 的 .prj 或先在 GIS 中把 SHP 与参考 TIF 转为同一坐标系。"
                    ) from exc
            geometry_bounds_value = geometry_bounds(geometry)
            if geometry_bounds_value is None:
                skipped_unlabeled += 1
                continue

            shapes.append(RasterShape(geometry=geometry, class_id=class_id, bounds=geometry_bounds_value))
            if feature_count:
                read_percent = 25 + int(index / feature_count * 30)
                if read_percent != last_read_percent or index == feature_count:
                    emit(read_percent, f"已读取有效要素 {len(shapes)} 个")
                    last_read_percent = read_percent

    if not shapes:
        raise ValueError("没有找到能匹配类别列表的有效 SHP 要素。")

    check_cancelled()
    output_path = Path(output_tif_path)
    emit(55, "开始按类别像素值分块栅格化...")
    rasterize_to_tif_by_windows(
        shapes=shapes,
        output_path=output_path,
        meta=meta,
        height=height,
        width=width,
        transform=transform,
        dtype=dtype,
        all_touched=all_touched,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
    )

    emit(100, "SHP 标签栅格化完成。")
    return ShpRasterizeResult(
        output_tif=str(output_path),
        label_to_id=label_to_id,
        burned_features=len(shapes),
        skipped_unlabeled_features=skipped_unlabeled,
        skipped_unknown_features=skipped_unknown,
        coordinate_note=coordinate_note,
    )


class ShpLabelRasterizeThread(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str, str)

    def __init__(
        self,
        jobs: list[ShpRasterizeJob],
        label_field: str,
        class_mapping: dict[str, int],
        all_touched: bool,
        encoding: Optional[str],
    ):
        super().__init__()
        self.jobs = jobs
        self.label_field = label_field
        self.class_mapping = class_mapping
        self.all_touched = all_touched
        self.encoding = encoding

    def run(self) -> None:
        try:
            results: list[ShpRasterizeResult] = []
            total = len(self.jobs)

            for job_index, job in enumerate(self.jobs):
                if self.isInterruptionRequested():
                    raise RasterizeCancelled()

                def emit_job_progress(value: int, message: str):
                    progress = int((job_index / total) * 100 + value / total) if total else value
                    self.progress_updated.emit(progress, f"[{job_index + 1}/{total}] {message}")

                result = rasterize_shp_labels(
                    job.shp_path,
                    job.reference_tif_path,
                    job.output_tif_path,
                    self.label_field,
                    self.class_mapping,
                    all_touched=self.all_touched,
                    encoding=self.encoding,
                    progress_callback=emit_job_progress,
                    should_cancel=self.isInterruptionRequested,
                )
                results.append(result)

            first_result = results[0]
            mapping = "\n".join(
                f"{class_id}: {label}"
                for label, class_id in sorted(first_result.label_to_id.items(), key=lambda item: item[1])
            )
            output_part = (
                first_result.output_tif
                if len(results) == 1
                else f"批量输出 {len(results)} 个 TIF，保存到：{Path(results[0].output_tif).parent}"
            )
            message = (
                f"SHP 标签转 TIF 完成。\n"
                f"输出：{output_part}\n\n"
                f"坐标处理：{first_result.coordinate_note}\n\n"
                f"像素值映射：\n0: 背景\n{mapping}\n\n"
                f"写入要素：{sum(item.burned_features for item in results)} 个；"
                f"无标签要素跳过：{sum(item.skipped_unlabeled_features for item in results)} 个；"
                f"未知类别跳过：{sum(item.skipped_unknown_features for item in results)} 个。"
            )
            self.finished_signal.emit(message, "info")
        except RasterizeCancelled:
            self.finished_signal.emit("操作已取消。", "info")
        except Exception as exc:
            details = traceback.format_exc()
            self.finished_signal.emit(f"SHP 标签转 TIF 失败：{exc}\n\n{details}", "error")


class ShpLabelRasterizerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHP 标签转 TIF")
        self.setGeometry(0, 0, 740, 660)
        self._center_window()
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_CommandLink))

        self.shp_path: Optional[str] = None
        self.reference_tif_path: Optional[str] = None
        self.output_tif_path: Optional[str] = None
        self.worker: Optional[ShpLabelRasterizeThread] = None

        self._init_ui()
        self._update_start_state()

        if not GIS_AVAILABLE:
            QMessageBox.critical(
                self,
                "依赖缺失",
                f"SHP 标签转 TIF 需要 fiona 和 rasterio。\n当前导入失败：{GIS_IMPORT_ERROR}",
            )
            for child in self.findChildren(QWidget):
                child.setEnabled(False)

    def _center_window(self) -> None:
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        input_group = QGroupBox("输入与输出")
        input_layout = QFormLayout(input_group)
        input_layout.setSpacing(10)

        self.batch_mode_checkbox = QCheckBox("批量处理文件夹")
        self.batch_mode_checkbox.setToolTip("开启后选择 SHP 文件夹和原始影像 TIF 文件夹，按同名文件一一配对。")
        self.batch_mode_checkbox.stateChanged.connect(self.on_batch_mode_changed)
        input_layout.addRow(self.batch_mode_checkbox)

        self.shp_btn = QPushButton("选择 SHP 文件")
        self.shp_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.shp_btn.clicked.connect(self.select_shp)
        input_layout.addRow(self.shp_btn)

        self.reference_tif_btn = QPushButton("选择原始影像/参考 TIF")
        self.reference_tif_btn.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.reference_tif_btn.clicked.connect(self.select_reference_tif)
        input_layout.addRow(self.reference_tif_btn)

        self.output_tif_btn = QPushButton("选择输出 TIF")
        self.output_tif_btn.setIcon(self.style().standardIcon(QStyle.SP_DriveHDIcon))
        self.output_tif_btn.clicked.connect(self.select_output_tif)
        input_layout.addRow(self.output_tif_btn)

        self.label_field_edit = QLineEdit("label")
        self.label_field_edit.setToolTip("SHP 属性表中的类别字段名，大小写不敏感；常见为 label 或 Label。")
        input_layout.addRow(QLabel("类别字段名:"), self.label_field_edit)

        self.encoding_edit = QLineEdit("")
        self.encoding_edit.setToolTip("留空自动识别；中文标签乱码时可填写 UTF-8 或 GBK。")
        input_layout.addRow(QLabel("SHP 编码:"), self.encoding_edit)

        self.all_touched_checkbox = QCheckBox("包含边界接触像素")
        self.all_touched_checkbox.setToolTip("开启后，只要像素接触到面边界就会被赋值；默认仅按像素中心判断。")
        input_layout.addRow(self.all_touched_checkbox)
        main_layout.addWidget(input_group)

        labels_group = QGroupBox("类别设置")
        labels_layout = QVBoxLayout(labels_group)
        labels_layout.setSpacing(8)

        load_labels_btn = QPushButton("从 SHP 读取类别")
        load_labels_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        load_labels_btn.clicked.connect(self.load_labels_from_shp)
        labels_layout.addWidget(load_labels_btn, alignment=Qt.AlignRight)

        self.labels_edit = QTextEdit()
        self.labels_edit.setText(DEFAULT_LABELS)
        self.labels_edit.setToolTip("每行一个类别映射，推荐写成 1=房子。也支持 房子=1；未覆盖区域固定为 0 背景。")
        labels_layout.addWidget(self.labels_edit)
        main_layout.addWidget(labels_group)

        self.start_btn = QPushButton("开始转换")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(130)
        main_layout.addWidget(self.log_edit)

    def _update_button_text(self, button: QPushButton, base_text: str, path: Optional[str]) -> None:
        if path:
            button.setText(f"{base_text}: {os.path.basename(path)}")
            button.setToolTip(path)
        else:
            button.setText(base_text)
            button.setToolTip("")

    def _shp_button_text(self) -> str:
        return "SHP 文件夹" if self.batch_mode_checkbox.isChecked() else "SHP 文件"

    def _reference_button_text(self) -> str:
        return "原始影像/参考 TIF 文件夹" if self.batch_mode_checkbox.isChecked() else "原始影像/参考 TIF"

    def _output_button_text(self) -> str:
        return "输出文件夹" if self.batch_mode_checkbox.isChecked() else "输出 TIF"

    def _update_start_state(self) -> None:
        batch_mode = self.batch_mode_checkbox.isChecked()
        shp_ready = bool(self.shp_path and (Path(self.shp_path).is_dir() if batch_mode else Path(self.shp_path).is_file()))
        tif_ready = bool(
            self.reference_tif_path
            and (Path(self.reference_tif_path).is_dir() if batch_mode else Path(self.reference_tif_path).is_file())
        )
        output_ready = bool(
            self.output_tif_path
            and (Path(self.output_tif_path).is_dir() if batch_mode else bool(Path(self.output_tif_path).suffix))
        )
        self.start_btn.setEnabled(
            bool(GIS_AVAILABLE and shp_ready and tif_ready and output_ready)
        )

    def on_batch_mode_changed(self, _state=None) -> None:
        batch_mode = self.batch_mode_checkbox.isChecked()
        self.shp_btn.setText("选择 SHP 文件夹" if batch_mode else "选择 SHP 文件")
        self.reference_tif_btn.setText("选择原始影像/参考 TIF 文件夹" if batch_mode else "选择原始影像/参考 TIF")
        self.output_tif_btn.setText("选择输出文件夹" if batch_mode else "选择输出 TIF")
        self._update_start_state()

    def select_shp(self) -> None:
        if self.batch_mode_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择 SHP 文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择 SHP 文件", "", "Shapefiles (*.shp);;All Files (*)")
        if path:
            self.shp_path = path
            self._update_button_text(self.shp_btn, self._shp_button_text(), path)
            if not self.batch_mode_checkbox.isChecked() and not self.output_tif_path:
                output_path = str(Path(path).with_name(f"{Path(path).stem}_label_id.tif"))
                self.output_tif_path = output_path
                self._update_button_text(self.output_tif_btn, self._output_button_text(), output_path)
            self._update_start_state()

    def select_reference_tif(self) -> None:
        if self.batch_mode_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择原始影像/参考 TIF 文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择原始影像/参考 TIF", "", "TIFF files (*.tif *.tiff);;All Files (*)")
        if path:
            self.reference_tif_path = path
            self._update_button_text(self.reference_tif_btn, self._reference_button_text(), path)
            self._update_start_state()

    def select_output_tif(self) -> None:
        if self.batch_mode_checkbox.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if path:
                self.output_tif_path = path
                self._update_button_text(self.output_tif_btn, self._output_button_text(), path)
                self._update_start_state()
            return

        default_path = self.output_tif_path or ""
        path, _ = QFileDialog.getSaveFileName(self, "选择输出 TIF", default_path, "TIFF files (*.tif *.tiff)")
        if path:
            if not path.lower().endswith((".tif", ".tiff")):
                path += ".tif"
            self.output_tif_path = path
            self._update_button_text(self.output_tif_btn, self._output_button_text(), path)
            self._update_start_state()

    def load_labels_from_shp(self) -> None:
        if not self.shp_path:
            QMessageBox.warning(self, "输入缺失", "请先选择 SHP 文件。")
            return

        shp_source = self.shp_path
        if self.batch_mode_checkbox.isChecked():
            shp_files = get_supported_shp_files(self.shp_path)
            if not shp_files:
                QMessageBox.warning(self, "输入缺失", "SHP 文件夹中没有找到 .shp 文件。")
                return
            shp_source = str(shp_files[0])

        label_field = self.label_field_edit.text().strip() or "label"
        try:
            encoding = self.encoding_edit.text().strip() or None
            open_kwargs = {"encoding": encoding} if encoding else {}
            with fiona.open(shp_source, "r", **open_kwargs) as shp:
                actual_field = infer_label_field(shp.schema.get("properties", {}), label_field)
                if actual_field is None:
                    fields = ", ".join(shp.schema.get("properties", {}).keys()) or "无"
                    QMessageBox.warning(self, "字段不存在", f"SHP 中没有字段 '{label_field}'。\n可用字段：{fields}")
                    return
                if actual_field != label_field:
                    self.label_field_edit.setText(actual_field)
                values = sorted(
                    {
                        str(feature["properties"].get(actual_field)).strip()
                        for feature in shp
                        if feature["properties"].get(actual_field) is not None
                        and str(feature["properties"].get(actual_field)).strip()
                    }
                )
            if not values:
                QMessageBox.warning(self, "未读取到类别", f"字段 '{label_field}' 中没有有效类别值。")
                return
            self.labels_edit.setText("\n".join(values))
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", f"读取 SHP 类别失败：{exc}")

    def start_processing(self) -> None:
        if not self.shp_path or not self.reference_tif_path or not self.output_tif_path:
            QMessageBox.warning(self, "输入缺失", "请选择 SHP、原始影像/参考 TIF 和输出位置。")
            return

        try:
            class_mapping = parse_class_mapping(self.labels_edit.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "类别配置错误", str(exc))
            return

        batch_mode = self.batch_mode_checkbox.isChecked()
        if batch_mode:
            jobs, missing_tifs = build_batch_jobs(self.shp_path, self.reference_tif_path, self.output_tif_path)
            if not jobs:
                QMessageBox.warning(self, "输入错误", "没有找到可处理的 SHP/TIF 同名配对。")
                return
            if missing_tifs:
                preview = "\n".join(missing_tifs[:10])
                suffix = "\n..." if len(missing_tifs) > 10 else ""
                QMessageBox.warning(
                    self,
                    "参考 TIF 缺失",
                    f"以下 SHP 没有找到同名原始影像/参考 TIF，已停止批量转换：\n{preview}{suffix}",
                )
                return
        else:
            jobs = [
                ShpRasterizeJob(
                    shp_path=self.shp_path,
                    reference_tif_path=self.reference_tif_path,
                    output_tif_path=self.output_tif_path,
                )
            ]

        label_field = self.label_field_edit.text().strip() or "label"
        encoding = self.encoding_edit.text().strip() or None
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        self.worker = ShpLabelRasterizeThread(
            jobs,
            label_field,
            class_mapping,
            self.all_touched_checkbox.isChecked(),
            encoding,
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.log_edit.append(message)
        self.log_edit.ensureCursorVisible()
        QApplication.processEvents()

    def on_finished(self, message: str, message_type: str) -> None:
        self._update_start_state()
        self.log_edit.append(f"--------------------\n{message}")
        self.log_edit.ensureCursorVisible()
        if message_type == "info":
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "操作完成", message)
        else:
            QMessageBox.critical(self, "操作失败", message)

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "退出确认",
                "SHP 转 TIF 正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.requestInterruption()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = ShpLabelRasterizerWindow()
    window.show()
    app.exec_()
