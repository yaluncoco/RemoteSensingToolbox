"""Explicit column layouts and validated numeric input for point-cloud previews."""
from dataclasses import dataclass
from itertools import islice
import warnings

import numpy as np


@dataclass(frozen=True)
class PointLayout:
    title: str
    columns: int
    label: int | None = None
    rgb: tuple[int, int, int] | None = None


# Column numbers shown to users start at 1. Extra columns are left unused.
POINT_LAYOUTS = {
    "xyz": PointLayout("XYZ（坐标）", 3),
    "xyz_label": PointLayout("XYZ + Label", 4, label=4),
    "xyz_rgb": PointLayout("XYZ + RGB", 6, rgb=(4, 5, 6)),
    "xyz_rgb_label": PointLayout("XYZ + RGB + Label", 7, label=7, rgb=(4, 5, 6)),
    "xyz_rgb_label_normal": PointLayout("XYZ + RGB + Label + 法向量", 10, label=7, rgb=(4, 5, 6)),
    "custom": PointLayout("自定义列位置…", 3),
}


@dataclass(frozen=True)
class VisualizationOptions:
    layout: str = "xyz_rgb_label"
    color_mode: str = "label"
    xyz_columns: tuple[int, int, int] = (1, 2, 3)
    label_column: int = 7
    rgb_columns: tuple[int, int, int] = (4, 5, 6)
    delimiter: str = "auto"
    skip_rows: int = 0
    rgb_range: str = "auto"
    uniform_color: tuple[float, float, float] = (0.5, 0.5, 0.5)


def read_visualization_text(path, options, preview=False):
    """Read numeric TXT/XYZ/CSV. Preview reads at most 64 KiB and five rows."""
    delimiters = {"whitespace": None, "comma": ",", "semicolon": ";", "tab": "\t"}
    if options.delimiter not in ("auto", *delimiters):
        raise ValueError("请选择有效的分隔符。")
    if not isinstance(options.skip_rows, int) or options.skip_rows < 0:
        raise ValueError("跳过行数必须是非负整数。")
    try:
        with open(path, "r", encoding="utf-8-sig") as stream:
            if preview:
                # Never read the whole cloud on the GUI thread, including long headers.
                block = stream.read(65537)
                lines = block[:65536].splitlines(keepends=True)
                if len(block) > 65536:
                    lines = lines[:-1]  # Do not parse a truncated record.
                source = iter(lines)
            else:
                source = iter(stream)
            for _ in range(options.skip_rows):
                next(source, None)
            first = None
            for line in source:
                if line.split("#", 1)[0].strip():
                    first = line
                    break
            if first is None:
                if preview and len(block) > 65536:
                    raise ValueError("前 64 KB 未找到数据，请检查跳过行数；完整文件可在开始时读取。")
                raise ValueError("文件为空，或跳过表头后没有数据。")
            delimiter = delimiters.get(options.delimiter)
            if options.delimiter == "auto":
                record = first.split("#", 1)[0]
                delimiter = "," if "," in record else ";" if ";" in record else None
            from itertools import chain
            rows = chain([first], source)
            if preview:
                rows = islice((line for line in rows if line.split("#", 1)[0].strip()), 5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                data = np.loadtxt(rows, delimiter=delimiter, ndmin=2)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"无法读取文件，请使用 UTF-8 文本并检查路径：{exc}") from exc
    except ValueError as exc:
        raise ValueError(f"读取失败：{exc}\n请检查分隔符、表头行数以及各行列数是否一致。") from exc
    if not data.size:
        raise ValueError("文件中没有有效点数据。")
    return data


def resolve_columns(options, column_count):
    layout = POINT_LAYOUTS.get(options.layout)
    if layout is None:
        raise ValueError("请选择有效的数据排列。")
    if options.color_mode not in ("label", "rgb", "uniform"):
        raise ValueError("请选择有效的着色方式。")
    custom = options.layout == "custom"
    xyz = options.xyz_columns if custom else (1, 2, 3)
    label = options.label_column if custom else layout.label
    rgb = options.rgb_columns if custom else layout.rgb
    columns = list(xyz)
    if len(columns) != 3:
        raise ValueError("请分别指定 X、Y、Z 三个坐标列。")
    if options.color_mode == "label":
        if label is None:
            raise ValueError("当前数据排列没有标签列，请选择单色或原始 RGB。")
        columns.append(label)
    elif options.color_mode == "rgb":
        if rgb is None or len(rgb) != 3:
            raise ValueError("当前数据排列没有 RGB 三列，请选择标签或单色。")
        columns.extend(rgb)
    if any(not isinstance(col, int) or col < 1 for col in columns):
        raise ValueError("列号从 1 开始，请填写正整数。")
    if len(set(columns)) != len(columns):
        raise ValueError("坐标、标签或 RGB 的列号不能重复。")
    required = max(columns) if custom else layout.columns
    if column_count < required:
        suggestion = {3: "XYZ（坐标）", 4: "XYZ + Label", 6: "XYZ + RGB"}.get(column_count)
        hint = f"若数据符合该排列，可选择「{suggestion}」。" if suggestion else "请改选数据排列或自定义列位置。"
        raise ValueError(f"文件有 {column_count} 列，当前设置需要至少 {required} 列。{hint}")
    return tuple(col - 1 for col in xyz), None if label is None else label - 1, (
        None if rgb is None else tuple(col - 1 for col in rgb))


def extract_visualization_data(data, options):
    """Validate only used columns; preserve label IDs, including negative IDs."""
    if data.ndim != 2 or not data.size:
        raise ValueError("点云数据必须是非空的二维数值表。")
    xyz, label_column, rgb_columns = resolve_columns(options, data.shape[1])
    points = data[:, xyz]
    if not np.isfinite(points).all():
        raise ValueError("坐标列含 NaN 或无穷值，请清理后重试。")
    labels, colors = None, None
    if options.color_mode == "label":
        values = data[:, label_column]
        if not np.isfinite(values).all() or not np.equal(values, np.trunc(values)).all():
            raise ValueError("标签列必须是有限整数，请检查标签列位置；不会自动取整或减 1。")
        # Float text cannot represent all integers outside this range exactly.
        if np.any(np.abs(values) > 2**53 - 1):
            raise ValueError("标签编号超出可精确读取的整数范围。")
        labels = values.astype(np.int64)
    elif options.color_mode == "rgb":
        colors = data[:, rgb_columns].copy()
        if not np.isfinite(colors).all() or np.any(colors < 0):
            raise ValueError("RGB 列必须是有限的非负数。")
        if options.rgb_range not in ("auto", "0-1", "0-255"):
            raise ValueError("请选择有效的 RGB 范围。")
        scale = 255 if options.rgb_range == "0-255" or (options.rgb_range == "auto" and colors.max() > 1) else 1
        if colors.max() > scale:
            raise ValueError(f"RGB 值超出 0–{scale}，请检查列位置或 RGB 范围。")
        colors /= scale
    else:
        color = np.asarray(options.uniform_color, dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError("单色必须是 0–1 范围的 RGB 三元组。")
        colors = np.broadcast_to(color, points.shape).copy()
    return points, labels, colors


def apply_label_colors(labels, color_map, default_color):
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    palette = np.asarray([color_map.get(int(label), default_color) for label in unique_labels], dtype=float)
    return palette[inverse]
