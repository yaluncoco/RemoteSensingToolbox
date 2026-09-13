from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_PROJECT_DIR = ROOT_DIR / "imageprocession"
POINT_CLOUD_PROJECT_DIR = ROOT_DIR / "PointCloud"
_DLL_DIRECTORY_HANDLES = []
_RUNTIME_CONFIGURED = False


@dataclass(frozen=True)
class ToolSpec:
    id: str
    title: str
    category: str
    summary: str
    module_name: str
    class_name: str
    project_dir: Path
    icon_name: str
    tags: tuple[str, ...] = field(default_factory=tuple)


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        id="image-cutter",
        title="图像切片",
        category="图像处理",
        summary="影像与标签同步切片，支持边缘补齐和无效区域过滤。",
        module_name="image_cutter",
        class_name="ImageCutterWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_FileDialogListView",
        tags=("遥感", "切片", "标签"),
    ),
    ToolSpec(
        id="image-visualizer",
        title="图像与标签可视化",
        category="图像处理",
        summary="对照查看影像与语义标签，自定义类别名称和颜色。",
        module_name="visualizer",
        class_name="VisualizationWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_DesktopIcon",
        tags=("可视化", "标签", "颜色"),
    ),
    ToolSpec(
        id="rgb-color-extractor",
        title="RGB 颜色提取",
        category="图像处理",
        summary="提取图例色块与 RGB 值，导出 CSV、JSON 或颜色字典。",
        module_name="rgb_color_extractor",
        class_name="RgbColorExtractorWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_DialogApplyButton",
        tags=("RGB", "颜色", "图例", "截图"),
    ),
    ToolSpec(
        id="tif-rgb-converter",
        title="TIF 转 RGB",
        category="图像处理",
        summary="选择 RGB 波段，批量输出 PNG、JPG 或 GeoTIFF。",
        module_name="tif_rgb_converter",
        class_name="TifRgbConverterWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_FileDialogDetailedView",
        tags=("TIF", "RGB", "格式转换"),
    ),
    ToolSpec(
        id="tif-rasterizer",
        title="TIF 矢量栅格化",
        category="图像处理",
        summary="将 GPKG 标注对齐到参考 TIF，输出类别与彩色标签。",
        module_name="tif_processor",
        class_name="TifProcessorWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_CommandLink",
        tags=("TIF", "GPKG", "栅格化"),
    ),
    ToolSpec(
        id="shp-label-rasterizer",
        title="SHP 标签转 TIF",
        category="图像处理",
        summary="按 SHP 属性生成标签 TIF，支持类别映射与批量配对。",
        module_name="shp_label_rasterizer",
        class_name="ShpLabelRasterizerWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_CommandLink",
        tags=("SHP", "Label", "TIF", "栅格化"),
    ),
    ToolSpec(
        id="voc-creator",
        title="VOC 数据集制作",
        category="数据集",
        summary="整理影像和标签，生成 VOC 目录并划分训练 / 验证集。",
        module_name="voc_creator",
        class_name="VOCDatasetCreatorWindow",
        project_dir=IMAGE_PROJECT_DIR,
        icon_name="SP_DriveHDIcon",
        tags=("VOC", "数据集", "训练集"),
    ),
    ToolSpec(
        id="point-cloud-workbench",
        title="点云处理工作台",
        category="点云处理",
        summary="点云预览、去噪、格式转换、标签分析与下采样。",
        module_name="gui",
        class_name="PointCloudProcessor",
        project_dir=POINT_CLOUD_PROJECT_DIR,
        icon_name="SP_ComputerIcon",
        tags=("点云", "Open3D", "去噪", "转换"),
    ),
)


def configure_runtime_environment() -> None:
    """Set compatibility variables before heavy scientific libraries are imported."""
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OPEN3D_DISABLE_WEB_VISUALIZER", "true")

    frozen_base_path = Path(getattr(sys, "_MEIPASS", sys.prefix))
    prefix_path = Path(sys.prefix)
    conda_base_path = (
        prefix_path.parents[1]
        if len(prefix_path.parents) >= 2 and prefix_path.parent.name.lower() == "envs"
        else prefix_path
    )
    dll_paths = [
        frozen_base_path,
        frozen_base_path / "Library" / "bin",
        prefix_path,
        prefix_path / "DLLs",
        prefix_path / "Library" / "bin",
        prefix_path / "Library" / "usr" / "bin",
        prefix_path / "Library" / "mingw-w64" / "bin",
        prefix_path / "Scripts",
    ]
    existing_paths = [path for path in dll_paths if path.exists()]

    gdal_data_candidates = [
        frozen_base_path / "gdal_data",
        frozen_base_path / "rasterio" / "gdal_data",
        frozen_base_path / "fiona" / "gdal_data",
        frozen_base_path / "Library" / "share" / "gdal",
        prefix_path / "Library" / "share" / "gdal",
        conda_base_path / "Library" / "share" / "gdal",
    ]
    proj_data_candidates = [
        frozen_base_path / "proj_data",
        frozen_base_path / "rasterio" / "proj_data",
        frozen_base_path / "fiona" / "proj_data",
        frozen_base_path / "Library" / "share" / "proj",
        prefix_path / "Library" / "share" / "proj",
        conda_base_path / "Library" / "share" / "proj",
    ]

    for env_name, candidates in [
        ("GDAL_DATA", gdal_data_candidates),
        ("PROJ_LIB", proj_data_candidates),
        ("PROJ_DATA", proj_data_candidates),
    ]:
        if os.environ.get(env_name) and not getattr(sys, "frozen", False):
            continue
        for path in candidates:
            if path.exists():
                os.environ[env_name] = str(path)
                break

    current_path = os.environ.get("PATH", "")
    for path in reversed(existing_paths):
        path_text = str(path)
        if path_text not in current_path:
            current_path = path_text + os.pathsep + current_path
    os.environ["PATH"] = current_path

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        _RUNTIME_CONFIGURED = True
        return

    for path in existing_paths:
        try:
            handle = add_dll_directory(str(path))
            _DLL_DIRECTORY_HANDLES.append(handle)
        except OSError:
            pass

    _RUNTIME_CONFIGURED = True


def ensure_project_on_path(project_dir: Path) -> None:
    project_path = str(project_dir)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def load_tool_class(spec: ToolSpec) -> Type:
    configure_runtime_environment()
    ensure_project_on_path(spec.project_dir)
    module = importlib.import_module(spec.module_name)
    try:
        return getattr(module, spec.class_name)
    except AttributeError as exc:
        raise ImportError(
            f"模块 {spec.module_name!r} 中没有找到窗口类 {spec.class_name!r}。"
        ) from exc
