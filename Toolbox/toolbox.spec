# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from glob import glob
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


toolbox_dir = Path(SPECPATH)
root_dir = toolbox_dir.parent
image_dir = root_dir / "imageprocession"
point_dir = root_dir / "PointCloud"
site_packages = Path(sys.prefix) / "Lib" / "site-packages"
rasterio_dir = site_packages / "rasterio"
fiona_dir = site_packages / "fiona"

conda_library_bin = Path(sys.prefix) / "Library" / "bin"
conda_prefix = Path(sys.prefix)
conda_base = conda_prefix.parents[1] if conda_prefix.parent.name.lower() == "envs" else conda_prefix
conda_base_library_bin = conda_base / "Library" / "bin"
conda_dll_paths = [
    conda_prefix,
    conda_prefix / "DLLs",
    conda_library_bin,
    conda_prefix / "Library" / "usr" / "bin",
    conda_prefix / "Library" / "mingw-w64" / "bin",
    conda_prefix / "Scripts",
]
os.environ["PATH"] = (
    os.pathsep.join(str(path) for path in conda_dll_paths if path.exists())
    + os.pathsep
    + os.environ.get("PATH", "")
)

runtime_binaries = []
for pattern in ("mkl*.dll", "libiomp*.dll", "tbb*.dll"):
    runtime_binaries.extend((dll_path, ".") for dll_path in glob(str(conda_library_bin / pattern)))

for dll_name in [
    "vcomp140.dll",
    "libiomp5md.dll",
    "libiompstubs5md.dll",
    "expat.dll",
    "libexpat.dll",
    "sqlite3.dll",
]:
    dll_path = conda_library_bin / dll_name
    if not dll_path.exists():
        dll_path = conda_base_library_bin / dll_name
    if dll_path.exists():
        runtime_binaries.append((str(dll_path), "."))

gis_package_binaries = []
for package_dir, target_dir in [(rasterio_dir, "rasterio"), (fiona_dir, "fiona")]:
    if package_dir.exists():
        gis_package_binaries.extend((str(path), target_dir) for path in package_dir.glob("*.pyd"))

gis_package_datas = []
for package_dir, target_dir in [(rasterio_dir, "rasterio"), (fiona_dir, "fiona")]:
    if package_dir.exists():
        gis_package_datas.append((str(package_dir), target_dir))

gis_runtime_datas = []
gdal_data_dir = conda_prefix / "Library" / "share" / "gdal"
proj_data_dir = conda_prefix / "Library" / "share" / "proj"
if gdal_data_dir.exists():
    gis_runtime_datas.append((str(gdal_data_dir), "gdal_data"))
if proj_data_dir.exists():
    gis_runtime_datas.append((str(proj_data_dir), "proj_data"))

try:
    open3d_datas = collect_data_files(
        "open3d",
        include_py_files=False,
        excludes=["**/examples/**", "**/labextension/**", "**/nbextension/**", "**/__pycache__/**"],
    ) + copy_metadata("open3d")
    open3d_binaries = collect_dynamic_libs("open3d")
    open3d_hiddenimports = ["open3d.cpu.pybind"]
except Exception:
    open3d_datas = []
    open3d_binaries = []
    open3d_hiddenimports = []

rasterio_hiddenimports = collect_submodules("rasterio")
fiona_hiddenimports = collect_submodules("fiona")
manual_gis_hiddenimports = [
    "rasterio",
    "rasterio._base",
    "rasterio._env",
    "rasterio._err",
    "rasterio._features",
    "rasterio._filepath",
    "rasterio._fill",
    "rasterio._io",
    "rasterio._transform",
    "rasterio._version",
    "rasterio._vsiopener",
    "rasterio._warp",
    "rasterio.crs",
    "rasterio.dtypes",
    "rasterio.env",
    "rasterio.errors",
    "rasterio.features",
    "rasterio.sample",
    "rasterio.transform",
    "rasterio.vrt",
    "rasterio.windows",
    "fiona",
    "fiona._env",
    "fiona._err",
    "fiona._geometry",
    "fiona._transform",
    "fiona._vsiopener",
    "fiona.crs",
    "fiona.drvsupport",
    "fiona.env",
    "fiona.errors",
    "fiona.ogrext",
    "fiona.schema",
    "fiona.session",
]

a = Analysis(
    ["main.py"],
    pathex=[str(toolbox_dir), str(image_dir), str(point_dir)],
    binaries=open3d_binaries + runtime_binaries + gis_package_binaries,
    datas=[("style.qss", ".")] + open3d_datas + gis_package_datas + gis_runtime_datas,
    hiddenimports=[
        "image_cutter",
        "visualizer",
        "rgb_color_extractor",
        "tif_rgb_converter",
        "voc_creator",
        "tif_processor",
        "shp_label_rasterizer",
        "gui",
        "workers",
        "visualization",
    ]
    + open3d_hiddenimports
    + rasterio_hiddenimports
    + fiona_hiddenimports
    + manual_gis_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="\u9065\u611f\u70b9\u4e91\u5de5\u5177\u7bb1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(point_dir / "images" / "point.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="\u9065\u611f\u70b9\u4e91\u5de5\u5177\u7bb1",
)
