# -*- mode: python ; coding: utf-8 -*-
"""Windows CPU distribution. Build with scripts/build_windows.ps1."""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from Toolbox.tool_registry import TOOLS

datas = [
    (str(root / "Toolbox" / "style.qss"), "."),
    (str(root / "Toolbox" / "quick_start.md"), "."),
    (str(root / "PointCloud" / "images" / "point.ico"), "."),
    (str(root / ".build" / "licenses"), "licenses"),
]
binaries = []
hiddenimports = [spec.module_name for spec in TOOLS] + ["workers", "visualization", "open3d.cpu.pybind", "backports.tarfile"]

for package in ("rasterio", "fiona"):
    datas += collect_data_files(package, excludes=["**/tests/**", "**/__pycache__/**"])
    binaries += collect_dynamic_libs(package)
    hiddenimports += collect_submodules(package, filter=lambda name: ".tests" not in name)

datas += collect_data_files("open3d", excludes=["**/examples/**", "**/labextension/**", "**/nbextension/**", "**/__pycache__/**"])
binaries += collect_dynamic_libs("open3d")
for distribution in ("PyQt5", "numpy", "Pillow", "rasterio", "fiona", "scipy", "scikit-learn", "matplotlib", "open3d"):
    datas += copy_metadata(distribution)

# Conda's Python may place standard-library/VC dependencies outside its DLLs directory.
# Pip wheels supply their own GIS and BLAS runtimes; do not copy an entire Conda runtime.
base = Path(sys.base_prefix)
base_conda = base.parents[1] if base.parent.name.lower() == "envs" else base
for filename in ("expat.dll", "libexpat.dll", "sqlite3.dll", "vcomp140.dll", "libffi-8.dll"):
    for directory in (base / "Library" / "bin", base_conda / "Library" / "bin", base / "DLLs"):
        candidate = directory / filename
        if candidate.is_file():
            binaries.append((str(candidate), "."))
            break

analysis = Analysis(
    [str(root / "run_toolbox.py")],
    pathex=[str(root), str(root / "imageprocession"), str(root / "PointCloud")],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={"matplotlib": {"backends": ["Qt5Agg", "Agg"]}},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "tensorboard", "open3d.ml.torch", "open3d.ml.tf",
              "IPython", "ipykernel", "ipywidgets", "jupyterlab", "notebook"],
    noarchive=False, optimize=1,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name="RemoteSensingToolbox", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, disable_windowed_traceback=False,
    icon=str(root / "PointCloud" / "images" / "point.ico"),
    version=str(root / ".build" / "version_info.txt"),
)
collect = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name="RemoteSensingToolbox")
