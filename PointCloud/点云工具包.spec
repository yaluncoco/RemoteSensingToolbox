# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

# 只收集本工具实际需要的 Open3D 运行资源，避免把示例、Jupyter 和 ML 可选依赖全部打进包。
open3d_datas = collect_data_files(
    'open3d',
    include_py_files=False,
    excludes=[
        '**/examples/**',
        '**/labextension/**',
        '**/nbextension/**',
        '**/__pycache__/**',
    ],
) + copy_metadata('open3d')
open3d_binaries = collect_dynamic_libs('open3d')
open3d_hiddenimports = ['open3d.cpu.pybind']

conda_library_bin = Path(sys.prefix) / 'Library' / 'bin'
conda_dll_paths = [
    Path(sys.prefix),
    Path(sys.prefix) / 'DLLs',
    conda_library_bin,
    Path(sys.prefix) / 'Library' / 'usr' / 'bin',
    Path(sys.prefix) / 'Library' / 'mingw-w64' / 'bin',
    Path(sys.prefix) / 'Scripts',
]
os.environ['PATH'] = os.pathsep.join(str(path) for path in conda_dll_paths if path.exists()) + os.pathsep + os.environ.get('PATH', '')

mkl_binaries = [(str(path), '.') for path in conda_library_bin.glob('mkl_*.dll')]
runtime_binaries = []
for dll_name in [
    'tbb12.dll',
    'vcomp140.dll',
    'libiomp5md.dll',
    'libiompstubs5md.dll',
    'expat.dll',
    'libexpat.dll',
    'sqlite3.dll',
]:
    dll_path = conda_library_bin / dll_name
    if dll_path.exists():
        runtime_binaries.append((str(dll_path), '.'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=open3d_binaries + mkl_binaries + runtime_binaries,
    datas=open3d_datas,
    hiddenimports=open3d_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='点云工具包',
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
    icon=['images\\point.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='点云工具包',
)
