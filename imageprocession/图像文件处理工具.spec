# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from glob import glob
from PyInstaller.utils.hooks import collect_submodules


runtime_bin_dir = os.path.join(sys.prefix, 'Library', 'bin')
runtime_dll_patterns = ('mkl*.dll', 'libiomp*.dll', 'tbb*.dll')
runtime_binaries = [
    (dll_path, '.')
    for pattern in runtime_dll_patterns
    for dll_path in glob(os.path.join(runtime_bin_dir, pattern))
]
rasterio_hiddenimports = collect_submodules('rasterio')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=runtime_binaries,
    datas=[('style.qss', '.')],
     hiddenimports=rasterio_hiddenimports + [
        'rasterio.sample',
        'rasterio.env',
        'rasterio.transform',
        'rasterio._base',
        'rasterio._io',
        'rasterio.coords',
        'rasterio.crs',
        'rasterio.dtypes',
        'rasterio.errors',
        'rasterio.features',
        'rasterio.windows',
        'rasterio.vrt',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='图像文件处理工具',
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
    icon=['favicon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='图像文件处理工具',
)
