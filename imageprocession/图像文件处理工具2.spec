# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
rasterio_hiddenimports = collect_submodules('rasterio')

a = Analysis(
    ['main.py'],  # 入口文件
    pathex=['.'],  # 当前目录下
    binaries=[],
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
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
    console=False,  # 设置为 True 可输出控制台调试信息
    icon='favicon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='图像文件处理工具'
)
