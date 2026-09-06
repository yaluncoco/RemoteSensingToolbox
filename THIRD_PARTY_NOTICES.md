# 第三方组件

本程序使用 PyQt5/Qt、NumPy、Pillow、Rasterio/GDAL/PROJ、Fiona、SciPy、scikit-learn、Matplotlib、Open3D 等第三方组件。各组件仍适用各自的许可证。

发行版的 `_internal/licenses/` 目录包含打包环境中的许可证文件和 `dependencies.json` 版本/许可索引。该索引包含部分仅在构建阶段使用的组件。

主要上游项目与对应源码：

- PyQt5（GPL v3）：https://www.riverbankcomputing.com/software/pyqt/ — https://pypi.org/project/PyQt5/5.15.11/#files
- Qt 5（LGPL/GPL，按模块适用）：https://download.qt.io/archive/qt/5.15/5.15.2/single/
- NumPy：https://github.com/numpy/numpy
- Pillow：https://github.com/python-pillow/Pillow
- Rasterio：https://github.com/rasterio/rasterio
- Fiona：https://github.com/Toblerity/Fiona
- GDAL：https://github.com/OSGeo/gdal
- PROJ：https://github.com/OSGeo/PROJ
- SciPy：https://github.com/scipy/scipy
- scikit-learn：https://github.com/scikit-learn/scikit-learn
- Matplotlib：https://github.com/matplotlib/matplotlib
- Open3D：https://github.com/isl-org/Open3D
- Inno Setup：https://github.com/jrsoftware/issrc

安装程序使用 Inno Setup 编译。中文安装界面翻译来自 Inno Setup 官方仓库中的社区翻译文件，原始署名保留在 `installer/languages/ChineseSimplified.isl` 中。
