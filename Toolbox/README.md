# 遥感点云工具箱

这是 `imageprocession` 和 `PointCloud` 两个项目的统一桌面入口。

主手册：

```text
E:\application\遥感点云工具箱_使用与维护手册.md
```

## 运行

```powershell
cd E:\application
conda run -n app python run_toolbox.py
```

或：

```powershell
cd E:\application
conda run -n app python Toolbox\main.py
```

## 打包

```powershell
cd E:\application
conda run -n app pyinstaller Toolbox\toolbox.spec --noconfirm
```

输出目录：

```text
E:\application\dist\遥感点云工具箱
```

发布时必须保留整个输出文件夹，不能只拷贝单个 exe。

## 新增工具

新增工具优先做成独立 `QWidget` 或 `QMainWindow`，然后在 `Toolbox\tool_registry.py` 的 `TOOLS` 中添加 `ToolSpec`。

如果新模块需要被 PyInstaller 打包，记得同步添加到 `Toolbox\toolbox.spec` 的 `hiddenimports`。

