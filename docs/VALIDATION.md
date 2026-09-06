# 1.0.0 验证记录

应用源代码提交：`df7851caf17d39f864c5026d811517c8124d61e3`。Windows x64，系统 build 26100，Python 3.10.20。

## 功能与安装验证

- 29 项回归测试通过，覆盖 TIFF/SHP/GPKG、16 位标签、点云采样、中文路径、输出保护与线程关闭。
- [GitHub Windows CI](https://github.com/yaluncoco/RemoteSensingToolbox/actions/runs/34025937554) 已通过依赖安装、回归测试和源码自检。
- 打包程序的 15 项离线自检通过。
- 安装到包含中文和空格的目录后，16 项自检通过，包括原生 OpenGL 三维渲染。
- 安装后的测试进程从工作目录之外启动，PATH 仅保留 Windows 系统目录，并清除了 Python、Conda、GDAL、PROJ 和 Qt 的外部路径变量。
- 当前用户静默安装、开始菜单快捷方式、原生 Windows 主窗口启动与正常退出均通过。
- 静默卸载成功，测试安装文件、卸载注册和开始菜单入口已清理。
- 便携 ZIP 的 1907 个文件通过 CRC 校验，安装包和 ZIP 的 SHA-256 与清单一致。

这是本机隔离环境变量后的验证，并非全新 Windows 虚拟机测试。其他机器的三维预览仍受显卡驱动和 OpenGL 支持影响。安装包未配置代码签名。

复现安装验证（构建机上不能已有本应用的安装记录或同名开始菜单目录）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_installer.ps1 -CheckRenderer
```

## 大图转换对比

合成影像：4096 × 4096、3 波段、float32；最小/最大值拉伸，输出 RGB GeoTIFF。两个版本在独立 Windows 进程中顺序运行，使用相同依赖和输入文件。基线为导入项目时的 `dd0e9d3`。

| 指标 | 原版本 | 优化版本 |
| --- | ---: | ---: |
| 峰值工作集 | 565.16 MiB | 143.71 MiB |
| 转换耗时 | 2.832 秒 | 1.761 秒 |

此样例峰值内存降低 **74.6%**。这是单次本机测量，耗时会随数据、缓存和硬件变化，不能代表所有任务。

复现：

```powershell
python scripts\benchmark_rgb.py
```

## 发行文件

完整机器可读报告随 Release 提供：`validation-1.0.0.json`。

- `RemoteSensingToolbox-1.0.0-windows-x64-portable.zip`：246,663,383 字节；SHA-256 `71f91ee2363b0eeab9d56d81699a15ba9118f220cfa0fd07d2603797f40ced35`。
- `RemoteSensingToolbox-1.0.0-windows-x64-setup.exe`：166,081,980 字节；SHA-256 `0f82c1eb6cb6256d752ab36bb4339d6e95868e90248787b06c9769bb09552920`。
