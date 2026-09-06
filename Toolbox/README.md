# 工具注册与维护

运行、依赖与打包命令见 [项目说明](../README.md)。

统一入口是 `run_toolbox.py`，窗口和任务管理位于 `Toolbox/`。工具模块保留在 `imageprocession/` 与 `PointCloud/`。

## 新增工具

1. 创建独立的 `QWidget` 或 `QMainWindow`，在 `Toolbox/tool_registry.py` 的 `TOOLS` 中添加 `ToolSpec`。打包配置会自动读取注册表。
2. 将耗时操作放入 `QThread`，通过信号更新界面。
3. 在线程启动前调用 `bind_task(window, "worker_attribute", callback)`；任务提供 `finished_signal`，回调会在真正的 `QThread.finished` 后执行。需要取消时实现 `stop()` 或检查 `isInterruptionRequested()`。
4. 输出文件使用 `Toolbox.file_io.atomic_output`，失败时保留已有文件。地理 TIFF 的掩膜应内嵌，避免临时 sidecar 丢失。
5. 补充算法/生命周期回归测试，并运行源码和发行版自检。

新增原生依赖时，还需检查 `Toolbox/toolbox.spec` 的 DLL、数据文件及隐藏导入收集。运行 `scripts/build_windows.ps1` 会同时收集第三方许可证。
