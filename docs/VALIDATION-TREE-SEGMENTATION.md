# 单木分割验证记录

日期：2026-11-06。Windows x64，独立 Python 3.12.7 环境，本机 CloudCompare `D:\apps\CloudCompare\CloudCompare.exe`。

单木分割随 v1.4.0 正式安装包和便携版提供；发行文件与安装验证见 [1.4.0 验证记录](VALIDATION-1.4.0.md)。以下预览构建记录保留用于追溯开发过程。

## CCB2 文件识别修复

- 用户实际输入为 `.laz` 扩展名、`CCB2` 文件头的 CloudCompare BIN。之前直接交给 laspy，导致 `Invalid file signature`。
- 新增按内容识别、BIN 工作副本转换、数值属性逐点验证、全局浮点坐标额外保存、格式错误提示，以及多点云容器拒绝处理。
- 修复后 96 项回归测试全部通过。源码与打包版均通过 22 项自检，包含真实生成一个扩展名为 `.laz` 的 BIN，再自动识别、转换及校验。
- 修复版可执行文件：`.build/forestry-bin-fix/RemoteSensingToolbox/RemoteSensingToolbox.exe`；原预览文件夹仍保留。
- `bin-fix-all-tests.log`、`bin-fix-self-test.json`、`bin-fix-frozen-self-test.json` 均保存在 `.build/`。

## 用户真实人工林数据（四份全部完成）

以下四份数据由源码 Python 运行，不能证明 v1.4.0 安装包完整流程通过。2026-11-07 确认 v1.4.0 安装包缺少 SVG 后端；v1.4.1 的安装版验证另行记录。

输入目录：`D:\Datasets\人工林（平原林）\处理后的点云`。四份文件均为扩展名 `.laz`、实际文件头 `CCB2` 的 CloudCompare BIN。采用平坦地形、0.2 米采样、默认 CSF / TreeIso 参数，开启逐树导出。

| 文件时间（2026-10-13） | 原始点数 | 单木候选 | 重点复核 | 端到端用时（秒） |
| --- | ---: | ---: | ---: | ---: |
| 12-31-27 | 10,972,811 | 224 | 14 | 505.0 |
| 12-39-54 | 10,414,470 | 283 | 47 | 387.5 |
| 15-42-09 | 25,183,760 | 656 | 73 | 2,223.4 |
| 16-24-51 | 10,353,528 | 240 | 25 | 502.9 |

共处理 56,924,569 点，导出 1,403 个单木候选文件，159 个由规则标记为重点复核。计数为逐文件相加，未跨文件去重。

四份均通过 BIN 数值属性转换校验、规范化源记录与点序比较、全点体素标签检查、逐树文件读回验证。整批结束后再次核对原始 BIN 的 SHA-256、候选清单与实际文件数量、完整点云头部及预览图，全部通过，原始文件未修改。

最大文件的 TreeIso 阶段耗时 1,647.88 秒，合并阶段观测内存约 13 GB；整份文件约需 37 分钟。当前 TreeIso 子进程超时为 1,800 秒，本次未触发。

总览和侧视图已抽看。15-42-09 文件的 0002 号较稀疏、0126 号冠层横向延伸明显；16-24-51 文件的 0198 号呈直线、矩形轮廓，应优先检查是否为非树物体。这些是复核线索，未用人工真值计算分割精度，未标记候选也可能误分。

完整结果、`测试报告.md`、`测试汇总.json` 和 `最终核验.json` 位于：`D:\Datasets\人工林（平原林）\分割后的点云\自动测试_20261106_184957`。

## 源码验证

- 90 项回归测试通过，包含原有影像、点云、更新、线程生命周期测试。
- 15 项新增测试覆盖属性与点序保留、逐树完整分区、源文件变更与几何变更拒绝、缺失插件、取消、输出目录隔离、大坐标、CRS 记录及 GUI 路径持久化。
- 其中一项调用真实 CloudCompare：在中文及空格路径下，对包含 6 棵合成树的 41,400 点数据完成 CSF、TreeIso、标签回写、LAZ 导出和逐点核验，生成 6 个候选。这是功能验证，不是林地分割精度评测。
- 源码自检 21 项通过，包括 9 个工具窗口、LAS/LAZ/PLY 读写，以及外部 CloudCompare 插件检测。
- 默认窗口和 640×480 小窗口已渲染检查；运行控制、取消、进度和结果入口固定可见。
- `pip check` 通过。

## 打包验证

- 已生成本地可执行预览版：`.build/forestry-preview/RemoteSensingToolbox/RemoteSensingToolbox.exe`。
- 打包后的程序在清理外部 Python、Conda、GIS、Qt 路径的环境下完成 21 项自检，全部通过；包含新增库的实际 LAZ / PLY 读写和真实 CloudCompare 插件探测。
- 预览版需要保留整个 `RemoteSensingToolbox` 文件夹；内部版本号暂保留 1.3.1，不作为正式 Release。未替换已有 `dist/`、安装程序或 GitHub 发行包。
- 本机日志与机器可读报告保存在 `.build/forestry-all-tests.log`、`.build/forestry-self-test.json`、`.build/forestry-frozen-self-test.json`。

## 关键兼容处理

CloudCompare 的 `SAVE_CLOUDS FILE` 会再次解析文件名列表。输出采用相对于任务工作目录的固定文件名，避免包含空格的父目录被拆分。

计算坐标以源数据最小坐标为局部原点，不依赖 LAS 头是否正确设置全局偏移；导出仍使用原始整数坐标、比例和偏移。预览原样复制投影 VLR / EVLR，不重新解释坐标系。

外部进程使用参数数组，并隔离工具箱的 Qt / 打包运行库路径。后台线程沿用工具箱的 `bind_task` 生命周期管理。

## 复现

```powershell
$env:TOOLBOX_TEST_CLOUDCOMPARE = 'D:\apps\CloudCompare\CloudCompare.exe'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_toolbox.py --self-test --check-cloudcompare $env:TOOLBOX_TEST_CLOUDCOMPARE --report .build\forestry-self-test.json
```

不设置 `TOOLBOX_TEST_CLOUDCOMPARE` 时，真实引擎测试跳过，其他回归测试仍运行。单木分割需要 v1.4.0 或更新版本；v1.3.1 安装包不包含该功能。
