# 单木分割（CloudCompare 后台流程）

在主界面选择 **点云处理 → 单木分割**。工具箱直接调用本机 CloudCompare 的命令行，普通使用不需要 MCP 服务、模型或桌面自动化。

本地源码启动：双击仓库根目录 `run_toolbox.bat`，优先使用 `.venv` 独立环境。测试记录见 [验证说明](VALIDATION-TREE-SEGMENTATION.md)。

修复 BIN 格式识别后的本地预览版位于 `.build/forestry-bin-fix/RemoteSensingToolbox/RemoteSensingToolbox.exe`，运行时需保留同目录的 `_internal` 文件夹。该预览版尚未发布为正式安装包。

## 使用

1. 选择 `CloudCompare.exe`，点击“检测插件”。路径会自动保存，下次打开恢复。
2. 选择 LAS / LAZ / BIN 文件和输出文件夹。按文件头识别内容，误写成 `.laz` 的 CloudCompare BIN 也可直接选择。
3. 确认 XYZ 坐标单位均为米，选择地形和采样间距。
4. 点击“开始分割”。任务在后台运行，可取消；关闭窗口时会等待任务安全退出。
5. 完成后打开结果文件夹，或点击“用 CloudCompare 查看结果”打开带实例颜色的采样预览。

路径查找顺序：界面保存的路径、`CLOUDCOMPARE_PATH` 环境变量、PATH、常见安装目录。明确指定的路径无效时会报错，不会偷偷切换到其他版本。用户可安装在任意目录，包括带空格和中文的路径。

CloudCompare 必须同时提供 `-CSF` 与 `-TREEISO` 命令。只有 `CloudCompare.exe` 文件存在还不够；程序每次运行前都会调用 `-HELP` 检测实际插件能力。CloudCompare 和插件需完整安装，工具箱不内置或自动下载该软件。

### 文件头为 CCB2 的“LAZ”

`Invalid file signature "b'CCB2'"` 表示内容是 CloudCompare BIN，扩展名与真实格式不一致。程序现在会识别 `LASF` 和 `CCB2/CCB3` 文件头；BIN 在任务目录中建立副本，使用 CloudCompare 导出 PLY，再转换为可验证的 LAZ 工作文件。不会修改、重命名或覆盖原始文件。包含多个点云或网格的 BIN 会提示先另存单个点云，避免只处理其中一部分。

BIN 的每个导出数值属性（包括双精度 XYZ、颜色、法线、标量字段）都保存在 LAZ 的 `cc000_…` 等额外字段中，原字段到新字段的对应关系记录在 `input.json`。常见颜色、强度、回波编号和 GPS 时间还会写入对应的 LAS 标准字段。LAS 整数 XYZ 通常使用 0.000001 米刻度，实际误差记录在 `input.json`；原始浮点 XYZ 另外保留，不因量化丢失。

BIN 转换会逐点比较 CloudCompare PLY 导出的全部数值属性，再进行后续分割。验证范围是点云属性，不涵盖 BIN 场景层级、传感器对象等元数据；程序不会推断 CRS。`input-conversion/normalized.laz` 和转换日志保留供检查。

## 流程与默认参数

体素代表点采样 → CSF 地面过滤 → 离地高度与稀疏点筛选 → TreeIso 三阶段分割 → 筛选、编号、复核标志 → 标签回写全部原始点 → 逐树导出与读回验证。

| 参数 | 默认值 |
| --- | --- |
| 采样间距 | 0.2 米，可调 0.05～0.5 米 |
| 地形 | 平坦，可选起伏、陡坡 |
| 最低候选树高 | 离地高度 P95 ≥ 2 米 |
| 最少点数 | 每组至少 100 个采样代表点 |
| CSF | 网格 0.5 米，分类阈值 0.3 米，500 次迭代 |
| 植被范围 | 离地 0.35～40 米，并通过邻域密度与地面距离筛选 |
| TreeIso | λ1=1、K1=5；λ2=20、K2=20、max_gap=2；ρ=0.5、vertical_overlap_weight=0.5 |

命令参数数组通过子进程直接传入，不经过 shell 拼接。运行使用所选安装的 DLL / Qt 环境；取消会结束正在运行的 CloudCompare 子进程。Python 数值计算会在当前计算步骤结束后响应取消。

## 输出

每次运行建立独立的 `forest_时间_标识` 子目录，原始文件不会被覆盖。

| 文件 | 内容 |
| --- | --- |
| `forest_instances.laz` | 全部原始点及原始属性，增加语义、实例和复核字段 |
| `instance_preview.laz` | 原始代表点，按单木编号着色；这是采样预览 |
| `trees/Tree_0001.laz` 等 | 每个候选的完整分辨率点云；可取消勾选 |
| `tree_inventory.csv` | 编号、位置、估计高度、水平范围、点数和复核原因 |
| `instance_overview.png` / `tree_gallery.png` | 总览和部分候选侧视图 |
| `engine.json`、`parameters.json`、`*-run.json` | 程序路径、可用命令和运行参数 |
| `input.json`、`input-conversion/` | 实际输入格式、BIN 转换记录、字段映射及规范化工作点云 |
| `result.json` | 整个任务成功完成的记录；失败或取消不会生成 |
| `progress.json` / `*-job/*.log` | 处理阶段、错误信息和 CloudCompare 日志 |

新增字段：`semantic_id`（0 未分类、1 地面候选、2 树木候选）、`instance_id`（0 无单木编号）、`review_status`（0 未分类、1 自动未复核、2 重点复核）、`source_index`（原始点行号）。原 LAS classification、颜色、强度、GPS 时间等字段保持原值，不会用候选类别替换原有分类。

导出会逐点比较原始属性、坐标比例与偏移、点序和体素标签；每棵树写出后再次读回验证。失败或取消的目录会保留诊断和中间文件，下次运行建立新目录。大点云需为完整 LAZ、临时 LAS、逐树文件和工作样本预留磁盘与内存。

## 适用范围

复用本地 CloudCompare MCP 工作区已经运行过的森林流程。支持 LAS / LAZ 及仅含单个点云的新版 CloudCompare BIN，不支持波形点格式；坐标单位由用户确认。TreeIso 更适合具有树干和三维枝冠结构的地面/移动扫描林地点云，纯航测冠顶点云需另行评估参数与方法。

所有结果都是**待复核的单木候选**。原始坐标精度得以保留，但边界仍取决于采样间距；近地树基、树冠接触、低密度区域可能有遗漏、过分割或合并。数量不可直接当作已核实的株数。

## 开发接口

核心入口是 `PointCloud.forestry_workflow.run_workflow`，与 PyQt5 和 MCP 无关：

```python
from PointCloud.forestry_workflow import run_workflow

result = run_workflow(
    source=r"E:\点云\forest.laz",
    output_parent=r"E:\结果",
    binary=r"D:\apps\CloudCompare\CloudCompare.exe",
    coordinates_in_metres=True,
    resolution=0.2,
    terrain="FLAT",
    extract_trees=True,
)
```

后续 MCP 工具可以调用此接口，不需要通过鼠标键盘操作界面。

上游说明：CloudCompare [命令行](https://www.cloudcompare.org/doc/wiki/index.php/Command_line_mode)、[TreeIso 插件](https://github.com/truebelief/cc-treeiso-plugin)。
