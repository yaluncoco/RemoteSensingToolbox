# 单木分割（CloudCompare 后台流程）

在主界面选择 **点云处理 → 林业功能 → 01 单木分割**。工具箱直接调用本机 CloudCompare 的命令行，普通使用不需要 MCP 服务、模型或桌面自动化。

本地源码启动：双击仓库根目录 `run_toolbox.bat`，优先使用 `.venv` 独立环境。测试记录见 [验证说明](VALIDATION-TREE-SEGMENTATION.md)。

v1.4.0 起，正式安装包和便携版均包含该功能及 BIN 格式识别修复。便携版需完整解压，运行时保留同目录的 `_internal` 文件夹。早期 `.build/forestry-bin-fix/` 仅为本地预览构建。

## 使用

1. 选择 `CloudCompare.exe`，点击“检测插件”。路径会自动保存，下次打开恢复。
2. 选择 LAS / LAZ / BIN 文件和输出文件夹。按文件头识别内容，误写成 `.laz` 的 CloudCompare BIN 也可直接选择。
3. 确认 XYZ 坐标单位均为米，选择地形和采样间距。默认开启“树干约束校正”，处理相邻树粘连和同树多编号；关闭可生成原始 TreeIso 对照结果。
4. 点击“开始分割”。任务在后台运行，可取消；关闭窗口时会等待任务安全退出。
5. 完成后打开结果文件夹，或点击“用 CloudCompare 查看结果”打开带实例颜色的采样预览。
6. 点击“02 · 提取林业参数”或“03 · 导出 / 转换训练数据”，直接带入本次完整分割结果。参数提取使用原实例编号与地面分类；训练导出无需先提取参数。正在运行的工具会提示先完成任务，保留当前输入。

路径查找顺序：界面保存的路径、`CLOUDCOMPARE_PATH` 环境变量、PATH、常见安装目录。明确指定的路径无效时会报错，不会偷偷切换到其他版本。用户可安装在任意目录，包括带空格和中文的路径。

CloudCompare 必须同时提供 `-CSF` 与 `-TREEISO` 命令。只有 `CloudCompare.exe` 文件存在还不够；程序每次运行前都会调用 `-HELP` 检测实际插件能力。CloudCompare 和插件需完整安装，工具箱不内置或自动下载该软件。

### 文件头为 CCB2 的“LAZ”

`Invalid file signature "b'CCB2'"` 表示内容是 CloudCompare BIN，扩展名与真实格式不一致。程序现在会识别 `LASF` 和 `CCB2/CCB3` 文件头；BIN 在任务目录中建立副本，使用 CloudCompare 导出 PLY，再转换为可验证的 LAZ 工作文件。不会修改、重命名或覆盖原始文件。包含多个点云或网格的 BIN 会提示先另存单个点云，避免只处理其中一部分。

BIN 的每个导出数值属性（包括双精度 XYZ、颜色、法线、标量字段）都保存在 LAZ 的 `cc000_…` 等额外字段中，原字段到新字段的对应关系记录在 `input.json`。常见颜色、强度、回波编号和 GPS 时间还会写入对应的 LAS 标准字段。LAS 整数 XYZ 通常使用 0.000001 米刻度，实际误差记录在 `input.json`；原始浮点 XYZ 另外保留，不因量化丢失。

BIN 转换会逐点比较 CloudCompare PLY 导出的全部数值属性，再进行后续分割。验证范围是点云属性，不涵盖 BIN 场景层级、传感器对象等元数据；程序不会推断 CRS。`input-conversion/normalized.laz` 和转换日志保留供检查。

## 流程与默认参数

体素代表点采样 → CSF 地面过滤 → 离地高度与稀疏点筛选 → TreeIso 三阶段分割 → 多切片树干约束校正 → 筛选、编号、复核标志 → 标签回写全部原始点 → 逐树导出与读回验证。

| 参数 | 默认值 |
| --- | --- |
| 采样间距 | 0.2 米，可调 0.05～0.5 米 |
| 地形 | 平坦，可选起伏、陡坡 |
| 最低候选树高 | 离地高度 P95 ≥ 2 米 |
| 最少点数 | 每组至少 100 个采样代表点 |
| CSF | 网格 0.5 米，分类阈值 0.3 米，500 次迭代 |
| 植被范围 | 离地 0.35～40 米，并通过邻域密度与地面距离筛选 |
| TreeIso | λ1=1、K1=5；λ2=20、K2=20、max_gap=2；ρ=0.5、vertical_overlap_weight=0.5 |
| 树干约束校正 | 默认开启；算法版本 `stem-graph-2` |

### v1.6.0 的粘连与碎片校正

树干检测重新读取原始点云的离地 0.4～2.6 米高度带，按 2.5 厘米体素保留点。树冠仍使用界面设置的采样间距，因此小树干不会只剩 20 厘米采样后的少量点。

- 在 0.70、1.25、1.80、2.35 米处取 22 厘米厚切片。圆拟合至少需要 12 点、65% 内点和 150° 圆弧覆盖；至少三个切片的中心、半径和倾斜轴一致才建立树干种子。圆拟合只提供分割约束，不作为胸径测量报告。
- 一个 TreeIso 块内有多个可靠树干时，在块内建立三维邻接图，从各树干传播最短路径，拆开粘连块。路径对长距离空隙增加代价；允许倾斜树干，保留三维枝冠连接信息。
- 多个块共享同一可靠树干时合并编号。没有树干的树冠碎片，仅在其下部与已有树木具有足够三维接触、且至少 85% 接触支持同一归属时合并。相近的竞争归属会弃权，不靠水平最近树干强行分配。
- 具有独立低位结构但树干拟合不足的块，以及归属不明或远离树干的块，保留独立候选并标记复核。纯冠顶、遮挡严重、低位分叉和稀疏点云仍可能有误分；校正不等于人工确认。
- 少量树干点混入大块低矮植被时，不据此整块合并：若无树干支持的低位点占该块至少一半且不少于 50 点，而已确认树干仅支持低位点的不到 10%，保留该块并标记 `mixed_low_vegetation`。这项保护来自新整理数据的局部检查。

采样间距默认 0.2 米。图连接距离为 `max(0.65 米, 2.5 × 采样间距)`。更粗的采样会损失枝冠细节，不能靠扩大连接距离恢复。每次新运行建立新目录；旧版缓存不能冒充已完成校正的结果，需重新运行。

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
| `refinement_comparison.png` | 同一视角的校正前后对照，黑色十字为通过几何检查的树干 |
| `refinement.json` | 树干轴、拆分与合并关系、接触支持和未解决组；仅开启校正时生成 |
| `refinement_labels.npz` / `sample_labels.npz` | 校正前后标签与原始采样索引；`raw_segment_id` 保留 TreeIso 原始编号加 1 |
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
    refine=True,
    extract_trees=True,
    training_format="txt",  # txt / numpy / both / none
)
```

后续 MCP 工具可以调用此接口，不需要通过鼠标键盘操作界面。

## 实例分割训练标签

默认额外生成 `training/points_labels.txt`，每点一行，坐标、原始点索引、语义及实例标签保持逐点对应。窗口可以选择 TXT、NumPy、两者同时输出或关闭训练导出。重点复核点保留原始编号，训练列设为 -1，并提供掩码。

已有分割结果可在主界面的“点云训练数据转换”中导出或转换，无需重新分割。输出是自动伪标签；字段顺序、类别/实例映射、读取方式及大坐标处理见 [训练数据格式](TRAINING-DATA.md)。

上游说明：CloudCompare [命令行](https://www.cloudcompare.org/doc/wiki/index.php/Command_line_mode)、[TreeIso 插件](https://github.com/truebelief/cc-treeiso-plugin)。
