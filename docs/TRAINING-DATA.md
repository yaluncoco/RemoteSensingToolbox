# 点云实例分割训练数据

v1.6.0 起，单木分割默认额外导出 `training/points_labels.txt`。窗口可选择 TXT、NumPy、两者同时导出或不导出训练数据。已有结果可从主界面的“点云训练数据转换”打开，选择完整分割目录或本工具生成的训练目录，再选择输出格式；转换无需重新运行 CloudCompare。

## TXT 字段

无表头、空格分隔、每点一行，顺序与完整标注点云 `forest_instances.laz` 及本次任务输入文件完全一致。重复坐标不会合并，未分类点不会删除。

```text
x y z intensity red green blue semantic_id instance_id source_index review_status train_semantic train_instance train_mask
```

| 列（从 1 开始） | 字段 | 含义 |
| --- | --- | --- |
| 1–3 | x / y / z | 原始世界坐标，单位米，17 位有效数字保留 float64 数值 |
| 4 | intensity | 原 LAS 强度值，未归一化 |
| 5–7 | red / green / blue | 原始 uint16 RGB 值域；无颜色时填 0，清单注明 `rgb_available=false` |
| 8 | semantic_id | 0 未分类、1 地面候选、2 树木候选 |
| 9 | instance_id | 0 无单木编号；正整数与 `trees/Tree_XXXX.laz` 对应 |
| 10 | source_index | 从 0 开始的原始行索引 |
| 11 | review_status | 0 未分类、1 自动待核验、2 重点复核 |
| 12 | train_semantic | -1 忽略、0 地面、1 树木 |
| 13 | train_instance | 可参与训练的树木取 `instance_id - 1`；地面和忽略点为 -1 |
| 14 | train_mask | 1 可作为伪标签使用，0 忽略 |

语义未分类或复核状态不为 1 时，原始标签仍保留，训练语义和实例设为 -1。重点复核包括可疑粘连、缺乏树干支持及校正后的冠部边界。`train_mask=1` **不代表人工审核通过**。自动分割标签始终标记为伪标签。

`manifest.json` 保存类别定义、原实例与训练编号映射、TreeIso 原编号、复核原因、坐标原点建议值、点数、源文件指纹、各输出文件和逐点内容的 SHA-256。实例编号只在本任务中有效，忽略后可能有空缺。若训练框架要求连续编号，对每个场景中非负 `train_instance` 做一次统一重编号，同时保留映射。

## NumPy 转换与读取

`points.npy` 为 N×7 float64，前七列点属性；`labels.npy` 为 N×7 int64，后七列标签。TXT ↔ NumPy 的坐标、属性、标签和点序完全一致。转换前核验文件指纹，转换后逐点读回检查，再发布新目录；发生取消或错误时不会留下成功清单。

分块读取示例：

```python
import json
from pathlib import Path
import numpy as np

root = Path("training")
meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
points = np.load(root / "points.npy", mmap_mode="r")
labels = np.load(root / "labels.npy", mmap_mode="r")
origin = np.asarray(meta["suggested_local_origin"], dtype=np.float64)

for start in range(0, len(points), 100_000):
    p = points[start:start + 100_000]
    y = labels[start:start + 100_000]
    xyz = (p[:, :3] - origin).astype(np.float32)
    semantic = y[:, 4]      # -1 / 0 / 1
    instance = y[:, 5]      # -1 或训练实例编号
    valid = y[:, 6].astype(bool)
    source_index = y[:, 2]
    # 下游裁块、筛选和打乱点序时，对 xyz、semantic、instance、valid、source_index 使用同一索引。
```

先在 float64 下减去统一原点，再转 float32，避免大坐标损失厘米级精度。训练框架中的 ignore_index 应与 -1 对应；不要把忽略点误当成地面。语义和实例监督使用相应掩码，地面不参与树木实例损失。

源码命令行入口：

```powershell
python -m PointCloud.forestry_training "D:\分割结果\forest_xxx" "D:\训练数据\scene01" --format txt
python -m PointCloud.forestry_training "D:\训练数据\scene01" "D:\训练数据\scene01_numpy" --format numpy
```

转换器支持本工具的完整清单格式。已有人工标注需先确认原始点索引和坐标，不能仅按点数拼接或用近邻把重复坐标混在一起。此版本没有自动匹配未知格式人工标注的功能。

## 数据身份和验证范围

用 `(source_sha256, source_index)` 识别一个点；不同场景的同号实例不是同一棵树。BIN 输入的索引对应经逐点验证的规范化 LAS/LAZ。裁剪测试中的索引对应 `buffered_roi.laz`，不是原始整片扫描的行号。

训练、验证和测试划分应按独立样地或采集批次进行；新数据目录中可能有重叠扫描，不能把同一林地的重叠点随机分到训练和验证集。具体测试数据和验证记录见 [v1.6.0 验证](VALIDATION-1.6.0.md)。
