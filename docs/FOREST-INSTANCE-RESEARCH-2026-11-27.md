# 手持 SLAM 森林点云实例分割：模型基线与数据组织

检索与代码核对日期：2026-11-27。针对手持 SLAM、场景级单木实例分割。本文是围绕工程问题的定向文献与作者代码分析，不是穷尽所有论文的系统综述。模型尚未在本项目数据上训练，下面的训练参数建议不是测得的最优值。

## 结论

推荐以 **TreeLearn + ForestFormer3D** 作为两个主要深度学习对照，以现有 **TreeIso / TreeIso 加树干约束校正** 作为几何方法对照。需要覆盖 2026 年进展时，增加 ForestMamba；SegmentAnyTreeV2 和 ForPT 适合后续研究，不宜只凭摘要替代已有可复现基线。

数据组织是“完整场景/样地统一标注与分组 → 按样地划分训练、验证、测试 → 在各自集合内裁出多个局部样本 → 分块推理并拼接到整片”。一个 LAZ 文件不等于一个 GPU 样本；一个训练块也通常不是一棵已分好的树。TreeLearn 和 ForestFormer3D 的公开训练代码分别采用预生成裁块和在线随机圆柱裁块。[1–4]

## 基线选择与研究状态

| 方法 | 核对到的状态 | 对手持 SLAM 的作用 |
| --- | --- | --- |
| TreeLearn | Ecological Informatics 2024 年 12 月，文章号 102888；作者代码、权重与训练指南公开 | 原本面向地面 LiDAR；用稀疏卷积预测语义和树基部偏移，再聚类。作为移动扫描的主基线很合适。[1,2] |
| ForestFormer3D | ICCV 2025，作者仓库标注 Oral；代码和预训练模型公开 | 面向航空、地面等点云的查询/掩码模型，同时处理语义与树实例；作为较新的统一强基线。[3,4] |
| SegmentAnyTree | Remote Sensing of Environment 2024，114367 | 传感器与平台泛化的已有对照，可复用预训练模型评估跨域效果；名称不表示直接调用二维 SAM。[5] |
| ForestMamba | 2026-06-01 预印本，检索到的修订日期为 2026-08-19；作者仓库报告 BMVC 2026 接收，并提供代码与权重 | 稀疏 Mamba 和冠层高度引导查询，可作 2026 年增强对照。当前公开权重与论文权重训练设置存在差异，实验应固定版本。[6] |
| SegmentAnyTreeV2 | 2026-06-06 预印本，2026-06-17 修订；本次未确认正式期刊/会议接收 | 改进非树/树分类、内部小片段和边界预测融合。适合研究碎片、边界和跨平台泛化；不把预印本视为已完成同行评审。[7] |
| ForPT | 2026-09-21 预印本 | 在 ALS/ULS/MLS 上自监督预训练 LitePT。适合少标注微调；已查到的仓库尚没有完整实例分割下载与配置，不能把“骨干已公开”等同于“实例分割流程已完全可复现”。[8] |
| SCATR-ITS | 2026-11-19 预印本及公开仓库 | 着重大面积推理速度与内存，包含国家级航空点云实验。可借鉴分块推理，但不能把航空扫描速度直接套到高密度手持 SLAM。[9] |

不同论文的传感器、数据集、标注、测试范围和指标不同，不应把其摘要中的分数拼成一个统一排行榜。“较新”不保证在本地手持 SLAM 上更好。针对 TLS 的 2026 年独立基准比较也表明森林结构会影响模型表现，需在一致条件下检验迁移效果。[10]

## 作者训练代码实际怎样读取数据

### TreeLearn：预先生成训练块

核对版本：`ecker-lab/TreeLearn@fd240ce7caa4c444fe3418aca454dc578bc557d4`。

- `configs/data_gen/gen_train_data.yaml`：从多个森林文件生成随机裁块；默认 `chunk_size=35`，即 XY 上 35×35 米；`n_samples_total=25000` 是作者示例配置，不能解释为任何新数据都必须生成 25,000 块。
- `configs/_modular/sample_generation.yaml`：默认预降采样体素边长 0.1 米。
- `configs/_modular/dataset_train.yaml`：训练目录是 `random_crops/npz`，中心监督区边长 8 米。
- `tree_learn/dataset/dataset.py`：一次 `__getitem__` 打开一个 NPZ，读取 `points`、`feat`、`instance_label`，产生语义、偏移目标及有效监督掩码。外围点用于上下文，中心区参与主要损失。
- 原始标签约定为 `-1=忽略，0=非树，正整数=树实例`。当前代码使用树的低位点计算基部位置；文件内明确注释其实现与论文描述有差异，复现实验需固定代码版本。[2]

因此，不能把软件的 14 列 TXT 原封不动交给 TreeLearn；需要选择正确字段、映射标签、生成特征和 NPZ。尤其不能默认“最后一列是实例编号”，因为本软件最后一列是 `train_mask`。

### ForestFormer3D：读取场景后在线裁圆柱

核对版本：`SmartForest-no/ForestFormer3D@6a75c3735e4a4108d02ee944a8b93177f2360a4f`。

`configs/oneformer3d_qs_radius16_qp300_2many.py` 的训练流水线是：

```text
读取 XYZ → 读取逐点语义/实例标签
→ 随机圆柱裁剪 radius=16 m
→ 0.2 m GridSample
→ 最多抽取 640,000 点
→ 实例标签映射 → 数据增强 → batch（默认 2）
```

圆柱只用 XY 距离判断成员，Z 不作高度截断，所以直径为 32 米、保留该位置的可见全高。`CylinderCrop` 同步裁剪坐标与标签，保存实例保留比例，并由裁剪前的实例计算投票中心；`PointSample_` 同步更新字段。预处理先转成数值数组及训练索引，训练并非每个 epoch 反复解析原始 LAZ/TXT。该配置只使用 XYZ，并不要求 RGB。[4]

作者默认语义为 ground/wood/leaf。本项目当前只有地面候选和树候选，没有可靠的木/叶标注。可先适配为地面/树两类，并相应修改模型、损失及评估映射；不能把全部树点假设成 wood 或 leaf 来伪造三类真值。

### 2026 年的流程是否改成整片训练

没有从这些已核对来源得到这样的结论。SegmentAnyTreeV2 的预处理仍使用半径 16 米、保持全高的圆柱，0.2 米网格，每柱最多 650,000 点；整片结果依靠局部预测合并。ForPT 是预训练策略的发展，也不能据此推导出任意亿点样地可直接作为一个 GPU 输入。[7,8]

ForPT 对 FOR-instanceV2 的描述给出了具体例子：48 个训练样地、17 个验证样地、29 个测试样地，共 11,028 个树实例。实例分割推理继承 ForestFormer3D 的重叠圆柱与按置信度合并流程。这里的 48 个训练样地并不等于 48 个固定 GPU 样本。[8]

不要混淆三种“patch”：原始地理样区、网络输入的局部点云、注意力/序列化算子内部的 token 分组。论文内部的 attention patch 数量不一定是地面米制裁块尺寸。

## 针对本项目数据的建议方案（待实测）

### 标注和集合划分

用户确认数据来自手持 SLAM，应该没有重复扫描。当前 16 个文件应先建立 `scene_id` 与实际样地的对应记录；“不同文件”和“地理上独立”仍不是同一个判定。时间戳不同也不能保证空间独立，局部坐标原点相同也不能证明重叠。

1. 在连续场景或选定连续子样区内统一编号：同一棵树所有可见的干、枝、叶属于同一实例；跨训练裁块保留同一个场景身份。多干树、倒木、灌木及严重缺失树的标注规则应提前统一。
2. 没必要先把 13 亿余点全部人工完成后才开始训练。可先选覆盖不同树密度、树径、地形和遮挡条件的连续样区，完整校正其中心区及缓冲区内可见对象；不确定的点保留为忽略。
3. 如果确认是 16 个独立样地，可将 10/3/3 作为训练/验证/测试的初始分组建议，再检查结构与难度分布。最终比例取决于独立地点数、树数和森林类型；这不是本项目已经执行的划分。
4. 先划分样地，再生成裁块。同一样地、空间相邻或重叠扫描归同一组。固定独立测试样地，模型与参数选择只看训练/验证；训练集较小时，可在开发集合内部做分组交叉验证。

自监督预训练与伪标签扩充也遵循这个分组：若目标是评估对未知样地的泛化，不把测试样地的未标注点混入本地预训练。确实使用测试域未标注数据时，应另列为传导式实验，不能与完全未见测试域的结果混写。

仅把同一森林中的随机点或重叠 patch 分到不同集合，会使相同树木与邻域出现在训练和测试中。单木实例号应在场景中稳定；输入模型前可以把当前 patch 的树 ID 重排为连续整数，同时保存返回场景 ID 的映射。[2,4]

### 建议的第一轮训练

| 项目 | 本项目建议 | 性质 |
| --- | --- | --- |
| 模型 | TreeIso、当前校正方法、TreeLearn、ForestFormer3D | 控制传统方法、现有改进、两个深度模型 |
| 起始参数 | 先复现各作者默认裁块、体素设置，再比较统一分辨率下的结果 | 不把自定参数冒充论文参数 |
| 后续裁块消融 | XY 24/32/40 米全高方块，或对应半径 12/16/20 米圆柱 | 待按冠幅、密度、显存验证 |
| 后续体素消融 | 0.1/0.2 米 | 比较细结构保留、速度、内存；不是树径估计的原始数据精度 |
| batch | 显存未知时先 1，再实测提高；保留点数上限 | 防止按原始亿点规模估计 GPU 可行性 |
| 输入 | 先 XYZ；有标定一致的 intensity 时再做加入强度的消融 | 不把不同采集的强度尺度直接混合 |
| 标签 | 人工纠正的实例标签为主要监督；自动结果单独标记为伪标签 | 伪标签扩充和真实标签微调分别报告 |
| 训练采样 | 在场景/网格中采中心，控制各样地贡献；检查密集、遮挡、边界难例覆盖 | 避免大文件和高密度区域支配样本 |

手持扫描的全高裁块应包含相邻树与枝冠竞争区域。如果先把树逐棵分好再把单树当作实例分割训练场景，模型没有学习在粘连邻树间确定归属的主要任务；这种单树输入更适合树种、属性或单树结构任务。

### 边界树与整片拼接

裁块不能重新定义“生物学上的一棵树”。边缘剩下的一截树冠不应因为局部几何中心改变，就被人工重新编号为新树。推荐保留完整场景的树编号、中心/树基部以及 crop 对该树的保留比例，按模型原有监督策略处理部分实例。TreeLearn 用中心监督掩码；ForestFormer3D 的裁剪变换保存完整实例信息，因此不宜先在外部将每块误当作独立整树场景再计算目标。[2,4]

推理时在 XY 上重叠裁块，提供外围上下文；结合重叠点、掩码相似度和可靠树干证据匹配实例，再赋予全场景唯一编号。不能直接拼接各 patch 的局部 ID，也不能只按树冠质心距离合并，后者会混淆密集邻树。具体采用中心区保留、非极大值抑制还是加权掩码融合，应遵循所复现模型的实现，再作消融。

现有 30×30 米测试样区、20×20 米中心区、5 米缓冲，是软件验证配置。训练时不保证 5 米对所有冠幅都足够；应依据实际树冠延伸和邻域验证增加缓冲。完整场景的测试分数需要先完成跨块拼接，再与人工实例匹配。

这些 30 米样区也不能提供一个完整 35 米训练块的全部上下文。它们可先验证数据读取和标签映射；复现 TreeLearn 的 35 米配置时，应从更大的原始连续场景生成训练块，并安排相应的人工标注范围。

### 标签对齐与格式转换

软件输出列详见 [TRAINING-DATA.md](TRAINING-DATA.md)。保留原始点的身份 `(source_sha256, source_index)`，坐标、特征、标签与掩码必须共同使用同一组采样索引。不要用浮点 XYZ 字符串连接标注表，重复坐标会导致歧义。

建议持久化下列层次：

```text
原始场景：原始坐标 / source_index / 语义 / 场景树编号 / 标注来源
场景索引：scene_id / plot_group / train-val-test / 坐标原点
训练缓存：局部坐标 / 特征 / 连续实例标签 / loss_mask / 原始索引映射
```

TXT 用于交换和人工检查；训练前转换为 NPZ/NPY、框架要求的二进制点文件与标签文件。原始世界坐标保留 float64，转局部坐标后再供网络使用 float32。体素化不应平均树木 ID；记录选中代表点或明确定义的标签聚合规则，并保存反向映射以恢复全分辨率结果。

本软件 `train_mask=1` 只说明通过当前自动筛选，不说明人工确认。已有“标记复核”与未知点的训练标签应忽略；需要新记录人工校正的来源与版本。用于论文最终测试的标签应独立人工检查，不能用同一算法生成的伪标签再证明该算法准确。

TreeLearn 适配示意：

```python
# 以下是规则示意，不是已经加入软件的框架适配器。
labels = np.full(n_points, -1, dtype=np.int64)
valid = train_mask.astype(bool)
labels[valid & (train_semantic == 0)] = 0  # 当前有效的非树标签；仍需区分伪标签与人工真值
tree = valid & (train_semantic == 1) & (train_instance >= 0)
scene_ids, local_ids = np.unique(train_instance[tree], return_inverse=True)
labels[tree] = local_ids + 1
# 保存 local_id -> scene_id；裁点、增强、降采样同步处理全部字段。
```

这一步只转换既有标签语义，不会自动将未知外部人工 TXT 对齐到原始点云。目前软件的 TXT/NumPy 互转也不等于已经实现所有模型的数据适配。

## 评价如何覆盖粘连与过分割

建议在固定独立样地的完整拼接结果上，使用预先固定的树实例匹配与 IoU 阈值，报告检测 Precision/Recall/F1、实例 IoU，并在采用 AP 时说明阈值集合和实现。另报告漏树、多个真树并为一实例、一真树拆成多个实例，以及边界/林下小树/密集邻树子集的结果。对匹配树的平均 IoU 要同时给出漏检数，不能只展示成功匹配对象。

训练与测试可以先在统一体素尺度评估，并保留原始分辨率的辅助核查；否则不同扫描密度会改变逐点指标的权重。效率报告应包含硬件、体素尺度、覆盖面积和点数，区分模型前向时间与预处理、拼接、回投影总时间。当前软件的结构检查、源数据不变核验与裁剪范围稳定性，都不能替代这些有人工真值的精度评价。

## 主要来源与可复核代码

以下链接为作者论文、作者仓库、出版方或数据发布页；没有把二手介绍用作参数依据。

1. Henrich et al. *TreeLearn: A deep learning method for segmenting individual trees from ground-based LiDAR forest point clouds*. Ecological Informatics, 2024, 102888. `https://doi.org/10.1016/j.ecoinf.2024.102888`。
2. TreeLearn 作者代码与训练指南：`https://github.com/ecker-lab/TreeLearn`。核对文件：`configs/data_gen/gen_train_data.yaml`、`configs/_modular/sample_generation.yaml`、`configs/_modular/dataset_train.yaml`、`tree_learn/dataset/dataset.py`、`docs/training.md`。
3. Xiang et al. *ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds*. ICCV 2025. `https://arxiv.org/abs/2506.16991`。
4. ForestFormer3D 作者代码：`https://github.com/SmartForest-no/ForestFormer3D`。核对文件：`configs/oneformer3d_qs_radius16_qp300_2many.py`、`oneformer3d/transforms_3d.py`、`data/ForAINetV2/load_forainetv2_data.py`、`data/ForAINetV2/batch_load_ForAINetV2_data.py`。数据与模型：`https://zenodo.org/records/16742708`。
5. Wielgosz et al. *SegmentAnyTree: A sensor and platform agnostic deep learning model for tree segmentation using laser scanning data*. Remote Sensing of Environment 2024, 114367. `https://arxiv.org/abs/2401.15739`；`https://doi.org/10.1016/j.rse.2024.114367`；代码 `https://github.com/SmartForest-no/SegmentAnyTree`。
6. Nguyen et al. *ForestMamba: Sparse Mamba with Geometry-guided Queries for 3D Forest Point Cloud Segmentation*. `https://arxiv.org/abs/2606.01549`；作者代码与接收公告 `https://github.com/thanhhff/ForestMamba`。核对提交 `f0c5239d0d83e95f56508cdc20287f025a0b98db`。
7. Wielgosz et al. *SegmentAnyTreeV2: Scaling Transformer-Based Tree Instance Segmentation Across Sensors, Platforms, and Forests*. `https://arxiv.org/abs/2606.08206`；预处理与拼接细节见 v2 的 3.2 节及方法部分。
8. Yue et al. *Toward A Foundation Model for Forest Point Clouds*. `https://arxiv.org/abs/2609.24787`；作者代码 `https://github.com/prs-eth/ForPT`。核对提交 `8304f57620fdffc35ac9a203ee844a015a875869`；README 中实例数据下载仍为占位，公开配置目录未发现完整的实例分割分支。
9. Ruoppa et al. *SCATR-ITS: a scalable transformer for large-area individual tree segmentation from LiDAR point clouds*. `https://arxiv.org/abs/2611.26715`；`https://github.com/ruoppa/scatr_its`。核对提交 `b58cd5264453ead87ef0508c98097e9b82939660`。
10. Cherlet et al. *Benchmarking tree instance segmentation of terrestrial laser scanning point clouds*. ISPRS Journal of Photogrammetry and Remote Sensing, 2026. `https://doi.org/10.1016/j.isprsjprs.2025.10.033`。

本页只给出复现与实验组织建议，未安装上述训练框架、下载大规模外部数据、启动 GPU 训练，亦未将伪标签声明为人工真值。
