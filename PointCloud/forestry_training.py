"""Streaming, row-aligned training export and lossless TXT/NumPy conversion.

Only completed toolbox jobs or this module's manifest-backed datasets are accepted.
Automatic labels are always identified as pseudo-labels, never as ground truth.
"""
from contextlib import ExitStack, closing
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import laspy
import numpy as np

from PointCloud.cloudcompare import SegmentationCancelled


SCHEMA = "forest-training-1"
POINT_COLUMNS = ["x", "y", "z", "intensity", "red", "green", "blue"]
LABEL_COLUMNS = ["semantic_id", "instance_id", "source_index", "review_status",
                 "train_semantic", "train_instance", "train_mask"]
CHUNK_SIZE = 100_000
FORMATS = ("txt", "numpy", "both")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _check(cancelled):
    if cancelled():
        raise SegmentationCancelled("训练数据导出已取消；源文件未修改。")


def _hash(path, cancelled):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            _check(cancelled)
            result.update(block)
    return result.hexdigest()


def _training_labels(raw):
    valid = (raw[:, 0] > 0) & (raw[:, 3] == 1)
    semantic = np.where(valid, raw[:, 0] - 1, -1)
    instance = np.where(valid & (raw[:, 0] == 2), raw[:, 1] - 1, -1)
    return np.column_stack((raw, semantic, instance, valid.astype("i8"))).astype("<i8")


def _validate_rows(points, labels, offset):
    if points.ndim != 2 or points.shape[1] != 7 or labels.shape != (len(points), 7):
        raise ValueError("训练数据列数不符，应为 7 个点属性和 7 个标签字段。")
    if not np.isfinite(points).all() or not np.isfinite(labels).all():
        raise ValueError("训练数据包含非有限值。")
    if not np.array_equal(labels, labels.astype("i8")):
        raise ValueError("标签和原始点索引必须是整数。")
    features = points[:, 3:]
    if np.any((features < 0) | (features > 65535)) or not np.array_equal(features, features.astype("u2")):
        raise ValueError("强度和 RGB 必须保留 LAS 中的 0～65535 整数值。")
    semantic, instance, source, review = labels[:, :4].T
    if not np.isin(semantic, [0, 1, 2]).all() or not np.isin(review, [0, 1, 2]).all():
        raise ValueError("不支持的语义或复核状态编号。")
    if np.any((instance < 0) | (instance > np.iinfo("u4").max)) or not np.array_equal(instance > 0, semantic == 2):
        raise ValueError("单木编号与语义类别不对应。")
    if not np.array_equal(source, np.arange(offset, offset + len(points))):
        raise ValueError("原始点索引错序、重复或缺失，拒绝错位标签。")
    if not np.array_equal(labels, _training_labels(labels[:, :4])):
        raise ValueError("训练标签与原始标签/忽略规则不一致。")


def _job_info(root, cancelled):
    exported = _read_json(root / "export.json")
    cloud = root / "forest_instances.laz"
    if _hash(cloud, cancelled) != exported["full_cloud_sha256"]:
        raise ValueError("已标注点云与导出记录不符。")
    with laspy.open(cloud) as reader:
        header = reader.header
        if not {"semantic_id", "instance_id", "source_index", "review_status"}.issubset(header.point_format.dimension_names):
            raise ValueError("缺少逐点语义/实例标签或原始点索引。")
        if header.point_count != exported["source_point_count"]:
            raise ValueError("完整点云点数与导出记录不符。")
        info = {"schema": SCHEMA, "point_count": int(header.point_count),
                "source_sha256": exported["source_sha256"], "labeled_cloud_sha256": exported["full_cloud_sha256"],
                "label_provenance": "automatic_pseudo_labels_require_human_review",
                "coordinate_system": "original_world_coordinates_in_metres",
                "suggested_local_origin": header.mins.tolist(),
                "las_scales": header.scales.tolist(), "las_offsets": header.offsets.tolist(),
                "rgb_available": "red" in header.point_format.dimension_names,
                "point_columns": POINT_COLUMNS, "label_columns": LABEL_COLUMNS,
                "semantic_mapping": {"0": "unclassified", "1": "ground_candidate", "2": "tree_candidate"},
                "training_semantic_mapping": {"-1": "ignore", "0": "ground", "1": "tree"},
                "review_mapping": {"0": "unclassified", "1": "automatic_unreviewed", "2": "flagged_for_review"},
                "ignore_rule": "Unclassified or review_status != 1: train_semantic/train_instance = -1, train_mask = 0",
                "instance_rule": "For eligible trees train_instance = instance_id - 1; ground and ignored points = -1. IDs may have gaps.",
                "identity_scope": "Use (source_sha256, source_index) for point identity; tree IDs are local to each job.",
                "instance_map": [{"instance_id": row["tree_id"], "train_instance": row["tree_id"] - 1,
                                  "review_status": row["review_status"], "review_flags": row["review_flags"],
                                  "raw_treeiso_ids": row.get("raw_treeiso_ids", str(row["raw_treeiso_id"]))}
                                 for row in _read_json(root / "tree_inventory.json")]}
    return info


def _job_chunks(root, cancelled, chunk_size):
    with laspy.open(root / "forest_instances.laz") as reader:
        rgb = "red" in reader.header.point_format.dimension_names
        for record in reader.chunk_iterator(chunk_size):
            _check(cancelled)
            points = np.empty((len(record), 7), dtype="<f8")
            points[:, :3] = np.column_stack((record.x, record.y, record.z))
            points[:, 3] = record.intensity
            points[:, 4:] = np.column_stack((record.red, record.green, record.blue)) if rgb else 0
            raw = np.column_stack([record[n] for n in LABEL_COLUMNS[:4]]).astype("<i8")
            yield points, _training_labels(raw)


def _dataset_info(root, cancelled):
    info = _read_json(root / "manifest.json")
    if (info.get("schema") != SCHEMA or info.get("point_columns") != POINT_COLUMNS or
            info.get("label_columns") != LABEL_COLUMNS or info.get("label_provenance") != "automatic_pseudo_labels_require_human_review"):
        raise ValueError("不支持的训练数据格式；请选择本工具导出的数据目录。")
    expected = {"txt": {"points_labels.txt"}, "numpy": {"points.npy", "labels.npy"},
                "both": {"points_labels.txt", "points.npy", "labels.npy"}}.get(info.get("format"))
    if expected is None or set(info.get("files", {})) != expected:
        raise ValueError("训练数据文件清单不完整。")
    for name, digest in info["files"].items():
        if _hash(root / name, cancelled) != digest:
            raise ValueError("训练数据文件已变化，无法保证点与标签对齐：" + name)
    return info


def _dataset_chunks(root, info, cancelled, chunk_size, use_numpy=False):
    if info["format"] == "numpy" or use_numpy:
        points = np.load(root / "points.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
        try:
            if points.dtype != np.dtype("<f8") or labels.dtype != np.dtype("<i8"):
                raise ValueError("NumPy 数据类型必须为 float64 坐标属性和 int64 标签。")
            if points.shape != (info["point_count"], 7) or labels.shape != (info["point_count"], 7):
                raise ValueError("NumPy 点数组与标签数组长度不一致。")
            for start in range(0, len(points), chunk_size):
                _check(cancelled)
                yield points[start:start+chunk_size], labels[start:start+chunk_size]
        finally:
            points._mmap.close()
            labels._mmap.close()
    else:
        with (root / "points_labels.txt").open("r", encoding="utf-8") as stream:
            while True:
                _check(cancelled)
                # Fixed bounded batches, without loading the entire text file.
                lines = []
                for _ in range(chunk_size):
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        raise ValueError("训练 TXT 中出现空行。")
                    lines.append(line)
                if not lines:
                    break
                data = np.loadtxt(lines, dtype="<f8", comments=None, ndmin=2)
                if data.shape[1] != 14:
                    raise ValueError("训练 TXT 必须每行 14 列。")
                yield data[:, :7], data[:, 7:]


def _digests(chunks, count, cancelled, notify):
    p_hash, l_hash = hashlib.sha256(), hashlib.sha256()
    offset, usable = 0, 0
    with closing(chunks):
        for points, labels in chunks:
            _check(cancelled)
            _validate_rows(points, labels, offset)
            p_hash.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
            l_hash.update(np.asarray(labels, dtype="<i8").tobytes(order="C"))
            offset += len(points)
            usable += int(labels[:, 6].sum())
            notify({"stage": "validating_training", "points": offset, "total": count})
    if offset != count:
        raise ValueError("训练数据点数与清单不一致。")
    return {"points_sha256": p_hash.hexdigest(), "labels_sha256": l_hash.hexdigest(), "eligible_points": usable}


def export_training(source, output, format="txt", *, cancelled=lambda: False, notify=lambda data: None,
                    chunk_size=CHUNK_SIZE):
    """Export a completed forest job, or convert an intact training dataset.

    Output must be a new directory. Publication follows complete read-back checks;
    cancellation and failures discard only the private staging directory.
    """
    if format not in FORMATS or chunk_size <= 0:
        raise ValueError("训练格式应为 txt、numpy 或 both。")
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("输出目录已存在，请使用新的目录。")
    if not output.parent.is_dir():
        raise ValueError("输出父目录不存在。")
    _check(cancelled)
    is_job = (source / "export.json").is_file()
    info = _job_info(source, cancelled) if is_job else _dataset_info(source, cancelled)
    count = info["point_count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("训练数据点数无效。")
    estimate = count * ((240 if format in ("txt", "both") else 0) + (112 if format in ("numpy", "both") else 0))
    if shutil.disk_usage(output.parent).free < estimate:
        raise OSError(f"输出磁盘空间不足，预计至少需要 {estimate / 1024**3:.1f} GiB。")
    stage = output.parent / (".training-" + uuid.uuid4().hex)
    stage.mkdir()
    maps = []
    try:
        p_hash, l_hash = hashlib.sha256(), hashlib.sha256()
        offset, usable = 0, 0
        with ExitStack() as stack:
            stream = stack.enter_context((stage / "points_labels.txt").open("w", encoding="utf-8", newline="\n")) if format in ("txt", "both") else None
            if format in ("numpy", "both"):
                p_map = np.lib.format.open_memmap(stage / "points.npy", mode="w+", dtype="<f8", shape=(count, 7))
                maps.append(p_map)
                l_map = np.lib.format.open_memmap(stage / "labels.npy", mode="w+", dtype="<i8", shape=(count, 7))
                maps.append(l_map)
            chunks = _job_chunks(source, cancelled, chunk_size) if is_job else _dataset_chunks(source, info, cancelled, chunk_size)
            stack.callback(chunks.close)
            for points, labels in chunks:
                _validate_rows(points, labels, offset)
                if offset + len(points) > count:
                    raise ValueError("训练数据行数超出清单。")
                labels = np.asarray(labels, dtype="<i8")
                if stream:
                    np.savetxt(stream, np.column_stack((points, labels)), fmt=["%.17g"]*3+["%d"]*11)
                if maps:
                    p_map[offset:offset+len(points)] = points
                    l_map[offset:offset+len(points)] = labels
                p_hash.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
                l_hash.update(labels.tobytes(order="C"))
                offset += len(points)
                usable += int(labels[:, 6].sum())
                _check(cancelled)
                notify({"stage": "writing_training", "points": offset, "total": count})
        for mapped in maps:
            mapped.flush()
            mapped._mmap.close()
        maps.clear()
        if offset != count:
            raise ValueError("训练数据行数与源文件不一致。")
        logical = {"points_sha256": p_hash.hexdigest(), "labels_sha256": l_hash.hexdigest(), "eligible_points": usable}
        if not is_job and logical != info["logical_content"]:
            raise ValueError("转换前数据与原始逐点摘要不一致。")
        result = {**info, "format": format, "logical_content": logical}
        checked = _digests(_dataset_chunks(stage, result, cancelled, chunk_size), count, cancelled, notify)
        if checked != logical:
            raise ValueError("读回校验失败：点属性或标签发生变化。")
        if format == "both" and _digests(_dataset_chunks(stage, result, cancelled, chunk_size, True), count, cancelled, notify) != logical:
            raise ValueError("TXT 与 NumPy 点序/标签不一致。")
        # Detect concurrent modification before publishing a completed dataset.
        current = _job_info(source, cancelled) if is_job else _dataset_info(source, cancelled)
        if current != info:
            raise ValueError("导出期间源数据发生变化。")
        result["files"] = {p.name: _hash(p, cancelled) for p in stage.iterdir() if p.suffix in (".txt", ".npy")}
        result["verification"] = {"every_row_read_back": True, "source_order_preserved": True,
                                  "labels_and_features_same_order": True, "ignore_rules_checked": True,
                                  "original_world_coordinates_preserved": True}
        _save_json(stage / "manifest.json", result)
        (stage / "README.md").write_text(
            "# 单木实例分割训练数据\n\n自动伪标签，未经人工核验。原始点云和已有人工标注未修改。\n\n"
            "TXT 无表头，空格分隔，每行对应原始文件同一行；列顺序：\n\n`" + " ".join(POINT_COLUMNS + LABEL_COLUMNS) + "`\n\n"
            "NumPy：points.npy 为 N×7 float64（坐标、强度、RGB）；labels.npy 为 N×7 int64（后七列）。"
            "可用 np.load(..., mmap_mode='r') 分块读取。TXT/NumPy 保留相同坐标、属性、标签、点序。\n\n"
            "原始语义：0 未分类，1 地面候选，2 树木候选；实例 0 表示无树木编号，正整数与 Tree_XXXX.laz 一致。"
            "source_index 从 0 开始；复核状态 0 未分类、1 自动待核验、2 重点复核。\n\n"
            "训练语义：-1 忽略，0 地面，1 树木。train_instance 为原实例编号减 1，地面及忽略点为 -1；"
            "编号可能有空缺。train_mask=1 仅表示可作为伪标签参与训练，不表示人工审核通过。"
            "未分类和重点复核点保留原标签，但训练标签为 -1、mask 为 0。\n\n"
            "XYZ 为米制原始世界坐标。训练前先在 float64 下减 manifest.json 中 suggested_local_origin，再转 float32；"
            "不要直接将大坐标转 float32。强度和 RGB 保留 LAS 的 uint16 值域；无 RGB 时填 0，见 rgb_available。\n\n"
            "单木编号只在本任务内有效。跨场景使用 source_sha256 + source_index 标识点；"
            "instance_map 保存原实例/训练编号、TreeIso 编号及复核原因。数据集划分应按独立样地/采集批次进行。\n\n"
            "转换入口：软件的“点云训练数据转换”，选择完整分割结果目录或本训练目录。"
            "也可运行 python -m PointCloud.forestry_training 源目录 新输出目录 --format txt|numpy|both。"
            "完整清单和数据须一起保留。外部人工标注不能仅凭行数与这里的标签直接拼接，应先核对原始点索引及坐标。\n",
            encoding="utf-8")
        _check(cancelled)
        # Same-volume directory rename; existing destination is never overwritten.
        if output.exists():
            raise FileExistsError("输出目录已被创建。")
        stage.rename(output)
        notify({"stage": "training_complete", "points": count, "total": count})
        return {"directory": str(output), **result}
    finally:
        for mapped in maps:
            mapped._mmap.close()
        if stage.exists():
            if stage.resolve().parent != output.parent or not stage.name.startswith(".training-"):
                raise RuntimeError("临时目录超出本次输出位置，停止清理。")
            shutil.rmtree(stage)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="逐点对齐的单木 TXT / NumPy 训练数据导出与转换")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--format", choices=FORMATS, default="txt")
    args = parser.parse_args()
    exported = export_training(args.source, args.output, args.format)
    print(json.dumps({"directory": exported["directory"], "points": exported["point_count"],
                      "verification": exported["verification"]}, ensure_ascii=False, indent=2))
