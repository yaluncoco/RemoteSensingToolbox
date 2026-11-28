"""Portable, bounded previews and lossless instance exports for forestry results."""
from contextlib import ExitStack
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

import laspy
import numpy as np

from PointCloud.cloudcompare import SegmentationCancelled

CHUNK = 500_000
PREVIEW_LIMIT = 200_000


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(path, check):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            check()
            digest.update(block)
    return digest.hexdigest()


def _read_rows(root):
    with (root / "林业参数.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [int(r["TreeID"]) for r in rows]
    if len(ids) != len(set(ids)) or any(i <= 0 for i in ids):
        raise ValueError("参数表中有重复或无效的单木编号。")
    return rows


def save_scene(root, source, xyz, labels, heights, origin, rows, summary, check):
    """Persist sampling only for display; retain the full label map for export."""
    root, source = Path(root), Path(source)
    np.save(root / "instance_labels.npy", labels, allow_pickle=False)
    # The stable grouping retains source order and samples every measured object.
    order = np.argsort(labels, kind="stable")
    ids, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    per_tree = max(1, min(6000, (PREVIEW_LIMIT - 15000) // max(1, len(rows))))
    indices = []
    for tree_id, start, count in zip(ids, starts, counts):
        check()
        take = min(int(count), 15000 if tree_id == 0 else per_tree)
        indices.append(order[start + np.linspace(0, count - 1, take, dtype=np.int64)])
    selected = np.sort(np.concatenate(indices))
    rgb = np.zeros((len(selected), 3), dtype=np.uint8)
    has_rgb = False
    with laspy.open(source) as reader:
        has_rgb = all(n in reader.header.point_format.dimension_names for n in ("red", "green", "blue"))
        if has_rgb:
            cursor = 0
            colors = np.zeros((len(selected), 3), dtype=np.uint16)
            for chunk in reader.chunk_iterator(CHUNK):
                check()
                lo, hi = np.searchsorted(selected, [cursor, cursor + len(chunk)])
                indices = selected[lo:hi] - cursor
                colors[lo:hi] = np.column_stack((chunk.red, chunk.green, chunk.blue))[indices]
                cursor += len(chunk)
            rgb = np.clip(colors / (256 if colors.max(initial=0) > 255 else 1), 0, 255).astype("u1")
    np.savez_compressed(root / "scene_preview.npz", xyz=np.asarray(xyz[selected], dtype="f4"),
                        ids=np.asarray(labels[selected], dtype="i8"), height=np.asarray(heights[selected]),
                        origin=np.asarray(origin, dtype="f8"), rgb=rgb, has_rgb=np.array(has_rgb))
    cloud = root / "参数标注.laz" if (root / "参数标注.laz").is_file() else source
    try:
        cloud_path = str(cloud.relative_to(root))
    except ValueError:
        cloud_path = str(cloud.resolve())
    manifest = dict(schema=1, cloud=cloud_path, cloud_sha256=_hash(cloud, check),
                    labels="instance_labels.npy", labels_sha256=_hash(root / "instance_labels.npy", check),
                    preview="scene_preview.npz", preview_sha256=_hash(root / "scene_preview.npz", check),
                    point_count=len(labels), label_field=summary.get("export_fields", {}).get("metric_tree_id"),
                    height_field=summary.get("export_fields", {}).get("metric_height"),
                    preview_points=len(selected), origin=np.asarray(origin).tolist())
    _json(root / "scene.json", manifest)
    summary["scene"] = {"preview_points": len(selected), "full_resolution_points": len(labels)}


def load_scene(directory, cancelled=lambda: False, notify=lambda data: None):
    root = Path(directory).resolve(strict=True)
    def check():
        if cancelled():
            raise SegmentationCancelled("已取消点云加载。")
    check()
    if not (root / "result.json").is_file():
        raise ValueError("请选择已完成的参数提取结果目录（包含 result.json）。")
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    rows = _read_rows(root)
    summary = result["summary"]
    manifest_path = root / "scene.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preview = root / manifest["preview"]
        if _hash(preview, check) != manifest["preview_sha256"]:
            raise ValueError("预览数据校验失败，请重新提取参数。")
        with np.load(preview, allow_pickle=False) as data:
            scene = {name: data[name].copy() for name in data.files}
    else:
        # v1.5 / v1.6 results are opened without recomputing any measurements.
        cloud = root / "参数标注.laz"
        if not cloud.exists():
            raise ValueError("旧结果没有参数标注 LAZ，无法恢复实例点云；请重新提取并勾选导出点云。")
        field = summary.get("export_fields", {}).get("metric_tree_id", "metric_tree_id")
        height_field = summary.get("export_fields", {}).get("metric_height", "metric_height")
        manifest = dict(schema=0, cloud=cloud.name, label_field=field, point_count=summary["source_points"],
                        cloud_sha256=summary.get("outputs", {}).get(cloud.name, {}).get("sha256"))
        quota = max(1, min(6000, (PREVIEW_LIMIT - 15000) // max(1, len(rows))))
        counts = {int(row["TreeID"]): int(row["NPoints"]) for row in rows}
        counts[0] = max(0, summary["source_points"] - summary["assigned_points"])
        seen, collected = {}, []
        with laspy.open(cloud) as reader:
            origin = np.asarray(reader.header.mins)
            names = set(reader.header.point_format.dimension_names)
            if field not in names:
                raise ValueError("点云缺少结果中记录的单木编号字段。")
            has_rgb = {"red", "green", "blue"} <= names
            for chunk in reader.chunk_iterator(CHUNK):
                check()
                labels = np.asarray(chunk[field], dtype="i8")
                selected = []
                for tree_id in np.unique(labels):
                    positions = np.flatnonzero(labels == tree_id)
                    before = seen.get(int(tree_id), 0)
                    limit = 15000 if tree_id == 0 else quota
                    count = counts.get(int(tree_id), len(positions))
                    step = max(1, int(np.ceil(count / limit)))
                    selected.append(positions[(np.arange(len(positions)) + before) % step == 0])
                    seen[int(tree_id)] = before + len(positions)
                selected = np.sort(np.concatenate(selected))
                xyz = np.column_stack((chunk.x, chunk.y, chunk.z))[selected] - origin
                heights = np.asarray(chunk[height_field])[selected] if height_field in names else xyz[:, 2]
                rgb = np.column_stack((chunk.red, chunk.green, chunk.blue))[selected] if has_rgb else np.zeros((len(selected), 3))
                collected.append((xyz, labels[selected], heights, rgb))
                notify({"stage": "scene", "points": sum(seen.values()), "total": reader.header.point_count})
        scene = dict(zip(("xyz", "ids", "height", "rgb"), (np.concatenate([v[j] for v in collected]) for j in range(4))))
        colors = scene["rgb"]
        scene["rgb"] = (colors / (256 if colors.max(initial=0) > 255 else 1)).astype("u1")
        scene.update(origin=origin, has_rgb=np.array(has_rgb))
    if len(scene["xyz"]) != len(scene["ids"]) or not np.isfinite(scene["xyz"]).all():
        raise ValueError("点云预览坐标或编号无效。")
    known = set(int(r["TreeID"]) for r in rows)
    if set(np.unique(scene["ids"])) - known - {0}:
        raise ValueError("点云实例与参数表编号不一致。")
    scene.update(root=str(root), rows=rows, summary=summary, manifest=manifest)
    return scene


def load_instance(scene, tree_id, cancelled=lambda: False, notify=lambda data: None):
    """Read one object from the full cloud, with a bounded high-detail preview."""
    def check():
        if cancelled():
            raise SegmentationCancelled("已取消单木细节加载。")
    root, manifest = Path(scene["root"]), scene["manifest"]
    cloud = (root / manifest["cloud"]).resolve(strict=True)
    row = next((r for r in scene["rows"] if int(r["TreeID"]) == int(tree_id)), None)
    if row is None:
        raise ValueError("单木编号不存在。")
    if _hash(cloud, check) != manifest.get("cloud_sha256"):
        raise ValueError("完整点云校验失败，无法加载单木细节。")
    expected_count = int(row["NPoints"])
    stride = max(1, int(np.ceil(expected_count / 250_000)))
    labels = None
    collected, count, cursor = [], 0, 0
    try:
        if manifest.get("labels"):
            label_path = root / manifest["labels"]
            if _hash(label_path, check) != manifest["labels_sha256"]:
                raise ValueError("完整实例编号索引校验失败。")
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        with laspy.open(cloud) as reader:
            names = set(reader.header.point_format.dimension_names)
            field = manifest.get("label_field")
            height_field = manifest.get("height_field") or scene["summary"].get("export_fields", {}).get("metric_height", "metric_height")
            has_rgb = {"red", "green", "blue"} <= names
            for chunk in reader.chunk_iterator(CHUNK):
                check()
                ids = labels[cursor:cursor + len(chunk)] if labels is not None else np.asarray(chunk[field])
                selected = np.flatnonzero(ids == tree_id)
                sampled = selected[(np.arange(len(selected)) + count) % stride == 0]
                count += len(selected)
                if len(sampled):
                    xyz = np.column_stack((chunk.x, chunk.y, chunk.z))[sampled] - scene["origin"]
                    heights = (np.asarray(chunk[height_field])[sampled] if height_field in names
                               else np.full(len(sampled), np.nan))
                    colors = np.column_stack((chunk.red, chunk.green, chunk.blue))[sampled] if has_rgb else np.zeros((len(sampled), 3))
                    collected.append((xyz, heights, colors))
                cursor += len(chunk)
                notify({"stage": "scene", "points": cursor, "total": reader.header.point_count})
        if count != expected_count or not collected:
            raise ValueError("单木点数与参数表不一致。")
        xyz, h, rgb = [np.concatenate([part[i] for part in collected]) for i in range(3)]
        rgb = (rgb / (256 if rgb.max(initial=0) > 255 else 1)).astype("u1")
        return dict(scene, xyz=xyz, ids=np.full(len(xyz), tree_id, dtype="i8"), height=h, rgb=rgb,
                    has_rgb=np.array(has_rgb), rows=[row], detail_tree_id=tree_id, detail_count=count)
    finally:
        if labels is not None:
            labels._mmap.close()


def export_instances(scene, tree_ids, output_parent, cancelled=lambda: False, notify=lambda data: None):
    """Export complete original records; bound open writers to 24 at a time."""
    def check():
        if cancelled():
            raise SegmentationCancelled("已取消实例导出；未完成文件不会作为结果发布。")
    check()
    root, manifest = Path(scene["root"]), scene["manifest"]
    cloud = (root / manifest["cloud"]).resolve(strict=True)
    rows = {int(row["TreeID"]): row for row in scene["rows"]}
    requested = sorted(set(int(i) for i in tree_ids))
    if not requested or any(i not in rows for i in requested):
        raise ValueError("请选择参数表中存在的单木。")
    parent = Path(output_parent).resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("输出位置必须是文件夹。")
    expected = manifest.get("cloud_sha256")
    if not expected or _hash(cloud, check) != expected:
        raise ValueError("完整点云已发生变化或缺少校验信息，不能安全导出。")
    labels = None
    if manifest.get("labels"):
        label_path = root / manifest["labels"]
        if _hash(label_path, check) != manifest["labels_sha256"]:
            raise ValueError("完整实例编号索引校验失败。")
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if labels.shape != (manifest["point_count"],):
            raise ValueError("编号索引长度与完整点云不同。")
    output = parent / ("instances_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    output.mkdir()
    files, counts = [], {}
    try:
        for begin in range(0, len(requested), 24):
            batch = requested[begin:begin + 24]
            with laspy.open(cloud) as reader, ExitStack() as stack:
                header = reader.header.copy()
                field = manifest.get("label_field")
                add_field = not field
                if add_field:
                    field = "metric_tree_id"
                    names = set(header.point_format.dimension_names)
                    while field in names:
                        field += "_copy"
                    header.add_extra_dim(laspy.ExtraBytesParams(name=field, type="i8"))
                hashes = {i: hashlib.sha256() for i in batch}
                writers = {i: stack.enter_context(laspy.open(output / f"Tree_{i}.partial.laz", mode="w", header=header)) for i in batch}
                counts.update({i: 0 for i in batch})
                cursor = 0
                for chunk in reader.chunk_iterator(CHUNK):
                    check()
                    ids = labels[cursor:cursor + len(chunk)] if labels is not None else np.asarray(chunk[field])
                    for i in batch:
                        mask = ids == i
                        if not mask.any():
                            continue
                        records = chunk[mask]
                        if add_field:
                            out = laspy.ScaleAwarePointRecord.zeros(len(records), header=header)
                            for name in records.array.dtype.names:
                                out.array[name] = records.array[name]
                            out[field] = np.full(len(records), i, dtype="i8")
                            records = out
                        hashes[i].update(records.array.tobytes())
                        writers[i].write_points(records)
                        counts[i] += len(records)
                    cursor += len(chunk)
                    notify({"stage": "instance_export", "points": cursor, "total": reader.header.point_count})
                if reader.header.evlrs:
                    for writer in writers.values():
                        writer.write_evlrs(reader.header.evlrs)
            for i in batch:
                check()
                path = output / f"Tree_{i}.partial.laz"
                digest = hashlib.sha256()
                with laspy.open(path) as reader:
                    if reader.header.point_count != counts[i] or counts[i] != int(rows[i]["NPoints"]):
                        raise ValueError(f"单木 {i} 点数与参数表不一致。")
                    def metadata(records):
                        return [(v.user_id, v.record_id, v.description, v.record_data_bytes()) for v in (records or [])
                                if v.user_id != "laszip encoded" and not (v.user_id == "LASF_Spec" and v.record_id == 4)]
                    if (not np.array_equal(reader.header.scales, header.scales)
                            or not np.array_equal(reader.header.offsets, header.offsets)
                            or metadata(reader.header.vlrs) != metadata(header.vlrs)
                            or metadata(reader.header.evlrs) != metadata(header.evlrs)):
                        raise ValueError("单木坐标精度或元数据未保留。")
                    for chunk in reader.chunk_iterator(CHUNK):
                        check()
                        digest.update(chunk.array.tobytes())
                        if not np.all(np.asarray(chunk[field]) == i):
                            raise ValueError("导出实例编号不一致。")
                if digest.digest() != hashes[i].digest():
                    raise ValueError("单木点云原始属性读回校验失败。")
                target = output / f"Tree_{i}.laz"
                path.rename(target)
                files.append(str(target))
        if _hash(cloud, check) != expected:
            raise ValueError("完整点云在导出期间发生变化，导出未发布。")
        with (output / "单木参数.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
            writer.writeheader()
            writer.writerows(rows[i] for i in requested)
        result = dict(output_directory=str(output), files=files, points=counts, original_records_verified=True,
                      source=str(cloud), tree_ids=requested)
        _json(output / "result.json", result)
        return result
    except Exception as exc:
        _json(output / "incomplete.json", {"error": str(exc)})
        raise
    finally:
        if labels is not None:
            labels._mmap.close()
