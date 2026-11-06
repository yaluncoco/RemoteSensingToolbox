"""File-based forest segmentation. No desktop input or GUI state is used.

CSF and TreeIso run in an isolated CloudCompare CLI process. Labels are proposals,
not botanical ground truth. Full-resolution LAS records keep their original order.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from PointCloud import cloudcompare as cli
from PointCloud.forestry_io import fingerprint, read_cloud, project_lock, publish_new


EXTRA_FIELDS = {"semantic_id": "u2", "instance_id": "u4", "review_status": "u1", "source_index": "u4"}


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def progress(root, stage, **values):
    cli.check_cancel()
    data = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **values}
    save_json(root / "progress.json", data)
    cli.report(data)


def write_ply(path, xyz, fields=None):
    fields = fields or {}
    data = np.empty(len(xyz), dtype=[(n, "f4") for n in "xyz"] + [(n, np.asarray(v).dtype) for n, v in fields.items()])
    for i, n in enumerate("xyz"):
        data[n] = xyz[:, i]
    for n, v in fields.items():
        data[n] = v
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(path)


def voxel_keys(xyz, metadata):
    cell = np.floor((xyz - np.asarray(metadata["voxel_origin_local"])) / metadata["voxel_resolution"]).astype(np.int64)
    shape = np.asarray(metadata["voxel_shape"], dtype=np.int64)
    if np.any(cell < 0) or np.any(cell >= shape):
        raise ValueError("Source point outside the prepared voxel grid")
    return (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]


def locate_keys(keys, sorted_keys):
    idx = np.searchsorted(sorted_keys, keys)
    if np.any(idx >= len(sorted_keys)) or not np.array_equal(sorted_keys[idx], keys):
        raise ValueError("Source voxel missing from sample; refusing to misalign labels")
    return idx


def validate_source(root):
    meta = json.loads((root / "source.json").read_text(encoding="utf-8"))
    source = Path(meta["source"])
    stat = source.stat()
    if stat.st_size != meta["source_size"] or stat.st_mtime_ns != meta["source_mtime_ns"]:
        raise ValueError("Source changed since sampling; use a new output directory")
    digest = fingerprint(source)
    if meta.get("source_sha256") and meta["source_sha256"] != digest:
        raise ValueError("Source fingerprint changed")
    meta["source_sha256"] = digest
    with laspy.open(source) as reader:
        if reader.header.point_count != meta["source_point_count"]:
            raise ValueError("Source point count changed")
    save_json(root / "source.json", meta)
    return meta


def prepare(source, root, resolution):
    """Stream the source and keep one actual source point per 3D voxel."""
    source, root = Path(source).resolve(strict=True), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "sample.npz").exists():
        raise FileExistsError("Prepared sample exists")
    before = fingerprint(source)
    stat = source.stat()
    keys, coords, source_ids, intensities = [], [], [], []
    with laspy.open(source) as reader:
        h = reader.header
        if h.point_count == 0 or h.point_count > np.iinfo(np.uint32).max:
            raise ValueError("This pipeline supports 1..2^32-1 LAS/LAZ points")
        if h.point_format.id in (4, 5, 9, 10):
            raise ValueError("Waveform formats require a separate preservation path")
        meta = {"source": str(source), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                "source_sha256": before, "source_point_count": int(h.point_count), "source_offsets": h.offsets.tolist(),
                "source_scales": h.scales.tolist(), "working_origin": h.mins.tolist(), "voxel_origin_local": [0., 0., 0.],
                "voxel_shape": (np.ceil((h.maxs - h.mins) / resolution).astype(np.int64) + 2).tolist(),
                "voxel_resolution": resolution, "coordinate_note": "local = world - working_origin; units must be metres"}
        if np.prod(np.asarray(meta["voxel_shape"], dtype=object)) > np.iinfo(np.int64).max:
            raise ValueError("Voxel grid exceeds int64 capacity")
        offset = 0
        for chunk in reader.chunk_iterator(1_000_000):
            cli.check_cancel()
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)) - np.asarray(meta["working_origin"])
            code = voxel_keys(xyz, meta)
            key, idx = np.unique(code, return_index=True)
            keys.append(key)
            coords.append(xyz[idx].astype("f4"))
            source_ids.append((idx + offset).astype("u4"))
            intensities.append(np.asarray(chunk.intensity)[idx])
            offset += len(chunk)
            if offset % 5_000_000 == 0 or offset == h.point_count:
                progress(root, "sampling", points=offset, total=int(h.point_count))
    if fingerprint(source) != before:
        raise ValueError("Source changed during sampling")
    unique, idx = np.unique(np.concatenate(keys), return_index=True)
    xyz = np.concatenate(coords)[idx]
    intensity = np.concatenate(intensities)[idx]
    np.savez_compressed(root / "sample.npz", xyz=xyz, source_index=np.concatenate(source_ids)[idx], intensity=intensity, voxel_key=unique)
    write_ply(root / "sample_local.ply", xyz, {"intensity": intensity})
    meta.update(sample_count=len(xyz), local_minimum=xyz.min(0).tolist(), local_maximum=xyz.max(0).tolist())
    save_json(root / "source.json", meta)
    return meta


def _ground_and_vegetation(root, params):
    ground_path = root / "sample_local_ground_points.ply"
    off_path = root / "sample_local_offground_points.ply"
    if not ground_path.exists() or not off_path.exists():
        progress(root, "ground_filtering")
        result = cli.run(["-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(root / "sample_local.ply"),
                          "-CSF", "-SCENES", params["terrain"], "-CLOTH_RESOLUTION", "0.5", "-CLASS_THRESHOLD", "0.3",
                          "-MAX_ITERATION", "500", "-EXPORT_GROUND", "-EXPORT_OFFGROUND"], root / "csf-job", 600)
        save_json(root / "csf-run.json", result)
    if (root / "vegetation.npz").exists():
        return
    ground = read_cloud(ground_path).xyz
    off = read_cloud(off_path).xyz
    progress(root, "vegetation_filtering")
    d, ii = cKDTree(ground[:, :2]).query(off[:, :2], k=min(8, len(ground)), workers=4)
    if d.ndim == 1:
        d, ii = d[:, None], ii[:, None]
    height = off[:, 2] - np.median(ground[ii, 2], axis=1)
    near, _ = cKDTree(off).query(off, k=min(8, len(off)), workers=4)
    if near.ndim == 1:
        near = near[:, None]
    keep = (height > .35) & (height < 40) & (near[:, -1] < .8) & (d[:, 0] < 3)
    xyz, height = off[keep].astype("f4"), height[keep].astype("f4")
    if len(xyz) < 20:
        raise ValueError("Too few vegetation points after ground and noise filtering")
    with np.load(root / "sample.npz") as sample:
        distance, ids = cKDTree(sample["xyz"]).query(xyz, workers=4)
    if np.max(distance) > 1e-5 or len(np.unique(ids)) != len(ids):
        raise ValueError("CSF output cannot be mapped unambiguously to working sample")
    np.savez_compressed(root / "vegetation.npz", xyz=xyz, height=height, sample_index=ids.astype("u4"))
    write_ply(root / "vegetation_local.ply", xyz, {"height_above_ground": height, "sample_index": ids.astype("u4")})


def _treeiso(root, resolution):
    if (root / "treeiso_local.ply").exists():
        if not (root / "treeiso-run.json").exists():
            raise ValueError("Existing TreeIso result needs its command/provenance JSON")
        run = json.loads((root / "treeiso-run.json").read_text(encoding="utf-8"))
        command = run["command"]
        for name, expected in (("-LAMBDA1", 1), ("-K1", 5), ("-LAMBDA2", 20), ("-K2", 20),
                               ("-MAX_GAP", 2), ("-RHO", .5), ("-VERTICAL_OVERLAP_WEIGHT", .5),
                               ("-DECIMATE_RESOLUTION1", resolution), ("-DECIMATE_RESOLUTION2", resolution)):
            if name not in command or float(command[command.index(name) + 1]) != expected:
                raise ValueError(f"Existing TreeIso parameter differs: {name}")
        return
    progress(root, "treeiso", sample_resolution=resolution)
    # CC parses SAVE_CLOUDS FILE as its own filename list. A relative path avoids
    # splitting a parent directory that contains spaces; cwd is root/treeiso-job.
    result = cli.run(["-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(root / "vegetation_local.ply"),
                      "-TREEISO", "-LAMBDA1", "1", "-K1", "5", "-DECIMATE_RESOLUTION1", str(resolution),
                      "-LAMBDA2", "20", "-K2", "20", "-MAX_GAP", "2", "-DECIMATE_RESOLUTION2", str(resolution),
                      "-RHO", "0.5", "-VERTICAL_OVERLAP_WEIGHT", "0.5", "-SAVE_CLOUDS", "FILE",
                      "../treeiso_local.ply"], root / "treeiso-job", 1800)
    save_json(root / "treeiso-run.json", result)


def analyze(root, metadata, min_sample_points=100, min_height=2.0):
    """Map TreeIso geometry back to source representatives, without assuming SF preservation."""
    with np.load(root / "sample.npz") as a:
        sample = {k: a[k] for k in a.files}
    with np.load(root / "vegetation.npz") as a:
        vegetation = {k: a[k] for k in a.files}
    result = read_cloud(root / "treeiso_local.ply")
    if len(result.xyz) != len(vegetation["xyz"]):
        raise ValueError("TreeIso point count differs from input")
    names = [n for n in result.fields if n in ("final_segs", "scalar_final_segs")]
    if len(names) != 1:
        raise ValueError("TreeIso did not produce exactly one final_segs scalar field")
    values = result.fields[names[0]]
    if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0):
        raise ValueError("Invalid TreeIso IDs")
    raw = values.astype("u4")
    if np.array_equal(result.xyz, vegetation["xyz"]):
        order = np.arange(len(raw))
    else:
        distance, order = cKDTree(vegetation["xyz"]).query(result.xyz, workers=4)
        if np.max(distance) > 1e-5 or len(np.unique(order)) != len(order):
            raise ValueError("TreeIso changed coordinates; label transfer is unsafe")
    xyz = result.xyz
    height = vegetation["height"][order]
    sample_ids = vegetation["sample_index"][order]
    if not np.array_equal(sample["xyz"][sample_ids], xyz):
        raise ValueError("Vegetation-to-sample provenance mismatch")
    semantic = np.zeros(len(sample["xyz"]), dtype="u2")
    instance = np.zeros(len(semantic), dtype="u4")
    review = np.zeros(len(semantic), dtype="u1")
    raw_sample = np.zeros(len(semantic), dtype="u4")
    # +1 reserves 0 for ground/non-vegetation even if an engine uses segment zero.
    raw_sample[sample_ids] = raw + 1
    ground = read_cloud(root / "sample_local_ground_points.ply").xyz
    distance, ground_ids = cKDTree(sample["xyz"]).query(ground, workers=4)
    if np.max(distance) > 1e-5 or len(np.unique(ground_ids)) != len(ground_ids):
        raise ValueError("Ground result does not match source representatives")
    if np.intersect1d(ground_ids, sample_ids).size:
        raise ValueError("Ground and vegetation sample sets overlap")
    semantic[ground_ids] = 1
    review[ground_ids] = 1
    rows, rejected = [], []
    grouped = np.argsort(raw, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(raw[grouped])) + 1, len(raw)]
    for start, end in zip(starts[:-1], starts[1:]):
        idx = grouped[start:end]
        pts, h = xyz[idx], height[idx]
        q = np.quantile(h, [.05, .5, .95, .99, 1])
        raw_id = int(raw[idx[0]])
        if len(idx) < min_sample_points or q[2] < min_height:
            rejected.append({"raw_treeiso_id": raw_id, "sample_points": len(idx), "height_p95_m": float(q[2])})
            continue
        low = (h >= .35) & (h < 1.5)
        width = np.ptp(pts[:, :2], axis=0)
        flags = []
        if len(idx) < 500:
            flags.append("sparse_segment")
        if np.count_nonzero(low) < 5:
            flags.append("no_low_stem_support")
        if width.max() > q[2] * 1.5:
            flags.append("wide_or_possible_merge")
        center = np.median(pts[low, :2] if np.count_nonzero(low) >= 5 else pts[:, :2], axis=0)
        rows.append({"raw_treeiso_id": raw_id, "sample_points": len(idx), "x_local": float(center[0]), "y_local": float(center[1]),
                     "x_world": float(center[0] + metadata["working_origin"][0]), "y_world": float(center[1] + metadata["working_origin"][1]),
                     "position_basis": "low_points_median" if np.count_nonzero(low) >= 5 else "crown_median",
                     "height_p99_m": float(q[3]), "height_max_m": float(q[4]), "crown_width_x_m": float(width[0]),
                     "crown_width_y_m": float(width[1]), "low_sample_points": int(low.sum()),
                     "review_status": 2 if flags else 1, "review_flags": "|".join(flags), "_idx": idx})
    rows.sort(key=lambda r: (r["y_local"], r["x_local"]))
    if not rows:
        raise ValueError("No plausible tree candidates; review ground filtering and thresholds")
    for tree_id, row in enumerate(rows, 1):
        ids = sample_ids[row.pop("_idx")]
        semantic[ids], instance[ids], review[ids] = 2, tree_id, row["review_status"]
        row["tree_id"] = tree_id
        row["file"] = f"trees/Tree_{tree_id:04d}.laz"
    np.savez_compressed(root / "sample_labels.npz", semantic_id=semantic, instance_id=instance, review_status=review, raw_segment_id=raw_sample)
    save_json(root / "tree_inventory.json", rows)
    save_json(root / "rejected_segments.json", rejected)
    _write_inventory(root / "tree_inventory.csv", rows)
    summary = {"source": metadata["source"], "source_sha256": metadata["source_sha256"] if "source_sha256" in metadata else None,
               "source_point_count": metadata["source_point_count"], "sample_count": len(semantic), "vegetation_sample_count": len(raw),
               "raw_segment_count": len(starts) - 1, "candidate_tree_count": len(rows),
               "flagged_candidate_count": sum(r["review_status"] == 2 for r in rows), "rejected_fragment_count": len(rejected),
               "sample_class_counts": {str(k): int((semantic == k).sum()) for k in (0, 1, 2)},
               "sample_resolution_m": metadata["voxel_resolution"], "min_sample_points": min_sample_points, "min_height_p95_m": min_height,
               "status": "automatic_candidates_require_review", "all_candidates_require_review": True,
               "semantic_classes": {"0": "unclassified", "1": "ground_candidate", "2": "tree_candidate"},
               "review_status": {"0": "unclassified", "1": "automatic_unreviewed", "2": "automatic_flagged_for_review"},
               "label_transfer": "same 0.2m source voxel" if metadata["voxel_resolution"] == .2 else "same source voxel",
               "limitations": ["No field inventory or ground truth; candidate count is not a verified tree census",
                               "Sparse edges and touching crowns may be split or merged; flagged candidates are retained",
                               "Tree bases below 0.35m are excluded from tree segmentation; voxel boundaries have sampling uncertainty",
                               "Height is estimated against CSF ground in source units assumed to be metres; no CRS is invented"],
               "outputs": {"inventory": str(root / "tree_inventory.csv"), "overview": str(root / "instance_overview.png"),
                           "map": str(root / "instance_map.svg"), "gallery": str(root / "tree_gallery.png")}}
    save_json(root / "segmentation.json", summary)
    cli.check_cancel()
    render_results(root, sample["xyz"], instance, rows)
    return summary


def _write_inventory(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def palette(count):
    colors = np.random.default_rng(20261106).uniform(.12, .9, (count + 1, 3))
    colors[0] = [.67, .67, .67]
    return colors


def render_results(root, xyz, ids, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    colors = palette(len(rows))
    keep = np.flatnonzero(ids)
    pick = keep[::max(1, len(keep) // 500000)]
    fig = Figure(figsize=(18, 11), layout="constrained")
    FigureCanvasAgg(fig)
    axs = fig.subplots(1, 2)
    pts = xyz[pick]
    axs[0].scatter(pts[:, 0], pts[:, 1], c=colors[ids[pick]], s=.4, linewidths=0, rasterized=True)
    # The second panel concentrates on denser candidates; every candidate is in the first.
    dense = [r for r in rows if r["sample_points"] >= 1000] or rows
    bounds = np.quantile([[r["x_local"], r["y_local"]] for r in dense], [.02, .98], axis=0)
    axs[1].scatter(pts[:, 0], pts[:, 1], c=colors[ids[pick]], s=.7, linewidths=0, rasterized=True)
    axs[1].set(xlim=(bounds[0, 0]-5, bounds[1, 0]+5), ylim=(bounds[0, 1]-5, bounds[1, 1]+5))
    for ax in axs:
        ax.set(aspect="equal", xlabel="X (local m)", ylabel="Y (local m)")
    axs[0].set_title(f"Entire forest: {len(rows)} automatic tree candidates")
    axs[1].set_title("Denser region: instance numbers")
    for r in rows:
        if bounds[0, 0]-5 < r["x_local"] < bounds[1, 0]+5 and bounds[0, 1]-5 < r["y_local"] < bounds[1, 1]+5:
            axs[1].text(r["x_local"], r["y_local"], str(r["tree_id"]), fontsize=4, ha="center",
                        bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none", "pad": .1})
    fig.suptitle("CSF + TreeIso | Automatic annotations - all require review")
    fig.savefig(root / "instance_overview.png", dpi=170)
    fig.savefig(root / "instance_map.svg")
    fig.clear()
    supported = [r for r in rows if r["review_status"] == 1]
    if not supported:
        supported = rows
    selected = [supported[i] for i in np.linspace(0, len(supported)-1, min(6, len(supported)), dtype=int)]
    fig = Figure(figsize=(15, 10), layout="constrained")
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    for ax, r in zip(axs.flat, selected):
        pts = xyz[ids == r["tree_id"]]
        ax.scatter(pts[:, 0] - r["x_local"], pts[:, 2] - pts[:, 2].min(), s=.5, c=[colors[r["tree_id"]]], linewidths=0)
        ax.set(title=f"Tree {r['tree_id']:04d} | {r['sample_points']:,} sample points", xlabel="X relative to centre (m)",
               ylabel="Z relative to segment bottom (m)", aspect="equal")
    fig.suptitle("Examples with low-point support - automatic, not individually verified")
    fig.savefig(root / "tree_gallery.png", dpi=150)
    fig.clear()


def segment(source_path, output_directory, coordinates_in_metres, resolution=.2, terrain="FLAT", min_sample_points=100,
            min_height=2.0, reuse_existing=False):
    source, root = Path(source_path).resolve(strict=True), Path(output_directory).resolve()
    if not coordinates_in_metres:
        raise ValueError("Explicitly confirm source coordinates are in metres for forestry parameters")
    if not np.isfinite([resolution, min_height]).all() or not .05 <= resolution <= .5 or min_sample_points < 20 or min_height < .5:
        raise ValueError("Require resolution .05..0.5m, min_sample_points >=20 and min_height >=0.5m")
    if terrain not in ("FLAT", "RELIEF", "SLOPE"):
        raise ValueError("Require FLAT/RELIEF/SLOPE terrain")
    # The workflow identifies contents before this stage; a valid LAS/LAZ may
    # itself have a misleading extension. laspy reads the actual file header.
    root.mkdir(parents=True, exist_ok=True)
    params = {"resolution": resolution, "terrain": terrain, "min_sample_points": min_sample_points, "min_height": min_height}
    with project_lock(root / "forest_job"):
        if (root / "source.json").exists():
            if not reuse_existing:
                raise FileExistsError("Use a new output directory or explicitly reuse existing matching stages")
            meta = validate_source(root)
            if Path(meta["source"]).resolve() != source or meta["voxel_resolution"] != resolution:
                raise ValueError("Prepared source or resolution differs")
            if (root / "parameters.json").exists() and json.loads((root / "parameters.json").read_text()) != params:
                raise ValueError("Existing job parameters differ; choose a new directory")
            if not (root / "parameters.json").exists() and terrain != "FLAT":
                raise ValueError("Legacy prepared stages only support their recorded FLAT terrain")
        else:
            meta = prepare(source, root, resolution)
        save_json(root / "parameters.json", params)
        if (root / "segmentation.json").exists():
            return json.loads((root / "segmentation.json").read_text(encoding="utf-8"))
        _ground_and_vegetation(root, params)
        _treeiso(root, resolution)
        progress(root, "numbering_candidates")
        result = analyze(root, meta, min_sample_points, min_height)
        progress(root, "segmentation_complete", candidates=result["candidate_tree_count"])
        return result


def status(output_directory):
    root = Path(output_directory).resolve(strict=True)
    return {name: json.loads((root / (name + ".json")).read_text(encoding="utf-8"))
            for name in ("progress", "segmentation", "export") if (root / (name + ".json")).exists()}


def _labeled_header(header):
    result = copy.deepcopy(header)
    if result.point_format.id in (4, 5, 9, 10):
        raise ValueError("Waveform LAS export is unsupported")
    if set(EXTRA_FIELDS) & set(result.point_format.dimension_names):
        raise ValueError("Source already has forestry label fields; refusing to replace existing annotations")
    for name, dtype in EXTRA_FIELDS.items():
        result.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    return result


def _validate_full(source, output, sample_keys, labels, meta, root):
    counts = np.zeros(int(labels["instance_id"].max()) + 1, dtype="i8")
    classes = np.zeros(3, dtype="i8")
    offset = 0
    with laspy.open(source) as src, laspy.open(output) as dst:
        if src.header.point_count != dst.header.point_count:
            raise ValueError("Export lost source points")
        if not np.array_equal(src.header.scales, dst.header.scales) or not np.array_equal(src.header.offsets, dst.header.offsets):
            raise ValueError("Export changed coordinate precision or origin")
        for a in src.chunk_iterator(1_000_000):
            cli.check_cancel()
            b = dst.read_points(len(a))
            # Byte comparisons include classification flags, GPS time, RGB, custom fields and exact integer XYZ.
            if any(a.array[n].tobytes() != b.array[n].tobytes() for n in a.array.dtype.names):
                raise ValueError(f"Original LAS attribute changed at source row {offset}")
            xyz = np.column_stack((a.x, a.y, a.z)) - np.asarray(meta["working_origin"])
            idx = locate_keys(voxel_keys(xyz, meta), sample_keys)
            for name in ("semantic_id", "instance_id", "review_status"):
                if not np.array_equal(b[name], labels[name][idx]):
                    raise ValueError(f"Exported {name} differs from source-voxel labels")
            if not np.array_equal(b.source_index, np.arange(offset, offset + len(a), dtype="u4")):
                raise ValueError("Export changed source row order")
            counts += np.bincount(np.asarray(b.instance_id), minlength=len(counts))
            classes += np.bincount(np.asarray(b.semantic_id), minlength=3)
            offset += len(a)
            if offset % 10_000_000 == 0 or offset == meta["source_point_count"]:
                progress(root, "validating_full_resolution", points=offset, total=meta["source_point_count"])
    return counts, classes


def export(root_path, extract_trees=True):
    """Stream all source records, add labels, verify every record, then split trees."""
    root = Path(root_path).resolve(strict=True)
    with project_lock(root / "forest_job"):
        meta = validate_source(root)
        summary = json.loads((root / "segmentation.json").read_text(encoding="utf-8"))
        if summary["source_sha256"] != meta["source_sha256"]:
            raise ValueError("Segmentation belongs to a different source fingerprint")
        if (root / "export.json").exists():
            previous = json.loads((root / "export.json").read_text(encoding="utf-8"))
            if extract_trees and not previous["extracted_trees"]:
                raise ValueError("Existing export omitted tree files; use a new job for a different export mode")
            if fingerprint(Path(previous["full_cloud"])) != previous["full_cloud_sha256"]:
                raise ValueError("Existing full-resolution export has changed")
            return previous
        target = root / "forest_instances.laz"
        if target.exists() or (root / ".export-work").exists():
            raise FileExistsError("Export files already exist without a completion manifest; inspect partial output first")
        work = root / ".export-work"
        work.mkdir()
        stage_las, stage_laz = work / "records.las", work / "forest_instances.laz"
        with np.load(root / "sample.npz") as a:
            sample = {k: a[k] for k in a.files}
        with np.load(root / "sample_labels.npz") as a:
            labels = {k: a[k] for k in a.files}
        rows = json.loads((root / "tree_inventory.json").read_text(encoding="utf-8"))
        if any(len(labels[n]) != len(sample["voxel_key"]) for n in ("semantic_id", "instance_id", "review_status")):
            raise ValueError("Label count does not match prepared sample")
        source_order = np.argsort(sample["source_index"])
        sorted_source_ids = sample["source_index"][source_order]
        preview_header = laspy.LasHeader(point_format=3, version="1.4")
        preview_header.offsets, preview_header.scales = np.asarray(meta["source_offsets"]), np.asarray(meta["source_scales"])
        preview_header = _labeled_header(preview_header)
        preview = laspy.LasData(preview_header, laspy.ScaleAwarePointRecord.zeros(len(source_order), header=preview_header))
        offset = 0
        source = Path(meta["source"])
        with laspy.open(source) as reader:
            header = _labeled_header(reader.header)
            # Preserve projection records verbatim, including EVLRs, without needing pyproj.
            from laspy.vlrs.vlrlist import VLRList
            preview.header.vlrs.extend(copy.deepcopy(reader.header.vlrs.get_by_id("LASF_Projection")))
            if reader.header.evlrs:
                preview.header.evlrs = VLRList(copy.deepcopy(reader.header.evlrs.get_by_id("LASF_Projection")))
            if reader.header.global_encoding.wkt:
                preview.header.global_encoding.wkt = True
            with laspy.open(stage_las, mode="w", header=header) as plain, laspy.open(stage_laz, mode="w", header=header) as compressed:
                for chunk in reader.chunk_iterator(1_000_000):
                    cli.check_cancel()
                    xyz = np.column_stack((chunk.x, chunk.y, chunk.z)) - np.asarray(meta["working_origin"])
                    idx = locate_keys(voxel_keys(xyz, meta), sample["voxel_key"])
                    record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                    for name in chunk.array.dtype.names:
                        record.array[name] = chunk.array[name]
                    for name in ("semantic_id", "instance_id", "review_status"):
                        record[name] = labels[name][idx]
                    record.source_index = np.arange(offset, offset + len(chunk), dtype="u4")
                    start, end = np.searchsorted(sorted_source_ids, [offset, offset + len(chunk)])
                    selected_rows = source_order[start:end]
                    local_rows = sorted_source_ids[start:end].astype("i8") - offset
                    # Validate the identity of every representative against the untouched source.
                    if not np.array_equal(xyz[local_rows].astype("f4"), sample["xyz"][selected_rows]):
                        raise ValueError("Working sample no longer corresponds to original source rows")
                    for name in ("X", "Y", "Z", "intensity", *EXTRA_FIELDS):
                        preview[name][selected_rows] = record[name][local_rows]
                    plain.write_points(record)
                    compressed.write_points(record)
                    offset += len(chunk)
                    if offset % 5_000_000 == 0 or offset == meta["source_point_count"]:
                        progress(root, "writing_full_resolution", points=offset, total=meta["source_point_count"])
                if reader.header.evlrs:
                    plain.write_evlrs(reader.header.evlrs)
                    compressed.write_evlrs(reader.header.evlrs)
        counts, classes = _validate_full(source, stage_laz, sample["voxel_key"], labels, meta, root)
        if fingerprint(source) != meta["source_sha256"]:
            raise ValueError("Source changed during export")
        colors = (palette(len(rows))[np.asarray(preview.instance_id)] * 65535).astype("u2")
        preview.red, preview.green, preview.blue = colors.T
        preview_path = root / "instance_preview.laz"
        preview.write(preview_path)
        check_preview = laspy.read(preview_path)
        if check_preview.points.array.tobytes() != preview.points.array.tobytes():
            raise ValueError("Preview did not round-trip exactly")
        del check_preview, preview
        tree_files = []
        for row in rows:
            row["point_count"] = int(counts[row["tree_id"]])
        if extract_trees:
            trees = root / "trees"
            trees.mkdir(exist_ok=False)
            progress(root, "indexing_tree_points", tree_count=len(rows))
            with laspy.mmap(stage_las) as mapped:
                ids = np.asarray(mapped.instance_id)
                nonzero = np.flatnonzero(ids)
                grouped = nonzero[np.argsort(ids[nonzero], kind="stable")]
                del nonzero, ids
                start = 0
                for row in rows:
                    cli.check_cancel()
                    count, tree_id = row["point_count"], row["tree_id"]
                    selected = grouped[start:start + count]
                    record = laspy.ScaleAwarePointRecord.zeros(count, header=header)
                    record.array[:] = mapped.points.array[selected]
                    final = trees / f"Tree_{tree_id:04d}.laz"
                    pending = work / f"Tree_{tree_id:04d}.laz"
                    with laspy.open(pending, mode="w", header=header) as writer:
                        writer.write_points(record)
                        if mapped.header.evlrs:
                            writer.write_evlrs(mapped.header.evlrs)
                    check = laspy.read(pending)
                    if check.points.array.tobytes() != record.array.tobytes() or not np.all(check.instance_id == tree_id):
                        raise ValueError(f"Tree {tree_id} did not round-trip exactly")
                    publish_new(pending, final)
                    tree_files.append({"tree_id": tree_id, "point_count": count, "file": row["file"], "sha256": fingerprint(final)})
                    start += count
                    if tree_id % 50 == 0 or tree_id == len(rows):
                        progress(root, "extracting_individual_trees", completed=tree_id, total=len(rows))
                if start != int(counts[1:].sum()):
                    raise ValueError("Individual tree files do not partition all labeled tree points")
                del grouped
            save_json(trees / "index.json", tree_files)
        publish_new(stage_laz, target)
        # Only delete the temporary file created by this export, after complete read-back validation.
        stage_las.unlink()
        save_json(root / "tree_inventory.json", rows)
        _write_inventory(root / "tree_inventory.csv", rows)
        result = {"full_cloud": str(target), "full_cloud_sha256": fingerprint(target), "preview_cloud": str(preview_path),
                  "tree_directory": str(root / "trees") if extract_trees else None, "extracted_trees": len(tree_files),
                  "source_point_count": meta["source_point_count"], "source_sha256": meta["source_sha256"],
                  "tree_labeled_point_count": int(counts[1:].sum()), "class_counts": {str(i): int(c) for i, c in enumerate(classes)},
                  "verification": {"all_source_records_and_attributes_compared": True, "coordinate_scale_offsets_preserved": True,
                                   "source_order_preserved": True, "voxel_labels_checked_for_every_point": True,
                                   "every_tree_file_read_back": bool(extract_trees), "source_unchanged": True},
                  "status": "automatic_candidates_require_review", "instance_count": len(rows),
                  "preview_note": "Representative source points with artificial instance colors; original RGB is retained in full cloud and tree files"}
        save_json(root / "export.json", result)
        progress(root, "complete", candidates=len(rows), full_resolution_points=meta["source_point_count"])
        return result
