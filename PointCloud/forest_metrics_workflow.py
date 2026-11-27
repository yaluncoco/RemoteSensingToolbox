"""Independent forestry inventory workflow with immutable sources and audited outputs."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import csv
import json
from pathlib import Path
import tempfile
import time
import uuid

import laspy
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree, QhullError

from PointCloud import cloudcompare as cli
from PointCloud.forest_metrics import MetricOptions, detect_stems, fit_circle, measure_tree
from PointCloud.forestry_input import identify_input, normalize_input
from PointCloud.forestry_io import fingerprint, read_cloud
from PointCloud.forestry import save_json, prepare


CHUNK = 400_000
METHODS = {
    "height": "P99 of terrain-normalized heights; maximum also exported; metres",
    "dbh": "horizontal slice at breast_height +/- dbh_band/2; three-point RANSAC and geometric least squares; cm",
    "dbh_quality": "minimum unique points, inlier fraction, occupied angular sectors, arc coverage and radius bounds; rejected fits are null",
    "crown_base": "first two consecutive profile slices >= half maximum 2-98 percentile XY width; geometric estimate, not observed live-branch height",
    "crown_width": "2-98 percentile X/Y extent above estimated crown base and below height P99; X/Y axes, not verified east/north",
    "crown_area": "2D convex hull of crown returns; m2",
    "crown_volume": "integral of per-slice convex hull area; empty slices not filled; null if coverage <80%; m3",
    "stem_volume": "pi*(DBH_m/2)^2*Height_P99*user_form_factor; model estimate, not measured merchantable volume; m3",
    "partition": "existing positive IDs preserved, or exact XY nearest validated stem within maximum distance; ID 0 unassigned",
    "ground": "median per occupied XY cell; linear triangulation, nearest fallback outside hull; support-distance cutoff",
    "review": "automatic estimates need review; ReviewRequired marks detected quality issues and is not accuracy confidence",
}


def _label_field(header, requested):
    names = list(header.point_format.dimension_names)
    if requested:
        if requested not in names:
            raise ValueError(f"点云中不存在编号字段：{requested}")
        return requested
    hits = [n for n in names if n.lower() in ("instance_id", "treeid", "tree_id", "metric_tree_id")]
    if len(hits) > 1:
        raise ValueError("发现多个单木编号字段，请在“编号字段”中明确指定：" + ", ".join(hits))
    return hits[0] if hits else None


def _ground_model(points, options):
    if len(points) < 3:
        raise ValueError("有效地面点不足。请使用含地面点的完整点云，或选择“Z 已是离地高度”。")
    cells = np.floor(points[:, :2]/options.ground_cell).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((points[:, 2], inverse))
    _, start, count = np.unique(inverse[order], return_index=True, return_counts=True)
    # Retain an actual median-height return per cell to avoid fabricated geometry.
    ground = points[order[start+count//2]]
    if len(ground) < 3:
        raise ValueError("地面覆盖范围不足，无法建立地形模型。")
    try:
        linear = LinearNDInterpolator(ground[:, :2], ground[:, 2])
    except QhullError as exc:
        raise ValueError("地面点共线，无法建立二维地形模型。") from exc
    return ground, cKDTree(ground[:, :2]), linear


def _tables_and_plot(root, rows, summary):
    from openpyxl import Workbook, load_workbook
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    columns = list(rows[0])
    with (root/"林业参数.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    book = Workbook()
    sheet = book.active
    sheet.title = "林业参数"
    sheet.append(columns)
    for row in rows:
        sheet.append([row[name] for name in columns])
    sheet.freeze_panes = "D2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = min(32, max(14, len(str(column[0].value))+2))
    methods = book.create_sheet("方法与单位")
    methods.append(["项目", "说明"])
    for name, value in METHODS.items():
        methods.append([name, value])
    methods.append(["参数", json.dumps(summary["parameters"], ensure_ascii=False)])
    methods.column_dimensions["A"].width = 25
    methods.column_dimensions["B"].width = 110
    book.save(root/"林业参数.xlsx")
    book.close()
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots(2, 2)
    positions = [r for r in rows if r["X"] is not None]
    if positions:
        scatter = ax[0, 0].scatter([r["X"] for r in positions], [r["Y"] for r in positions],
                                    c=[r["Height_m"] for r in positions], s=16, cmap="viridis")
        fig.colorbar(scatter, ax=ax[0, 0], label="Height P99 (m)")
        ax[0, 0].set_aspect("equal")
    ax[0, 0].set(xlabel="Source X (m)", ylabel="Source Y (m)", title=f"Tree candidates: {len(rows)}")
    for axes, field, title in ((ax[0, 1], "DBH_cm", "Accepted DBH (cm)"),
                               (ax[1, 0], "Height_m", "Height P99 (m)"),
                               (ax[1, 1], "CrownDiameter_m", "Crown diameter (m)")):
        values = [r[field] for r in rows if r[field] is not None]
        axes.hist(values, bins=min(25, max(1, int(np.sqrt(len(values))))), color="#337da8")
        axes.set(title=f"{title}; n={len(values)}", ylabel="Count")
    fig.savefig(root/"参数分布.png", dpi=140)
    fig.clear()
    # Reading the produced formats is part of successful completion, including frozen builds.
    with (root/"林业参数.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
        if [int(r["TreeID"]) for r in csv_rows] != [r["TreeID"] for r in rows]:
            raise ValueError("CSV 读回编号校验失败。")
    check_book = load_workbook(root/"林业参数.xlsx", read_only=True, data_only=True)
    try:
        values = list(check_book["林业参数"].values)
        if len(values) != len(rows)+1 or list(values[0]) != columns:
            raise ValueError("Excel 读回校验失败。")
        for expected, actual in zip(rows, values[1:]):
            for name, value in zip(columns, actual):
                want = expected[name]
                if isinstance(want, float):
                    if value is None or not np.isclose(want, value, rtol=1e-12, atol=1e-12):
                        raise ValueError("Excel 数值读回校验失败："+name)
                elif value != (want if want != "" else None):
                    raise ValueError("Excel 内容读回校验失败："+name)
    finally:
        check_book.close()
    with Image.open(root/"参数分布.png") as image:
        image.load()
        if min(image.size) < 100 or np.asarray(image.convert("RGB")).std() < 1:
            raise ValueError("分布图未正确生成。")


def _export(source, target, labels, heights, rows, check, notify):
    with laspy.open(source) as reader:
        header = reader.header.copy()
        used = set(header.point_format.dimension_names)
        fields = {}
        for name, dtype in (("metric_tree_id", "i8"), ("metric_height", "f4"), ("metric_review", "u1")):
            new = name
            suffix = 1
            while new in used:
                new = f"{name}_{suffix}"
                suffix += 1
            fields[name] = new
            header.add_extra_dim(laspy.ExtraBytesParams(name=new, type=dtype))
            used.add(new)
        ids = np.array([r["TreeID"] for r in rows], dtype=np.int64)
        review = np.array([2 if r["ReviewRequired"] else 1 for r in rows], dtype=np.uint8)
        start = 0
        with laspy.open(target, mode="w", header=header) as writer:
            for chunk in reader.chunk_iterator(CHUNK):
                check()
                end = start+len(chunk)
                out = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                for name in chunk.array.dtype.names:
                    out.array[name] = chunk.array[name]
                out[fields["metric_tree_id"]] = labels[start:end]
                out[fields["metric_height"]] = heights[start:end]
                status = np.zeros(len(chunk), dtype=np.uint8)
                valid = labels[start:end] > 0
                status[valid] = review[np.searchsorted(ids, labels[start:end][valid])]
                out[fields["metric_review"]] = status
                writer.write_points(out)
                start = end
                notify({"stage": "export", "points": start, "total": len(labels)})
            if reader.header.evlrs:
                writer.write_evlrs(reader.header.evlrs)
    # Original packed dimensions, CRS/VLRs and row order are preserved, then read back.
    with laspy.open(source) as original, laspy.open(target) as exported:
        if original.header.point_count != exported.header.point_count:
            raise ValueError("点云导出点数不一致。")
        np.testing.assert_array_equal(original.header.scales, exported.header.scales)
        np.testing.assert_array_equal(original.header.offsets, exported.header.offsets)
        def metadata(records):
            return [(v.user_id, v.record_id, v.description, v.record_data_bytes()) for v in (records or [])
                    if not (v.user_id == "LASF_Spec" and v.record_id == 4)
                    and v.user_id != "laszip encoded"]
        if (metadata(original.header.vlrs) != metadata(exported.header.vlrs)
                or metadata(original.header.evlrs) != metadata(exported.header.evlrs)):
            raise ValueError("点云坐标参考系或元数据未保留。")
        start = 0
        for chunk in original.chunk_iterator(CHUNK):
            check()
            out = exported.read_points(len(chunk))
            for name in chunk.array.dtype.names:
                if chunk.array[name].tobytes() != out.array[name].tobytes():
                    raise ValueError("点云原始属性校验失败："+name)
            end = start+len(chunk)
            np.testing.assert_array_equal(out[fields["metric_tree_id"]], labels[start:end])
            np.testing.assert_array_equal(out[fields["metric_height"]], heights[start:end])
            start = end
    return fields


def run_metrics(source, output_parent, binary="", *, coordinates_in_metres=False,
                mode="auto", label_field="", ground_mode="auto", terrain="FLAT",
                export_cloud=True, options=None, cancelled=lambda: False, notify=lambda data: None):
    """mode: auto/reuse/stems; ground_mode: auto/normalized. Never overwrite inputs."""
    options = (options or MetricOptions()).validate()
    if not coordinates_in_metres:
        raise ValueError("请确认 XYZ 单位为米。")
    if mode not in ("auto", "reuse", "stems") or ground_mode not in ("auto", "normalized"):
        raise ValueError("无效的处理模式。")
    if terrain not in ("FLAT", "RELIEF", "SLOPE"):
        raise ValueError("无效的地形类型。")
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        source = (source/"forest_instances.laz").resolve(strict=True)
    kind = identify_input(source)
    parent = Path(output_parent).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("请选择存在的输出文件夹。")
    def check():
        if cancelled():
            raise cli.SegmentationCancelled("任务已取消；本次中间结果不会作为完成结果发布。")
    check()
    # Preflight every optional output dependency before reading millions of points.
    import openpyxl  # noqa: F401
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    root = parent/("metrics_"+datetime.now().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8])
    root.mkdir()
    started = time.perf_counter()
    timings = {}
    def stage(name, **values):
        now = time.perf_counter()
        timings[name+"_start_s"] = round(now-started, 3)
        check()
        save_json(root/"progress.json", {"stage": name, **values})
        notify({"stage": name, "output_directory": str(root), **values})
    try:
        stage("reading")
        before = fingerprint(source)
        with ExitStack() as stack:
            if kind != "las":
                stack.enter_context(cli.execution(binary, cancelled, notify))
            normalized, input_info = normalize_input(source, root, kind)
            save_json(root/"input.json", input_info)
            work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="work-", dir=root)))
            def array(name, dtype, shape):
                value = np.memmap(work/name, mode="w+", dtype=dtype, shape=shape)
                stack.callback(value._mmap.close)
                return value
            with laspy.open(normalized) as reader:
                header = reader.header
                n = header.point_count
                if n == 0:
                    raise ValueError("输入点云为空。")
                if header.point_format.id in (4, 5, 9, 10):
                    raise ValueError("当前流程不支持带波形的 LAS 格式。")
                field = _label_field(header, label_field) if mode != "stems" else None
                if mode == "reuse" and not field:
                    raise ValueError("未找到 instance_id / TreeID 编号字段。请先分割，或改用自动树干模式。")
                origin = np.asarray(header.mins).copy()
                xyz, labels = array("xyz", "f8", (n, 3)), array("ids", "i8", (n,))
                ground_mask = array("ground", "?", (n,))
                names = set(header.point_format.dimension_names)
                cursor = 0
                for chunk in reader.chunk_iterator(CHUNK):
                    check()
                    end = cursor+len(chunk)
                    p = np.column_stack((chunk.x, chunk.y, chunk.z))
                    if not np.isfinite(p).all():
                        raise ValueError("点云包含非有限坐标。")
                    xyz[cursor:end] = p-origin
                    if field:
                        values = np.asarray(chunk[field])
                        if not np.isfinite(values).all() or (values < 0).any() or (values >= 2**63).any() or (values != np.floor(values)).any():
                            raise ValueError("单木编号必须为非负整数（0 为未分配），且小于 2^63。")
                        labels[cursor:end] = values
                    else:
                        labels[cursor:end] = 0
                    ground_mask[cursor:end] = (np.asarray(chunk.semantic_id) == 1) if "semantic_id" in names else np.asarray(chunk.classification) == 2
                    cursor = end
            stage("ground")
            heights, distances = array("height", "f4", (n,)), array("distance", "f4", (n,))
            if ground_mode == "normalized":
                ground_method = "user_declared_Z_is_height_above_ground"
                for start in range(0, n, CHUNK):
                    check()
                    heights[start:start+CHUNK] = xyz[start:start+CHUNK, 2]+origin[2]
                    distances[start:start+CHUNK] = 0
            else:
                if np.count_nonzero(ground_mask) >= 3:
                    ground = xyz[ground_mask]
                    ground_method = "semantic_id=1" if "semantic_id" in names else "LAS_classification=2"
                else:
                    capability = cli.probe(binary, cancelled, required=("CSF",))
                    if not capability["available"]:
                        raise ValueError("原始点云没有地面分类，所选 CloudCompare 必须提供 CSF 插件。")
                    save_json(root/"engine.json", capability)
                    with cli.execution(capability["binary"], cancelled, notify):
                        ground_root = root/"ground"
                        prepare(normalized, ground_root, .2)
                        run = cli.run(["-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(ground_root/"sample_local.ply"),
                                       "-CSF", "-SCENES", terrain, "-CLOTH_RESOLUTION", str(options.ground_cell),
                                       "-CLASS_THRESHOLD", "0.3", "-MAX_ITERATION", "500", "-EXPORT_GROUND"], ground_root/"csf-job", 600)
                        save_json(root/"csf-run.json", run)
                        ground = read_cloud(ground_root/"sample_local_ground_points.ply").xyz
                    ground_method = "CloudCompare_CSF_on_0.2m_sample"
                ground, tree, linear = _ground_model(ground, options)
                for start in range(0, n, CHUNK):
                    check()
                    p = xyz[start:start+CHUNK]
                    d, ids = tree.query(p[:, :2], workers=4)
                    z = linear(p[:, :2])
                    missing = ~np.isfinite(z)
                    z[missing] = ground[ids[missing], 2]
                    h = p[:, 2]-z
                    h[d > options.ground_support] = np.nan
                    heights[start:start+len(p)] = h
                    distances[start:start+len(p)] = d
                del ground, tree, linear
            stage("stems" if not field else "grouping")
            diagnostics = {}
            if not field:
                seeds = detect_stems(xyz, heights, options, check)
                candidate_count = len(seeds)
                breast = np.flatnonzero(np.isfinite(heights) & (np.abs(heights-options.breast_height) <= options.dbh_band/2))
                index = cKDTree(xyz[breast, :2])
                accepted, rejected = [], []
                for seed in seeds:
                    check()
                    ids = index.query_ball_point(seed, options.max_radius*2+.2)
                    fit = fit_circle(xyz[breast[ids], :2], options, check)
                    if fit["DBH_status"] == "ok" and np.linalg.norm(np.asarray(fit["center"])-seed) > options.max_radius+.1:
                        fit["DBH_status"] = "center_shift"
                    if fit["DBH_status"] == "ok":
                        accepted.append(fit["center"])
                    else:
                        rejected.append({"x":float(seed[0]+origin[0]), "y":float(seed[1]+origin[1]), "reason":fit["DBH_status"]})
                save_json(root/"rejected_stems.json", rejected)
                if not accepted:
                    raise ValueError("没有通过胸径质量校验的树干。可检查切片高度和容差，或使用已复核的单木编号提取。")
                # Deduplicate centers that converged to the same stem circle.
                accepted = np.asarray(accepted)
                accepted = accepted[np.lexsort((accepted[:, 0], accepted[:, 1]))]
                keep = []
                index = cKDTree(accepted)
                used = np.zeros(len(accepted), bool)
                for i in range(len(accepted)):
                    if not used[i]:
                        keep.append(i)
                        used[index.query_ball_point(accepted[i], options.stem_merge)] = True
                seeds = accepted[keep]
                index = cKDTree(seeds)
                for start in range(0, n, CHUNK):
                    check()
                    d, ids = index.query(xyz[start:start+CHUNK, :2], workers=4)
                    h = heights[start:start+CHUNK]
                    eligible = np.isfinite(h) & (h > options.stem_low) & (d <= options.max_assign_distance)
                    labels[start:start+CHUNK] = np.where(eligible, ids+1, 0)
                diagnostics = {"stem_grid_candidates": candidate_count, "accepted_stems": len(seeds),
                               "rejected_stems":len(rejected), "merged_stems":len(accepted)-len(seeds)}
            stage("measuring")
            valid_ids = np.flatnonzero(labels > 0)
            if not len(valid_ids):
                raise ValueError("点云没有正整数单木编号。")
            order = valid_ids[np.argsort(labels[valid_ids], kind="stable")]
            ids, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
            rows = []
            for i, (tree_id, start, count) in enumerate(zip(ids, starts, counts)):
                check()
                selected = order[start:start+count]
                rows.append(measure_tree(tree_id, xyz[selected], heights[selected], distances[selected], origin, options, check,
                                         stem_center=seeds[int(tree_id)-1] if not field else None))
                notify({"stage":"measuring", "trees":i+1, "total":len(ids)})
            del valid_ids, order
            summary = dict(tree_count=len(rows), accepted_dbh_count=sum(r["DBH_cm"] is not None for r in rows),
                           flagged_count=sum(r["ReviewRequired"] for r in rows), source_points=int(n),
                           assigned_points=int(np.count_nonzero(labels)), ground_supported_points=int(np.isfinite(heights).sum()),
                           mode="reuse" if field else "stems", label_field=field, ground_method=ground_method,
                           parameters=asdict(options), methods=METHODS, diagnostics=diagnostics,
                           source=str(source), source_sha256=before, working_origin=origin.tolist(),
                           terrain=terrain, all_results_require_review=True)
            stage("writing")
            _tables_and_plot(root, rows, summary)
            if export_cloud:
                stage("export")
                summary["export_fields"] = _export(normalized, root/"参数标注.laz", labels, heights, rows, check, notify)
                summary["source_fields_verified"] = True
            check()
        if fingerprint(source) != before:
            raise ValueError("输入文件在运行期间发生变化，请使用稳定的源文件重试。")
        summary.update(source_unchanged=True, seconds=round(time.perf_counter()-started, 3), stages=timings,
                       outputs={p.name: {"bytes":p.stat().st_size, "sha256":fingerprint(p)}
                                for p in root.iterdir() if p.suffix in (".csv", ".xlsx", ".png", ".laz")})
        save_json(root/"汇总.json", summary)
        stage("complete")
        result = {"output_directory":str(root), "summary":summary}
        # This marker is written last, only after format readback and source checks.
        save_json(root/"result.json", result)
        return result
    except Exception as exc:
        save_json(root/"progress.json", {"stage":"cancelled" if isinstance(exc, cli.SegmentationCancelled) else "failed", "error":str(exc)})
        raise
