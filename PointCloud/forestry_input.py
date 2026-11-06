"""Identify file contents and normalize single-cloud CloudCompare BIN inputs."""
from pathlib import Path
import re

import laspy
import numpy as np
from plyfile import PlyData

from PointCloud import cloudcompare as cli
from PointCloud.forestry_io import fingerprint


def identify_input(path):
    with Path(path).open("rb") as stream:
        signature = stream.read(4)
    if signature == b"LASF":
        return "las"
    if signature[:3] == b"CCB" and signature[3:] in (b"2", b"3"):
        return "cloudcompare_bin"
    raise ValueError(
        f"文件内容不是 LAS/LAZ 或支持的 CloudCompare BIN（文件头 {signature!r}）。"
        "修改扩展名不会转换格式；请在 CloudCompare 中选择 LAS/LAZ 格式另存。"
    )


def _chunks(vertex, size=500_000):
    for start in range(0, len(vertex), size):
        cli.check_cancel()
        yield start, vertex[start:start + size]


def _properties(vertex):
    names = vertex.dtype.names or ()
    if not set("xyz").issubset(names) or not len(vertex):
        raise ValueError("BIN 没有可处理的三维点云。")
    for name in names:
        dtype = vertex.dtype.fields[name][0]
        if dtype.kind not in "iuf" or dtype.subdtype is not None:
            raise ValueError(f"暂不支持 BIN 转出字段类型：{name} / {dtype}")
    fields = {}
    for index, name in enumerate(names):
        # Stable, ASCII and collision-free names fit LAS's 32-byte name limit.
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", name.removeprefix("scalar_"))[:20]
        fields[name] = f"cc{index:03d}_{suffix}"
    return fields


def _standard_fields(points, block):
    # Always preserve the original PLY property as an extra dimension as well.
    for name in ("red", "green", "blue"):
        if name in block.dtype.names:
            values = block[name]
            if values.dtype == np.dtype("u1"):
                points[name] = values.astype("u2") * 257
    for source, target, maximum in (("scalar_Intensity", "intensity", 65535),
                                     ("scalar_Return_Number", "return_number", 7),
                                     ("scalar_Number_Of_Returns", "number_of_returns", 7)):
        if source in block.dtype.names:
            values = block[source]
            valid = np.isfinite(values) & (values >= 0) & (values <= maximum) & (values == np.floor(values))
            points[target] = np.where(valid, values, 0).astype("u2")
    if "scalar_Gps_Time" in block.dtype.names:
        points.gps_time = np.where(np.isfinite(block["scalar_Gps_Time"]), block["scalar_Gps_Time"], 0.)


def ply_to_laz(ply_path, target, notify=lambda data: None):
    """Keep every exported numeric property, including exact floating point XYZ.

    LAS integer XYZ is quantized to <=1 micrometre per axis for ordinary plot
    extents. Exact CloudCompare-exported coordinates also remain in extra fields.
    """
    ply = PlyData.read(ply_path, mmap="r")
    if any(element.name != "vertex" and element.count for element in ply.elements):
        raise ValueError("BIN 包含网格等非点云实体，请先另存单个点云。")
    vertex = ply["vertex"].data
    fields = _properties(vertex)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for _, block in _chunks(vertex):
        xyz = np.column_stack([block[name] for name in "xyz"])
        if not np.isfinite(xyz).all():
            raise ValueError("BIN 坐标包含 NaN/Inf，无法生成可靠的分割输入。")
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = minimum
    header.scales = np.maximum(1e-6, (maximum - minimum) / (np.iinfo(np.int32).max - 100))
    for name, extra in fields.items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=extra, type=vertex.dtype.fields[name][0].newbyteorder("=")))
    with laspy.open(target, mode="w", header=header) as writer:
        for start, block in _chunks(vertex):
            points = laspy.ScaleAwarePointRecord.zeros(len(block), header=header)
            points.x, points.y, points.z = [block[name] for name in "xyz"]
            for name, extra in fields.items():
                points[extra] = block[name]
            _standard_fields(points, block)
            writer.write_points(points)
            notify({"stage": "converting_bin", "points": start + len(block), "total": len(vertex)})
    max_error = np.zeros(3)
    with laspy.open(target) as reader:
        if reader.header.point_count != len(vertex):
            raise ValueError("BIN 转换后的点数不一致。")
        for start, block in _chunks(vertex):
            points = reader.read_points(len(block))
            for name, extra in fields.items():
                # Compare bits, preserving NaNs and signed zero in scalar properties too.
                original = np.asarray(block[name], dtype=vertex.dtype.fields[name][0].newbyteorder("="))
                if np.asarray(points[extra]).tobytes() != original.tobytes():
                    raise ValueError(f"BIN 转换字段校验失败：{name}")
            xyz = np.column_stack([block[name] for name in "xyz"])
            restored = np.column_stack((points.x, points.y, points.z))
            max_error = np.maximum(max_error, np.abs(restored - xyz).max(axis=0))
            tolerance = header.scales * .501 + np.spacing(np.abs(xyz).max(axis=0)) * 2
            if np.any(max_error > tolerance):
                raise ValueError("BIN 转换后的 LAS 坐标误差超过记录的精度。")
            expected = laspy.ScaleAwarePointRecord.zeros(len(block), header=header)
            _standard_fields(expected, block)
            for name in ("red", "green", "blue", "intensity", "return_number", "number_of_returns", "gps_time"):
                if not np.array_equal(np.asarray(expected[name]), np.asarray(points[name])):
                    raise ValueError(f"BIN 标准字段校验失败：{name}")
            notify({"stage": "validating_bin", "points": start + len(block), "total": len(vertex)})
    return {"point_count": len(vertex), "property_mapping": fields,
            "scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
            "max_coordinate_rounding_error": max_error.tolist(),
            "verification": {"all_exported_properties_compared": True, "point_order_preserved": True,
                             "exact_exported_xyz_retained_as_extra_fields": True, "standard_fields_compared": True},
            "limitations": ["BIN scene hierarchy and non-point metadata are not represented in LAS",
                            "No CRS is inferred; CloudCompare-exported global XYZ are used",
                            "Preservation is checked against CloudCompare's PLY export, not the BIN serializer"]}


def normalize_input(source, root, kind):
    from PointCloud.forestry import progress, save_json
    if kind == "las":
        return source, {"original_source": str(source), "detected_format": kind, "converted": False}
    work = root / "input-conversion"
    work.mkdir(exist_ok=False)
    staged = work / "source.bin"
    progress(root, "reading_bin")
    before = fingerprint(source)
    with source.open("rb") as src, staged.open("xb") as dst:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            cli.check_cancel()
            dst.write(block)
    if fingerprint(staged) != before:
        raise ValueError("输入在复制过程中发生变化，请使用稳定的源文件。")
    try:
        result = cli.run(["-C_EXPORT_FMT", "PLY", "-PLY_EXPORT_FMT", "BINARY_LE", "-O", str(staged),
                          "-SAVE_CLOUDS", "FILE", "../cloud.ply"], work / "job", timeout=900)
    except RuntimeError as exc:
        if "file names, but there are" in str(exc):
            raise ValueError("BIN 包含多个点云，请先在 CloudCompare 中单独另存需要分割的点云。") from exc
        raise
    save_json(work / "cloudcompare-run.json", result)
    log = Path(result["log"]).read_text(encoding="utf8", errors="replace")
    clouds = re.findall(r"Found one cloud with (\d+) points", log)
    if len(clouds) != 1 or re.search(r"Found (?:one|\d+) mesh", log, re.I):
        raise ValueError("BIN 必须只包含一个点云；请在 CloudCompare 中单独另存需要分割的点云。")
    target = work / "normalized.laz"
    details = ply_to_laz(work / "cloud.ply", target,
                         lambda value: progress(root, **value))
    if details["point_count"] != int(clouds[0]):
        raise ValueError("BIN 点数与 CloudCompare 转出点数不一致。")
    if fingerprint(source) != before:
        raise ValueError("输入在转换过程中发生变化。")
    details.update(original_source=str(source), original_sha256=before, detected_format=kind,
                   converted=True, normalized_source=str(target), normalized_sha256=fingerprint(target),
                   source_unchanged=True)
    save_json(root / "input.json", details)
    # These are newly created internal copies; preserve the normalized source and provenance.
    staged.unlink()
    (work / "cloud.ply").unlink()
    return target, details
