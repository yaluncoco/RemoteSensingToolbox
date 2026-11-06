"""Public application service for a complete, isolated single-tree job."""
from datetime import datetime
from pathlib import Path
import uuid

from PointCloud import cloudcompare
from PointCloud.forestry_input import identify_input, normalize_input


def run_workflow(source, output_parent, binary, *, coordinates_in_metres=False,
                 resolution=.2, terrain="FLAT", min_sample_points=100, min_height=2.,
                 extract_trees=True, cancelled=lambda: False, notify=lambda data: None):
    from PointCloud import forestry
    source = Path(source).expanduser().resolve(strict=True)
    if not coordinates_in_metres:
        raise ValueError("请确认点云的 XYZ 坐标单位均为米。")
    kind = identify_input(source)
    parent = Path(output_parent).expanduser().resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("请选择输出文件夹。")
    notify({"stage": "checking_plugins"})
    capability = cloudcompare.probe(binary, cancelled)
    if not capability["available"]:
        raise RuntimeError("所选 CloudCompare 缺少命令行插件：" + ", ".join(capability["missing"]) +
                           "。请选择带有 CSF 和 TreeIso 的安装目录。")
    root = parent / ("forest_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    root.mkdir(exist_ok=False)
    notify({"stage": "starting", "output_directory": str(root)})
    forestry.save_json(root / "engine.json", capability)
    with cloudcompare.execution(capability["binary"], cancelled, notify):
        try:
            normalized, input_details = normalize_input(source, root, kind)
            forestry.save_json(root / "input.json", input_details)
            summary = forestry.segment(str(normalized), str(root), True, resolution, terrain,
                                       min_sample_points, min_height)
            exported = forestry.export(root, extract_trees)
            if kind == "cloudcompare_bin":
                from PointCloud.forestry_io import fingerprint
                if fingerprint(source) != input_details["original_sha256"]:
                    raise ValueError("原始 BIN 在任务过程中发生变化，请重新运行。")
        except Exception as exc:
            state = "cancelled" if isinstance(exc, cloudcompare.SegmentationCancelled) else "failed"
            forestry.save_json(root / "progress.json", {"stage": state, "error": str(exc)})
            raise
    result = {"output_directory": str(root), "segmentation": summary, "export": exported,
              "binary": capability["binary"], "input": input_details}
    forestry.save_json(root / "result.json", result)
    return result
