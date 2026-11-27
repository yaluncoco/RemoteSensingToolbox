"""Executable forestry checks shared by source and frozen application self-tests."""
from pathlib import Path
import json
import tempfile
import time
import xml.etree.ElementTree as ET

import laspy
import numpy as np

from PointCloud import forestry


def prepared_forest(root, zero_offsets=False):
    ground = np.array([[x, y, 0.] for x in np.arange(.01, 10, .4) for y in np.arange(.01, 2, .4)])
    trees = np.array([[base + x, y, z] for base in (1., 7.) for x in (.01, .25, .49)
                      for y in (.01, .25, .49) for z in np.arange(.41, 4.5, .21)])
    representative = np.concatenate((ground, trees))
    xyz = np.repeat(representative, 3, axis=0)
    xyz[:, 0] += np.tile([0., .002, .004], len(representative))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [.001] * 3
    header.offsets = [434917.6331222008, 4460951.865954002, 92.4817]
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = (xyz + header.offsets).T
    cloud.intensity = np.arange(len(xyz), dtype="u2")
    cloud.classification = np.full(len(xyz), 7, dtype="u1")
    cloud.synthetic = np.ones(len(xyz), dtype="u1")
    cloud.withheld = (np.arange(len(xyz)) % 2).astype("u1")
    cloud.gps_time = np.arange(len(xyz), dtype=float) + 123.25
    cloud.red, cloud.green, cloud.blue = [np.full(len(xyz), v, dtype="u2") for v in (100, 200, 300)]
    if zero_offsets:
        cloud.change_scaling(scales=[.01]*3, offsets=[0.]*3)
    from laspy.vlrs.known import WktCoordinateSystemVlr
    # Deliberately preserve, rather than parse or reinterpret, a supplied CRS record.
    cloud.header.vlrs.append(WktCoordinateSystemVlr('LOCAL_CS["metre survey",UNIT["metre",1]]'))
    source = root / "林地 原始点云.laz"
    cloud.write(source)
    output = root / "单木 结果"
    meta = forestry.prepare(source, output, .2)
    with np.load(output / "sample.npz") as archive:
        sample = archive["xyz"]
    mask = sample[:, 2] > .3
    vegetation = sample[mask]
    np.savez_compressed(output / "vegetation.npz", xyz=vegetation, height=vegetation[:, 2],
                        sample_index=np.flatnonzero(mask).astype("u4"))
    forestry.write_ply(output / "sample_local_ground_points.ply", sample[~mask])
    forestry.write_ply(output / "treeiso_local.ply", vegetation[::-1],
                       {"scalar_final_segs": np.where(vegetation[::-1, 0] < 5, 7, 19).astype("f4")})
    return source, output, meta


def verify_plots(root):
    from PIL import Image
    sizes = {}
    for name in ("instance_overview.png", "tree_gallery.png"):
        with Image.open(root / name) as image:
            image.load()
            if image.format != "PNG" or min(image.size) < 100:
                raise AssertionError(f"Invalid forestry figure: {name}")
            if np.asarray(image.convert("RGB")).std() < 1:
                raise AssertionError(f"Empty forestry figure: {name}")
            sizes[name] = list(image.size)
    svg = ET.parse(root / "instance_map.svg").getroot()
    if svg.tag != "{http://www.w3.org/2000/svg}svg" or not list(svg):
        raise AssertionError("Invalid SVG instance map")
    sizes["instance_map.svg"] = (root / "instance_map.svg").stat().st_size
    return sizes


def check_postprocessing():
    """No external engine: real analysis, figures and full LAZ/tree exports."""
    with tempfile.TemporaryDirectory(prefix="toolbox-forest-outputs-") as directory:
        source, root, meta = prepared_forest(Path(directory))
        forestry.check_output_runtime()
        summary = forestry.analyze(root, meta, 20, 2.)
        plots = verify_plots(root)
        exported = forestry.export(root)
        original, full = laspy.read(source), laspy.read(exported["full_cloud"])
        if summary["candidate_tree_count"] != 2 or exported["extracted_trees"] != 2:
            raise AssertionError("Fixture did not export both tree candidates")
        for name in original.points.array.dtype.names:
            np.testing.assert_array_equal(original[name], full[name])
        np.testing.assert_array_equal(full.source_index, np.arange(len(original.points)))
        indices = []
        for tree_id in (1, 2):
            tree = laspy.read(root / "trees" / f"Tree_{tree_id:04d}.laz")
            idx = np.asarray(tree.source_index)
            if tree.points.array.tobytes() != full.points.array[idx].tobytes():
                raise AssertionError("Tree records differ from full export")
            if not np.all(tree.instance_id == tree_id):
                raise AssertionError("Wrong instance IDs in tree file")
            indices.extend(idx.tolist())
        np.testing.assert_array_equal(sorted(indices), np.flatnonzero(full.instance_id))
        if not all(exported["verification"].values()):
            raise AssertionError("Forestry export verification failed")
        return {"fixture": "known segmentation (no external engine)", "plots": plots,
                "source_points": len(original.points), "trees": 2, "verification": exported["verification"]}


def check_refinement():
    """Both merge and split errors with known tree identity, through full export."""
    with tempfile.TemporaryDirectory(prefix="toolbox-stem-refine-") as directory:
        root = Path(directory)
        ground = np.array([[x, y, 0.] for x in np.arange(-2., 6., .2) for y in np.arange(-2., 2., .2)])
        parts, truth = [ground], [np.zeros(len(ground), dtype="u1")]
        h = np.arange(.45, 5.05, .025)
        theta = np.arange(60)*2*np.pi/60
        radius = .16+np.clip((h-2.5)/1.5, 0, 1)*.7
        for tree_id, base in enumerate((0., 1.9), 1):
            p = np.column_stack((np.repeat(base+.4*(h-1.3), len(theta))+np.repeat(radius, len(theta))*np.tile(np.cos(theta), len(h)),
                                 np.repeat(radius, len(theta))*np.tile(np.sin(theta), len(h)), np.repeat(h, len(theta))))
            parts.append(p)
            truth.append(np.full(len(p), tree_id, dtype="u1"))
        xyz = np.concatenate(parts)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets, header.scales = [430000., 4460000., 90.], [.001]*3
        header.add_extra_dim(laspy.ExtraBytesParams(name="known_tree", type="u1"))
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = (xyz+header.offsets).T
        cloud.known_tree = np.concatenate(truth)
        cloud.intensity = np.arange(len(xyz), dtype="u2")
        source = root/"已知双树.laz"
        cloud.write(source)
        job = root/"校正结果"
        meta = forestry.prepare(source, job, .12)
        with np.load(job/"sample.npz") as sample:
            p = sample["xyz"]
            known = np.asarray(cloud.known_tree)[sample["source_index"]]
        vegetation = known > 0
        heights = p[vegetation, 2]
        # Both trees are initially merged; tree 2's upper crown is also wrongly
        # detached. No known_tree values are passed to the correction algorithm.
        bad_labels = ((known[vegetation] == 2) & (heights > 3.8)).astype("u4")
        np.savez_compressed(job/"vegetation.npz", xyz=p[vegetation], height=heights,
                            sample_index=np.flatnonzero(vegetation).astype("u4"))
        forestry.write_ply(job/"sample_local_ground_points.ply", p[~vegetation])
        forestry.write_ply(job/"treeiso_local.ply", p[vegetation], {"scalar_final_segs": bad_labels.astype("f4")})
        summary = forestry.analyze(job, meta, 20, 2.)
        correction = summary["refinement"]
        if correction["active_anchor_count"] != 2 or correction["split_segment_count"] != 1 or correction["merged_group_count"] != 1:
            raise AssertionError("Known split/merge errors were not corrected: "+str(correction))
        if summary["candidate_tree_count"] != 2:
            raise AssertionError("Correction did not retain exactly two trees")
        result = forestry.export(job)
        exported = laspy.read(result["full_cloud"])
        truth = np.asarray(exported.known_tree)
        labels = np.asarray(exported.instance_id)
        mask = truth > 0
        def score(a, b):
            return sum(np.unique(b[a == k], return_counts=True)[1].max() for k in np.unique(a))/len(a)
        agreement = min(score(truth[mask], labels[mask]), score(labels[mask], truth[mask]))
        if agreement < .97:
            raise AssertionError(f"Known partition agreement is only {agreement:.3f}")
        from PIL import Image
        with Image.open(job/"refinement_comparison.png") as image:
            image.load()
            if np.asarray(image.convert("RGB")).std() < 1:
                raise AssertionError("Blank refinement comparison")
        return {"fixture": "two leaning trees with touching crowns; merged lower parts and detached upper crown",
                "source_points": len(xyz), "known_trees": 2, "partition_agreement": agreement,
                "refinement": correction, "verification": result["verification"]}


def synthetic_forest(root):
    ground = np.array([[x, y, 0.] for x in np.arange(-3, 15, .2) for y in np.arange(-3, 9, .2)])
    rng = np.random.default_rng(12)
    parts = [ground]
    for x, y in ((0., 0.), (6., 0.), (12., 0.), (0., 6.), (6., 6.), (12., 6.)):
        angle = rng.uniform(0, 2*np.pi, 1500)
        stem = np.column_stack((x + .18*np.cos(angle), y + .18*np.sin(angle), rng.uniform(.4, 4., 1500)))
        crown = rng.normal(size=(4500, 3)) * [.65, .65, .8] + [x, y, 4.5]
        parts += [stem, crown]
    xyz = np.concatenate(parts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets, header.scales = [430000., 4460000., 90.], [.001]*3
    source = laspy.LasData(header)
    source.x, source.y, source.z = (xyz + header.offsets).T
    source.intensity = np.arange(len(xyz), dtype="u2")
    path = root / "人工林.laz"
    source.write(path)
    return path


def check_gui_workflow(app, binary, source=None, output_parent=None, screenshot=None):
    """Click the actual window's start button and await its real QThread result."""
    from PyQt5.QtCore import QEvent, QSettings, QThread
    from PyQt5.QtWidgets import QMessageBox
    from PointCloud.forestry_gui import ForestryWindow
    with tempfile.TemporaryDirectory(prefix="toolbox-forest-gui-") as directory:
        root = Path(directory) / "中文 空格"
        root.mkdir()
        source = Path(source) if source else synthetic_forest(root)
        parent = Path(output_parent) if output_parent else root
        settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
        window = ForestryWindow(settings)
        messages = []
        warning = QMessageBox.warning
        QMessageBox.warning = lambda _parent, title, message, *args: messages.append(f"{title}: {message}") or QMessageBox.Ok
        try:
            window.binary_edit.setText(str(binary))
            window.source_edit.setText(str(source))
            window.output_edit.setText(str(parent))
            window.metres.setChecked(True)
            window.show()
            window.run_button.click()
            deadline = time.monotonic() + 3600
            while window.worker is not None:
                app.processEvents()
                QThread.msleep(20)
                if time.monotonic() > deadline:
                    window.worker.stop()
                    raise TimeoutError("Forestry GUI workflow exceeded one hour")
            if messages or window.result is None:
                raise AssertionError("; ".join(messages) or "GUI produced no result")
            if not all(button.isEnabled() for button in (window.run_button, window.folder_button, window.preview_button)):
                raise AssertionError("Completed GUI did not enable its result actions")
            if window.cancel_button.isEnabled() or window.progress.value() != 1:
                raise AssertionError("Completed GUI still reports an active job")
            result = window.result
            job = Path(result["output_directory"])
            plots = verify_plots(job)
            correction = result["segmentation"]["refinement"]
            if not correction["enabled"] or not (job/"refinement.json").is_file() or not (job/"refinement_comparison.png").is_file():
                raise AssertionError("Default GUI workflow did not run stem correction")
            if not all(result["export"]["verification"].values()):
                raise AssertionError("Workflow export verification failed")
            training = result.get("training")
            if not training or training["format"] != "txt" or training["point_count"] != result["export"]["source_point_count"]:
                raise AssertionError("Default workflow did not export aligned TXT training data")
            if not all(training["verification"].values()):
                raise AssertionError("Training point/label alignment failed")
            if len(list((job / "trees").glob("*.laz"))) != result["segmentation"]["candidate_tree_count"]:
                raise AssertionError("Missing tree output files")
            if json.loads((job / "progress.json").read_text(encoding="utf8"))["stage"] != "complete":
                raise AssertionError("Workflow did not reach completion")
            if not (job / "result.json").is_file():
                raise AssertionError("Workflow completion manifest missing")
            if screenshot:
                app.processEvents()
                if not window.grab().save(str(screenshot)):
                    raise AssertionError("Could not save completed forestry window")
            return {"workflow": result, "plots": plots, "gui_status": window.status_label.text(),
                    "result_actions_enabled": True, "output_retained": output_parent is not None}
        finally:
            if window.worker is not None:
                window.worker.stop()
                while window.worker is not None:
                    app.processEvents()
                    QThread.msleep(20)
            QMessageBox.warning = warning
            window.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
