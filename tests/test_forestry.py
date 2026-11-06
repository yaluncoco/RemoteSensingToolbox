import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import laspy
import numpy as np

from PointCloud import cloudcompare, forestry
from PointCloud.forestry_workflow import run_workflow


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


class ForestryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_full_export_preserves_source_attributes_and_tree_partition(self):
        source, root, meta = prepared_forest(self.root)
        summary = forestry.analyze(root, meta, 20, 2.)
        self.assertEqual(summary["candidate_tree_count"], 2)
        result = forestry.export(root)
        original, exported = laspy.read(source), laspy.read(result["full_cloud"])
        self.assertEqual(len(original.points), len(exported.points))
        for name in original.points.array.dtype.names:
            self.assertEqual(original.points.array[name].tobytes(), exported.points.array[name].tobytes())
        self.assertTrue(np.array_equal(exported.source_index, np.arange(len(original.points))))
        selected = []
        for tree_id in (1, 2):
            tree = laspy.read(root / "trees" / f"Tree_{tree_id:04d}.laz")
            self.assertTrue(np.all(tree.instance_id == tree_id))
            idx = np.asarray(tree.source_index)
            self.assertEqual(tree.points.array.tobytes(), exported.points.array[idx].tobytes())
            selected.extend(idx.tolist())
        self.assertEqual(sorted(selected), np.flatnonzero(exported.instance_id != 0).tolist())
        self.assertTrue(all(result["verification"].values()))
        preview = laspy.read(result["preview_cloud"])
        self.assertEqual(preview.header.vlrs.get_by_id("LASF_Projection")[0].record_data_bytes(),
                         original.header.vlrs.get_by_id("LASF_Projection")[0].record_data_bytes())

    def test_large_world_coordinates_with_zero_las_offsets(self):
        source, root, meta = prepared_forest(self.root, zero_offsets=True)
        self.assertEqual(meta["source_offsets"], [0., 0., 0.])
        with np.load(root / "sample.npz") as sample:
            self.assertLess(float(np.abs(sample["xyz"]).max()), 20.)
        forestry.analyze(root, meta, 20, 2.)
        result = forestry.export(root, extract_trees=False)
        original, exported = laspy.read(source), laspy.read(result["full_cloud"])
        for name in ("X", "Y", "Z"):
            np.testing.assert_array_equal(original[name], exported[name])

    def test_modified_geometry_is_rejected(self):
        _, root, meta = prepared_forest(self.root)
        with np.load(root / "vegetation.npz") as archive:
            xyz = archive["xyz"].copy()
        xyz[0, 0] += .01
        forestry.write_ply(root / "treeiso_local.ply", xyz, {"scalar_final_segs": np.ones(len(xyz), dtype="f4")})
        with self.assertRaisesRegex(ValueError, "changed coordinates"):
            forestry.analyze(root, meta, 20, 2.)

    def test_changed_source_and_missing_voxel_rejected(self):
        source, root, _ = prepared_forest(self.root)
        with source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            forestry.validate_source(root)
        with self.assertRaisesRegex(ValueError, "missing"):
            forestry.locate_keys(np.array([1, 3]), np.array([1, 2]))

    def test_no_tree_files_option(self):
        _, root, meta = prepared_forest(self.root)
        forestry.analyze(root, meta, 20, 2.)
        result = forestry.export(root, extract_trees=False)
        self.assertEqual(result["extracted_trees"], 0)
        self.assertFalse((root / "trees").exists())
        self.assertTrue(Path(result["full_cloud"]).is_file())

    def test_invalid_configured_path_does_not_fallback(self):
        with patch.dict(os.environ, {"CLOUDCOMPARE_PATH": str(self.root / "missing.exe")}):
            self.assertIsNone(cloudcompare.find_binary())
        self.assertIsNone(cloudcompare.find_binary(str(self.root / "missing.exe")))

    def test_metres_confirmation_required_before_engine_launch(self):
        source = self.root / "empty.las"
        source.touch()
        with patch.object(cloudcompare, "probe") as probe:
            with self.assertRaisesRegex(ValueError, "单位"):
                run_workflow(source, self.root, "missing.exe")
            probe.assert_not_called()

    def test_missing_plugin_rejected_before_output_created(self):
        source = self.root / "empty.las"
        source.write_bytes(b"LASF")
        with patch.object(cloudcompare, "probe", return_value={"available": False, "missing": ["TREEISO"]}):
            with self.assertRaisesRegex(RuntimeError, "TREEISO"):
                run_workflow(source, self.root, "missing.exe", coordinates_in_metres=True)
        self.assertEqual(list(self.root.glob("forest_*")), [])

    def test_cancel_terminates_running_child(self):
        binary = self.root / "CloudCompare.exe"
        binary.touch()
        started = time.monotonic()
        class Child:
            killed = False
            waited = False
            def poll(self):
                return 1 if self.killed else None
            def kill(self):
                self.killed = True
            def wait(self):
                self.waited = True
        child = Child()
        with cloudcompare.execution(binary, lambda: time.monotonic() - started > .15), \
                patch.object(cloudcompare, "_popen", return_value=child):
            with self.assertRaises(cloudcompare.SegmentationCancelled):
                cloudcompare.run(["-HELP"], self.root / "job")
        self.assertTrue(child.killed and child.waited)

    def test_failed_task_has_no_success_manifest_and_uses_separate_directory(self):
        source = self.root / "empty.las"
        source.write_bytes(b"LASF")
        binary = self.root / "CloudCompare.exe"
        binary.touch()
        probe = {"available": True, "binary": str(binary), "missing": [], "commands": []}
        with patch.object(cloudcompare, "probe", return_value=probe), \
                patch.object(forestry, "segment", side_effect=RuntimeError("test failure")):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    run_workflow(source, self.root, binary, coordinates_in_metres=True)
        outputs = list(self.root.glob("forest_*"))
        self.assertEqual(len(outputs), 2)
        for root in outputs:
            self.assertFalse((root / "result.json").exists())
            self.assertEqual(json.loads((root / "progress.json").read_text())["stage"], "failed")


@unittest.skipUnless(os.environ.get("TOOLBOX_TEST_CLOUDCOMPARE"), "Set TOOLBOX_TEST_CLOUDCOMPARE to run the installed engine")
class InstalledEngineTests(unittest.TestCase):
    def test_actual_plugins_and_complete_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "中文 空格"
            root.mkdir()
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
            header.offsets = [430000., 4460000., 90.]
            header.scales = [.001]*3
            source = laspy.LasData(header)
            source.x, source.y, source.z = (xyz + header.offsets).T
            source.intensity = np.arange(len(xyz), dtype="u2")
            input_path = root / "人工林.laz"
            source.write(input_path)
            # Valid compressed LAS content with a misleading extension must also work.
            renamed = root / "人工林.bin"
            input_path.rename(renamed)
            input_path = renamed
            result = run_workflow(input_path, root, os.environ["TOOLBOX_TEST_CLOUDCOMPARE"],
                                  coordinates_in_metres=True, min_sample_points=20)
            self.assertGreater(result["segmentation"]["candidate_tree_count"], 0)
            self.assertTrue(all(result["export"]["verification"].values()))
            self.assertEqual(result["export"]["source_point_count"], len(xyz))
            print("Real CloudCompare:", result["segmentation"]["candidate_tree_count"], "candidates;", len(xyz), "source points")


if __name__ == "__main__":
    unittest.main()
