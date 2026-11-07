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


from Toolbox.forestry_self_test import prepared_forest


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

    def test_render_failure_does_not_publish_segmentation_manifest(self):
        _, root, meta = prepared_forest(self.root)
        with patch.object(forestry, "render_results", side_effect=ModuleNotFoundError("backend_svg")):
            with self.assertRaisesRegex(ModuleNotFoundError, "backend_svg"):
                forestry.analyze(root, meta, 20, 2.)
        self.assertFalse((root / "segmentation.json").exists())
        self.assertFalse((root / "export.json").exists())

    def test_reuse_repairs_legacy_missing_plots_without_rerunning_engine(self):
        from Toolbox.forestry_self_test import verify_plots
        source, root, meta = prepared_forest(self.root)
        summary = forestry.analyze(root, meta, 20, 2.)
        labels = (root / "sample_labels.npz").read_bytes()
        (root / "instance_map.svg").unlink()
        (root / "tree_gallery.png").write_bytes(b"")
        with patch.object(forestry, "_ground_and_vegetation") as ground, patch.object(forestry, "_treeiso") as treeiso:
            restored = forestry.segment(source, root, True, min_sample_points=20, reuse_existing=True)
            ground.assert_not_called()
            treeiso.assert_not_called()
        self.assertEqual(restored, summary)
        self.assertEqual((root / "sample_labels.npz").read_bytes(), labels)
        verify_plots(root)

    def test_failed_legacy_plot_repair_removes_incomplete_manifest(self):
        source, root, meta = prepared_forest(self.root)
        forestry.analyze(root, meta, 20, 2.)
        (root / "instance_map.svg").unlink()
        with patch.object(forestry, "render_results", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                forestry.segment(source, root, True, min_sample_points=20, reuse_existing=True)
        self.assertFalse((root / "segmentation.json").exists())

    def test_missing_output_runtime_fails_before_engine_or_output_creation(self):
        source = self.root / "input.las"
        source.write_bytes(b"LASF")
        with patch.object(forestry, "check_output_runtime", side_effect=RuntimeError("backend_svg")), \
                patch.object(cloudcompare, "probe") as probe:
            with self.assertRaisesRegex(RuntimeError, "backend_svg"):
                run_workflow(source, self.root, "unused.exe", coordinates_in_metres=True)
            probe.assert_not_called()
        self.assertEqual(list(self.root.glob("forest_*")), [])

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
