import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np

from PointCloud import forestry_training as training
from PointCloud.cloudcompare import SegmentationCancelled
from Toolbox.training_self_test import training_fixture


class TrainingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job, self.cloud = training_fixture(self.root)

    def test_txt_preserves_duplicate_points_order_world_precision_and_labels(self):
        result = training.export_training(self.job, self.root / "training", "both", chunk_size=2)
        data = np.loadtxt(self.root / "training" / "points_labels.txt")
        np.testing.assert_array_equal(data[:, :3], np.column_stack((self.cloud.x, self.cloud.y, self.cloud.z)))
        np.testing.assert_array_equal(data[:, 3], self.cloud.intensity)
        np.testing.assert_array_equal(data[:, 4:7], np.column_stack((self.cloud.red, self.cloud.green, self.cloud.blue)))
        np.testing.assert_array_equal(data[:, 7:11], np.column_stack([self.cloud[n] for n in training.LABEL_COLUMNS[:4]]))
        np.testing.assert_array_equal(data[:, 11], [-1, 0, 1, 1, -1, -1, -1])
        np.testing.assert_array_equal(data[:, 12], [-1, -1, 1, 1, -1, -1, -1])
        np.testing.assert_array_equal(data[:, 13], [0, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(np.load(self.root / "training" / "points.npy"), data[:, :7])
        np.testing.assert_array_equal(np.load(self.root / "training" / "labels.npy"), data[:, 7:])
        self.assertEqual(result["logical_content"]["eligible_points"], 3)
        self.assertIn("pseudo", result["label_provenance"])
        self.assertEqual(result["instance_map"][1]["instance_id"], 4)

    def test_all_format_directions_preserve_logical_content_across_chunk_sizes(self):
        a = training.export_training(self.job, self.root / "a", "txt", chunk_size=3)
        b = training.export_training(self.root / "a", self.root / "b", "numpy", chunk_size=2)
        c = training.export_training(self.root / "b", self.root / "c", "txt", chunk_size=5)
        self.assertEqual(a["logical_content"], b["logical_content"])
        self.assertEqual(a["logical_content"], c["logical_content"])
        self.assertEqual((self.root / "a/points_labels.txt").read_bytes(), (self.root / "c/points_labels.txt").read_bytes())

    def test_source_label_modification_is_detected(self):
        cloud = laspy.read(self.job / "forest_instances.laz")
        cloud.instance_id[2] = 4
        cloud.write(self.job / "forest_instances.laz")
        with self.assertRaisesRegex(ValueError, "不符"):
            training.export_training(self.job, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_changed_txt_labels_are_not_silently_converted(self):
        training.export_training(self.job, self.root / "a")
        path = self.root / "a/points_labels.txt"
        path.write_text(path.read_text().replace(" 2 2 ", " 2 4 "), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "已变化"):
            training.export_training(self.root / "a", self.root / "out", "numpy")

    def test_shuffled_indices_rejected_even_with_updated_file_hash(self):
        training.export_training(self.job, self.root / "a")
        path = self.root / "a/points_labels.txt"
        lines = path.read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines)+"\n", encoding="utf8")
        manifest = training._read_json(self.root / "a/manifest.json")
        manifest["files"][path.name] = training._hash(path, lambda: False)
        training._save_json(self.root / "a/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "错序"):
            training.export_training(self.root / "a", self.root / "out", "numpy")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob(".training-*")))

    def test_cancel_partial_export_leaves_source_and_no_completed_dataset(self):
        original = (self.job / "forest_instances.laz").read_bytes()
        cancelled = [False]
        def notify(data):
            if data["stage"] == "writing_training":
                cancelled[0] = True
        with self.assertRaises(SegmentationCancelled):
            training.export_training(self.job, self.root / "out", "both", chunk_size=2,
                                     cancelled=lambda: cancelled[0], notify=notify)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob(".training-*")))
        self.assertEqual(original, (self.job / "forest_instances.laz").read_bytes())

    def test_existing_output_is_never_replaced(self):
        output = self.root / "output"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            training.export_training(self.job, output)
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_mismatched_numpy_lengths_rejected(self):
        training.export_training(self.job, self.root / "a", "numpy")
        path = self.root / "a/labels.npy"
        np.save(path, np.load(path)[:-1])
        manifest = training._read_json(self.root / "a/manifest.json")
        manifest["files"][path.name] = training._hash(path, lambda: False)
        training._save_json(self.root / "a/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "长度不一致"):
            training.export_training(self.root / "a", self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_missing_rgb_is_explicit(self):
        old = self.cloud
        header = laspy.LasHeader(point_format=1, version="1.2")
        for name, dtype in (("semantic_id", "u2"), ("instance_id", "u4"), ("source_index", "u4"), ("review_status", "u1")):
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        cloud = laspy.LasData(header)
        for name in ("X", "Y", "Z", "intensity", *training.LABEL_COLUMNS[:4]):
            cloud[name] = old[name]
        cloud.write(self.job / "forest_instances.laz")
        info = training._read_json(self.job / "export.json")
        info["full_cloud_sha256"] = training._hash(self.job / "forest_instances.laz", lambda: False)
        training._save_json(self.job / "export.json", info)
        result = training.export_training(self.job, self.root / "a", "numpy")
        self.assertFalse(result["rgb_available"])
        self.assertTrue(np.all(np.load(self.root / "a/points.npy")[:, 4:] == 0))

    def test_partial_disk_failure_has_no_success_manifest(self):
        with patch.object(np, "savetxt", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                training.export_training(self.job, self.root / "out", "both")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(list(self.root.glob(".training-*")))


if __name__ == "__main__":
    unittest.main()
