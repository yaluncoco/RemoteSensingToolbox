import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Toolbox.tool_registry import configure_runtime_environment
configure_runtime_environment()

import numpy as np
from PointCloud.processing import downsample_points, SamplingCancelled
from PointCloud.workers import CheckShapeWorker, DownsampleWorker, _save_txt_pointcloud, _load_txt_pointcloud
from PointCloud.file_io import read_point_cloud, write_point_cloud


class PointProcessingTests(unittest.TestCase):
    def setUp(self):
        self.points = np.column_stack([np.arange(100), np.zeros(100), np.zeros(100)])
        self.features = np.column_stack([np.full(100, 255), np.zeros((100, 2)), np.arange(100), np.arange(100) * 2])
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_uniform_sampling_covers_entire_cloud(self):
        points, features = downsample_points(self.points, self.features, "uniform", {"target_points": 60})
        self.assertEqual(len(points), 60)
        self.assertEqual(points[0, 0], 0)
        self.assertEqual(points[-1, 0], 99)
        np.testing.assert_array_equal(points[:, 0], features[:, 3])

    def test_random_sampling_is_reproducible_and_keeps_features(self):
        args = (self.points, self.features, "random", {"target_points": 20, "seed": 42})
        points, features = downsample_points(*args)
        again, _ = downsample_points(*args)
        np.testing.assert_array_equal(points, again)
        np.testing.assert_array_equal(points[:, 0], features[:, 3])
        self.assertEqual(len(np.unique(features[:, 3])), 20)

    def test_farthest_sampling_does_not_duplicate_coincident_points(self):
        points = np.ones((10, 3))
        _, features = downsample_points(points, np.arange(10)[:, None], "farthest", {"target_points": 7, "seed": 3})
        self.assertEqual(len(np.unique(features)), 7)

    def test_farthest_sampling_handles_large_coordinate_offsets(self):
        points = np.column_stack([1e12 + np.arange(5), np.ones(5) * 1e12, np.ones(5) * 1e12])
        _, ids = downsample_points(points, np.arange(5)[:, None], "farthest", {"target_points": 3, "seed": 0})
        self.assertEqual(set(ids.ravel()), {0, 2, 4})

    def test_voxel_sampling_preserves_original_rows_and_rgb_scale(self):
        points = np.array([[0, 0, 0], [.1, 0, 0], [5, 0, 0], [5.1, 0, 0]])
        features = np.array([[255, 0, 0, 7, 100], [255, 0, 0, 8, 101], [0, 255, 0, 9, 102], [0, 255, 0, 10, 103]])
        selected, retained = downsample_points(points, features, "voxel", {"voxel_size": 1})
        self.assertEqual(selected.shape, (2, 3))
        self.assertEqual(retained.shape, (2, 5))
        for point, row in zip(selected, retained):
            index = np.flatnonzero(np.all(points == point, axis=1))[0]
            np.testing.assert_array_equal(row, features[index])

    def test_rejects_invalid_parameters_and_supports_cancellation(self):
        for target in [0, -1, 1.5]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                downsample_points(self.points, self.features, "random", {"target_points": target})
        with self.assertRaises(SamplingCancelled):
            downsample_points(self.points, self.features, "uniform", {"target_points": 5}, should_cancel=lambda: True)
        with self.assertRaises(ValueError):
            downsample_points(np.full((2, 3), np.nan), None, "uniform", {"target_points": 1})

    def test_failed_save_preserves_existing_output(self):
        output = self.root / "points.txt"
        output.write_text("completed result", encoding="utf-8")
        with patch("numpy.savetxt", side_effect=OSError("disk full")), self.assertRaises(IOError):
            _save_txt_pointcloud(output, self.points, self.features)
        self.assertEqual(output.read_text(encoding="utf-8"), "completed result")
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_single_row_point_cloud_is_supported(self):
        path = self.root / "single.txt"
        path.write_text("1 2 3 255 0 0 17", encoding="utf-8")
        points, features = _load_txt_pointcloud(path)
        self.assertEqual(points.shape, (1, 3))
        self.assertEqual(features[0, -1], 17)

    def test_single_column_input_is_not_mistaken_for_one_xyz_point(self):
        path = self.root / "bad.txt"
        path.write_text("1\n2\n3\n", encoding="utf-8")
        with self.assertRaises(IOError):
            _load_txt_pointcloud(path)

    def test_batch_excludes_its_output_subdirectory(self):
        input_dir = self.root / "input"
        output_dir = input_dir / "output"
        output_dir.mkdir(parents=True)
        np.savetxt(input_dir / "cloud.txt", self.points)
        np.savetxt(output_dir / "previous.txt", self.points)
        worker = DownsampleWorker(str(input_dir), str(output_dir), "uniform", {"target_points": 4}, True)
        messages = []
        worker.status_signal.connect(messages.append)
        worker.run()
        self.assertTrue((output_dir / "cloud.txt").exists())
        self.assertFalse((output_dir / "output").exists())
        self.assertIn("成功: 1", messages[-1])

    def test_npy_preview_is_bounded_for_multidimensional_arrays(self):
        path = self.root / "volume.npy"
        np.save(path, np.zeros((2, 4, 10000), dtype=np.float32))
        worker = CheckShapeWorker(str(path))
        results = []
        worker.result_signal.connect(results.append)
        worker.run()
        self.assertIn("(2, 4, 10000)", results[0])
        self.assertLess(len(results[0]), 1000)
        path.unlink()  # The memory mapping must not hold the Windows file open.

    def test_open3d_handles_chinese_paths_and_preserves_failed_outputs(self):
        import open3d as o3d
        folder = self.root / "中文 点云"
        folder.mkdir()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.points)
        for suffix in ("pcd", "ply", "xyz"):
            with self.subTest(format=suffix):
                output = folder / ("点云." + suffix)
                self.assertTrue(write_point_cloud(output, cloud))
                np.testing.assert_allclose(np.asarray(read_point_cloud(output).points), self.points)
                previous = output.read_bytes()
                with patch("open3d.io.write_point_cloud", return_value=False), self.assertRaises(OSError):
                    write_point_cloud(output, cloud)
                self.assertEqual(output.read_bytes(), previous)
