import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import laspy
import numpy as np
from PyQt5.QtWidgets import QApplication

from PointCloud.forest_metrics import MetricOptions, fit_circle, area, measure_tree
from PointCloud.forest_metrics_workflow import run_metrics, _ground_model
from PointCloud.cloudcompare import SegmentationCancelled
from Toolbox.metrics_self_test import synthetic_inventory, check_metrics_gui


class MetricsNumericalTests(unittest.TestCase):
    def test_noisy_circle_with_outliers_and_large_offset(self):
        rng = np.random.default_rng(1)
        angle = np.linspace(0, 2*np.pi, 250, endpoint=False)
        p = .18*np.column_stack((np.cos(angle), np.sin(angle)))+rng.normal(0, .002, (250, 2))
        p = np.concatenate((p, rng.uniform(-.5, .5, (60, 2))))+[430000., 4460000.]
        result = fit_circle(p, MetricOptions())
        self.assertEqual(result["DBH_status"], "ok")
        self.assertAlmostEqual(result["DBH_cm"], 36., delta=.3)
        self.assertGreater(result["DBH_fit_inlier"], .75)
        self.assertLess(result["DBH_fit_rmse_m"], .006)

    def test_partial_arc_stays_missing(self):
        angle = np.linspace(0, np.pi/3, 100)
        result = fit_circle(.18*np.column_stack((np.cos(angle), np.sin(angle))), MetricOptions())
        self.assertIsNone(result["DBH_cm"])
        self.assertEqual(result["DBH_status"], "partial_arc")

    def test_duplicates_not_independent_support(self):
        p = np.tile([[0., 0.], [.1, .1], [.2, 0.]], (100, 1))
        self.assertIsNone(fit_circle(p, MetricOptions())["DBH_cm"])

    def test_invalid_parameters(self):
        for values in ({"ground_cell":0}, {"min_radius":.5}, {"min_arc":361}, {"fit_tolerance":np.nan}, {"form_factor":1.1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(MetricOptions(), **values).validate()

    def test_sloped_ground_and_exact_area(self):
        x, y = np.meshgrid(np.arange(8.), np.arange(8.))
        p = np.column_stack((x.ravel(), y.ravel(), (.2*x+.1*y).ravel()))
        _, _, model = _ground_model(p, MetricOptions())
        np.testing.assert_allclose(model([[2.3, 4.6], [5.2, 2.1]]), [.92, 1.25], atol=1e-10)
        self.assertAlmostEqual(area(np.array([[0,0],[2,0],[2,3],[0,3],[1,1]])), 6.)
        self.assertIsNone(area(np.array([[0,0],[1,1],[2,2]])))

    def test_ground_unsupported_keeps_tree_row(self):
        row = measure_tree(9007, np.ones((20, 3)), np.full(20, np.nan), np.full(20, 50.),
                           np.zeros(3), MetricOptions())
        self.assertEqual(row["TreeID"], 9007)
        self.assertIsNone(row["DBH_cm"])
        self.assertIsNone(row["Height_m"])
        self.assertEqual(row["QualityFlags"], "ground_unsupported")


class MetricsWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_gui_and_original_records(self):
        result = check_metrics_gui(self.app)
        self.assertEqual(result["summary"]["accepted_dbh_count"], 2)

    def test_raw_stems_with_existing_ground(self):
        source = synthetic_inventory(self.root, labeled=False)
        result = run_metrics(source, self.root, coordinates_in_metres=True)
        self.assertEqual(result["summary"]["tree_count"], 2)
        self.assertEqual(result["summary"]["accepted_dbh_count"], 2)
        with (Path(result["output_directory"])/"林业参数.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            self.assertAlmostEqual(float(row["CrownBase_m"]), 3., delta=.25)
            self.assertAlmostEqual(float(row["CrownArea_m2"]), np.pi*1.2**2, delta=.02)
            expected = np.pi*1.2**2*(float(row["Height_m"])-3.)
            self.assertAlmostEqual(float(row["CrownVolume_m3"]), expected, delta=.4)

    def test_failure_cannot_publish_success(self):
        source = synthetic_inventory(self.root)
        with patch("PointCloud.forest_metrics_workflow._tables_and_plot", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                run_metrics(source, self.root, coordinates_in_metres=True)
        folder = next(self.root.glob("metrics_*"))
        self.assertFalse((folder/"result.json").exists())
        self.assertEqual(json.loads((folder/"progress.json").read_text())["stage"], "failed")
        self.assertFalse(list(folder.glob("work-*")))

    def test_cancellation_cleans_work_and_does_not_publish(self):
        source = synthetic_inventory(self.root)
        state = {"stop":False}
        def progress(data):
            if data["stage"] == "ground":
                state["stop"] = True
        with self.assertRaises(SegmentationCancelled):
            run_metrics(source, self.root, coordinates_in_metres=True,
                        cancelled=lambda:state["stop"], notify=progress)
        folder = next(self.root.glob("metrics_*"))
        self.assertFalse((folder/"result.json").exists())
        self.assertFalse(list(folder.glob("work-*")))

    def test_reuse_requires_labels(self):
        source = synthetic_inventory(self.root, labeled=False)
        with self.assertRaisesRegex(ValueError, "未找到"):
            run_metrics(source, self.root, coordinates_in_metres=True, mode="reuse")

    def test_invalid_labels_never_silently_cast(self):
        source = synthetic_inventory(self.root)
        cloud = laspy.read(source)
        cloud.TreeID[0] = -1
        cloud.write(source)
        with self.assertRaisesRegex(ValueError, "非负整数"):
            run_metrics(source, self.root, coordinates_in_metres=True)


if __name__ == "__main__":
    unittest.main()
