import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import laspy
import numpy as np
from PyQt5.QtCore import QSettings, Qt, QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from PointCloud.forest_metrics_workflow import run_metrics
from PointCloud.forest_scene import load_scene, export_instances
from PointCloud.forest_metrics_gui import ForestMetricsWindow
from PointCloud.forest_viewer import ForestCanvas
from PointCloud.forestry_io import fingerprint
from Toolbox.metrics_self_test import synthetic_inventory

app = QApplication.instance() or QApplication([])


def wait_for(window):
    end = time.monotonic() + 30
    while window.scene_worker is not None:
        app.processEvents()
        QTest.qWait(5)
        if time.monotonic() > end:
            raise TimeoutError("Scene worker did not finish")
    app.processEvents()


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = synthetic_inventory(cls.root)
        cls.original_hash = fingerprint(cls.source)
        cls.result = run_metrics(cls.source, cls.root, coordinates_in_metres=True)
        cls.output = Path(cls.result["output_directory"])
        cls.scene = load_scene(cls.output)

    @classmethod
    def tearDownClass(cls):
        app.processEvents()
        cls.temp.cleanup()

    def test_portable_preview_keeps_nonconsecutive_ids_and_all_fields(self):
        self.assertEqual(set(np.unique(self.scene["ids"])), {0, 7, 100003})
        self.assertLessEqual(len(self.scene["ids"]), 200_000)
        self.assertEqual(len(self.scene["rows"][0]), 27)
        self.assertEqual(fingerprint(self.source), self.original_hash)

    def test_full_instance_exports_preserve_original_records_and_metadata(self):
        result = export_instances(self.scene, [100003, 7], self.root)
        original = laspy.read(self.output / "参数标注.laz")
        for path in result["files"]:
            target = laspy.read(path)
            tree_id = int(Path(path).stem.split("_")[1])
            expected = original[original.metric_tree_id == tree_id]
            self.assertEqual(target.points.array.tobytes(), expected.points.array.tobytes())
            np.testing.assert_array_equal(target.header.scales, original.header.scales)
            np.testing.assert_array_equal(target.header.offsets, original.header.offsets)
            vlrs = lambda h: [(v.user_id, v.record_id, v.record_data_bytes()) for v in h.vlrs
                             if v.user_id != "laszip encoded" and not (v.user_id == "LASF_Spec" and v.record_id == 4)]
            self.assertEqual(vlrs(target.header), vlrs(original.header))
            self.assertEqual(target.header.evlrs[0].record_data, original.header.evlrs[0].record_data)
        self.assertTrue(result["original_records_verified"])
        self.assertEqual(fingerprint(self.source), self.original_hash)

    def test_export_without_full_annotated_cloud_retains_original_dimensions(self):
        result = run_metrics(self.source, self.root, coordinates_in_metres=True, export_cloud=False)
        scene = load_scene(result["output_directory"])
        exported = export_instances(scene, [7], self.root)
        original = laspy.read(self.source)
        target = laspy.read(exported["files"][0])
        expected = original[original.TreeID == 7]
        for name in original.points.array.dtype.names:
            self.assertEqual(target.points.array[name].tobytes(), expected.points.array[name].tobytes())
        self.assertTrue(np.all(target.metric_tree_id == 7))

    def test_old_results_open_without_recomputing_measurements(self):
        path = self.output / "scene.json"
        data = path.read_bytes()
        path.unlink()
        try:
            scene = load_scene(self.output)
            self.assertEqual(scene["manifest"]["schema"], 0)
            self.assertEqual(set(np.unique(scene["ids"])), {0, 7, 100003})
            self.assertEqual(scene["rows"], self.scene["rows"])
        finally:
            path.write_bytes(data)

    def test_changed_label_index_is_rejected(self):
        path = self.output / "instance_labels.npy"
        content = path.read_bytes()
        path.write_bytes(content + b"changed")
        try:
            with self.assertRaisesRegex(ValueError, "编号索引校验失败"):
                export_instances(self.scene, [7], self.root)
        finally:
            path.write_bytes(content)

    def test_cancelled_export_creates_no_completed_result(self):
        from PointCloud.cloudcompare import SegmentationCancelled
        with self.assertRaises(SegmentationCancelled):
            export_instances(self.scene, [7], self.root, cancelled=lambda: True)

    def test_canvas_depth_picking_and_orientation(self):
        canvas = ForestCanvas()
        canvas.resize(600, 450)
        scene = dict(self.scene, xyz=np.array([[0., 1., 0.], [0., 2., 0.], [0., 0., 1.]]),
                     ids=np.array([7, 100003, 7]), height=np.array([0., 0., 1.]), rgb=np.zeros((3, 3), dtype="u1"))
        canvas.set_scene(scene)
        canvas.show_labels = False
        canvas.reset_view("front")
        projected, _, _ = canvas._project(scene["xyz"])
        self.assertLess(projected[2, 1], projected[0, 1])
        self.assertEqual(canvas.pick_at(*projected[0]), 100003)
        canvas.close()

    def test_click_table_inspector_isolation_and_export_work_together(self):
        window = ForestMetricsWindow(QSettings(str(self.root / "gui.ini"), QSettings.IniFormat))
        window.show()
        try:
            self.assertTrue(window.open_result_directory(self.output))
            self.assertFalse(window.use_segmentation_result({"output_directory": "other"}))
            wait_for(window)
            work = window.workspace
            self.assertEqual(work.table.rowCount(), 2)
            canvas = work.canvas
            canvas.show_labels = False
            canvas._render()
            visible = np.flatnonzero(canvas._pixel_ids == 100003)
            self.assertGreater(len(visible), 0)
            x, y = canvas._pixels[visible[len(visible)//2]]
            QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(int(x), int(y)))
            self.assertEqual(work.selected_id, 100003)
            self.assertAlmostEqual(float(work.detail_labels["DBH_cm"].text().split()[0]), 50., delta=.1)
            self.assertEqual(work.table.item(work.table.currentRow(), 0).text(), "100003")
            work.isolate_button.click()
            wait_for(window)
            self.assertTrue(canvas.isolated)
            self.assertEqual(len(canvas.scene["ids"]), int(work.rows[100003]["NPoints"]))
            work.overview_button.click()
            self.assertFalse(canvas.isolated)
            work.table.sortItems(1, Qt.DescendingOrder)
            work.select_tree(7)
            self.assertAlmostEqual(float(work.detail_labels["DBH_cm"].text().split()[0]), 30., delta=.1)
            work.export_to(self.root, [7])
            wait_for(window)
            self.assertTrue(work.last_export["original_records_verified"])
        finally:
            if window.scene_worker is not None:
                window.scene_worker.stop()
                window.scene_worker.wait(30000)
                app.processEvents()
            window.close()


if __name__ == "__main__":
    unittest.main()
