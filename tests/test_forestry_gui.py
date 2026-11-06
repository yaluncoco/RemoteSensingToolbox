import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QSettings
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox

from PointCloud.forestry_gui import ForestryWindow
from Toolbox.tool_registry import TOOLS


app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    if not predicate():
        raise AssertionError("Worker did not finish")


class ForestryGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        self.window = ForestryWindow(self.settings)
        self.addCleanup(self.window.close)

    def test_path_saved_and_restored(self):
        self.window.binary_edit.setText(r"D:\中文 路径\CloudCompare.exe")
        self.window._save_binary()
        other = ForestryWindow(self.settings)
        try:
            self.assertEqual(other.binary_edit.text(), self.window.binary_edit.text())
        finally:
            other.close()
        self.assertTrue(any(s.id == "tree-segmentation" and s.category == "点云处理" for s in TOOLS))

    def test_probe_is_background_task_and_cannot_report_missing_plugin_as_success(self):
        def delayed_probe(*args):
            time.sleep(.15)
            return {"available": False, "missing": ["TREEISO"]}
        with patch("PointCloud.cloudcompare.probe", side_effect=delayed_probe):
            self.window._probe()
            self.assertFalse(self.window.run_button.isEnabled())
            self.assertTrue(self.window.cancel_button.isEnabled())
            wait_until(lambda: self.window.worker is None)
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIn("TREEISO", self.window.engine_status.text())
        self.assertFalse(self.window.preview_button.isEnabled())

    def test_units_required_before_worker_start(self):
        source = self.root / "input.las"
        source.touch()
        self.window.source_edit.setText(str(source))
        self.window.output_edit.setText(str(self.root))
        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()
        warning.assert_called_once()
        self.assertIsNone(self.window.worker)

    def test_finished_result_enables_folder_and_preview(self):
        result = {"output_directory": str(self.root), "binary": "engine.exe",
                  "segmentation": {"candidate_tree_count": 12, "flagged_candidate_count": 3},
                  "export": {"preview_cloud": str(self.root / "instance_preview.laz")}}
        with patch("PointCloud.forestry_workflow.run_workflow", return_value=result):
            self.window._launch({})
            wait_until(lambda: self.window.worker is None)
        self.assertTrue(self.window.folder_button.isEnabled())
        self.assertTrue(self.window.preview_button.isEnabled())
        self.assertIn("12", self.window.status_label.text())
        self.assertIn("复核", self.window.status_label.text())


if __name__ == "__main__":
    unittest.main()
