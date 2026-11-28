import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QEvent, QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from Toolbox.main import ToolboxWindow
from Toolbox.tool_registry import TOOLS
from PointCloud.forest_metrics_gui import ForestMetricsWindow
from PointCloud.forestry_training_gui import TrainingWindow
from test_forestry_gui import wait_until

app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)
ORDER = ["tree-segmentation", "forest-metrics", "forest-training"]


class ForestryNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = {"output_directory": str(self.root / "中文 分割"), "binary": "engine.exe"}
        self.dashboard = ToolboxWindow(persist_settings=False, check_updates=False)
        self.dashboard.show()
        app.processEvents()

    def tearDown(self):
        self.dashboard.close()
        app.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_group_navigation_search_and_small_layout(self):
        for category in ("全部", "点云处理"):
            self.dashboard.set_category(category)
            self.assertFalse(self.dashboard.forestry_entry.isHidden())
            self.assertTrue(all(self.dashboard.tool_cards[name].isHidden() for name in ORDER))
        self.dashboard.set_category("林业功能")
        self.assertEqual([s.id for s in self.dashboard.filtered_tools()], ORDER)
        self.assertTrue(self.dashboard.category_buttons["林业功能"].isChecked())
        for width in (640, 960, 1280):
            self.dashboard.resize(width, 600)
            app.processEvents()
            self.dashboard.render_tools()
            app.processEvents()
            positions = [self.dashboard.tools_layout.getItemPosition(
                self.dashboard.tools_layout.indexOf(self.dashboard.tool_cards[name]))[:2] for name in ORDER]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(self.dashboard.tool_scroll.horizontalScrollBar().maximum(), 0)
        self.dashboard.set_category("全部")
        self.dashboard.search_edit.setText("林业 TXT")
        self.dashboard.render_tools()
        self.assertEqual([s.id for s in self.dashboard.filtered_tools()], ["forest-training"])
        self.assertFalse(self.dashboard.tool_cards["forest-training"].isHidden())
        self.assertTrue(self.dashboard.forestry_entry.isHidden())
        self.dashboard.search_edit.setText("no matching tool")
        self.dashboard.render_tools()
        self.assertFalse(self.dashboard.empty_label.isHidden())

    def test_async_handoff_and_metrics_to_training_uses_segmentation_directory(self):
        self.dashboard.open_forestry_step("forest-metrics", self.result)
        wait_until(lambda: self.dashboard._loader is None)
        metrics = self.dashboard.open_windows["forest-metrics"]
        self.assertEqual(metrics.source_edit.text(), self.result["output_directory"])
        self.assertEqual(metrics.mode.currentData(), "reuse")
        self.assertEqual(metrics.label_edit.text(), "instance_id")
        metrics.output_directory = str(self.root / "metrics-output")
        metrics.training_button.click()
        wait_until(lambda: self.dashboard._loader is None)
        training = self.dashboard.open_windows["forest-training"]
        self.assertEqual(training.source_edit.text(), self.result["output_directory"])
        self.assertIsNone(training.worker)  # Navigation does not start processing.

    def test_busy_target_cannot_be_replaced_or_queued_for_later(self):
        training = TrainingWindow()
        self.dashboard.open_windows["forest-training"] = training
        training.source_edit.setText("in-progress-source")
        training.worker = object()
        with patch.object(QMessageBox, "information") as info:
            self.dashboard.open_forestry_step("forest-training", self.result)
        self.assertEqual(training.source_edit.text(), "in-progress-source")
        info.assert_called_once()
        training.worker = None
        self.assertEqual(training.source_edit.text(), "in-progress-source")
        self.dashboard.open_forestry_step("forest-training", self.result)
        self.assertEqual(training.source_edit.text(), self.result["output_directory"])

    def test_metrics_context_reset_and_source_change_prevents_stale_continuation(self):
        metrics = ForestMetricsWindow(QSettings(str(self.root / "test.ini"), QSettings.IniFormat))
        self.addCleanup(metrics.close)
        metrics.mode.setCurrentIndex(metrics.mode.findData("stems"))
        metrics.ground_mode.setCurrentIndex(metrics.ground_mode.findData("normalized"))
        metrics.label_edit.setText("other_label")
        self.assertTrue(metrics.use_segmentation_result(self.result))
        self.assertEqual(metrics.mode.currentData(), "reuse")
        self.assertEqual(metrics.ground_mode.currentData(), "auto")
        self.assertTrue(metrics.metres.isChecked())
        emitted = []
        metrics.continue_requested.connect(lambda *args: emitted.append(args))
        metrics.source_edit.setText("another-source")
        metrics._continue_training()
        self.assertFalse(emitted)
        self.assertFalse(metrics.training_button.isEnabled())
        metrics.worker = object()
        self.assertFalse(metrics.use_segmentation_result(self.result))
        self.assertEqual(metrics.source_edit.text(), "another-source")
        metrics.worker = None

    def test_training_inputs_locked_during_work_and_restored_after_failure(self):
        training = TrainingWindow()
        self.addCleanup(training.close)
        training.source_edit.setText(str(self.root))
        training.output_edit.setText(str(self.root))
        with patch("PointCloud.forestry_training.export_training", side_effect=ValueError("invalid source")), \
                patch.object(QMessageBox, "warning"):
            training.run_button.click()
            self.assertFalse(training.inputs.isEnabled())
            self.assertFalse(training.use_segmentation_result(self.result))
            wait_until(lambda: training.worker is None)
        self.assertTrue(training.inputs.isEnabled())
        self.assertFalse(training.folder_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
