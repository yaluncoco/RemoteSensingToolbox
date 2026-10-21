import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PyQt5 import sip
from PyQt5.QtCore import Qt, QEvent, QPoint, QPointF
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QSpinBox

from Toolbox.tool_registry import configure_runtime_environment
configure_runtime_environment()
from PointCloud.visualization_data import (
    VisualizationOptions, read_visualization_text, extract_visualization_data, apply_label_colors,
)
from PointCloud.visualization import parse_color_map_string_for_vis, _run_custom_color_visualization
from PointCloud.gui import PointCloudProcessor

app = QApplication.instance() or QApplication([])
app.setQuitOnLastWindowClosed(False)


class PointVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / "中文 点云.txt"

    def read(self, text, options=None, preview=False):
        self.path.write_text(text, encoding="utf-8-sig")
        return read_visualization_text(self.path, options or VisualizationOptions(), preview)

    def test_presets_and_legacy_extra_columns(self):
        cases = [
            ("xyz", "uniform", "1 2 3", None),
            ("xyz_label", "label", "1 2 3 -1", -1),
            ("xyz_rgb", "rgb", "1 2 3 255 1 0", None),
            ("xyz_rgb_label", "label", "1 2 3 255 1 0 9", 9),
            ("xyz_rgb_label_normal", "label", "1 2 3 255 1 0 9 0 0 1", 9),
            ("xyz_rgb_label", "label", "1 2 3 255 1 0 9 0 0 1", 9),
        ]
        for layout, mode, row, label in cases:
            with self.subTest(layout=layout, row=row):
                options = VisualizationOptions(layout=layout, color_mode=mode)
                points, labels, colors = extract_visualization_data(self.read(row), options)
                np.testing.assert_array_equal(points, [[1, 2, 3]])
                if label is not None:
                    np.testing.assert_array_equal(labels, [label])
                elif mode == "rgb":
                    np.testing.assert_allclose(colors, [[1, 1 / 255, 0]])
                else:
                    np.testing.assert_allclose(colors, [[.5, .5, .5]])

    def test_custom_columns_and_label_ids_are_preserved(self):
        data = self.read("42 10 20 30 0\n-1 11 21 31 0\n1 12 22 32 0")
        options = VisualizationOptions(layout="custom", xyz_columns=(2, 3, 4), label_column=1)
        points, labels, _ = extract_visualization_data(data, options)
        np.testing.assert_array_equal(points[:, 0], [10, 11, 12])
        np.testing.assert_array_equal(labels, [42, -1, 1])
        colors = apply_label_colors(labels, {42: [1, 0, 0], 1: [0, 1, 0]}, [.5, .5, .5])
        np.testing.assert_array_equal(colors, [[1, 0, 0], [.5, .5, .5], [0, 1, 0]])
        options = replace(options, color_mode="rgb", rgb_columns=(5, 1, 6))
        data = self.read("255 10 20 30 0 128")
        _, _, colors = extract_visualization_data(data, options)
        np.testing.assert_allclose(colors, [[0, 1, 128 / 255]])

    def test_delimiters_headers_comments_and_utf8_bom(self):
        for delimiter in [" ", "\t", ",", ";"]:
            for selected in ["auto", {" ": "whitespace", "\t": "tab", ",": "comma", ";": "semicolon"}[delimiter]]:
                with self.subTest(delimiter=delimiter, selected=selected):
                    options = VisualizationOptions(layout="xyz_label", delimiter=selected, skip_rows=1)
                    text = "X,Y,Z,Label\n# 注释\n\n" + delimiter.join(["1", "2", "3", "5"]) + " # comment\n"
                    for preview in (True, False):
                        data = self.read(text, options, preview)
                        np.testing.assert_array_equal(data, [[1, 2, 3, 5]])

    def test_invalid_input_is_rejected_without_guessing_or_rounding(self):
        for text in ("", "# comment", "1\n2\n3", "1 2 3", "1 2 3 .5", "nan 2 3 1", "1 2 3 inf",
                     "1 2 3 999999999999999999", "1 2 3 1\n1 2 3", "X Y Z Label\n1 2 3 0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_visualization_data(self.read(text), VisualizationOptions(layout="xyz_label"))
        data = self.read("1 2 3 4")
        with self.assertRaisesRegex(ValueError, "至少 7 列"):
            extract_visualization_data(data, VisualizationOptions())
        for options in [VisualizationOptions(layout="custom", label_column=3),
                        VisualizationOptions(layout="custom", label_column=0),
                        VisualizationOptions(layout="custom", label_column=8),
                        VisualizationOptions(layout="xyz", color_mode="label")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                extract_visualization_data(data, options)

    def test_preview_is_bounded_and_full_read_checks_later_rows(self):
        options = VisualizationOptions(layout="xyz_label")
        text = "1 2 3 0\n" * 10000 + "1 2 3 .5\n"
        preview = self.read(text, options, preview=True)
        self.assertEqual(preview.shape, (5, 4))
        extract_visualization_data(preview, options)
        with self.assertRaisesRegex(ValueError, "有限整数"):
            extract_visualization_data(read_visualization_text(self.path, options), options)
        with self.assertRaisesRegex(ValueError, "64 KB"):
            self.read("#" + " " * 70000 + "\n1 2 3 0", options, preview=True)

    def test_rgb_ranges_use_whole_array_not_each_component(self):
        options = VisualizationOptions(layout="xyz_rgb", color_mode="rgb")
        data = self.read("1 2 3 255 1 0\n4 5 6 1 1 1")
        _, _, colors = extract_visualization_data(data, options)
        np.testing.assert_allclose(colors, [[1, 1 / 255, 0], [1 / 255] * 3])
        with self.assertRaisesRegex(ValueError, "超出"):
            extract_visualization_data(data, replace(options, rgb_range="0-1"))
        for rgb in ("-1 0 0", "256 0 0", "nan 0 0"):
            with self.subTest(rgb=rgb), self.assertRaises(ValueError):
                extract_visualization_data(self.read("1 2 3 " + rgb), options)
        data = self.read("1 2 3 1 0 0")
        self.assertEqual(extract_visualization_data(data, replace(options, rgb_range="0-255"))[2][0, 0], 1 / 255)

    def test_color_map_rejects_mistakes_and_normalizes_consistently(self):
        colors, default = parse_color_map_string_for_vis("# comment\n1: [255, 1, 0]\n2: [0.5, 1, 0]")
        np.testing.assert_allclose(colors[1], [1, 1 / 255, 0])
        self.assertEqual(colors[2], [.5, 1, 0])
        self.assertEqual(default, [.5, .5, .5])
        for text in ("1: [256, 0, 0]", "1: [0, 0]", "bad", "1: [True, 0, 0]", "1: [0, 0, 0]\n1: [1, 1, 1]"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "颜色映射第"):
                parse_color_map_string_for_vis(text)

    def test_runner_validates_full_input_before_creating_window(self):
        self.read("1 2 3 0\n" * 8 + "1 2 3 .5")
        statuses, done = [], Mock()
        with patch("PointCloud.visualization.o3d.visualization.Visualizer") as constructor:
            _run_custom_color_visualization(self.path, "", status_update_func=statuses.append,
                                            options=VisualizationOptions(layout="xyz_label"), finished_func=done)
        constructor.assert_not_called()
        self.assertIn("有限整数", statuses[-1])
        done.assert_called_once()

    def test_renderer_failure_and_cleanup_are_reported(self):
        self.read("1 2 3")
        for failure in ("create", "run"):
            with self.subTest(failure=failure):
                renderer = Mock()
                renderer.create_window.return_value = failure != "create"
                if failure == "run":
                    renderer.run.side_effect = RuntimeError("renderer failed")
                statuses, done = [], Mock()
                with patch("PointCloud.visualization.o3d.visualization.Visualizer", return_value=renderer):
                    _run_custom_color_visualization(self.path, "ignored for uniform", status_update_func=statuses.append,
                                                    options=VisualizationOptions(layout="xyz", color_mode="uniform"),
                                                    finished_func=done)
                self.assertIn("出错", statuses[-1])
                self.assertEqual(renderer.destroy_window.call_count, int(failure == "run"))
                done.assert_called_once()

    def test_capture_none_return_and_unicode_png_jpeg(self):
        self.read("1 2 3 1")
        renderer = Mock()
        renderer.capture_screen_image.side_effect = lambda path, **kwargs: Image.new("RGB", (8, 8), "red").save(path)
        statuses = []
        for suffix, expected in (("png", "PNG"), ("jpg", "JPEG")):
            output = self.path.parent / ("中文 截图." + suffix)
            with patch("PointCloud.visualization.o3d.visualization.Visualizer", return_value=renderer):
                _run_custom_color_visualization(self.path, "1: [255, 0, 0]", output, statuses.append,
                                                VisualizationOptions(layout="xyz_label"))
            with Image.open(output) as image:
                self.assertEqual(image.format, expected)
            self.assertIn("截图已保存", statuses[-1])
        output.write_bytes(b"previous output")
        renderer.capture_screen_image.side_effect = None
        with patch("PointCloud.visualization.o3d.visualization.Visualizer", return_value=renderer):
            _run_custom_color_visualization(self.path, "", output, statuses.append, VisualizationOptions(layout="xyz_label"))
        self.assertEqual(output.read_bytes(), b"previous output")
        self.assertIn("保存截图失败", statuses[-1])


class PointVisualizationUiTests(unittest.TestCase):
    def setUp(self):
        self.window = PointCloudProcessor()
        self.window.switch_page(3)
        self.window.show()
        self.addCleanup(lambda: None if sip.isdeleted(self.window) else self.window.close())
        self.panel = self.window.vis_format_widget

    def choose(self, combo, value):
        combo.setCurrentIndex(combo.findData(value))
        app.processEvents()

    def test_controls_follow_format_and_preserve_color_map(self):
        original = self.window.vis_custom_color_map_edit.toPlainText()
        self.choose(self.panel.layout_combo, "xyz")
        self.assertEqual(self.panel.options().color_mode, "uniform")
        self.assertTrue(self.window.vis_color_map_widget.isHidden())
        self.choose(self.panel.layout_combo, "xyz_rgb")
        self.assertEqual(self.panel.options().color_mode, "rgb")
        self.choose(self.panel.layout_combo, "xyz_label")
        self.assertFalse(self.window.vis_color_map_widget.isHidden())
        self.assertEqual(self.window.vis_custom_color_map_edit.toPlainText(), original)
        self.assertTrue(self.panel.details.isHidden())
        self.panel.details_toggle.setChecked(True)
        self.assertFalse(self.panel.details.isHidden())

    def test_wheel_scrolls_page_without_changing_format_even_after_focus(self):
        self.window.resize(900, 600)
        self.choose(self.panel.layout_combo, "custom")
        self.panel.details_toggle.setChecked(True)
        scroll = self.window.stacked_widget.currentWidget()
        bar = scroll.verticalScrollBar()
        for mode in ("label", "rgb"):
            self.choose(self.panel.mode_combo, mode)
            controls = [self.panel.layout_combo, self.panel.mode_combo,
                        self.panel.delimiter_combo, self.panel.skip_spin, *self.panel.xyz_spins]
            controls += ([self.panel.label_spin] if mode == "label" else
                         [self.panel.range_combo, *self.panel.rgb_spins])
            for control in controls:
                for focused in (False, True):
                    for delta in (120, -120):
                        with self.subTest(mode=mode, control=control, focused=focused, delta=delta):
                            (control if focused else self.panel.details_toggle).setFocus()
                            scroll.ensureWidgetVisible(control)
                            app.processEvents()
                            self.assertEqual(control.hasFocus(), focused)
                            before = self.panel.options()
                            position = bar.value()
                            target = control.lineEdit() if isinstance(control, QSpinBox) else control
                            point = target.rect().center()
                            global_point = target.mapToGlobal(point)
                            event = QWheelEvent(QPointF(self.window.mapFromGlobal(global_point)), QPointF(global_point),
                                                QPoint(), QPoint(0, delta), Qt.NoButton,
                                                Qt.NoModifier, Qt.NoScrollPhase, False)
                            app.sendEvent(self.window.windowHandle(), event)
                            app.processEvents()
                            self.assertEqual(self.panel.options(), before)
                            if delta > 0 and position > bar.minimum():
                                self.assertLess(bar.value(), position)
                            elif delta < 0 and position < bar.maximum():
                                self.assertGreater(bar.value(), position)

    def test_format_can_still_be_selected_by_click_and_keyboard(self):
        combo = self.panel.layout_combo
        self.window.stacked_widget.currentWidget().ensureWidgetVisible(combo)
        app.processEvents()
        QTest.mouseClick(combo, Qt.LeftButton)
        QTest.qWait(QApplication.doubleClickInterval() + 20)
        view = combo.view()
        index = combo.model().index(combo.findData("xyz_label"), 0)
        view.scrollTo(index)
        app.processEvents()
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(index).center())
        self.assertEqual(self.panel.options().layout, "xyz_label")
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key_Down)
        self.assertEqual(self.panel.options().layout, "xyz_rgb")
        self.choose(combo, "custom")
        self.panel.label_spin.setFocus()
        self.panel.label_spin.selectAll()
        QTest.keyClicks(self.panel.label_spin, "5")
        QTest.keyClick(self.panel.label_spin, Qt.Key_Enter)
        self.assertEqual(self.panel.options().label_column, 5)

    def test_merge_resets_read_options_and_label_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "merged.txt"
            path.write_text("1 2 3 255 0 0 9 0 0 1", encoding="utf-8")
            self.choose(self.panel.layout_combo, "custom")
            self.choose(self.panel.delimiter_combo, "comma")
            self.panel.skip_spin.setValue(10)
            self.window.on_merge_prediction_result(str(path))
            options = self.panel.options()
            self.assertEqual((options.layout, options.color_mode, options.delimiter, options.skip_rows),
                             ("xyz_rgb_label", "label", "auto", 0))
            self.assertIn("样本检查通过", self.panel.preview_label.text())

    def test_background_failure_reaches_ui_and_allows_retry(self):
        self.window.vis_custom_input_txt_path = "missing-test-input.txt"
        self.choose(self.panel.layout_combo, "xyz_label")
        self.window.start_custom_color_visualization()
        self.assertFalse(self.window.vis_custom_start_button.isEnabled())
        self.window.visualization_thread.join(timeout=3)
        QTest.qWait(100)
        self.assertTrue(self.window.vis_custom_start_button.isEnabled())
        self.assertIn("无法读取文件", self.window.vis_custom_status_label.text())

    def test_closing_tool_before_renderer_finishes_is_safe(self):
        release, errors = threading.Event(), []

        def finish_later(path, colors, screenshot, status, options, finished_func):
            release.wait(3)
            try:
                status("renderer finished after closing tool")
                finished_func()
            except Exception as exc:
                errors.append(exc)

        self.window.setAttribute(Qt.WA_DeleteOnClose)
        self.window.vis_custom_input_txt_path = "sample.txt"
        with patch("PointCloud.gui._run_custom_color_visualization", side_effect=finish_later):
            self.window.start_custom_color_visualization()
            thread = self.window.visualization_thread
            self.window.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            self.assertTrue(sip.isdeleted(self.window))
            release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            app.processEvents()
            self.assertEqual(errors, [])

    def test_minimum_window_does_not_overflow_horizontally(self):
        self.window.resize(840, 580)
        scroll = self.window.stacked_widget.currentWidget()
        for layout in ("xyz", "xyz_label", "xyz_rgb_label_normal", "custom"):
            self.choose(self.panel.layout_combo, layout)
            self.panel.details_toggle.setChecked(True)
            QTest.qWait(30)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0, layout)
            for widget in (self.panel.layout_combo, self.panel.mode_combo, self.panel.custom_widget, self.panel.details):
                if not widget.isHidden():
                    self.assertLessEqual(widget.geometry().right(), self.panel.width())
