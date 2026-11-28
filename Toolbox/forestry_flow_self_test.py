"""Exercise the real dashboard, handoff signals and three background jobs."""
import csv
import json
from pathlib import Path
import tempfile
import time

from PyQt5.QtCore import QEvent, QSettings, QThread
from PyQt5.QtWidgets import QMessageBox


def check_forestry_flow(app, binary, source=None, output_parent=None, screenshots=None):
    from Toolbox.main import ToolboxWindow
    from Toolbox.tool_registry import TOOLS
    from Toolbox.forestry_self_test import synthetic_forest

    with tempfile.TemporaryDirectory(prefix="toolbox-forest-flow-") as directory:
        root = Path(directory)
        settings = QSettings(str(root / "settings.ini"), QSettings.IniFormat)
        dashboard = ToolboxWindow(persist_settings=False, check_updates=False)
        messages = []
        originals = {name: getattr(QMessageBox, name) for name in ("warning", "critical", "information")}
        for name in originals:
            setattr(QMessageBox, name, lambda parent, title, text, *args: messages.append(f"{title}: {text}") or QMessageBox.Ok)

        def wait(predicate):
            deadline = time.monotonic() + 3600
            while not predicate():
                app.processEvents()
                QThread.msleep(10)
                if time.monotonic() > deadline:
                    raise TimeoutError("Forestry flow exceeded one hour")
            if messages:
                raise AssertionError(messages)

        def opened(tool_id):
            wait(lambda: dashboard._loader is None)
            window = dashboard.open_windows[tool_id]
            if hasattr(window, "settings"):
                window.settings = settings
            return window

        def capture(window, name):
            app.processEvents()
            if screenshots:
                path = Path(screenshots)
                path.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(path / (name + ".png"))):
                    raise AssertionError("Screenshot failed")

        def require(condition, message):
            # Frozen builds use optimize=1; ordinary assert would be removed.
            if not condition:
                raise AssertionError(message)

        try:
            dashboard.set_category("林业功能")
            dashboard.show()
            expected = ["tree-segmentation", "forest-metrics", "forest-training"]
            require([s.id for s in dashboard.filtered_tools()] == expected, "Forestry step order changed")
            capture(dashboard, "forestry-group")
            dashboard.open_tool(next(s for s in TOOLS if s.id == "tree-segmentation"))
            segmentation = opened("tree-segmentation")
            segmentation.source_edit.setText(str(source or synthetic_forest(root)))
            segmentation.output_edit.setText(str(output_parent or root))
            segmentation.binary_edit.setText(str(binary))
            segmentation.metres.setChecked(True)
            # The separate third step must also work when inline export is off.
            segmentation.training_format.setCurrentIndex(segmentation.training_format.findData("none"))
            segmentation.run_button.click()
            wait(lambda: segmentation.worker is None)
            require(segmentation.result, segmentation.status_label.text())
            result = segmentation.result
            capture(segmentation, "forestry-segmentation-complete")
            segmentation.metrics_button.click()
            metrics = opened("forest-metrics")
            require(metrics.source_edit.text() == result["output_directory"], "Metrics received wrong source")
            require(metrics.mode.currentData() == "reuse", "Metrics would regroup the trees")
            require(metrics.label_edit.text() == "instance_id", "Metrics lost the original label field")
            metrics.run_button.click()
            wait(lambda: metrics.worker is None and metrics.scene_worker is None)
            require(metrics.result, metrics.status_label.text())
            require(metrics.result["summary"]["tree_count"] == result["segmentation"]["candidate_tree_count"], "Tree count changed during handoff")
            require(metrics.result["summary"]["source_fields_verified"], "Metrics source fields not verified")
            capture(metrics, "forestry-metrics-complete")
            metrics.training_button.click()
            training = opened("forest-training")
            require(training.source_edit.text() == result["output_directory"], "Training received metrics output instead of segmentation")
            training.format.setCurrentIndex(training.format.findData("both"))
            training.run_button.click()
            wait(lambda: training.worker is None)
            require(training.result, training.status.text())
            require(training.result["point_count"] == result["export"]["source_point_count"], "Training lost source points")
            require(all(training.result["verification"].values()), "Training labels failed alignment validation")
            with (Path(metrics.result["output_directory"]) / "林业参数.csv").open(encoding="utf-8-sig", newline="") as stream:
                metric_ids = {int(row["TreeID"]) for row in csv.DictReader(stream)}
            training_ids = {row["instance_id"] for row in training.result["instance_map"]}
            require(metric_ids == training_ids, "Metrics and training disagree on original tree IDs")
            capture(training, "forestry-training-complete")
            first = training.result
            training.source_edit.setText(first["directory"])
            training.format.setCurrentIndex(training.format.findData("numpy"))
            training.run_button.click()
            wait(lambda: training.worker is None)
            require(training.result, training.status.text())
            require(training.result["logical_content"] == first["logical_content"], "Conversion changed points or labels")
            detail = {"source": str(source) if source else "synthetic forest", "output_directory": result["output_directory"],
                      "points": result["export"]["source_point_count"],
                      "candidates": result["segmentation"]["candidate_tree_count"],
                      "metrics": metrics.result, "training": first, "conversion": training.result,
                      "handoff": "segmentation -> metrics -> original segmentation -> training",
                      "tree_ids_preserved": True,
                      "retained": output_parent is not None}
            if output_parent:
                (Path(result["output_directory"]) / "flow-validation.json").write_text(
                    json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
            return detail
        finally:
            for window in dashboard.open_windows.values():
                worker = getattr(window, "worker", None)
                if worker is not None:
                    worker.stop()
                    while getattr(window, "worker", None) is not None:
                        app.processEvents()
                        QThread.msleep(20)
            dashboard.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            for name, original in originals.items():
                setattr(QMessageBox, name, original)
