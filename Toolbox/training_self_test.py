"""Real training conversion, shared by regression and frozen/installed checks."""
import json
from pathlib import Path
import tempfile
import time

import laspy
import numpy as np


def training_fixture(root):
    from PointCloud.forestry_io import fingerprint
    from PointCloud.forestry import EXTRA_FIELDS, save_json
    job = Path(root) / "中文 标注结果"
    job.mkdir()
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales, header.offsets = [.001]*3, [430000.123456789, 4460000.987654321, 90.12]
    for name, dtype in EXTRA_FIELDS.items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    cloud = laspy.LasData(header)
    cloud.X, cloud.Y, cloud.Z = [4, 4, -50, 0, 30, 1500, 70], [3, 3, 4, 5, 6, 7, 8], [0, 0, 1000, 2000, 3000, 4000, 5000]
    cloud.intensity = [0, 65535, 80, 120, 90, 8, 7]
    cloud.red, cloud.green, cloud.blue = [10]*7, [65535]*7, [0]*7
    cloud.semantic_id = [0, 1, 2, 2, 2, 2, 0]
    cloud.instance_id = [0, 0, 2, 2, 4, 4, 0]
    cloud.review_status = [0, 1, 1, 1, 2, 2, 0]
    cloud.source_index = np.arange(7, dtype="u4")
    path = job / "forest_instances.laz"
    cloud.write(path)
    save_json(job / "export.json", {"full_cloud_sha256": fingerprint(path), "source_point_count": 7,
                                    "source_sha256": "fixture-source"})
    save_json(job / "tree_inventory.json", [{"tree_id": i, "raw_treeiso_id": i + 20,
                                            "review_status": r, "review_flags": f}
                                           for i, r, f in ((2, 1, ""), (4, 2, "possible_merge"))])
    return job, cloud


def check_training_gui(app, source=None, output_parent=None):
    from PyQt5.QtCore import QThread, QEvent
    from PyQt5.QtWidgets import QMessageBox
    from PointCloud.forestry_training import export_training
    from PointCloud.forestry_training_gui import TrainingWindow
    with tempfile.TemporaryDirectory(prefix="toolbox-training-") as directory:
        parent = Path(output_parent) if output_parent else Path(directory)
        source = Path(source) if source else training_fixture(Path(directory))[0]
        window = TrainingWindow()
        messages = []
        original_warning = QMessageBox.warning
        QMessageBox.warning = lambda _p, _t, message, *args: messages.append(message) or QMessageBox.Ok
        try:
            window.source_edit.setText(str(source))
            window.output_edit.setText(str(parent))
            window.format.setCurrentIndex(window.format.findData("both"))
            window.show()
            window.run_button.click()
            deadline = time.monotonic() + 7200
            while window.worker is not None:
                app.processEvents()
                QThread.msleep(20)
                if time.monotonic() > deadline:
                    window.worker.stop()
                    raise TimeoutError("Training export exceeded two hours")
            if messages or not window.result:
                raise AssertionError("; ".join(messages) or "Training GUI produced no result")
            if not window.folder_button.isEnabled() or window.cancel_button.isEnabled() or window.progress.value() != 1:
                raise AssertionError("Training GUI completion state is wrong")
            result = window.result
            if not all(result["verification"].values()):
                raise AssertionError("Training export verification failed")
            # Also test NumPy -> TXT on the small fixture without rerunning engines.
            if output_parent is None:
                numpy = export_training(result["directory"], parent / "转换 NumPy", "numpy", chunk_size=2)
                text = export_training(numpy["directory"], parent / "转回 TXT", "txt", chunk_size=3)
                if text["logical_content"] != result["logical_content"]:
                    raise AssertionError("TXT/NumPy round trip changed labels or coordinates")
            return {"training": result, "gui_status": window.status.text(), "result_actions_enabled": True}
        finally:
            if window.worker is not None:
                window.worker.stop()
                while window.worker is not None:
                    app.processEvents()
                    QThread.msleep(20)
            QMessageBox.warning = original_warning
            window.close()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
