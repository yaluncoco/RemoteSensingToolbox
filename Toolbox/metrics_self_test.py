"""Known-dimension fixtures and full GUI output checks, also run when frozen."""
import csv
import json
from pathlib import Path
import tempfile
import time

import laspy
import numpy as np


def synthetic_inventory(root, labeled=True, ground_class=True):
    root = Path(root)
    rng = np.random.default_rng(413)
    gx, gy = np.meshgrid(np.arange(-3, 10, .25), np.arange(-3, 4, .25))
    ground = np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size)))
    clouds, labels, classes = [ground], [np.zeros(len(ground), dtype="i8")], [np.full(len(ground), 2 if ground_class else 0)]
    for tree_id, x, radius in ((7, 0., .15), (100003, 6., .25)):
        angle = np.tile(np.linspace(0, 2*np.pi, 80, endpoint=False), 70)
        z = np.repeat(np.linspace(.35, 3., 70), 80)
        trunk = np.column_stack((x+radius*np.cos(angle), radius*np.sin(angle), z))
        angle = rng.uniform(0, 2*np.pi, 16000)
        crown = np.column_stack((x+1.2*np.cos(angle), 1.2*np.sin(angle), rng.uniform(3., 8., len(angle))))
        p = np.concatenate((trunk, crown))
        clouds.append(p)
        labels.append(np.full(len(p), tree_id, dtype="i8"))
        classes.append(np.zeros(len(p), dtype="u1"))
    xyz = np.concatenate(clouds)
    xyz[:, 2] += .08*xyz[:, 0]+.03*xyz[:, 1]
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = [430000., 4460000., 90.]
    header.scales = [.0001]*3
    header.add_extra_dim(laspy.ExtraBytesParams(name="custom_measurement", type="f4"))
    if labeled:
        header.add_extra_dim(laspy.ExtraBytesParams(name="TreeID", type="i8"))
    from rasterio.crs import CRS
    header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(CRS.from_epsg(32650).to_wkt()))
    header.global_encoding.wkt = True
    header.evlrs = laspy.vlrs.vlrlist.VLRList([laspy.VLR(user_id="toolbox_test", record_id=77, record_data=b"preserve-extended-metadata")])
    cloud = laspy.LasData(header)
    cloud.x, cloud.y, cloud.z = (xyz+header.offsets).T
    cloud.classification = np.concatenate(classes)
    cloud.intensity = np.arange(len(xyz), dtype="u2")
    cloud.custom_measurement = np.arange(len(xyz), dtype="f4")*.125
    if labeled:
        cloud.TreeID = np.concatenate(labels)
    path = root/("已有编号.laz" if labeled else "原始林地.laz")
    cloud.write(path)
    return path


def check_metrics_gui(app, binary="", source=None, output_parent=None, raw=False, screenshot=None):
    from PyQt5.QtCore import QSettings, QThread
    from PyQt5.QtWidgets import QMessageBox
    from PointCloud.forest_metrics_gui import ForestMetricsWindow
    with tempfile.TemporaryDirectory(prefix="toolbox-metrics-") as directory:
        root = Path(directory)
        fixture = source is None
        source = Path(source) if source else synthetic_inventory(root, labeled=not raw, ground_class=not raw)
        parent = Path(output_parent) if output_parent else root
        window = ForestMetricsWindow(QSettings(str(root/"settings.ini"), QSettings.IniFormat))
        messages = []
        warning = QMessageBox.warning
        QMessageBox.warning = lambda _parent, title, message, *args: messages.append(f"{title}: {message}") or QMessageBox.Ok
        try:
            window.source_edit.setText(str(source))
            window.output_edit.setText(str(parent))
            window.binary_edit.setText(str(binary))
            window.metres.setChecked(True)
            window.show()
            app.processEvents()
            window.run_button.click()
            deadline = time.monotonic()+1800
            while window.worker is not None:
                app.processEvents()
                QThread.msleep(10)
                if time.monotonic() > deadline:
                    window.worker.stop()
                    window.worker.wait(30000)
                    raise TimeoutError("林业参数 GUI 流程超时。")
            if messages or not window.result:
                raise AssertionError(messages or window.status_label.text())
            result = window.result
            folder = Path(result["output_directory"])
            if not (folder/"result.json").is_file() or not window.folder_button.isEnabled():
                raise AssertionError("GUI completion did not expose verified outputs")
            summary = result["summary"]
            with (folder/"林业参数.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != summary["tree_count"] or not summary.get("source_fields_verified"):
                raise AssertionError("Inventory or original-field verification missing")
            if fixture:
                if summary["tree_count"] != 2 or summary["accepted_dbh_count"] != 2:
                    raise AssertionError(f"Expected two measured trees: {summary}")
                dbh = sorted(float(r["DBH_cm"]) for r in rows)
                np.testing.assert_allclose(dbh, [30., 50.], atol=.5 if raw else .1)
                np.testing.assert_allclose([float(r["Height_m"]) for r in rows], [7.93, 7.93], atol=.2)
                if not raw:
                    if [int(r["TreeID"]) for r in rows] != [7, 100003]:
                        raise AssertionError("Nonconsecutive IDs were changed")
                    with laspy.open(source) as a, laspy.open(folder/"参数标注.laz") as b:
                        if [v.record_data for v in a.header.evlrs] != [v.record_data for v in b.header.evlrs]:
                            raise AssertionError("Extended VLRs were lost")
            if screenshot:
                app.processEvents()
                if not window.grab().save(str(screenshot)):
                    raise AssertionError("Cannot render metrics result window")
            return {"fixture":fixture, "raw":raw, "summary":summary, "output_directory":str(folder),
                    "gui_preview_rows":window.preview.rowCount()}
        finally:
            QMessageBox.warning = warning
            if window.worker is not None:
                window.worker.stop()
                window.worker.wait(30000)
                app.processEvents()
            window.close()
