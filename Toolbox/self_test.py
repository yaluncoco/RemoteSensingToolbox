"""Offline checks that run in both Python and the installed Windows application."""
from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

from PyQt5.QtCore import QEvent, QThread
from PyQt5.QtWidgets import QMessageBox

from Toolbox.version import APP_VERSION


def run_self_test(app, report_path: Path, screenshots: Path | None = None, check_renderer=False) -> int:
    from Toolbox.main import ToolboxWindow
    from Toolbox.tool_registry import TOOLS

    started = time.perf_counter()
    checks = []
    app.setQuitOnLastWindowClosed(False)
    if screenshots:
        screenshots.mkdir(parents=True, exist_ok=True)

    def check(name, callback):
        before = time.perf_counter()
        try:
            detail = callback()
            checks.append({"name": name, "ok": True, "detail": detail,
                           "seconds": round(time.perf_counter() - before, 3)})
        except Exception:
            error = traceback.format_exc()
            logging.error("Self-test %s failed: %s", name, error)
            checks.append({"name": name, "ok": False, "error": error})

    def save_window(window, name):
        deadline = time.monotonic() + .18
        while time.monotonic() < deadline:
            app.processEvents()
            QThread.msleep(10)
        if screenshots:
            if not window.grab().save(str(screenshots / f"{name}.png")):
                raise RuntimeError(f"Could not save screenshot: {name}")

    errors = []
    original_critical = QMessageBox.critical
    QMessageBox.critical = lambda _parent, title, message, *args, **kwargs: errors.append(f"{title}: {message}") or QMessageBox.Ok
    dashboard = ToolboxWindow(persist_settings=False, check_updates=False)
    dashboard.show()
    try:
        check("dashboard", lambda: save_window(dashboard, "dashboard"))

        def update_network_runtime():
            import ssl
            ssl.create_default_context()
            return ssl.OPENSSL_VERSION
        check("update-network-runtime", update_network_runtime)

        dashboard.resize(760, 600)
        dashboard.render_tools()
        check("dashboard-small-screen", lambda: save_window(dashboard, "dashboard-small"))
        for spec in TOOLS:
            def open_tool(spec=spec):
                error_start = len(errors)
                dashboard.open_tool(spec)
                deadline = time.monotonic() + 120
                while dashboard._loader is not None:
                    app.processEvents()
                    QThread.msleep(10)
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Tool import did not finish: {spec.id}")
                if errors[error_start:]:
                    raise RuntimeError("\n".join(errors[error_start:]))
                window = dashboard.open_windows[spec.id]
                save_window(window, spec.id)
                window.close()
                app.sendPostedEvents(None, QEvent.DeferredDelete)
                return spec.title
            check(f"window:{spec.id}", open_tool)
    finally:
        QMessageBox.critical = original_critical
        dashboard.close()
        app.sendPostedEvents(None, QEvent.DeferredDelete)

    with tempfile.TemporaryDirectory(prefix="toolbox-self-test-") as directory:
        root = Path(directory) / "中文 路径"
        root.mkdir()

        def raster_io():
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
            from PIL import Image
            from imageprocession.tif_rgb_converter import convert_tif_to_rgb
            values = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
            values[0, 0, 0] = -9999
            with rasterio.open(root / "source.tif", "w", driver="GTiff", height=16, width=16,
                               count=1, dtype="float32", nodata=-9999, crs="EPSG:3857",
                               transform=from_origin(0, 16, 1, 1)) as output:
                output.write(values)
            for fmt, name in [("RGB TIF", "rgb.tif"), ("PNG", "rgb.png"), ("JPG", "rgb.jpg")]:
                convert_tif_to_rgb(str(root / "source.tif"), str(root / name), fmt, stretch_percent=0)
            with rasterio.open(root / "rgb.tif") as image:
                if image.crs.to_epsg() != 3857 or image.count != 3 or image.nodata is not None:
                    raise AssertionError("RGB TIFF metadata mismatch")
                if image.dataset_mask()[0, 0] != 0 or image.dataset_mask()[0, 1] != 255:
                    raise AssertionError("RGB TIFF mask mismatch")
            for name in ("rgb.png", "rgb.jpg"):
                with Image.open(root / name) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (16, 16):
                        raise AssertionError("RGB image mismatch")
            return "GeoTIFF, PNG and JPEG; nodata mask; Unicode paths"
        check("raster-conversion", raster_io)

        def vector_io():
            import fiona
            import rasterio
            from imageprocession.shp_label_rasterizer import rasterize_shp_labels
            from imageprocession.tif_processor import TifProcessThread
            schema = {"geometry": "Polygon", "properties": {"Label": "str"}}
            geometry = {"type": "Polygon", "coordinates": [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]]}
            for driver, name in [("ESRI Shapefile", "labels.shp"), ("GPKG", "labels.gpkg")]:
                with fiona.open(root / name, "w", driver=driver, crs="EPSG:3857", schema=schema) as output:
                    output.write({"geometry": geometry, "properties": {"Label": "tree"}})
            rasterize_shp_labels(str(root / "labels.shp"), str(root / "source.tif"),
                                 str(root / "labels.tif"), "Label", {"tree": 300})
            with rasterio.open(root / "labels.tif") as output:
                if output.dtypes != ("uint16",) or output.read(1)[11, 4] != 300:
                    raise AssertionError("16-bit raster labels mismatch")
            worker = TifProcessThread(str(root / "labels.gpkg"), str(root / "source.tif"),
                                      str(root / "gpkg"), "2026", "1", ["tree"], {"tree": [20, 100, 200]})
            results = []
            worker.finished_signal.connect(lambda *args: results.append(args))
            worker.run()
            if not results or results[-1][1] != "info" or len(list((root / "gpkg").glob("*.tif"))) != 3:
                raise AssertionError(f"GPKG processing failed: {results}")
            return "SHP/GPKG read/write; 16-bit labels; three georeferenced GPKG outputs"
        check("vector-rasterization", vector_io)

        def preview_and_tiles():
            import numpy as np
            from imageprocession.visualizer import read_tif_rgb_preview, get_pyplot
            from imageprocession.image_cutter import cut_image_tiles
            preview = read_tif_rgb_preview(str(root / "source.tif"))
            if preview.shape != (16, 16, 3):
                raise AssertionError("Preview shape mismatch")
            plot = get_pyplot()
            figure, axes = plot.subplots()
            axes.imshow(preview)
            figure.canvas.draw()
            plot.close(figure)
            cut_image_tiles(str(root / "source.tif"), str(root / "labels.tif"), str(root / "tiles"), 8, 0, 1)
            images = list((root / "tiles" / "images").glob("*"))
            labels = list((root / "tiles" / "labels").glob("*"))
            if len(images) != 1 or [path.stem for path in images] != [path.stem for path in labels]:
                raise AssertionError("Expected one labeled tile after filtering background-only tiles")
            from PIL import Image
            with Image.open(labels[0]) as label:
                if set(np.unique(np.asarray(label))) != {0, 300}:
                    raise AssertionError("Tile label IDs changed")
            return "TIFF preview, matplotlib rendering, paired 16-bit label tiles and background filtering"
        check("preview-and-tiles", preview_and_tiles)

        def point_cloud_io():
            import numpy as np
            import open3d as o3d
            from PointCloud.processing import downsample_points
            from PointCloud.file_io import read_point_cloud, write_point_cloud
            points = np.array([[0., 0, 0], [.1, 0, 0], [5, 0, 0], [5.1, 0, 0]])
            features = np.array([[255, 0, 0, 7], [255, 0, 0, 8], [0, 255, 0, 9], [0, 255, 0, 10]])
            sampled, retained = downsample_points(points, features, "voxel", {"voxel_size": 1})
            if sampled.shape != (2, 3) or retained.shape != (2, 4) or retained[:, :3].max() != 255:
                raise AssertionError("Point attributes were not retained")
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(sampled)
            for suffix in ("ply", "pcd"):
                path = str(root / ("cloud." + suffix))
                if not write_point_cloud(path, cloud) or len(read_point_cloud(path).points) != 2:
                    raise AssertionError(f"Open3D {suffix} I/O failed")
            return "Open3D PLY/PCD I/O; voxel sampling with RGB and label preservation"
        check("point-cloud-processing", point_cloud_io)

        def update_installer_validation():
            import hashlib
            import threading
            from Toolbox.updater import parse_release
            from Toolbox.update_installer import verify_installer
            content = b"MZ" + b"offline installer integrity fixture"
            name = "RemoteSensingToolbox-99.0.0-windows-x64-setup.exe"
            release = parse_release({"tag_name": "v99.0.0", "assets": [{
                "name": name, "state": "uploaded", "size": len(content),
                "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            }]})
            installer = root / name
            installer.write_bytes(content)
            verify_installer(installer, release.installer, threading.Event())
            return "Installer metadata, SHA-256/size verification, Unicode cache path; no installer executed"
        check("update-installer-validation", update_installer_validation)

        def prediction_visualization():
            import numpy as np
            from PointCloud.visualization_data import (
                VisualizationOptions, read_visualization_text, extract_visualization_data, apply_label_colors,
            )
            path = root / "预测 点云.csv"
            path.write_text("X,Y,Z,Label\n1,2,3,1\n4,5,6,-1\n", encoding="utf-8-sig")
            options = VisualizationOptions(layout="xyz_label", skip_rows=1)
            points, labels, _ = extract_visualization_data(read_visualization_text(path, options), options)
            np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_array_equal(labels, [1, -1])
            np.testing.assert_array_equal(apply_label_colors(labels, {1: [1, 0, 0]}, [.5, .5, .5]),
                                          [[1, 0, 0], [.5, .5, .5]])
            return "XYZ + Label; CSV/BOM/header/Unicode; original label IDs and fallback colors"
        check("prediction-visualization", prediction_visualization)

        def colors_and_voc():
            import numpy as np
            from PIL import Image, ImageDraw
            from imageprocession.rgb_color_extractor import detect_color_blocks
            from imageprocession.voc_creator import VOCCreatorThread
            legend = Image.new("RGB", (300, 160), "white")
            draw = ImageDraw.Draw(legend)
            for index, color in enumerate([(255, 0, 0), (0, 180, 0), (0, 0, 255)]):
                draw.rectangle((20, 20 + index * 40, 60, 40 + index * 40), fill=color)
            blocks = detect_color_blocks(legend, 60, 8, 6, 24, False, "按列从上到下")
            if len(blocks) != 3:
                raise AssertionError("Color block extraction failed")
            images, labels = root / "voc-images", root / "voc-labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (8, 8), "green").save(images / "sample.PNG")
            Image.fromarray(np.full((8, 8), 300, dtype=np.uint16)).save(labels / "sample.TIF")
            worker = VOCCreatorThread(str(images), str(labels), str(root / "voc"), "VOC2026", .8)
            results = []
            worker.finished_signal.connect(lambda *args: results.append(args))
            worker.run()
            if not results or results[-1][1] != "info":
                raise AssertionError(f"VOC processing failed: {results}")
            with Image.open(root / "voc/VOCdevkit/VOC2026/SegmentationClass/sample.png") as output:
                if not np.all(np.asarray(output) == 300):
                    raise AssertionError("VOC class IDs changed")
            return "Three RGB colors; VOC split with 16-bit labels"
        check("colors-and-voc", colors_and_voc)

    if check_renderer:
        def render_point_cloud():
            import numpy as np
            import open3d as o3d
            visualizer = o3d.visualization.Visualizer()
            if not visualizer.create_window(window_name="Toolbox renderer check", width=320, height=240, visible=False):
                raise RuntimeError("OpenGL window could not be created")
            try:
                visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
                visualizer.poll_events()
                visualizer.update_renderer()
                pixels = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
                if pixels.shape != (240, 320, 3) or not np.isfinite(pixels).all() or pixels.max() - pixels.min() < .1:
                    raise AssertionError("The native renderer returned an empty frame")
                return "Native OpenGL frame rendered successfully (320 x 240)"
            finally:
                visualizer.destroy_window()
        check("native-3d-renderer", render_point_cloud)

    versions = {}
    for name in ("PyQt5", "numpy", "Pillow", "rasterio", "fiona", "scipy", "scikit-learn", "matplotlib", "open3d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "metadata unavailable"
    report = {"version": APP_VERSION, "frozen": bool(getattr(sys, "frozen", False)),
              "platform": platform.platform(), "python": platform.python_version(), "dependencies": versions,
              "ok": all(item["ok"] for item in checks), "checks": checks,
              "seconds": round(time.perf_counter() - started, 3)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
