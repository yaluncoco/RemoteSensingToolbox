import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

from Toolbox.tool_registry import configure_runtime_environment
configure_runtime_environment()

import numpy as np
from PIL import Image
import fiona
import rasterio
from rasterio.transform import from_origin
from imageprocession.tif_processor import TifProcessThread
from imageprocession.voc_creator import VOCCreatorThread


class DatasetProcessingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_gpkg_outputs_share_geometry_and_do_not_inherit_invalid_nodata(self):
        source = self.root / "source.tif"
        transform = from_origin(0, 16, 1, 1)
        with rasterio.open(source, "w", driver="GTiff", height=16, width=16, count=1,
                           dtype="float32", nodata=-9999, crs="EPSG:3857", transform=transform) as dst:
            dst.write(np.ones((1, 16, 16), dtype=np.float32))
        vector = self.root / "labels.gpkg"
        with fiona.open(vector, "w", driver="GPKG", crs="EPSG:3857",
                        schema={"geometry": "Polygon", "properties": {"Label": "str"}}) as dst:
            dst.write({"geometry": {"type": "Polygon", "coordinates": [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]]},
                       "properties": {"Label": "tree"}})
        worker = TifProcessThread(str(vector), str(source), str(self.root / "out"), "2026", "1", ["tree"], {"tree": [20, 100, 200]})
        result = []
        worker.finished_signal.connect(lambda *args: result.append(args))
        worker.run()
        self.assertEqual(result[-1][1], "info", result)
        outputs = list((self.root / "out").glob("*.tif"))
        self.assertEqual(len(outputs), 3)
        for path in outputs:
            with rasterio.open(path) as dst:
                self.assertEqual(dst.crs.to_epsg(), 3857)
                self.assertEqual(dst.transform, transform)
                if dst.count == 3:
                    np.testing.assert_array_equal(dst.read()[:, 11, 4], [20, 100, 200])
                    self.assertIsNone(dst.nodata)
                else:
                    self.assertEqual(dst.read(1)[11, 4], 1)

    def test_voc_keeps_16_bit_class_ids_and_matches_uppercase_extensions(self):
        images, labels = self.root / "images", self.root / "labels"
        images.mkdir()
        labels.mkdir()
        Image.new("RGB", (8, 8), "green").save(images / "sample.PNG")
        Image.fromarray(np.full((8, 8), 300, dtype=np.uint16)).save(labels / "sample.TIF")
        worker = VOCCreatorThread(str(images), str(labels), str(self.root / "voc"), "VOC2026", .8)
        result = []
        worker.finished_signal.connect(lambda *args: result.append(args))
        worker.run()
        self.assertEqual(result[-1][1], "info", result)
        dataset = self.root / "voc" / "VOCdevkit" / "VOC2026"
        with Image.open(dataset / "SegmentationClass" / "sample.png") as output:
            np.testing.assert_array_equal(np.asarray(output), 300)
        self.assertEqual((dataset / "ImageSets" / "Segmentation" / "train.txt").read_text(encoding="utf-8"), "sample")

    def test_voc_rejects_duplicate_stems_and_invalid_dataset_names(self):
        images, labels = self.root / "images", self.root / "labels"
        images.mkdir()
        labels.mkdir()
        for suffix in ("png", "jpg"):
            Image.new("RGB", (2, 2)).save(images / ("sample." + suffix))
        for name in ("VOC2026", "../escape"):
            worker = VOCCreatorThread(str(images), str(labels), str(self.root / "out"), name, .8)
            result = []
            worker.finished_signal.connect(lambda *args: result.append(args))
            worker.run()
            self.assertEqual(result[-1][1], "error", result)
