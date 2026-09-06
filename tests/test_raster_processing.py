import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

from Toolbox.tool_registry import configure_runtime_environment
configure_runtime_environment()

import numpy as np
import rasterio
import fiona
from PIL import Image
from rasterio.transform import from_origin
from imageprocession.tif_rgb_converter import convert_tif_to_rgb, normalize_band_to_uint8, ConversionCancelled
from imageprocession.shp_label_rasterizer import rasterize_shp_labels, RasterizeCancelled
from imageprocession.image_cutter import cut_image_tiles


class RasterProcessingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="toolbox-tests-")
        self.root = Path(self.scratch.name) / "中文 空格"
        self.root.mkdir()

    def tearDown(self):
        self.scratch.cleanup()

    def write_tif(self, name="input.tif", data=None, nodata=-9999):
        if data is None:
            data = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
            data[0, 0, 0] = nodata
        path = self.root / name
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                           count=data.shape[0], dtype=data.dtype, nodata=nodata,
                           crs="EPSG:3857", transform=from_origin(0, data.shape[1], 1, 1)) as dst:
            dst.write(data)
        return path

    def test_masked_values_do_not_change_stretch(self):
        band = np.ma.array([-9999.0, 10, 20, 30, np.nan], mask=[1, 0, 0, 0, 0])
        np.testing.assert_array_equal(normalize_band_to_uint8(band, 0), [0, 0, 127, 255, 0])

    def test_rgb_tif_preserves_crs_and_distinguishes_valid_black(self):
        source = self.write_tif()
        output = self.root / "rgb.tif"
        convert_tif_to_rgb(str(source), str(output), "RGB TIF", stretch_percent=0)
        with rasterio.open(source) as src, rasterio.open(output) as dst:
            self.assertEqual(dst.count, 3)
            self.assertEqual(dst.dtypes, ("uint8",) * 3)
            self.assertEqual(dst.crs, src.crs)
            self.assertEqual(dst.transform, src.transform)
            self.assertIsNone(dst.nodata)
            self.assertEqual(int(dst.dataset_mask()[0, 0]), 0)
            self.assertEqual(int(dst.dataset_mask()[0, 1]), 255)
            self.assertEqual(int(dst.read(1)[0, 1]), 0)
        self.assertFalse(Path(str(output) + ".msk").exists())

    def test_global_stretch_has_no_tile_seams(self):
        values = np.arange(1025, dtype=np.float32)[None, None, :].repeat(3, 0).repeat(2, 1)
        source = self.write_tif(data=values)
        output = self.root / "seams.tif"
        convert_tif_to_rgb(str(source), str(output), "RGB TIF", stretch_percent=0)
        with rasterio.open(output) as dst:
            expected = (np.arange(1025) * (255.0 / 1024)).astype(np.uint8)
            np.testing.assert_array_equal(dst.read(1)[0], expected)

    def test_png_and_jpeg_are_readable_rgb_images(self):
        source = self.write_tif()
        for fmt, extension in [("PNG", ".png"), ("JPG", ".jpg")]:
            with self.subTest(fmt=fmt):
                output = self.root / ("preview" + extension)
                convert_tif_to_rgb(str(source), str(output), fmt)
                with Image.open(output) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (8, 8))
                self.assertFalse(Path(str(output) + ".aux.xml").exists())

    def test_cancel_preserves_previous_output_and_cleans_temporary_files(self):
        source = self.write_tif(data=np.ones((1, 20, 1025), np.float32))
        output = self.root / "previous.tif"
        output.write_bytes(b"previous completed output")
        cancelled = [False]
        with self.assertRaises(ConversionCancelled):
            convert_tif_to_rgb(str(source), str(output), "RGB TIF", stretch_percent=0,
                               progress_callback=lambda value, _: cancelled.__setitem__(0, value >= 30),
                               should_cancel=lambda: cancelled[0])
        self.assertEqual(output.read_bytes(), b"previous completed output")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["input.tif", "previous.tif"])

    def test_rejects_source_overwrite_and_invalid_parameters(self):
        source = self.write_tif()
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            convert_tif_to_rgb(str(source), str(source), "RGB TIF")
        self.assertEqual(source.read_bytes(), original)
        for value in [-1, 50, float("nan")]:
            with self.subTest(stretch=value), self.assertRaises(ValueError):
                convert_tif_to_rgb(str(source), str(self.root / "bad.tif"), "RGB TIF", stretch_percent=value)
        two_band = self.write_tif("two.tif", np.ones((2, 4, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            convert_tif_to_rgb(str(two_band), str(self.root / "bad.tif"), "RGB TIF")
        self.assertFalse((self.root / "bad.tif").exists())

    def vector_fixture(self):
        source = self.write_tif(data=np.ones((1, 16, 16), dtype=np.float32))
        shape = self.root / "标注.shp"
        with fiona.open(shape, "w", driver="ESRI Shapefile", crs="EPSG:3857",
                        schema={"geometry": "Polygon", "properties": {"Label": "str"}}) as dst:
            dst.write({"geometry": {"type": "Polygon", "coordinates": [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]]},
                       "properties": {"Label": "tree"}})
        return source, shape

    def test_shp_rasterization_preserves_large_class_ids(self):
        source, shape = self.vector_fixture()
        output = self.root / "label.tif"
        rasterize_shp_labels(str(shape), str(source), str(output), "Label", {"tree": 300})
        with rasterio.open(output) as result:
            self.assertEqual(result.dtypes, ("uint16",))
            self.assertEqual(int(result.read(1)[11, 4]), 300)
            self.assertEqual(int(result.read(1)[0, 0]), 0)

    def test_cancelled_shp_output_does_not_replace_completed_file(self):
        source, shape = self.vector_fixture()
        output = self.root / "label.tif"
        output.write_bytes(b"old labels")
        cancelled = [False]
        with self.assertRaises(RasterizeCancelled):
            rasterize_shp_labels(str(shape), str(source), str(output), "Label", {"tree": 1},
                                 progress_callback=lambda value, _: cancelled.__setitem__(0, value >= 80),
                                 should_cancel=lambda: cancelled[0])
        self.assertEqual(output.read_bytes(), b"old labels")

    def test_image_and_label_tiles_remain_aligned(self):
        source = self.write_tif(data=np.arange(100, dtype=np.float32).reshape(1, 10, 10))
        label = self.write_tif("labels.tif", np.full((1, 10, 10), 7, dtype=np.uint8), nodata=0)
        output = self.root / "tiles"
        cut_image_tiles(str(source), str(label), str(output), 8, 0, 1)
        images = sorted((output / "images").glob("*"))
        labels = sorted((output / "labels").glob("*"))
        self.assertEqual(len(images), 4)
        self.assertEqual([p.stem for p in images], [p.stem for p in labels])
        for path in labels:
            with Image.open(path) as image:
                self.assertEqual(image.size, (8, 8))
                self.assertTrue(set(np.asarray(image).ravel()).issubset({0, 7}))

    def test_tiles_preserve_16_bit_class_ids(self):
        source = self.write_tif(data=np.arange(100, dtype=np.float32).reshape(1, 10, 10))
        label = self.write_tif("labels.tif", np.full((1, 10, 10), 300, dtype=np.uint16), nodata=0)
        output = self.root / "tiles"
        cut_image_tiles(str(source), str(label), str(output), 8, 0, 1)
        for path in (output / "labels").glob("*.png"):
            with Image.open(path) as image:
                values = set(np.unique(np.asarray(image)))
                self.assertIn(300, values)
                self.assertTrue(values.issubset({0, 300}))
