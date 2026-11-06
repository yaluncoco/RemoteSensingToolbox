import os
from pathlib import Path
import tempfile
import unittest

import laspy
import numpy as np
from plyfile import PlyData, PlyElement

from PointCloud import cloudcompare
from PointCloud.forestry_input import identify_input, normalize_input, ply_to_laz
from PointCloud.forestry_io import fingerprint


class InputFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_content_detection_overrides_extension(self):
        path = self.root / "wrong.laz"
        path.write_bytes(b'CCB2' + bytes(16))
        self.assertEqual(identify_input(path), "cloudcompare_bin")
        path = self.root / "wrong.bin"
        path.write_bytes(b'LASF' + bytes(16))
        self.assertEqual(identify_input(path), "las")

    def test_invalid_signature_has_actionable_error(self):
        path = self.root / "wrong.laz"
        path.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, "修改扩展名不会转换格式"):
            identify_input(path)

    def test_ply_normalization_keeps_exact_coordinates_colors_and_all_fields(self):
        vertex = np.zeros(4, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                         ('scalar_Intensity', 'f4'), ('scalar_Gps_Time', 'f4'),
                         ('scalar_very_long_name_that_needs_shortening', 'f4'), ('nx', 'f4')])
        for i, name in enumerate('xyz'):
            vertex[name] = 434000.123456789 + i*1000000 + np.arange(4)*.123456789
        vertex['red'] = [0, 1, 128, 255]
        vertex['scalar_Intensity'] = [0, 12, 65535, 12.5]
        vertex['scalar_Gps_Time'] = [123.456, 1, 2, 3]
        vertex['scalar_very_long_name_that_needs_shortening'] = [0, np.nan, np.inf, -0.]
        path = self.root / "cloud.ply"
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)
        result = ply_to_laz(path, self.root / 'normalized.laz')
        cloud = laspy.read(self.root / 'normalized.laz')
        self.assertLessEqual(max(result['max_coordinate_rounding_error']), 5.1e-7)
        for name, extra in result['property_mapping'].items():
            self.assertEqual(np.asarray(cloud[extra]).tobytes(), vertex[name].tobytes())
        np.testing.assert_array_equal(cloud.red, [0, 257, 32896, 65535])
        np.testing.assert_array_equal(cloud.intensity, [0, 12, 65535, 0])

    def test_nan_coordinates_rejected(self):
        vertex = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        vertex['x'] = np.nan
        path = self.root / 'bad.ply'
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(path)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            ply_to_laz(path, self.root / 'out.laz')


@unittest.skipUnless(os.environ.get('TOOLBOX_TEST_CLOUDCOMPARE'), 'Set TOOLBOX_TEST_CLOUDCOMPARE for BIN integration tests')
class BinIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '中文 空格'
        self.root.mkdir()
        self.binary = os.environ['TOOLBOX_TEST_CLOUDCOMPARE']

    def make_bin(self, multiple=False):
        xyz = np.array([(x*.1, y*.1, x*.2) for x in range(10) for y in range(10)])
        from PointCloud.forestry import write_ply
        write_ply(self.root / 'cloud.ply', xyz, {'Intensity': np.arange(100, dtype='f4')})
        args = ['-C_EXPORT_FMT', 'BIN', '-O', str(self.root / 'cloud.ply')]
        if multiple:
            args += ['-O', str(self.root / 'cloud.ply')]
        args += ['-SAVE_CLOUDS', 'ALL_AT_ONCE', 'FILE', '../disguised.laz']
        with cloudcompare.execution(self.binary):
            cloudcompare.run(args, self.root / 'make-bin')
        return self.root / 'disguised.laz'

    def test_misnamed_bin_is_converted_and_source_unchanged(self):
        source = self.make_bin()
        before = fingerprint(source)
        output = self.root / 'job'
        output.mkdir()
        with cloudcompare.execution(self.binary):
            target, info = normalize_input(source, output, identify_input(source))
        self.assertEqual(fingerprint(source), before)
        self.assertEqual(info['point_count'], 100)
        self.assertEqual(laspy.read(target).header.point_count, 100)
        self.assertTrue(info['converted'])
        self.assertTrue(all(info['verification'].values()))

    def test_multiple_clouds_are_not_silently_dropped(self):
        source = self.make_bin(multiple=True)
        output = self.root / 'job'
        output.mkdir()
        with cloudcompare.execution(self.binary):
            with self.assertRaises((ValueError, RuntimeError)):
                normalize_input(source, output, identify_input(source))
        self.assertFalse((output / 'input.json').exists())


if __name__ == '__main__':
    unittest.main()
