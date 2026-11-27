"""Known-identity partitions: touching canopies, false fragments and missing stems."""
import unittest
import numpy as np

from PointCloud.tree_refinement import detect_anchors, refine_labels


def tube(center=(0., 0.), lean=(0., 0.), step=.12, angular=18, crown=True):
    h = np.arange(.45, 5.05 if crown else 2.6, step)
    theta = np.arange(angular)*2*np.pi/angular
    radius = .16 + (np.clip((h-2.5)/1.5, 0, 1)*.7 if crown else np.zeros(len(h)))
    p = np.column_stack((np.repeat(center[0]+lean[0]*(h-1.3), angular)+np.repeat(radius, angular)*np.tile(np.cos(theta), len(h)),
                         np.repeat(center[1]+lean[1]*(h-1.3), angular)+np.repeat(radius, angular)*np.tile(np.sin(theta), len(h)),
                         np.repeat(h, angular)))
    return p


def anchor(center, lean=(0., 0.)):
    return dict(center=list(center), lean=list(lean), radius=.16, slices=4,
                axis_residual_m=0., fit_rmse_m=0., height_min=.7, height_max=2.35)


def purity(expected, actual):
    # Both directions: prevents one giant group or one group per point passing.
    def direction(a, b):
        return sum(np.unique(b[a == k], return_counts=True)[1].max() for k in np.unique(a))/len(a)
    return min(direction(expected, actual), direction(actual, expected))


class RefinementTests(unittest.TestCase):
    def test_detect_multislice_close_and_leaning_stems(self):
        full = np.concatenate([tube((x, 0), (.15, .04), .02, 60, False) for x in (0., .65)])
        anchors = detect_anchors(full, full[:, 2])
        self.assertEqual(len(anchors), 2)
        centers = sorted([a['center'] for a in anchors])
        np.testing.assert_allclose(centers, [[0, 0], [.65, 0]], atol=.02)
        for a in anchors:
            np.testing.assert_allclose(a['lean'], [.15, .04], atol=.02)

    def test_one_slice_or_partial_arc_cannot_seed_a_tree(self):
        p = tube(step=.02, angular=60, crown=False)
        p = p[np.abs(p[:, 2]-1.25) < .1]
        self.assertEqual(detect_anchors(p, p[:, 2]), [])
        p = tube(step=.02, angular=60, crown=False)
        p = p[(p[:, 0] > .12) & (p[:, 1] > 0)]
        self.assertEqual(detect_anchors(p, p[:, 2]), [])

    def test_split_touching_crowns_by_stem_paths(self):
        p = np.concatenate([tube((x, 0)) for x in (0., 1.9)])
        truth = np.repeat([1, 2], len(p)//2)
        labels, _, report = refine_labels(p, p[:, 2], np.zeros(len(p), int), [anchor((0., 0)), anchor((1.9, 0))])
        self.assertEqual(report['split_segment_count'], 1)
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertGreater(purity(truth, labels), .98)

    def test_merge_three_fragments_of_one_stem(self):
        p = tube()
        # Longitudinal split: every fragment has independent low stem support.
        angle = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2*np.pi)
        raw = np.minimum((angle/(2*np.pi)*3).astype(int), 2)
        labels, _, report = refine_labels(p, p[:, 2], raw, [anchor((0., 0))])
        self.assertEqual(len(np.unique(labels)), 1)
        self.assertEqual(report['merged_group_count'], 1)

    def test_attach_detached_upper_fragment_with_3d_contacts(self):
        p = tube()
        raw = (p[:, 2] > 3.8).astype(int)
        labels, _, report = refine_labels(p, p[:, 2], raw, [anchor((0., 0))])
        self.assertEqual(len(np.unique(labels)), 1)
        self.assertEqual(report['attached_fragment_count'], 1)

    def test_stray_stem_points_do_not_absorb_large_understory_block(self):
        stem = tube()
        rng = np.random.default_rng(551)
        understory = rng.uniform([.6, -.8, .45], [3., .8, 2.4], (3000, 3))
        p = np.concatenate((stem, understory))
        raw = np.r_[np.zeros(len(stem), int), np.ones(len(understory), int)]
        # A few genuine points from the same bole were misassigned to the
        # understory proposal by the upstream engine; both raw groups have
        # enough seed height support to trigger the old whole-group merge.
        contamination = np.flatnonzero((stem[:, 2] < 2.5) & (np.arange(len(stem)) % 18 == 0))
        raw[contamination] = 1
        labels, info, report = refine_labels(p, p[:, 2], raw, [anchor((0., 0))])
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(report['merged_group_count'], 0)
        self.assertEqual(report['attached_fragment_count'], 0)
        self.assertIn('mixed_low_vegetation', [v['reason'] for v in info.values()])

    def test_two_close_distinct_stems_never_merge(self):
        p = np.concatenate([tube((x, 0), crown=False) for x in (0., .65)])
        truth = np.repeat([1, 2], len(p)//2)
        labels, _, report = refine_labels(p, p[:, 2], truth, [anchor((0., 0)), anchor((.65, 0))])
        self.assertEqual(purity(truth, labels), 1.)
        self.assertEqual(report['merged_group_count'], 0)

    def test_leaning_paths_preserve_crown_ownership(self):
        p = np.concatenate([tube((x, 0), (.4, 0)) for x in (0., 1.9)])
        truth = np.repeat([1, 2], len(p)//2)
        labels, _, _ = refine_labels(p, p[:, 2], np.zeros(len(p), int), [anchor((0., 0), (.4, 0)), anchor((1.9, 0), (.4, 0))])
        self.assertGreater(purity(truth, labels), .97)
        # Horizontal nearest stem demonstrably misassigns the leaning crown.
        nearest = (p[:, 0] > .95).astype(int)+1
        self.assertGreater(purity(truth, labels), purity(truth, nearest)+.1)

    def test_missing_stems_preserves_original_partition(self):
        p = np.concatenate([tube((x, 0)) for x in (0., 1.9)])
        raw = np.repeat([6, 19], len(p)//2)
        labels, _, report = refine_labels(p, p[:, 2], raw, [])
        self.assertEqual(purity(raw, labels), 1.)
        self.assertEqual(report['split_segment_count'], 0)
        self.assertEqual(report['merged_group_count'], 0)

    def test_unverified_low_stem_not_absorbed(self):
        p = np.concatenate([tube((x, 0)) for x in (0., 1.9)])
        raw = np.repeat([6, 19], len(p)//2)
        labels, info, _ = refine_labels(p, p[:, 2], raw, [anchor((0., 0))])
        self.assertEqual(purity(raw, labels), 1.)
        self.assertIn('unverified_low_stem', [v['reason'] for v in info.values()])

    def test_far_fragment_not_forced_to_nearest_stem(self):
        p = np.concatenate([tube(), tube((8., 0))+[0, 0, 8]])
        raw = np.repeat([0, 1], len(p)//2)
        labels, _, _ = refine_labels(p, p[:, 2], raw, [anchor((0., 0))])
        self.assertEqual(purity(raw, labels), 1.)

    def test_ambiguous_crown_fragment_remains_separate(self):
        p1, p2 = tube((-.4, 0), crown=False), tube((.4, 0), crown=False)
        # A fragment equally close to the tops of two distinct stems.
        cap = np.array([[0, y, h] for y in np.linspace(-.1, .1, 15) for h in np.linspace(2.7, 3., 12)])
        p = np.concatenate((p1, p2, cap))
        raw = np.r_[np.zeros(len(p1), int), np.ones(len(p2), int), np.full(len(cap), 2)]
        labels, info, report = refine_labels(p, p[:, 2], raw, [anchor((-.4, 0)), anchor((.4, 0))])
        self.assertEqual(purity(raw, labels), 1.)
        self.assertEqual(report['attached_fragment_count'], 0)

    def test_point_permutation_preserves_partition(self):
        p = np.concatenate([tube((x, 0)) for x in (0., 1.9)])
        raw = np.zeros(len(p), int)
        anchors = [anchor((0., 0)), anchor((1.9, 0))]
        a, _, _ = refine_labels(p, p[:, 2], raw, anchors)
        order = np.random.default_rng(41).permutation(len(p))
        b, _, _ = refine_labels(p[order], p[order, 2], raw[order], anchors)
        self.assertEqual(purity(a[order], b), 1.)

    def test_cancelled_refinement_stops(self):
        def cancel():
            raise RuntimeError('cancelled')
        p = tube()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            refine_labels(p, p[:, 2], np.zeros(len(p), int), [anchor((0, 0))], check=cancel)

    def test_known_partition_through_full_source_export(self):
        from Toolbox.forestry_self_test import check_refinement
        self.assertGreater(check_refinement()['partition_agreement'], .97)


if __name__ == '__main__':
    unittest.main()
