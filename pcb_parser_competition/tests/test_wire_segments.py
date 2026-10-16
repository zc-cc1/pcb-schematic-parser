import unittest
import numpy as np

from pcb_parser.wire_segments import decode_wire_segments, export_candidate_edges, component_bridge_guard


class WireSegmentTests(unittest.TestCase):
    def decode(self, wire, **kwargs):
        return decode_wire_segments(wire.astype(float), **kwargs)

    def test_straight_wire_has_two_original_endpoints(self):
        p = np.zeros((60, 90)); p[20, 5:81] = 1
        result = self.decode(p)
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(sorted(result['segments'][0]['points']), [[5., 20.], [80., 20.]])

    def test_small_gap_merges_without_rewriting_pixels(self):
        p = np.zeros((60, 90)); p[20, 5:40] = 1; p[20, 43:81] = 1
        original = p.copy(); result = self.decode(p)
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['summary']['merged_gaps'], 1)
        self.assertTrue(np.array_equal(p, original))

    def test_long_gap_is_not_merged(self):
        p = np.zeros((60, 90)); p[20, 5:35] = 1; p[20, 45:81] = 1
        self.assertEqual(len(self.decode(p)['segments']), 2)

    def test_staggered_parallel_wires_do_not_join(self):
        p = np.zeros((60, 90)); p[20, 5:40] = 1; p[23, 42:81] = 1
        self.assertEqual(len(self.decode(p)['segments']), 2)

    def test_L_corner_stays_two_segments(self):
        p = np.zeros((60, 90)); p[20, 5:41] = 1; p[20:51, 40] = 1
        self.assertEqual(len(self.decode(p)['segments']), 2)

    def test_T_junction_is_not_merged_across(self):
        p = np.zeros((60, 90)); p[20, 5:81] = 1; p[20:51, 40] = 1
        result = self.decode(p)
        self.assertEqual(len(result['segments']), 3)
        self.assertEqual(result['summary']['merged_gaps'], 0)

    def test_diagonal_is_preserved(self):
        p = np.zeros((60, 90))
        for i in range(5, 45): p[i, i] = 1
        result = self.decode(p)
        self.assertEqual(len(result['segments']), 1)
        np.testing.assert_allclose(sorted(result['segments'][0]['points']), [[5., 5.], [44., 44.]], atol=1e-10)

    def test_short_spur_is_pruned_but_isolated_short_wire_is_kept(self):
        p = np.zeros((60, 90)); p[20, 5:81] = 1; p[19, 40] = 1; p[40, 5:8] = 1
        result = self.decode(p)
        self.assertEqual(len(result['segments']), 2)
        self.assertTrue(any(abs(np.linalg.norm(np.diff(s['points'], axis=0))-2) < 1e-6 for s in result['segments']))

    def test_component_body_block_prevents_gap_bridge(self):
        p = np.zeros((60, 90)); p[20, 5:40] = 1; p[20, 43:81] = 1
        blocked = np.zeros_like(p, bool); blocked[19:22, 40:43] = True
        self.assertEqual(len(self.decode(p, blocked_mask=blocked)['segments']), 2)

    def test_two_predicted_pin_tips_prevent_shorting_across_a_component(self):
        p = np.zeros((60, 90)); p[20, 5:40] = 1; p[20, 43:81] = 1
        self.assertEqual(len(self.decode(p, pin_points=[[39, 20], [43, 20]])['segments']), 2)

    def test_symbol_body_guard_blocks_bridging_without_removing_existing_pixels(self):
        p = np.zeros((60, 90)); p[20, 5:40] = 1; p[20, 43:81] = 1
        guard = component_bridge_guard(p.shape, [dict(bbox_top_left=[35, 12, 48, 28], confidence=.9)])
        before = p.copy()
        result = self.decode(p, bridge_blocked_mask=guard)
        self.assertEqual(len(result['segments']), 2)
        self.assertTrue(np.array_equal(p, before))
        self.assertEqual(result['summary']['before_gap_merging'], 2)

    def test_export_uses_two_points_and_leaves_network_assignment_pending(self):
        p = np.zeros((60, 90)); p[20, 5:81] = 1
        exported = export_candidate_edges(self.decode(p))
        self.assertFalse(exported['nets_assigned'])
        self.assertEqual(sorted(exported['edges']['edge_1'], key=lambda x:x['x']), [{'x':5.,'y':40.},{'x':80.,'y':40.}])

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(self.decode(np.zeros((40, 40)))['segments'], [])
        for p in (np.zeros(5), np.full((3, 3), np.nan), np.full((3, 3), 2.)):
            with self.assertRaises(ValueError): self.decode(p)
        with self.assertRaises(ValueError): self.decode(np.zeros((3, 3)), blocked_mask=np.zeros((2, 2)))

    def test_three_fragments_merge_deterministically(self):
        p = np.zeros((60, 120)); p[20, 5:35] = 1; p[20, 38:68] = 1; p[20, 71:101] = 1
        a = self.decode(p); b = self.decode(p)
        self.assertEqual(a, b)
        self.assertEqual(len(a['segments']), 1)
        self.assertEqual(a['summary']['merged_gaps'], 2)

    def test_closed_rectangle_remains_four_segments(self):
        p = np.zeros((60, 90)); p[10, 10:41] = 1; p[40, 10:41] = 1; p[10:41, 10] = 1; p[10:41, 40] = 1
        result = self.decode(p)
        self.assertEqual(len(result['segments']), 4)
        self.assertEqual(result['summary']['merged_gaps'], 0)

    def test_short_fragments_cannot_justify_large_invented_span(self):
        p = np.zeros((40, 40)); p[20, 5:8] = 1; p[20, 11:14] = 1
        self.assertEqual(len(self.decode(p)['segments']), 2)


if __name__ == '__main__': unittest.main()
