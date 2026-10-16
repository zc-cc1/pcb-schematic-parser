from copy import deepcopy
import unittest
import numpy as np

from pcb_parser.wire_segment_nodes import refine_segment_nodes


def source(*segments):
    return dict(size=[100,100], coordinate_origin='top_left', summary={'segments':len(segments)},
                segments=[dict(segment_id=f'edge_{i+1}', points=[list(a),list(b)], net_id=None, bridges=[])
                          for i,(a,b) in enumerate(segments)])


class NodeTests(unittest.TestCase):
    def test_confident_tip_snaps_endpoint_without_changing_source(self):
        original=source(((10,20),(50,20))); saved=deepcopy(original)
        result=refine_segment_nodes(original,[dict(point=[8,20],confidence=.95)])
        self.assertEqual(result['segments'][0]['points'][0],[8,20])
        self.assertEqual(original,saved)

    def test_low_score_and_distant_parallel_tip_do_not_move_endpoint(self):
        original=source(((10,20),(50,20)))
        for pin in [dict(point=[8,20],confidence=.7),dict(point=[10,23],confidence=.95)]:
            r=refine_segment_nodes(original,[pin])
            self.assertEqual(r['segments'][0]['points'],original['segments'][0]['points'])

    def test_ambiguous_tips_abstain(self):
        original=source(((10,20),(50,20)))
        r=refine_segment_nodes(original,[dict(point=[8,20],confidence=.95),dict(point=[10,18],confidence=.96)])
        self.assertEqual(r['segments'][0]['points'][0],[10,20])

    def test_interior_pin_splits_long_wire(self):
        r=refine_segment_nodes(source(((10,20),(90,20))),[dict(point=[50,21],confidence=.95)],dict(split_at_pins=True))
        self.assertEqual([s['points'] for s in r['segments']],[[[10,20],[50,20]],[[50,20],[90,20]]])

    def test_endpoint_pin_does_not_create_tiny_piece(self):
        r=refine_segment_nodes(source(((10,20),(90,20))),[dict(point=[13,20],confidence=.95)],dict(split_at_pins=True))
        self.assertEqual(len(r['segments']),1)

    def test_gapped_T_endpoint_splits_trunk(self):
        r=refine_segment_nodes(source(((10,50),(90,50)),((50,10),(50,48))),[],dict(split_at_T_nodes=True))
        self.assertEqual(len(r['segments']),3)
        self.assertEqual(r['summary']['node_interior_splits'],1)
        self.assertEqual(r['summary']['node_T_projections'],1)

    def test_opposing_arms_at_crossing_remain_unresolved(self):
        r=refine_segment_nodes(source(((10,50),(90,50)),((50,10),(50,48)),((50,52),(50,90))),[],dict(split_at_T_nodes=True))
        self.assertEqual(len(r['segments']),3)
        self.assertEqual(r['summary']['node_interior_splits'],0)
        self.assertTrue(r['node_refinement']['uncertain_geometric_nodes'])

    def test_guard_blocks_endpoint_extension(self):
        guard=np.zeros((100,100),bool);guard[19:22,8:10]=True
        r=refine_segment_nodes(source(((10,20),(50,20))),[dict(point=[8,20],confidence=.95)],bridge_guard=guard)
        self.assertEqual(r['segments'][0]['points'][0],[10,20])

    def test_disabled_mode_is_geometry_identical(self):
        original=source(((10,20),(50,20)))
        r=refine_segment_nodes(original,[dict(point=[8,20],confidence=.95)],dict(enabled=False))
        self.assertEqual(r['segments'],original['segments'])

    def test_invalid_points_and_guard_are_rejected(self):
        original=source(((10,20),(50,20)))
        with self.assertRaises(ValueError):refine_segment_nodes(original,[dict(point=[float('nan'),20],confidence=.95)])
        with self.assertRaises(ValueError):refine_segment_nodes(original,[],bridge_guard=np.zeros((5,5)))


if __name__=='__main__':unittest.main()
