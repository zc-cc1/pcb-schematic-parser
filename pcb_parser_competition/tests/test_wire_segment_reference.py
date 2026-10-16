import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_wire_segment_nodes_v2 import overlap_union


def segment(a,b,edge='wire'):
    return dict(points=[a,b],edge_id=edge,length=abs(b[0]-a[0])+abs(b[1]-a[1]))


class ReferenceInterpretationTests(unittest.TestCase):
    def test_overlapping_same_edge_is_counted_as_one_in_sensitivity_view(self):
        refs=[segment([0,5],[10,5]),segment([12,5],[2,5]),segment([0,5],[2,5])]
        result=overlap_union(refs)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['points'],[[0.,5.],[12.,5.]])

    def test_merely_touching_pieces_keep_their_split(self):
        result=overlap_union([segment([0,5],[10,5]),segment([10,5],[20,5])])
        self.assertEqual(len(result),2)

    def test_different_source_edges_are_never_merged(self):
        result=overlap_union([segment([0,5],[10,5],'a'),segment([2,5],[12,5],'b')])
        self.assertEqual(len(result),2)

    def test_short_offset_and_corner_are_not_deleted(self):
        refs=[segment([0,5],[10,5]),segment([0,6],[1,6]),segment([1,6],[1,5])]
        self.assertEqual(len(overlap_union(refs)),3)
        self.assertEqual(len(refs),3)


if __name__=='__main__':unittest.main()
