import unittest
from copy import deepcopy

import cv2
import numpy as np

from pcb_parser.wire_span_consolidation import consolidate_wire_spans


def edge(a,b):return [{'x':float(a[0]),'y':float(a[1])},{'x':float(b[0]),'y':float(b[1])}]


class WireSpanConsolidationTests(unittest.TestCase):
    def fixture(self,*,gap_color=(0,0,190)):
        target={'components':{},'pins':{},'nets':{
            'main':{'hyperGraph':'(A.1,B.1,C.1)','edges':{
                'edge_1':edge((180,80),(300,80)), 'edge_2':edge((300,80),(300,200))}},
            'fragment':{'hyperGraph':'(D.1)','edges':{
                'edge_1':edge((40,80),(180,80)), 'edge_2':edge((180,80),(230,80))}}}}
        rgb=np.full((240,400,3),255,np.uint8)
        cv2.line(rgb,(40,160),(300,160),(0,160,0),1)
        cv2.line(rgb,(300,160),(300,40),(0,160,0),1)
        rgb[160,210:213]=gap_color
        graph={'image_sha256':'image','size':[400,240],'coordinate_origin':'top_left','prediction_only':True}
        return target,rgb,graph

    def run_rule(self,data,dots=()):
        return consolidate_wire_spans(*data,'image',dots,cfg={'enabled':True})

    def test_overlapping_spans_cross_short_blue_text_without_membership_changes(self):
        data=self.fixture();before=deepcopy(data[0]);out,audit=self.run_rule(data)
        self.assertEqual(data[0],before)
        self.assertEqual(out['nets']['main']['edges']['edge_1'],edge((40,80),(300,80)))
        self.assertEqual(out['nets']['fragment']['edges'],{})
        self.assertEqual({k:v['hyperGraph'] for k,v in out['nets'].items()},
                         {k:v['hyperGraph'] for k,v in before['nets'].items()})
        self.assertEqual(len(audit['changes']),1)
        self.assertEqual(self.run_rule((out,data[1],data[2]))[0],out)

    def test_white_gap_is_not_bridged(self):
        data=self.fixture(gap_color=(255,255,255))
        self.assertEqual(self.run_rule(data)[0],data[0])

    def test_restored_simplified_duplicate_is_removed_on_pipeline_replay(self):
        data=self.fixture();complete=self.run_rule(data)[0]
        replay=deepcopy(complete)
        replay['nets']['fragment']['edges']={'edge_1':edge((40,80),(230,80))}
        self.assertEqual(self.run_rule((replay,data[1],data[2]))[0],complete)

    def test_observed_internal_junction_is_kept(self):
        data=self.fixture()
        self.assertEqual(self.run_rule(data,[[180.,80.]])[0],data[0])

    def test_image_binding_is_required(self):
        data=self.fixture();data[2]['image_sha256']='other'
        with self.assertRaises(ValueError):self.run_rule(data)


if __name__=='__main__':unittest.main()
