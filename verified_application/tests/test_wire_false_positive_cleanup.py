import unittest
from copy import deepcopy

import cv2
import numpy as np

from pcb_parser.wire_false_positive_cleanup import cleanup_wire_false_positives


def edge(a,b):
    return [{'x':float(a[0]),'y':float(a[1])},{'x':float(b[0]),'y':float(b[1])}]


def fixture(edges, *, components=None, pins=None, members='(A.1)'):
    target={'components':components or {},'pins':pins or {},
            'nets':{'N':{'hyperGraph':members,
                         'edges':{'edge_'+str(i):e for i,e in enumerate(edges,1)}}}}
    rgb=np.full((400,400,3),255,np.uint8)
    graph={'image_sha256':'image','size':[400,400],
           'coordinate_origin':'top_left','prediction_only':True}
    return target,rgb,graph


def run(data,dots=(),**cfg):
    return cleanup_wire_false_positives(*data,'image',dots,cfg={'enabled':True,**cfg})


class WireFalsePositiveCleanupTests(unittest.TestCase):
    def test_long_unsupported_diagonal_and_covered_edge_are_removed(self):
        data=fixture([edge((30,300),(360,270)),edge((30,100),(230,100)),
                      edge((60,100),(180,100))])
        original=deepcopy(data[0])
        out,audit=run(data)
        self.assertEqual(data[0],original)
        self.assertEqual(len(out['nets']['N']['edges']),1)
        self.assertEqual(out['nets']['N']['hyperGraph'],original['nets']['N']['hyperGraph'])
        self.assertEqual({event['rule'] for event in audit['changes']},
                         {'unsupported_long_diagonal','within_net_covered'})
        self.assertEqual(run((out,data[1],data[2]))[0],out)

    def test_red_resistor_lead_is_kept_at_observed_junction(self):
        components={'R1':{'type':'r','bbox':[190,40,210,50]}}
        pins={'R1':{'pin_1':{'point':{'x':200.,'y':60.}}}}
        data=fixture([edge((200,60),(200,80))],components=components,pins=pins,members='(R1.1)')
        cv2.line(data[1],(200,340),(200,320),(190,0,0),2)
        self.assertEqual(len(run(data)[0]['nets']['N']['edges']),0)
        self.assertEqual(len(run(data,[[200.,80.]])[0]['nets']['N']['edges']),0)
        short=fixture([edge((200,60),(200,70))],components=components,pins=pins,members='(R1.1)')
        cv2.line(short[1],(200,340),(200,330),(190,0,0),2)
        self.assertEqual(run(short,[[200.,70.]])[0],short[0])

    def test_long_wire_extends_only_along_ink_to_unique_observed_dot(self):
        data=fixture([edge((100,20),(100,250))])
        cv2.line(data[1],(100,150),(100,122),(0,160,0),2)
        cv2.circle(data[1],(100,122),4,(190,0,0),-1)
        out,audit=run(data,[[100.,278.]])
        self.assertEqual(out['nets']['N']['edges']['edge_1'][1],{'x':100.,'y':278.})
        self.assertEqual(audit['changes'][0]['rule'],'long_wire_endpoint_to_observed_junction')
        self.assertEqual(run((out,data[1],data[2]),[[100.,278.]])[0],out)

    def test_tiny_segment_and_budget(self):
        data=fixture([edge((20,20),(23,20))])
        out,audit=run(data)
        self.assertEqual(len(out['nets']['N']['edges']),0)
        self.assertEqual(audit['changes'][0]['rule'],'tiny_fragment')
        held,report=run(data,maximum_changes=0)
        self.assertEqual(held,data[0])
        self.assertEqual(report['status'],'withheld_budget')

    def test_stale_image_evidence_is_rejected(self):
        data=fixture([edge((20,20),(23,20))])
        data[2]['image_sha256']='different'
        with self.assertRaises(ValueError):run(data)

    def test_box_pin_overrun_trims_only_observed_red_symbol_lead(self):
        components={'U':{'type':'box','bbox':[120,40,170,100]}}
        pins={'U':{'pin_1':{'point':{'x':100.,'y':70.}}}}
        data=fixture([edge((30,70),(120,70))],components=components,pins=pins,members='(U.1)')
        cv2.line(data[1],(30,330),(100,330),(0,160,0),2)
        cv2.line(data[1],(100,330),(120,330),(190,0,0),2)
        out,audit=run(data)
        self.assertEqual(out['nets']['N']['edges']['edge_1'][1],{'x':100.,'y':70.})
        self.assertEqual(audit['changes'][-1]['rule'],'box_pin_red_lead_trim')
        self.assertEqual(run((out,data[1],data[2]))[0],out)
        green=data[1].copy()
        cv2.line(green,(30,330),(120,330),(0,160,0),2)
        self.assertEqual(run((data[0],green,data[2]))[0],data[0])

    def test_capacitor_pin_overrun_stops_at_measured_color_boundary(self):
        components={'C':{'type':'c','bbox':[120,40,150,100]}}
        pins={'C':{'pin_1':{'point':{'x':100.,'y':70.}}}}
        data=fixture([edge((30,70),(120,70))],components=components,pins=pins,members='(C.1)')
        cv2.line(data[1],(30,330),(100,330),(0,160,0),2)
        cv2.line(data[1],(100,330),(120,330),(190,0,0),2)
        out,_=run(data)
        self.assertEqual(out['nets']['N']['edges']['edge_1'][1],{'x':100.,'y':70.})


if __name__=='__main__':unittest.main()
