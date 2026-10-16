"""Rendered frame evidence, abstention and final Target invariants."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_parser.ic_geometry import locate_ic_rectangle,refine_ic_boxes


def canvas():return np.full((360,360,3),255,np.uint8)


class ICGeometryTests(unittest.TestCase):
    def test_observed_frame_replaces_expanded_proposal(self):
        rgb=canvas();cv2.rectangle(rgb,(100,80),(260,280),(0,0,0),2)
        actual,audit=locate_ic_rectangle(rgb,[80,60,280,300])
        self.assertIsNotNone(actual)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,[100,80,260,280])),1)
        self.assertEqual(audit['reason'],'four_sided_frame_observed')

    def test_supports_offcenter_prediction_without_forcing_original_center(self):
        rgb=canvas();cv2.rectangle(rgb,(100,80),(260,280),(0,0,0),2)
        actual,_=locate_ic_rectangle(rgb,[70,45,275,290])
        self.assertIsNotNone(actual)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,[100,80,260,280])),1)

    def test_colored_frame_and_external_leads_do_not_expand_frame(self):
        rgb=canvas();cv2.rectangle(rgb,(100,80),(260,280),(150,45,45),2)
        for y in [120,160,200,240]:
            cv2.line(rgb,(30,y),(100,y),(0,140,0),2);cv2.line(rgb,(260,y),(330,y),(0,140,0),2)
        actual,_=locate_ic_rectangle(rgb,[80,60,280,300])
        self.assertIsNotNone(actual)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,[100,80,260,280])),1)

    def test_missing_side_keeps_original(self):
        rgb=canvas();cv2.line(rgb,(100,80),(260,80),(0,0,0),2)
        cv2.line(rgb,(100,80),(100,280),(0,0,0),2);cv2.line(rgb,(260,80),(260,280),(0,0,0),2)
        self.assertIsNone(locate_ic_rectangle(rgb,[80,60,280,300])[0])

    def test_nested_competing_rectangles_are_not_guessed(self):
        rgb=canvas();cv2.rectangle(rgb,(80,80),(260,260),(0,0,0),2);cv2.rectangle(rgb,(100,100),(240,240),(0,0,0),2)
        actual,audit=locate_ic_rectangle(rgb,[65,65,275,275])
        self.assertIsNone(actual);self.assertEqual(audit['reason'],'competing_rectangular_frames')

    def test_two_adjacent_boxes_are_not_enclosed_as_one_frame(self):
        rgb=canvas();cv2.rectangle(rgb,(80,80),(140,240),(0,0,0),1);cv2.rectangle(rgb,(147,80),(207,240),(0,0,0),1)
        self.assertIsNone(locate_ic_rectangle(rgb,[65,65,220,255])[0])

    def test_internal_row_line_does_not_truncate_complete_vertical_sides(self):
        rgb=canvas();cv2.rectangle(rgb,(100,80),(260,280),(0,0,0),2);cv2.line(rgb,(100,130),(260,130),(0,0,0),2)
        actual,_=locate_ic_rectangle(rgb,[80,60,280,300])
        self.assertIsNotNone(actual)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,[100,80,260,280])),1)

    def test_small_box_stays_unchanged_even_if_internal_rectangle_exists(self):
        rgb=canvas();cv2.rectangle(rgb,(105,105),(125,125),(0,0,0),2)
        actual,audit=locate_ic_rectangle(rgb,[100,100,135,135])
        self.assertIsNone(actual);self.assertEqual(audit['reason'],'symbol_too_small_for_four_side_evidence')

    def test_weak_unobserved_edges_are_not_fabricated(self):
        rgb=canvas();cv2.rectangle(rgb,(100,80),(260,280),(220,220,220),2)
        self.assertIsNone(locate_ic_rectangle(rgb,[80,60,280,300])[0])

    def test_bottom_left_conversion_and_non_geometry_fields_preserved(self):
        rgb=canvas();cv2.rectangle(rgb,(100,50),(260,230),(0,0,0),2)
        p={'components':{'U1':{'type':'box','Name':'LM358','value':None,'bbox':[80,110,280,330]},
                         'R1':{'type':'r','Name':'R1','value':'10k','bbox':[10,10,18,30]}},
           'pins':{'U1':{'pin_1':{'pinname':'IN','point':{'x':90,'y':240}}}},'nets':{}}
        saved=deepcopy(p);actual,audit=refine_ic_boxes(rgb,p)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(actual['components']['U1']['bbox'],[100,130,260,310])),1)
        actual['components']['U1']['bbox']=saved['components']['U1']['bbox']
        self.assertEqual(actual,saved);self.assertEqual(p,saved);self.assertEqual(audit['changed_IC_boxes'],1)

    def test_no_ic_leaves_everything_identical(self):
        p={'components':{'R1':{'type':'r','Name':'R1','value':'10k','bbox':[10,10,18,30]}},'pins':{},'nets':{}}
        actual,audit=refine_ic_boxes(canvas(),p)
        self.assertEqual(actual,p);self.assertEqual(audit['entries'],[])

    def test_invalid_proposals_rejected(self):
        for b in [[10,10,5,20],[10,10,float('nan'),20]]:
            with self.assertRaises(ValueError):locate_ic_rectangle(canvas(),b)


if __name__=='__main__':unittest.main()
