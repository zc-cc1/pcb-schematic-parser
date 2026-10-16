import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.pin_ownership import normalized_components,associate_pins
sys.path.insert(0,str(ROOT/'scripts'))
from run_pin_ownership import validate_cache
CFG=json.loads((ROOT/'configs/pin_ownership_v1.json').read_text(encoding='utf-8'))


class OwnershipTests(unittest.TestCase):
    def component(self,box,kind='resistor'):
        return {'bbox':box,'class_name':kind,'confidence':.95}

    def pin(self,xy):
        return {'point':xy,'confidence':.9}

    def run_case(self,pins,detections,wire=None):
        components=normalized_components(detections,{})
        return associate_pins(pins,components,CFG,[200,200],wire)

    def test_clear_owner_does_not_move_tip_or_invent_number(self):
        pins=[self.pin([30.25,50.5])]
        original=copy.deepcopy(pins)
        result=self.run_case(pins,[self.component([30,40,70,60])])
        found=result['pin_candidates'][0]
        self.assertEqual(found['owner'],'DET_0001')
        self.assertEqual(found['point'],[30.25,50.5])
        self.assertIsNone(found['number'])
        self.assertEqual(pins,original)

    def test_two_equally_plausible_owners_remain_ambiguous(self):
        result=self.run_case([self.pin([50,50])],[self.component([20,40,50,60]),self.component([50,40,80,60])])
        self.assertEqual(result['pin_candidates'][0]['owner_status'],'ambiguous_owner')
        self.assertIsNone(result['pin_candidates'][0]['owner'])

    def test_background_and_ic_corner_are_retained_for_review(self):
        result=self.run_case([self.pin([150,150]),self.pin([20,20])],[self.component([20,20,100,100],'ic')])
        self.assertEqual(result['summary']['candidate_count'],2)
        self.assertEqual(result['pin_candidates'][0]['owner_status'],'no_component_nearby')
        self.assertEqual(result['pin_candidates'][1]['owner_status'],'large_component_corner_needs_review')

    def test_low_wire_probability_does_not_remove_unconnected_pin(self):
        result=self.run_case([self.pin([30,50])],[self.component([30,40,70,60])],np.zeros((200,200),np.float32))
        self.assertEqual(result['pin_candidates'][0]['owner_status'],'assigned')
        self.assertEqual(result['pin_candidates'][0]['wire_context_status'],'no_confident_wire_observed')

    def test_wrong_ocr_record_box_rejected(self):
        with self.assertRaises(ValueError):
            normalized_components([self.component([30,40,70,60])],{},[{'detection_index':0,'component_id':'R1','bbox_top_left':[0,0,10,10]}])

    def test_ocr_identity_attaches_without_number_guess(self):
        components=normalized_components([self.component([30,40,70,60])],{},
                   [{'detection_index':0,'component_id':'R7','designator':'R7','bbox_top_left':[30,40,70,60]}])
        result=associate_pins([self.pin([30,50])],components,CFG,[200,200])
        self.assertEqual(result['pin_candidates'][0]['owner'],'R7')
        self.assertIsNone(result['pin_candidates'][0]['number'])

    def test_prediction_cache_requires_matching_image_and_origin(self):
        payload={'image_sha256':'same','size':[200,200],'coordinate_origin':'top_left'}
        validate_cache(payload,'same',[200,200])
        with self.assertRaises(ValueError):
            validate_cache(payload,'different',[200,200])
        with self.assertRaises(ValueError):
            validate_cache({**payload,'coordinate_origin':'bottom_left'},'same',[200,200])

    def test_map_shape_and_out_of_bounds_points_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case([self.pin([30,50])],[self.component([30,40,70,60])],np.zeros((20,20)))
        with self.assertRaises(ValueError):
            self.run_case([self.pin([200,50])],[self.component([30,40,70,60])])


if __name__=='__main__':
    unittest.main()
