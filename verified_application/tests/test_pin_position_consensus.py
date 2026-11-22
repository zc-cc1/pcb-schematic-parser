"""Position-only invariants, geometry consensus safeguards and entrypoint integration."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT
sys.path.insert(0,str(LIVE))
sys.path.insert(0,str(LIVE/'src'))
sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:
    pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser.pin_position_consensus import correct_pin_positions_by_side


def fixture(count=7):
    pins={f'pin_{i+1}':{'pinname':f'SIGNAL{i}', 'point':{'x':102.,'y':35.+20*i}}
          for i in range(count)}
    raw={'U1':copy.deepcopy(pins)}
    for pin in pins.values():pin['point']['x']=130.
    target={'components':{'U1':{'type':'box','bbox':[20,20,100,180]}},
            'pins':{'U1':pins},'nets':{'N1':{'hyperGraph':'(U1.1,U1.2)','edges':{}}}}
    return target,raw,np.full((200,220,3),255,dtype=np.uint8)


class ConsensusTests(unittest.TestCase):
    def run_lengths(self,target,raw,image,lengths):
        return correct_pin_positions_by_side(target,raw,image,lambda *args:next(lengths))

    def test_short_majority_rejects_long_outliers_and_preserves_fields(self):
        target,raw,image=fixture();before=copy.deepcopy(target);raw_before=copy.deepcopy(raw)
        audit=self.run_lengths(target,raw,image,iter([8,8,10,12,12,30,30]))
        self.assertEqual(audit['modified_pin_count'],7)
        self.assertEqual(audit['groups'][0]['consensus_length'],10)
        self.assertEqual(target['components'],before['components'])
        self.assertEqual(target['nets'],before['nets'])
        self.assertEqual(raw,raw_before)
        for key,pin in target['pins']['U1'].items():
            self.assertEqual(pin['point']['x'],110)
            self.assertEqual(pin['point']['y'],before['pins']['U1'][key]['point']['y'])
            self.assertEqual(pin['pinname'],before['pins']['U1'][key]['pinname'])

    def test_two_rows_cannot_establish_consensus(self):
        target,raw,image=fixture(2);before=copy.deepcopy(target)
        self.run_lengths(target,raw,image,iter([8,8]))
        self.assertEqual(target,before)

    def test_same_row_duplicates_do_not_manufacture_support(self):
        target,raw,image=fixture(3);before=copy.deepcopy(target)
        for pin in raw['U1'].values():pin['point']['y']=50
        self.run_lengths(target,raw,image,iter([8,8,8]))
        self.assertEqual(target,before)

    def test_half_support_is_not_a_majority(self):
        target,raw,image=fixture(6);before=copy.deepcopy(target)
        self.run_lengths(target,raw,image,iter([8,8,8,30,30,30]))
        self.assertEqual(target,before)

    def test_failed_measurements_count_against_majority(self):
        target,raw,image=fixture(7);before=copy.deepcopy(target)
        self.run_lengths(target,raw,image,iter([8,8,8,None,None,None,None]))
        self.assertEqual(target,before)

    def test_opposite_sides_have_independent_lengths(self):
        target,raw,image=fixture(3)
        for i,y in enumerate([35,55,75]):
            key=f'pin_L{i}'
            raw['U1'][key]={'pinname':'L','point':{'x':18.,'y':y}}
            target['pins']['U1'][key]={'pinname':'L','point':{'x':2.,'y':y}}
        self.run_lengths(target,raw,image,iter([8,8,8,30,30,30]))
        self.assertEqual(target['pins']['U1']['pin_1']['point']['x'],108)
        self.assertEqual(target['pins']['U1']['pin_L0']['point']['x'],-10)

    def test_large_correction_is_not_applied(self):
        target,raw,image=fixture(3)
        target['pins']['U1']['pin_1']['point']['x']=190
        audit=self.run_lengths(target,raw,image,iter([8,8,8]))
        self.assertEqual(target['pins']['U1']['pin_1']['point']['x'],190)
        self.assertEqual(audit['modified_pin_count'],2)

    def test_passive_components_are_untouched(self):
        target,raw,image=fixture(3);target['components']['U1']['type']='resistor'
        before=copy.deepcopy(target)
        audit=correct_pin_positions_by_side(target,raw,image,lambda *args:self.fail('Unexpected measurement'))
        self.assertEqual(target,before);self.assertEqual(audit['modified_pin_count'],0)

    def test_actual_image_corrects_legacy_long_snap(self):
        target,raw,image=fixture(3)
        for pin in raw['U1'].values():
            y=int(image.shape[0]-pin['point']['y'])
            image[y,100:109]=[150,0,0];image[y,109:180]=[0,160,0]
        audit=correct_pin_positions_by_side(target,raw,image,agent.measure_lead)
        self.assertEqual(audit['modified_pin_count'],3)
        for pin in target['pins']['U1'].values():
            self.assertGreaterEqual(pin['point']['x'],106)
            self.assertLessEqual(pin['point']['x'],109)

    def test_entrypoint_opt_out_and_fixed_mode(self):
        target,raw,image=fixture(3);target['pins']=copy.deepcopy(raw)
        audit={}
        with patch.dict(os.environ,{'PCB_PIN_SIDE_CONSENSUS':'0'}):
            agent.pin_lead_snap(target,18,image,'color',side_audit=audit)
        self.assertFalse(audit['enabled'])
        self.assertTrue(all(p['point']['x']==118 for p in target['pins']['U1'].values()))
        target['pins']=copy.deepcopy(raw)
        with patch.dict(os.environ,{'PCB_PIN_SIDE_CONSENSUS':'1'}):
            agent.pin_lead_snap(target,18,image,'fixed',side_audit=audit)
        self.assertFalse(audit['enabled'])

    def test_entrypoint_correction_and_per_call_audit(self):
        target,raw,image=fixture(3);target['pins']=copy.deepcopy(raw)
        for pin in raw['U1'].values():
            y=int(image.shape[0]-pin['point']['y'])
            image[y,100:109]=[150,0,0];image[y,109:180]=[0,160,0]
        audit={}
        with patch.dict(os.environ,{'PCB_PIN_SIDE_CONSENSUS':'1'}):
            agent.pin_lead_snap(target,18,image,'color',side_audit=audit)
        self.assertTrue(audit['enabled']);self.assertEqual(audit['modified_pin_count'],3)
        previous=copy.deepcopy(audit)
        other={}
        agent.pin_lead_snap({'components':{},'pins':{}},18,None,side_audit=other)
        self.assertEqual(audit,previous);self.assertFalse(other['enabled'])


if __name__=='__main__':unittest.main()
