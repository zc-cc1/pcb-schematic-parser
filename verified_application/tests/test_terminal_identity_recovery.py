import unittest
from copy import deepcopy
from types import SimpleNamespace
import cv2
import numpy as np

from pcb_parser.terminal_identity_recovery import (
    bar_profile,capacitor_candidates,ground_proof,isolate_caption,recover_capacitors)


def fixture(ground=False):
    rgb=np.full((110,130,3),255,np.uint8)
    if ground:
        for x,length in [(40,18),(46,28),(52,40)]:
            cv2.line(rgb,(x,50-length//2),(x,50+length//2),(190,0,0),1)
        cv2.line(rgb,(52,50),(72,50),(190,0,0),1)
        cv2.line(rgb,(72,50),(110,50),(0,160,0),1)
        records=[{'component_id':'ground','type':'gnd','detection_confidence':.9,'bbox_top_left':[36,29,66,72]},
                 {'component_id':'false','type':'capacitor','detection_confidence':.7,'designator':None,'Name':None}]
        pins=[{'candidate_id':'terminal','point':[61.,50.],'confidence':.99,'owner':None,'owner_status':'ambiguous_owner',
               'owner_candidates':[{'component_id':'ground'},{'component_id':'false'}]}]
    else:
        for x in [40,48]:cv2.line(rgb,(x,34),(x,66),(190,0,0),1)
        cv2.line(rgb,(20,50),(40,50),(190,0,0),1)
        cv2.line(rgb,(48,50),(72,50),(190,0,0),1)
        cv2.line(rgb,(2,50),(20,50),(0,160,0),1)
        cv2.line(rgb,(72,50),(110,50),(0,160,0),1)
        cv2.putText(rgb,'C8',(31,25),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,180),1)
        records=[{'component_id':'cap','type':'capacitor','detection_confidence':.95,'designator':None,
                  'Name':None,'duplicate_of_detection_index':None,'bbox_top_left':[25,30,63,70],'value':'100nF'}]
        pins=[{'candidate_id':str(i),'point':[x,50.],'number':str(i),'confidence':.99,
               'owner':'cap','owner_status':'assigned'} for i,x in [(2,30.),(1,60.)]]
    r,g,b=[rgb[:,:,i].astype(int) for i in range(3)]
    draft=SimpleNamespace(rgb=rgb,h=110,w=130,red=(r>g+35)&(r>b+35)&(g<150)&(b<150),
        green=(g>r+20)&(g>b+20)&(g>60),records=records,rmap={r['component_id']:r for r in records},
        pin_document={'pin_candidates':pins},out={'components':{},'pins':{}},source_points={},
        words=[{'text':'GND','confidence':.999,'view':str(i),'bbox':[20,35,30,65]} for i in range(2)],
        events=[],withheld=[],ocr_calls=0,cfg={'maximum_caption_views':8})
    draft.box=lambda c:[c['bbox'][0],110-c['bbox'][3],c['bbox'][2],110-c['bbox'][1]]
    draft.export_pin=lambda p,name='':{'pinname':str(name),'point':{'x':float(p[0]),'y':110-float(p[1])}}
    draft.terminal_pair=lambda *args:(0,[[19.,50.],[73.,50.]])
    draft.reader=lambda rgb,box,scale,view:[{'text':'C8','confidence':.999,'view':view+str(scale),'bbox':box}]
    return draft


class TerminalIdentityRecoveryTests(unittest.TestCase):
    def test_two_equal_plates_accept_capacitor(self):
        d=fixture();self.assertIsNotNone(capacitor_candidates(d,d.records[0]))

    def test_missing_text_is_not_inferred_from_value_or_sequence(self):
        d=fixture();d.reader=lambda *a:[]
        self.assertEqual(recover_capacitors(d),set());self.assertEqual(d.out['components'],{})

    def test_recovery_retains_observed_terminal_numbers(self):
        d=fixture();self.assertEqual(recover_capacitors(d),{'C8'})
        self.assertEqual(d.out['pins']['C8']['pin_2']['point']['x'],19.)
        self.assertEqual(d.out['pins']['C8']['pin_1']['point']['x'],73.)

    def test_disagreeing_readings_are_withheld(self):
        d=fixture();d.reader=lambda rgb,box,scale,view:[{'text':'C8' if scale==3 else 'C9','confidence':.999,'view':str(scale)}]
        self.assertFalse(recover_capacitors(d));self.assertEqual(d.out['components'],{})

    def test_repeated_view_is_not_consensus(self):
        d=fixture();d.reader=lambda *a:[{'text':'C8','confidence':.999,'view':'same'}]
        self.assertFalse(recover_capacitors(d))

    def test_low_confidence_text_is_withheld(self):
        d=fixture();d.reader=lambda *a:[{'text':'C8','confidence':.97,'view':str(a[2])}]
        self.assertFalse(recover_capacitors(d))

    def test_shared_budget_is_respected(self):
        d=fixture();d.cfg['maximum_caption_views']=1
        self.assertFalse(recover_capacitors(d));self.assertEqual(d.ocr_calls,1)

    def test_existing_identity_cannot_be_overwritten(self):
        d=fixture();d.out['components']['C8']={'type':'c','bbox':[95,20,115,40]}
        old=deepcopy(d.out);self.assertFalse(recover_capacitors(d));self.assertEqual(d.out,old)

    def test_already_exported_same_symbol_is_not_reidentified(self):
        d=fixture();d.out['components']['C8']={'type':'c','bbox':[25,40,63,80]}
        self.assertIsNone(capacitor_candidates(d,d.records[0]))

    def test_ambiguous_or_weak_capacitor_endpoints_are_rejected(self):
        for change in [{'owner_status':'ambiguous_owner'},{'confidence':.89},{'number':None}]:
            d=fixture();d.pin_document['pin_candidates'][0].update(change)
            self.assertIsNone(capacitor_candidates(d,d.records[0]))

    def test_white_gap_blocks_capacitor_recovery(self):
        d=fixture();d.red[48:53,25:28]=False
        self.assertFalse(recover_capacitors(d));self.assertEqual(d.out['components'],{})

    def test_three_graded_bars_and_text_resolve_ground(self):
        d=fixture(True);self.assertIsNotNone(ground_proof(d,d.records[0],d.pin_document['pin_candidates'][0]))

    def test_ground_cannot_steal_assigned_candidate(self):
        d=fixture(True);p=d.pin_document['pin_candidates'][0];p.update(owner='false',owner_status='assigned')
        self.assertIsNone(ground_proof(d,d.records[0],p))

    def test_named_or_stronger_or_other_kind_opponent_blocks_ground(self):
        for change in [{'designator':'C1'},{'detection_confidence':.89},{'type':'resistor'}]:
            d=fixture(True);d.records[1].update(change)
            self.assertIsNone(ground_proof(d,d.records[0],d.pin_document['pin_candidates'][0]))

    def test_missing_ground_text_or_wire_is_rejected(self):
        for case in ['text','wire']:
            d=fixture(True)
            if case=='text':d.words=d.words[:1]
            else:d.red[48:53,66:70]=False
            self.assertIsNone(ground_proof(d,d.records[0],d.pin_document['pin_candidates'][0]))

    def test_two_plates_are_not_ground_and_three_bars_are_not_capacitor(self):
        d=fixture();self.assertIsNone(bar_profile(d.red,d.records[0]['bbox_top_left'],0,3,[61.,50.]))
        d=fixture(True);self.assertIsNone(bar_profile(d.red,d.records[0]['bbox_top_left'],0,2))

    def test_caption_extraction_rejects_empty_or_outside_region(self):
        d=fixture()
        for box in [[-20,-20,-5,-5],[140,120,160,140],[80,5,95,20]]:
            self.assertIsNone(isolate_caption(d.rgb,box))

    def test_bar_geometry_supports_rotated_symbol(self):
        d=fixture();rot=np.rot90(d.red);h,w=d.red.shape;a,b,c,e=d.records[0]['bbox_top_left']
        self.assertIsNotNone(bar_profile(rot,[b,w-c,e,w-a],1,2))


if __name__=='__main__':unittest.main()
