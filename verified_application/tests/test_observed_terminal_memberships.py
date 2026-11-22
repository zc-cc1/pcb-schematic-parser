import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.observed_terminal_memberships import capacitor_axis_proof,local_attachment,recover_terminal_memberships


class ObservedTerminalMembershipTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((240,360,3),255,np.uint8)
        for x in [175,185]:cv2.line(rgb,(x,100),(x,140),(190,0,0),2)
        cv2.line(rgb,(130,120),(175,120),(190,0,0),1)
        cv2.line(rgb,(185,120),(230,120),(190,0,0),1)
        cv2.line(rgb,(100,120),(130,120),(0,150,0),1)
        cv2.line(rgb,(230,120),(260,120),(0,150,0),1)
        record={'component_id':'C19','type':'capacitor','designator':'C19','bbox_top_left':[157,96,204,144],'detection_confidence':.85}
        return rgb,record

    def test_actual_capacitor_plates_override_wrong_square_box_axis(self):
        rgb,record=self.fixture();proof=capacitor_axis_proof(rgb,record)
        self.assertIsNotNone(proof);self.assertEqual(proof['axis'],0)

    def test_rectangle_is_not_capacitor_plate_evidence(self):
        rgb,record=self.fixture();cv2.line(rgb,(175,100),(185,100),(190,0,0),2)
        cv2.line(rgb,(175,140),(185,140),(190,0,0),2)
        self.assertIsNone(capacitor_axis_proof(rgb,record))

    def test_rotated_plate_pair_is_detected(self):
        rgb,record=self.fixture();rgb=rgb.transpose(1,0,2).copy()
        a,b,c,d=record['bbox_top_left'];record['bbox_top_left']=[b,a,d,c]
        self.assertEqual(capacitor_axis_proof(rgb,record)['axis'],1)

    def test_nearby_export_wire_adds_only_missing_contact(self):
        rgb=np.full((100,100,3),255,np.uint8);cv2.line(rgb,(20,50),(90,50),(0,150,0),1)
        target={'components':{},'pins':{'P':{'pin_1':{'point':{'x':25,'y':50}}}},'nets':{
            'a':{'hyperGraph':'(A.1,B.1)','edges':{'e':[{'x':20,'y':50},{'x':90,'y':50}]}}}}
        event=local_attachment(target,'P.1',rgb,maximum=12)
        self.assertIsNotNone(event);self.assertEqual(target['nets']['a']['hyperGraph'],'(A.1,B.1,P.1)')

    def test_white_gap_does_not_bind_nearby_wire(self):
        rgb=np.full((100,100,3),255,np.uint8);cv2.line(rgb,(20,50),(90,50),(0,150,0),1)
        target={'components':{},'pins':{'P':{'pin_1':{'point':{'x':10,'y':50}}}},'nets':{
            'a':{'hyperGraph':'(A.1,B.1)','edges':{'e':[{'x':20,'y':50},{'x':90,'y':50}]}}}}
        self.assertIsNone(local_attachment(target,'P.1',rgb,maximum=12))

    def test_two_wire_candidates_at_undotted_crossing_are_withheld(self):
        rgb=np.full((100,100,3),255,np.uint8)
        for a,b in [((10,50),(90,50)),((50,10),(50,90))]:cv2.line(rgb,a,b,(0,150,0),1)
        target={'components':{},'pins':{'P':{'pin_1':{'point':{'x':50,'y':50}}}},'nets':{
            'a':{'hyperGraph':'(A.1,B.1)','edges':{'e':[{'x':10,'y':50},{'x':90,'y':50}]}},
            'b':{'hyperGraph':'(C.1,D.1)','edges':{'e':[{'x':50,'y':10},{'x':50,'y':90}]}}}}
        old=deepcopy(target);self.assertIsNone(local_attachment(target,'P.1',rgb,maximum=12));self.assertEqual(target,old)

    def test_single_perpendicular_export_at_undotted_crossing_is_withheld(self):
        rgb=np.full((100,100,3),255,np.uint8)
        cv2.line(rgb,(20,50),(90,50),(0,150,0),1);cv2.line(rgb,(30,10),(30,90),(0,150,0),1)
        target={'components':{},'pins':{'P':{'pin_1':{'point':{'x':25,'y':50}}}},'nets':{
            'b':{'hyperGraph':'(C.1,D.1)','edges':{'e':[{'x':30,'y':10},{'x':30,'y':90}]}}}}
        old=deepcopy(target);self.assertIsNone(local_attachment(target,'P.1',rgb,maximum=12));self.assertEqual(target,old)

    def test_perpendicular_export_with_observed_dot_can_bind(self):
        rgb=np.full((100,100,3),255,np.uint8)
        cv2.line(rgb,(20,50),(90,50),(0,150,0),1);cv2.line(rgb,(30,10),(30,90),(0,150,0),1)
        cv2.circle(rgb,(30,50),6,(190,0,0),-1)
        target={'components':{},'pins':{'P':{'pin_1':{'point':{'x':25,'y':50}}}},'nets':{
            'b':{'hyperGraph':'(C.1,D.1)','edges':{'e':[{'x':30,'y':10},{'x':30,'y':90}]}}}}
        self.assertIsNotNone(local_attachment(target,'P.1',rgb,maximum=12))

    def test_evidence_binding_mismatch_fails_before_mutation(self):
        rgb,record=self.fixture();target={'components':{},'pins':{},'nets':{}}
        evidence={'image_sha256':'wrong','size':[360,240],'prediction_only':True,'coordinate_origin':'top_left'}
        with self.assertRaisesRegex(ValueError,'evidence_differs_from_image'):
            recover_terminal_memberships(target,rgb,evidence,evidence,evidence,evidence,'correct',cfg={'enabled':True})

    def recovery_fixture(self):
        rgb,record=self.fixture()
        cv2.line(rgb,(100,70),(100,170),(0,150,0),1)
        cv2.circle(rgb,(100,120),6,(190,0,0),-1)
        target={'components':{'C19':{'type':'c','Name':'C19','value':'10nF','bbox':[157,96,204,144]},
                              'U8':{'type':'box','Name':'U8','value':None,'bbox':[70,150,85,175]}},
                'pins':{'C19':{'pin_1':{'pinname':'1','point':{'x':130,'y':120}},'pin_2':{'pinname':'2','point':{'x':230,'y':120}}},
                        'U8':{'pin_1':{'pinname':'X','point':{'x':100,'y':170}}}},
                'nets':{'bus':{'hyperGraph':'(U8.1)','edges':{'edge_1':[{'x':100,'y':170},{'x':100,'y':70}]}}}}
        binding={'image_sha256':'fixture','size':[360,240],'prediction_only':True,'coordinate_origin':'top_left'}
        return target,rgb,binding,{**binding,'records':[record]},{**binding,'pin_candidates':[]},{**binding,'texts':[]},'fixture'

    def test_full_axial_recovery_is_immutable_idempotent_and_keeps_ic(self):
        args=self.recovery_fixture();before=deepcopy(args[0])
        result,audit=recover_terminal_memberships(*args,cfg={'enabled':True})
        self.assertEqual(args[0],before);self.assertEqual(audit['new_edges'],1)
        self.assertEqual(result['nets']['bus']['hyperGraph'],'(C19.1,U8.1)')
        self.assertEqual(result['pins']['U8'],before['pins']['U8'])
        again,_=recover_terminal_memberships(result,*args[1:],cfg={'enabled':True})
        self.assertEqual(again,result)

    def test_contact_and_new_edge_budgets_rollback_whole_change(self):
        args=self.recovery_fixture()
        for setting in ['maximum_attachments','maximum_new_edges']:
            out,audit=recover_terminal_memberships(*args,cfg={'enabled':True,setting:0})
            self.assertEqual(out,args[0]);self.assertEqual(audit['status'],'withheld_budget')

    def test_axial_white_gap_is_not_filled(self):
        args=self.recovery_fixture();args[1][117:124,111:118]=255
        out,audit=recover_terminal_memberships(*args,cfg={'enabled':True})
        self.assertEqual(out,args[0]);self.assertEqual(audit['new_edges'],0)

    def test_ray_does_not_turn_into_an_undotted_crossing(self):
        args=self.recovery_fixture();rgb=args[1];rgb[113:128,93:108]=255
        cv2.line(rgb,(100,70),(100,170),(0,150,0),1);cv2.line(rgb,(90,120),(130,120),(0,150,0),1)
        out,audit=recover_terminal_memberships(*args,cfg={'enabled':True})
        self.assertEqual(out,args[0]);self.assertEqual(audit['new_edges'],0)


if __name__=='__main__':unittest.main()
