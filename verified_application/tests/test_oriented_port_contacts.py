import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.oriented_port_contacts import port_proof,attach_contact,clear_shared_contact_slivers,recover_oriented_port_contacts


class OrientedPortContactsTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((350,450,3),255,np.uint8)
        cv2.polylines(rgb,[np.array([[200,180],[220,168],[270,168],[270,192],[220,192]])],True,(190,0,0),2)
        cv2.line(rgb,(170,180),(200,180),(190,0,0),1)
        cv2.line(rgb,(140,180),(170,180),(0,160,0),1)
        record={'component_id':'SOURCE_24','type':'vbus_in','bbox_top_left':[195,164,275,196],'detection_confidence':.55}
        words=[{'text':'INPUT','bbox':[280,172,330,190],'confidence':.999,'view':v} for v in ['a','b','c']]
        return rgb,record,words

    def test_arrow_contact_follows_the_external_stem(self):
        proof=port_proof(*self.fixture())
        self.assertIsNotNone(proof)
        self.assertEqual(proof['caption'],'INPUT')
        self.assertLess(abs(proof['point'][0]-170),3)
        self.assertLess(abs(proof['point'][1]-180),2)

    def test_rotated_port_is_not_lost(self):
        rgb,record,words=self.fixture();rgb=np.transpose(rgb,(1,0,2)).copy()
        a,b,c,d=record['bbox_top_left'];record['bbox_top_left']=[b,a,d,c]
        for t in words:
            a,b,c,d=t['bbox'];t['bbox']=[b,a,d,c]
        proof=port_proof(rgb,record,words)
        self.assertIsNotNone(proof)
        self.assertLess(abs(proof['point'][0]-180),2)
        self.assertLess(abs(proof['point'][1]-170),3)

    def test_minor_raster_corner_does_not_hide_five_sided_frame(self):
        rgb,record,words=self.fixture();rgb[164:197,195:276]=255
        cv2.polylines(rgb,[np.array([[200,180],[220,168],[269,168],[272,171],[270,192],[220,192]])],True,(190,0,0),2)
        cv2.line(rgb,(170,180),(200,180),(190,0,0),1)
        self.assertIsNotNone(port_proof(rgb,record,words))

    def test_white_gap_is_not_crossed_to_a_remote_green_line(self):
        rgb,record,words=self.fixture();rgb[177:184,180:185]=255
        self.assertIsNone(port_proof(rgb,record,words))

    def test_one_ocr_view_does_not_create_an_identity(self):
        rgb,record,words=self.fixture()
        self.assertIsNone(port_proof(rgb,record,words[:1]))

    def test_rectangle_without_an_arrow_is_rejected(self):
        rgb,record,words=self.fixture();rgb[160:200,195:276]=255
        cv2.rectangle(rgb,(200,168),(270,192),(190,0,0),2)
        self.assertIsNone(port_proof(rgb,record,words))

    def test_local_attachment_does_not_merge_crossing_networks(self):
        rgb=np.full((100,100,3),255,np.uint8)
        cv2.line(rgb,(10,50),(90,50),(0,160,0),1)
        cv2.line(rgb,(50,10),(50,90),(0,160,0),1)
        out={'pins':{'PORT':{'pin_1':{'point':{'x':12,'y':50}}}},'nets':{
            'a':{'hyperGraph':'(A.1,B.1)','edges':{'e':[{'x':10,'y':50},{'x':90,'y':50}]}},
            'b':{'hyperGraph':'(C.1,D.1)','edges':{'e':[{'x':50,'y':10},{'x':50,'y':90}]}}}}
        attach_contact(out,'PORT','1',rgb)
        self.assertEqual(out['nets']['a']['hyperGraph'],'(A.1,B.1,PORT.1)')
        self.assertEqual(out['nets']['b']['hyperGraph'],'(C.1,D.1)')

    def test_contact_gap_does_not_attach_to_nearby_network(self):
        rgb=np.full((100,100,3),255,np.uint8)
        cv2.line(rgb,(20,50),(90,50),(0,160,0),1)
        out={'pins':{'PORT':{'pin_1':{'point':{'x':10,'y':50}}}},'nets':{
            'a':{'hyperGraph':'(A.1,B.1)','edges':{'e':[{'x':20,'y':50},{'x':90,'y':50}]}}}}
        self.assertIsNone(attach_contact(out,'PORT','1',rgb))
        self.assertEqual(out['nets']['a']['hyperGraph'],'(A.1,B.1)')

    def test_coincident_red_symbol_stub_is_not_a_wire(self):
        rgb=np.full((100,100,3),255,np.uint8);cv2.line(rgb,(48,50),(53,50),(190,0,0),2)
        out={'nets':{'a':{'hyperGraph':'(R7.2,LED8.1)','edges':{'e':[{'x':48,'y':50},{'x':53,'y':50}]}}}}
        removed=clear_shared_contact_slivers(out,{'R7.2','LED8.1'},[50,50],rgb)
        self.assertEqual(len(removed),1);self.assertEqual(out['nets']['a']['edges'],{})

    def test_green_wire_extending_beyond_shared_contact_is_preserved(self):
        rgb=np.full((100,100,3),255,np.uint8);cv2.line(rgb,(43,50),(53,50),(0,160,0),2)
        out={'nets':{'a':{'hyperGraph':'(R7.2,LED8.1)','edges':{'e':[{'x':43,'y':50},{'x':53,'y':50}]}}}}
        self.assertEqual(clear_shared_contact_slivers(out,{'R7.2','LED8.1'},[50,50],rgb),[])
        self.assertIn('e',out['nets']['a']['edges'])

    def document_fixture(self):
        rgb,record,words=self.fixture()
        binding={'image_sha256':'fixture','size':[450,350],'coordinate_origin':'top_left','prediction_only':True}
        # Observation parser consumes engine OCR records, not runtime words.
        return rgb,record,binding

    def test_disabled_stage_preserves_input_without_evidence(self):
        rgb,_,_=self.document_fixture();target={'components':{},'pins':{},'nets':{}}
        result,audit=recover_oriented_port_contacts(target,rgb,{},{},{},{},'fixture')
        self.assertEqual(result,target);self.assertIsNot(result,target);self.assertEqual(audit['status'],'disabled')

    def test_cross_image_evidence_is_rejected(self):
        rgb,record,binding=self.document_fixture();wrong={**binding,'image_sha256':'different'}
        with self.assertRaisesRegex(ValueError,'evidence_differs_from_image'):
            recover_oriented_port_contacts({'components':{},'pins':{},'nets':{}},rgb,binding,
                {**binding,'records':[record]},binding,wrong,'fixture',cfg={'enabled':True})

    def test_proven_port_budget_rollback_and_idempotence(self):
        from unittest.mock import patch
        rgb,record,binding=self.document_fixture();proof=port_proof(*self.fixture())
        target={'components':{},'pins':{},'nets':{}};snapshot=deepcopy(target)
        args=(rgb,binding,{**binding,'records':[record]},binding,binding,'fixture')
        with patch('pcb_parser.oriented_port_contacts.port_proof',return_value=proof):
            result,audit=recover_oriented_port_contacts(target,*args,cfg={'enabled':True,'maximum_ports':0})
            self.assertEqual(result,snapshot);self.assertEqual(audit['status'],'withheld_budget')
            result,audit=recover_oriented_port_contacts(target,*args,cfg={'enabled':True})
            self.assertIn('INPUT',result['components']);self.assertEqual(target,snapshot)
            again,_=recover_oriented_port_contacts(result,*args,cfg={'enabled':True})
            self.assertEqual(again,result)


if __name__=='__main__':unittest.main()
