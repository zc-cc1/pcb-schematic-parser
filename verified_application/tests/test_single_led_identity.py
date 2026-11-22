import unittest
from types import SimpleNamespace
import cv2
import numpy as np

from pcb_parser.single_led_identity import single_led_proof


class SingleLedIdentityTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((150,240,3),255,np.uint8)
        cv2.fillConvexPoly(rgb,np.array([[90,60],[74,90],[106,90]]),(190,0,0))
        cv2.line(rgb,(90,45),(90,60),(190,0,0),2)
        cv2.line(rgb,(90,90),(90,110),(190,0,0),2)
        cv2.line(rgb,(90,110),(90,135),(0,160,0),2)
        cv2.rectangle(rgb,(50,61),(58,69),(190,0,0),-1)
        cv2.rectangle(rgb,(62,49),(70,57),(190,0,0),-1)
        cv2.putText(rgb,'LED1',(115,79),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,190),1)
        channels=rgb.astype(np.int16);r,g,b=[channels[:,:,i] for i in range(3)]
        draft=SimpleNamespace(rgb=rgb,h=150,w=240,
            red=(r>g+35)&(r>b+35)&(g<150)&(b<150),
            green=(g>r+20)&(g>b+20)&(g>60),
            pin_document={'pin_candidates':[
                {'candidate_id':'top','owner':'source','owner_status':'assigned','number':'2','point':[90.,48.],'confidence':.7},
                {'candidate_id':'bottom','owner':'source','owner_status':'assigned','number':'1','point':[90.,102.],'confidence':.96}]})
        record={'component_id':'source','type':'led','detection_confidence':.9,
                'bbox_top_left':[43,38,111,116]}
        return draft,record

    def test_model_triangle_emission_marks_and_caption_prove_candidate(self):
        draft,record=self.fixture()
        proof=single_led_proof(draft,record)
        self.assertIsNotNone(proof)
        self.assertEqual(proof['source_candidates'],['top','bottom'])
        self.assertEqual(proof['numbering'],['2','1'])
        self.assertLess(proof['body_box'][0],74)

    def test_without_emission_marks_is_withheld(self):
        draft,record=self.fixture();draft.red[45:72,45:72]=False
        self.assertIsNone(single_led_proof(draft,record))


if __name__=='__main__':unittest.main()
