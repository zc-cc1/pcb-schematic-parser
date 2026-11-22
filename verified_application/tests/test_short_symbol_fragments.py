"""Guard red symbol cleanup against real green branches and electrical changes."""
import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.wire_false_positive_cleanup import cleanup_wire_false_positives


class ShortSymbolFragmentsTests(unittest.TestCase):
    def fixture(self,kind='c',*,pin_gap=0):
        target={'components':{'X':{'type':kind,'bbox':[80,40,100,60]}},
                'pins':{'X':{'pin_1':{'point':{'x':100.,'y':100.+pin_gap}}}},
                'nets':{'N':{'hyperGraph':'(X.1)','edges':{'edge_1':[
                    {'x':100.,'y':100.},{'x':100.,'y':110.}]}}}}
        rgb=np.full((200,200,3),255,np.uint8)
        cv2.line(rgb,(100,100),(100,90),(190,0,0),2)
        rgb[90:92,100]=(0,160,0)  # tiny pin marker, not a green conductor
        graph={'image_sha256':'same','size':[200,200],'coordinate_origin':'top_left','prediction_only':True}
        return target,rgb,graph

    def run_rule(self,data,dots=(),**cfg):
        return cleanup_wire_false_positives(*data,'same',dots,cfg={'enabled':True,**cfg})

    def test_red_short_symbol_leads_with_tiny_markers_are_removed(self):
        for kind in ['c','l','r','box','ic','gnd']:
            with self.subTest(kind=kind):
                data=self.fixture(kind,pin_gap=-6);before=deepcopy(data[0])
                out,audit=self.run_rule(data)
                self.assertEqual(out['nets']['N']['edges'],{})
                self.assertEqual(data[0],before)
                self.assertEqual(out['components'],before['components'])
                self.assertEqual(out['pins'],before['pins'])
                self.assertEqual(out['nets']['N']['hyperGraph'],'(X.1)')
                self.assertEqual(audit['changes'][0]['rule'],'short_red_symbol_fragment')
                self.assertEqual(self.run_rule((out,data[1],data[2]))[0],out)

    def test_green_continuation_at_endpoint_is_kept(self):
        data=self.fixture()
        cv2.line(data[1],(100,90),(100,65),(0,160,0),1)
        self.assertEqual(self.run_rule(data)[0],data[0])

    def test_real_green_short_wire_is_kept(self):
        data=self.fixture();cv2.line(data[1],(100,100),(100,90),(0,160,0),2)
        self.assertEqual(self.run_rule(data)[0],data[0])

    def test_observed_junction_is_kept(self):
        data=self.fixture()
        self.assertEqual(self.run_rule(data,[[100.,110.]])[0],data[0])

    def test_no_member_pin_evidence_keeps_red_geometry(self):
        data=self.fixture();data[0]['pins']={}
        self.assertEqual(self.run_rule(data)[0],data[0])

    def test_long_red_conductor_is_kept(self):
        data=self.fixture();data[0]['nets']['N']['edges']['edge_1'][1]['y']=140.
        cv2.line(data[1],(100,100),(100,60),(190,0,0),2)
        self.assertEqual(self.run_rule(data)[0],data[0])

    def test_budget_reverts_symbol_deletion(self):
        data=self.fixture();out,audit=self.run_rule(data,maximum_changes=0)
        self.assertEqual(out,data[0]);self.assertEqual(audit['status'],'withheld_budget')


if __name__=='__main__':unittest.main()
