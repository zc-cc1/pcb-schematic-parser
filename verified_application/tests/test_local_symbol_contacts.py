import unittest
import cv2
import numpy as np
from pcb_parser.local_symbol_contacts import parallel_led_proof, read_contact_numbers, _route, recover_local_symbol_contacts
from copy import deepcopy


class LocalSymbolContactsTests(unittest.TestCase):
    def led_fixture(self):
        rgb=np.full((350,450,3),255,np.uint8)
        for y in [180,210]:
            cv2.line(rgb,(160,y),(290,y),(190,0,0),1)
            cv2.line(rgb,(220,y-14),(220,y+14),(190,0,0),2)
            cv2.polylines(rgb,[np.array([[220,y],[240,y-12],[240,y+12]])],True,(190,0,0),2)
            rgb[y,160]=(0,160,0);rgb[y,290]=(0,160,0)
        record={'component_id':'LED31','type':'led','designator':'LED31','Name':'lamp',
                'detection_confidence':.95,'bbox_top_left':[200,155,245,230]}
        words=[{'text':name,'confidence':.999,'bbox':box,'view':view}
               for name,box in [('LED31',[248,140,284,153]),('LED74',[202,230,238,243])]
               for view in ['a','b']]
        return rgb,record,words

    def test_parallel_horizontal_leds_use_observed_names_and_contacts(self):
        proof=parallel_led_proof(*self.led_fixture())
        self.assertIsNotNone(proof)
        self.assertEqual([s['owner'] for s in proof['symbols']],['LED31','LED74'])
        self.assertEqual(proof['symbols'][1]['contacts'],[[160.,210.],[290.,210.]])

    def test_missing_second_caption_is_not_filled_by_sequence(self):
        rgb,record,words=self.led_fixture();words=[w for w in words if w['text']=='LED31']
        self.assertIsNone(parallel_led_proof(rgb,record,words))

    def test_white_gap_in_external_stem_rejects_split(self):
        rgb,record,words=self.led_fixture();rgb[175:186,180:185]=255
        self.assertIsNone(parallel_led_proof(rgb,record,words))

    def test_reverse_facing_cathode_is_not_assigned_left_pin_two(self):
        rgb,record,words=self.led_fixture();rgb=cv2.flip(rgb,1)
        record['bbox_top_left']=[204,155,249,230]
        for t in words:t['bbox'][0],t['bbox'][2]=449-t['bbox'][2],449-t['bbox'][0]
        self.assertIsNone(parallel_led_proof(rgb,record,words))

    def invoke(self,*,budget=4,stale=False,collision=False):
        rgb,record,words=self.led_fixture();target={'components':{'LED31':{'type':'led','Name':'lamp','value':None,'bbox':[200,120,245,195]}},'pins':{'LED31':{}},'nets':{}}
        if collision:
            target['components']['LED74']={'type':'box','bbox':[300,10,340,50]};target['pins']['LED74']={}
        common={'image_sha256':'wrong' if stale else 'same','size':[450,350],'coordinate_origin':'top_left','prediction_only':True}
        graph={**common,'nodes':[],'edges':[],'networks':[]};records={**common,'records':[record]}
        pins={**common,'pin_candidates':[]};ocr={**common,'texts':words}
        result=recover_local_symbol_contacts(target,rgb,graph,records,pins,ocr,'same',cfg={'enabled':True,'maximum_groups':budget})
        return target,result

    def test_whole_symbol_budget_rolls_back(self):
        target,(out,audit)=self.invoke(budget=0)
        self.assertEqual(out,target);self.assertEqual(audit['status'],'withheld_budget')

    def test_stale_source_image_is_rejected(self):
        with self.assertRaises(ValueError):self.invoke(stale=True)

    def test_existing_ic_identity_is_protected_from_collision(self):
        target,(out,audit)=self.invoke(collision=True)
        self.assertEqual(out,target)

    def test_contact_numbers_follow_ocr_not_spatial_order(self):
        shape={'contacts':[[100.,100.],[140.,100.],[100.,200.],[140.,200.]]}
        def reader(rgb,box,scale,view,rotation=0):
            text={82:'3',122:'1'}[box[0]] if box[1]<150 else {82:'4',122:'2'}[box[0]]
            return [{'text':text,'confidence':.999,'view':str((box,scale,rotation))}]
        result=read_contact_numbers(np.full((250,200,3),255,np.uint8),shape,reader,16)
        self.assertEqual(result['numbers'],['3','1','4','2'])

    def test_conflicting_contact_ocr_withholds_whole_switch(self):
        shape={'contacts':[[100.,100.],[140.,100.],[100.,200.],[140.,200.]]}
        def reader(rgb,box,scale,view,rotation=0):
            return [{'text':'1' if scale==3 else '2','confidence':.999,'view':str(scale)}]
        self.assertIsNone(read_contact_numbers(np.full((250,200,3),255,np.uint8),shape,reader,16))

    def test_external_green_wire_survives_symbol_frame_margin(self):
        rgb=np.full((200,200,3),255,np.uint8);cv2.line(rgb,(50,113),(150,113),(0,160,0),1)
        target={'components':{'X':{'type':'led','bbox':[90,90,110,110]},'U':{'type':'box','bbox':[155,60,180,110]}},
                'pins':{'X':{'pin_1':{'point':{'x':50.,'y':87.}}},'U':{'pin_1':{'point':{'x':150.,'y':87.}}}},
                'nets':{'N':{'hyperGraph':'(X.1)','edges':{}}}}
        out=deepcopy(target);_route(out,target,rgb,{'X'},[])
        self.assertEqual(out['nets']['N']['hyperGraph'],'(U.1,X.1)')

    def test_white_gap_does_not_join_unconnected_contacts(self):
        rgb=np.full((200,200,3),255,np.uint8)
        cv2.line(rgb,(50,113),(95,113),(0,160,0),1);cv2.line(rgb,(105,113),(150,113),(0,160,0),1)
        target={'components':{'X':{'type':'led','bbox':[10,10,20,20]},'U':{'type':'box','bbox':[155,60,180,110]}},
                'pins':{'X':{'pin_1':{'point':{'x':50.,'y':87.}}},'U':{'pin_1':{'point':{'x':150.,'y':87.}}}},
                'nets':{'N':{'hyperGraph':'(X.1)','edges':{}}}}
        out=deepcopy(target);_route(out,target,rgb,{'X'},[])
        self.assertEqual(out,target)


if __name__=='__main__':unittest.main()
