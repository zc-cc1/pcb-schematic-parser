"""Portable CPU tests; synthetic images only, no paths, OCR, GT or model calls."""
import copy
import unittest
import numpy as np
try:
    from pcb_parser.precise_pin_contacts import refine_pin_contacts,make_binding
except ModuleNotFoundError:
    from precise_pin_contacts import refine_pin_contacts,make_binding
RED=np.array([160,20,20],np.uint8);GREEN=np.array([20,140,20],np.uint8)
def scene(rows=(70,100)):
    rgb=np.full((160,220,3),255,np.uint8)
    rgb[40:121,80:82]=RED;rgb[40:121,160:162]=RED
    rgb[40:42,80:162]=RED;rgb[119:121,80:162]=RED
    pins={}
    for i,y in enumerate(rows,1):
        rgb[y-1:y+2,60:81]=RED;rgb[y-1:y+2,20:60]=GREEN
        pins[f'pin_{i}']={'pinname':f'P{i}','point':{'x':61,'y':160-y}}
    target={'components':{'A':{'type':'box','bbox':[80,40,161,120]}},'pins':{'A':pins},'nets':{'net':{'hyperGraph':'(A.1,A.2)','edges':{}}}}
    return target,rgb
def run(t,rgb,**kwargs):
    return refine_pin_contacts(t,rgb,enabled=True,binding=make_binding(t,rgb),**kwargs)
class PortablePinContacts(unittest.TestCase):
    def test_tangent_offset_is_corrected_by_unique_red_stem(self):
        t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8
        original=copy.deepcopy(t);out,a=run(t,rgb)
        self.assertLess(abs(out['pins']['A']['pin_1']['point']['y']-90),1)
        self.assertEqual(t,original);self.assertEqual(out['components'],t['components']);self.assertEqual(out['nets'],t['nets'])
        self.assertEqual(len(a['changes']),1)
        self.assertEqual(a['changes'][0]['kind'],'red_green_contact')
    def test_red_disc_center_on_bilateral_green_wire(self):
        t,rgb=scene(rows=(85,));rgb[10:150,49:52]=GREEN;rgb[84:87,50:81]=RED
        yy,xx=np.ogrid[:160,:220];rgb[(xx-50)**2+(yy-85)**2<=25]=RED
        t['pins']['A']['pin_1']['point']={'x':44.5,'y':75}
        out,a=run(t,rgb);point=out['pins']['A']['pin_1']['point']
        self.assertLessEqual(abs(point['x']-50),1);self.assertLessEqual(abs(point['y']-75),1)
        self.assertEqual(a['changes'][0]['kind'],'red_disc_green_center')
    def test_three_same_side_contacts_correct_red_continuation(self):
        t,rgb=scene(rows=(55,75,95,110));rgb[109:112,20:60]=255;rgb[109:112,35:81]=RED
        t['pins']['A']['pin_4']['point']['x']=50.5
        out,a=run(t,rgb)
        self.assertLessEqual(abs(out['pins']['A']['pin_4']['point']['x']-60),1)
        self.assertEqual(a['changes'][0]['kind'],'same_side_visible_contact_consensus')
        self.assertEqual(run(out,rgb)[0],out)
    def test_two_contacts_are_insufficient_for_consensus(self):
        t,rgb=scene(rows=(55,75,110));rgb[109:112,20:60]=255;rgb[109:112,35:81]=RED
        t['pins']['A']['pin_3']['point']['x']=50.5
        self.assertEqual(run(t,rgb)[0],t)
    def test_blank_short_stem_overlap_and_crossing_withhold(self):
        for mode in ['blank','short','overlap','cross']:
            with self.subTest(mode=mode):
                t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=6
                if mode=='blank':rgb[:]=255
                elif mode=='short':rgb[65:80,20:80]=255;rgb[69:72,76:81]=RED
                elif mode=='overlap':rgb[74:77,60:81]=RED;rgb[74:77,20:60]=GREEN
                else:rgb[69:72,40:81]=RED;rgb[65:86,58:61]=GREEN;rgb[75:78,50:70]=GREEN
                self.assertEqual(run(t,rgb)[0],t)
    def test_wide_green_crossings_are_not_contact_anchors(self):
        for width in [3,6,10,18,30]:
            with self.subTest(width=width):
                t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8
                rgb[69:72,20:60]=255;rgb[69:72,35:81]=RED
                rgb[60:81,66-width:66]=GREEN
                self.assertEqual(run(t,rgb)[0],t)
    def test_one_white_pixel_gap_does_not_create_contact(self):
        t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8
        rgb[69:72,59:60]=255
        self.assertEqual(run(t,rgb)[0],t)
    def test_white_gap_between_body_and_stem_does_not_create_contact(self):
        t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8
        rgb[69:72,79:80]=255
        self.assertEqual(run(t,rgb)[0],t)
    def test_white_gap_to_body_cannot_borrow_three_contact_consensus(self):
        t,rgb=scene(rows=(55,75,95,110));rgb[109:112,20:60]=255;rgb[109:112,35:81]=RED
        t['pins']['A']['pin_4']['point']['x']=50.5
        rgb[109:112,79:80]=255
        self.assertEqual(run(t,rgb)[0],t)
    def test_white_gap_before_red_disc_withholds_center_correction(self):
        t,rgb=scene(rows=(85,));rgb[10:150,49:52]=GREEN;rgb[84:87,50:81]=RED
        yy,xx=np.ogrid[:160,:220];rgb[(xx-50)**2+(yy-85)**2<=25]=RED
        t['pins']['A']['pin_1']['point']={'x':44.5,'y':75}
        rgb[84:87,70:71]=255
        self.assertEqual(run(t,rgb)[0],t)
    def test_green_crossing_cannot_borrow_three_contact_consensus(self):
        t,rgb=scene(rows=(55,75,95,110));rgb[109:112,20:60]=255;rgb[109:112,35:81]=RED
        rgb[103:118,59:62]=GREEN
        t['pins']['A']['pin_4']['point']['x']=50.5
        self.assertEqual(run(t,rgb)[0],t)
    def test_correct_points_and_small_motion_are_preserved(self):
        t,rgb=scene();self.assertEqual(run(t,rgb)[0],t)
        t['pins']['A']['pin_1']['point']['y']+=3;self.assertEqual(run(t,rgb)[0],t)
    def test_binding_rejects_wrong_image_target_size_origin_and_provenance(self):
        t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8
        good=make_binding(t,rgb)
        for field,value in [('origin','bottom_left'),('prediction_only',False),('size',[1,1]),('image_rgb_sha256','0'*64),('target_sha256','0'*64)]:
            bad={**good,field:value};out,a=refine_pin_contacts(t,rgb,enabled=True,binding=bad)
            self.assertEqual(out,t);self.assertFalse(a['enabled'])
        changed=rgb.copy();changed[0,0,0]^=1
        self.assertEqual(refine_pin_contacts(t,changed,enabled=True,binding=good)[0],t)
    def test_default_disabled_strict_whole_pass_budget_and_boundary(self):
        t,rgb=scene();t['pins']['A']['pin_1']['point']['y']+=8;good=make_binding(t,rgb)
        self.assertEqual(refine_pin_contacts(t,rgb,binding=good)[0],t)
        for budget in [True,False,-1,1.,float('nan'),None,'500',4097,0,1]:
            self.assertEqual(run(t,rgb,max_pins=budget)[0],t,repr(budget))
        self.assertNotEqual(run(t,rgb,max_pins=2)[0],t)
if __name__=='__main__':unittest.main()




