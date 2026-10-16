"""LED geometry guards and coordinate contracts, without model downloads."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pcb_parser.led_geometry import lead_axis, refine_led_boxes


def payload(box, key='LED1'):
    return {'components':{key:{'type':'led','Name':'MODEL1','value':None,'bbox':box},
                          'U1':{'type':'box','Name':'CHIP','value':None,'bbox':[10,10,50,80]}},
            'pins':{'U1':{'pin_1':{'pinname':'VCC','point':{'x':10,'y':40}}}}, 'nets':{}}


def projected(raw):
    result = deepcopy(raw)
    for c in result['components'].values():
        if c['type']=='led':
            b=c['bbox'];x=(b[0]+b[2])/2;y=(b[1]+b[3])/2
            c['bbox']=[x-10,y-7,x+10,y+7]
    return result


def marker_predictor(rgb):
    mask = (rgb[:,:,1]>200)&(rgb[:,:,0]<20)&(rgb[:,:,2]<20)
    ys,xs=np.where(mask)
    if not len(xs):return []
    return [{'class_name':'led','confidence':.95,'bbox':[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]}]


class LEDGeometryTests(unittest.TestCase):
    def test_opposite_leads_determine_axis_for_square_box(self):
        rgb=np.full((180,180,3),255,np.uint8)
        cv2.line(rgb,(90,20),(90,160),(0,0,0),2)
        cv2.polylines(rgb,[np.array([[82,82],[98,82],[90,98]])],True,(0,0,0),2)
        self.assertEqual(lead_axis(rgb,[74,74,106,106])['axis'],'vertical')
        self.assertEqual(lead_axis(np.rot90(rgb).copy(),[74,74,106,106])['axis'],'horizontal')

    def test_crossed_or_missing_leads_are_ambiguous(self):
        rgb=np.full((180,180,3),255,np.uint8)
        self.assertIsNone(lead_axis(rgb,[74,74,106,106])['axis'])
        cv2.line(rgb,(20,90),(160,90),(0,0,0),2)
        cv2.line(rgb,(90,20),(90,160),(0,0,0),2)
        self.assertIsNone(lead_axis(rgb,[74,74,106,106])['axis'])

    def test_orientation_does_not_recenter_on_diode_body(self):
        rgb=np.full((180,180,3),255,np.uint8)
        cv2.line(rgb,(84,20),(84,160),(0,0,0),2)
        raw=payload([74,74,106,106]);before=projected(raw)
        after,audit=refine_led_boxes(rgb,raw,before)
        self.assertEqual(after['components']['LED1']['bbox'],[83,80,97,100])
        self.assertEqual(audit['changed_LED_centers'],0)

    def test_two_contexts_recover_shifted_center_without_touching_fields(self):
        rgb=np.full((800,800,3),255,np.uint8);rgb[330:360,300:330]=[0,255,0]
        raw=payload([324,440,354,470]);before=projected(raw);saved=deepcopy((raw,before))
        after,audit=refine_led_boxes(rgb,raw,before,marker_predictor)
        self.assertEqual(after['components']['LED1']['bbox'],[305,448,325,462])
        self.assertEqual(audit['changed_LED_centers'],1)
        self.assertEqual(after['components']['U1'],before['components']['U1'])
        self.assertEqual(after['pins'],before['pins']);self.assertEqual(after['nets'],before['nets'])
        for field in ['Name','type','value']:
            self.assertEqual(after['components']['LED1'][field],before['components']['LED1'][field])
        self.assertEqual((raw,before),saved)

    def test_disagreeing_views_do_not_move_center(self):
        rgb=np.full((800,800,3),255,np.uint8);rgb[330:360,300:330]=[0,255,0]
        calls=[]
        def predict(crop):
            calls.append(1);rows=marker_predictor(crop)
            if len(calls)==2:rows[0]['bbox'][0]+=12;rows[0]['bbox'][2]+=12
            return rows
        raw=payload([324,440,354,470]);before=projected(raw)
        after,audit=refine_led_boxes(rgb,raw,before,predict)
        self.assertEqual(after,before)
        self.assertEqual(audit['entries'][0]['local_detection']['reason'],'local_views_disagree')

    def test_one_view_or_low_confidence_is_insufficient(self):
        rgb=np.full((800,800,3),255,np.uint8);rgb[330:360,300:330]=[0,255,0]
        def weak(crop):
            rows=marker_predictor(crop)
            for row in rows:row['confidence']=.6
            return rows
        raw=payload([324,440,354,470]);before=projected(raw)
        self.assertEqual(refine_led_boxes(rgb,raw,before,weak)[0],before)

    def test_repeated_small_image_context_is_not_independent(self):
        rgb=np.full((120,120,3),255,np.uint8);rgb[40:60,40:60]=[0,255,0]
        raw=payload([55,60,75,80]);before=projected(raw)
        after,audit=refine_led_boxes(rgb,raw,before,marker_predictor)
        self.assertEqual(after,before)
        self.assertEqual(audit['entries'][0]['local_detection']['reason'],'repeated_context_is_not_independent')

    def test_correction_cannot_cross_nearby_led_ownership(self):
        rgb=np.full((800,800,3),255,np.uint8);rgb[330:360,300:330]=[0,255,0]
        raw=payload([324,440,354,470]);raw['components']['LED2']={**raw['components']['LED1'],'bbox':[300,440,330,470]}
        before=projected(raw);after,audit=refine_led_boxes(rgb,raw,before,marker_predictor)
        self.assertEqual(after,before)
        self.assertEqual(audit['entries'][0]['local_detection']['reason'],'would_cross_neighboring_LED_ownership')

    def test_no_leds_means_no_local_prediction(self):
        rgb=np.full((120,120,3),255,np.uint8);raw=payload([10,10,40,40]);del raw['components']['LED1']
        def forbidden(_):raise AssertionError('unnecessary model call')
        self.assertEqual(refine_led_boxes(rgb,raw,raw,forbidden)[0],raw)

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):lead_axis(np.zeros((120,120,3),np.uint8),[20,20,10,30])


if __name__=='__main__':unittest.main()
