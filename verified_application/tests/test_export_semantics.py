import copy
import unittest
import cv2
import numpy as np
from pcb_parser.export_semantics import normalize_export_semantics

SHA='a'*64

def documents(size=(200,200),texts=None,records=None):
    meta=dict(image_sha256=SHA,size=list(size),coordinate_origin='top_left',prediction_only=True)
    return dict(meta,records=records or []),dict(meta,texts=texts or [])

def word(text,box,views=('a','b')):
    return [dict(text=text,bbox=box,confidence=.999,view=v) for v in views]

def part(kind='switch',name='SW7',key='SW7'):
    return {'components':{key:dict(type=kind,Name=name,value=None,bbox=[50,90,90,150])},'pins':{},'nets':{}}

def normalize(target,rgb=None,texts=None,records=None,profile='observed'):
    if rgb is None:rgb=np.full((200,200,3),255,np.uint8)
    rec,ocr=documents((rgb.shape[1],rgb.shape[0]),texts,records)
    return normalize_export_semantics(target,rgb,rec,ocr,SHA,profile)

class ExportSemanticsTests(unittest.TestCase):
    def test_flag_direction_follows_terminal_for_rotations_and_mirrors(self):
        # Tip at x=60; terminal on tip side means input, on flat side means output.
        for terminal,want in [((45,90),'net_input'),((145,90),'net_output')]:
            for rotation in range(4):
                for mirror in (False,True):
                    rgb=np.full((200,200,3),255,np.uint8)
                    cv2.polylines(rgb,[np.array([[60,90],[78,78],[130,78],[130,102],[78,102]])],True,(130,5,5),2)
                    point=np.array(terminal);box=np.array([[55,73],[135,107]])
                    if mirror:
                        rgb=np.fliplr(rgb).copy();point[0]=199-point[0];box[:,0]=199-box[:,0]
                    for _ in range(rotation):
                        rgb=np.rot90(rgb).copy();point=np.array([point[1],199-point[0]]);box=np.stack([box[:,1],199-box[:,0]],axis=1)
                    lo=box.min(0);hi=box.max(0)
                    target={'components':{'SIGNAL':dict(type='net_bidirection',Name='SIGNAL',value='SIGNAL',bbox=[int(lo[0]),int(200-hi[1]),int(hi[0]),int(200-lo[1])])},'pins':{'SIGNAL':{'pin_1':dict(pinname='SIG',point=dict(x=int(point[0]),y=int(200-point[1])))}},'nets':{'n':{'hyperGraph':'(SIGNAL.1)','edges':{}}}}
                    before=copy.deepcopy(target);out,_=normalize(target,rgb)
                    with self.subTest(terminal=terminal,rotation=rotation,mirror=mirror):
                        self.assertEqual(out['components']['SIGNAL']['type'],want)
                        self.assertIsNone(out['components']['SIGNAL']['value'])
                        self.assertEqual(out['pins'],before['pins']);self.assertEqual(out['nets'],before['nets']);self.assertEqual(target,before)
                        self.assertEqual(out['components']['SIGNAL']['bbox'],before['components']['SIGNAL']['bbox'])

    def test_unknown_port_without_geometry_is_preserved(self):
        target=part('net_bidirection','SIGNAL','SIGNAL');target['components']['SIGNAL']['value']='SIGNAL'
        self.assertEqual(normalize(target)[0],target)

    def test_two_terminals_are_not_forced_to_input(self):
        rgb=np.full((200,200,3),255,np.uint8);cv2.polylines(rgb,[np.array([[60,90],[78,78],[130,78],[130,102],[78,102]])],True,(180,0,0),2)
        target=part('net_bidirection','BUS','BUS');target['components']['BUS']['bbox']=[55,90,135,130]
        target['pins']['BUS']={f'pin_{i}':dict(pinname='',point=dict(x=x,y=110)) for i,x in enumerate([45,145])}
        self.assertEqual(normalize(target,rgb)[0],target)

    def test_only_model_bearing_classes_clear_self_designator(self):
        for kind in ['led','switch','box','ic','connector']:
            out,_=normalize(part(kind,'X7','X7'));self.assertIsNone(out['components']['X7']['Name'])
        for kind in ['r','c','l','pin','net_bidirection']:
            target=part(kind,'X7','X7');self.assertEqual(normalize(target)[0],target)

    def test_observed_model_and_description_are_preserved(self):
        for name in ['PART987Z','USB 5V input',None]:
            target=part('box',name);self.assertEqual(normalize(target)[0],target)

    def test_switch_model_uses_local_multiview_caption_beside_reference(self):
        target=part();texts=word('SW7',[96,51,123,64])+word('ABC987Z',[96,70,156,84])
        out,audit=normalize(target,texts=texts)
        self.assertEqual(out['components']['SW7']['Name'],'ABC987Z');self.assertTrue(audit['changes'])

    def test_ambiguous_models_do_not_guess(self):
        texts=word('SW7',[96,51,123,64])+word('ABC987Z',[96,70,156,84])+word('XYZ654K',[96,87,156,101])
        self.assertIsNone(normalize(part(),texts=texts)[0]['components']['SW7']['Name'])

    def test_single_view_model_is_not_adopted(self):
        texts=word('SW7',[96,51,123,64])+word('ABC987Z',[96,70,156,84],('a',))
        self.assertIsNone(normalize(part(),texts=texts)[0]['components']['SW7']['Name'])

    def test_caption_claimed_by_other_component_is_not_stolen(self):
        texts=word('SW7',[96,51,123,64])+word('ABC987Z',[96,70,156,84])
        records=[dict(component_id='U8',fields={'Name':dict(text='ABC987Z',bbox=[96,70,156,84])},bbox_top_left=[160,50,190,110])]
        self.assertIsNone(normalize(part(),texts=texts,records=records)[0]['components']['SW7']['Name'])

    def test_legacy_profile_is_identical(self):
        target=part();self.assertEqual(normalize(target,profile='legacy')[0],target)

    def test_mismatched_evidence_is_rejected_without_mutation(self):
        target=part();before=copy.deepcopy(target);rec,ocr=documents();ocr['image_sha256']='b'*64
        with self.assertRaises(ValueError):normalize_export_semantics(target,np.zeros((200,200,3),np.uint8),rec,ocr,SHA)
        self.assertEqual(target,before)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):normalize(part(),profile='typo')

    def test_model_below_another_component_reference_is_not_stolen(self):
        target=part();target['components']['U8']=dict(type='ic',Name=None,value=None,bbox=[160,90,190,150])
        texts=word('SW7',[96,26,123,39])+word('U8',[96,45,120,58])+word('ABC987Z',[96,65,156,79])
        out,_=normalize(target,texts=texts)
        self.assertIsNone(out['components']['SW7']['Name'])
        self.assertIsNone(out['components']['U8']['Name'])

    def test_nonfinite_ocr_confidence_cannot_support_a_model(self):
        for confidence in [float('nan'),float('inf'),-float('inf')]:
            texts=word('SW7',[96,51,123,64])+word('ABC987Z',[96,70,156,84])
            for t in texts:
                if t['text']=='ABC987Z':t['confidence']=confidence
            self.assertIsNone(normalize(part(),texts=texts)[0]['components']['SW7']['Name'])
