import sys
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from pcb_parser.component_caption_fields import model_name_proof,caption_patch,literal_caption_patch,recover_component_caption_fields


class ComponentCaptionFieldsTests(unittest.TestCase):
 def fixture(self):
  out={'components':{'U17':{'type':'box','Name':'U17','value':None,'bbox':[100,100,180,180]}},'pins':{'U17':{}},'nets':{}}
  rec={'component_id':'U17','type':'ic','designator':'U17','detection_confidence':.95,'bbox_top_left':[95,115,185,205]}
  words=[{'text':text,'confidence':.999,'bbox':box,'view':v} for text,box in [('U17',[100,100,126,116]),('AX23-5.0',[100,201,180,218])]
         for v in ['a','b','c']]
  return out,rec,words

 def test_external_model_caption_is_attached_to_designator(self):
  out,rec,words=self.fixture();proof=model_name_proof(out,rec,words,300,[rec])
  self.assertEqual(proof['text'],'AX23-5.0')

 def test_pin_label_inside_body_is_not_a_model(self):
  out,rec,words=self.fixture();words=[t for t in words if t['text']=='U17']
  words +=[{'text':'GPIO12','confidence':.999,'bbox':[120,140,175,156],'view':v} for v in ['a','b','c']]
  self.assertIsNone(model_name_proof(out,rec,words,300,[rec]))

 def test_truncated_prefix_can_use_independently_observed_suffix(self):
  out,rec,words=self.fixture();out['components']['U17']['Name']='AX23-5.'
  self.assertEqual(model_name_proof(out,rec,words,300,[rec])['text'],'AX23-5.0')

 def test_conflicting_models_are_not_resolved_by_guessing(self):
  out,rec,words=self.fixture();words +=[{'text':'BX23-5.0','confidence':.999,'bbox':[100,201,180,218],'view':v} for v in ['d','e','f']]
  self.assertIsNone(model_name_proof(out,rec,words,300,[rec]))

 def test_one_view_is_insufficient(self):
  out,rec,words=self.fixture();self.assertIsNone(model_name_proof(out,rec,[t for t in words if t['view']=='a'],300,[rec]))

 def test_competing_ic_without_designator_ocr_still_blocks_model_assignment(self):
  out,rec,words=self.fixture();other={**rec,'component_id':'U99'}
  out['components']['U99']=deepcopy(out['components']['U17'])
  self.assertIsNone(model_name_proof(out,rec,words,300,[rec,other]))

 def test_evidence_binding_and_input_immutability(self):
  out,rec,words=self.fixture();rgb=np.full((300,400,3),255,np.uint8)
  common={'image_sha256':'same','size':[400,300],'prediction_only':True,'coordinate_origin':'top_left'}
  args=(rgb,common,{**common,'records':[rec]},{**common,'pin_candidates':[]},{**common,'texts':words},'same')
  before=deepcopy(out);result,audit=recover_component_caption_fields(out,*args,cfg={'enabled':True})
  self.assertEqual(out,before);self.assertEqual(result['components']['U17']['Name'],'AX23-5.0')
  self.assertEqual(result['pins'],before['pins']);self.assertEqual(result['nets'],before['nets'])
  again,_=recover_component_caption_fields(result,*args,cfg={'enabled':True});self.assertEqual(again,result)
  rolled,audit=recover_component_caption_fields(out,*args,cfg={'enabled':True,'maximum_changes':0});self.assertEqual(rolled,out)
  with self.assertRaisesRegex(ValueError,'evidence_differs_from_image'):
   recover_component_caption_fields(out,*args[:-1],'other',cfg={'enabled':True})

 def test_caption_patch_supports_long_blue_models(self):
  rgb=np.full((50,260,3),255,np.uint8);rgb[15:30,15:220]=[0,0,160]
  patch=caption_patch(rgb,[10,10,230,35]);self.assertIsNotNone(patch);self.assertGreater(patch.shape[1],180)

 def test_literal_caption_retains_faint_comma_and_removes_red_symbol(self):
  rgb=np.full((40,100,3),255,np.uint8);rgb[20,30]=[180,180,190];rgb[25,50]=[170,0,0]
  patch=literal_caption_patch(rgb,[0,0,100,40])
  self.assertEqual(patch[20,30].tolist(),[180,180,190]);self.assertEqual(patch[25,50].tolist(),[255,255,255])

 def supply_fixture(self,value=None):
  rgb=np.full((120,160,3),255,np.uint8)
  target={'components':{'PORT':{'type':'vcc','Name':None,'value':value,'bbox':[60,40,80,60]}},'pins':{},'nets':{}}
  common={'image_sha256':'same','size':[160,120],'prediction_only':True,'coordinate_origin':'top_left'}
  rec={'component_id':'PORT','type':'vcc','detection_confidence':.95,'bbox_top_left':[60,60,80,80]}
  pins={**common,'pin_candidates':[{'owner':'PORT','owner_status':'assigned','point':[70,80],'confidence':.99}]}
  return target,rgb,common,{**common,'records':[rec]},pins

 def test_supply_requires_wire_evidence_and_unanimous_crop_votes(self):
  target,rgb,common,recs,pins=self.supply_fixture();ocr={**common,'texts':[{'text':'VCC','confidence':.9,'bbox':[55,45,85,62],'view':'source'}]}
  reader=lambda rgb,box,scale,view:[{'text':'VCC','confidence':.99,'view':str(scale)}]
  args=(rgb,common,recs,pins,ocr,'same')
  with patch('pcb_parser.component_caption_fields.outward_wire',return_value=None):
   out,_=recover_component_caption_fields(target,*args,reader=reader,cfg={'enabled':True});self.assertEqual(out,target)
  with patch('pcb_parser.component_caption_fields.outward_wire',return_value={'proof':True}):
   out,_=recover_component_caption_fields(target,*args,reader=reader,cfg={'enabled':True});self.assertEqual(out['components']['PORT']['value'],'VCC')
   conflict=lambda rgb,box,scale,view:[{'text':'VDD' if scale==4 else 'VCC','confidence':.99,'view':str(scale)}]
   out,_=recover_component_caption_fields(target,*args,reader=conflict,cfg={'enabled':True});self.assertEqual(out,target)

 def test_nearby_competing_power_owner_is_withheld(self):
  target,rgb,common,recs,pins=self.supply_fixture();target['components']['PORT2']=deepcopy(target['components']['PORT'])
  recs['records'].append({**recs['records'][0],'component_id':'PORT2','bbox_top_left':[75,60,95,80]})
  pins['pin_candidates'].append({**pins['pin_candidates'][0],'owner':'PORT2','point':[85,80]})
  ocr={**common,'texts':[{'text':'VCC','confidence':.9,'bbox':[55,45,85,62],'view':'source'}]}
  reader=lambda rgb,box,scale,view:[{'text':'VCC','confidence':.99,'view':str(scale)}]
  with patch('pcb_parser.component_caption_fields.outward_wire',return_value={'proof':True}):
   out,_=recover_component_caption_fields(target,rgb,common,recs,pins,ocr,'same',reader=reader,cfg={'enabled':True});self.assertEqual(out,target)

 def test_punctuation_requires_all_three_local_reads_and_no_digit_conflict(self):
  target,rgb,common,recs,pins=self.supply_fixture('1.8V')
  ocr={**common,'texts':[{'text':'1,8V','confidence':.9,'bbox':[55,45,85,62],'view':'source'}]}
  args=(rgb,common,recs,pins,ocr,'same')
  with patch('pcb_parser.component_caption_fields.outward_wire',return_value={'proof':True}):
   reader=lambda rgb,box,scale,view:[{'text':'1,8V','confidence':.99,'view':str(scale)}]
   out,_=recover_component_caption_fields(target,*args,reader=reader,cfg={'enabled':True});self.assertEqual(out['components']['PORT']['value'],'1,8V')
   conflict=lambda rgb,box,scale,view:[{'text':'1.8V' if scale==4 else '1,8V','confidence':.9 if scale==4 else .99,'view':str(scale)}]
   out,_=recover_component_caption_fields(target,*args,reader=conflict,cfg={'enabled':True});self.assertEqual(out,target)
   ocr['texts'].append({'text':'1.9V','confidence':.99,'bbox':[55,45,85,62],'view':'conflicting_source'})
   out,_=recover_component_caption_fields(target,*args,reader=reader,cfg={'enabled':True});self.assertEqual(out,target)


if __name__=='__main__':unittest.main()
