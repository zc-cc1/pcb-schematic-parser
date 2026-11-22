import unittest
from copy import deepcopy
import cv2
import numpy as np
try:
 from pcb_parser.component_symbol_boxes import recover_component_symbol_boxes
except ImportError:
 recover_component_symbol_boxes=None

class ComponentSymbolBoxesTests(unittest.TestCase):
 def invoke(self,rgb,target,records,config=None,binding=None):
  self.assertIsNotNone(recover_component_symbol_boxes,'image-grounded bbox recovery is missing')
  h,w=rgb.shape[:2];bind={'image_sha256':'same','size':[w,h],'coordinate_origin':'top_left','prediction_only':True}
  if binding:bind.update(binding)
  before=deepcopy(target);source=deepcopy(records)
  out,audit=recover_component_symbol_boxes(target,rgb,bind,{**bind,'records':records},'same',cfg=config)
  self.assertEqual(target,before);self.assertEqual(records,source)
  self.assertEqual(out['pins'],target['pins']);self.assertEqual(out['nets'],target['nets'])
  for owner,c in out['components'].items():
   self.assertEqual({k:v for k,v in c.items() if k!='bbox'},{k:v for k,v in target['components'][owner].items() if k!='bbox'})
  return out,audit

 def ground(self,rotated=False):
  rgb=np.full((240,240,3),255,np.uint8)
  for x,length in [(140,56),(148,38),(156,22),(164,6)]:cv2.line(rgb,(x,150-length//2),(x,150+length//2),(180,0,0),2)
  cv2.line(rgb,(110,150),(140,150),(180,0,0),2);cv2.circle(rgb,(110,150),4,(180,0,0),-1)
  bbox=[130,80,150,100];recordbox=[105,115,170,185]
  if rotated:
   rgb=np.transpose(rgb,(1,0,2)).copy();bbox=[140,90,160,110];recordbox=[115,105,185,170]
  target={'components':{'GROUND':{'type':'gnd','Name':'GND','value':'GND','extra':'retain','bbox':bbox},'R1':{'type':'resistor','Name':'R1','value':'1k','bbox':[1,1,5,5]}},'pins':{'GROUND':{'pin_1':{'point':{'x':150 if rotated else 110,'y':130 if rotated else 90}}}},'nets':{'N':{'hyperGraph':'(GROUND.1)','edges':{}}}}
  records=[{'component_id':'GROUND','type':'gnd','detection_confidence':.9,'bbox_top_left':recordbox}]
  return rgb,target,records

 def test_ground_frame_contains_all_four_bars_and_lead(self):
  out,audit=self.invoke(*self.ground(),{'enabled':True});a,b,c,d=out['components']['GROUND']['bbox']
  self.assertGreaterEqual(d-b,57);self.assertLessEqual(a,110);self.assertGreaterEqual(c,165)
  self.assertEqual(audit['changes'][0]['bar_count'],4)

 def test_horizontal_comb_rotates_the_observed_frame(self):
  out,_=self.invoke(*self.ground(True),{'enabled':True});a,b,c,d=out['components']['GROUND']['bbox']
  self.assertGreaterEqual(c-a,57);self.assertGreaterEqual(d-b,55)

 def test_ground_frame_stops_at_existing_contact_instead_of_following_wire(self):
  rgb,target,records=self.ground();cv2.line(rgb,(100,150),(140,150),(180,0,0),2)
  target['pins']['GROUND']['pin_1']['point']['x']=125;cv2.circle(rgb,(125,150),4,(180,0,0),-1)
  out,_=self.invoke(rgb,target,records,{'enabled':True})
  self.assertEqual(out['components']['GROUND']['bbox'][0],125.)

 def test_plain_wire_contact_does_not_expand_ground_comb_frame(self):
  rgb,target,records=self.ground();rgb[144:157,104:116]=255;cv2.line(rgb,(100,150),(140,150),(180,0,0),2)
  out,_=self.invoke(rgb,target,records,{'enabled':True})
  self.assertGreaterEqual(out['components']['GROUND']['bbox'][0],139.)

 def test_ground_lead_uses_visible_junction_when_contact_is_slightly_offset(self):
  rgb,target,records=self.ground();target['pins']['GROUND']['pin_1']['point']['x']=116
  out,_=self.invoke(rgb,target,records,{'enabled':True})
  self.assertLessEqual(out['components']['GROUND']['bbox'][0],111.)

 def test_adequate_ground_frame_is_preserved(self):
  rgb,target,records=self.ground();target['components']['GROUND']['bbox']=[108,60,167,120]
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)

 def test_unproven_ground_single_bar_is_withheld(self):
  rgb,target,records=self.ground();rgb[:,145:]=255
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)

 def test_default_disabled_and_budget_rollback(self):
  args=self.ground();out,audit=self.invoke(*args);self.assertEqual(out,args[1]);self.assertEqual(audit['status'],'disabled')
  out,audit=self.invoke(*args,{'enabled':True,'maximum_changes':0});self.assertEqual(out,args[1]);self.assertEqual(audit['status'],'withheld_budget')

 def test_replay_is_idempotent(self):
  rgb,target,records=self.ground();out,_=self.invoke(rgb,target,records,{'enabled':True});again,audit=self.invoke(rgb,out,records,{'enabled':True})
  self.assertEqual(again,out);self.assertEqual(audit['status'],'no_effect')

 def test_stale_evidence_is_rejected(self):
  for binding in [{'image_sha256':'other'},{'size':[1,1]},{'coordinate_origin':'bottom_left'},{'prediction_only':False}]:
   with self.assertRaisesRegex(ValueError,'evidence_differs'):self.invoke(*self.ground(),{'enabled':True},binding)

 def test_low_confidence_and_shared_source_combs_are_withheld(self):
  rgb,target,records=self.ground();records[0]['detection_confidence']=.69
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)
  records[0]['detection_confidence']=.9;target['components']['GROUND_symbol_2']={**target['components']['GROUND'],'bbox':[135,65,169,105]}
  target['pins']['GROUND_symbol_2']={}
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)

 def led(self):
  rgb=np.full((220,220,3),255,np.uint8);red=(180,0,0)
  cv2.line(rgb,(80,60),(80,140),red,2);cv2.polylines(rgb,[np.array([[70,90],[90,90],[80,110]])],True,red,2);cv2.line(rgb,(70,110),(90,110),red,2)
  for x,y in [(102,100),(113,89)]:
   cv2.line(rgb,(x,y),(x+12,y+12),red,2);cv2.fillPoly(rgb,[np.array([[x+12,y+12],[x+4,y+9],[x+9,y+4]])],red)
  target={'components':{'D':{'type':'led','Name':'LAMP','value':None,'bbox':[83,100,97,120]}},'pins':{'D':{'pin_1':{'point':{'x':80,'y':80}}}},'nets':{}}
  records=[{'component_id':'D','type':'led','detection_confidence':.65,'bbox_top_left':[65,80,132,125]}]
  return rgb,target,records

 def test_led_frame_includes_two_proven_emission_arrows(self):
  out,audit=self.invoke(*self.led(),{'enabled':True});a,b,c,d=out['components']['D']['bbox']
  self.assertLessEqual(a,70);self.assertGreaterEqual(c,125);self.assertGreaterEqual(d-b,23)
  self.assertEqual(audit['changes'][0]['arrow_count'],2)

 def test_two_plain_diagonal_strokes_are_not_emission_arrows(self):
  rgb,target,records=self.led();rgb[:,99:]=255
  for x,y in [(102,100),(113,89)]:cv2.line(rgb,(x,y),(x+12,y+12),(180,0,0),2)
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)

 def test_led_without_two_arrows_is_withheld(self):
  rgb,target,records=self.led();rgb[:99,100:]=255
  out,_=self.invoke(rgb,target,records,{'enabled':True});self.assertEqual(out,target)

if __name__=='__main__':unittest.main()
