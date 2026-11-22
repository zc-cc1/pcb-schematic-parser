import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.ground_symbol_identity import ground_comb_group,split_ground_identities


class GroundSymbolIdentityTests(unittest.TestCase):
 def fixture(self):
  rgb=np.full((300,400,3),255,np.uint8)
  for y in [150,170]:
   cv2.line(rgb,(200,y),(220,y),(190,0,0),1)
   for x,half in [(220,20),(226,12),(232,5)]:cv2.line(rgb,(x,y-half),(x,y+half),(190,0,0),2)
  record={'component_id':'SOURCE_GROUND_7','type':'gnd','detection_confidence':.9,'bbox_top_left':[195,125,239,196]}
  return rgb,record,[[200.,150.],[200.,170.]]

 def test_two_short_comb_teeth_prove_two_physical_symbols(self):
  proof=ground_comb_group(*self.fixture());self.assertIsNotNone(proof)
  self.assertEqual(len(proof['symbols']),2)
  self.assertGreater(proof['symbols'][0]['body'][3]-proof['symbols'][0]['body'][1],35)

 def test_multiple_contacts_do_not_copy_one_ground_symbol(self):
  rgb,record,points=self.fixture();rgb[155:177,230:235]=255
  self.assertIsNone(ground_comb_group(rgb,record,points))

 def test_a_white_gap_in_the_lead_withholds_the_group(self):
  rgb,record,points=self.fixture();rgb[147:154,206:212]=255
  self.assertIsNone(ground_comb_group(rgb,record,points))

 def invoke(self,*,budget=8,collision=False,stale=False):
  rgb,record,points=self.fixture();height=rgb.shape[0];owner=record['component_id']
  pin=lambda p:{'pinname':'','point':{'x':p[0],'y':height-p[1]}}
  target={'components':{owner:{'type':'gnd','Name':'GND','value':'GND','bbox':[215,140,235,160]},
                         'D8':{'type':'led','Name':'lamp','value':None,'bbox':[160,140,180,160]},
                         'D9':{'type':'led','Name':'lamp','value':None,'bbox':[160,120,180,140]}},
          'pins':{owner:{'pin_1':pin(points[0]),'pin_2':pin(points[1])},'D8':{'pin_1':pin(points[0])},'D9':{'pin_1':pin(points[1])}},
          'nets':{'a':{'hyperGraph':'(D8.1,'+owner+'.1)','edges':{}},'b':{'hyperGraph':'(D9.1,'+owner+'.2)','edges':{}}}}
  if collision:target['components'][owner+'_symbol_2']={'type':'box','Name':'X','value':None,'bbox':[10,10,30,30]};target['pins'][owner+'_symbol_2']={}
  binding={'image_sha256':'wrong' if stale else 'same','size':[400,300],'coordinate_origin':'top_left','prediction_only':True}
  before=deepcopy(target)
  out,audit=split_ground_identities(target,rgb,binding,{**binding,'records':[record]},'same',cfg={'enabled':True,'maximum_new_symbols':budget})
  self.assertEqual(target,before)
  return target,out,audit,rgb,binding,record

 def test_split_moves_each_contact_reference_and_is_idempotent(self):
  target,out,audit,rgb,binding,record=self.invoke();owner=record['component_id'];child=owner+'_symbol_2'
  self.assertIn(child,out['components']);self.assertEqual(len(out['components']),len(target['components'])+1)
  self.assertEqual(set(out['pins'][owner]),{'pin_1'});self.assertEqual(set(out['pins'][child]),{'pin_1'})
  self.assertEqual(out['pins'][child]['pin_1'],target['pins'][owner]['pin_2'])
  self.assertEqual(out['nets']['b']['hyperGraph'],'(D9.1,'+child+'.1)')
  again,_=split_ground_identities(out,rgb,binding,{**binding,'records':[record]},'same',cfg={'enabled':True})
  self.assertEqual(again,out)

 def test_existing_unrelated_identity_is_not_overwritten(self):
  target,out,audit,*_=self.invoke(collision=True);self.assertEqual(out,target)

 def test_budget_rolls_back_the_entire_group(self):
  target,out,audit,*_=self.invoke(budget=0);self.assertEqual(out,target);self.assertEqual(audit['status'],'withheld_budget')

 def test_existing_child_ground_memberships_are_retained_on_replay(self):
  original,out,_,rgb,binding,record=self.invoke();owner=record['component_id'];child=owner+'_symbol_2'
  out['pins'][owner]=deepcopy(original['pins'][owner])
  out['nets']['extra']={'hyperGraph':'(D8.1,'+child+'.1)','edges':{}}
  result,_=split_ground_identities(out,rgb,binding,{**binding,'records':[record]},'same',cfg={'enabled':True})
  self.assertEqual(result['nets']['extra']['hyperGraph'],'(D8.1,'+child+'.1)')

 def test_stale_source_image_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'evidence_differs_from_image'):self.invoke(stale=True)


if __name__=='__main__':unittest.main()
