"""Portable synthetic contract tests; no cached output, GT, or drive dependency."""
import unittest, hashlib
from copy import deepcopy
import numpy as np
from PIL import Image,ImageFilter
from pcb_parser.precise_wire_contacts import refine_wire_contacts

def binding(rgb,sha='fixture'):
 return {'image_sha256':sha,'rgb_sha256':hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
         'size':[rgb.shape[1],rgb.shape[0]],'coordinate_origin':'top_left','prediction_only':True}
def doc(edge):
 return {'components':{'R':{'type':'resistor','bbox':[15,15,25,35]}},'pins':{'R':{'pin_1':{'point':{'x':20,'y':40}}}},
         'nets':{'n':{'hyperGraph':'(R.1, X.1)','edges':{'e':[dict(x=x,y=y) for x,y in edge]}}}}
def fixture():
 rgb=np.full((140,180,3),255,np.uint8)
 rgb[68:71,20:41]=[150,0,0]
 rgb[68:71,41:151]=[0,130,0]
 return rgb,doc([(30,70),(150,70)])
class ContactTests(unittest.TestCase):
 def run_refine(self,target,rgb,**kw):
  return refine_wire_contacts(target,rgb,'fixture',image_binding=binding(rgb),cfg={'enabled':True,**kw})
 def test_contact_and_idempotence(self):
  rgb,p=fixture();original=deepcopy(p);out,a=self.run_refine(p,rgb)
  self.assertEqual(out['nets']['n']['edges']['e'][0],{'x':41.0,'y':71.0})
  again,b=self.run_refine(out,rgb);self.assertEqual(out,again);self.assertEqual(b['changes'],[])
  self.assertEqual(p,original);self.assertEqual(p['components'],out['components']);self.assertEqual(p['pins'],out['pins'])
  self.assertEqual(set(p['nets']),set(out['nets']))
  self.assertEqual(set(p['nets']['n']['edges']),set(out['nets']['n']['edges']))
  self.assertEqual(p['nets']['n']['hyperGraph'],out['nets']['n']['hyperGraph'])
 def test_absent_red_lead_does_not_guess_free_end(self):
  rgb,p=fixture();rgb[68:71,20:41]=255
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_white_gap_is_not_a_color_contact(self):
  rgb,p=fixture();rgb[68:71,38:41]=255
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_even_one_white_pixel_gap_is_withheld(self):
  rgb,p=fixture();rgb[68:71,40:41]=255
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_half_pixel_endpoint_sampling_cannot_skip_white_column(self):
  rgb,p=fixture();p['nets']['n']['edges']['e'][0]['x']=30.5
  rgb[68:71,39:40]=255
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_short_or_fragmented_evidence_is_withheld(self):
  rgb,p=fixture();rgb[68:71,48:]=255
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_blurred_color_boundary_without_thin_axis_is_withheld(self):
  rgb,p=fixture();rgb=np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(3)))
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_rotation_and_translation_preserve_observed_contact_rule(self):
  rgb,p=fixture()
  # A vertical and translated example uses exactly the same runtime rule.
  rgb2=np.rot90(rgb).copy();h,w=rgb.shape[:2]
  edge=p['nets']['n']['edges']['e']
  rotated=doc([(h-v['y'],v['x']+1) for v in edge])
  out,a=self.run_refine(rotated,rgb2)
  self.assertEqual(out['nets']['n']['edges']['e'][0],{'x':69.0,'y':42.0})
 def test_shared_endpoint_is_immutable(self):
  rgb,p=fixture();p['nets']['n']['edges']['other']=[{'x':30,'y':70},{'x':30,'y':110}]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_existing_t_contact_at_edge_interior_is_immutable(self):
  rgb,p=fixture();p['nets']['n']['edges']['other']=[{'x':30,'y':40},{'x':30,'y':110}]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_proposal_cannot_create_a_contact_at_edge_interior(self):
  rgb,p=fixture();p['nets']['n']['edges']['other']=[{'x':41,'y':40},{'x':41,'y':110}]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_movement_cannot_cross_another_net_segment(self):
  rgb,p=fixture();p['nets']['other']={'hyperGraph':'(Z.1,W.1)',
       'edges':{'cross':[{'x':35,'y':40},{'x':35,'y':110}]}}
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_cross_net_t_contact_at_source_is_also_immutable(self):
  rgb,p=fixture();p['nets']['other']={'hyperGraph':'(Z.1,W.1)',
       'edges':{'cross':[{'x':30,'y':40},{'x':30,'y':110}]}}
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_lateral_recenter_cannot_create_contact_midway_along_edge(self):
  rgb,p=fixture();p['nets']['other']={'hyperGraph':'(Z.1,W.1)',
       'edges':{'parallel':[{'x':80,'y':72.5},{'x':100,'y':72.5}]}}
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_lateral_recenter_cannot_remove_existing_mid_edge_t(self):
  rgb,p=fixture();p['nets']['n']['edges']['e']=[{'x':30,'y':67},{'x':150,'y':67}]
  p['nets']['other']={'hyperGraph':'(Z.1,W.1)',
       'edges':{'branch':[{'x':60,'y':20},{'x':60,'y':67}]}}
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p);self.assertEqual(a['status'],'withheld_topology')
 def test_final_topology_failure_atomically_withholds_all_proposals(self):
  rgb,p=fixture();p['nets']['n']['edges']['e']=[{'x':30,'y':67},{'x':150,'y':67}]
  p['nets']['other']={'hyperGraph':'(Z.1,W.1)',
       'edges':{'branch':[{'x':60,'y':20},{'x':60,'y':67}]}}
  rgb[29:32,20:41]=[150,0,0];rgb[29:32,41:151]=[0,130,0]
  p['nets']['unrelated']={'hyperGraph':'(J.1,K.1)',
       'edges':{'safe':[{'x':30,'y':110},{'x':150,'y':110}]}}
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
  self.assertEqual(a['status'],'withheld_topology');self.assertGreaterEqual(a['proposed_changes'],2)
  self.assertEqual(a['changes'],[])
 def test_endpoint_extra_metadata_is_preserved(self):
  rgb,p=fixture();p['nets']['n']['edges']['e'][0].update(confidence=.9,evidence={'source':'observed'})
  out,a=self.run_refine(p,rgb);point=out['nets']['n']['edges']['e'][0]
  self.assertIn('confidence',point);self.assertEqual(point['x'],41.);self.assertEqual(point['confidence'],.9)
  self.assertEqual(point['evidence'],{'source':'observed'})
 def test_ambiguous_parallel_green_lines_are_withheld(self):
  rgb,p=fixture();rgb[74:77,41:151]=[0,130,0]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_unmarked_crossing_is_withheld(self):
  rgb,p=fixture();rgb[35:105,38:41]=[0,130,0]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_dot_contact_is_withheld(self):
  rgb,p=fixture()
  yy,xx=np.ogrid[:140,:180];rgb[(xx-40)**2+(yy-69)**2<=36]=[150,0,0]
  out,a=self.run_refine(p,rgb);self.assertEqual(out,p)
 def test_budget_is_atomic(self):
  rgb,p=fixture();out,a=self.run_refine(p,rgb,maximum_changes=0)
  self.assertEqual(out,p);self.assertEqual(a['status'],'withheld_budget')
 def test_image_binding_checks_actual_rgb(self):
  rgb,p=fixture();meta=binding(rgb);rgb[0,0]=0
  with self.assertRaisesRegex(ValueError,'image'):
   refine_wire_contacts(p,rgb,'fixture',image_binding=meta,cfg={'enabled':True})
 def test_image_binding_rejects_different_source_sha(self):
  rgb,p=fixture()
  with self.assertRaisesRegex(ValueError,'image'):
   refine_wire_contacts(p,rgb,'other',image_binding=binding(rgb),cfg={'enabled':True})
 def test_disabled_is_an_immutable_copy(self):
  rgb,p=fixture()
  out,a=refine_wire_contacts(p,rgb,'fixture',image_binding=binding(rgb))
  self.assertEqual(p,out);self.assertIsNot(out,p);self.assertEqual(a['status'],'disabled')
 def test_unknown_configuration_is_rejected(self):
  rgb,p=fixture()
  with self.assertRaisesRegex(ValueError,'setting'):
   self.run_refine(p,rgb,page_number=2)
if __name__=='__main__':unittest.main()
