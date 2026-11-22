import copy,unittest
import cv2,numpy as np
from pcb_parser.connectivity_recovery import recover_connectivity

def fixture(existing=True):
    image=np.full((128,128,3),255,np.uint8)
    cv2.line(image,(10,50),(110,50),(0,160,0),3);cv2.circle(image,(60,90),8,(190,0,0),-1)
    target={'components':{'CHIP':{'type':'box','Name':'MODEL','value':None,'bbox':[5,68,15,88]},'PORT':{'type':'pin','Name':'PORT','value':None,'bbox':[105,68,115,88]}},
        'pins':{'CHIP':{'pin_1':{'pinname':'OUT','point':{'x':20.,'y':78.}}},'PORT':{'pin_1':{'pinname':'1','point':{'x':100.,'y':78.}}}},'nets':{}}
    if existing:target['nets']['N']={'hyperGraph':'(CHIP.1)','edges':{'e':[{'x':20.,'y':78.},{'x':100.,'y':78.}]}}
    meta={'image_sha256':'1'*64,'size':[128,128],'coordinate_origin':'top_left','prediction_only':True}
    graph={**meta,'edges':[{'edge_id':'W','polyline':[[20.,50.],[100.,50.]]}],
        'physical_networks':[{'network_id':'P','edge_ids':['W'],'pin_candidate_ids':[],'unresolved_junction_ids':[]}],
        'networks':[{'network_id':'N','physical_network_ids':['P']}],'junction_candidates':[]}
    pins={**meta,'pin_candidates':[]}
    return target,graph,pins,image

class ConnectivityRecoveryTests(unittest.TestCase):
    def run_case(self,**cfg):
        t,g,p,im=fixture();return recover_connectivity(t,g,p,im,cfg=cfg)
    def test_adds_unique_known_pin(self):
        out,a=self.run_case();self.assertEqual(out['nets']['N']['hyperGraph'],'(CHIP.1,PORT.1)');self.assertEqual(a['new_contact_count'],1)
    def test_does_not_mutate_inputs(self):
        t,g,p,im=fixture();before=copy.deepcopy((t,g,p));recover_connectivity(t,g,p,im);self.assertEqual((t,g,p),before)
    def test_idempotent(self):
        t,g,p,im=fixture();out,a=recover_connectivity(t,g,p,im);again,b=recover_connectivity(out,g,p,im);self.assertEqual(out,again);self.assertFalse(b['changes'])
    def test_old_fields_and_edges_preserved(self):
        t,g,p,im=fixture();out,a=recover_connectivity(t,g,p,im);self.assertEqual(t['pins'],out['pins']);self.assertEqual(t['components'],out['components']);self.assertEqual(t['nets']['N']['edges'],out['nets']['N']['edges'])
    def test_disabled_needs_no_artifacts(self):
        t,*_=fixture();out,a=recover_connectivity(t,{}, {},None,cfg={'enabled':False});self.assertEqual(t,out)
    def test_monochrome_skips(self):
        t,g,p,im=fixture();out,a=recover_connectivity(t,g,p,np.full_like(im,255));self.assertEqual(t,out);self.assertEqual(a['status'],'unsupported_color_profile')
    def test_wrong_hash_rejected(self):
        t,g,p,im=fixture();p['image_sha256']='2'*64
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_wrong_dimensions_rejected(self):
        t,g,p,im=fixture();g['size']=[127,128]
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_wrong_origin_rejected(self):
        t,g,p,im=fixture();p['coordinate_origin']='bottom_left'
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_nonprediction_artifact_rejected(self):
        t,g,p,im=fixture();g['prediction_only']=False
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_invalid_distance_rejected(self):
        with self.assertRaises(ValueError):self.run_case(maximum_distance_px=float('nan'))
    def test_missing_edge_rejected(self):
        t,g,p,im=fixture();g['edges']=[]
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_outside_edge_rejected(self):
        t,g,p,im=fixture();g['edges'][0]['polyline'][0]=[-1,50]
        with self.assertRaises(ValueError):recover_connectivity(t,g,p,im)
    def test_no_connect_never_attaches(self):
        t,g,p,im=fixture();t['pins']['PORT']['pin_1']['pinname']=' NC '
        out,a=recover_connectivity(t,g,p,im);self.assertEqual(t,out)
    def test_absent_wire_not_inferred(self):
        t,g,p,im=fixture();t['pins']['PORT']['pin_1']['point']['y']=65
        out,a=recover_connectivity(t,g,p,im);self.assertEqual(t,out)
    def test_nearby_network_ambiguity_withheld(self):
        t,g,p,im=fixture();g['edges'].append({'edge_id':'W2','polyline':[[90.,51.],[110.,51.]]});g['physical_networks'].append({'network_id':'P2','edge_ids':['W2'],'pin_candidate_ids':[],'unresolved_junction_ids':[]})
        out,a=recover_connectivity(t,g,p,im);self.assertEqual(t,out);self.assertEqual(a['proposals'][0]['status'],'ambiguous_physical_networks')
    def test_near_contact_crossing_withheld(self):
        t,g,p,im=fixture();g['junction_candidates']=[{'point':[100,50],'resolved':False}]
        out,a=recover_connectivity(t,g,p,im);self.assertEqual(t,out)
    def test_restores_two_vertex_branch(self):
        t,g,p,im=fixture(False);out,a=recover_connectivity(t,g,p,im);self.assertEqual(a['new_branch_count'],1);self.assertEqual(len(out['nets']),1)
    def test_does_not_restore_single_component_fragment(self):
        t,g,p,im=fixture(False);t['components']['CHIP']['type']='switch';t['pins']['CHIP']['pin_2']=t['pins'].pop('PORT')['pin_1'];del t['components']['PORT']
        out,a=recover_connectivity(t,g,p,im);self.assertFalse(out['nets'])
    def test_does_not_restore_uncertain_branch(self):
        t,g,p,im=fixture(False);g['physical_networks'][0]['unresolved_junction_ids']=['J']
        out,a=recover_connectivity(t,g,p,im);self.assertFalse(out['nets'])
    def test_contact_budget_atomic(self):
        out,a=self.run_case(maximum_new_contacts=0);self.assertFalse(a['changes'])
    def test_branch_budget(self):
        t,g,p,im=fixture(False);out,a=recover_connectivity(t,g,p,im,cfg={'maximum_new_branches':0});self.assertFalse(out['nets'])
    def test_fragment_pruning_is_opt_in(self):
        t,g,p,im=fixture();t['nets']['F']={'hyperGraph':'(PORT.1)','edges':{}};t['components']['PORT']['type']='box';t['pins']['PORT']['pin_1']['point']['y']=20
        out,a=recover_connectivity(t,g,p,im);self.assertIn('F',out['nets']);self.assertTrue(a['fragment_candidates'])
        out,a=recover_connectivity(t,g,p,im,cfg={'prune_zero_wire_fragments':True});self.assertNotIn('F',out['nets']);self.assertEqual(out['pins'],t['pins'])
    def test_source_order_independence(self):
        t,g,p,im=fixture();out,a=recover_connectivity(t,g,p,im);t['pins']=dict(reversed(list(t['pins'].items())));g['physical_networks'].reverse();again,b=recover_connectivity(t,g,p,im);self.assertEqual(out,again)

if __name__=='__main__':unittest.main()
