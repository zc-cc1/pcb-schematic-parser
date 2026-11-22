import copy,json,unittest
from pathlib import Path
import cv2,numpy as np

from pcb_parser.color_trace import make_graph
from pcb_parser import wire_graph
from pcb_parser.structural_patterns import StructuralDraft
from pcb_parser.structural_recovery import recover_image_structures,ImageStructurePolicy
from pcb_parser.repair_engine import Edit,Proposal,RepairContext,RepairPipeline,RuleSpec
from test_connectivity_recovery import fixture

WIRE=json.loads((Path(wire_graph.__file__).resolve().parents[2]/'configs/wire_graph_v1.json').read_text(encoding='utf-8'))


def docs():
    target,graph,pins,rgb=fixture()
    meta={k:graph[k] for k in ('image_sha256','size','coordinate_origin','prediction_only')}
    return target,pins,{**meta,'records':[]},{**meta,'texts':[]},rgb


def draft(target=None,rgb=None,records=None,pins=None,ocr=None,cfg=None):
    t,p,r,o,im=docs()
    return StructuralDraft(target or t,im if rgb is None else rgb,r if records is None else records,
                           p if pins is None else pins,o if ocr is None else ocr,WIRE,cfg or {})


class ColoredTopologyTests(unittest.TestCase):
    def colored(self,dot=False,gap=None):
        im=np.full((101,101,3),255,np.uint8)
        cv2.line(im,(10,50),(90,50),(0,150,0),2)
        if gap:
            cv2.rectangle(im,(46,48),(52,52),(40,60,180) if gap=='blue' else (255,255,255),-1)
        else:cv2.line(im,(50,10),(50,90),(0,150,0),2)
        if dot:cv2.circle(im,(50,50),5,(190,0,0),-1)
        return make_graph(im,{},WIRE)

    def test_no_dot_crossing_is_not_short_circuited(self):
        nets,bind,proof,mask=self.colored()
        self.assertNotEqual(bind([20,50]),bind([50,20]))
        self.assertEqual(bind([20,50]),bind([80,50]))

    def test_observed_red_dot_connects_crossing(self):
        nets,bind,proof,mask=self.colored(dot=True)
        self.assertEqual(bind([20,50]),bind([50,20]))

    def test_blue_text_can_bridge_short_collinear_gap(self):
        nets,bind,proof,mask=self.colored(gap='blue')
        self.assertEqual(bind([20,50]),bind([80,50]))
        self.assertGreater(proof['blue_text_occlusion_pixels'],0)

    def test_plain_white_gap_is_never_filled(self):
        nets,bind,proof,mask=self.colored(gap='white')
        self.assertNotEqual(bind([20,50]),bind([80,50]))
        self.assertEqual(proof['blue_text_occlusion_pixels'],0)

    def test_lead_extension_stops_at_component_keepout(self):
        im=np.full((101,101,3),255,np.uint8)
        cv2.line(im,(10,50),(40,50),(190,0,0),2)
        cv2.line(im,(40,50),(90,50),(0,150,0),2)
        keepout=np.zeros((101,101),bool);keepout[45:56,20:30]=True
        nets,bind,proof,mask=make_graph(im,{'A.1':[10,50]},WIRE,keepout)
        self.assertFalse(proof['pin_lead_extensions'])


class StructuralPatternTests(unittest.TestCase):
    def red_pair_draft(self,inward=False):
        im=np.full((128,128,3),255,np.uint8)
        end=80 if inward else 10
        cv2.line(im,(50,40),(50,end),(190,0,0),2)
        target={'components':{'CAP':{'type':'c','bbox':[45,68,55,88]},
                              'GROUND':{'type':'gnd','bbox':[45,123 if not inward else 43,55,127 if not inward else 47]}},
                'pins':{'CAP':{'pin_1':{'pinname':'','point':{'x':50.,'y':88.}}},
                        'GROUND':{'pin_1':{'pinname':'','point':{'x':50.,'y':128.-end}}}},'nets':{}}
        return draft(target=target,rgb=im)

    def test_continuous_red_cannot_go_inward_through_capacitor(self):
        d=self.red_pair_draft(True);old=copy.deepcopy(d.out)
        self.assertFalse(d.direct_red_pair('CAP.1','GROUND.1'))
        self.assertEqual(d.out,old)

    def test_continuous_outward_red_stem_can_connect(self):
        d=self.red_pair_draft(False)
        self.assertTrue(d.direct_red_pair('CAP.1','GROUND.1'))
        self.assertEqual(len(d.out['nets']),1)

    def test_collinear_white_gap_does_not_form_red_pair(self):
        d=self.red_pair_draft(False);d.red[20:30,:]=False
        self.assertFalse(d.direct_red_pair('CAP.1','GROUND.1'))

    def test_unaligned_red_contacts_are_rejected(self):
        d=self.red_pair_draft(False);d.out['pins']['GROUND']['pin_1']['point']['x']=65
        self.assertFalse(d.direct_red_pair('CAP.1','GROUND.1'))

    def test_opposing_rails_cannot_be_merged_by_red_pair(self):
        d=self.red_pair_draft(False)
        d.out['components']['CAP']['type']='v';d.out['components']['CAP']['value']='VCC'
        self.assertFalse(d.direct_red_pair('CAP.1','GROUND.1'))

    def test_no_two_visible_stems_means_no_terminal_correction(self):
        d=draft();d.out['components']['PART']={'type':'r','bbox':[45,45,55,55]}
        d.out['pins']['PART']={'pin_1':{'pinname':'','point':{'x':45.,'y':50.}},'pin_2':{'pinname':'','point':{'x':55.,'y':50.}}}
        d.rmap['PART']={'bbox_top_left':[45,73,55,83]};before=copy.deepcopy(d.out)
        self.assertFalse(d.normalize_passive('PART'))
        self.assertEqual(d.out,before)

    def test_nc_is_excluded_from_image_binding(self):
        t,p,r,o,im=docs();t['pins']['PORT']['pin_1']['pinname']=' NC '
        d=draft(t,im,r,p,o);d.rebuild_color_networks()
        self.assertFalse(any('PORT.1' in e.get('members',[]) for e in d.events))

    def test_fragment_cleanup_preserves_pin_records(self):
        d=draft();d.out['pins']['PORT']['pin_1']['point']['y']=20
        d.out['components']['PORT']['type']='box';d.out['nets']['F']={'hyperGraph':'(PORT.1)','edges':{}}
        before=copy.deepcopy(d.out['pins']);d.prune_fragments()
        self.assertNotIn('F',d.out['nets']);self.assertEqual(d.out['pins'],before)


class StructuralExecutionTests(unittest.TestCase):
    def execute(self,**kwargs):
        t,p,r,o,im=docs()
        return recover_image_structures(t,im,r,p,o,WIRE,**kwargs)

    def test_disabled_requires_no_evidence(self):
        t,*_=docs()
        out,a=recover_image_structures(t,None,None,None,None,None,cfg={'enabled':False})
        self.assertEqual(t,out)

    def test_wrong_ocr_hash_rejected(self):
        t,p,r,o,im=docs();o['image_sha256']='different'
        with self.assertRaisesRegex(ValueError,'differs from current image'):
            recover_image_structures(t,im,r,p,o,WIRE)

    def test_wrong_record_origin_rejected(self):
        t,p,r,o,im=docs();r['coordinate_origin']='bottom_left'
        with self.assertRaises(ValueError):recover_image_structures(t,im,r,p,o,WIRE)

    def test_unpredicted_records_rejected(self):
        t,p,r,o,im=docs();r['prediction_only']=False
        with self.assertRaises(ValueError):recover_image_structures(t,im,r,p,o,WIRE)

    def test_unknown_pattern_does_not_silently_execute(self):
        with self.assertRaises(ValueError):self.execute(cfg={'patterns':{'typo':True}})

    def test_pattern_switch_is_boolean(self):
        with self.assertRaises(ValueError):self.execute(cfg={'patterns':{'visible_terminal_pair':'false'}})

    def test_dependent_geometry_cannot_execute_without_topology(self):
        with self.assertRaisesRegex(ValueError,'require_topology'):
            self.execute(cfg={'patterns':{'colored_wire_network':False}})

    def test_invalid_budgets_are_rejected(self):
        for key,value in [('maximum_caption_views',-1),('maximum_color_networks',0),('maximum_changes',1.5)]:
            with self.assertRaises(ValueError):self.execute(cfg={key:value})

    def test_unknown_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'unknown_structural'):
            self.execute(cfg={'experimental_typo':1})

    def test_no_new_ocr_without_supplied_reader(self):
        out,a=self.execute()
        self.assertFalse(a['errors'])
        self.assertEqual(a['proposals'][0]['evidence']['caption_reader_calls'],0)

    def test_zero_edit_budget_withholds_whole_reconstruction(self):
        t,p,r,o,im=docs();t['nets']={}
        out,a=recover_image_structures(t,im,r,p,o,WIRE,cfg={'maximum_changes':0})
        self.assertEqual(out,t)
        self.assertEqual(a['proposals'][0]['rejection'],'structural_edit_budget')

    def test_color_budget_exhaustion_discards_partial_draft(self):
        t,p,r,o,im=docs();t['nets']={}
        for owner in ('CHIP','PORT'):
            t['components'][owner+'2']=copy.deepcopy(t['components'][owner])
            t['pins'][owner+'2']=copy.deepcopy(t['pins'][owner])
            t['pins'][owner+'2']['pin_1']['point']['y']=108.
        cv2.line(im,(10,20),(110,20),(0,160,0),3)
        out,a=recover_image_structures(t,im,r,p,o,WIRE,cfg={'maximum_color_networks':1})
        self.assertEqual(out,t)
        self.assertFalse(a['accepted'])
        self.assertIn('color_network_budget_exhausted',a['errors'][0]['reason'])

    def test_protected_chip_modification_rolls_back_atomically(self):
        t,p,r,o,im=docs()
        class BadRule:
            spec=RuleSpec('bad','1','bad',('image_verified_structure',))
            def propose(self,c):
                changed=copy.deepcopy(c.target['pins']['CHIP']);changed['pin_1']['point']['x']=35
                yield Proposal('bad',(Edit.set(c.target,('pins','CHIP'),changed),),
                               {'pattern_changes':[{'rule':'visible_terminal_pair','owner':'CHIP'}]})
        out,a=RepairPipeline([BadRule()],extension=ImageStructurePolicy()).run(RepairContext(t,{},128,128))
        self.assertEqual(out,t)
        self.assertEqual(a['proposals'][0]['rejection'],'protected_component_or_pin_changed')


if __name__=='__main__':unittest.main()
