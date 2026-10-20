"""Small synthetic export checks; never loads a detector, OCR model or GT page."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import agent
from pcb_parser.power_wire_export import restore_power_wires

spec = importlib.util.spec_from_file_location('production_wire', ROOT/'src/pcb_parser/wire_graph.py')
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)


def inputs():
    base = {'components': {'G1': {'type':'gnd','value':'GND'}, 'R1': {'type':'r'}},
            'pins': {'R1': {'pin_1': {'pinname':'','point':{'x':0,'y':50}}}},
            'nets': {'n1': {'hyperGraph':'(R1.1)','edges':{'e1':[{'x':0,'y':50},{'x':50,'y':50}]}}}}
    primary = {'candidate_id':'p1','owner':'G1','owner_status':'assigned','number':'1',
               'point':[50,50],'pinname':None}
    duplicate = {**primary,'candidate_id':'p2','number':'2',
                 'pinname_status':'duplicate_tip_hypothesis','text_hypothesis_primary_candidate_id':'p1'}
    provenance = {'image_sha256':'current','size':[100,100],'coordinate_origin':'top_left','prediction_only':True}
    pins = {**provenance,'pin_candidates':[primary,duplicate]}
    graph = {**provenance,'edges':[],'physical_networks':[],
             'networks':[{'network_id':'n1','pin_candidate_ids':['p1','p2']}],
             'pin_attachments':[{'candidate_id':'p1','point':[50,50],
                                 'attachment_status':'attached','network_id':'n1'}],
             'power_merge_decisions':[{'label':'GND','withheld_reasons':[],
                                       'status':'power_logical_connectivity_proposal'}]}
    return base,graph,pins


class PowerExportTests(unittest.TestCase):
    def run_export(self, base, graph, pins):
        return restore_power_wires(base,graph,pins,agent.cleanup_wire_edges)

    def test_primary_alias_recovers_contact_and_preserves_geometry(self):
        base,graph,pins = inputs()
        original = copy.deepcopy((base,graph,pins))
        result,audit = self.run_export(base,graph,pins)
        self.assertEqual(result['nets']['n1']['hyperGraph'],'(G1.1,R1.1)')
        self.assertEqual(result['nets']['n1']['edges'],base['nets']['n1']['edges'])
        self.assertEqual(result['pins']['G1']['pin_1']['point'],{'x':50.,'y':50.})
        self.assertNotIn('pin_2',result['pins']['G1'])
        self.assertEqual(audit['primary_contact_count'],1)
        self.assertEqual((base,graph,pins),original)
        self.assertEqual(self.run_export(result,graph,pins)[0],result)

    def test_uncertain_or_inconsistent_alias_is_not_promoted(self):
        for problem in ['two_primaries','foreign_primary','unattached','wrong_point','unknown_number',
                        'missing_rail_value','unassigned_owner','absent_model_member','missing_output_net']:
            base,graph,pins = inputs()
            if problem=='two_primaries': pins['pin_candidates'][1].pop('text_hypothesis_primary_candidate_id')
            if problem=='foreign_primary': pins['pin_candidates'][1]['text_hypothesis_primary_candidate_id']='other'
            if problem=='unattached': graph['pin_attachments'][0]['attachment_status']='ambiguous_nearby_networks'
            if problem=='wrong_point': graph['pin_attachments'][0]['point']=[60,60]
            if problem=='unknown_number': pins['pin_candidates'][0]['number']=None
            if problem=='missing_rail_value': base['components']['G1']['value']=''
            if problem=='unassigned_owner': pins['pin_candidates'][0]['owner_status']='ambiguous'
            if problem=='absent_model_member': graph['networks'][0]['pin_candidate_ids']=[]
            if problem=='missing_output_net': base['nets']={}
            with self.subTest(problem=problem):
                self.assertEqual(self.run_export(base,graph,pins)[0],base)

    def test_unique_contact_recovers_without_upstream_rail_merge(self):
        base,graph,pins=inputs()
        pins['pin_candidates']=pins['pin_candidates'][:1]
        graph['power_merge_decisions']=[]
        result,audit=self.run_export(base,graph,pins)
        self.assertEqual(result['nets']['n1']['hyperGraph'],'(G1.1,R1.1)')
        self.assertEqual(result['nets']['n1']['edges'],base['nets']['n1']['edges'])
        self.assertEqual(audit['existing_power_contact_count'],1)
        self.assertEqual(audit['existing_power_contacts'][0]['reason'],'unique_observed_power_tip')
        self.assertEqual(self.run_export(result,graph,pins)[0],result)

    def test_duplicate_primary_recovers_without_upstream_rail_merge(self):
        base,graph,pins=inputs()
        graph['power_merge_decisions']=[]
        result,audit=self.run_export(base,graph,pins)
        self.assertIn('G1.1',result['nets']['n1']['hyperGraph'])
        self.assertNotIn('pin_2',result['pins']['G1'])
        self.assertEqual(audit['existing_power_contacts'][0]['duplicate_count'],1)

    def test_no_new_net_is_created_for_a_missing_output_network(self):
        base,graph,pins=inputs()
        base['nets']={}
        result,audit=self.run_export(base,graph,pins)
        self.assertEqual(result,base)
        self.assertEqual(audit['existing_power_contact_count'],0)

    def test_source_artifact_mismatches_are_rejected(self):
        for field,value in [('size',[200,100]),('coordinate_origin','bottom_left'),('prediction_only',False)]:
            base,graph,pins=inputs()
            pins[field]=value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):self.run_export(base,graph,pins)

    def test_foreign_image_is_rejected_without_mutation(self):
        base,graph,pins = inputs()
        pins['image_sha256']='other'
        original=copy.deepcopy(base)
        with self.assertRaises(ValueError): self.run_export(base,graph,pins)
        self.assertEqual(base,original)

    def test_ordinary_contacts_are_never_added(self):
        base,graph,pins=inputs()
        base['components']['G1']['type']='r'
        self.assertEqual(self.run_export(base,graph,pins)[0],base)

    def test_existing_pin_values_are_never_overwritten(self):
        base,graph,pins=inputs()
        base['pins']['G1']={'pin_1':{'pinname':'observed','point':{'x':49,'y':51}}}
        result,_=self.run_export(base,graph,pins)
        self.assertEqual(result['pins']['G1'],base['pins']['G1'])

    def test_contact_already_in_other_net_cannot_be_stolen(self):
        base,graph,pins=inputs()
        base['nets']['other']={'hyperGraph':'(G1.1)','edges':{}}
        self.assertEqual(self.run_export(base,graph,pins)[0],base)

    def test_switch_disables_artifact_reads(self):
        base,_,_=inputs()
        with patch.dict(os.environ,{'PCB_POWER_WIRE_EXPORT':'off'}):
            result,audit=agent.finalize_power_wire_export(base,Path('nonexistent'), 'current')
        self.assertIs(result,base)
        self.assertFalse(audit['enabled'])

    def test_branch_qualification_and_physical_geometry(self):
        base,graph,pins=inputs()
        pins['pin_candidates']=pins['pin_candidates'][:1]+[
            {'candidate_id':'r','owner':'R1','owner_status':'assigned','number':'1','point':[0,50]}]
        graph['networks']=[{'network_id':'power','power_logical_merge':True,'power_names':['GND'],
                            'power_site_ids':['s'],'edge_ids':['w']}]
        graph['physical_networks']=[{'network_id':'physical','pin_candidate_ids':['p1','r'],
                                     'unresolved_junction_ids':[],'edge_ids':['w']}]
        graph['power_sites']=[{'site_id':'s','candidate_id':'p1','component_id':'G1',
                               'label':'GND','withheld_reasons':[]}]
        graph['edges']=[{'edge_id':'w','polyline':[[0,50],[50,50]]}]
        graph['pin_attachments'][0]['network_id']='power'
        fixed,audit=self.run_export(base,graph,pins)
        self.assertEqual(audit['restored_net_count'],1)
        self.assertEqual(fixed['nets']['RESTORED_physical']['hyperGraph'],'(G1.1,R1.1)')
        self.assertEqual(fixed['nets']['n1'],base['nets']['n1'])
        covered=copy.deepcopy(base)
        covered['nets']['power']={'hyperGraph':'(G1.1,R1.1)','edges':{}}
        self.assertEqual(self.run_export(covered,graph,pins)[0],covered)
        graph['power_sites'][0]['withheld_reasons']=['unresolved_caption']
        rejected,audit=self.run_export(base,graph,pins)
        self.assertEqual(rejected,base)
        self.assertEqual(audit['restored_net_count'],0)


class PathSplittingTests(unittest.TestCase):
    def graph(self, probability, points):
        # Production settings relevant to the small geometry, without model inference.
        config_path=ROOT/'configs/wire_graph_v1.json'
        if not config_path.is_file():
            config_path=Path('D:/pcb5_submission/configs/wire_graph_v1.json')
        cfg={**json.loads(config_path.read_text(encoding='utf-8')),
             'body_guard':False,'bridge_max_gap_px':0}
        pins={'size':list(probability.shape[::-1]),'coordinate_origin':'top_left','components':[],
              'pin_candidates':[{'candidate_id':f'p{i}','point':p,'owner_status':'unassigned',
                                  'owner_detection_index':None,'owner':None} for i,p in enumerate(points)]}
        return wire.build_wire_graph(probability,pins,cfg)[0]

    def test_bend_preserved_and_chord_pin_rejected(self):
        p=np.zeros((100,100),dtype=np.float32)
        p[10,10:81]=1;p[10:81,80]=1
        off=self.graph(p,[[45.,45.]])
        self.assertEqual(off['pin_attachments'][0]['attachment_status'],'no_wire_within_radius')
        real=self.graph(p,[[80.,45.]])
        self.assertEqual(real['pin_attachments'][0]['attachment_status'],'attached')
        for edge in real['edges']:
            for a,b in zip(edge['polyline'],edge['polyline'][1:]):
                self.assertTrue(abs(a[0]-b[0])<=1.1 or abs(a[1]-b[1])<=1.1)

    def test_dense_path_and_length_preserved_after_multiple_cuts(self):
        p=np.zeros((100,100),dtype=np.float32);p[10,10:91]=1
        result=self.graph(p,[[40.,10.],[70.,15.]])
        self.assertTrue(all(a['attachment_status']=='attached' for a in result['pin_attachments']))
        self.assertAlmostEqual(sum(e['length_px'] for e in result['edges']),80)
        self.assertEqual(len(result['networks']),1)

    def test_no_pin_path_unchanged(self):
        p=np.zeros((100,100),dtype=np.float32);p[10,10:91]=1
        result=self.graph(p,[])
        self.assertEqual(len(result['edges']),1)
        self.assertAlmostEqual(result['edges'][0]['length_px'],80)


if __name__=='__main__': unittest.main()
