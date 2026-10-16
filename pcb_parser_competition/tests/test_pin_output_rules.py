"""Regression for OCR-vs-convention evidence and incomplete power members."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pcb_parser.pin_output_rules import complete_pin_output_names, power_network_guard, apply_power_export_gate, simplify_model_polyline, compress_candidate_edges
from run_network_labels import logical_partial_export

CFG=json.loads((ROOT/'configs/pin_output_rules_v1.json').read_text(encoding='utf-8'))


def example(kind='resistor', number='1'):
    record={'detection_index':0,'component_id':'R1','designator':'R1','type':kind,'Name':None,'value':'10k',
            'bbox_top_left':[10,30,40,50]}
    ev={'text':number,'source_text':number,'source_view':'global','confidence':.999,'bbox':[9,20,17,29],
        'approximate_split_geometry':False,'global_assignment_margin':20,
        'alternatives':[{'text':number,'source_text':number,'source_view':'tile_r0','confidence':.999,
                         'bbox':[9,20,17,29],'approximate_split_geometry':False}]}
    pin={'candidate_id':'P0','point':[10,40],'owner':'R1','owner_status':'assigned','owner_detection_index':0,
         'number':number,'pinname':None,'number_status':'observed_OCR_text','text_evidence':{'number':ev},'text_candidates':{}}
    pins={'size':[180,100],'components':[{'detection_index':0,'component_id':'R1','type':kind,'bbox_top_left':[10,30,40,50]}],
          'pin_candidates':[pin]}
    graph={'pin_attachments':[{'candidate_id':'P0','point':[10,40],'attachment_status':'attached'}],
           'networks':[{'network_id':'N','pin_candidate_ids':['P0'],'edge_ids':['E'],'unresolved_junction_ids':[]}],
           'edges':[{'edge_id':'E','polyline':[[10,40],[80,40]]}],'gap_candidates':[],'label_sites':[],
           'summary':{'logical_label_merges':0}}
    return pins,graph,[record]


class PinOutputTests(unittest.TestCase):
    def test_passive_extension_is_candidate_only_and_number_is_unchanged(self):
        p,g,r=example();before=deepcopy((p,g,r));strict,sg,candidate,cg,a=complete_pin_output_names(p,g,r,CFG)
        self.assertIsNone(strict['pin_candidates'][0]['pinname'])
        self.assertEqual(candidate['pin_candidates'][0]['pinname'],'1')
        self.assertTrue(candidate['pin_candidates'][0]['text_evidence']['pinname']['provisional'])
        self.assertEqual(candidate['pin_candidates'][0]['number'],'1')
        self.assertEqual((p,g,r),before);self.assertEqual(g['edges'],cg['edges']);self.assertEqual(g['networks'],cg['networks'])
        self.assertFalse(a['summary']['recognition_accuracy_gain_claimed'])

    def test_package_code_or_out_of_range_reading_never_becomes_passive_name(self):
        for number in ['0402','10','A5','01']:
            p,g,r=example(number=number);_,_,c,_,_=complete_pin_output_names(p,g,r,CFG)
            self.assertIsNone(c['pin_candidates'][0]['pinname'])

    def test_unobserved_numbers_are_not_spatially_guessed(self):
        p,g,r=example(number=None);p['pin_candidates'][0]['text_evidence']={}
        _,_,c,_,_=complete_pin_output_names(p,g,r,CFG)
        self.assertIsNone(c['pin_candidates'][0]['number']);self.assertIsNone(c['pin_candidates'][0]['pinname'])

    def test_IC_diode_and_polarity_symbols_do_not_use_passive_convention(self):
        for kind in ['ic','diode','led','mosfet','gnd','vcc','switch']:
            p,g,r=example(kind=kind);_,_,c,_,_=complete_pin_output_names(p,g,r,CFG)
            self.assertIsNone(c['pin_candidates'][0]['pinname'])

    def test_distinct_views_must_read_same_position_and_literal(self):
        for change in ['duplicate','far','split','wrong','weak']:
            p,g,r=example();alt=p['pin_candidates'][0]['text_evidence']['number']['alternatives'][0]
            if change=='duplicate':alt['source_view']='global'
            elif change=='far':alt['bbox']=[90,20,98,29]
            elif change=='split':alt['approximate_split_geometry']=True
            elif change=='wrong':alt['source_text']='R1'
            else:alt['confidence']=.97
            _,_,c,_,_=complete_pin_output_names(p,g,r,CFG);self.assertIsNone(c['pin_candidates'][0]['pinname'])

    def test_existing_name_and_ambiguous_name_candidates_are_preserved(self):
        p,g,r=example();p['pin_candidates'][0]['pinname']='KNOWN'
        _,_,c,_,_=complete_pin_output_names(p,g,r,CFG);self.assertEqual(c['pin_candidates'][0]['pinname'],'KNOWN')
        p['pin_candidates'][0]['pinname']=None;p['pin_candidates'][0]['text_candidates']={'pinname':[{'text':'NC'}]}
        _,_,c,_,_=complete_pin_output_names(p,g,r,CFG);self.assertIsNone(c['pin_candidates'][0]['pinname'])

    def test_unique_observed_label_name_follows_PDF_example(self):
        p,g,r=example(kind='vbus_in');r[0].update(component_id='RUN',designator='RUN',Name='RUN')
        p['components'][0]['component_id']='RUN';p['pin_candidates'][0]['owner']='RUN'
        g['label_sites']=[{'detection_index':0,'candidate_id':'P0','site_id':'S','label':'RUN','confidence':.99,
                          'distinct_view_votes':{'RUN':3},'status':'bound_ordinary_label','evidence_kind':'detected_symbol_and_owned_pin'}]
        strict,_,candidate,_,a=complete_pin_output_names(p,g,r,CFG)
        self.assertEqual(strict['pin_candidates'][0]['pinname'],'RUN');self.assertEqual(a['summary']['ordinary_label_names_added'],1)
        self.assertFalse(strict['pin_candidates'][0]['text_evidence']['pinname']['OCR_recognition_improved'])

    def test_label_roles_or_identity_spelling_conflicts_cannot_create_name(self):
        for status,designator in [('power_semantics_pending','RUN'),('bound_ordinary_label',None),('bound_ordinary_label','RUN_1')]:
            p,g,r=example(kind='vbus_in');r[0].update(designator=designator,Name='RUN')
            g['label_sites']=[{'detection_index':0,'candidate_id':'P0','site_id':'S','label':'RUN','confidence':.99,
                              'distinct_view_votes':{'RUN':3},'status':status,'evidence_kind':'detected_symbol_and_owned_pin'}]
            strict,_,_,_,_=complete_pin_output_names(p,g,r,CFG);self.assertIsNone(strict['pin_candidates'][0]['pinname'])

    def test_missing_power_tip_blocks_adjacent_wire_without_inventing_pin(self):
        p,g,r=example();r.append({'detection_index':1,'component_id':'UNRESOLVED_2','designator':None,'type':'vcc',
                                 'bbox_top_left':[76,20,88,36]})
        guard=power_network_guard(p,g,r,CFG)
        self.assertEqual(guard['summary']['power_without_owned_tip'],1)
        self.assertEqual(guard['blocks'][0]['network_id'],'N')
        self.assertEqual(guard['summary']['pin_points_created'],0)
        out,cov=apply_power_export_gate({'nets':{'N':{}}},{'withheld_networks':[]},guard)
        self.assertFalse(out['nets']);self.assertEqual(cov['power_guard_excluded_network_ids'],['N'])

    def test_distant_power_symbol_does_not_block_unrelated_network(self):
        p,g,r=example();r.append({'detection_index':1,'component_id':'GND','designator':None,'type':'gnd',
                                 'bbox_top_left':[150,80,160,90]})
        self.assertFalse(power_network_guard(p,g,r,CFG)['blocks'])

    def test_represented_power_tip_does_not_create_duplicate_omission(self):
        p,g,r=example(kind='gnd');self.assertFalse(power_network_guard(p,g,r,CFG)['blocks'])

    def test_name_completion_does_not_make_partial_net_pass(self):
        p,g,r=example();_,_,c,cg,_=complete_pin_output_names(p,g,r,CFG)
        out,cov=logical_partial_export(c,cg,r)
        self.assertFalse(out['nets']);self.assertEqual(cov['network_reason_counts']['single_pin_fragment'],1)

    def test_mismatched_tip_input_rejected(self):
        p,g,r=example();g['pin_attachments'][0]['point']=[11,40]
        with self.assertRaises(ValueError):complete_pin_output_names(p,g,r,CFG)

    def test_unknown_second_member_is_not_removed_to_export_a_net(self):
        p,g,r=example();second=deepcopy(p['pin_candidates'][0]);second.update(candidate_id='P1',point=[40,40],number=None)
        p['pin_candidates'].append(second);g['pin_attachments'].append({'candidate_id':'P1','point':[40,40]})
        g['networks'][0]['pin_candidate_ids'].append('P1')
        _,_,c,cg,_=complete_pin_output_names(p,g,r,CFG);out,cov=logical_partial_export(c,cg,r)
        self.assertFalse(out['nets']);self.assertEqual(cg['networks'][0]['pin_candidate_ids'],['P0','P1'])

    def test_edge_compression_preserves_corners_endpoints_and_deviation_bound(self):
        line=[[0,0],[1,.2],[2,0],[3,0],[3,1],[3,2],[3,3]]
        result,dev=simplify_model_polyline(line,.3)
        self.assertEqual(result,[[0,0],[3,0],[3,3]]);self.assertLessEqual(dev,.3)
        for invalid in [2,-1,float('nan')]:
            with self.assertRaises(ValueError):simplify_model_polyline(line,invalid)

    def test_compression_keeps_separate_paths_and_flywire_zero_geometry(self):
        p,g,r=example();g['edges'][0]['polyline']=[[10,40],[20,40],[30,40]]
        g['edges'].append({'edge_id':'E2','polyline':[[70,40],[80,40],[90,40]]});g['networks'][0]['edge_ids'].append('E2')
        inp={'nets':{'N':{'hyperGraph':'(R1.1,R1.2)','edges':{}}}};before=deepcopy((g,inp))
        out,a=compress_candidate_edges(inp,g,100,1)
        self.assertEqual(a['output_segments'],2);self.assertEqual((g,inp),before)
        self.assertEqual(out['nets']['N']['edges']['edge_1'],[{'x':10,'y':60},{'x':30,'y':60}])
        g['networks'][0]['flywire']=True;inp['nets']['N']['edges']={'edge_1':[{'x':10,'y':60},{'x':10,'y':60}]}
        out,a=compress_candidate_edges(inp,g,100,1);self.assertEqual(out,inp);self.assertEqual(a['output_segments'],0)


if __name__=='__main__':unittest.main()
