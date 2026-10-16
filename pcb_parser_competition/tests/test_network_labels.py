"""Electrical safety invariants and difficult OCR/geometry contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pcb_parser.network_labels import merge_ordinary_labels,observations,power_name,bind_network_labels,symbol_matches
from run_network_labels import logical_partial_export,validate_inputs

CFG=json.loads((ROOT/'configs/network_labels_v1.json').read_text(encoding='utf-8'))


def graph():
    return {'size':[160,100],'coordinate_origin':'top_left','image_sha256':'image','pin_input_sha256':'pins',
            'nodes':[{'node_id':f'N{i}','point':[20+80*i,50],'kind':'endpoint'} for i in range(2)],
            'edges':[{'edge_id':f'E{i}','source':f'N{i}','target':f'N{i}','polyline':[[20+80*i,50],[30+80*i,50]],'length_px':10} for i in range(2)],
            'networks':[{'network_id':f'NET{i}','node_ids':[f'N{i}'],'edge_ids':[f'E{i}'],'pin_candidate_ids':[f'P{i}'],
                         'official_pin_references':[],'unresolved_junction_ids':[],'status':'physical'} for i in range(2)],
            'pin_attachments':[{'candidate_id':f'P{i}','point':[20+80*i,50],'network_id':f'NET{i}',
                                'attachment_status':'attached','alternatives':[{'network_id':f'NET{i}'}]} for i in range(2)],
            'gap_candidates':[],'summary':{}}


def site(index,label='RUN',kind='detected_symbol_and_owned_pin'):
    return {'site_id':f'S{index}','label':label,'network_id':f'NET{index}','network_ids':[f'NET{index}'],
            'point':[20+80*index,50],'merge_eligible':True,'evidence_kind':kind,
            'status':'bound_ordinary_label','candidate_id':f'P{index}'}


def ocr(text,box,view='global',confidence=.99,index=0):
    return {'text_id':f'T{index}','text':text,'bbox':box,'confidence':confidence,'view':view,'observations':[]}


class NetworkLabelTests(unittest.TestCase):
    def test_two_ordinary_symbol_sites_merge_without_remote_wire_or_tip_changes(self):
        source=graph();before=deepcopy(source);result=merge_ordinary_labels(source,[site(0),site(1)],CFG)
        self.assertEqual(len(result['networks']),1);self.assertEqual(result['edges'],source['edges'])
        self.assertEqual([p['point'] for p in result['pin_attachments']],[p['point'] for p in source['pin_attachments']])
        self.assertEqual(source,before);self.assertTrue(result['networks'][0]['flywire'])

    def test_one_occurrence_is_not_a_flywire_even_with_two_contact_nets(self):
        row=site(0);row['network_ids']=['NET0','NET1']
        result=merge_ordinary_labels(graph(),[row],CFG)
        self.assertEqual(len(result['networks']),2)
        self.assertIn('single_label_occurrence',result['label_merge_decisions'][0]['withheld_reasons'])

    def test_differential_explanation_text_cannot_merge_endpoint_only_nets(self):
        rows=[site(i,'Differential','model_wire_endpoint_next_to_OCR') for i in range(2)]
        result=merge_ordinary_labels(graph(),rows,CFG)
        self.assertEqual(len(result['networks']),2)
        self.assertIn('label_symbol_evidence_missing',result['label_merge_decisions'][0]['withheld_reasons'])

    def test_power_names_never_enter_ordinary_merge(self):
        for label in ['GND','DGND','VCC 3.3','PLL_VDD','+5V','3V3','VIN','VCC5V0_SYS']:
            self.assertTrue(power_name(label,CFG),label)
        rows=[site(i,'GND') for i in range(2)]
        # Public planner must protect against a caller mislabeling a power site.
        result=merge_ordinary_labels(graph(),rows,CFG)
        self.assertEqual(len(result['networks']),2)

    def test_case_space_and_underscore_names_remain_distinct(self):
        for a,b in [('RUN','run'),('ADC_IN12','ADC IN12'),('SCL','SDA')]:
            result=merge_ordinary_labels(graph(),[site(0,a),site(1,b)],CFG)
            self.assertEqual(len(result['networks']),2)

    def test_uncertain_junction_and_gap_stop_merge(self):
        for kind in ['junction','gap']:
            source=graph()
            if kind=='junction':source['networks'][0]['unresolved_junction_ids']=['J1']
            else:source['gap_candidates']=[{'network_ids':['NET0','NET1']}]
            result=merge_ordinary_labels(source,[site(0),site(1)],CFG)
            self.assertEqual(len(result['networks']),2)

    def test_disable_merge_switch_preserves_membership(self):
        result=merge_ordinary_labels(graph(),[site(0),site(1)],{**CFG,'enable_ordinary_label_merges':False})
        self.assertEqual(len(result['networks']),2)

    def test_conflicting_strong_labels_on_one_net_stop_merge(self):
        extra=site(0,'OTHER');extra['site_id']='OTHER0'
        result=merge_ordinary_labels(graph(),[site(0),site(1),extra],CFG)
        self.assertEqual(len(result['networks']),2)

    def test_weak_pin_text_does_not_override_strong_symbol_label(self):
        extra=site(0,'PB11','model_wire_endpoint_next_to_OCR');extra['site_id']='PB11'
        result=merge_ordinary_labels(graph(),[site(0),site(1),extra],CFG)
        self.assertEqual(len(result['networks']),1)

    def test_repeated_views_are_one_site_but_remote_same_words_are_two(self):
        rows=[ocr('RUN',[10,10,30,20],'global'),ocr('RUN',[10,10,30,20],'tile',index=1),ocr('RUN',[100,10,120,20],index=2)]
        groups=observations(rows,[160,100]);self.assertEqual(len(groups),2)
        self.assertEqual(len(groups[0]),2)

    def test_partial_word_does_not_create_separate_label_site(self):
        rows=[ocr('IIC SCL',[10,10,65,25]),ocr('IIC',[10,10,32,25],'tile',index=1)]
        self.assertEqual(len(observations(rows,[160,100])),1)

    def test_invalid_ocr_and_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError):observations([ocr('RUN',[-1,0,20,20])],[160,100])
        with self.assertRaises(ValueError):observations([ocr('RUN',[0,0,20,20],confidence=float('nan'))],[160,100])
        with self.assertRaises(ValueError):validate_inputs('different',[160,100],graph(),{}, {},{})

    def test_cross_image_and_duplicate_site_ids_rejected(self):
        rows=[site(0),site(1)];rows[1]['image_sha256']='different'
        with self.assertRaises(ValueError):merge_ordinary_labels(graph(),rows,CFG)
        rows=[site(0),site(1)];rows[1]['site_id']=rows[0]['site_id']
        with self.assertRaises(ValueError):merge_ordinary_labels(graph(),rows,CFG)

    def test_mismatched_pin_coordinate_cannot_use_cached_graph(self):
        source=graph();pins={'image_sha256':'image','size':[160,100],'coordinate_origin':'top_left',
                             'pin_candidates':[{'candidate_id':f'P{i}','point':[21+80*i,50]} for i in range(2)],'components':[]}
        ocr_payload={'image_sha256':'image','size':[160,100],'coordinate_origin':'top_left','prediction_only':True}
        records={'image_sha256':'image','records':[]}
        with self.assertRaises(ValueError):validate_inputs('image',[160,100],source,pins,ocr_payload,records)

    def test_strict_flywire_export_has_zero_length_edges_and_complete_refs(self):
        source=graph();logical=merge_ordinary_labels(source,[site(0),site(1)],CFG)
        pins={'size':[160,100],'pin_candidates':[{'candidate_id':f'P{i}','owner':f'RUN_{i+1}',
               'owner_status':'assigned','number':'1','pinname':'RUN','point':[20+80*i,50]} for i in range(2)]}
        records=[{'component_id':f'RUN_{i+1}','designator':f'RUN_{i+1}','bbox_top_left':[15+80*i,40,30+80*i,60],
                  'type':'vbus_in','Name':None,'value':None} for i in range(2)]
        output,coverage=logical_partial_export(pins,logical,records)
        self.assertEqual(coverage['exported_nets'],1)
        net=next(iter(output['nets'].values()));self.assertEqual(net['hyperGraph'],'(RUN_1.1,RUN_2.1)')
        self.assertTrue(all(edge[0]==edge[1] for edge in net['edges'].values()))

    def test_missing_label_pin_blocks_complete_flywire_export(self):
        rows=[site(0),site(1)];rows[1]['candidate_id']=None
        logical=merge_ordinary_labels(graph(),rows,CFG)
        pins={'size':[160,100],'pin_candidates':[{'candidate_id':f'P{i}','owner':f'U{i}',
               'owner_status':'assigned','number':'1','pinname':'X','point':[20+80*i,50]} for i in range(2)]}
        records=[{'component_id':f'U{i}','designator':f'U{i}','bbox_top_left':[15+80*i,40,30+80*i,60],
                  'type':'ic','Name':'X','value':None} for i in range(2)]
        output,coverage=logical_partial_export(pins,logical,records)
        self.assertFalse(output['nets']);self.assertEqual(coverage['exported_nets'],0)


if __name__=='__main__':unittest.main()
