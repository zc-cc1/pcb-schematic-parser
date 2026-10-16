"""Canonical-key, electrical-reference and field-evidence regression tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pcb_parser.submission_fields import complete_fields,network_field_queue
from run_network_labels import logical_partial_export

CFG=json.loads((ROOT/'configs/submission_fields_v1.json').read_text(encoding='utf-8'))


def inputs(labels=('SIG_A','SIG_B'),types=('vbus_out','vbus_out')):
    records=[];components=[];pins=[];sites=[];attachments=[]
    for i,(label,kind) in enumerate(zip(labels,types)):
        cid=f'UNRESOLVED_DET_{i+1:04d}';point=[20+80*i,50];bb=[15+80*i,40,30+80*i,60]
        records.append({'detection_index':i,'component_id':cid,'type':kind,'designator':None,'Name':None,'value':None,'bbox_top_left':bb,'fields':{}})
        components.append({'detection_index':i,'component_id':cid,'type':kind,'class_name':kind,'bbox_top_left':bb})
        pins.append({'candidate_id':f'P{i}','point':point,'owner':cid,'owner_status':'assigned','owner_detection_index':i,
                     'number':None,'pinname':None,'text_candidates':{},'text_evidence':{},'number_status':'no_reliable_text'})
        sites.append({'site_id':f'S{i}','label':label,'confidence':.99,'bbox':bb,'component_id':cid,'detection_index':i,
                      'component_type':kind,'candidate_id':f'P{i}','status':'bound_ordinary_label',
                      'evidence_kind':'detected_symbol_and_owned_pin','distinct_view_votes':{label:3}})
        attachments.append({'candidate_id':f'P{i}','point':point,'owner':cid,'owner_status':'assigned'})
    graph={'size':[180,100],'summary':{'logical_label_merges':0},'edges':[{'edge_id':'E','polyline':[[20,50],[100,50]]}],
           'pin_attachments':attachments,'label_sites':sites,'gap_candidates':[],
           'networks':[{'network_id':'N','pin_candidate_ids':['P0','P1'],'unresolved_junction_ids':[],'edge_ids':['E']}]}
    result={'size':[180,100],'components':components,'pin_candidates':pins}
    return result,graph,records


class SubmissionFieldTests(unittest.TestCase):
    def test_unique_symbol_identity_and_single_tip_number_use_distinct_provenance(self):
        pins,graph,records=inputs();before=deepcopy((pins,graph,records))
        updated,g,r,proposals=complete_fields(pins,graph,records,CFG)
        self.assertEqual(proposals['summary']['new_label_identities'],2)
        self.assertEqual([p['owner'] for p in updated['pin_candidates']],['SIG_A','SIG_B'])
        self.assertEqual([p['number'] for p in updated['pin_candidates']],['1','1'])
        self.assertFalse(updated['pin_candidates'][0]['text_evidence']['number']['observed_number_glyph'])
        self.assertEqual([p['pinname'] for p in updated['pin_candidates']],[None,None])
        self.assertEqual(before,(pins,graph,records));self.assertEqual(g['edges'],graph['edges'])
        self.assertEqual(g['networks'],graph['networks'])

    def test_format_number_does_not_unlock_missing_pinname_or_network(self):
        pins,graph,records=inputs();updated,g,r,_=complete_fields(pins,graph,records,CFG)
        output,coverage=logical_partial_export(updated,g,r)
        self.assertEqual(len(output['components']),2);self.assertFalse(output['pins']);self.assertFalse(output['nets'])
        self.assertEqual(coverage['pin_reason_counts'],{'pin_name_unresolved':2})

    def test_repeated_label_suffix_proposals_never_enter_strict_components(self):
        pins,graph,records=inputs(('RUN','RUN'));updated,g,r,proposals=complete_fields(pins,graph,records,CFG)
        self.assertEqual([p['proposed_component_key'] for p in proposals['components']],['RUN_1','RUN_2'])
        self.assertFalse(any(p['new_identity_accepted'] for p in proposals['components']))
        self.assertTrue(all(p['designator'] is None for p in r))
        output,_=logical_partial_export(updated,g,r);self.assertFalse(output['components'])

    def test_weak_duplicate_cannot_make_other_site_appear_unique(self):
        pins,graph,records=inputs(('RUN','RUN'));graph['label_sites'][0]['distinct_view_votes']={'RUN':1}
        _,_,_,proposals=complete_fields(pins,graph,records,CFG)
        self.assertTrue(all('duplicate_suffix_source_component_order_unobservable' in p['withheld_reasons'] for p in proposals['components']))

    def test_space_underscore_alternatives_block_identity_without_merging(self):
        pins,graph,records=inputs(('ADC IN11','ADC_IN11'))
        updated,g,r,proposals=complete_fields(pins,graph,records,CFG)
        self.assertTrue(all('similar_label_spelling_may_hide_duplicate_identity' in p['withheld_reasons'] for p in proposals['components']))
        self.assertEqual([p['raw_label_name'] for p in proposals['components']],['ADC IN11','ADC_IN11'])
        self.assertEqual(g['networks'],graph['networks']);self.assertTrue(all(c['designator'] is None for c in r))

    def test_two_tip_and_passive_or_IC_numbers_are_never_ordered(self):
        for kind in ['net_short','ic','resistor']:
            pins,graph,records=inputs(types=(kind,kind))
            if kind!='net_short':graph['label_sites']=[]
            updated,_,_,_=complete_fields(pins,graph,records,CFG)
            self.assertTrue(all(p['number'] is None for p in updated['pin_candidates']))

    def test_existing_observed_number_is_preserved(self):
        pins,graph,records=inputs();pins['pin_candidates'][0]['number']='A5'
        updated,_,_,_=complete_fields(pins,graph,records,CFG)
        self.assertEqual(updated['pin_candidates'][0]['number'],'A5')

    def test_conflicting_number_evidence_prevents_auto_number(self):
        pins,graph,records=inputs();pins['pin_candidates'][0]['text_candidates']['number']=[{'text':'2'},{'text':'3'}]
        updated,_,_,_=complete_fields(pins,graph,records,CFG)
        self.assertIsNone(updated['pin_candidates'][0]['number'])

    def test_component_designator_collision_is_withheld(self):
        pins,graph,records=inputs();records[1].update(designator='SIG_A',component_id='SIG_A')
        pins['components'][1]['component_id']='SIG_A';pins['pin_candidates'][1]['owner']='SIG_A';graph['label_sites'][1]['component_id']='SIG_A'
        _,_,_,proposals=complete_fields(pins,graph,records,CFG)
        self.assertIn('label_key_collides_with_recognized_designator',proposals['components'][0]['withheld_reasons'])

    def test_decimal_in_label_cannot_create_ambiguous_hypergraph_reference(self):
        pins,graph,records=inputs(('SIG.1','SIG_B'));_,_,_,proposals=complete_fields(pins,graph,records,CFG)
        self.assertFalse(proposals['components'][0]['new_identity_accepted'])

    def test_power_and_endpoint_text_cannot_create_label_identity(self):
        for status in ['power_semantics_pending','endpoint_text_role_unverified']:
            pins,graph,records=inputs(('GND','VCC'))
            for s in graph['label_sites']:s['status']=status
            _,_,_,proposals=complete_fields(pins,graph,records,CFG)
            self.assertEqual(proposals['summary']['new_label_identities'],0)

    def test_mismatch_and_duplicate_network_membership_rejected(self):
        for mode in ['point','owner','net']:
            pins,graph,records=inputs()
            if mode=='point':graph['pin_attachments'][0]['point']=[21,50]
            elif mode=='owner':pins['pin_candidates'][0]['owner']='wrong'
            else:graph['networks'].append({**graph['networks'][0],'network_id':'OTHER'})
            with self.assertRaises(ValueError):complete_fields(pins,graph,records,CFG)

    def test_queue_distinguishes_field_and_topology_blockers(self):
        pins,graph,records=inputs();graph['networks'][0]['unresolved_junction_ids']=['J']
        updated,g,r,_=complete_fields(pins,graph,records,CFG);_,coverage=logical_partial_export(updated,g,r)
        queue=network_field_queue(updated,g,r,coverage)
        self.assertEqual(queue['networks'][0]['missing_field_counts'],{'pin_name_unresolved':2})
        self.assertEqual(queue['networks'][0]['topology_blockers'],['junction_semantics_unresolved'])
        self.assertEqual(queue['topology_clear_but_field_incomplete'],0)

    def test_queue_keeps_power_and_omitted_member_export_blockers(self):
        p,g,r=inputs();updated,graph,records,_=complete_fields(p,g,r,CFG)
        _,coverage=logical_partial_export(updated,graph,records)
        graph['networks'][0]['power_logical_merge']=True
        coverage['withheld_networks'].append({'network_id':'N','reasons':['detected_power_symbol_near_wire_without_attached_member']})
        row=network_field_queue(updated,graph,records,coverage)['networks'][0]
        self.assertEqual(row['topology_blockers'],[])
        self.assertIn('power_logical_membership_provisional',row['non_field_export_blockers'])
        self.assertIn('detected_power_symbol_near_wire_without_attached_member',row['non_field_export_blockers'])
        self.assertFalse(row['field_only_candidate'])

    def test_identity_tasks_for_same_component_are_shared_across_its_pins(self):
        p,g,r=inputs();p['pin_candidates'][1].update(owner=p['pin_candidates'][0]['owner'],owner_detection_index=0)
        coverage={'pending_pins':[{'candidate_id':p['candidate_id'],'reasons':['component_identity_unresolved']} for p in p['pin_candidates']],
                  'withheld_networks':[],'exported_network_ids':[]}
        row=network_field_queue(p,g,r,coverage)['networks'][0]
        self.assertEqual(row['missing_field_total'],2)
        self.assertEqual(row['completion_task_count'],1)
        self.assertEqual(row['completion_tasks'][0]['affected_candidate_ids'],['P0','P1'])

    def test_field_only_network_ranks_ahead_of_power_proposal_with_fewer_fields(self):
        p,g,r=inputs();g['networks'][0]['power_logical_merge']=True
        g['networks'].append({**g['networks'][0],'network_id':'CLEAR','power_logical_merge':False})
        coverage={'pending_pins':[{'candidate_id':p['candidate_id'],'reasons':['pin_name_unresolved']} for p in p['pin_candidates']],
                  'withheld_networks':[],'exported_network_ids':[]}
        queue=network_field_queue(p,g,r,coverage)
        self.assertEqual(queue['networks'][0]['network_id'],'CLEAR')
        self.assertEqual(queue['field_only_incomplete_networks'],1)


if __name__=='__main__':unittest.main()
