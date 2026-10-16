"""Power-domain separation, model-anchor evidence, and conservative topology."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.power_networks import propose_power_networks, qualified_name

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
CFG=read(ROOT/'configs/power_networks_v1.json')
GUARD=read(ROOT/'configs/pin_output_rules_v1.json')


def fixture(labels=('GND','GND'),kind='gnd'):
    pins={'image_sha256':'image','size':[360,140],'pin_candidates':[],'components':[]}
    graph={'image_sha256':'image','size':pins['size'],'coordinate_origin':'top_left','networks':[],
           'pin_attachments':[],'edges':[],'gap_candidates':[],'label_sites':[],'summary':{}}
    records=[];texts=[]
    for i,label in enumerate(labels):
        x=30+i*130;cid=f'P{i}';nid=f'N{i}';owner=f'UNRESOLVED_{i}';box=[x-10,45,x+10,58];text_box=[x-12,60,x+12,72]
        record={'component_id':owner,'detection_index':i,'type':kind,'bbox_top_left':box,'designator':None}
        records.append(record);pins['components'].append(record)
        pin={'candidate_id':cid,'owner':owner,'owner_detection_index':i,'owner_status':'assigned','point':[x,44],
             'number':None,'pinname':None,'confidence':.99,'text_evidence':{}}
        pins['pin_candidates'].append(pin)
        graph['edges'].append({'edge_id':f'E{i}','polyline':[[x,20],[x,44]]})
        graph['networks'].append({'network_id':nid,'node_ids':[f'A{i}',f'B{i}'],'edge_ids':[f'E{i}'],'pin_candidate_ids':[cid],
                                  'official_pin_references':[],'unresolved_junction_ids':[],'flywire':False,
                                  'logical_label_names':[],'logical_label_sites':[],'physical_network_ids':[nid]})
        graph['pin_attachments'].append({**pin,'attachment_status':'attached','network_id':nid,'alternatives':[{'network_id':nid}]})
        graph['label_sites'].append({'site_id':f'S{i}','image_sha256':'image','component_type':kind,'evidence_kind':'detected_symbol_and_owned_pin',
                                    'detection_index':i,'candidate_id':cid,'label':label,'point':pin['point'],'bbox':text_box,'confidence':.999})
        texts.extend({'text_id':f'T{i}_{j}','text':label,'bbox':text_box,'confidence':.999,'view':view}
                     for j,view in enumerate(['global','tile_r0','tile_r90']))
    return pins,graph,records,texts


def run(p,g,r,t):return propose_power_networks(p,g,r,t,CFG,GUARD)


class PowerNetworkTests(unittest.TestCase):
    def test_exact_named_power_group_keeps_all_pins_and_physical_geometry(self):
        p,g,r,t=fixture();original=deepcopy((p,g,r,t));new,a=run(p,g,r,t)
        self.assertEqual(a['summary']['power_groups_applied'],1)
        self.assertEqual(len(new['networks']),1);self.assertEqual(new['networks'][0]['pin_candidate_ids'],['P0','P1'])
        self.assertEqual(new['edges'],g['edges']);self.assertFalse(new['networks'][0]['flywire'])
        self.assertEqual((p,g,r,t),original);self.assertEqual([p['point'] for p in new['pin_attachments']],[[30,44],[160,44]])
        self.assertTrue(all(p['number'] is None and p['pinname'] is None for p in new['pin_attachments']))

    def test_different_ground_domains_and_literal_spellings_are_not_aliased(self):
        for labels in [('GND','DGND'),('GND','AGND'),('GND','gnd'),('GND_ISO','GND')]:
            p,g,r,t=fixture(labels);new,a=run(p,g,r,t)
            self.assertEqual(len(new['networks']),2);self.assertEqual(a['summary']['power_groups_applied'],0)
        p,g,r,t=fixture(('VCC_3.3','3V3'),'vcc');new,a=run(p,g,r,t);self.assertEqual(len(new['networks']),2)

    def test_supply_name_supports_IO_suffix_without_accepting_caption_fragments(self):
        for name in ['VDDIO_EXT','VCC5V0_SYS','PVDD','+1V8','3.3V']:
            self.assertTrue(qualified_name(name,'vcc',CFG))
        for name in ['F','cc','GPIO','GND']:
            self.assertFalse(qualified_name(name,'vcc',CFG))

    def test_repeat_reads_in_same_view_are_not_independent_support(self):
        p,g,r,t=fixture()
        for row in t:row['view']='global'
        new,a=run(p,g,r,t);self.assertEqual(a['summary']['observed_power_sites'],0)

    def test_low_confidence_or_distant_reads_do_not_supply_votes(self):
        for failure in ['weak','far']:
            p,g,r,t=fixture()
            if failure=='weak':t[2]['confidence']=.95
            else:t[2]['bbox']=[300,100,330,120]
            new,a=run(p,g,r,t);self.assertEqual(a['summary']['power_groups_applied'],0)

    def test_conflicting_supported_caption_is_held(self):
        p,g,r,t=fixture();t.extend([{**row,'text':'DGND'} for row in t[:2]])
        new,a=run(p,g,r,t);self.assertEqual(a['summary']['power_groups_applied'],0)

    def test_uncertain_junction_or_gap_prevents_power_join(self):
        for failure in ['junction','gap','flywire']:
            p,g,r,t=fixture()
            if failure=='junction':g['networks'][0]['unresolved_junction_ids']=['J']
            elif failure=='gap':g['gap_candidates']=[{'network_ids':['N0','N1']}]
            else:g['networks'][0]['flywire']=True
            new,a=run(p,g,r,t);self.assertEqual(a['summary']['power_groups_applied'],0)

    def test_missing_power_tip_near_member_wire_blocks_join(self):
        p,g,r,t=fixture();r.append({'component_id':'MISSING','detection_index':2,'type':'vcc','designator':None,'bbox_top_left':[25,20,35,30]})
        new,a=run(p,g,r,t);self.assertEqual(a['summary']['power_groups_applied'],0)
        self.assertIn('unrepresented_power_symbol_near_member_wire',a['decisions'][0]['withheld_reasons'])

    def test_multiple_learned_tips_and_unattached_tips_stay_held(self):
        for failure in ['extra','unattached']:
            p,g,r,t=fixture()
            if failure=='extra':
                extra={**p['pin_candidates'][0],'candidate_id':'EXTRA','point':[31,44]};p['pin_candidates'].append(extra)
                g['pin_attachments'].append({**extra,'attachment_status':'attached','network_id':'N0','alternatives':[]})
            else:g['pin_attachments'][0]['attachment_status']='ambiguous'
            new,a=run(p,g,r,t);self.assertEqual(a['summary']['power_groups_applied'],0)

    def test_cross_image_or_changed_tip_anchor_is_rejected(self):
        for failure in ['image','point','owner']:
            p,g,r,t=fixture()
            if failure=='image':g['label_sites'][0]['image_sha256']='another'
            elif failure=='point':g['label_sites'][0]['point']=[30,45]
            else:p['pin_candidates'][0]['owner_detection_index']=1
            with self.assertRaises(ValueError):run(p,g,r,t)

    def test_all_other_members_of_a_merged_network_are_retained(self):
        p,g,r,t=fixture();g['networks'][0]['pin_candidate_ids'].append('IC_PIN')
        p['pin_candidates'].append({'candidate_id':'IC_PIN','owner':'U1','owner_status':'assigned','owner_detection_index':3,'point':[30,20],'number':'5','pinname':'VDD'})
        g['pin_attachments'].append({**p['pin_candidates'][-1],'attachment_status':'attached','network_id':'N0','alternatives':[]})
        new,a=run(p,g,r,t);self.assertEqual(new['networks'][0]['pin_candidate_ids'],['IC_PIN','P0','P1'])

    def test_second_application_and_policy_that_edits_fields_are_rejected(self):
        p,g,r,t=fixture();new,a=run(p,g,r,t)
        with self.assertRaises(ValueError):run(p,new,r,t)
        with self.assertRaises(ValueError):propose_power_networks(p,g,r,t,{**CFG,'change_pin_fields':True},GUARD)


if __name__=='__main__':unittest.main()
