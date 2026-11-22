"""Synthetic atomic rekey/network tests; no evaluation images or OCR weights."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser import pin_pairing as pairing


def fixture():
    rgb=np.full((240,220,3),255,np.uint8)
    cv2.rectangle(rgb,(50,40),(150,160),(150,0,0),1)
    for y in [80,100,120]:cv2.line(rgb,(30,y),(50,y),(150,0,0),1)
    token={'token_id':'SHARED','text':'3','confidence':.999,'bbox':[40,105,48,113]}
    other={'candidate_id':'ROW_A','owner':'X','owner_status':'assigned','point':[35.,100.],
           'number':'3','pinname':'OUT','text_candidates':{'number':[copy.deepcopy(token)]}}
    missing={'candidate_id':'ROW_B','owner':'X','owner_status':'assigned','point':[35.,120.],
             'number':None,'pinname':'IN','number_status':'text_used_by_other_pin',
             'text_candidates':{'number':[copy.deepcopy(token)]}}
    document={'pin_candidates':[other,missing]}
    target={'components':{'X':{'type':'box','bbox':[50,80,150,200]},'Y':{'type':'c','bbox':[10,10,20,20]}},
            'pins':{'X':{'pin_1':{'pinname':'GND','point':{'x':30.,'y':160.}},
                         'pin_3':{'pinname':'OUT','point':{'x':30.,'y':140.}}}},
            'nets':{'N_OUT':{'hyperGraph':'(X.3,Y.1)','edges':{'e':[{'x':10,'y':140},{'x':30,'y':140}]}}}}
    graph={'networks':[{'network_id':'N_OUT','pin_candidate_ids':['ROW_A'],'edge_ids':[],'unresolved_junction_ids':[]},
                       {'network_id':'N_IN','pin_candidate_ids':['ROW_B'],'edge_ids':[],'unresolved_junction_ids':[]}],
           'pin_attachments':[{'candidate_id':p['candidate_id'],'point':p['point'].copy(),'attachment_status':'attached',
                               'network_id':nid,'physical_network_id':'PHYSICAL_IN' if nid=='N_IN' else 'PHYSICAL_OUT'}
                              for p,nid in [(other,'N_OUT'),(missing,'N_IN')]],'physical_networks':[]}
    records=[{'component_id':'X','bbox_top_left':[50,40,150,160]}]
    def reader(image,box,scale,rotation,recognition_only,view):
        if recognition_only:
            return [{'text':'2' if 'ROW_A' in view else '3','confidence':.999,'bbox':box,'view':view}]
        return [{'text':text,'confidence':.999,'bbox':bbox,'view':view} for text,bbox in
                [('OUT',[60,95,85,105]),('IN',[60,115,80,125])]]
    return target,document,graph,records,rgb,reader


def no_power(target,*args):return copy.deepcopy(target),{'restored_networks':[]}


class PairingTests(unittest.TestCase):
    def run_fixture(self,data=None,reader=None,**kwargs):
        target,document,graph,records,rgb,default_reader=data or fixture()
        with patch.object(pairing,'restore_power_wires',side_effect=no_power):
            return pairing.reconcile_pin_pairs(target,document,graph,records,rgb,reader=reader or default_reader,**kwargs)

    def test_complete_rekey_preserves_old_geometry_and_removes_stale_reference(self):
        data=fixture();before=copy.deepcopy(data[:4]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:4],before)
        self.assertEqual(len(audit['changes']),2)
        self.assertEqual(out['pins']['X']['pin_2'],data[0]['pins']['X']['pin_3'])
        self.assertEqual(out['pins']['X']['pin_3']['pinname'],'IN')
        self.assertEqual(out['pins']['X']['pin_3']['point']['y'],120.)
        self.assertEqual(out['nets']['N_OUT']['hyperGraph'],'(X.2,Y.1)')
        self.assertEqual(out['nets']['N_OUT']['edges'],data[0]['nets']['N_OUT']['edges'])
        self.assertEqual(out['components'],data[0]['components'])
        self.assertEqual(out['pins']['X']['pin_1'],data[0]['pins']['X']['pin_1'])

    def test_second_run_does_not_rekey_again(self):
        data=fixture();out,_=self.run_fixture(data)
        again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(again,out);self.assertEqual(audit['ocr_calls'],0)

    def test_existing_destination_network_gets_only_confirmed_new_member(self):
        data=fixture();data[0]['nets']['N_IN']={'hyperGraph':'(Z.1)','edges':{}}
        out,audit=self.run_fixture(data)
        self.assertEqual(out['nets']['N_IN']['hyperGraph'],'(X.3,Z.1)')
        self.assertEqual(len(audit['network_changes']),2)

    def test_no_commit_for_partial_or_conflicting_ocr(self):
        data=fixture();original=data[-1]
        for failure in ['low_number','wrong_name','wrong_missing_number','duplicate_view','conflicting_name']:
            def reader(*args):
                tokens=original(*args)
                if failure=='low_number' and args[4]:tokens[0]['confidence']=.97
                if failure=='wrong_name' and not args[4]:tokens[0]['text']='BAD'
                if failure=='wrong_missing_number' and args[4] and 'ROW_B' in args[5]:tokens[0]['text']='9'
                if failure=='duplicate_view':
                    for t in tokens:t['view']='one'
                if failure=='conflicting_name' and not args[4] and args[2]==3:tokens[0]['text']='BAD'
                return tokens
            out,audit=self.run_fixture(data,reader=reader)
            self.assertEqual(out,data[0],failure);self.assertEqual(audit['changes'],[],failure)

    def test_reserved_target_key_or_model_number_blocks_transaction(self):
        data=fixture();data[0]['pins']['X']['pin_2']={'pinname':'RESERVED','point':{'x':20,'y':20}}
        self.assertEqual(self.run_fixture(data)[0],data[0])
        data=fixture();data[1]['pin_candidates'].append({'candidate_id':'RESERVED','owner':'X','owner_status':'assigned','number':'2'})
        self.assertEqual(self.run_fixture(data)[0],data[0])

    def test_reserved_network_reference_blocks_transaction(self):
        data=fixture();data[0]['nets']['OTHER']={'hyperGraph':'(X.2,Z.1)','edges':{}}
        self.assertEqual(self.run_fixture(data)[0],data[0])

    def test_changed_old_record_is_not_overwritten(self):
        for update in [{'pinname':'NEW'},{'point':{'x':30,'y':130}}]:
            data=fixture();data[0]['pins']['X']['pin_3'].update(update)
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_wrong_absent_or_multiple_old_memberships_block_before_ocr(self):
        for kind in ['absent','wrong','multiple']:
            data=fixture()
            if kind=='absent':data[0]['nets']={}
            if kind=='wrong':data[0]['nets']={'OTHER':data[0]['nets']['N_OUT']}
            if kind=='multiple':data[0]['nets']['OTHER']=copy.deepcopy(data[0]['nets']['N_OUT'])
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_attachments_require_exact_point_unique_membership_and_resolved_junctions(self):
        for kind in ['point','missing','duplicate','unattached','candidate_membership','junction']:
            data=fixture()
            if kind=='point':data[2]['pin_attachments'][1]['point']=[36.,120.]
            if kind=='missing':data[2]['pin_attachments'].pop()
            if kind=='duplicate':data[2]['pin_attachments'].append(copy.deepcopy(data[2]['pin_attachments'][0]))
            if kind=='unattached':data[2]['pin_attachments'][0]['attachment_status']='unattached'
            if kind=='candidate_membership':data[2]['networks'][1]['pin_candidate_ids']=[]
            if kind=='junction':data[2]['networks'][0]['unresolved_junction_ids']=['J']
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_shared_token_must_have_unique_owner_rival_and_cross_row_position(self):
        for kind in ['ambiguous_rival','no_shared_token','inside','wrong_row','low_confidence','approximate']:
            data=fixture()
            if kind=='ambiguous_rival':data[1]['pin_candidates'].append({**copy.deepcopy(data[1]['pin_candidates'][0]),'candidate_id':'OTHER'})
            if kind=='no_shared_token':data[1]['pin_candidates'][0]['text_candidates']['number'][0]['token_id']='ELSEWHERE'
            if kind=='inside':data[1]['pin_candidates'][1]['text_candidates']['number'][0]['bbox']=[70,105,78,113]
            if kind=='wrong_row':data[1]['pin_candidates'][1]['text_candidates']['number'][0]['bbox']=[40,80,48,90]
            if kind=='low_confidence':data[1]['pin_candidates'][1]['text_candidates']['number'][0]['confidence']=.97
            if kind=='approximate':data[1]['pin_candidates'][1]['text_candidates']['number'][0]['approximate_split_geometry']=True
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_unsupported_frame_or_missing_lead_abstains(self):
        data=fixture();data[4][:,:,1:]=data[4][:,:,:1]
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)
        data=fixture();data[4][118:123,28:49]=255
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_short_row_gap_abstains(self):
        data=fixture();data[1]['pin_candidates'][1]['point'][1]=115.
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_ocr_budget_never_leaves_half_rekey(self):
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':6})
        self.assertEqual(out,data[0]);self.assertTrue(audit['budget_exhausted']);self.assertEqual(audit['ocr_calls'],6)

    def test_zero_group_budget_avoids_ocr(self):
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_groups':0})
        self.assertEqual(out,data[0]);self.assertTrue(audit['budget_exhausted']);self.assertEqual(audit['ocr_calls'],0)

    def test_power_export_failure_does_not_mutate_inputs(self):
        data=fixture();before=copy.deepcopy(data[:4])
        with patch.object(pairing,'restore_power_wires',side_effect=RuntimeError('export failed')):
            with self.assertRaises(RuntimeError):
                pairing.reconcile_pin_pairs(*data[:5],reader=data[-1])
        self.assertEqual(data[:4],before)

    def test_disabled_wrapper_needs_no_artifacts(self):
        target={'unchanged':True}
        with patch.dict(os.environ,{'PCB_PIN_PAIRING':'0'}):result,audit=agent.finalize_pin_pairing(target,Path('missing'),None,'missing')
        self.assertIs(result,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_provenance_and_coordinate_errors(self):
        for update in [{'image_sha256':'wrong'},{'prediction_only':False},{'coordinate_origin':'bottom_left'},{'size':[21,20]}]:
            with tempfile.TemporaryDirectory() as temp:
                stage=Path(temp);(stage/'export').mkdir()
                document={'image_sha256':'current','prediction_only':True,'coordinate_origin':'top_left','size':[20,20]}
                for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                    (stage/'export'/name).write_text(json.dumps({**document,**update}),encoding='utf-8')
                (stage/'export/completed_component_records.json').write_text('{"records":[]}',encoding='utf-8')
                with self.assertRaises(ValueError):agent.finalize_pin_pairing({},stage,np.zeros((20,20,3),np.uint8),'current')


class PowerBranchTests(unittest.TestCase):
    def fixture(self):
        out={'components':{'X':{'type':'box'},'P':{'type':'v','value':'+5V'}},
             'pins':{'X':{'pin_3':{'pinname':'IN','point':{'x':30,'y':120}}}},'nets':{}}
        model={'pin_candidates':[{'candidate_id':'ROW_B','owner':'X','number':'3','point':[35.,120.]}]}
        graph={'networks':[{'network_id':'N_IN','pin_candidate_ids':['ROW_B']}],
               'pin_attachments':[{'candidate_id':'ROW_B','point':[35.,120.],'attachment_status':'attached',
                                   'network_id':'N_IN','physical_network_id':'PHYSICAL_IN'}],
               'physical_networks':[{'network_id':'PHYSICAL_IN','pin_candidate_ids':['ROW_B','POWER']} ]}
        restored=copy.deepcopy(out);restored['pins']['P']={'pin_1':{'pinname':'+5V','point':{'x':10,'y':120}}}
        restored['nets']['RESTORED_PHYSICAL_IN']={'hyperGraph':'(P.1,X.3)','edges':{'e':[{'x':10,'y':120},{'x':30,'y':120}]}}
        entries=[{'network_id':'RESTORED_PHYSICAL_IN','observed_rail':'+5V'}]
        return out,model,graph,restored,entries

    def run_fixture(self,data):
        out,model,graph,restored,entries=data
        with patch.object(pairing,'restore_power_wires',return_value=(restored,{'restored_networks':entries})):
            audit=pairing._restore_changed_power_branches(out,graph,model,{'ROW_B'},None)
        return out,audit

    def test_only_matching_two_contact_branch_is_copied(self):
        data=self.fixture();data[3]['nets']['UNRELATED']={'hyperGraph':'(P.2,Z.1)','edges':{}}
        data[3]['pins']['UNRELATED']={'pin_1':{'pinname':'X','point':{'x':0,'y':0}}}
        data[4].append({'network_id':'UNRELATED'})
        out,audit=self.run_fixture(data)
        self.assertEqual(len(audit),1);self.assertEqual(list(out['nets']),['RESTORED_PHYSICAL_IN'])
        self.assertNotIn('UNRELATED',out['pins']);self.assertIn('pin_1',out['pins']['P'])

    def test_complex_or_uncertain_branch_is_not_copied(self):
        for kind in ['three_contacts','unresolved','wrong_physical','not_recovered']:
            data=self.fixture()
            if kind=='three_contacts':data[2]['physical_networks'][0]['pin_candidate_ids'].append('THIRD')
            if kind=='unresolved':data[2]['physical_networks'][0]['unresolved_junction_ids']=['J']
            if kind=='wrong_physical':data[2]['pin_attachments'][0]['physical_network_id']='OTHER'
            if kind=='not_recovered':data[2]['physical_networks'][0]['pin_candidate_ids']=['OTHER','POWER']
            out,audit=self.run_fixture(data);self.assertEqual(out['nets'],{},kind);self.assertEqual(audit,[],kind)

    def test_existing_ic_membership_is_not_stolen(self):
        data=self.fixture();data[0]['nets']['EXISTING']={'hyperGraph':'(X.3,Z.1)','edges':{}}
        out,audit=self.run_fixture(data);self.assertEqual(list(out['nets']),['EXISTING']);self.assertEqual(audit,[])

    def test_existing_power_pin_cannot_be_overwritten(self):
        data=self.fixture();data[0]['pins']['P']={'pin_1':{'pinname':'OTHER','point':{'x':0,'y':0}}}
        before=copy.deepcopy(data[0]);out,audit=self.run_fixture(data)
        self.assertEqual(out,before);self.assertEqual(audit,[])


if __name__=='__main__':unittest.main()
