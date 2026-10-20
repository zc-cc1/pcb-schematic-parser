"""Synthetic field/attachment transactions; no OCR weights or reference pages."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIVE=Path('D:/pcb5_submission') if not (ROOT/'adapter.py').exists() else ROOT
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'));sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser import pin_field_recovery as recovery


def pin(cid,number=None,name=None,point=(10,60),**extra):
    return {'candidate_id':cid,'owner':'U1','owner_status':'assigned','point':list(point),
            'number':number,'pinname':name,'confidence':.99,**extra}


def fixture():
    target={'components':{'U1':{'type':'box','bbox':[20,20,100,180]}},
            'pins':{'U1':{'pin_4':{'pinname':'VM','point':{'x':2,'y':140}}}},
            'nets':{'N1':{'hyperGraph':'(U1.4)','edges':{}}}}
    model={'pin_candidates':[pin('a','4','VM',number_source='library_pin_name'),pin('b',None,'VM',(10,40))]}
    graph={'networks':[{'network_id':'N1','pin_candidate_ids':['a','b']}],
           'pin_attachments':[{'candidate_id':p['candidate_id'],'point':p['point'],
                               'attachment_status':'attached','network_id':'N1'} for p in model['pin_candidates']]}
    groups=recovery.build_groups(target,model,200)
    for g in groups:
        g['number']='5' if g['id']=='a' else '4';g['number_audit']={'reason':'accepted'}
    return target,model,graph,groups,np.full((200,220,3),255,dtype=np.uint8)


class RecoveryTests(unittest.TestCase):
    def test_repeated_views_are_not_duplicate_tokens(self):
        token={'text':'5','confidence':.999,'view':'v1'}
        self.assertIsNone(recovery._vote([token,token],recovery.DEFAULT)[0])
        self.assertEqual(recovery._vote([token,{**token,'view':'v2'}],recovery.DEFAULT)[0],'5')

    def test_low_confidence_and_conflicts_remain_unresolved(self):
        ts=[{'text':'6','confidence':.97,'view':f'v{i}'} for i in range(3)]
        self.assertIsNone(recovery._vote(ts,recovery.DEFAULT)[0])
        ts=[{'text':s,'confidence':.999,'view':f'{s}_{i}'} for s in ['6','9'] for i in range(2)]
        self.assertEqual(recovery._vote(ts,recovery.DEFAULT)[1]['reason'],'conflicting_spelling')

    def test_explicit_primary_merges_fields_without_moving_geometry(self):
        target,model,graph,groups,image=fixture()
        model['pin_candidates']=[pin('a',None,'D-'),pin('b','2',None,(15,60),text_hypothesis_primary_candidate_id='a')]
        grouped=recovery.build_groups(target,model,200)
        self.assertEqual(len(grouped),1);self.assertEqual(grouped[0]['number'],'2')
        self.assertEqual(grouped[0]['name'],'D-');self.assertEqual(grouped[0]['point'],[10,60])

    def test_conflicting_alias_numbers_do_not_get_selected(self):
        target,model,graph,groups,image=fixture()
        model['pin_candidates']=[pin('a','1','D-'),pin('b','2',None,(15,60),text_hypothesis_primary_candidate_id='a')]
        grouped=recovery.build_groups(target,model,200)
        self.assertFalse(recovery.select_groups(grouped,recovery.DEFAULT))

    def test_vertical_duplicates_keep_outward_primary(self):
        target,model,graph,groups,image=fixture()
        model['pin_candidates']=[pin('a','6',None,(60,18)),pin('b',None,None,(60,0))]
        grouped=recovery.build_groups(target,model,200)
        self.assertEqual(len(grouped),1);self.assertEqual(grouped[0]['id'],'b')
        self.assertEqual(grouped[0]['number'],'6')

    def test_owner_boundaries_are_not_merged(self):
        target,model,graph,groups,image=fixture();target['components']['U2']=copy.deepcopy(target['components']['U1'])
        model['pin_candidates']=[pin('a','6',None,(60,18)),{**pin('b',None,None,(60,0)),'owner':'U2'}]
        self.assertEqual(len(recovery.build_groups(target,model,200)),2)

    def test_complete_renumber_pair_updates_keys_and_existing_net(self):
        target,model,graph,groups,image=fixture();original=copy.deepcopy((target,model,graph))
        with patch.object(recovery,'collect',return_value=([],[],{'ocr_calls':0,'budget_exhausted':False})),patch.object(recovery,'bind',return_value=groups):
            output,audit=recovery.recover_pin_fields(target,model,graph,[],image,reader=None,
                position_corrector=lambda t:None,raw_pins=target['pins'])
        self.assertEqual(set(output['pins']['U1']),{'pin_4','pin_5'})
        self.assertEqual(output['pins']['U1']['pin_4']['point']['y'],160)
        self.assertEqual(output['pins']['U1']['pin_5']['point']['y'],140)
        self.assertEqual(output['nets']['N1']['hyperGraph'],'(U1.4,U1.5)')
        self.assertEqual((target,model,graph),original)

    def test_renumber_across_unproven_networks_withholds_entire_pair(self):
        target,model,graph,groups,image=fixture();graph['pin_attachments'][1]['network_id']='N2'
        with patch.object(recovery,'collect',return_value=([],[],{'ocr_calls':0,'budget_exhausted':False})),patch.object(recovery,'bind',return_value=groups):
            output,audit=recovery.recover_pin_fields(target,model,graph,[],image,reader=None,
                position_corrector=lambda t:None,raw_pins=target['pins'])
        self.assertEqual(output,target);self.assertEqual(audit['changes'],[])

    def test_ocr_budget_is_hard_limit(self):
        target,model,graph,groups,image=fixture();calls=[]
        def reader(*args):calls.append(1);return []
        _,_,audit=recovery.collect(image,groups,[{'component_id':'U1','bbox_top_left':[20,20,100,180]}],model,target,reader,
                                   {**recovery.DEFAULT,'max_ocr_calls':3})
        self.assertEqual(len(calls),3);self.assertTrue(audit['budget_exhausted'])

    def test_unknown_owner_or_missing_geometry_is_not_invented(self):
        target,model,graph,groups,image=fixture();model['pin_candidates']=[pin('a',None,'IN',owner_status='ambiguous_owner')]
        output,audit=recovery.recover_pin_fields(target,model,graph,[],image,
            reader=lambda *a:self.fail('Unexpected OCR'),position_corrector=lambda *a:self.fail('Unexpected geometry'),raw_pins=target['pins'])
        self.assertEqual(output,target);self.assertEqual(audit['ocr_calls'],0)

    def test_inner_digit_name_needs_actual_two_view_evidence(self):
        g={'id':'a','owner':'U1','point':[60,0],'side':'top','box':[20,20,100,180],
           'members':['a'],'number':'6','name':None,'original_number':'6','original_name':None,'library_number':False}
        target,model,graph,groups,image=fixture();model['pin_candidates']=[pin('a','6',None,(60,0))]
        text={'text':'6','confidence':.999,'owner':'U1','bbox':[55,28,65,38],'rotation':0,'view':'v1'}
        result=recovery.bind([g],[g],[],[text],model,target,recovery.DEFAULT)
        self.assertIsNone(result[0]['name'])
        result=recovery.bind([g],[g],[],[text,{**text,'view':'v2'}],model,target,recovery.DEFAULT)
        self.assertEqual(result[0]['name'],'6')

    def test_existing_attachment_must_match_candidate_coordinates(self):
        target,model,graph,groups,image=fixture();graph['pin_attachments'][0]['point']=[999,999]
        out=copy.deepcopy(target);audit=recovery.sync_existing_networks(out,graph,[groups[0]])
        self.assertEqual(out,target);self.assertEqual(audit,[])

    def test_existing_member_is_never_attached_to_second_net(self):
        target,model,graph,groups,image=fixture();target['nets']['N2']={'hyperGraph':'(U1.5)','edges':{}}
        out=copy.deepcopy(target);recovery.sync_existing_networks(out,graph,[groups[0]])
        self.assertEqual(out,target)

    def test_no_new_network_or_wire_is_created(self):
        target,model,graph,groups,image=fixture();target['nets']={}
        out=copy.deepcopy(target);audit=recovery.sync_existing_networks(out,graph,groups)
        self.assertEqual(out,target);self.assertEqual(audit[0]['status'],'withheld_new_net')

    def test_feature_opt_out_avoids_loading_artifacts_or_models(self):
        target,model,graph,groups,image=fixture()
        with patch.dict(os.environ,{'PCB_PIN_FIELD_RECOVERY':'0'}):
            out,audit=agent.finalize_pin_field_recovery(target,Path('missing-stage'),image,'digest',target['pins'])
        self.assertIs(out,target);self.assertFalse(audit['enabled'])

    def test_mismatched_image_provenance_is_rejected(self):
        target,model,graph,groups,image=fixture();graph['image_sha256']='another-image'
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder);export=stage/'export';export.mkdir()
            for name,data in [('engineering_pin_text_result.json',model),('engineering_graph_proposals.json',graph),('completed_component_records.json',{'records':[]})]:
                (export/name).write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'differ'):
                agent.finalize_pin_field_recovery(target,stage,image,'current-image',target['pins'])


if __name__=='__main__':unittest.main()
