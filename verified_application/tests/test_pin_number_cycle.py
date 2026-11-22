"""Closed permutation safety, including different existing physical networks."""
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
from pcb_parser.pin_number_cycle import reconcile_bottom_number_cycles


def fixture():
    rgb=np.full((240,260,3),255,np.uint8);cv2.rectangle(rgb,(50,40),(170,160),(150,0,0),1)
    candidates=[];pins={};nets={};networks=[];attachments=[]
    for cid,number,name,x,nid in [('IN','2','IN',80,'N_A'),('GND','1','GND',110,'N_B'),('OUT','3','OUT',140,'N_C')]:
        cv2.line(rgb,(x,160),(x,180),(150,0,0),1)
        pin={'pinname':name,'point':{'x':float(x),'y':60.},'provenance_tag':cid}
        pins['pin_'+number]=pin
        candidate={'candidate_id':cid,'owner':'X','owner_status':'assigned','point':[float(x),168.],
                   'number':number,'pinname':name,'confidence':.97,'text_candidates':{'number':[]}}
        if cid=='GND':candidate['number_source']='library_pin_name'
        if cid=='IN':candidate['number_source']='pin_number_strip_OCR'
        if cid=='OUT':candidate['text_candidates']['number']=[{'text':'3','confidence':.999,'bbox':[132,162,138,168],'source_view':v} for v in ['a','b']]
        candidates.append(candidate)
        nets[nid]={'hyperGraph':f'(X.{number},PEER_{cid}.1)','edges':{'e':[{'x':x,'y':60},{'x':x,'y':10}]}}
        networks.append({'network_id':nid,'pin_candidate_ids':[cid],'edge_ids':['E_'+cid],'unresolved_junction_ids':[]})
        attachments.append({'candidate_id':cid,'point':[float(x),168.],'network_id':nid,'attachment_status':'attached','physical_network_id':'P_'+cid})
    target={'components':{'X':{'type':'box','bbox':[50,80,170,200]}},'pins':{'X':pins},'nets':nets}
    document={'pin_candidates':candidates};graph={'networks':networks,'pin_attachments':attachments,'physical_networks':[]}
    records=[{'component_id':'X','bbox_top_left':[50,40,170,160]}]
    def reader(image,box,scale,rotation,recognition_only,view):
        if recognition_only:
            cid=next(c for c in ['IN','GND','OUT'] if f'cycle_glyph_{c}_' in view)
            value={'IN':'1','GND':'2','OUT':'3'}[cid] if '_r90_' in view else '8'
            return [{'text':value,'confidence':.999,'bbox':box,'view':view}]
        return [{'text':name,'confidence':.999,'bbox':[x-6,125,x+8,150],'view':view}
                for name,x in [('IN',80),('GND',110),('OUT',140)]]
    return target,document,graph,records,rgb,reader


class CycleTests(unittest.TestCase):
    def run_fixture(self,data=None,reader=None,**kwargs):
        data=data or fixture()
        return reconcile_bottom_number_cycles(*data[:5],reader=reader or data[-1],**kwargs)

    def test_swap_preserves_physical_records_and_updates_two_networks(self):
        data=fixture();original=copy.deepcopy(data[:4]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:4],original);self.assertEqual(len(audit['changes']),2)
        self.assertEqual(out['pins']['X']['pin_1'],data[0]['pins']['X']['pin_2'])
        self.assertEqual(out['pins']['X']['pin_2'],data[0]['pins']['X']['pin_1'])
        self.assertEqual(out['pins']['X']['pin_3'],data[0]['pins']['X']['pin_3'])
        self.assertEqual(out['nets']['N_A']['hyperGraph'],'(PEER_IN.1,X.1)')
        self.assertEqual(out['nets']['N_B']['hyperGraph'],'(PEER_GND.1,X.2)')
        self.assertEqual(out['components'],data[0]['components'])
        for nid,net in data[0]['nets'].items():self.assertEqual(out['nets'][nid]['edges'],net['edges'])

    def test_idempotence_avoids_second_ocr(self):
        data=fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(out,again);self.assertEqual(audit['ocr_calls'],0)

    def test_correct_control_row_is_verified_unchanged(self):
        data=fixture();original=data[-1]
        def reader(*args):
            tokens=original(*args)
            if args[4] and '_r90_' in args[5]:
                if 'cycle_glyph_IN_' in args[5]:tokens[0]['text']='2'
                if 'cycle_glyph_GND_' in args[5]:tokens[0]['text']='1'
            return tokens
        out,audit=self.run_fixture(data,reader=reader)
        self.assertEqual(out,data[0]);self.assertEqual(audit['groups'][0]['status'],'verified_unchanged')

    def test_same_network_swap_does_not_collapse_references(self):
        data=fixture();data[0]['nets'].pop('N_A');data[0]['nets'].pop('N_B')
        data[0]['nets']['SHARED']={'hyperGraph':'(X.1,X.2,X.10,Y.1)','edges':{}}
        data[2]['networks']=[n for n in data[2]['networks'] if n['network_id']=='N_C']
        data[2]['networks'].append({'network_id':'SHARED','pin_candidate_ids':['IN','GND'],'edge_ids':[]})
        for a in data[2]['pin_attachments']:
            if a['candidate_id']!='OUT':a['network_id']='SHARED'
        out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),2)
        self.assertEqual(out['nets']['SHARED']['hyperGraph'],'(X.1,X.2,X.10,Y.1)')
        self.assertEqual(out['pins']['X']['pin_1']['pinname'],'IN')

    def restored_fixture(self):
        data=fixture();data[0]['nets']['RESTORED_P_IN']=data[0]['nets'].pop('N_A')
        source=data[2]['networks'][0];source['network_id']='POWER_LOGICAL';source['physical_network_ids']=['P_IN']
        data[2]['pin_attachments'][0]['network_id']='POWER_LOGICAL'
        data[2]['physical_networks']=[{'network_id':'P_IN','pin_candidate_ids':['IN'],'edge_ids':['E_IN'],'unresolved_junction_ids':[]}]
        return data

    def test_existing_restored_physical_membership_is_supported(self):
        data=self.restored_fixture();out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),2)
        self.assertIn('X.1',out['nets']['RESTORED_P_IN']['hyperGraph'])
        self.assertNotIn('X.2',out['nets']['RESTORED_P_IN']['hyperGraph'])

    def test_unproven_restored_physical_provenance_blocks_transaction(self):
        for kind in ['candidate','physical_id','edge','junction','exported_name']:
            data=self.restored_fixture()
            if kind=='candidate':data[2]['physical_networks'][0]['pin_candidate_ids']=[]
            if kind=='physical_id':data[2]['networks'][0]['physical_network_ids']=[]
            if kind=='edge':data[2]['physical_networks'][0]['edge_ids']=['UNRELATED']
            if kind=='junction':data[2]['physical_networks'][0]['unresolved_junction_ids']=['J']
            if kind=='exported_name':data[0]['nets']['OTHER']=data[0]['nets'].pop('RESTORED_P_IN')
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_existing_direct_junctions_are_recorded_and_topology_is_preserved(self):
        data=fixture();data[2]['networks'][1]['unresolved_junction_ids']=['J_OLD']
        before=copy.deepcopy(data[2]);out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),2)
        self.assertEqual(audit['groups'][0]['preserved_unresolved_junctions'],{'N_B':['J_OLD']})
        self.assertEqual(data[2],before)
        self.assertEqual(out['nets']['N_B']['edges'],data[0]['nets']['N_B']['edges'])

    def test_all_three_fields_need_corroboration_and_closed_number_set(self):
        data=fixture();original=data[-1]
        for kind in ['wrong_name','low_number','nonclosed','duplicate_number','duplicate_view','conflicting_name']:
            def reader(*args):
                tokens=original(*args)
                if kind=='wrong_name' and not args[4]:tokens[0]['text']='OTHER'
                if kind=='low_number' and args[4] and 'cycle_glyph_GND_' in args[5]:tokens[0]['confidence']=.97
                if kind=='nonclosed' and args[4] and 'cycle_glyph_IN_' in args[5]:tokens[0]['text']='4'
                if kind=='duplicate_number' and args[4] and 'cycle_glyph_IN_' in args[5]:tokens[0]['text']='2'
                if kind=='duplicate_view':
                    for t in tokens:t['view']='one'
                if kind=='conflicting_name' and not args[4] and args[2]==3.:tokens[0]['text']='OTHER'
                return tokens
            out,audit=self.run_fixture(data,reader=reader)
            self.assertEqual(out,data[0],kind);self.assertEqual(audit['changes'],[],kind)

    def test_orientation_must_be_unique(self):
        data=fixture();original=data[-1]
        def reader(*args):
            tokens=original(*args)
            if args[4] and 'cycle_glyph_OUT_' in args[5]:tokens[0]['text']='3'
            return tokens
        out,audit=self.run_fixture(data,reader=reader)
        self.assertEqual(out,data[0]);self.assertEqual(audit['groups'][0]['status'],'orientation_not_uniquely_corroborated')

    def test_conflicting_direct_anchor_is_not_overridden(self):
        data=fixture();data[1]['pin_candidates'][0]['text_candidates']['number']=[
            {'text':'2','confidence':.999,'bbox':[72,162,78,168],'source_view':v} for v in ['a','b']]
        out,audit=self.run_fixture(data)
        self.assertEqual(out,data[0]);self.assertEqual(audit['groups'][0]['status'],'observed_anchor_would_change')

    def test_changed_target_records_are_protected(self):
        for update in [{'pinname':'NEW'},{'point':{'x':84.,'y':60.}},{'point':{'x':80.,'y':90.}}]:
            data=fixture();data[0]['pins']['X']['pin_2'].update(update)
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,update)

    def test_missing_or_duplicate_network_memberships_block_before_ocr(self):
        for kind in ['missing','duplicate']:
            data=fixture()
            if kind=='missing':data[0]['nets'].pop('N_A')
            else:data[0]['nets']['OTHER']=copy.deepcopy(data[0]['nets']['N_A'])
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_model_attachments_need_exact_unique_point_and_source_membership(self):
        for kind in ['point','duplicate','unattached','source_member','wrong_network']:
            data=fixture()
            if kind=='point':data[2]['pin_attachments'][0]['point']=[81.,168.]
            if kind=='duplicate':data[2]['pin_attachments'].append(copy.deepcopy(data[2]['pin_attachments'][0]))
            if kind=='unattached':data[2]['pin_attachments'][0]['attachment_status']='unattached'
            if kind=='source_member':data[2]['networks'][0]['pin_candidate_ids']=[]
            if kind=='wrong_network':data[2]['pin_attachments'][0]['network_id']='OTHER'
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_anchor_needs_two_observed_source_views(self):
        for kind in ['missing','duplicate','approximate','low']:
            data=fixture();tokens=data[1]['pin_candidates'][2]['text_candidates']['number']
            if kind=='missing':tokens.clear()
            if kind=='duplicate':
                for t in tokens:t['source_view']='one'
            if kind=='approximate':
                for t in tokens:t['approximate_split_geometry']=True
            if kind=='low':
                for t in tokens:t['confidence']=.97
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_unsupported_frame_or_missing_visible_lead_abstains(self):
        data=fixture();data[4][:,:,1:]=data[4][:,:,:1]
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)
        data=fixture();data[4][162:173,78:83]=255
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_duplicate_numbers_low_confidence_or_no_library_trigger_abstains(self):
        for kind in ['number','confidence','library','extra_candidate']:
            data=fixture()
            if kind=='number':data[1]['pin_candidates'][0]['number']='1'
            if kind=='confidence':data[1]['pin_candidates'][0]['confidence']=.89
            if kind=='library':data[1]['pin_candidates'][1]['number_source']='observed_OCR_text'
            if kind=='extra_candidate':data[1]['pin_candidates'].append({**data[1]['pin_candidates'][0],'candidate_id':'DUP'})
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_ocr_budget_never_commits_half_a_permutation(self):
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':10})
        self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],10);self.assertTrue(audit['budget_exhausted'])

    def test_zero_component_budget_avoids_ocr(self):
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_components':0})
        self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0);self.assertTrue(audit['budget_exhausted'])

    def test_ocr_failure_never_mutates_inputs(self):
        data=fixture();before=copy.deepcopy(data[:4])
        def reader(*args):raise RuntimeError('OCR failed')
        with self.assertRaises(RuntimeError):self.run_fixture(data,reader=reader)
        self.assertEqual(data[:4],before)

    def test_disabled_wrapper_needs_no_artifacts(self):
        target={'unchanged':True}
        with patch.dict(os.environ,{'PCB_PIN_NUMBER_CYCLE':'0'}):
            out,audit=agent.finalize_pin_number_cycle(target,Path('missing'),None,'missing')
        self.assertIs(out,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_foreign_or_nonprediction_artifacts(self):
        for update in [{'image_sha256':'wrong'},{'prediction_only':False},{'coordinate_origin':'bottom_left'},{'size':[21,20]}]:
            with tempfile.TemporaryDirectory() as temp:
                stage=Path(temp);(stage/'export').mkdir()
                document={'image_sha256':'current','prediction_only':True,'coordinate_origin':'top_left','size':[20,20]}
                for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                    (stage/'export'/name).write_text(json.dumps({**document,**update}),encoding='utf-8')
                (stage/'export/completed_component_records.json').write_text('{"records":[]}',encoding='utf-8')
                with self.assertRaises(ValueError):agent.finalize_pin_number_cycle({},stage,np.zeros((20,20,3),np.uint8),'current')


if __name__=='__main__':unittest.main()
