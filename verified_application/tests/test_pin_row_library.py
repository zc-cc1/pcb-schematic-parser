"""Exact-model, row ownership, alphanumeric IDs and one remaining contact."""
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
from pcb_parser.pin_row_library import reconcile_rows_with_library,pin_table,strict_vote


def fixture(template=False):
    height=280;rgb=np.full((height,330,3),255,np.uint8)
    frame=[80,40,200,190 if template else 160];cv2.rectangle(rgb,(80,40),(200,frame[3]),(150,0,0),1)
    model='MODEL5' if template else 'PART8'
    definitions=([('left',100,'5','FRAME'),('right',70,'1','SIGA'),('right',100,'2','SIGB'),('right',130,'3','SIGC'),('right',160,'4','SIGD')]
                 if template else [('left',90,'7','EH'),('right',70,'B5','CC2'),('right',100,'A5','CC1'),('right',130,'6','VDD')])
    records=[{'component_id':'X','bbox_top_left':frame}]
    old={} if template else {'pin_9':{'pinname':'VDD','point':{'x':240.,'y':150.},'keep':'physical_record'}}
    target={'components':{'X':{'type':'box','Name':'X','bbox':[80,height-frame[3],200,height-40]}},'pins':{'X':old},'nets':{}}
    doc={'pin_candidates':[]};graph={'networks':[],'pin_attachments':[],'edges':[]};whole=[]
    whole.append({'text':model,'confidence':.999,'bbox':[82,frame[3]+2,143,frame[3]+18]})
    library={model:{'pins':[{'number':n,'name':name} for side,row,n,name in definitions]}}
    for side,row,number,name in definitions:
        edge=80 if side=='left' else 200;tip=40 if side=='left' else 240
        cv2.line(rgb,(edge,row),(tip,row),(150,0,0),1)
        if name in ['CC1','CC2','SIGB','SIGC']:
            for slope in [-1,1]:cv2.line(rgb,(tip-8,row-8*slope),(tip+8,row+8*slope),(0,130,0),1)
        elif name!='FRAME':cv2.line(rgb,(10 if side=='left' else 241,row),(39 if side=='left' else 300,row),(0,130,0),1)
        cx=60 if side=='left' else 220;cid='P'+number
        candidate={'candidate_id':cid,'owner':'X','owner_status':'assigned','point':[float(cx),float(row)],'number':None,'pinname':None,'text_candidates':{}}
        if name=='VDD':
            candidate.update(number='9',pinname='VDD',point=[240.,130.],text_candidates={'number':[{'text':'9','confidence':.95,'bbox':[207,116,217,128],'source_view':'original'}]})
            cv2.putText(rgb,'6',(208,127),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
        doc['pin_candidates'].append(candidate)
        if name!='FRAME':
            name_box=[83,row-5,110,row+8] if side=='left' else [166,row-5,196,row+8]
            whole.append({'text':name,'confidence':.999,'bbox':name_box})
        if not template:
            number_box=[64,row-13,74,row-1] if side=='left' else [208,row-14,230 if len(number)>1 else 217,row-2]
            whole.append({'text':'9' if name=='VDD' else number,'confidence':.95 if name=='VDD' else .999,'bbox':number_box})
        if name in ['EH','VDD','SIGA','SIGD']:
            nid='N_'+name;target['nets'][nid]={'hyperGraph':f'(PEER_{name}.1'+(',X.9)' if name=='VDD' else ')'),
                'edges':{'e':[{'x':float(tip),'y':float(height-row)},{'x':10. if side=='left' else 300.,'y':float(height-row)}]}}
            eid='E_'+name;poly=[[float(tip),float(row)],[10. if side=='left' else 300.,float(row)]]
            graph['edges'].append({'edge_id':eid,'polyline':poly})
            graph['networks'].append({'network_id':nid,'edge_ids':[eid],'pin_candidate_ids':[cid],'unresolved_junction_ids':[]})
            graph['pin_attachments'].append({'candidate_id':cid,'point':candidate['point'].copy(),'network_id':nid,'attachment_status':'attached'})
    def reader(image,box,scale,rotation,rec,view):
        if not rec:return [{**copy.deepcopy(t),'view':view} for t in whole]
        return [{'text':'6','confidence':.999,'bbox':box,'view':view}]
    return target,doc,graph,records,library,rgb,reader


class RowLibraryTests(unittest.TestCase):
    def run_fixture(self,data=None,reader=None,**kwargs):
        data=data or fixture()
        return reconcile_rows_with_library(*data[:6],reader=reader or data[-1],**kwargs)

    def test_observed_four_fields_recover_without_moving_the_rekeyed_record(self):
        data=fixture();before=copy.deepcopy(data[:5]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:5],before);self.assertEqual(len(audit['changes']),4)
        self.assertEqual(out['pins']['X']['pin_6'],data[0]['pins']['X']['pin_9']);self.assertNotIn('pin_9',out['pins']['X'])
        self.assertEqual(set(out['pins']['X']),{'pin_6','pin_7','pin_A5','pin_B5'})
        self.assertEqual(out['components']['X']['Name'],'PART8')
        self.assertEqual(out['nets']['N_VDD']['hyperGraph'],'(PEER_VDD.1,X.6)')
        self.assertEqual(out['nets']['N_EH']['hyperGraph'],'(PEER_EH.1,X.7)')
        for nid,net in data[0]['nets'].items():self.assertEqual(net['edges'],out['nets'][nid]['edges'])

    def test_template_uses_four_visible_names_and_one_remaining_contact(self):
        data=fixture(True);out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),5);self.assertEqual(out['pins']['X']['pin_5']['pinname'],'FRAME')
        inferred=next(r for r in audit['groups'][0]['rows'] if r['number']=='5')
        self.assertEqual(inferred['method'],'unique_remaining_library_contact')
        self.assertEqual(out['components']['X']['Name'],'MODEL5')
        self.assertEqual({r['number'] for r in audit['groups'][0]['rows']},{'1','2','3','4','5'})

    def test_crossed_and_floating_contacts_remain_unconnected(self):
        out,audit=self.run_fixture(fixture(True));all_members=','.join(n['hyperGraph'] for n in out['nets'].values())
        self.assertIn('X.1',all_members);self.assertIn('X.4',all_members)
        for n in ['2','3','5']:self.assertNotIn('X.'+n,all_members)
        self.assertEqual({r['reason'] for r in audit['network_withheld']},{'visible_no_connect_cross','no_visible_green_contact'})

    def test_library_is_not_a_single_first_name_lookup(self):
        record={'pins':[{'number':'7','name':'EH'},{'number':'7','name':'GND'},{'number':'7','name':'EH'}]}
        self.assertEqual(pin_table(record),{'7':{'EH','GND'}})

    def test_library_template_is_not_tied_to_standard_usb_names(self):
        out,audit=self.run_fixture(fixture(True))
        self.assertEqual([out['pins']['X']['pin_'+str(i)]['pinname'] for i in range(1,6)],['SIGA','SIGB','SIGC','SIGD','FRAME'])

    def test_idempotence_after_name_and_pin_recovery(self):
        data=fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(out,again);self.assertEqual(audit['ocr_calls'],0)

    def test_existing_real_model_name_is_not_reinterpreted(self):
        data=fixture();data[0]['components']['X']['Name']='KNOWN_OTHER'
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_model_requires_multiple_high_confidence_views(self):
        for mode in ['one_view','low']:
            data=fixture();original=data[-1]
            def reader(*args):
                values=original(*args)
                if not args[4]:
                    for t in values:
                        if t['text']=='PART8':
                            if mode=='low':t['confidence']=.979
                            else:t['view']='same'
                return values
            out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_no_fuzzy_model_name_is_accepted(self):
        data=fixture();original=data[-1]
        def reader(*args):
            values=original(*args)
            for t in values:
                if t['text']=='PART8':t['text']='PART9'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_model_must_be_outside_but_near_its_own_frame(self):
        for mode in ['inside','far']:
            data=fixture();original=data[-1]
            def reader(*args):
                values=original(*args)
                for t in values:
                    if t['text']=='PART8':t['bbox']=[100,80,140,100] if mode=='inside' else [270,220,315,240]
                return values
            out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_library_pin_count_must_match_visible_leads(self):
        data=fixture();data[4]['PART8']['pins'].append({'number':'11','name':'EXTRA'})
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_strong_existing_number_is_never_overwritten(self):
        data=fixture();source=next(p for p in data[1]['pin_candidates'] if p['number']=='9')
        source['text_candidates']['number'][0]['confidence']=.999
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_rekey_target_cannot_already_exist(self):
        data=fixture();data[0]['pins']['X']['pin_6']={'pinname':'VDD','point':{'x':239.,'y':150.}}
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_multiple_existing_memberships_block_rekey(self):
        data=fixture();data[0]['nets']['OTHER']={'hyperGraph':'(X.9,OTHER.1)','edges':{}}
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_missing_current_network_is_audited_without_inventing_one(self):
        data=fixture();data[0]['nets'].pop('N_VDD');out,audit=self.run_fixture(data)
        self.assertIn('pin_6',out['pins']['X']);self.assertNotIn('N_VDD',out['nets'])
        self.assertIn({'member':'X.6','reason':'source_model_route_not_exported'},audit['network_withheld'])

    def test_existing_membership_must_agree_with_exact_original_attachment(self):
        data=fixture();a=next(a for a in data[2]['pin_attachments'] if a['candidate_id']=='P6');a['point']=[0.,0.]
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_one_strong_conflicting_new_reading_rejects_component(self):
        data=fixture();original=data[-1]
        def reader(*args):
            values=original(*args)
            if args[4] and args[2]==5.:values[0]['text']='8'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_same_physical_row_cannot_take_name_from_opposite_side(self):
        data=fixture();original=data[-1]
        def reader(*args):
            values=original(*args)
            for t in values:
                if t['text']=='EH':t['bbox']=[165,85,195,98]
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_no_assigned_candidate_means_no_new_identity(self):
        data=fixture();data[1]['pin_candidates']=[p for p in data[1]['pin_candidates'] if p['candidate_id']!='P7']
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_unprinted_contact_requires_its_own_real_candidate(self):
        data=fixture(True);data[1]['pin_candidates']=[p for p in data[1]['pin_candidates'] if p['candidate_id']!='P5']
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_two_unresolved_rows_cannot_be_inferred_by_number_order(self):
        data=fixture(True);original=data[-1]
        def reader(*args):return [t for t in original(*args) if t['text']!='SIGA']
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_remaining_library_contact_must_have_one_unambiguous_name(self):
        data=fixture(True);data[4]['MODEL5']['pins'].append({'number':'5','name':'OTHER'})
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_observed_name_mapping_must_be_unique(self):
        data=fixture(True);data[4]['MODEL5']['pins'].append({'number':'2','name':'SIGA'})
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_blank_contact_inference_rejects_ocr_text_that_could_not_be_bound(self):
        data=fixture(True);original=data[-1]
        def reader(*args):
            values=original(*args)
            if not args[4]:values.append({'text':'UNKNOWN','confidence':.8,'bbox':[83,95,110,108],'view':args[-1]})
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_wrong_library_pairing_cannot_override_printed_alphanumeric_id(self):
        data=fixture();data[4]['PART8']['pins']=[p if p['number']!='A5' else {'number':'A5','name':'WRONG'} for p in data[4]['PART8']['pins']]
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_unresolved_original_route_does_not_gain_new_member(self):
        data=fixture();next(n for n in data[2]['networks'] if n['network_id']=='N_EH')['unresolved_junction_ids']=['J']
        out,audit=self.run_fixture(data);self.assertIn('pin_7',out['pins']['X'])
        self.assertEqual(out['nets']['N_EH'],data[0]['nets']['N_EH'])

    def test_budget_cannot_partially_commit_a_component(self):
        for budget in [0,4,15]:
            data=fixture();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':budget})
            self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_extra_calls_per_component':0})
        self.assertEqual(out,data[0]);self.assertTrue(audit['budget_exhausted'])

    def test_zero_component_budget_skips_ocr(self):
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_components':0})
        self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_ocr_failure_does_not_mutate_input_or_library(self):
        data=fixture();before=copy.deepcopy(data[:5])
        def failing(*args):raise RuntimeError('OCR failed')
        with self.assertRaises(RuntimeError):self.run_fixture(data,reader=failing)
        self.assertEqual(data[:5],before)

    def test_strict_vote_withholds_one_high_confidence_conflict(self):
        tokens=[{'text':'6','confidence':.999,'view':v} for v in ['one','two']]+[{'text':'9','confidence':.999,'view':'other'}]
        self.assertIsNone(strict_vote(tokens,{'min_confidence':.98,'min_views':2})[0])

    def test_disabled_wrapper_does_not_load_library_or_artifacts(self):
        target={'sample':True}
        with patch.dict(os.environ,{'PCB_PIN_ROW_LIBRARY':'0'}):out,audit=agent.finalize_pin_row_library(target,Path('missing'),None,'bad')
        self.assertIs(out,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_foreign_image_before_ocr(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'export').mkdir()
            for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                (stage/'export'/name).write_text(json.dumps({'prediction_only':True,'image_sha256':'foreign'}),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_pin_row_library(data[0],stage,data[5],'current',reader=data[-1])

    def test_wrapper_rejects_a_changed_bundled_library_before_ocr(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);stage=root/'stage';(stage/'export').mkdir(parents=True);(root/'configs').mkdir()
            header={'prediction_only':True,'image_sha256':'current','coordinate_origin':'top_left','size':[330,280]}
            for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                (stage/'export'/name).write_text(json.dumps(header),encoding='utf-8')
            (stage/'export/completed_component_records.json').write_text('{"records":[]}',encoding='utf-8')
            (root/'configs/pin_row_library_v1.json').write_text(json.dumps({'library_path':'lib.json','library_sha256':'0'*64}),encoding='utf-8')
            (root/'lib.json').write_text('{}',encoding='utf-8')
            with patch.object(agent,'ROOT',root),self.assertRaisesRegex(ValueError,'fingerprint'):
                agent.finalize_pin_row_library(data[0],stage,data[5],'current',reader=data[-1])


if __name__=='__main__':unittest.main()
