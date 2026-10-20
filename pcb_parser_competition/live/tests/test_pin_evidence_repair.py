"""Evidence boundaries for full labels and singleton collision correction."""
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
sys.path.insert(0,'D:/pcb5_submission');sys.path.insert(0,'D:/pcb5_submission/src');sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser.pin_evidence_repair import full_label_extends, repair_observed_pin_evidence


def label_fixture():
    rgb=np.full((240,260,3),255,np.uint8);cv2.rectangle(rgb,(50,40),(180,160),(150,0,0),1)
    pins={};candidates=[]
    for number,name,full,row in [('1','GND','ADJ(GND)',70),('2','OUT','OUT(TAB)',100)]:
        cv2.line(rgb,(30,row),(50,row),(150,0,0),1)
        pins['pin_'+number]={'pinname':name,'point':{'x':30.,'y':240.-row}}
        candidates.append({'candidate_id':'C'+number,'owner':'X','owner_status':'assigned','point':[40.,row],
                           'number':number,'pinname':name,'pinname_source':'library_pin_name',
                           'text_candidates':{'pinname':[{'text':full,'confidence':.999,'bbox':[60,row-5,110,row+10],
                                                        'source_view':view} for view in ['a','b']]}})
    target={'components':{'X':{'type':'box','bbox':[50,80,180,200]}},'pins':{'X':pins},
            'nets':{'N':{'hyperGraph':'(X.1,Y.1)','edges':{}}}}
    doc={'pin_candidates':candidates};graph={'networks':[],'pin_attachments':[]}
    records=[{'component_id':'X','bbox_top_left':[50,40,180,160]}]
    def reader(image,box,scale,rotation,rec,view):
        return [{'text':text,'confidence':.999,'bbox':[60,row-5,110,row+10],'view':view}
                for text,row in [('ADJ(GND)',70),('OUT(TAB)',100)]]
    return target,doc,graph,records,rgb,reader


def collision_fixture():
    rgb=np.full((240,300,3),255,np.uint8);cv2.rectangle(rgb,(100,40),(200,180),(150,0,0),1)
    for row in [80,120,140]:cv2.line(rgb,(70,row),(100,row),(150,0,0),1)
    for row in [80,120,140,160]:cv2.line(rgb,(200,row),(220,row),(150,0,0),1)
    pins={'pin_1':{'pinname':'GND','point':{'x':70.,'y':100.}},'pin_4':{'pinname':'EN','point':{'x':70.,'y':100.}}}
    weak={'candidate_id':'WEAK','owner':'X','owner_status':'assigned','point':[85.,140.],
          'number':'1','pinname':'GND','number_source':'library_pin_name','confidence':.74}
    anchor={'candidate_id':'ANCHOR','owner':'X','owner_status':'assigned','point':[85.,140.],
            'number':'4','pinname':'EN','confidence':.96}
    candidates=[weak,anchor]
    for n,name,row in [('2','OUT',80),('3','FB',120),('7','PP',140)]:
        pins['pin_'+n]={'pinname':name,'point':{'x':220.,'y':240.-row}}
        candidates.append({'candidate_id':'C'+n,'owner':'X','owner_status':'assigned','point':[215.,row],'number':n,'pinname':name})
    target={'components':{'X':{'type':'box','bbox':[100,60,200,200]}},'pins':{'X':pins},
            'nets':{'SINGLE':{'hyperGraph':'(X.1)','edges':{}}}}
    doc={'pin_candidates':candidates}
    graph={'networks':[{'network_id':'SINGLE','pin_candidate_ids':['WEAK'],'edge_ids':[],'unresolved_junction_ids':[]}],
           'pin_attachments':[{'candidate_id':'WEAK','point':[85.,140.],'attachment_status':'attached','network_id':'SINGLE'}]}
    records=[{'component_id':'X','bbox_top_left':[100,40,200,180]}]
    def reader(image,box,scale,rotation,rec,view):
        tokens=[]
        for number,name,row in [('2','OUT',80),('3','FB',120),('7','PP',140),('1','GND',160)]:
            tokens.append({'text':number,'confidence':.999,'bbox':[207,row-10,213,row-4],'view':view})
            tokens.append({'text':name,'confidence':.999,'bbox':[167,row-5,187,row+5],'view':view})
        return tokens
    return target,doc,graph,records,rgb,reader


class EvidenceTests(unittest.TestCase):
    def run_fixture(self,data,reader=None,**kwargs):
        return repair_observed_pin_evidence(*data[:5],reader=reader or data[-1],**kwargs)

    def test_only_explicit_balanced_whole_token_aliases_qualify(self):
        for full,short in [('ADJ(GND)','GND'),('OUT(TAB)','OUT')]:self.assertTrue(full_label_extends(full,short))
        for full,short in [('AGND','GND'),('OUT(TAB','OUT'),('OUT(TAB))','OUT'),('VIN(TAB)','IN'),('A(B(C))','C')]:
            self.assertFalse(full_label_extends(full,short))

    def test_full_labels_recovered_without_coordinate_or_network_changes(self):
        data=label_fixture();original=copy.deepcopy(data[:4]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:4],original);self.assertEqual(len(audit['changes']),2)
        self.assertEqual(out['pins']['X']['pin_1']['pinname'],'ADJ(GND)')
        self.assertEqual(out['pins']['X']['pin_2']['pinname'],'OUT(TAB)')
        self.assertEqual(out['nets'],data[0]['nets']);self.assertEqual(out['components'],data[0]['components'])
        for key,pin in data[0]['pins']['X'].items():self.assertEqual(out['pins']['X'][key]['point'],pin['point'])
        self.assertEqual(audit['ocr_calls'],4)

    def test_label_step_is_idempotent(self):
        data=label_fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(again,out);self.assertEqual(audit['ocr_calls'],0)

    def test_observed_names_or_changed_records_are_protected(self):
        for kind in ['observed_source','changed_name','changed_point','duplicate']:
            data=label_fixture()
            for c in data[1]['pin_candidates']:
                if kind=='observed_source':c['pinname_source']='observed_OCR_text'
                if kind=='changed_name':data[0]['pins']['X']['pin_'+c['number']]['pinname']='NEW'
                if kind=='changed_point':data[0]['pins']['X']['pin_'+c['number']]['point']['x']=0
            if kind=='duplicate':
                data[1]['pin_candidates'] += [{**copy.deepcopy(c),'candidate_id':'D'+c['candidate_id']} for c in data[1]['pin_candidates']]
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_existing_duplicate_views_or_low_confidence_do_not_trigger_labels(self):
        for kind in ['duplicate','low','outside']:
            data=label_fixture()
            for c in data[1]['pin_candidates']:
                for t in c['text_candidates']['pinname']:
                    if kind=='duplicate':t['source_view']='same'
                    if kind=='low':t['confidence']=.97
                    if kind=='outside':t['bbox']=[190,50,220,70]
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_fresh_text_must_corroborate_exact_full_spelling(self):
        data=label_fixture();original=data[-1]
        for kind in ['low','partial','duplicate','conflict']:
            def reader(*args):
                tokens=original(*args)
                for t in tokens:
                    if kind=='low':t['confidence']=.97
                    if kind=='partial':t['text']='GND' if t['text'].startswith('ADJ') else 'OUT'
                    if kind=='duplicate':t['view']='one'
                    if kind=='conflict' and args[2]==3:t['text']='OTHER'
                return tokens
            out,audit=self.run_fixture(data,reader=reader)
            self.assertEqual(out,data[0],kind);self.assertEqual(audit['changes'],[],kind)

    def test_collision_point_is_replaced_but_isolated_net_is_preserved(self):
        data=collision_fixture();original=copy.deepcopy(data[:4]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:4],original);self.assertEqual(len(audit['changes']),1)
        self.assertGreater(out['pins']['X']['pin_1']['point']['x'],200)
        self.assertEqual(out['pins']['X']['pin_1']['point']['y'],80.)
        for key,pin in data[0]['pins']['X'].items():
            if key!='pin_1':self.assertEqual(out['pins']['X'][key],pin)
        self.assertEqual(out['nets'],data[0]['nets'])
        self.assertEqual(audit['network_changes'][0]['status'],'withheld')

    def test_collision_step_is_idempotent(self):
        data=collision_fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(out,again);self.assertEqual(audit['ocr_calls'],0)

    def test_active_or_multi_member_network_cannot_be_moved(self):
        for kind in ['wire','member','multiple','missing']:
            data=collision_fixture()
            if kind=='wire':data[0]['nets']['SINGLE']['edges']={'e':[{'x':0,'y':0},{'x':1,'y':1}]}
            if kind=='member':data[0]['nets']['SINGLE']['hyperGraph']='(X.1,Y.1)'
            if kind=='multiple':data[0]['nets']['OTHER']=copy.deepcopy(data[0]['nets']['SINGLE'])
            if kind=='missing':data[0]['nets']={}
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_model_singleton_and_attachment_must_be_unambiguous(self):
        for kind in ['wire','member','junction','point','wrong_net','duplicate_attachment','unattached']:
            data=collision_fixture()
            if kind=='wire':data[2]['networks'][0]['edge_ids']=['E']
            if kind=='member':data[2]['networks'][0]['pin_candidate_ids'].append('OTHER')
            if kind=='junction':data[2]['networks'][0]['unresolved_junction_ids']=['J']
            if kind=='point':data[2]['pin_attachments'][0]['point']=[86.,140.]
            if kind=='wrong_net':data[2]['pin_attachments'][0]['network_id']='OTHER'
            if kind=='duplicate_attachment':data[2]['pin_attachments']*=2
            if kind=='unattached':data[2]['pin_attachments'][0]['attachment_status']='unattached'
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_high_confidence_or_nonlibrary_collision_is_protected(self):
        for update in [{'confidence':.9},{'number_source':'observed_OCR_text'},{'text_hypothesis_primary_candidate_id':'OTHER'}]:
            data=collision_fixture();data[1]['pin_candidates'][0].update(update)
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,update)

    def test_collision_requires_exactly_one_strong_other_pin(self):
        for kind in ['weak_anchor','no_collision','multiple_anchors','changed_name']:
            data=collision_fixture()
            if kind=='weak_anchor':data[1]['pin_candidates'][1]['confidence']=.85
            if kind=='no_collision':data[0]['pins']['X']['pin_4']['point']['y']=110.
            if kind=='changed_name':data[0]['pins']['X']['pin_1']['pinname']='OTHER'
            if kind=='multiple_anchors':
                data[0]['pins']['X']['pin_5']={'pinname':'OTHER','point':{'x':70.,'y':100.}}
                data[1]['pin_candidates'].append({**data[1]['pin_candidates'][1],'candidate_id':'SECOND','number':'5','pinname':'OTHER'})
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0,kind)

    def test_counterpart_must_have_supported_matching_number_and_name(self):
        data=collision_fixture();original=data[-1]
        for kind in ['name','number','view']:
            def reader(*args):
                tokens=original(*args)
                for t in tokens:
                    if kind=='name' and t['text']=='GND':t['text']='OTHER'
                    if kind=='number' and t['text']=='1':t['text']='9'
                    if kind=='view':t['view']='one'
                return tokens
            self.assertEqual(self.run_fixture(data,reader=reader)[0],data[0],kind)

    def test_missing_frame_prevents_both_repairs(self):
        for create in [label_fixture,collision_fixture]:
            data=create();data[4][:,:,1:]=data[4][:,:,:1]
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_hard_ocr_budget_and_component_budget(self):
        for create in [label_fixture,collision_fixture]:
            data=create();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':1})
            self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],1);self.assertTrue(audit['budget_exhausted'])
            out,audit=self.run_fixture(data,cfg={'max_components':0})
            self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0);self.assertTrue(audit['budget_exhausted'])

    def test_ocr_failure_does_not_mutate_inputs(self):
        data=label_fixture();before=copy.deepcopy(data[:4])
        with self.assertRaises(RuntimeError):self.run_fixture(data,reader=lambda *args:(_ for _ in ()).throw(RuntimeError('OCR failed')))
        self.assertEqual(data[:4],before)

    def test_disabled_wrapper_needs_no_artifacts(self):
        target={'unchanged':True}
        with patch.dict(os.environ,{'PCB_PIN_EVIDENCE_REPAIR':'0'}):
            out,audit=agent.finalize_pin_evidence_repair(target,Path('missing'),None,'missing')
        self.assertIs(out,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_foreign_image_before_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'export').mkdir()
            for name,doc in [('engineering_pin_text_result.json',{'image_sha256':'wrong'}),
                             ('engineering_graph_proposals.json',{'image_sha256':'wrong'}),
                             ('completed_component_records.json',{'records':[]})]:
                (stage/'export'/name).write_text(json.dumps(doc),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_pin_evidence_repair({},stage,np.zeros((20,20,3),np.uint8),'current')


if __name__=='__main__':unittest.main()
