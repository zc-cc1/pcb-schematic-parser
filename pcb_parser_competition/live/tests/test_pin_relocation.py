"""Evidence and attachment safeguards for replacing existing pin coordinates."""
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
from pcb_parser.pin_relocation import DEFAULT, relocate_unsupported_pins, unsupported_candidate


def fixture():
    rgb=np.full((240,220,3),255,np.uint8)
    cv2.rectangle(rgb,(50,40),(130,180),(150,0,0),1)
    pins={};candidates=[]
    for n,row in [('1',70),('3',130),('4',160)]:
        cv2.line(rgb,(30,row),(50,row),(150,0,0),1)
        pins['pin_'+n]={'pinname':'FIELD'+n,'point':{'x':30.,'y':240.-row}}
        candidates.append({'candidate_id':'C'+n,'owner':'X','point':[30.,row], 'number':n,
                           'number_source':'observed_OCR_text','text_candidates':{'number':[
                               {'text':n,'confidence':.999,'bbox':[41,row-10,47,row-4]}]}})
    cv2.line(rgb,(30,100),(50,100),(150,0,0),1)
    cv2.line(rgb,(10,100),(29,100),(0,150,0),1)
    cv2.putText(rgb,'2',(40,97),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
    pins['pin_2']={'pinname':'VCC','point':{'x':128.,'y':62.}}
    weak={'candidate_id':'WEAK','owner':'X','owner_status':'assigned','point':[128.,178.],
          'number':'2','pinname':'VCC','number_source':'library_elimination',
          'pinname_source':'library_elimination','confidence':.66,'nearby_wire_probability_max':.01,
          'text_evidence':{}}
    candidates.append(weak)
    target={'components':{'X':{'type':'box','bbox':[50,60,130,200]}},'pins':{'X':pins},'nets':{}}
    doc={'pin_candidates':candidates}
    graph={'pin_attachments':[{'candidate_id':'WEAK','point':[128.,178.],
                              'attachment_status':'no_wire_within_radius','network_id':None}], 'networks':[],'edges':[]}
    records=[{'component_id':'X','bbox_top_left':[50,40,130,180]}]
    def reader(image,box,scale,rotation,recognition_only,view):
        return [{'text':'2' if recognition_only else 'VCC','confidence':.999,
                 'bbox':box if recognition_only else [60,96,85,108],'view':view}]
    return target,doc,graph,records,rgb,reader


class RelocationTests(unittest.TestCase):
    def run_fixture(self,items=None,**kwargs):
        target,doc,graph,records,rgb,reader=items or fixture()
        return relocate_unsupported_pins(target,doc,graph,records,rgb,reader=kwargs.pop('reader',reader),**kwargs)

    def test_valid_relocation_and_idempotence_without_input_mutation(self):
        data=fixture();before=copy.deepcopy(data[:4]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:4],before)
        self.assertEqual(len(audit['changes']),1)
        self.assertEqual(out['pins']['X']['pin_2']['point']['y'],140.)
        self.assertEqual(out['pins']['X']['pin_2']['pinname'],'VCC')
        self.assertEqual(out['components'],data[0]['components'])
        again,_=self.run_fixture((out,*data[1:]));self.assertEqual(out,again)

    def test_directly_observed_or_high_confidence_fields_are_protected(self):
        for update in [{'text_evidence':{'number':{'text':'2'}}}, {'number_source':'observed_OCR_text'},
                       {'pinname_source':'observed_OCR_text'}, {'confidence':.9},
                       {'text_candidates':{'number':[{'text':'2','confidence':.999}]}},
                       {'nearby_wire_probability_max':.3}, {'text_hypothesis_primary_candidate_id':'OTHER'}]:
            data=fixture();data[1]['pin_candidates'][-1].update(update)
            self.assertEqual(self.run_fixture(data)[1]['changes'],[],update)

    def test_existing_network_membership_is_protected(self):
        data=fixture();data[0]['nets']['N']={'hyperGraph':'(X.2,Y.1)','edges':{}}
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_contradictory_model_network_membership_is_protected(self):
        data=fixture();data[2]['networks']=[{'network_id':'N','pin_candidate_ids':['WEAK']}]
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_attached_missing_duplicate_or_mismatched_attachment_is_protected(self):
        for update in [{'attachment_status':'attached','network_id':'N'}, {'point':[127,178]}]:
            data=fixture();data[2]['pin_attachments'][0].update(update)
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)
        data=fixture();data[2]['pin_attachments']=[]
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)
        data=fixture();data[2]['pin_attachments']*=2
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_already_changed_point_or_field_is_protected(self):
        for update in [{'point':{'x':125.,'y':62.}}, {'pinname':'OTHER'}]:
            data=fixture();data[0]['pins']['X']['pin_2'].update(update)
            self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_duplicate_number_candidate_is_protected(self):
        data=fixture();data[1]['pin_candidates'].append({**data[1]['pin_candidates'][-1],'candidate_id':'DUP'})
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_multiple_empty_rows_and_too_few_known_rows_abstain(self):
        data=fixture();cv2.line(data[4],(30,115),(50,115),(150,0,0),1)
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)
        data=fixture();data[0]['pins']['X'].pop('pin_4')
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_old_point_with_visible_green_wire_is_protected(self):
        data=fixture();data[4][176:180,126:130]=[0,150,0]
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_no_observed_number_layout_abstains(self):
        data=fixture()
        for c in data[1]['pin_candidates']:c['text_candidates']={}
        self.assertEqual(self.run_fixture(data)[1]['ocr_calls'],0)

    def test_conflicting_or_unconfirmed_fields_abstain(self):
        data=fixture();original=data[-1]
        for kind in ['number','name','conflict','duplicate_view']:
            def reader(*args):
                result=original(*args)
                if kind=='number' and args[4]:result[0]['text']='9'
                if kind=='name' and not args[4]:result[0]['text']='OTHER'
                if kind=='conflict' and not args[4] and args[2]==3.:result[0]['text']='OTHER'
                if kind=='duplicate_view':result[0]['view']='one'
                return result
            self.assertEqual(self.run_fixture(data,reader=reader)[1]['changes'],[],kind)

    def test_ocr_budget_is_hard_limit(self):
        out,audit=self.run_fixture(cfg={'max_ocr_calls':1})
        self.assertEqual(audit['changes'],[]);self.assertEqual(audit['ocr_calls'],1)
        self.assertTrue(audit['budget_exhausted'])

    def test_unique_existing_network_attaches_and_preserves_wire_geometry(self):
        data=fixture();data[0]['nets']['N']={'hyperGraph':'(Y.1)','edges':{'e':[{'x':10,'y':140},{'x':30,'y':140}]}}
        data[2]['networks']=[{'network_id':'N','edge_ids':['E'],'unresolved_junction_ids':[]}]
        data[2]['edges']=[{'edge_id':'E','polyline':[[10,100],[30,100]]}]
        out,audit=self.run_fixture(data)
        self.assertEqual(out['nets']['N']['hyperGraph'],'(X.2,Y.1)')
        self.assertEqual(out['nets']['N']['edges'],data[0]['nets']['N']['edges'])
        self.assertIn('after',audit['network_changes'][0])

    def test_missing_exported_network_is_withheld_without_dropping_pin_fix(self):
        data=fixture();data[2]['networks']=[{'network_id':'N','edge_ids':['E']}]
        data[2]['edges']=[{'edge_id':'E','polyline':[[10,100],[30,100]]}]
        out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),1);self.assertEqual(out['nets'],{})
        self.assertEqual(audit['network_changes'][0]['candidate_networks'],['N'])
        self.assertNotIn('after',audit['network_changes'][0])

    def test_disabled_wrapper_needs_no_artifacts(self):
        target={'unchanged':True}
        with patch.dict(os.environ,{'PCB_PIN_RELOCATION':'0'}):
            result,audit=agent.finalize_pin_relocation(target,Path('missing'),None,'missing')
        self.assertIs(result,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_wrong_image_before_ocr(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'export').mkdir()
            for name,doc in [('engineering_pin_text_result.json',{'image_sha256':'wrong'}),
                             ('engineering_graph_proposals.json',{'image_sha256':'wrong'}),
                             ('completed_component_records.json',{'records':[]})]:
                (stage/'export'/name).write_text(json.dumps(doc),encoding='utf-8')
            with self.assertRaises(ValueError):
                agent.finalize_pin_relocation({},stage,np.zeros((20,20,3),np.uint8),'expected')


if __name__=='__main__':unittest.main()
