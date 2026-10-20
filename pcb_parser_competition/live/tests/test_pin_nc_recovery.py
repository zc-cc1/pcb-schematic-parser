"""Evidence, row isolation and network preservation for missing NC recovery."""
import copy
import json
import os
from pathlib import Path
import re
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
from pcb_parser.pin_nc_recovery import recover_crossed_nc,neutral_group,crossed_lead
from pcb_parser.pin_geometry_recovery import LeadEvidence,DEFAULT


def fixture(side='left'):
    rgb=np.full((300,320,3),255,np.uint8)
    cv2.rectangle(rgb,(90,40),(230,255),(150,0,0),1)
    edge=90 if side=='left' else 230;sign=-1 if side=='left' else 1
    pins={};candidates=[]
    for row,number,name in [(60,'7','A'),(85,'8','B'),(110,'1','NC'),(135,'2','NC'),(160,'16','NC'),(185,'3','C'),(210,'4','D'),(235,'5','E')]:
        x=edge+sign*18
        cv2.line(rgb,(edge,row),(x,row),(150,0,0),1)
        if name=='NC':
            for slope in [-1,1]:cv2.line(rgb,(x-8,row-8*slope),(x+8,row+8*slope),(0,130,0),1)
        text_x=edge-20 if side=='left' else edge+4
        cv2.putText(rgb,number,(text_x,row-3),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
        if number not in {'2','16'}:
            pins['pin_'+number]={'pinname':name,'point':{'x':float(x),'y':float(300-row)},'keep':'original'}
            candidate={'candidate_id':'P'+number,'owner':'X','owner_status':'assigned','number':number,'pinname':name,
                       'point':[float(x),float(row)],'text_candidates':{'number':[]}}
            if number in {'7','8'}:
                candidate['text_candidates']['number']=[{'text':number,'confidence':.999,'bbox':[text_x,row-13,text_x+10,row-2],'source_view':v} for v in ['one','two']]
            candidates.append(candidate)
    target={'components':{'X':{'type':'box','bbox':[90,45,230,260]}},'pins':{'X':pins},
            'nets':{'N':{'hyperGraph':'(X.7,PEER.1)','edges':{'e':[{'x':72,'y':240},{'x':30,'y':240}]}}}}
    def reader(image,box,scale,rotation,rec,view):
        row=float(re.search(r'_(135|160)(?:\.0)?_',view).group(1));number='2' if row==135 else '16'
        if view.startswith('nc_name_'):return [{'text':'NC','confidence':.999,'bbox':box,'view':view}]
        if view.startswith('nc_group_'):return [{'text':number,'confidence':.999,'bbox':box,'view':view}]
        tx=edge-20 if side=='left' else edge+4
        return [{'text':number,'confidence':.999,'bbox':[tx,row-13,tx+12,row-1],'view':view}]
    return target,{'pin_candidates':candidates},rgb,reader


class NCRecoveryTests(unittest.TestCase):
    def run_fixture(self,data=None,reader=None,**kwargs):
        data=data or fixture()
        return recover_crossed_nc(*data[:3],reader=reader or data[3],**kwargs)

    def test_two_missing_nc_added_without_mutating_any_existing_records_or_nets(self):
        data=fixture();snapshot=copy.deepcopy(data[:2]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:2],snapshot);self.assertEqual(len(audit['changes']),2)
        self.assertEqual(out['components'],data[0]['components']);self.assertEqual(out['nets'],data[0]['nets'])
        self.assertEqual(out['pins']['X']['pin_16']['pinname'],'NC')
        for key,value in data[0]['pins']['X'].items():self.assertEqual(value,out['pins']['X'][key])
        self.assertEqual(audit['ocr_calls'],12)

    def test_idempotence_has_zero_extra_ocr(self):
        data=fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(again,out);self.assertEqual(audit['ocr_calls'],0)

    def test_right_side_uses_outside_digits_and_inside_names(self):
        out,audit=self.run_fixture(fixture('right'))
        self.assertEqual({r['number'] for r in audit['changes']},{'2','16'})
        self.assertGreater(out['pins']['X']['pin_16']['point']['x'],230)

    def test_missing_cross_or_only_one_diagonal_is_rejected_before_ocr(self):
        for single in [False,True]:
            data=fixture();rgb=data[2]
            for row in [135,160]:
                rgb[row-9:row+10,63:82]=255
                cv2.line(rgb,(90,row),(72,row),(150,0,0),1)
                if single:cv2.line(rgb,(64,row-8),(80,row+8),(0,130,0),1)
            out,audit=self.run_fixture(data)
            self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_outgoing_wire_blocks_nc_addition(self):
        data=fixture()
        for row in [135,160]:cv2.line(data[2],(25,row),(60,row),(0,130,0),1)
        out,audit=self.run_fixture(data)
        self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_unsupported_frame_is_rejected(self):
        data=fixture();data[2][40:256,89:92]=255
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_row_spacing_prevents_neighbor_text_ambiguity(self):
        data=fixture()
        for row in [135,160]:cv2.line(data[2],(90,row+10),(72,row+10),(150,0,0),1)
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_requires_two_distinct_observed_anchor_numbers(self):
        for kind in ['library','one_anchor','one_view']:
            data=fixture()
            if kind=='library':
                for p in data[1]['pin_candidates']:p['number_source']='library_pin_name'
            if kind=='one_anchor':data[1]['pin_candidates'][1]['text_candidates']['number']=[]
            if kind=='one_view':
                for p in data[1]['pin_candidates']:p['text_candidates']['number']=p['text_candidates']['number'][:1]
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_low_confidence_or_duplicate_views_do_not_confirm_number(self):
        for kind in ['low','duplicate']:
            data=fixture();base=data[3]
            def reader(*args):
                values=base(*args)
                if not args[-1].startswith('nc_name_'):
                    for v in values:
                        if kind=='low':v['confidence']=.979
                        else:v['view']='same'
                return values
            out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_one_strong_conflicting_number_abstains(self):
        data=fixture();base=data[3]
        def reader(*args):
            values=base(*args)
            if args[-1].startswith('nc_group_') and args[2]==3.:values[0]['text']='99'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_nc_name_must_be_exact_and_corroborated(self):
        for label in ['NCS','GND','N.C.']:
            data=fixture();base=data[3]
            def reader(*args):
                values=base(*args)
                if args[-1].startswith('nc_name_'):values[0]['text']=label
                return values
            out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_existing_number_is_never_overwritten(self):
        data=fixture();base=data[3]
        def reader(*args):
            values=base(*args)
            if not args[-1].startswith('nc_name_'):values[0]['text']='7'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_number_capacity_rejects_outlandish_ocr(self):
        data=fixture();base=data[3]
        def reader(*args):
            values=base(*args)
            if not args[-1].startswith('nc_name_'):values[0]['text']='99'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_network_reference_without_pin_record_blocks_identity(self):
        data=fixture();data[0]['nets']['N']['hyperGraph']='(X.2,X.16,PEER.1)'
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_conflicting_model_identity_is_preserved(self):
        data=fixture()
        for number in ['2','16']:data[1]['pin_candidates'].append({'owner':'X','number':number,'pinname':'NC','point':[72,235]})
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_duplicate_proposals_are_rejected_as_a_group(self):
        data=fixture();base=data[3]
        def reader(*args):
            values=base(*args)
            if not args[-1].startswith('nc_name_'):values[0]['text']='2'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_budget_never_commits_partial_ocr_row(self):
        for budget,expected in [(0,0),(5,0),(6,1),(11,1),(12,2)]:
            data=fixture();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':budget})
            self.assertEqual(len(audit['changes']),expected);self.assertLessEqual(audit['ocr_calls'],budget)
            self.assertEqual(out['nets'],data[0]['nets'])

    def test_zero_component_and_proposal_budgets(self):
        for cfg in [{'max_components':0},{'max_proposals':0}]:
            data=fixture();out,audit=self.run_fixture(data,cfg=cfg)
            self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_ocr_failure_keeps_input_untouched(self):
        data=fixture();before=copy.deepcopy(data[:2])
        def failing(*args):raise RuntimeError('OCR failed')
        with self.assertRaises(RuntimeError):self.run_fixture(data,reader=failing)
        self.assertEqual(data[:2],before)

    def test_neutral_group_preserves_fragmented_leading_digit(self):
        rgb=np.full((60,90,3),255,np.uint8)
        cv2.rectangle(rgb,(24,19),(26,23),(0,0,0),-1)
        cv2.rectangle(rgb,(25,25),(26,28),(0,0,0),-1)
        cv2.rectangle(rgb,(33,19),(39,28),(0,0,0),-1)
        image,box,parts=neutral_group(rgb,[10,15,45,32],30)
        self.assertEqual(parts,3);self.assertEqual(box,[24,19,40,29])
        self.assertEqual(image.shape[:2],(26,32))

    def test_clipped_glyphs_and_invalid_boxes_abstain(self):
        rgb=np.full((60,90,3),255,np.uint8);cv2.rectangle(rgb,(10,19),(18,28),(0,0,0),-1)
        self.assertIsNone(neutral_group(rgb,[10,15,45,32],30))
        self.assertIsNone(neutral_group(rgb,[-1,15,45,32],30))

    def test_disabled_wrapper_reads_no_artifacts(self):
        target={'sample':True}
        with patch.dict(os.environ,{'PCB_PIN_NC_RECOVERY':'0'}):
            result,audit=agent.finalize_pin_nc_recovery(target,Path('missing'),None,'bad')
        self.assertIs(result,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_foreign_image_before_ocr(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'export').mkdir()
            for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                (stage/'export'/name).write_text(json.dumps({'image_sha256':'foreign','prediction_only':True}),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_pin_nc_recovery(data[0],stage,data[2],'current',reader=data[3])


if __name__=='__main__':unittest.main()
