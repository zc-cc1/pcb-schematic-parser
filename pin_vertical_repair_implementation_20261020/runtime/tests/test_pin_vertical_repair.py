"""Atomic vertical contact identity, endpoint and physical-branch safeguards."""
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
from pcb_parser.pin_vertical_repair import repair_vertical_triplet


def fixture():
    rgb=np.full((400,350,3),255,np.uint8);cv2.rectangle(rgb,(80,130),(240,220),(150,0,0),1)
    cv2.line(rgb,(160,50),(160,89),(0,130,0),1);cv2.line(rgb,(160,90),(160,130),(150,0,0),1)
    for x in [110,210]:
        cv2.line(rgb,(x,220),(x,260),(150,0,0),1);cv2.line(rgb,(x,261),(x,320),(0,130,0),1)
    old={'pin_1':{'pinname':'Vout','point':{'x':110.,'y':162.},'keep':'one'},
         'pin_2':{'pinname':'Vin','point':{'x':210.,'y':162.},'keep':'two'},
         'pin_3':{'pinname':'GND','point':{'x':211.,'y':162.},'keep':'three'}}
    target={'components':{'X':{'type':'box','bbox':[80,180,240,270]},'G':{'type':'gnd','value':'GND','bbox':[150,345,170,365]}},
            'pins':{'X':old,'G':{'pin_1':{'pinname':'','point':{'x':160.,'y':350.}}}},
            'nets':{'LEFT':{'hyperGraph':'(X.1,PEER_A.1)','edges':{'e':[{'x':110,'y':162},{'x':110,'y':80}]}},
                    'RIGHT':{'hyperGraph':'(X.2,X.3,PEER_B.1)','edges':{'e':[{'x':210,'y':162},{'x':210,'y':80}]}}}}
    def tokens(text,box):return [{'text':text,'bbox':box,'confidence':.999,'source_view':v} for v in ['a','b']]
    candidates=[
        {'candidate_id':'TOP','owner':'X','owner_status':'assigned','number':None,'pinname':None,'point':[160.,90.],
         'text_candidates':{'number':tokens('2',[148,112,158,122]),'pinname':tokens('GND',[155,135,165,165])}},
        {'candidate_id':'A','owner':'X','owner_status':'assigned','number':'1','pinname':'Vout','point':[110.,238.],
         'number_source':'library_pin_name','text_candidates':{'number':[],'pinname':tokens('Vout',[105,183,115,218])}},
        {'candidate_id':'B','owner':'X','owner_status':'assigned','number':'2','pinname':'Vin','point':[210.,238.],
         'number_source':'library_pin_name','text_candidates':{'number':tokens('3',[198,228,208,238]),'pinname':tokens('Vin',[205,185,215,218])}},
        {'candidate_id':'C','owner':'X','owner_status':'assigned','number':'3','pinname':'GND','point':[211.,238.],
         'text_candidates':{'number':tokens('3',[198,228,208,238])}},
        {'candidate_id':'GROUND','owner':'G','owner_status':'assigned','number':'1','pinname':None,'point':[160.,50.],'text_candidates':{}}]
    mapping={'TOP':('GROUND_NET','PG'),'GROUND':('GROUND_NET','PG'),'A':('LEFT','PA'),'B':('RIGHT','PB'),'C':('RIGHT','PB')}
    attachments=[{'candidate_id':p['candidate_id'],'point':p['point'].copy(),'network_id':mapping[p['candidate_id']][0],
                  'physical_network_id':mapping[p['candidate_id']][1],'attachment_status':'attached'} for p in candidates]
    graph={'pin_attachments':attachments,'edges':[{'edge_id':'E_G','polyline':[[160.,50.],[160.,90.]]},
           {'edge_id':'E_A','polyline':[[110.,238.],[110.,320.]]},{'edge_id':'E_B','polyline':[[210.,238.],[210.,320.]]}],
           'networks':[{'network_id':'GROUND_NET','physical_network_ids':['PG'],'pin_candidate_ids':['TOP','GROUND'],'edge_ids':['E_G'],'unresolved_junction_ids':[]},
                       {'network_id':'LEFT','physical_network_ids':['PA'],'pin_candidate_ids':['A'],'edge_ids':['E_A'],'unresolved_junction_ids':['J_A']},
                       {'network_id':'RIGHT','physical_network_ids':['PB'],'pin_candidate_ids':['B','C'],'edge_ids':['E_B'],'unresolved_junction_ids':['J_B']}],
           'physical_networks':[{'network_id':'PG','pin_candidate_ids':['TOP','GROUND'],'edge_ids':['E_G'],'unresolved_junction_ids':[]}]}
    def reader(image,box,scale,rotation,rec,view):
        if '_top_' in view:number,name='2','GND'
        elif '_110.0_' in view or '_110_' in view:number,name='1','Vout'
        else:number,name='3','Vin'
        if view.startswith('vertical_glyph_'):text=number if '_r90_' in view else '?'
        else:text=name.casefold() if name=='Vin' else name
        return [{'text':text,'confidence':.999,'bbox':box,'view':view}]
    return target,{'pin_candidates':candidates},graph,rgb,reader


class VerticalRepairTests(unittest.TestCase):
    def run_fixture(self,data=None,reader=None,**kwargs):
        data=data or fixture()
        return repair_vertical_triplet(*data[:4],reader=reader or data[-1],**kwargs)

    def test_complete_repair_preserves_existing_wires_and_other_members(self):
        data=fixture();before=copy.deepcopy(data[:3]);out,audit=self.run_fixture(data)
        self.assertEqual(data[:3],before);self.assertEqual(len(audit['changes']),3)
        self.assertEqual(out['components'],data[0]['components']);self.assertEqual(audit['ocr_calls'],24)
        self.assertEqual(out['nets']['RIGHT']['hyperGraph'],'(PEER_B.1,X.3)')
        self.assertEqual(out['nets']['LEFT'],data[0]['nets']['LEFT'])
        self.assertEqual(out['nets']['RIGHT']['edges'],data[0]['nets']['RIGHT']['edges'])
        self.assertEqual(out['nets']['RESTORED_PG']['hyperGraph'],'(G.1,X.2)')
        self.assertEqual(out['pins']['X']['pin_2']['pinname'],'GND')
        self.assertEqual(out['pins']['X']['pin_3']['pinname'],'Vin')
        self.assertEqual(out['pins']['X']['pin_1']['keep'],'one')
        self.assertEqual(out['pins']['G'],data[0]['pins']['G'])

    def test_endpoints_are_measured_instead_of_using_the_old_length(self):
        out,audit=self.run_fixture()
        self.assertAlmostEqual(out['pins']['X']['pin_1']['point']['y'],139.,delta=1)
        self.assertAlmostEqual(out['pins']['X']['pin_2']['point']['y'],311.,delta=1)
        self.assertEqual([r['length'] for r in audit['groups'][0]['rows']],[41.,41.,41.])

    def test_copied_ground_segment_is_exactly_the_model_polyline(self):
        out,audit=self.run_fixture()
        self.assertEqual(out['nets']['RESTORED_PG']['edges'],{'edge_1':[{'x':160.,'y':350.},{'x':160.,'y':310.}]})
        self.assertEqual(audit['restored_networks'][0]['source_edge_ids'],['E_G'])

    def test_existing_junction_ambiguity_is_preserved_in_audit(self):
        out,audit=self.run_fixture()
        self.assertEqual(audit['groups'][0]['preserved_unresolved_junctions'],{'LEFT':['J_A'],'RIGHT':['J_B']})

    def test_idempotence_skips_ocr_after_correction(self):
        data=fixture();out,_=self.run_fixture(data);again,audit=self.run_fixture((out,*data[1:]))
        self.assertEqual(again,out);self.assertEqual(audit['ocr_calls'],0)

    def test_source_spelling_is_preserved_after_case_only_corroboration(self):
        out,audit=self.run_fixture();self.assertEqual(out['pins']['X']['pin_3']['pinname'],'Vin')
        row=next(r for r in audit['groups'][0]['rows'] if r['number']=='3')
        self.assertEqual({r['text'] for r in row['name_readings']},{'vin'})

    def test_correctly_separated_old_records_are_protected(self):
        data=fixture();data[0]['pins']['X']['pin_3']['point']['x']=230.
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_requires_three_records_and_a_library_assignment(self):
        for mode in ['missing','no_library']:
            data=fixture()
            if mode=='missing':data[0]['pins']['X'].pop('pin_3')
            else:
                for p in data[1]['pin_candidates']:p.pop('number_source',None)
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_missing_frame_or_inconsistent_lead_length_abstains(self):
        for mode in ['frame','length']:
            data=fixture()
            if mode=='frame':data[3][129:132,80:241]=255
            else:data[3][90:101,159:162]=255
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_missing_visible_wire_continuation_abstains(self):
        data=fixture();data[3][265:280,109:112]=255
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_existing_identity_requires_one_exact_model_attachment(self):
        for mode in ['point','duplicate','status']:
            data=fixture();a=data[2]['pin_attachments'][2]
            if mode=='point':a['point']=[0.,0.]
            if mode=='duplicate':data[2]['pin_attachments'].append(copy.deepcopy(a))
            if mode=='status':a['attachment_status']='ambiguous'
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_existing_ref_in_multiple_nets_abstains(self):
        data=fixture();data[0]['nets']['EXTRA']={'hyperGraph':'(X.2,PEER.1)','edges':{}}
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_original_names_require_two_reliable_views(self):
        for mode in ['low','single_view']:
            data=fixture()
            for p in data[1]['pin_candidates']:
                for t in p.get('text_candidates',{}).get('pinname',[]):
                    if mode=='low':t['confidence']=.979
                    else:t['source_view']='same'
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_fresh_number_low_confidence_or_duplicate_views_cannot_commit(self):
        for mode in ['low','same_view']:
            data=fixture();original=data[-1]
            def reader(*args):
                values=original(*args)
                if args[-1].startswith('vertical_glyph_'):
                    if mode=='low':values[0]['confidence']=.979
                    else:values[0]['view']='same'
                return values
            out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_ambiguous_rotation_cannot_commit(self):
        data=fixture();original=data[-1]
        def reader(*args):return original(*args[:-1],args[-1].replace('_r270_','_r90_'))
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_nonclosed_numbering_cannot_commit(self):
        data=fixture();original=data[-1]
        def reader(*args):
            values=original(*args)
            if args[-1].startswith('vertical_glyph_') and '_110' in args[-1] and '_r90_' in args[-1]:values[0]['text']='9'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_strong_fresh_name_conflict_rejects_entire_transaction(self):
        data=fixture();original=data[-1]
        def reader(*args):
            values=original(*args)
            if args[-1].startswith('vertical_tight_') and '_raw_' in args[-1] and args[2]==3.:values[0]['text']='OTHER'
            return values
        out,audit=self.run_fixture(data,reader=reader);self.assertEqual(out,data[0])

    def test_ambiguous_model_route_is_not_selected_by_proximity(self):
        data=fixture();top=copy.deepcopy(data[1]['pin_candidates'][0]);top['candidate_id']='ALT';data[1]['pin_candidates'].append(top)
        data[2]['pin_attachments'].append({'candidate_id':'ALT','point':top['point'],'network_id':'ALT_NET','physical_network_id':'ALT_PHYS','attachment_status':'attached'})
        data[2]['networks'].append({'network_id':'ALT_NET','pin_candidate_ids':['ALT'],'edge_ids':['E_G'],'unresolved_junction_ids':[]})
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_new_ground_branch_cannot_have_unresolved_junctions(self):
        for mode in ['logical','physical']:
            data=fixture();source=data[2]['networks'][0] if mode=='logical' else data[2]['physical_networks'][0]
            source['unresolved_junction_ids']=['J_UNRESOLVED']
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_new_ground_branch_requires_exact_two_contact_provenance(self):
        for mode in ['extra_candidate','edge_mismatch','physical_id']:
            data=fixture()
            if mode=='extra_candidate':data[2]['physical_networks'][0]['pin_candidate_ids'].append('EXTRA')
            if mode=='edge_mismatch':data[2]['physical_networks'][0]['edge_ids']=['E_A']
            if mode=='physical_id':data[2]['pin_attachments'][0]['physical_network_id']='OTHER'
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_ground_symbol_must_be_ground_and_have_matching_existing_tip(self):
        for mode in ['label','type','moved']:
            data=fixture()
            if mode=='label':data[0]['components']['G']['value']='VCC'
            if mode=='type':data[0]['components']['G']['type']='v'
            if mode=='moved':data[0]['pins']['G']['pin_1']['point']['x']=170.
            out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_ground_contact_already_exported_elsewhere_cannot_be_stolen(self):
        data=fixture();data[0]['nets']['EXTRA']={'hyperGraph':'(G.1,PEER.1)','edges':{}}
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_ground_segment_must_match_contact_endpoints(self):
        data=fixture();data[2]['edges'][0]['polyline'][0]=[160.,60.]
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_ground_wire_needs_visible_support_along_the_whole_path(self):
        data=fixture();data[3][60:70,159:162]=255
        out,audit=self.run_fixture(data);self.assertEqual(out,data[0])

    def test_missing_ground_tip_record_can_be_copied_from_exact_model_contact(self):
        data=fixture();data[0]['pins'].pop('G');out,audit=self.run_fixture(data)
        self.assertEqual(len(audit['changes']),3)
        self.assertEqual(out['pins']['G']['pin_1'],{'pinname':'','point':{'x':160.,'y':350.}})

    def test_budget_reserves_a_complete_component_transaction(self):
        for budget,expected in [(0,0),(23,0),(24,3)]:
            data=fixture();out,audit=self.run_fixture(data,cfg={'max_ocr_calls':budget})
            self.assertEqual(len(audit['changes']),expected);self.assertLessEqual(audit['ocr_calls'],budget)
            if not expected:self.assertEqual(out,data[0])
        data=fixture();out,audit=self.run_fixture(data,cfg={'max_components':0})
        self.assertEqual(out,data[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_ocr_failure_keeps_all_inputs_untouched(self):
        data=fixture();before=copy.deepcopy(data[:3]);original=data[-1];calls=0
        def reader(*args):
            nonlocal calls
            calls+=1
            if calls==20:raise RuntimeError('OCR failed')
            return original(*args)
        with self.assertRaises(RuntimeError):self.run_fixture(data,reader=reader)
        self.assertEqual(data[:3],before)

    def test_disabled_wrapper_reads_no_artifacts(self):
        target={'sample':True}
        with patch.dict(os.environ,{'PCB_PIN_VERTICAL_REPAIR':'0'}):
            out,audit=agent.finalize_pin_vertical_repair(target,Path('missing'),None,'bad')
        self.assertIs(out,target);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_foreign_image_before_ocr(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'export').mkdir()
            for name in ['engineering_pin_text_result.json','engineering_graph_proposals.json']:
                (stage/'export'/name).write_text(json.dumps({'prediction_only':True,'image_sha256':'foreign'}),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_pin_vertical_repair(data[0],stage,data[3],'current',reader=data[-1])


if __name__=='__main__':unittest.main()
