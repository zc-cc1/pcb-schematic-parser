from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.targeted_pin_ocr import adopt_missing_fields, collect_readings, existing_readings, select_rois

read=lambda name:json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))
CFG=read('targeted_pin_ocr_v1.json');TEXT=read('pin_text_v2.json');PASSIVE=read('pin_output_rules_v1.json')


def fixture(kind='ic'):
    record={'component_id':'U1','detection_index':0,'type':kind,'designator':'U1','Name':None,'value':None,
            'bbox_top_left':[100,100,300,300],'fields':{}}
    component={**record,'class_name':kind,'confidence':.99}
    pins={'size':[500,500],'coordinate_origin':'top_left','components':[component],
          'pin_candidates':[{'candidate_id':f'P{i}','point':[100,y],'owner':'U1','owner_status':'assigned',
                             'owner_detection_index':0,'confidence':.99,'number':None,'pinname':None,
                             'number_status':'no_reliable_text','pinname_status':'no_reliable_text','text_candidates':{},'text_evidence':{}}
                            for i,y in enumerate([180,220])]}
    graph={'size':pins['size'],'pin_attachments':deepcopy(pins['pin_candidates']),
           'networks':[{'network_id':'N','pin_candidate_ids':['P0','P1'],'unresolved_junction_ids':[],'edge_ids':['E']}],
           'edges':[{'edge_id':'E','polyline':[[100,180],[100,220]]}],'gap_candidates':[]}
    return pins,graph,[record]


def readings(value='3',box=(78,174,96,186),views=None,score=.999):
    views=views or ['missing_pin_P0_s2.0_r0','missing_pin_P0_s3.0_r0']
    return [{'text_id':str(i),'text':value,'bbox':list(box),'confidence':score,'view':view,
             'polygon':[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]}
            for i,view in enumerate(views)]


def run(p,g,r,texts,ids=None,cfg=CFG):
    return adopt_missing_fields(p,g,r,texts,{'P0'} if ids is None else ids,cfg,TEXT,PASSIVE)


class TargetedPinOCRTests(unittest.TestCase):
    def test_fills_actual_number_and_name_without_changing_inputs_or_geometry(self):
        p,g,r=fixture();snapshot=deepcopy((p,g,r))
        updated,graph,audit=run(p,g,r,readings()+readings('CLK',[120,174,155,186]))
        self.assertEqual(updated['pin_candidates'][0]['number'],'3')
        self.assertEqual(updated['pin_candidates'][0]['pinname'],'CLK')
        self.assertEqual(audit['summary']['missing_numbers_added'],1)
        self.assertEqual(audit['summary']['missing_names_added'],1)
        self.assertEqual(graph['edges'],g['edges']);self.assertEqual(graph['networks'],g['networks'])
        self.assertEqual((p,g,r),snapshot)

    def test_existing_fields_are_not_overwritten_by_local_disagreement(self):
        p,g,r=fixture();p['pin_candidates'][0].update(number='7',pinname='RESET')
        updated,_,audit=run(p,g,r,readings()+readings('CLK',[120,174,155,186]))
        self.assertEqual(updated['pin_candidates'][0]['number'],'7')
        self.assertEqual(updated['pin_candidates'][0]['pinname'],'RESET')
        self.assertEqual(audit['changes'],[])

    def test_duplicate_number_on_same_owner_is_withheld(self):
        p,g,r=fixture();p['pin_candidates'][1]['number']='3'
        updated,_,audit=run(p,g,r,readings())
        self.assertIsNone(updated['pin_candidates'][0]['number'])
        self.assertIn('number_already_claimed_on_component',audit['summary']['withheld_reason_counts'])

    def test_occupied_name_glyph_cannot_be_reassigned_as_number(self):
        p,g,r=fixture();p['pin_candidates'][1].update(pinname='OLD',text_evidence={'pinname':{'bbox':[78,174,96,186]}})
        updated,_,audit=run(p,g,r,readings())
        self.assertIsNone(updated['pin_candidates'][0]['number'])

    def test_global_views_alone_cannot_trigger_local_field_recovery(self):
        p,g,r=fixture();updated,_,audit=run(p,g,r,readings(views=['global','tile']))
        self.assertIsNone(updated['pin_candidates'][0]['number'])
        self.assertIn('no_matching_local_ROI_observation',audit['summary']['withheld_reason_counts'])

    def test_one_local_view_plus_original_same_glyph_is_supported(self):
        p,g,r=fixture();updated,_,audit=run(p,g,r,readings(views=['global','missing_pin_P0_s2.0_r0']))
        self.assertEqual(updated['pin_candidates'][0]['number'],'3')

    def test_low_confidence_and_single_view_remain_unresolved(self):
        for texts in [readings(score=.95),readings(views=['missing_pin_P0_s2.0_r0'])]:
            p,g,r=fixture();updated,_,_=run(p,g,r,texts)
            self.assertIsNone(updated['pin_candidates'][0]['number'])

    def test_conflicting_supported_same_glyph_spellings_are_not_selected(self):
        p,g,r=fixture();texts=readings()+readings('4',views=['missing_pin_P0_s2.0_r90','missing_pin_P0_s3.0_r90'])
        updated,_,_=run(p,g,r,texts)
        self.assertIsNone(updated['pin_candidates'][0]['number'])

    def test_passive_package_number_cannot_be_reintroduced(self):
        p,g,r=fixture('capacitor');updated,_,audit=run(p,g,r,readings('0402',[80,174,98,186]))
        self.assertIsNone(updated['pin_candidates'][0]['number'])
        self.assertIn('passive_number_outside_supported_literal_range',audit['summary']['withheld_reason_counts'])

    def test_non_target_candidate_is_not_modified(self):
        p,g,r=fixture();updated,_,audit=run(p,g,r,readings(),ids={'P1'})
        self.assertEqual(updated,p)

    def test_fragment_geometry_is_not_promoted_to_whole_glyph(self):
        p,g,r=fixture();p['pin_candidates'][0]['text_candidates']={'number':[
            {'text':'0402','source_text':'cap_0402','source_view':'global','confidence':.99,'bbox':[78,174,96,186]},
            {'text':'1','source_text':'1 2','source_view':'tile','confidence':.99,'bbox':[78,174,96,186],'approximate_split_geometry':True}]}
        self.assertEqual(existing_readings(p),[])

    def test_candidate_reconstruction_keeps_position_view_and_literal(self):
        p,g,r=fixture();row={'text':'3','source_text':'3','source_view':'global','confidence':.99,'bbox':[78,174,96,186]}
        p['pin_candidates'][0]['text_candidates']={'number':[row,row]}
        result=existing_readings(p)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['bbox'],row['bbox']);self.assertEqual(result[0]['view'],'global')

    def test_model_owner_mismatch_is_rejected(self):
        p,g,r=fixture();p['pin_candidates'][0]['owner']='WRONG'
        with self.assertRaises(ValueError):run(p,g,r,readings())

    def test_ROI_budget_prefers_visible_evidence_and_skips_unresolved_identity(self):
        p,g,r=fixture();p['pin_candidates'][1]['text_candidates']={'number':[{'confidence':.999,'approximate_split_geometry':False}]}
        rois,skipped=select_rois(p,r,{'networks':[]},{**CFG,'max_rois':1},TEXT)
        self.assertEqual(rois[0]['candidate_id'],'P1');self.assertEqual(skipped,1)
        r[0]['designator']=None
        self.assertEqual(select_rois(p,r,{'networks':[]},CFG,TEXT)[0],[])

    def test_rotated_and_scaled_boxes_return_to_original_pixels(self):
        crop=[60,150,170,210];rois=[{'candidate_id':'P0','crop':crop}]
        calls=[]
        def engine(image,**options):
            i=len(calls);calls.append(options)
            rotation=CFG['rotations'][i//len(CFG['scales'])];scale=CFG['scales'][i%len(CFG['scales'])]
            points=np.array([[78,174],[96,174],[96,186],[78,186]],float)-[60,150]
            if rotation==90:points=np.column_stack((59-points[:,1],points[:,0]))
            return SimpleNamespace(boxes=[points*scale],txts=['3'],scores=[.999])
        texts,audit=collect_readings(np.zeros((500,500,3),np.uint8),rois,engine,CFG)
        self.assertEqual(len(texts),6)
        for row in texts:np.testing.assert_allclose(row['bbox'],[78,174,96,186])

    def test_duplicate_scales_are_rejected(self):
        with self.assertRaises(ValueError):collect_readings(np.zeros((500,500,3),np.uint8),[],None,{**CFG,'scales':[2,2]})

    def test_repeated_rotated_fragment_is_withheld_when_whole_spelling_is_observed(self):
        p,g,r=fixture()
        texts=readings('N',[120,174,138,186],views=['missing_pin_P0_s2.0_r0','missing_pin_P0_s3.0_r0','missing_pin_P0_s4.0_r90'])+readings('IN',[120,174,138,186],views=['global'])
        updated,_,audit=run(p,g,r,texts)
        self.assertIsNone(updated['pin_candidates'][0]['pinname'])
        self.assertIn('supported_longer_observed_spelling_requires_review',audit['summary']['withheld_reason_counts'])

    def test_whole_string_at_a_different_row_does_not_block_a_valid_single_letter(self):
        p,g,r=fixture()
        texts=readings('F',[120,174,138,186])+readings('FB',[120,214,138,226],views=['global'])
        updated,_,audit=run(p,g,r,texts)
        self.assertEqual(updated['pin_candidates'][0]['pinname'],'F')

    def test_new_number_updates_derived_reference_without_changing_members(self):
        p,g,r=fixture();g['networks'][0]['official_pin_references']=[]
        updated,graph,audit=run(p,g,r,readings())
        self.assertEqual(graph['networks'][0]['official_pin_references'],['U1.3'])
        self.assertEqual(graph['networks'][0]['pin_candidate_ids'],g['networks'][0]['pin_candidate_ids'])
        self.assertEqual(audit['derived_reference_additions'][0]['reference'],'U1.3')
