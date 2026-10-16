"""Image-text evidence, row isolation and hypothesis preservation tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.pin_text_v2 import bind_pin_text_v2,glyph_groups,resolve_spelling,adopt_pin_fields
from pcb_parser.pin_text import bind_pin_text
from pcb_parser.component_identity_recovery import recover_component_identities

CFG=json.loads((ROOT/'configs/pin_text_v2.json').read_text(encoding='utf-8'))


def ownership(points,box=(100,20,300,300)):
    return {'size':[500,500],'coordinate_origin':'top_left','components':[
        {'component_id':'UNRESOLVED_DET_0001','detection_index':0,'class_name':'ic','type':'ic','bbox_top_left':list(box),'confidence':.99}],
        'pin_candidates':[{'candidate_id':f'P{i}','point':list(p),'confidence':.99-i*.01,
                           'owner':'UNRESOLVED_DET_0001','owner_status':'assigned','owner_detection_index':0} for i,p in enumerate(points)]}


def text(value,box,view='global',confidence=.99):
    return {'text_id':view+value,'text':value,'bbox':list(box),'confidence':confidence,'view':view,
            'polygon':[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]}


def record(box=(100,20,300,300)):
    return {'component_id':'UNRESOLVED_DET_0001','detection_index':0,'type':'ic','bbox_top_left':list(box),
            'designator':None,'Name':None,'value':None,'fields':{},'duplicate_of_detection_index':None}


class PinTextV2Tests(unittest.TestCase):
    def test_near_tip_hypotheses_preserved_and_single_OCR_assignment(self):
        source=ownership([[100,100],[101,101],[100,140]]);before=deepcopy(source)
        result,debug=bind_pin_text_v2(source,[text('3',[78,94,96,106]),text('CLK',[120,94,156,106])],CFG)
        self.assertEqual([p['point'] for p in result['pin_candidates']],[p['point'] for p in source['pin_candidates']])
        self.assertEqual(result['pin_candidates'][0]['number'],'3');self.assertEqual(result['pin_candidates'][0]['pinname'],'CLK')
        self.assertEqual(result['pin_candidates'][1]['number_status'],'duplicate_tip_hypothesis')
        self.assertEqual(debug['tip_hypothesis_secondary_to_primary'],{'P1':'P0'});self.assertEqual(source,before)

    def test_distinct_dense_rows_keep_repeated_GND_names(self):
        source=ownership([[100,100],[100,110],[100,120]])
        readings=[text('GND',[120,y-4,150,y+4],f'row{y}') for y in [100,110,120]]
        result,_=bind_pin_text_v2(source,readings,CFG)
        self.assertEqual([p['pinname'] for p in result['pin_candidates']],['GND']*3)

    def test_same_row_normal_direction_candidates_compete_without_moving_points(self):
        source=ownership([[300,100],[313,100],[300,115]])
        source['pin_candidates'][1]['confidence']=1.0
        result,debug=bind_pin_text_v2(source,[text('GND',[260,94,289,106])],CFG)
        self.assertEqual(result['pin_candidates'][0]['pinname'],'GND')
        self.assertIsNone(result['pin_candidates'][1]['pinname'])
        self.assertEqual(debug['tip_hypothesis_secondary_to_primary'],{'P1':'P0'})
        self.assertEqual(result['pin_candidates'][1]['point'],[313,100])

    def test_wide_crop_cannot_bridge_two_row_glyph_groups(self):
        tokens=[{'text':'GND','bbox':b,'confidence':.99} for b in [[120,95,150,105],[120,115,150,125],[120,95,150,125]]]
        self.assertEqual(len(glyph_groups(tokens)),3)

    def test_same_glyph_conflicting_full_spellings_stay_unresolved(self):
        source=ownership([[100,100]])
        result,_=bind_pin_text_v2(source,[text('6',[78,94,96,106]),text('8',[78,94,96,106],'tile')],CFG)
        self.assertIsNone(result['pin_candidates'][0]['number'])

    def test_observed_full_number_preferred_to_weaker_partial_crop(self):
        source=ownership([[100,100]])
        ts=[text('12',[78,94,96,106],v) for v in ['global','tile0','tile90']]+[text('2',[78,94,96,106],'other',.97)]
        result,debug=bind_pin_text_v2(source,ts,CFG)
        self.assertEqual(result['pin_candidates'][0]['number'],'12');self.assertTrue(debug['spelling_decisions'])

    def test_repeated_same_view_does_not_create_dominant_votes(self):
        ranked=[(1,{'text':'BOOT0','confidence':.99,'source_view':'global'})]*4+[(1.1,{'text':'BOOTO','confidence':.99,'source_view':'tile'})]
        chosen,_=resolve_spelling(ranked,CFG);self.assertIsNone(chosen)

    def test_low_confidence_unvoted_partial_is_handled_without_vote_mutation(self):
        ranked=[(1,{'text':'12','confidence':.99,'source_view':v}) for v in ['global','tile']]
        ranked.append((1.1,{'text':'2','confidence':.9,'source_view':'weak'}))
        chosen,_=resolve_spelling(ranked,CFG);self.assertEqual(chosen[1]['text'],'12')

    def test_internal_D0_is_name_without_using_same_glyph_as_ball_number(self):
        result,_=bind_pin_text_v2(ownership([[100,100]]),[text('D0',[107,94,123,106])],CFG)
        self.assertIsNone(result['pin_candidates'][0]['number']);self.assertEqual(result['pin_candidates'][0]['pinname'],'D0')

    def test_external_net_name_does_not_supply_pinname(self):
        result,_=bind_pin_text_v2(ownership([[100,100]]),[text('NET_ALERT',[25,94,90,106])],CFG)
        self.assertIsNone(result['pin_candidates'][0]['pinname'])

    def test_complete_observed_multiword_name_is_preserved(self):
        result,_=bind_pin_text_v2(ownership([[100,100]]),[text('ADC Input',[120,94,195,106])],CFG)
        self.assertEqual(result['pin_candidates'][0]['pinname'],'ADC Input')

    def test_name_number_line_straddling_body_is_not_a_multiword_name(self):
        result,_=bind_pin_text_v2(ownership([[300,100]]),[text('GND 15',[270,94,320,106])],CFG)
        self.assertIsNone(result['pin_candidates'][0]['pinname'])

    def test_component_header_glyph_excluded_from_pin_fields(self):
        result,_=bind_pin_text_v2(ownership([[100,100]]),[text('CHIP123',[120,94,170,106])],CFG,
                                 [{'fields':{'Name':{'bbox':[120,94,170,106]}}}])
        self.assertIsNone(result['pin_candidates'][0]['pinname'])

    def test_large_top_left_visible_reference_recovers_without_center_bias(self):
        box=[100,100,400,450];source=ownership([[100,150],[400,150],[100,400]],box)
        ts=[text('U1',[104,75,124,95],v) for v in ['global','tile']]
        records,debug=recover_component_identities(source,ts,[record(box)],CFG['identity_recovery'])
        self.assertEqual(records[0]['designator'],'U1');self.assertEqual(debug['new_identities'],1)

    def test_unit_suffix_and_connector_title_inside_predicted_box_are_supported(self):
        for name in ['U1H','H1','CN1','LD2']:
            source=ownership([[100,100],[300,100],[100,200],[300,200]])
            ts=[text(name,[105,22,140,40],v) for v in ['global','tile']]
            records,_=recover_component_identities(source,ts,[record()],CFG['identity_recovery'])
            self.assertEqual(records[0]['designator'],name)

    def test_side_ball_identifier_cannot_become_component_designator(self):
        source=ownership([[100,100],[100,200],[300,100],[300,200]])
        ts=[text('U18',[75,150,95,165],v) for v in ['global','tile']]
        records,_=recover_component_identities(source,ts,[record()],CFG['identity_recovery'])
        self.assertIsNone(records[0]['designator'])

    def test_existing_identity_and_unknown_fields_are_not_replaced(self):
        source=ownership([[100,100],[300,100]])
        old=record();old.update(component_id='U9',designator='U9',Name='Chip123')
        records,_=recover_component_identities(source,[text('U1',[105,0,130,18],'global')],[old],CFG['identity_recovery'])
        self.assertEqual(records,[old]);result,_=bind_pin_text_v2(source,[],CFG)
        self.assertTrue(all(p['number'] is None and p['pinname'] is None for p in result['pin_candidates']))

    def test_adoption_preserves_old_numbers_and_fills_only_supported_names(self):
        source=ownership([[100,100]]);legacy=json.loads((ROOT/'configs/pin_text_v1.json').read_text(encoding='utf-8'))
        baseline,_=bind_pin_text(source,[],legacy);baseline['pin_candidates'][0].update(number='7',number_status='observed_OCR_text')
        baseline['pin_candidates'][0]['text_evidence']['number']={'bbox':[78,94,96,106],'text':'7'}
        exp,_=bind_pin_text_v2(source,[text('CLK',[120,94,156,106],v) for v in ['global','tile']],CFG)
        adopted,debug=adopt_pin_fields(baseline,exp,CFG['adoption'])
        self.assertEqual(adopted['pin_candidates'][0]['number'],'7');self.assertEqual(adopted['pin_candidates'][0]['pinname'],'CLK')
        self.assertEqual(len(debug['adopted_name_additions']),1)

    def test_new_name_cannot_reuse_a_glyph_already_assigned_as_number(self):
        source=ownership([[100,100]]);legacy=json.loads((ROOT/'configs/pin_text_v1.json').read_text(encoding='utf-8'))
        baseline,_=bind_pin_text(source,[],legacy);baseline['pin_candidates'][0].update(number='7',number_status='observed_OCR_text')
        baseline['pin_candidates'][0]['text_evidence']['number']={'bbox':[120,94,156,106],'text':'7'}
        exp,_=bind_pin_text_v2(source,[text('CLK',[120,94,156,106],v) for v in ['global','tile']],CFG)
        adopted,_=adopt_pin_fields(baseline,exp,CFG['adoption'])
        self.assertIsNone(adopted['pin_candidates'][0]['pinname'])

    def test_single_view_name_is_only_an_experiment(self):
        source=ownership([[100,100]]);legacy=json.loads((ROOT/'configs/pin_text_v1.json').read_text(encoding='utf-8'))
        baseline,_=bind_pin_text(source,[],legacy);exp,_=bind_pin_text_v2(source,[text('CLK',[120,94,156,106])],CFG)
        adopted,_=adopt_pin_fields(baseline,exp,CFG['adoption'])
        self.assertIsNone(adopted['pin_candidates'][0]['pinname'])

    def test_duplicate_detection_zero_index_is_not_recovered(self):
        source=ownership([[100,100],[300,100]]);old=record();old['duplicate_of_detection_index']=0
        records,_=recover_component_identities(source,[text('U1',[105,0,130,18],v) for v in ['global','tile']],[old],CFG['identity_recovery'])
        self.assertIsNone(records[0]['designator'])

    def test_repeated_visible_header_names_do_not_receive_invented_canonical_suffixes(self):
        a=ownership([[100,100],[300,100]]);b=ownership([[100,400],[300,400]],[100,320,300,480])
        b['components'][0].update(detection_index=1,component_id='UNRESOLVED_DET_0002')
        for i,p in enumerate(b['pin_candidates']):p.update(candidate_id=f'B{i}',owner_detection_index=1,owner='UNRESOLVED_DET_0002')
        a['components']+=b['components'];a['pin_candidates']+=b['pin_candidates']
        r0=record();r1=record([100,320,300,480]);r1.update(detection_index=1,component_id='UNRESOLVED_DET_0002')
        ts=[text('CN1',box,view) for box in [[105,0,140,18],[105,302,140,318]] for view in ['global','tile']]
        records,debug=recover_component_identities(a,ts,[r0,r1],CFG['identity_recovery'])
        self.assertTrue(all(r['designator'] is None for r in records));self.assertEqual(len(debug['withheld_duplicates']),1)

    def test_name_cannot_reuse_number_glyph_from_other_candidate(self):
        source=ownership([[100,100],[100,110]]);legacy=json.loads((ROOT/'configs/pin_text_v1.json').read_text(encoding='utf-8'))
        baseline,_=bind_pin_text(source,[],legacy)
        baseline['pin_candidates'][1].update(number='7',number_status='observed_OCR_text')
        baseline['pin_candidates'][1]['text_evidence']['number']={'bbox':[120,94,156,106],'text':'7'}
        exp,_=bind_pin_text_v2(source,[text('CLK',[120,96,156,104],v) for v in ['global','tile']],CFG)
        adopted,_=adopt_pin_fields(baseline,exp,CFG['adoption']);self.assertIsNone(adopted['pin_candidates'][0]['pinname'])


if __name__=='__main__':unittest.main()
