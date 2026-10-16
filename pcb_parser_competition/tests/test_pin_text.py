"""Pin text evidence, role ambiguity, and geometry regression cases."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from pcb_parser.pin_text import bind_pin_text,local_views
from run_pin_text import validate_image,attach_identities

CFG=json.loads((ROOT/'configs/pin_text_v1.json').read_text(encoding='utf-8'))


def text(value,box,confidence=.99):
    x1,y1,x2,y2=box
    return {'text':value,'bbox':box,'polygon':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],'confidence':confidence,'view':'test'}


def owned(points,kind='ic'):
    return {'size':[400,400],'coordinate_origin':'top_left','image_sha256':'same',
            'components':[{'component_id':'U1','detection_index':0,'type':kind,'class_name':kind,
                           'bbox_top_left':[100,20,300,300],'confidence':.95}],
            'pin_candidates':[{'candidate_id':f'p{i}','point':xy,'confidence':.99,
                               'owner_status':'assigned','owner':'U1','owner_detection_index':0,'owner_candidates':[]} for i,xy in enumerate(points)]}


class PinTextTests(unittest.TestCase):
    def run_case(self,points,texts,kind='ic',records=None):
        return bind_pin_text(owned(points,kind),texts,CFG,records)[0]

    def test_number_and_name_use_separate_text_preserve_tip_and_case(self):
        point=[99.25,100.5]
        result=self.run_case([point],[text('R8',[75,87,97,103]),text('XTALn',[120,94,175,106])])
        pin=result['pin_candidates'][0]
        self.assertEqual((pin['number'],pin['pinname']),('R8','XTALn'))
        self.assertEqual(pin['point'],point)

    def test_no_text_does_not_guess_sequence(self):
        result=self.run_case([[100,60],[100,100]],[])
        self.assertTrue(all(p['number'] is None for p in result['pin_candidates']))

    def test_empty_detector_and_pin_results_have_explicit_zero_counts(self):
        source=owned([])
        source['components']=[]
        result,_=bind_pin_text(source,[],CFG)
        for key in ['candidate_count','owned_candidates','number_observed','name_observed','number_and_name_observed']:
            self.assertEqual(result['pin_text_summary'][key],0)

    def test_unresolved_owner_cannot_gain_number(self):
        source=owned([[100,100]])
        source['pin_candidates'][0].update(owner=None,owner_status='ambiguous_owner')
        result,_=bind_pin_text(source,[text('3',[78,90,95,105])],CFG)
        self.assertIsNone(result['pin_candidates'][0]['number'])
        self.assertEqual(result['pin_candidates'][0]['number_status'],'owner_unresolved')

    def test_outside_net_label_is_not_internal_pinname(self):
        result=self.run_case([[100,100]],[text('NET_ALERT',[25,94,90,106])])
        self.assertIsNone(result['pin_candidates'][0]['pinname'])

    def test_component_field_text_is_excluded(self):
        records=[{'fields':{'Name':{'bbox':[120,94,185,106]}}}]
        result=self.run_case([[100,100]],[text('Chip123',[120,94,185,106])],records=records)
        self.assertIsNone(result['pin_candidates'][0]['pinname'])

    def test_conflicting_observed_number_readings_remain_unknown(self):
        result=self.run_case([[100,100]],[text('R9',[75,87,97,103]),text('6',[75,87,97,103],.985)])
        self.assertIsNone(result['pin_candidates'][0]['number'])
        self.assertEqual(result['pin_candidates'][0]['number_status'],'ambiguous_text')

    def test_one_number_cannot_silently_attach_to_two_nearby_pins(self):
        result=self.run_case([[100,100],[100,110]],[text('7',[78,100,95,110])])
        self.assertTrue(all(p['number'] is None for p in result['pin_candidates']))

    def test_duplicate_number_on_same_component_is_unresolved(self):
        result=self.run_case([[100,70],[100,140]],[text('1',[78,60,95,76]),text('1',[78,130,95,146])])
        self.assertTrue(all(p['number'] is None and p['number_status']=='duplicate_number_on_same_component' for p in result['pin_candidates']))

    def test_same_glyph_cannot_supply_both_number_and_name(self):
        result=self.run_case([[100,100]],[text('D0',[105,94,119,106])])
        pin=result['pin_candidates'][0]
        self.assertIsNone(pin['number'])
        self.assertIsNone(pin['pinname'])
        self.assertEqual(pin['number_status'],'same_glyph_number_name_role_ambiguous')

    def test_vertical_pin_number_and_name(self):
        result=self.run_case([[200,20]],[text('12',[191,0,209,16]),text('CLK',[190,45,210,70])])
        self.assertEqual(result['pin_candidates'][0]['number'],'12')
        self.assertEqual(result['pin_candidates'][0]['pinname'],'CLK')

    def test_local_views_stay_in_image_and_enforce_budget(self):
        source=owned([[100,50],[100,100],[100,150]])
        binding,_=bind_pin_text(source,[],CFG)
        options=deepcopy(CFG)
        options['local_ocr']['max_rois']=1
        views,skipped=local_views(source,binding,options)
        self.assertEqual((len(views),skipped),(1,2))
        self.assertTrue(all(0<=v<=400 for v in views[0]['crop']))

    def test_identity_join_changes_id_without_moving_or_reassigning_tip(self):
        source=owned([[100,100]])
        records=[{'detection_index':0,'component_id':'IC2','designator':'IC2','bbox_top_left':[100,20,300,300]}]
        result=attach_identities(source,records,{'class_mapping':{}})
        self.assertEqual(result['pin_candidates'][0]['owner'],'IC2')
        self.assertEqual(result['pin_candidates'][0]['point'],source['pin_candidates'][0]['point'])

    def test_cache_rejects_wrong_image_or_origin(self):
        validate_image(owned([]),'same',[400,400])
        with self.assertRaises(ValueError):
            validate_image(owned([]),'other',[400,400])
        with self.assertRaises(ValueError):
            validate_image({**owned([]),'coordinate_origin':'bottom_left'},'same',[400,400])


if __name__=='__main__':
    unittest.main()
