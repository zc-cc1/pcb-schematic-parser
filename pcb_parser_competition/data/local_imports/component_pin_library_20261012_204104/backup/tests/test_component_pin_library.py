"""Observable-name/package binding, ambiguity and provenance regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from pcb_parser.component_pin_library import bind_library_pins, number_for_name, validate_library
from run_component_pin_library import load_library

read = lambda p: json.loads(p.read_text(encoding='utf-8'))
CFG = read(ROOT/'configs/component_pin_library_v1.json')
TEXT_CFG = read(ROOT/'configs/pin_text_v2.json')
LIBRARY = read(ROOT/CFG['library_path'])


def fixture(name='PA5/AN5/SPI1_SCK'):
    model_box = [20,115,115,126]
    record = {'detection_index':0, 'component_id':'U1', 'type':'ic', 'Name':'STM32F103C8T6',
              'bbox_top_left':[20,20,120,110], 'fields':{'Name':{'text':'STM32F103C8T6', 'confidence':.999,
              'bbox':model_box, 'observations':[{'text':'STM32F103C8T6', 'confidence':.999, 'view':v, 'bbox':model_box}
                                              for v in ['global','tile']]}}}
    pins = {'components':[{'detection_index':0, 'component_id':'U1', 'type':'ic', 'bbox_top_left':record['bbox_top_left']}],
            'size':[160,160], 'pin_candidates':[{'candidate_id':'P1', 'point':[20,50], 'confidence':.99,
             'owner_detection_index':0, 'owner_status':'assigned', 'owner':'U1', 'number':None, 'pinname':None,
             'number_status':'no_reliable_text', 'pinname_status':'no_reliable_text', 'text_evidence':{}, 'text_candidates':{}}],
            'pin_text_summary':{}}
    texts = [{'text':name, 'bbox':[35,45,95,55], 'confidence':.999, 'view':v} for v in ['global','tile']]
    return pins, texts, [record]


def bind(p, t, r):
    return bind_library_pins(p,t,r,LIBRARY,CFG,TEXT_CFG)


class LibraryTests(unittest.TestCase):
    def test_observed_literal_name_and_library_number_have_separate_evidence(self):
        p,t,r = fixture(); before = deepcopy((p,t,r)); out,a = bind(p,t,r); pin = out['pin_candidates'][0]
        self.assertEqual(pin['number'],'15'); self.assertEqual(pin['pinname'],'PA5/AN5/SPI1_SCK')
        self.assertFalse(pin['text_evidence']['number']['number_observed_in_image'])
        self.assertEqual(a['summary']['number_observed_OCR_count'],0)
        self.assertEqual((p,t,r),before); self.assertEqual(pin['point'],[20,50])
        self.assertEqual(pin['owner_detection_index'],0)

    def test_primary_tokens_do_not_confuse_PA1_PA10_or_package_power_names(self):
        part = LIBRARY['parts'][0]
        for text, number in [('PA1/AN1','11'),('PA10/RX1','31'),('PA0_WKUP/AN0','10'),('VDD_2','36'),('VSS_3','47')]:
            self.assertEqual(number_for_name(text,part),number)
        for text in ['PA11x','PAI/AN1','PA16','VDD 2','VDD','RTC','PC13X','pb5']:
            self.assertIsNone(number_for_name(text,part))

    def test_exact_observed_part_is_required(self):
        for model in ['STM32F103C8','STM32F103C8U6','STM32F103C8T7','STM32F103CBT6',None]:
            p,t,r = fixture(); r[0]['Name'] = model
            out,a = bind(p,t,r)
            self.assertEqual(out['pin_candidates'],p['pin_candidates']); self.assertEqual(a['summary']['supported_components'],0)

    def test_model_must_have_two_matching_nearby_strong_views(self):
        for failure in ['single','weak','far','text','duplicate_detection']:
            p,t,r = fixture(); ev = r[0]['fields']['Name']
            if failure=='single':ev['observations']=ev['observations'][:1]
            elif failure=='weak':ev['observations'][1]['confidence']=.9
            elif failure=='far':ev['observations'][1]['bbox']=[0,0,10,10]
            elif failure=='text':ev['text']='STM32F103CBT6'
            else:r[0]['duplicate_of_detection_index']=1
            out,a = bind(p,t,r); self.assertEqual(out['pin_candidates'],p['pin_candidates'])

    def test_pin_name_needs_distinct_views_and_full_inside_body_geometry(self):
        for failure in ['single','same_view','weak','outside','other_row','split']:
            p,t,r = fixture()
            if failure=='single':t=t[:1]
            elif failure=='same_view':t[1]['view']='global'
            elif failure=='weak':t[1]['confidence']=.9
            elif failure=='outside':
                for item in t:item['bbox']=[5,45,15,55]
            elif failure=='other_row':
                for item in t:item['bbox']=[35,95,95,105]
            else:
                for item in t:item['text']='PA5/AN5 SPI1_SCK'
            out,a = bind(p,t,r); self.assertIsNone(out['pin_candidates'][0]['number'])

    def test_supported_conflicting_spellings_are_withheld(self):
        p,t,r = fixture(); t += [{**row,'text':'PA5/AN5/SPI1_SCL'} for row in t]
        out,a = bind(p,t,r); self.assertIsNone(out['pin_candidates'][0]['number'])
        self.assertTrue(any(x['reason']=='conflicting_observed_full_spellings' for x in a['withheld']))

    def test_valid_observed_number_is_never_replaced_by_another_number(self):
        p,t,r = fixture(); p['pin_candidates'][0]['number']='14'
        out,a = bind(p,t,r); self.assertEqual(out['pin_candidates'][0]['number'],'14')
        self.assertIsNone(out['pin_candidates'][0]['pinname'])
        self.assertTrue(any(x['reason']=='existing_number_disagrees_with_name' for x in a['withheld']))

    def test_invalid_number_is_quarantined_only_on_supported_model(self):
        p,t,r = fixture(); p['pin_candidates'][0]['number']='PB5'
        out,a = bind(p,t,r)
        self.assertEqual(a['summary']['invalid_numbers_withheld'],1); self.assertEqual(out['pin_candidates'][0]['number'],'15')
        r[0]['Name']='OTHER_PART'; out,a = bind(p,t,r); self.assertEqual(out['pin_candidates'][0]['number'],'PB5')

    def test_number_claimed_by_another_tip_is_not_reused(self):
        p,t,r = fixture(); other=deepcopy(p['pin_candidates'][0]);other.update(candidate_id='P2',point=[20,85],number='15')
        p['pin_candidates'].append(other);out,a = bind(p,t,r)
        self.assertIsNone(out['pin_candidates'][0]['number']);self.assertEqual(out['pin_candidates'][1]['number'],'15')

    def test_two_equal_row_hypotheses_are_not_silently_assigned(self):
        p,t,r = fixture();p['pin_candidates'][0]['point']=[20,48]
        other=deepcopy(p['pin_candidates'][0]);other.update(candidate_id='P2',point=[20,52]);p['pin_candidates'].append(other)
        out,a = bind(p,t,r);self.assertTrue(all(x['number'] is None for x in out['pin_candidates']))

    def test_secondary_tip_keeps_geometry_without_receiving_a_second_number(self):
        p,t,r = fixture();other=deepcopy(p['pin_candidates'][0]);other.update(candidate_id='P2',point=[19,52])
        p['pin_candidates'].append(other);out,a = bind(p,t,r)
        self.assertEqual(out['pin_candidates'][0]['number'],'15');self.assertIsNone(out['pin_candidates'][1]['number'])
        self.assertEqual([x['point'] for x in out['pin_candidates']],[[20,50],[19,52]])

    def test_fragment_can_be_completed_only_from_literal_observed_full_text(self):
        p,t,r = fixture();p['pin_candidates'][0]['pinname']='SPI1_SCK'
        out,a=bind(p,t,r);self.assertEqual(out['pin_candidates'][0]['pinname'],'PA5/AN5/SPI1_SCK')
        self.assertEqual(a['summary']['partial_names_completed'],1)
        p['pin_candidates'][0]['pinname']='PA5/SPI1_CLK';out,a=bind(p,t,r)
        self.assertEqual(out['pin_candidates'][0]['pinname'],'PA5/SPI1_CLK');self.assertIsNone(out['pin_candidates'][0]['number'])

    def test_library_rejects_duplicate_numbers_and_aliases(self):
        for field in ['number','observed_primary_tokens']:
            library=deepcopy(LIBRARY);library['parts'][0]['pins'][1][field]=library['parts'][0]['pins'][0][field]
            with self.assertRaises(ValueError):validate_library(library)

    def test_runtime_rejects_modified_library_hash(self):
        cfg={**CFG,'library_sha256':'0'*64}
        with self.assertRaises(ValueError):load_library(cfg)
        self.assertEqual(load_library(CFG)[0],LIBRARY)


if __name__=='__main__':
    unittest.main()
