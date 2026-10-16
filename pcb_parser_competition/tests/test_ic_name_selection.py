"""Priorities, retained readings and physical text ownership for final IC Name."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_parser.ic_name_selection import select_ic_names


def text(value, box, index=1, confidence=.99):
    a,b,c,d=box
    return {'text':value,'bbox':box,'polygon':[[a,b],[c,b],[c,d],[a,d]],
            'confidence':confidence,'text_id':f'text_{index}','view':'global'}


def fixture(name=None, reference=True, key='U1', box=None):
    box=box or [100,100,300,300]
    fields={}
    if reference:fields['designator']=text(key,[(box[0]+box[2])/2-15,box[1]-22,(box[0]+box[2])/2+15,box[1]-7],9)
    if name is not None:fields['Name']=text(name,[150,150,230,165],8)
    record={'component_id':key,'type':'ic','designator':key if reference else None,
            'Name':name,'value':None,'bbox_top_left':box,'fields':fields,'detection_index':0}
    payload={'components':{key:{'type':'box','Name':name,'value':None,'bbox':[100,100,300,300]},
                           'R1':{'type':'r','Name':'R1','value':'10k','bbox':[400,400,408,420]}},
             'pins':{key:{'pin_1':{'pinname':'IN','point':{'x':100,'y':180}}}},'nets':{}}
    meta={'prediction_only':True,'coordinate_origin':'top_left','image_sha256':'a'*64,'size':[500,500]}
    return payload,{**meta,'records':[record]},{**meta,'texts':[]}


class ICNameSelectionTests(unittest.TestCase):
    def test_interior_two_letter_model_has_priority_over_reference_and_description(self):
        p,r,o=fixture();o['texts']=[text('LM358',[150,150,230,165]),text('T_V',[160,180,220,195],2)]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'LM358')
        self.assertEqual(audit['components'][0]['selected_category'],'model')
        self.assertEqual({c['text'] for c in audit['components'][0]['candidates']},{'LM358','T_V'})

    def test_observed_reference_precedes_other_description(self):
        p,r,o=fixture();o['texts']=[text('2.4GHz',[150,150,230,165])]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'U1')
        self.assertEqual(audit['components'][0]['selected_category'],'reference')
        self.assertEqual(audit['components'][0]['candidates'][0]['text'],'2.4GHz')

    def test_other_description_used_when_no_model_or_observed_reference(self):
        p,r,o=fixture(reference=False,key='UNRESOLVED_DET_0001');o['texts']=[text('T_V',[160,160,220,175])]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['UNRESOLVED_DET_0001']['Name'],'T_V')
        self.assertEqual(audit['components'][0]['selected_category'],'description')

    def test_full_multiword_description_is_retained_without_concatenating_boxes(self):
        p,r,o=fixture(reference=False);o['texts']=[text('TYPE-C 16PIN',[140,160,260,175])]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'TYPE-C 16PIN')
        self.assertIn('TYPE-C 16PIN',[c['text'] for c in audit['components'][0]['candidates']])

    def test_existing_observed_model_is_not_replaced_by_reference_or_new_reading(self):
        p,r,o=fixture(name='PAW3395DM-T6QU');o['texts']=[text('LM358',[150,150,230,165])]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual,p)
        self.assertEqual(audit['changed_Name_count'],0)

    def test_no_observed_reference_means_no_key_based_guess(self):
        p,r,o=fixture(reference=False)
        self.assertIsNone(select_ic_names(p,r,o)[0]['components']['U1']['Name'])

    def test_pin_and_rail_tokens_are_retained_but_not_used(self):
        for value in ['GPIO17','PA3','VCC','GND','RESET','EN','NC','U2','12']:
            with self.subTest(value=value):
                p,r,o=fixture(reference=False);o['texts']=[text(value,[150,150,230,165])]
                actual,audit=select_ic_names(p,r,o)
                self.assertIsNone(actual['components']['U1']['Name'])
                self.assertEqual(audit['components'][0]['candidates'][0]['text'],value)

    def test_side_pin_region_is_not_a_model_caption(self):
        p,r,o=fixture();o['texts']=[text('CAGC1',[102,155,135,168])]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'U1')
        self.assertFalse(audit['components'][0]['candidates'][0]['eligible'])

    def test_low_confidence_model_does_not_override_observed_reference(self):
        p,r,o=fixture();o['texts']=[text('LM358',[150,150,230,165],confidence=.75)]
        self.assertEqual(select_ic_names(p,r,o)[0]['components']['U1']['Name'],'U1')

    def test_conflicting_same_glyph_model_readings_fall_back_to_reference(self):
        p,r,o=fixture();t=text('LM358',[150,150,230,165]);t['observations']=[{'text':'NE555','bbox':t['bbox'],'confidence':.99,'view':'tile'}];o['texts']=[t]
        actual,audit=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'U1')
        self.assertEqual({c['text'] for c in audit['components'][0]['candidates']},{'LM358','NE555'})
        self.assertTrue(audit['assignment_rejections'])

    def test_same_model_at_two_distinct_positions_can_name_two_ics(self):
        p,r,o=fixture(box=[100,100,200,250]);p2,r2,_=fixture(key='U2',box=[250,100,350,250])
        p['components']['U2']=p2['components']['U2'];r['records'].append(r2['records'][0])
        o['texts']=[text('LM358',[130,150,170,165]),text('LM358',[280,150,320,165],2)]
        actual,_=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'LM358');self.assertEqual(actual['components']['U2']['Name'],'LM358')

    def test_shared_ambiguous_model_glyph_does_not_name_two_ics(self):
        p,r,o=fixture(box=[100,100,200,250]);p2,r2,_=fixture(key='U2',box=[220,100,320,250])
        p['components']['U2']=p2['components']['U2'];r['records'].append(r2['records'][0])
        o['texts']=[text('LM358',[195,70,225,85])]
        actual,_=select_ic_names(p,r,o)
        self.assertEqual(actual['components']['U1']['Name'],'U1');self.assertEqual(actual['components']['U2']['Name'],'U2')

    def test_glyph_owned_by_other_component_is_not_stolen(self):
        p,r,o=fixture();other=deepcopy(r['records'][0]);other.update(component_id='U2',designator='U2',bbox_top_left=[350,100,450,250]);other['fields']={'Name':text('LM358',[150,150,230,165])};r['records'].append(other)
        o['texts']=[text('LM358',[150,150,230,165])]
        self.assertEqual(select_ic_names(p,r,o)[0]['components']['U1']['Name'],'U1')

    def test_all_other_fields_and_input_documents_stay_identical(self):
        p,r,o=fixture();o['texts']=[text('LM358',[150,150,230,165])];saved=deepcopy((p,r,o))
        actual,_=select_ic_names(p,r,o);actual['components']['U1']['Name']=p['components']['U1']['Name']
        self.assertEqual(actual,p);self.assertEqual((p,r,o),saved)

    def test_cross_image_evidence_is_rejected(self):
        p,r,o=fixture();o['image_sha256']='b'*64
        with self.assertRaises(ValueError):select_ic_names(p,r,o)


if __name__=='__main__':unittest.main()
