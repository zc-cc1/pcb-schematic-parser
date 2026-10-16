"""Learning-output agreement, preservation and unknown-number regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.component_tip_refinement import refine_component_tips

CFG=json.loads((ROOT/'configs/component_tip_refinement_v1.json').read_text(encoding='utf-8'))
OWNER=json.loads((ROOT/'configs/pin_ownership_v1.json').read_text(encoding='utf-8'))


def inputs(kind='vcc',peaks=((30,35,.7),),zoom_peaks=None):
    component={'detection_index':0,'component_id':'C0','type':kind,'class_name':kind,'confidence':.99,'bbox_top_left':[20,20,50,50]}
    p={'size':[100,80],'coordinate_origin':'top_left','components':[component],'pin_candidates':[]}
    native=np.zeros((2,80,100),np.float32);zoom=native.copy();native[0]=.9;zoom[0]=.9
    for array,items in [(native,peaks),(zoom,zoom_peaks if zoom_peaks is not None else peaks)]:
        yy,xx=np.mgrid[:80,:100]
        for x,y,prob in items:array[1]=np.maximum(array[1],prob*np.exp(-((xx-x)**2+(yy-y)**2)/2))
    return p,native,zoom


class ComponentTipTests(unittest.TestCase):
    def test_supported_power_peak_added_without_number_or_bbox_snapping(self):
        p,n,z=inputs(peaks=((31.25,35.4,.73),));before=deepcopy(p)
        out,a=refine_component_tips(p,n,z,CFG,OWNER);self.assertEqual(a['summary']['added_by_type'],{'vcc':1})
        pin=out['pin_candidates'][0];self.assertAlmostEqual(pin['point'][0],31.25,places=4)
        self.assertAlmostEqual(pin['point'][1],35.4,places=4);self.assertIsNone(pin['number']);self.assertIsNone(pin['pinname'])
        self.assertEqual(p,before);self.assertFalse(pin['localization_evidence']['coordinate_snapped_to_component_box'])

    def test_one_scale_only_or_disagreement_is_withheld(self):
        for zoom in [(),((40,35,.72),)]:
            p,n,z=inputs(zoom_peaks=zoom);out,_=refine_component_tips(p,n,z,CFG,OWNER)
            self.assertFalse(out['pin_candidates'])

    def test_multiple_power_peaks_do_not_choose_one_by_expected_count(self):
        p,n,z=inputs(peaks=((28,30,.75),(42,43,.7)));out,a=refine_component_tips(p,n,z,CFG,OWNER)
        self.assertFalse(out['pin_candidates']);self.assertIn('more_plausible_peaks_than_missing_tip_slots',a['summary']['component_withheld_reasons'])

    def test_no_learned_wire_evidence_prevents_addition(self):
        p,n,z=inputs();n[0]=0;out,_=refine_component_tips(p,n,z,CFG,OWNER);self.assertFalse(out['pin_candidates'])

    def test_existing_candidates_and_fields_are_not_modified_or_removed(self):
        p,n,z=inputs();old={'candidate_id':'PIN_CAND_0001','point':[30,35],'confidence':.92,'owner':'C0',
                          'owner_detection_index':0,'owner_status':'assigned','number':'A5','pinname':'OBSERVED'}
        p['pin_candidates']=[old];out,a=refine_component_tips(p,n,z,CFG,OWNER)
        self.assertEqual(out['pin_candidates'],[old]);self.assertEqual(a['summary']['added_candidates'],0)

    def test_unresolved_nearby_original_tip_is_not_reassigned_or_duplicated(self):
        p,n,z=inputs();old={'candidate_id':'PIN_CAND_0001','point':[30,35],'confidence':.9,'owner':None,
                          'owner_detection_index':None,'owner_status':'ambiguous_owner','number':None,'pinname':None}
        p['pin_candidates']=[old];out,_=refine_component_tips(p,n,z,CFG,OWNER);self.assertEqual(out['pin_candidates'],[old])

    def test_second_passive_tip_requires_opposed_complete_pair(self):
        p,n,z=inputs(kind='resistor',peaks=((45,35,.72),));p['components'][0]['bbox_top_left']=[20,30,50,40]
        old={'candidate_id':'PIN_CAND_0001','point':[22,35],'confidence':.9,'owner':'C0','owner_status':'assigned',
             'owner_detection_index':0,'number':None,'pinname':None};p['pin_candidates']=[old]
        out,a=refine_component_tips(p,n,z,CFG,OWNER);self.assertEqual(a['summary']['added_by_type'],{'resistor':1})
        self.assertEqual(out['pin_candidates'][0],old);self.assertIsNone(out['pin_candidates'][1]['number'])

    def test_incomplete_passive_or_same_side_pair_is_not_completed(self):
        for kind,existing in [('capacitor',[]),('resistor',[{'candidate_id':'P','point':[28,35],'owner_status':'assigned','owner_detection_index':0}])]:
            p,n,z=inputs(kind=kind);p['pin_candidates']=existing
            out,_=refine_component_tips(p,n,z,CFG,OWNER);self.assertEqual(out['pin_candidates'],existing)

    def test_unselected_symbol_types_never_get_template_pin_count(self):
        for kind in ['ic','diode','mosfet','led','connector']:
            p,n,z=inputs(kind=kind);out,_=refine_component_tips(p,n,z,CFG,OWNER);self.assertFalse(out['pin_candidates'])

    def test_nonfinite_or_mismatched_probability_rejected(self):
        for mode in ['nan','shape','range']:
            p,n,z=inputs()
            if mode=='nan':n[1,0,0]=np.nan
            elif mode=='shape':n=n[:,:70]
            else:n[1,0,0]=1.1
            with self.assertRaises(ValueError):refine_component_tips(p,n,z,CFG,OWNER)

    def test_ambiguous_owner_across_close_detections_does_not_add(self):
        p,n,z=inputs();p['components'].append({**p['components'][0],'detection_index':1,'component_id':'C1'})
        out,_=refine_component_tips(p,n,z,CFG,OWNER);self.assertFalse(out['pin_candidates'])

    def test_duplicate_ids_or_number_guessing_policy_rejected(self):
        p,n,z=inputs();old={'candidate_id':'P','point':[30,35],'owner_status':'ambiguous_owner'};p['pin_candidates']=[old,old]
        with self.assertRaises(ValueError):refine_component_tips(p,n,z,CFG,OWNER)
        p,n,z=inputs();cfg={**CFG,'new_numbers_policy':'spatial_sort'}
        with self.assertRaises(ValueError):refine_component_tips(p,n,z,cfg,OWNER)


if __name__=='__main__':unittest.main()
