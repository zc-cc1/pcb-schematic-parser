"""Independent row evidence, consensus positions and local T contact guards."""
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
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser.pin_numeric_connector import recover_numeric_connectors,observations

def fixture():
    h=240;rgb=np.full((h,300,3),255,np.uint8)
    cv2.rectangle(rgb,(120,30),(190,165),(150,0,0),1)
    for row,number in [(60,'7'),(90,'12'),(120,'43')]:
        cv2.line(rgb,(90,row),(120,row),(150,0,0),1)
        cv2.putText(rgb,number,(127,row+14),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    cv2.line(rgb,(90,120),(90,148),(150,0,0),1)
    cv2.line(rgb,(75,148),(105,148),(150,0,0),1)
    cv2.putText(rgb,'VIN',(74,170),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,230),1)
    target={'components':{'X':{'type':'box','Name':'X','value':None,'bbox':[120,h-165,190,h-30]}},
            'pins':{'X':{'pin_7':{'pinname':'7','point':{'x':90.,'y':h-60.}},'pin_12':{'pinname':'12','point':{'x':90.,'y':h-90.}}}},
            'nets':{'KEEP':{'hyperGraph':'(OTHER.1)','edges':{'e':[{'x':10.,'y':20.},{'x':20.,'y':20.}]}}}}
    doc={'pin_candidates':[{'candidate_id':'P','owner':'X','owner_status':'assigned','number':None,'pinname':None,'point':[90.,120.]}]}
    graph={'networks':[],'edges':[],'pin_attachments':[{'candidate_id':'P','point':[90.,120.],'attachment_status':'no_wire_within_radius','network_id':None}]}
    ocr={'texts':[]}
    for row,number in [(60,'7'),(90,'12'),(120,'43')]:
        for v in ['a','b']:ocr['texts'].append({'text':number,'confidence':.999,'bbox':[124,row+1,147,row+19],'view':v})
    def reader(image,box,scale,rotation,rec,view):
        return [{'text':'VIN' if view.startswith('connector_supply_') else '43','confidence':.999,'bbox':box,'view':view}]
    return target,doc,graph,ocr,rgb,reader

class NumericConnectorTests(unittest.TestCase):
    def run_data(self,data=None,reader=None,cfg=None):
        d=data or fixture()
        return recover_numeric_connectors(*d[:5],reader=reader or d[5],cfg=cfg)
    def unchanged(self,d):
        out,audit=self.run_data(d);self.assertEqual(out,d[0]);self.assertEqual(audit['ocr_calls'],0)

    def test_nonsequential_identity_and_inputs_preserved(self):
        d=fixture();before=copy.deepcopy(d[:4]);out,audit=self.run_data(d)
        self.assertEqual(d[:4],before)
        self.assertEqual(out['pins']['X']['pin_43'],{'pinname':'43','point':{'x':90.,'y':120.}})
        self.assertNotIn('pin_3',out['pins']['X']);self.assertEqual(len(audit['changes']),1)

    def test_power_has_only_local_contact_net(self):
        out,audit=self.run_data()
        self.assertEqual(out['components']['RECOVERED_SUPPLY_P']['value'],'VIN')
        self.assertEqual(out['nets']['OBSERVED_CONTACT_P'],{'hyperGraph':'(RECOVERED_SUPPLY_P.1,X.43)','edges':{}})
        self.assertEqual(audit['network_withheld'][0]['reason'],'upstream_route_not_reconstructed')
        self.assertEqual(out['nets']['KEEP'],fixture()[0]['nets']['KEEP'])

    def test_idempotence_needs_no_extra_ocr(self):
        d=fixture();out,_=self.run_data(d)
        def no_read(*args):raise AssertionError('Unexpected OCR')
        again,audit=self.run_data((out,*d[1:]),reader=no_read)
        self.assertEqual(out,again);self.assertEqual(audit['ocr_calls'],0)

    def test_one_cache_view_is_insufficient(self):
        d=fixture();d[3]['texts']=[t for t in d[3]['texts'] if t['text']!='43' or t['view']=='a'];self.unchanged(d)

    def test_duplicate_views_not_independent(self):
        d=fixture()
        for t in d[3]['texts']:
            if t['text']=='43':t['view']='a'
        self.unchanged(d)

    def test_single_high_confidence_conflict_abstains(self):
        d=fixture();d[3]['texts'].append({**d[3]['texts'][-1],'text':'99','view':'c'});self.unchanged(d)

    def test_merged_multirow_digits_not_split(self):
        d=fixture();d[3]['texts']=[t for t in d[3]['texts'] if t['text']!='43']
        for view in ['a','b']:d[3]['texts'].append({'text':'123','confidence':.999,'bbox':[124,62,147,140],'view':view})
        self.unchanged(d)

    def test_fresh_glyph_disagreement_abstains(self):
        d=fixture()
        def wrong(*args):return [{'text':'44','confidence':.999,'view':args[-1]}]
        out,audit=self.run_data(d,reader=wrong);self.assertEqual(out,d[0]);self.assertEqual(audit['ocr_calls'],2)

    def test_conflicting_fresh_scales_abstain(self):
        d=fixture()
        def wrong(*args):return [{'text':'43' if args[2]==3. else '44','confidence':.999,'view':args[-1]}]
        out,_=self.run_data(d,reader=wrong);self.assertEqual(out,d[0])

    def test_requires_digit_name_convention(self):
        d=fixture();d[0]['pins']['X']['pin_7']['pinname']='SIGNAL';self.unchanged(d)

    def test_requires_two_anchor_rows(self):
        d=fixture();d[0]['pins']['X'].pop('pin_12');self.unchanged(d)

    def test_inconsistent_anchor_lengths_rejected(self):
        d=fixture();d[0]['pins']['X']['pin_12']['point']['x']=96;self.unchanged(d)

    def test_requires_existing_real_candidate(self):
        d=fixture();d[1]['pin_candidates']=[];self.unchanged(d)

    def test_ambiguous_owner_rejected(self):
        d=fixture();d[1]['pin_candidates'][0]['owner_status']='ambiguous';self.unchanged(d)

    def test_duplicate_candidates_rejected(self):
        d=fixture();d[1]['pin_candidates'].append({**d[1]['pin_candidates'][0],'candidate_id':'Q'});self.unchanged(d)

    def test_existing_source_number_not_overwritten(self):
        d=fixture();d[1]['pin_candidates'][0]['number']='99';self.unchanged(d)

    def test_off_column_digits_not_borrowed(self):
        d=fixture()
        for t in d[3]['texts']:
            if t['text']=='43':t['bbox']=[164,121,185,139]
        self.unchanged(d)

    def test_broken_red_frame_rejected(self):
        d=fixture();d[4][30:166,119:122]=255;self.unchanged(d)

    def test_supply_requires_cap(self):
        d=fixture();d[4][146:151,70:110]=255;out,audit=self.run_data(d)
        self.assertIn('pin_43',out['pins']['X']);self.assertFalse(audit['component_changes']);self.assertEqual(out['nets'],d[0]['nets'])

    def test_supply_requires_unbroken_stem(self):
        d=fixture();d[4][130:138,87:94]=255;out,audit=self.run_data(d)
        self.assertIn('pin_43',out['pins']['X']);self.assertFalse(audit['component_changes'])

    def test_unrecognized_power_label_not_exported(self):
        d=fixture();original=d[5]
        def wrong(*args):return [{'text':'FOO','confidence':.999,'view':args[-1]}] if args[-1].startswith('connector_supply_') else original(*args)
        out,audit=self.run_data(d,reader=wrong)
        self.assertIn('pin_43',out['pins']['X']);self.assertFalse(audit['component_changes']);self.assertEqual(out['nets'],d[0]['nets'])

    def test_weak_power_label_not_exported(self):
        d=fixture();original=d[5]
        def weak(*args):
            values=original(*args)
            if args[-1].startswith('connector_supply_'):values[0]['confidence']=.979
            return values
        _,audit=self.run_data(d,reader=weak);self.assertFalse(audit['component_changes'])

    def test_existing_power_symbol_not_duplicated(self):
        d=fixture();d[0]['components']['V']={'type':'v','Name':None,'value':'VIN','bbox':[75,92,105,120]}
        out,audit=self.run_data(d);self.assertFalse(audit['component_changes']);self.assertEqual(out['nets'],d[0]['nets'])

    def test_pin_budget_exhaustion_commits_nothing(self):
        d=fixture();out,audit=self.run_data(d,cfg={'max_ocr_calls':1});self.assertEqual(out,d[0]);self.assertTrue(audit['budget_exhausted'])

    def test_supply_budget_exhaustion_keeps_verified_pin_only(self):
        d=fixture();out,audit=self.run_data(d,cfg={'max_ocr_calls':2})
        self.assertIn('pin_43',out['pins']['X']);self.assertFalse(audit['component_changes']);self.assertEqual(out['nets'],d[0]['nets'])

    def test_observation_flattening_preserves_views(self):
        value={'text':'9','confidence':.999,'bbox':[1,2,3,4],'view':'a'}
        self.assertEqual(len(observations({'texts':[{**value,'observations':[value,{**value,'view':'b'}]}]})),2)

    def attach(self,d):
        d[2]['pin_attachments'][0].update(attachment_status='attached',network_id='KEEP')
        d[2]['networks']=[{'network_id':'KEEP','pin_candidate_ids':['P'],'edge_ids':['E'],'unresolved_junction_ids':[]}]
        d[2]['edges']=[{'edge_id':'E','polyline':[[70,120],[90,120]]}]
        cv2.line(d[4],(50,120),(89,120),(0,130,0),1)

    def test_unique_attachment_with_original_segment_can_join_existing_net(self):
        d=fixture();self.attach(d);out,_=self.run_data(d)
        self.assertIn('X.43',out['nets']['KEEP']['hyperGraph']);self.assertNotIn('OBSERVED_CONTACT_P',out['nets'])
        self.assertEqual(out['nets']['KEEP']['edges'],d[0]['nets']['KEEP']['edges'])

    def test_missing_original_segment_cannot_join_upstream_net(self):
        d=fixture();self.attach(d);d[2]['edges']=[]
        out,_=self.run_data(d);self.assertEqual(out['nets']['KEEP'],d[0]['nets']['KEEP'])

    def test_ambiguous_junction_does_not_join_upstream_net(self):
        d=fixture();self.attach(d);d[2]['networks'][0]['unresolved_junction_ids']=['J']
        out,_=self.run_data(d);self.assertEqual(out['nets']['KEEP'],d[0]['nets']['KEEP'])

    def test_image_green_support_is_required_for_original_attachment(self):
        d=fixture();self.attach(d);d[4][119:122,50:90]=255
        out,_=self.run_data(d);self.assertEqual(out['nets']['KEEP'],d[0]['nets']['KEEP'])

    def test_wrong_attachment_point_does_not_join_upstream_net(self):
        d=fixture();self.attach(d);d[2]['pin_attachments'][0]['point']=[85,120]
        out,_=self.run_data(d);self.assertEqual(out['nets']['KEEP'],d[0]['nets']['KEEP'])

    def test_observed_local_contact_key_never_overwrites_existing_net(self):
        d=fixture();d[0]['nets']['OBSERVED_CONTACT_P']={'hyperGraph':'(OTHER.2)','edges':{}}
        out,_=self.run_data(d);self.assertEqual(out['nets']['OBSERVED_CONTACT_P'],d[0]['nets']['OBSERVED_CONTACT_P'])

    def test_wrapper_optout_needs_no_files(self):
        d=fixture()
        with patch.dict(os.environ,{'PCB_PIN_NUMERIC_CONNECTOR':'0'}):out,audit=agent.finalize_pin_numeric_connector(d[0],Path('missing'),None,'digest',None)
        self.assertEqual(out,d[0]);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_other_image(self):
        d=fixture()
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder);(stage/'export').mkdir()
            for name,value in [('engineering_pin_text_result.json',d[1]),('engineering_graph_proposals.json',d[2])]:
                (stage/'export'/name).write_text(json.dumps({**value,'image_sha256':'wrong','prediction_only':True,'coordinate_origin':'top_left','size':[300,240]}),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_pin_numeric_connector(d[0],stage,d[4],'right',d[3],reader=d[5])

if __name__=='__main__':unittest.main()
