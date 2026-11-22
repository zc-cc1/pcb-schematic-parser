"""Real symbol structure, class conflicts, physical rekeys and abstention."""
import copy,json,os,re,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import pcb_parser
if str(ROOT/'src/pcb_parser') not in pcb_parser.__path__:pcb_parser.__path__.insert(0,str(ROOT/'src/pcb_parser'))
import agent
from pcb_parser.component_kind_repair import recover_component_kinds,contact_rows,LeadEvidence,GEOMETRY_DEFAULT

def tokens(text,box):return [{'text':text,'bbox':box,'confidence':.999,'view':v} for v in ['a','b']]

def terminal():
    h=160;rgb=np.full((h,240,3),255,np.uint8)
    cv2.rectangle(rgb,(100,50),(140,70),(128,0,0),1);cv2.line(rgb,(80,60),(100,60),(128,0,0),1)
    cv2.putText(rgb,'1',(106,66),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    comp={'type':'box','Name':'H7','value':None,'bbox':[88,h-78,148,h-45]}
    pin={'pinname':'1','point':{'x':90.,'y':h-60.}}
    target={'components':{'H7':comp},'pins':{'H7':{'pin_1':pin}},'nets':{'N':{'hyperGraph':'(H7.1,PEER.1)','edges':{'e':[{'x':60,'y':100},{'x':80,'y':100}]}}}}
    doc={'pin_candidates':[{'candidate_id':'P','owner':'H7','owner_status':'assigned','point':[90.,60.],'number':'1','pinname':'1'}]}
    graph={'pin_attachments':[],'networks':[],'edges':[]}
    ocr={'texts':tokens('H7',[99,29,122,45])+tokens('1',[105,53,115,68])}
    def reader(image,box,scale,rotation,rec,view):return [{'text':'1','confidence':.999,'view':view}]
    return target,doc,graph,ocr,rgb,reader

def switch():
    h=360;rgb=np.full((h,400,3),255,np.uint8);cv2.rectangle(rgb,(120,80),(280,240),(128,0,0),1)
    ids={('top',150):'12',('top',200):'11',('top',250):'10',('bottom',150):'1',('bottom',200):'2',('bottom',250):'3',('left',140):'13',('right',180):'14'}
    doc={'pin_candidates':[]};ocr={'texts':tokens('SW9',[120,52,155,69])}
    for (side,axis),number in ids.items():
        if side in ['top','bottom']:
            edge=80 if side=='top' else 240;y=60 if side=='top' else 260;x=axis
            cv2.line(rgb,(x,edge),(x,y),(128,0,0),1)
            cv2.putText(rgb,'8',(x-13,edge-5 if side=='top' else edge+15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            box=[x-16,edge-18 if side=='top' else edge+2,x-2,edge-3 if side=='top' else edge+18]
        else:
            edge=120 if side=='left' else 280;x=100 if side=='left' else 300;y=axis
            cv2.line(rgb,(edge,y),(x,y),(128,0,0),1)
            cv2.putText(rgb,'8',(edge-17 if side=='left' else edge+5,y-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            box=[edge-23 if side=='left' else edge+2,y-18,edge-2 if side=='left' else edge+23,y-2]
        doc['pin_candidates'].append({'candidate_id':'P'+number,'owner':'SW9','owner_status':'assigned','point':[float(x),float(y)],'number':None,'pinname':None})
        ocr['texts']+=tokens(number,box)
    for y in [140,190]:
        for x in [150,200,250]:cv2.circle(rgb,(x,y),3,(128,0,0),1)
    target={'components':{'SW9':{'type':'box','Name':'SW9','value':None,'bbox':[120,h-240,280,h-80]}},
            'pins':{'SW9':{}},'nets':{'N':{'hyperGraph':'(PEER.1,SW9.2)','edges':{'e':[{'x':200,'y':300},{'x':200,'y':320}]}}}}
    for key,point in [('1',(150,260)),('2',(200,60)),('13',(100,140)),('14',(300,180))]:target['pins']['SW9']['pin_'+key]={'pinname':key,'point':{'x':float(point[0]),'y':float(h-point[1])}}
    def reader(image,box,scale,rotation,rec,view):
        m=re.search(r'switch_SW9_(top|bottom|left|right)_([\d.]+)',view)
        side=m[1];coordinate=float(m[2])
        if side in ['left','right']:
            m2=re.search(r'switch_SW9_\w+_[\d.]+_([\d.]+)',view);coordinate=float(m2[1])
        axis=min((a for s,a in ids if s==side),key=lambda a:abs(a-coordinate));number=ids[side,axis]
        return [{'text':number if '_r270_' not in view else 'Z','confidence':.999,'view':view}]
    return target,doc,{'pin_attachments':[],'networks':[],'edges':[]},ocr,rgb,reader

class KindRepairTests(unittest.TestCase):
    def run_data(self,d,reader=None,cfg=None):return recover_component_kinds(*d[:5],reader=reader or d[5],cfg=cfg)
    def unchanged(self,d):
        out,audit=self.run_data(d);self.assertEqual(out,d[0]);return audit

    def test_single_terminal_is_classified_and_actual_frame_and_tip_measured(self):
        d=terminal();before=copy.deepcopy(d[:4]);out,audit=self.run_data(d)
        self.assertEqual(d[:4],before);self.assertEqual(out['components']['H7']['type'],'pin')
        self.assertLess(abs(out['components']['H7']['bbox'][0]-100),2)
        self.assertLess(abs(out['pins']['H7']['pin_1']['point']['x']-80),3)
        self.assertEqual(out['nets'],d[0]['nets']);self.assertEqual(audit['ocr_calls'],2)

    def test_three_row_header_is_not_reclassified_as_single_pin(self):
        d=terminal();d[0]['pins']['H7']['pin_2']={'pinname':'2','point':{'x':90.,'y':70.}};self.unchanged(d)

    def test_terminal_requires_observed_designator_in_two_views(self):
        d=terminal();d[3]['texts']=[t for t in d[3]['texts'] if t['text']!='H7' or t['view']=='a'];self.unchanged(d)

    def test_real_model_name_is_protected(self):
        d=terminal();d[0]['components']['H7']['Name']='PART123';self.unchanged(d)

    def test_terminal_requires_single_real_lead(self):
        d=terminal();cv2.line(d[4],(140,60),(160,60),(128,0,0),1);self.unchanged(d)

    def test_broken_terminal_frame_abstains(self):
        d=terminal();d[4][49:53,99:142]=255;self.unchanged(d)

    def test_terminal_requires_digit_name_convention(self):
        d=terminal();d[0]['pins']['H7']['pin_1']['pinname']='BAT';self.unchanged(d)

    def test_wrong_candidate_owner_is_not_borrowed(self):
        d=terminal();d[1]['pin_candidates'][0]['owner_status']='ambiguous';self.unchanged(d)

    def test_terminal_fresh_ocr_disagreement_abstains(self):
        d=terminal()
        def wrong(*a):return [{'text':'2','confidence':.999,'view':a[-1]}]
        out,_=self.run_data(d,reader=wrong);self.assertEqual(out,d[0])

    def test_terminal_does_not_create_a_missing_route(self):
        d=terminal();d[0]['nets']={};out,audit=self.run_data(d)
        self.assertEqual(out['nets'],{});self.assertEqual(audit['network_withheld'][0]['member'],'H7.1')

    def test_nonsequential_switch_labels_are_individually_read(self):
        d=switch();before=copy.deepcopy(d[:4]);out,audit=self.run_data(d)
        self.assertEqual(d[:4],before);self.assertEqual(out['components']['SW9']['type'],'switch')
        self.assertEqual(set(out['pins']['SW9']),{'pin_'+s for s in ['1','2','3','10','11','12','13','14']})
        self.assertEqual(out['pins']['SW9']['pin_11']['point'],d[0]['pins']['SW9']['pin_2']['point'])

    def test_rekey_follows_the_old_physical_contact_without_reassigning_new_two(self):
        d=switch();out,audit=self.run_data(d)
        self.assertEqual(out['nets']['N']['hyperGraph'],'(PEER.1,SW9.11)')
        self.assertEqual(out['nets']['N']['edges'],d[0]['nets']['N']['edges'])
        self.assertNotEqual(out['pins']['SW9']['pin_2']['point'],out['pins']['SW9']['pin_11']['point'])

    def test_switch_prefix_without_contact_structure_cannot_change_an_ic(self):
        d=switch()
        for y in [140,190]:d[4][y-5:y+6,140:261]=255
        self.unchanged(d)

    def test_switch_requires_multiple_sides_of_real_leads(self):
        d=switch();d[4][55:80,140:261]=255;self.unchanged(d)

    def test_switch_requires_unique_read_numbers(self):
        d=switch();orig=d[5]
        def wrong(*a):
            v=orig(*a)
            if '_top_150' in a[-1]:v[0]['text']='13'
            return v
        out,_=self.run_data(d,reader=wrong);self.assertEqual(out,d[0])

    def test_unconfirmed_existing_physical_record_blocks_transaction(self):
        d=switch();d[0]['pins']['SW9']['pin_2']['point']={'x':200.,'y':180.};self.unchanged(d)

    def test_unreadable_new_contact_is_withheld_without_numbering_by_order(self):
        d=switch();orig=d[5]
        def partly(*a):
            if '_top_150' in a[-1]:return [{'text':'?','confidence':.5,'view':a[-1]}]
            return orig(*a)
        out,audit=self.run_data(d,reader=partly)
        self.assertEqual(out['components']['SW9']['type'],'switch');self.assertNotIn('pin_12',out['pins']['SW9'])
        self.assertEqual(len(audit['unresolved_contacts']),1)

    def test_classified_outputs_are_idempotent_without_ocr(self):
        for d in [terminal(),switch()]:
            out,_=self.run_data(d)
            def no_read(*a):raise AssertionError('Extra OCR')
            again,audit=self.run_data((out,*d[1:]),reader=no_read);self.assertEqual(out,again);self.assertEqual(audit['ocr_calls'],0)

    def test_budget_does_not_commit_unconfirmed_terminal(self):
        d=terminal();out,audit=self.run_data(d,cfg={'max_ocr_calls':1});self.assertEqual(out,d[0]);self.assertTrue(audit['budget_exhausted'])

    def test_wrapper_optout_needs_no_artifacts(self):
        d=terminal()
        with patch.dict(os.environ,{'PCB_COMPONENT_KIND_REPAIR':'0'}):out,audit=agent.finalize_component_kind_repair(d[0],Path('missing'),None,'x',None)
        self.assertEqual(out,d[0]);self.assertFalse(audit['enabled'])

    def test_wrapper_rejects_wrong_image_before_reading(self):
        d=terminal()
        with tempfile.TemporaryDirectory() as folder:
            stage=Path(folder);(stage/'export').mkdir()
            for file,v in [('engineering_pin_text_result.json',d[1]),('engineering_graph_proposals.json',d[2])]:
                (stage/'export'/file).write_text(json.dumps({**v,'image_sha256':'wrong','prediction_only':True,'coordinate_origin':'top_left','size':[240,160]}),encoding='utf-8')
            with self.assertRaises(ValueError):agent.finalize_component_kind_repair(d[0],stage,d[4],'right',d[3],reader=d[5])

if __name__=='__main__':unittest.main()
