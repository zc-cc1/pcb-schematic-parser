"""Bounded image-only relocation pilot. Reference labels are read after export."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
import agent
from pcb_parser.pin_geometry_recovery import DEFAULT, LeadEvidence, side_of, vote
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
case='2';path=Path(f'D:/pcb_open_gt_in/{case}.png')
rgb=np.asarray(Image.open(path).convert('RGB'));height,width=rgb.shape[:2]
digest=hashlib.sha256(path.read_bytes()).hexdigest()
inventory=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']==case)
stage=Path(inventory['model_cache'])/'p'
graph=read(stage/'export/engineering_graph_proposals.json');assert graph['image_sha256']==digest
model=read(stage/'export/engineering_pin_text_result.json')
records={r['component_id']:r for r in read(stage/'export/completed_component_records.json')['records']}
base=read(WORK/f'pin_geometry_implementation_20261020/deployed/{case}_Pred.json');out=copy.deepcopy(base)
evidence=LeadEvidence(rgb,DEFAULT);audit={'image_sha256':digest,'proposals':[],'changes':[],'ocr_calls':0}
engine=None;cache=ROOT/'ocr';cache.mkdir(exist_ok=True)
def ocr(image,box,scale,rotation,rec,view):
    global engine
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,rec)).encode()).hexdigest()
    dest=cache/(key+'.json')
    previous=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')
    if not dest.exists() and previous.exists():write(dest,read(previous))
    if dest.exists():return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    tokens=recognize(engine,image,box,scale,rotation,rec,view);audit['ocr_calls']+=1
    write(dest,tokens);return tokens

def clean_neutral(box, expected_center=None):
    patch=rgb[box[1]:box[3],box[0]:box[2]];c=patch.astype(np.int16)
    neutral=(c.max(axis=2)-c.min(axis=2)<35)&(c.mean(axis=2)<205)
    neutral=neutral.astype(np.uint8)
    neutral&=1-cv2.morphologyEx(neutral,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
    if expected_center is not None:
        count,labels,stats,centers=cv2.connectedComponentsWithStats(neutral,8)
        keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_HEIGHT]>=6
              and stats[i,cv2.CC_STAT_AREA]>=8 and abs(box[1]+centers[i,1]-expected_center)<=6]
        neutral=np.isin(labels,keep).astype(np.uint8)
    isolated=np.full_like(patch,255);isolated[neutral>0]=patch[neutral>0]
    if expected_center is not None and neutral.any():
        yy,xx=np.where(neutral>0)
        isolated=isolated[yy.min():yy.max()+1,xx.min():xx.max()+1]
        return cv2.copyMakeBorder(isolated,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
    return cv2.copyMakeBorder(isolated,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))

for candidate in model['pin_candidates']:
    owner=candidate.get('owner');number=str(candidate.get('number'));name=candidate.get('pinname')
    # Only distrust both fields filled by elimination at a weak, unsupported point.
    if not(owner in base['components'] and candidate.get('owner_status')=='assigned'
           and candidate.get('number_source')=='library_elimination'
           and candidate.get('pinname_source')=='library_elimination'
           and not candidate.get('text_evidence') and candidate.get('confidence',1)<.8
           and candidate.get('nearby_wire_probability_max',1)<.2):continue
    old=base['pins'].get(owner,{}).get('pin_'+number)
    if old is None or old['pinname']!=name:continue
    a,b,c,d=base['components'][owner]['bbox'];found=evidence.frame([a,height-d,c,height-b])
    if not found:continue
    box=found[0];x,y=candidate['point']
    if not(box[0]-2<=x<=box[2]+2 and box[1]-2<=y<=box[3]+2):continue
    old_tl=[old['point']['x'],height-old['point']['y']]
    if math.dist(old_tl,candidate['point'])>2:continue
    known=[(key,p,[p['point']['x'],height-p['point']['y']]) for key,p in base['pins'].get(owner,{}).items() if key!='pin_'+number]
    rows=[]
    for side in ['left','right']:
        same=[item for item in known if side_of(item[2],box)==side]
        if len(same)<3:continue
        rows.extend(p for p in evidence.leads(box,side) if not any(abs(pt[1]-p['row'])<=6 for key,pin,pt in same))
    if len(rows)!=1:continue
    proposal=rows[0];row=proposal['row'];edge=proposal['edge'];side=proposal['side']
    # Derive numeral offset/column from observed numbers on other correct rows.
    observed=[];name_offsets=[]
    for p in model['pin_candidates']:
        if p.get('owner')!=owner or not p.get('number') or 'library' in p.get('number_source',''):continue
        for t in p.get('text_candidates',{}).get('number',[]):
            if t.get('text')==p['number'] and t.get('confidence',0)>=.98:
                observed.append(((t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2-p['point'][1]))
        for t in p.get('text_candidates',{}).get('pinname',[]):
            if t.get('text')==p.get('pinname') and t.get('confidence',0)>=.98:
                name_offsets.append((t['bbox'][1]+t['bbox'][3])/2-p['point'][1])
    if not observed:continue
    column=statistics.median(v[0] for v in observed);offset=statistics.median(v[1] for v in observed)
    expected=row+offset
    number_box=[max(0,math.floor(column-11)),max(0,math.floor(expected-12)),min(width,math.ceil(column+11)),min(height,math.ceil(expected+12))]
    name_box=[math.ceil(box[0]+3),max(0,math.floor(row-9)),math.floor(box[2]-3),min(height,math.ceil(row+20))]
    # Names remain in the near half on wider components; this pilot handles left rows.
    if side!='left':continue
    reading={};tokens_by_kind={}
    for kind,cropbox in [('number',number_box),('name',name_box)]:
        image=clean_neutral(cropbox,row+statistics.median(name_offsets) if kind=='name' and name_offsets else None);texts=[]
        Image.fromarray(image).save(ROOT/f'{owner}_{kind}_neutral.png')
        local=[0,0,image.shape[1],image.shape[0]]
        for scale in [3.,5.]:
            texts.extend(ocr(image,local,scale,0,True,f'{kind}_isolated_s{scale}'))
        if kind=='name':
            for scale in [3.,5.]:
                texts.extend(ocr(rgb,cropbox,scale,0,False,f'name_row_detect_s{scale}'))
            old_box=records[owner]['bbox_top_left']
            component_box=[max(0,math.floor(old_box[0]-32)),max(0,math.floor(old_box[1]-32)),
                           min(width,math.ceil(old_box[2]+32)),min(height,math.ceil(old_box[3]+32))]
            for scale in [2.,3.,5.]:
                for rotation in [0,90]:
                    observations=ocr(rgb,component_box,scale,rotation,False,f'name_component_s{scale}_r{rotation}')
                    texts.extend(t for t in observations if box[0]<(t['bbox'][0]+t['bbox'][2])/2<box[2]
                                 and abs((t['bbox'][1]+t['bbox'][3])/2-row)<=12)
        accepted=[t for t in texts if (t['text'].isdigit() if kind=='number' else any(ch.isalpha() for ch in t['text']))]
        value,support=vote(accepted,DEFAULT);reading[kind]={'value':value,'support':support,'tokens':texts,'crop':cropbox}
        tokens_by_kind[kind]=texts
    entry={'owner':owner,'source_candidate_id':candidate['candidate_id'],'old_point':old_tl,
           'old_number':number,'old_name':name,'proposal':proposal,'reading':reading}
    audit['proposals'].append(entry)
    if reading['number']['value']!=number or reading['name']['value']!=name:
        entry['status']='insufficient_or_conflicting_observed_fields';continue
    if math.dist(old_tl,proposal['point'])<25:entry['status']='not_large_displacement';continue
    # This first experiment moves the pin only; memberships/wires are held fixed.
    out['pins'][owner]['pin_'+number]['point']={'x':proposal['point'][0],'y':height-proposal['point'][1]}
    entry['status']='relocated_by_visible_lead_and_observed_fields';audit['changes'].append(entry)

agent.validate_target(out)
assert out['components']==base['components'] and out['nets']==base['nets']
write(ROOT/'2_Pred.json',out);write(ROOT/'generation_audit.json',audit)
# Reference labels are first accessed here, after the candidate export is frozen.
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/2').glob('*.json')))
result={'before':metrics.score_pins(base,truth,20)[0],'after':metrics.score_pins(out,truth,20)[0],
        'changes':len(audit['changes']),'new_local_ocr_calls':audit['ocr_calls'],'external_api_calls':0,
        'full_model_runs':0,'production_modified':False,'components_and_networks_unchanged':True}
write(ROOT/'results.json',result)
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
