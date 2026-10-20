"""Inspect three-pin bottom-row numbering using same-image OCR only."""
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
sys.dont_write_bytecode=True;sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
from pcb_parser.pin_geometry_recovery import DEFAULT,IC_TYPES,LeadEvidence,side_of,vote
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
case='6';entry=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']==case)
stage=Path(entry['model_cache'])/'p';model=read(stage/'export/engineering_pin_text_result.json')
records={r['component_id']:r for r in read(stage/'export/completed_component_records.json')['records']}
base=read(WORK/f'pin_evidence_implementation_20261020/deployed/{case}_Pred.json')
image=Path(f'D:/pcb_open_gt_in/{case}.png');rgb=np.asarray(Image.open(image).convert('RGB'));h,w=rgb.shape[:2]
assert hashlib.sha256(image.read_bytes()).hexdigest()==model['image_sha256']
evidence=LeadEvidence(rgb,DEFAULT);engine=None;new=reused=0;cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
def ocr(image,box,scale,rotation,rec,view):
    global engine,new,reused
    patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,rec)).encode()).hexdigest();dest=cache/(key+'.json')
    if dest.exists():reused+=1;return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    texts=recognize(engine,image,box,scale,rotation,rec,view);write(dest,texts);new+=1;return texts

owners=[]
for owner,comp in sorted(base['components'].items()):
    if comp['type'] not in IC_TYPES or owner not in records:continue
    a,b,c,d=comp['bbox'];found=evidence.frame([a,h-d,c,h-b])
    if not found:continue
    frame=found[0]
    candidates=[p for p in model['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                and not p.get('text_hypothesis_primary_candidate_id') and side_of(p['point'],frame)=='bottom'
                and str(p.get('number','')).isdigit() and p.get('pinname')]
    if len(candidates)!=3 or not any(p.get('number_source')=='library_pin_name' for p in candidates):continue
    candidates.sort(key=lambda p:p['point'][0]);offsets=[];anchors=[]
    for p in candidates:
        if p.get('number_source')=='library_pin_name':continue
        for t in p.get('text_candidates',{}).get('number',[]):
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if str(t.get('text'))==str(p['number']) and t.get('confidence',0)>=.98 and 0<cy-frame[3]<20 and abs(cx-p['point'][0])<15:
                offsets.append([cx-p['point'][0],cy-frame[3]]);anchors.append(p['candidate_id'])
    if not offsets:continue
    dx=statistics.median(v[0] for v in offsets);dy=statistics.median(v[1] for v in offsets)
    box=records[owner]['bbox_top_left'];crop=[max(0,math.floor(box[0]-32)),max(0,math.floor(box[1]-32)),min(w,math.ceil(box[2]+32)),min(h,math.ceil(box[3]+32))]
    whole=[]
    for scale in [1.,3.]:
        for rotation in [0,90]:whole.extend(ocr(rgb,crop,scale,rotation,False,f'cycle_component_{owner}_s{scale}_r{rotation}'))
    group={'owner':owner,'frame':frame,'offset':[dx,dy],'anchors':sorted(set(anchors)),'pins':[],'whole':whole}
    for p in candidates:
        cx=p['point'][0]+dx;cy=frame[3]+dy
        box=[max(0,math.floor(cx-8)),max(0,math.floor(cy-8)),min(w,math.ceil(cx+8)),min(h,math.ceil(cy+8))]
        patch=rgb[box[1]:box[3],box[0]:box[2]];colors=patch.astype(np.int16)
        mask=(colors.max(axis=2)-colors.min(axis=2)<35)&(colors.mean(axis=2)<205)
        isolated=np.full_like(patch,255);isolated[mask]=patch[mask]
        isolated=cv2.copyMakeBorder(isolated,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
        Image.fromarray(isolated).save(ROOT/f"{owner}_{p['candidate_id']}_number.png")
        results={}
        for rotation,code in [(90,cv2.ROTATE_90_CLOCKWISE),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            oriented=cv2.rotate(isolated,code);tokens=[]
            for scale in [3.,5.]:tokens.extend(ocr(oriented,[0,0,oriented.shape[1],oriented.shape[0]],scale,0,True,f"cycle_glyph_{p['candidate_id']}_r{rotation}_s{scale}"))
            value,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
            results[str(rotation)]={'number':value,'support':support,'tokens':tokens}
        group['pins'].append({'candidate_id':p['candidate_id'],'number':p['number'],'name':p['pinname'],'point':p['point'],'crop':box,'readings':results})
    owners.append(group)
write(ROOT/'collection.json',{'image_sha256':model['image_sha256'],'groups':owners,'new_ocr_calls':new,'reused_ocr_calls':reused})
for group in owners:
    print(json.dumps({k:v for k,v in group.items() if k!='whole'},ensure_ascii=False))
print('NEW',new,'REUSED',reused)
