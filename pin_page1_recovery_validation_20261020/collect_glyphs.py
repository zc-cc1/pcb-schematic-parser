"""Recognize isolated ink components in predicted number columns; no reference input."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_text_ocr import create_engine
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
model=read(Path(trace['folder'])/'p/export/engineering_pin_text_result.json')
pred=read(WORK/'pin_position_implementation_20261020/deployed/1_Pred.json')
image=cv2.imdecode(np.fromfile('D:/pcb_open_gt_in/1.png',np.uint8),cv2.IMREAD_COLOR)
h,w=image.shape[:2]
engine,models=create_engine({'onnx_threads':4})
out=[];calls=0
for owner,names in [('U3',{'VM'}),('USB1',{'DN1','SBU1'})]:
    texts=[t for p in (WORK/'pin_crop_experiment_20261020/ocr').glob(f'1_{owner}_*.json') for t in read(p)['texts']]
    bbox=pred['components'][owner]['bbox'];edge=bbox[0] if owner=='U3' else bbox[2]
    candidates=[p for p in model['pin_candidates'] if p.get('owner')==owner and p.get('pinname') in names
                and not p.get('text_hypothesis_primary_candidate_id')]
    digits=[t for t in texts if t['text'].isdigit() and t['confidence']>=.98
            and abs((t['bbox'][0]+t['bbox'][2])/2-edge)<30]
    column=float(np.median([(t['bbox'][0]+t['bbox'][2])/2 for t in digits]))
    # Learn printed number offset from reliable existing number/name pairs on
    # this component; fall back across its two horizontal sides if necessary.
    offsets=[]
    for p in model['pin_candidates']:
        if p.get('owner')!=owner or not p.get('number') or p.get('number_source')=='library_pin_name':continue
        for t in texts:
            cy=(t['bbox'][1]+t['bbox'][3])/2
            if t['text']==p['number'] and t['confidence']>=.98 and abs(cy-p['point'][1])<=16:
                offsets.append(cy-p['point'][1])
    offset=float(np.median(offsets)) if offsets else 0.
    for p in candidates:
        expected=p['point'][1]+offset
        x1=max(0,int(column-13));x2=min(w,int(column+14))
        y1=max(0,int(expected-13));y2=min(h,int(expected+14))
        patch=image[y1:y2,x1:x2].copy()
        blue,green,red=patch[:,:,0].astype(int),patch[:,:,1].astype(int),patch[:,:,2].astype(int)
        green_mask=(green>red+20)&(green>blue+20)&(green>60)
        # Printed digits are neutral dark ink; exclude colored body/lead strokes
        # that touch them and would otherwise form one boundary-spanning blob.
        neutral=(abs(red-green)<35)&(abs(red-blue)<35)&(abs(green-blue)<35)
        mask=(((blue+green+red)/3<205)&~green_mask&neutral).astype(np.uint8)
        long_strokes=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
        mask=mask & (1-long_strokes)
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
        glyphs=[]
        for i in range(1,count):
            a,b,ww,hh,area=map(int,stats[i]);cx,cy=centroids[i]
            if not(2<=ww<=20 and 6<=hh<=22 and area>=8):continue
            if abs(x1+cx-column)>8 or abs(y1+cy-expected)>9:continue
            if a==0 or b==0 or a+ww>=patch.shape[1] or b+hh>=patch.shape[0]:continue
            box=[x1+a,y1+b,x1+a+ww,y1+b+hh]
            isolated=np.full((hh+16,ww+16,3),255,dtype=np.uint8)
            ink=patch[b:b+hh,a:a+ww]
            chosen=labels[b:b+hh,a:a+ww]==i
            isolated[8:8+hh,8:8+ww][chosen]=ink[chosen]
            for scale in [3.,5.]:
                scaled=cv2.resize(isolated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                result=engine(scaled,use_det=False,use_cls=False,use_rec=True)
                calls+=1
                for text,confidence in zip(result.txts or [],result.scores or []):
                    glyphs.append({'text':str(text),'confidence':float(confidence),'bbox':box,
                                   'view':f"glyph_{p['candidate_id']}_{i}_s{scale}_r0",'source':'isolated_glyph'})
            folder=ROOT/'glyphs';folder.mkdir(exist_ok=True)
            cv2.imencode('.png',isolated)[1].tofile(str(folder/f"{p['candidate_id']}_{i}.png"))
        out.append({'owner':owner,'candidate_id':p['candidate_id'],'column':column,'offset':offset,'texts':glyphs})
        print(p['candidate_id'],'column',column,'offset',offset,[(t['text'],round(t['confidence'],4)) for t in glyphs])
(ROOT/'glyph_collection.json').write_text(json.dumps({'rows':out,'new_ocr_calls':calls,'models':models,
    'external_api_calls':0,'uses_reference':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
