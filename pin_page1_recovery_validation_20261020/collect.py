"""Prediction-only local number-strip OCR for the requested page-one pilot."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.component_text import restore_polygon

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
model=read(Path(trace['folder'])/'p/export/engineering_pin_text_result.json')
pred=read(WORK/'pin_position_implementation_20261020/deployed/1_Pred.json')
source=Path('D:/pcb_open_gt_in/1.png')
image=cv2.imdecode(np.fromfile(str(source),np.uint8),cv2.IMREAD_COLOR)
h,w=image.shape[:2]
# Chosen from predicted missing/conflicting fields, never from reference coordinates.
selected=[p for p in model['pin_candidates'] if
          (p.get('owner')=='U3' and p.get('pinname')=='VM') or
          (p.get('owner')=='USB1' and p.get('pinname') in {'DN1','SBU1'}) or
          (p.get('owner')=='USB2' and p['candidate_id'] in {'PIN_CAND_0107','PIN_CAND_0135','PIN_CAND_0155'})]
rois=[]
for p in selected:
    x,y=p['point'];b=pred['components'][p['owner']]['bbox'];x1,by1,x2,by2=b;y1,y2=h-by2,h-by1
    side=min({'left':abs(x-x1),'right':abs(x-x2),'top':abs(y-y1),'bottom':abs(y-y2)},
             key=lambda s:{'left':abs(x-x1),'right':abs(x-x2),'top':abs(y-y1),'bottom':abs(y-y2)}[s])
    if side in {'left','right'}:
        a,b=(x1-35,x1+16) if side=='left' else (x2-16,x2+35)
        boxes=[(a,y-21,b,y+2),(a,y-2,b,y+21)]
    else:
        edge=y1 if side=='top' else y2
        boxes=[(x-22,edge-40,x+22,edge+40)]
    for index,box in enumerate(boxes):
        box=[max(0,int(box[0])),max(0,int(box[1])),min(w,int(box[2])+1),min(h,int(box[3])+1)]
        rois.append({'candidate_id':p['candidate_id'],'owner':p['owner'],'side':side,'part':index,'box':box})

engine,models=create_engine({'onnx_threads':4})
views=[];new=0;reused=0
for roi in rois:
    x1,y1,x2,y2=roi['box'];crop=image[y1:y2,x1:x2]
    folder=ROOT/'ocr';folder.mkdir(exist_ok=True)
    for scale in [3.,5.]:
        for rotation in [0,90]:
            dest=folder/f"{roi['candidate_id']}_p{roi['part']}_s{scale}_r{rotation}.json"
            if dest.exists():
                doc=read(dest);assert doc['roi']==roi;reused+=1;views.append(doc);continue
            rotated=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE) if rotation else crop
            patch=cv2.resize(rotated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            start=time.perf_counter()
            result=engine(np.ascontiguousarray(patch),use_det=True,use_cls=True,use_rec=True)
            texts=[]
            if result.boxes is not None:
                for polygon,text,confidence in zip(result.boxes,result.txts,result.scores):
                    polygon=np.asarray(restore_polygon(np.asarray(polygon,dtype=float),roi['box'],scale,rotation),dtype=float)
                    a=polygon.min(axis=0);b=polygon.max(axis=0)
                    if a[0]<=x1+1 or a[1]<=y1+1 or b[0]>=x2-1 or b[1]>=y2-1:continue
                    texts.append({'text':str(text),'confidence':float(confidence),'bbox':[float(a[0]),float(a[1]),float(b[0]),float(b[1])],
                                  'view':dest.stem,'source':'targeted_strip'})
            doc={'roi':roi,'scale':scale,'rotation':rotation,'texts':texts,'wall_seconds':time.perf_counter()-start}
            dest.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            views.append(doc);new+=1
            print(roi['candidate_id'],roi['part'],scale,rotation,[(t['text'],round(t['confidence'],4)) for t in texts],flush=True)
(ROOT/'collection.json').write_text(json.dumps({'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'models':models,'rois':rois,'views':views,'new_ocr_calls':new,'reused_calls':reused,'external_api_calls':0,
    'uses_reference_for_crop':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Completed',len(views),'local OCR views;',new,'new calls.',flush=True)
