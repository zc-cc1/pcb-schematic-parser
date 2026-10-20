"""Whole-component OCR on predicted boxes, using bundled local weights only."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.component_text import restore_polygon

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

inventory=read(ROOT/'inventory.json')
print('Loading bundled local OCR models; no detector/U-Net inference.',flush=True)
engine,models=create_engine({'onnx_threads':4})
begin=time.perf_counter()
views=[]
new_calls=reused=0
for row in inventory:
    image=Path(row['image'])
    assert hashlib.sha256(image.read_bytes()).hexdigest()==row['image_sha256']
    bgr=cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
    height,width=bgr.shape[:2]
    for component in row['components']:
        x1,y1,x2,y2=component['bbox']
        margin=32
        box=[max(0,math.floor(x1-margin)),max(0,math.floor(y1-margin)),
             min(width,math.ceil(x2+margin)),min(height,math.ceil(y2+margin))]
        x1,y1,x2,y2=box
        crop=bgr[y1:y2,x1:x2]
        path=ROOT/f"crops/{row['case']}_{component['owner']}.png"
        path.parent.mkdir(exist_ok=True)
        cv2.imencode('.png',crop)[1].tofile(str(path))
        for scale in [1.0,3.0]:
            for rotation in [0,90]:
                destination=ROOT/f"ocr/{row['case']}_{component['owner']}_s{scale}_r{rotation}.json"
                if destination.exists():
                    result=read(destination)
                    assert result['image_sha256']==row['image_sha256'] and result['crop']==box
                    views.append(result)
                    reused+=1
                    print('Reused',destination.name,flush=True)
                    continue
                rotated=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE) if rotation==90 else crop
                patch=cv2.resize(rotated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                start=time.perf_counter()
                detected=engine(np.ascontiguousarray(patch),use_det=True,use_cls=True,use_rec=True)
                new_calls+=1
                texts=[]
                if detected.boxes is not None:
                    for polygon,text,confidence in zip(detected.boxes,detected.txts,detected.scores):
                        if float(confidence)<.80:continue
                        polygon=restore_polygon(np.asarray(polygon,dtype=float),box,scale,rotation)
                        xs,ys=np.asarray(polygon).T
                        text_box=[float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max())]
                        if text_box[0]<=x1+1 or text_box[1]<=y1+1 or text_box[2]>=x2-1 or text_box[3]>=y2-1:
                            continue
                        texts.append({'text':str(text),'confidence':float(confidence),'bbox':text_box,
                                      'polygon':np.asarray(polygon).tolist(),
                                      'view':f"component_crop_{component['index']}_s{scale}_r{rotation}"})
                result={'case':row['case'],'owner':component['owner'],'index':component['index'],
                        'image_sha256':row['image_sha256'],'crop':box,'scale':scale,'rotation':rotation,
                        'texts':texts,'wall_seconds':time.perf_counter()-start,'models':models}
                save(destination,result)
                views.append(result)
                print(f"case={row['case']} owner={component['owner']} scale={scale} rotation={rotation} "
                      f"texts={len(texts)} seconds={result['wall_seconds']:.2f}",flush=True)
save(ROOT/'ocr_collection.json',{'views':views,'model_fingerprints':models,
    'total_unique_ocr_views':len(views),'new_local_ocr_calls':new_calls,'reused_views':reused,
    'external_api_calls':0,'detector_unet_calls':0,
    'gt_used_for_crop_or_ocr':False,'wall_seconds':time.perf_counter()-begin})
print('Completed',len(views),'local OCR views.',flush=True)
