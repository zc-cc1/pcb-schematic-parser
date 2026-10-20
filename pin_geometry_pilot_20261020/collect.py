"""Additional component OCR from predicted boxes, no reference labels."""
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
model_folder=Path(trace['folder'])
records=read(model_folder/'p/export/completed_component_records.json')['records']
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'))
engine=None;views=[];calls=0
for owner in ['U1','U2','USB1']:
    record=next(r for r in records if r['component_id']==owner)
    a,b,c,d=record['bbox_top_left'];box=[max(0,math.floor(a-32)),max(0,math.floor(b-32)),min(rgb.shape[1],math.ceil(c+32)),min(rgb.shape[0],math.ceil(d+32))]
    for scale in [1.,3.]:
        for rotation in [0,90]:
            old=WORK/f'pin_crop_experiment_20261020/ocr/1_{owner}_s{scale}_r{rotation}.json'
            view=f'component_{owner}_s{scale}_r{rotation}'
            if old.exists() and read(old)['crop']==box:
                texts=[{**t,'view':view} for t in read(old)['texts']]
            else:
                if engine is None:engine,_=create_engine({'onnx_threads':4})
                texts=recognize(engine,rgb,box,scale,rotation,False,view);calls+=1
            views.append({'owner':owner,'scale':scale,'rotation':rotation,'box':box,'texts':texts})
            print(owner,scale,rotation,[(t['text'],round(t['confidence'],4)) for t in texts if t['text'].isdigit() or t['text'] in ['EN','EP']])
(ROOT/'collection.json').write_text(json.dumps({'views':views,'new_local_ocr_calls':calls,'external_api_calls':0,'uses_reference':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
