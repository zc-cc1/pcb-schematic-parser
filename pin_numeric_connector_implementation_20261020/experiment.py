"""Local crop feasibility only; no reference labels or network guesses."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.pin_field_recovery import recognize
ROOT=Path(__file__).resolve().parent
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/13.png').convert('RGB'))
engine,fingerprints=create_engine({'onnx_threads':4})
output=[]
for name,box,color in [('pin',(1295,374,1314,399),'neutral'),('supply',(1236,413,1283,439),'blue')]:
    patch=rgb[box[1]:box[3],box[0]:box[2]].copy()
    r,g,b=[patch[:,:,i].astype(int) for i in range(3)]
    if color=='neutral':mask=(abs(r-g)<35)&(abs(r-b)<35)&(abs(g-b)<35)&((r+g+b)/3<230)
    else:mask=(b>r+35)&(b>g+35)&(r<180)&(g<180)
    y,x=np.where(mask)
    assert len(x)
    isolated=np.full((max(y)-min(y)+9,max(x)-min(x)+9,3),255,np.uint8)
    isolated[4:-4,4:-4][mask[min(y):max(y)+1,min(x):max(x)+1]]=0
    Image.fromarray(isolated).save(ROOT/(name+'_isolated.png'))
    readings=[]
    for scale in [3.,5.]:
        readings.extend(recognize(engine,isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'{name}_{scale}'))
    output.append({'kind':name,'source_box':box,'readings':readings})
(ROOT/'experiment.json').write_text(json.dumps({'readings':output,'models':fingerprints},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=True))
