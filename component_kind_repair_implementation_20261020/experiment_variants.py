from pathlib import Path
import sys,json,cv2
import numpy as np
from PIL import Image
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
ROOT=Path(__file__).resolve().parent
doc=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'));engine,_=create_engine({'onnx_threads':4})
results=[]
for i in [0,4,5,6,7,8,9,10,15]:
    row=doc['rows'][i];box=row['box'];crop=rgb[box[1]:box[3],box[0]:box[2]]
    v=crop.astype(int);neutral=(v.max(2)-v.min(2)<35)&(v.mean(2)<230);dark=neutral|(v.max(2)<100)
    variants={'raw':crop.copy()}
    for name,mask,binary in [('neutral',neutral,True),('dark_binary',dark,True),('dark_gray',dark,False)]:
        z=np.full_like(crop,255);z[mask]=0 if binary else crop[mask];variants[name]=z
    values=[]
    for name,patch in variants.items():
        patch=cv2.copyMakeBorder(patch,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
        if row['side'] in ['top','bottom']:patch=cv2.rotate(patch,cv2.ROTATE_90_CLOCKWISE)
        Image.fromarray(patch).save(ROOT/f'variant_{i}_{name}.png')
        for scale in [3.,5.]:
            vals=recognize(engine,patch,[0,0,patch.shape[1],patch.shape[0]],scale,0,True,f'{name}_{i}_{scale}')
            values.extend({'variant':name,**t} for t in vals)
    results.append({'index':i,'side':row['side'],'values':values})
(ROOT/'experiment_variants.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'index':r['index'],'values':[(t['variant'],t['text'],t['confidence']) for t in r['values']]} for r in results],ensure_ascii=True))
