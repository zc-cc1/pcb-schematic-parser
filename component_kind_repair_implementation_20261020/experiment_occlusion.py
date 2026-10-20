"""Test generic interpolation across the observed red frame band, not digit templates."""
from pathlib import Path
import sys,json,cv2
import numpy as np
from PIL import Image
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
ROOT=Path(__file__).resolve().parent
doc=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'));rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'))
engine,_=create_engine({'onnx_threads':4});results=[]
for i in [4,5,6,7,8,9]:
    l=doc['rows'][i];x=l['point'][0];edge=l['edge'];box=[int(x-20),int(edge-25),int(x-1),int(edge+12)]
    crop=rgb[box[1]:box[3],box[0]:box[2]];v=crop.astype(int)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<230)).astype('uint8')
    values=[]
    for size in [2,3,4]:
        closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((size,1),np.uint8))
        # Only bridge in the frame's three-pixel occlusion band.
        repaired=mask.copy();lo=max(0,int(edge)-box[1]-1);hi=min(mask.shape[0],lo+4)
        repaired[lo:hi]=closed[lo:hi]
        n,labels,stats,_=cv2.connectedComponentsWithStats(repaired,8)
        repaired=np.isin(labels,[j for j in range(1,n) if stats[j,4]>=5])
        yy,xx=np.where(repaired)
        z=np.full((max(yy)-min(yy)+9,max(xx)-min(xx)+9,3),255,np.uint8)
        z[4:-4,4:-4][repaired[min(yy):max(yy)+1,min(xx):max(xx)+1]]=0
        z=cv2.rotate(z,cv2.ROTATE_90_CLOCKWISE)
        Image.fromarray(z).save(ROOT/f'occlusion_{i}_{size}.png')
        for scale in [3.,5.]:values.extend({'size':size,**t} for t in recognize(engine,z,[0,0,z.shape[1],z.shape[0]],scale,0,True,f'occlusion_{i}_{size}_{scale}'))
    results.append({'index':i,'readings':values})
(ROOT/'experiment_occlusion.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'index':r['index'],'readings':[(t['size'],t['text'],t['confidence']) for t in r['readings']]} for r in results],ensure_ascii=True))
