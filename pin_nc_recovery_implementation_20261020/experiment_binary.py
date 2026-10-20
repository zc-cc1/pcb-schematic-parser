"""Compare standard binary/vertical closing preprocessing at fixed scales."""
import hashlib,json,math,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_geometry_recovery import DEFAULT,vote
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/8.png').convert('RGB'))
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
engine=None;calls=0
def ocr(image,scale,view):
    global engine,calls
    box=[0,0,image.shape[1],image.shape[0]];crop=np.ascontiguousarray(image)
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,0,True)).encode()).hexdigest();dest=ROOT/'ocr_cache'/(key+'.json')
    if dest.exists():return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    value=recognize(engine,image,box,scale,0,True,view);calls+=1;dest.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');return value
result=[]
for entry in read(ROOT/'experiment_grouped.json'):
    box=entry['crop'];patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<205)).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    chosen=[]
    for i in range(1,count):
        x,y,cw,ch,area=map(int,stats[i]);cx,cy=centroids[i]
        if 1<=cw<=24 and 2<=ch<=17 and area>=2 and -14<=box[1]+cy-entry['lead']['row']<=0:chosen.append(i)
    mask=np.isin(labels,chosen).astype(np.uint8)
    for close in [False,True]:
        joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)) if close else mask
        ys,xs=np.where(joined);a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
        for pad in [4,8]:
            isolated=np.repeat(((1-joined[b:d,a:c])*255)[:,:,None],3,axis=2).astype(np.uint8)
            isolated=cv2.copyMakeBorder(isolated,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255))
            tokens=[]
            for scale in [3.,5.]:tokens.extend(ocr(isolated,scale,f"binary_{entry['lead']['row']}_{close}_{pad}_{scale}"))
            number,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
            result.append({'row':entry['lead']['row'],'close':close,'pad':pad,'number':number,'support':support,'readings':tokens})
            Image.fromarray(isolated).save(ROOT/f"binary_{entry['lead']['row']}_{close}_{pad}.png")
(ROOT/'experiment_binary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False));print('NEW',calls)
