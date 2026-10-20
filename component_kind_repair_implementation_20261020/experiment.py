"""Image-only switch crop pilot. References are not loaded."""
from pathlib import Path
import sys,json
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_geometry_recovery import LeadEvidence,DEFAULT
from pcb_parser.pin_vertical_repair import vertical_leads
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
target=read(WORK/'pin_numeric_connector_implementation_20261020/deployed/1_Pred.json')
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'));h,w=rgb.shape[:2]
e=LeadEvidence(rgb,DEFAULT);a,b,c,d=target['components']['SW2']['bbox'];frame=e.frame([a,h-d,c,h-b])[0]
leads=e.leads(frame,'left')+e.leads(frame,'right')+vertical_leads(e,frame,'top',{'min_lead_px':8,'max_lead_px':60})+vertical_leads(e,frame,'bottom',{'min_lead_px':8,'max_lead_px':60})
engine,_=create_engine({'onnx_threads':4});rows=[]
for i,l in enumerate(leads):
    side=l['side'];x,y=l['point'];edge=l['edge']
    if side in ['top','bottom']:
        box=[int(x-20),int(edge-28 if side=='top' else edge-6),int(x+9 if side=='top' else x+2),int(edge+18 if side=='top' else edge+28)]
        rotations=[90,270]
    else:
        box=[int(edge-28 if side=='left' else edge+1),int(y-21),int(edge-1 if side=='left' else edge+28),int(y-1)];rotations=[0]
    p=rgb[box[1]:box[3],box[0]:box[2]];v=p.astype(int)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<230))|(v.max(2)<45)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    mask=np.isin(labels,[j for j in range(1,count) if stats[j,cv2.CC_STAT_AREA]>=4
                        and (side!='top' or box[0]+stats[j,0]+stats[j,2]/2<=x+3)])
    yy,xx=np.where(mask)
    if not len(xx):rows.append({**l,'box':box,'empty':True});continue
    z=np.full((max(yy)-min(yy)+9,max(xx)-min(xx)+9,3),255,np.uint8)
    z[4:-4,4:-4][mask[min(yy):max(yy)+1,min(xx):max(xx)+1]]=p[min(yy):max(yy)+1,min(xx):max(xx)+1][mask[min(yy):max(yy)+1,min(xx):max(xx)+1]]
    Image.fromarray(z).save(ROOT/f'crop_{i}.png')
    readings=[]
    for rotation in rotations:
        p=cv2.rotate(z,cv2.ROTATE_90_CLOCKWISE if rotation==90 else cv2.ROTATE_90_COUNTERCLOCKWISE) if rotation else z
        for scale in [3.,5.]:
            values=recognize(engine,p,[0,0,p.shape[1],p.shape[0]],scale,0,True,f'pilot_{side}_{i}_r{rotation}_s{scale}')
            readings.extend({**t,'rotation':rotation} for t in values)
    rows.append({**l,'box':box,'readings':readings})
(ROOT/'experiment.json').write_text(json.dumps({'frame':frame,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'frame':frame,'rows':[{**{k:v for k,v in l.items() if k!='readings'},'reads':[(t['rotation'],t['text'],t['confidence']) for t in l.get('readings',[])]} for l in rows]},ensure_ascii=True))
