"""Vertical three-contact experiment from image geometry and local OCR only."""
import hashlib,json,math,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;ROOT.mkdir(exist_ok=True)
sys.dont_write_bytecode=True;sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_geometry_recovery import DEFAULT,LeadEvidence,vote
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
inv=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='10')
stage=Path(inv['model_cache'])/'p';document=read(stage/'export/engineering_pin_text_result.json')
records=read(stage/'export/completed_component_records.json')['records']
base=read(WORK/'pin_nc_recovery_implementation_20261020/deployed/10_Pred.json')
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/10.png').convert('RGB'));height,width=rgb.shape[:2]
evidence=LeadEvidence(rgb,DEFAULT);engine=None;calls=0;cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
def reader(image,box,scale,rotation,rec,view):
    global engine,calls
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,rec)).encode()).hexdigest();dest=cache/(key+'.json')
    if dest.exists():return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    value=recognize(engine,image,box,scale,rotation,rec,view);calls+=1
    dest.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');return value
def vertical_leads(frame,side):
    edge=frame[1] if side=='top' else frame[3];sign=-1 if side=='top' else 1;columns=[]
    for x in range(math.ceil(frame[0])+6,math.floor(frame[2])-5):
        if sum(evidence.hit(x,edge+sign*d) for d in range(3,11))<6:continue
        last=gap=0
        for d in range(1,66):
            if evidence.hit(x,edge+sign*d):last=d;gap=0
            else:gap+=1
            if gap>2:break
        if 24<=last<60:columns.append((x,last))
    clusters=[]
    for p in columns:
        if not clusters or p[0]-clusters[-1][-1][0]>1:clusters.append([])
        clusters[-1].append(p)
    import statistics
    return [{'side':side,'edge':edge,'point':[statistics.median(p[0] for p in g),edge+sign*statistics.median(p[1] for p in g)],'length':statistics.median(p[1] for p in g)} for g in clusters if len(g)<=7]
owner='U2';a,b,c,d=base['components'][owner]['bbox'];frame=evidence.frame([a,height-d,c,height-b])[0]
leads=vertical_leads(frame,'top')+vertical_leads(frame,'bottom')
record=next(r for r in records if r['component_id']==owner)['bbox_top_left']
crop=[math.floor(record[0]-32),math.floor(record[1]-32),math.ceil(record[2]+32),math.ceil(record[3]+32)]
whole=[]
for scale in [1.,3.]:
    for rotation in [0,90]:whole.extend(reader(rgb,crop,scale,rotation,False,f'vertical_component_{owner}_s{scale}_r{rotation}'))
rows=[]
for lead in leads:
    x,y=lead['point'];edge=lead['edge'];side=lead['side'];sign=-1 if side=='top' else 1
    # Original observed numbers at the matching side supply layout offsets.
    tokens=[]
    for p in document['pin_candidates']:
        if p.get('owner')!=owner:continue
        for t in p.get('text_candidates',{}).get('number',[]):
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if 0<sign*(cy-edge)<24 and abs(cx-x)<15:tokens.append(t)
    if not tokens:
        for p in document['pin_candidates']:
            if p.get('owner')!=owner:continue
            for t in p.get('text_candidates',{}).get('number',[]):
                cy=(t['bbox'][1]+t['bbox'][3])/2
                if 0<sign*(cy-edge)<24:tokens.append(t)
        anchor=next(l for l in leads if l['side']==side and any(abs((t['bbox'][0]+t['bbox'][2])/2-l['point'][0])<15 for t in tokens))
        ax=anchor['point'][0]
    else:ax=x
    import statistics
    dx=statistics.median((t['bbox'][0]+t['bbox'][2])/2-ax for t in tokens)
    dy=statistics.median((t['bbox'][1]+t['bbox'][3])/2-edge for t in tokens)
    cx=x+dx;cy=edge+dy;box=[math.floor(cx-9),math.floor(cy-9),math.ceil(cx+9),math.ceil(cy+9)]
    patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=(v.max(2)-v.min(2)<35)&(v.mean(2)<230);isolated=np.full_like(patch,255);isolated[mask]=patch[mask]
    isolated=cv2.copyMakeBorder(isolated,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
    readings={}
    for rotation,code in [(90,cv2.ROTATE_90_CLOCKWISE),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        oriented=cv2.rotate(isolated,code);values=[]
        for scale in [3.,5.]:values.extend(reader(oriented,[0,0,oriented.shape[1],oriented.shape[0]],scale,0,True,f'vertical_glyph_{owner}_{side}_{x}_r{rotation}_s{scale}'))
        value,support=vote([t for t in values if t['text'].isdigit()],DEFAULT);readings[str(rotation)]={'number':value,'support':support,'readings':values}
    names=[t for t in whole if any(v.isalpha() for v in t['text']) and frame[0]<(t['bbox'][0]+t['bbox'][2])/2<frame[2]
           and frame[1]<(t['bbox'][1]+t['bbox'][3])/2<frame[3] and abs((t['bbox'][0]+t['bbox'][2])/2-x)<12
           and ((t['bbox'][1]+t['bbox'][3])/2<(frame[1]+frame[3])/2 if side=='top' else (t['bbox'][1]+t['bbox'][3])/2>(frame[1]+frame[3])/2)]
    label,support=vote(names,DEFAULT)
    rows.append({'lead':lead,'number_crop':box,'layout':[dx,dy],'names':names,'name':label,'name_support':support,'readings':readings})
    Image.fromarray(isolated).save(ROOT/f'number_{side}_{x}.png')
result={'frame':frame,'whole':whole,'rows':rows,'new_local_ocr_calls':calls,'external_api_calls':0}
(ROOT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='whole'},ensure_ascii=False))
