"""Image-only grouped numeral experiment, before any code or label changes."""
import hashlib
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_geometry_recovery import DEFAULT,LeadEvidence,vote
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
base=read(WORK/'pin_number_cycle_implementation_20261020/deployed/8_Pred.json')
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/8.png').convert('RGB'));h,w=rgb.shape[:2]
evidence=LeadEvidence(rgb,DEFAULT);a,b,c,d=base['components']['U1']['bbox']
frame=evidence.frame([a,h-d,c,h-b])[0]
ROOT.mkdir(exist_ok=True);cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
engine=None;calls=0
def ocr(image,box,scale,rec,view):
    global engine,calls
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,0,rec)).encode()).hexdigest();dest=cache/(key+'.json')
    if dest.exists():return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    result=recognize(engine,image,box,scale,0,rec,view);calls+=1
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
def isolated(box):
    patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=(v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<205)
    result=np.full_like(patch,255);result[mask]=patch[mask]
    return cv2.copyMakeBorder(result,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
results=[]
# Candidate rows derive only from actual red short leads. Read the entire NC
# cluster, including already known controls, without using GT coordinates.
for lead in evidence.leads(frame,'left'):
    row=lead['row'];edge=lead['edge']
    if not 12<=lead['length']<=24:continue
    crop=[math.floor(edge-32),math.floor(row-19),math.ceil(edge-2),math.ceil(row-1)]
    number=isolated(crop);tokens=[]
    for scale in [3.,5.]:tokens.extend(ocr(number,[0,0,number.shape[1],number.shape[0]],scale,True,f'nc_number_{row}_s{scale}'))
    names=[];namebox=[math.ceil(edge+3),math.floor(row-1),math.ceil(edge+60),math.ceil(row+19)]
    name=isolated(namebox)
    for scale in [3.,5.]:names.extend(ocr(name,[0,0,name.shape[1],name.shape[0]],scale,True,f'nc_name_{row}_s{scale}'))
    value,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
    label,nsupport=vote(names,DEFAULT)
    results.append({'lead':lead,'number':value,'support':support,'number_readings':tokens,'name':label,'name_support':nsupport,'name_readings':names,'crop':crop})
    Image.fromarray(number).save(ROOT/f'number_row_{row}.png')
result={'frame':frame,'rows':results,'new_local_ocr_calls':calls,'external_api_calls':0}
(ROOT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
