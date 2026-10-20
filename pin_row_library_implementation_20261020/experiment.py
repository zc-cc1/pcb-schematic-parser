"""Image-only horizontal row OCR and exact bundled library lookup."""
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
baseline=read(WORK/'pin_vertical_repair_implementation_20261020/deployed/11_Pred.json')
entry=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='11')
stage=Path(entry['model_cache'])/'p';records=read(stage/'export/completed_component_records.json')['records']
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/11.png').convert('RGB'));h,w=rgb.shape[:2]
evidence=LeadEvidence(rgb,DEFAULT);cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
cache_roots=[WORK/f'{folder}/ocr_cache' for folder in ['pin_field_implementation_20261020','pin_geometry_implementation_20261020']]
engine=None;calls=0;reused=0
def reader(image,box,scale,rotation,rec,view):
    global engine,calls,reused
    crop=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(crop.tobytes()+str((crop.shape,box,scale,rotation,rec)).encode()).hexdigest();dest=cache/(key+'.json')
    source=next((p/(key+'.json') for p in cache_roots if (p/(key+'.json')).exists()),None)
    if not dest.exists() and source:dest.write_bytes(source.read_bytes())
    if dest.exists():reused+=1;return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    result=recognize(engine,image,box,scale,rotation,rec,view);calls+=1
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
norm=lambda s:''.join(c.casefold() for c in s if c.isalnum())
library=read('D:/pcb5_submission/data/component_library/BoxICLibrary.json')
index={norm(k):k for k in library}
groups=[]
for owner in ['U1','USB1','USB2']:
    comp=baseline['components'][owner];a,b,c,d=comp['bbox'];frame=evidence.frame([a,h-d,c,h-b])[0]
    record=next(r for r in records if r['component_id']==owner)['bbox_top_left']
    box=[max(0,math.floor(record[0]-32)),max(0,math.floor(record[1]-32)),min(w,math.ceil(record[2]+32)),min(h,math.ceil(record[3]+32))]
    whole=[]
    for scale in [1.,3.]:
        for rotation in [0,90]:whole.extend(reader(rgb,box,scale,rotation,False,f'row_component_{owner}_s{scale}_r{rotation}'))
    matches=[{**t,'model':index[norm(t['text'])]} for t in whole if norm(t['text']) in index]
    rows=[]
    for side in ['left','right']:
        for lead in evidence.leads(frame,side):
            row=lead['row'];edge=lead['edge'];names=[];numbers=[]
            for t in whole:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                inward=cx-frame[0] if side=='left' else frame[2]-cx
                outward=frame[0]-cx if side=='left' else cx-frame[2]
                if 0<inward<.5*(frame[2]-frame[0]) and abs(cy-row)<=12 and any(c.isalpha() for c in t['text']):names.append(t)
                if 0<=outward<=52 and -16<=cy-row<=2 and t['text'].isalnum() and any(c.isdigit() for c in t['text']):numbers.append(t)
            name,ns=vote(names,DEFAULT);number,ds=vote(numbers,DEFAULT)
            rows.append({'lead':lead,'name':name,'name_support':ns,'number':number,'number_support':ds,'names':names,'numbers':numbers})
    groups.append({'owner':owner,'crop':box,'frame':frame,'models':matches,'whole':whole,'rows':rows})
result={'groups':groups,'new_local_ocr_calls':calls,'cached_calls':reused,'external_api_calls':0,'library_sha256':hashlib.sha256(Path('D:/pcb5_submission/data/component_library/BoxICLibrary.json').read_bytes()).hexdigest()}
(ROOT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for g in groups:print(json.dumps({k:v for k,v in g.items() if k!='whole'},ensure_ascii=True))
print('NEW',calls,'REUSED',reused)
