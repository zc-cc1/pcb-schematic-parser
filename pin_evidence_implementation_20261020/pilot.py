"""Generate the page-five pilot without reading reference labels until export."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
sys.dont_write_bytecode=True;sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
import pcb_parser
pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
from pcb_parser.pin_evidence_repair import repair_observed_pin_evidence
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_text_ocr import create_engine

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
case='5';item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']==case)
stage=Path(item['model_cache'])/'p';model=read(stage/'export/engineering_pin_text_result.json');graph=read(stage/'export/engineering_graph_proposals.json')
records=read(stage/'export/completed_component_records.json')['records']
image=Path(f'D:/pcb_open_gt_in/{case}.png');rgb=np.asarray(Image.open(image).convert('RGB'))
assert graph['image_sha256']==hashlib.sha256(image.read_bytes()).hexdigest()
base=read(WORK/f'pin_pairing_implementation_20261020/deployed/{case}_Pred.json');original=copy.deepcopy((base,model,graph,records))
cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True);engine=None;new=reused=0
def reader(image,box,scale,rotation,recognition_only,view):
    global engine,new,reused
    patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,recognition_only)).encode()).hexdigest();dest=cache/(key+'.json')
    previous=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')
    if not dest.exists() and previous.exists():write(dest,read(previous))
    if dest.exists():reused+=1;return [{**t,'view':view} for t in read(dest)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    tokens=recognize(engine,image,box,scale,rotation,recognition_only,view);write(dest,tokens);new+=1;return tokens
out,audit=repair_observed_pin_evidence(base,model,graph,records,rgb,reader=reader)
assert original==(base,model,graph,records);agent.validate_target(out)
assert base['components']==out['components'] and base['nets']==out['nets']
write(ROOT/'5_pilot_Pred.json',out);write(ROOT/'pilot_audit.json',audit)
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/5').glob('*.json')))
result={'before':metrics.score_pins(base,truth,20)[0],'after':metrics.score_pins(out,truth,20)[0],
        'new_local_ocr_calls':new,'reused_observations':reused,'external_api_calls':0,'full_model_runs':0,'production_modified':False}
write(ROOT/'pilot_result.json',result)
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
