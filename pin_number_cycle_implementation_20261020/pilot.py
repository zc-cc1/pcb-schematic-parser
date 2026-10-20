"""Run the closed-number swap on same-image caches, then evaluate frozen output."""
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
from pcb_parser.pin_number_cycle import reconcile_bottom_number_cycles
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
case='6';entry=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']==case)
stage=Path(entry['model_cache'])/'p';model=read(stage/'export/engineering_pin_text_result.json');graph=read(stage/'export/engineering_graph_proposals.json')
records=read(stage/'export/completed_component_records.json')['records']
image=Path(f'D:/pcb_open_gt_in/{case}.png');rgb=np.asarray(Image.open(image).convert('RGB'))
assert model['image_sha256']==hashlib.sha256(image.read_bytes()).hexdigest()
base=read(WORK/f'pin_evidence_implementation_20261020/deployed/{case}_Pred.json');original=copy.deepcopy((base,model,graph,records))
calls=0
def reader(image,box,scale,rotation,recognition_only,view):
    global calls
    patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,recognition_only)).encode()).hexdigest()
    path=ROOT/'ocr_cache'/(key+'.json');assert path.exists(),view
    calls+=1;return [{**t,'view':view} for t in read(path)]
out,audit=reconcile_bottom_number_cycles(base,model,graph,records,rgb,reader=reader)
agent.validate_target(out);assert original==(base,model,graph,records)
assert out['components']==base['components'] and out['nets'].keys()==base['nets'].keys()
for nid,n in base['nets'].items():assert out['nets'][nid]['edges']==n['edges']
write(ROOT/'6_pilot_Pred.json',out);write(ROOT/'pilot_audit.json',audit)
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/6').glob('*.json')))
def score(doc):
    pn,tn=logical_nets(doc),logical_nets(truth)
    conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    pin=metrics.score_pins(doc,truth,20)[0];wire=score_wire(doc,truth);comp=metrics.score_components(doc,truth)[0]
    return {'pin':pin,'connectivity':conn,'wire':wire,'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1'])}
result={'before':score(base),'after':score(out),'ocr_replay_calls':calls,'external_api_calls':0,'full_model_runs':0,'production_modified':False}
write(ROOT/'pilot_result.json',result)
print(json.dumps(audit,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
