"""Freeze image-only predictions before reading reference labels for scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
parser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');parser.add_argument('--fresh',action='store_true');parser.add_argument('--cases',default='13');args=parser.parse_args()
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.pin_field_recovery import recognize
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
mode='deployed' if args.deployed else 'staged';dest=ROOT/mode;dest.mkdir(exist_ok=True)
cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
inventory={r['case']:r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case']}
engine=None;new_calls=0;reused=0
def reader(image,box,scale,rotation,rec,view):
    global engine,new_calls,reused
    patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
    key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,rec)).encode()).hexdigest()
    path=cache/(key+'.json')
    if path.exists() and not args.fresh:
        reused+=1;return [{**t,'view':view} for t in read(path)]
    if engine is None:engine,_=create_engine({'onnx_threads':4})
    tokens=recognize(engine,image,box,scale,rotation,rec,view)
    new_calls+=1;write(path,tokens);return tokens
stats=[]
for case in args.cases.split(','):
    baseline=read(WORK/f'pin_row_library_implementation_20261020/deployed/{case}_Pred.json')
    ocr=read(f'D:/pcb_open_gt_v37dots_b/logs/{case}_ocr_texts.json')
    unchanged=copy.deepcopy((baseline,ocr))
    path=Path(f'D:/pcb_open_gt_in/{case}.png');image=np.asarray(Image.open(path).convert('RGB'))
    result,audit=agent.finalize_pin_numeric_connector(baseline,Path(inventory[case]['model_cache'])/'p',image,hashlib.sha256(path.read_bytes()).hexdigest(),ocr,reader=reader)
    assert (baseline,ocr)==unchanged
    agent.validate_target(result)
    for owner,comp in baseline['components'].items():assert result['components'][owner]==comp
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():assert result['pins'][owner][key]==pin
    for nid,net in baseline['nets'].items():assert result['nets'][nid]['edges']==net['edges']
    # Idempotence is verified without any extra inference.
    def no_read(*unused):raise AssertionError('Idempotent pass performed OCR')
    repeated,second=agent.finalize_pin_numeric_connector(result,Path(inventory[case]['model_cache'])/'p',image,hashlib.sha256(path.read_bytes()).hexdigest(),ocr,reader=no_read)
    assert repeated==result and second['ocr_calls']==0
    if args.deployed:assert result==read(ROOT/f'staged/{case}_Pred.json')
    if args.fresh:
        assert result==read(dest/f'{case}_Pred.json')
        write(dest/f'{case}_fresh_audit.json',audit)
    else:
        write(dest/f'{case}_Pred.json',result);write(dest/f'{case}_audit.json',audit)
    stats.append({'case':case,'changed_pins':len(audit['changes']),'changed_components':len(audit['component_changes']),
                  'network_changes':len(audit['network_changes']),'ocr_calls':audit['ocr_calls'],'unchanged':result==baseline})

# All results have now been frozen; references are loaded only below.
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
def score(pred,truth):
    pin=metrics.score_pins(pred,truth,20)[0];comp=metrics.score_components(pred,truth)[0];wire=score_wire(pred,truth)
    pn,tn=logical_nets(pred),logical_nets(truth)
    conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    return {'pin':pin,'components':comp,'connectivity':conn,'wire':wire,'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1'])}
for row in stats:
    case=row['case'];truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    row['before']=score(read(WORK/f'pin_row_library_implementation_20261020/deployed/{case}_Pred.json'),truth)
    row['after']=score(read(dest/f'{case}_Pred.json'),truth)
report={'mode':mode,'fresh':args.fresh,'cases':stats,'new_local_ocr_calls':new_calls,'reused_views':reused,
        'external_api_calls':0,'full_model_runs':0,'predictions_frozen_before_reference_reads':True,
        'regressed_pages':[r['case'] for r in stats if r['after']['score_100']+1e-7<r['before']['score_100']],
        'macro_before':sum(r['before']['score_100'] for r in stats)/len(stats),'macro_after':sum(r['after']['score_100'] for r in stats)/len(stats)}
write(ROOT/(mode+('_fresh_check.json' if args.fresh else '_cached_check.json')),report)
print(json.dumps(report,ensure_ascii=True))
