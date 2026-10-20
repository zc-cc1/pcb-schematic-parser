"""Automatic closed number permutation on retained image-model caches, then independent scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;LIVE=Path('D:/pcb5_submission')
parser=argparse.ArgumentParser();parser.add_argument('--deployed',action='store_true');parser.add_argument('--cases',default='1,2,3,4,5,6,7,8,9,10,11,12,13')
args=parser.parse_args();cases=args.cases.split(',')
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE));sys.path.insert(0,str(LIVE/'src'))
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
from pcb_parser.pin_field_recovery import recognize
from pcb_parser.pin_geometry_recovery import recover_image_leads
from pcb_parser.pin_text_ocr import create_engine
import agent
mode='deployed' if args.deployed else 'staged';out=ROOT/mode;out.mkdir(exist_ok=True)
cache=ROOT/'ocr_cache';cache.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
inventory={r['case']:r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case']}
engine=None;new_calls=seeded=reused=0;stats=[]
for case in cases:
    baseline=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/{case}_Pred.json')
    raw=read(LIVE.parent/f'pcb_open_gt_v37dots_b/component_results/{case}_Pred.json')
    folder=Path(inventory[case]['model_cache']);model=read(folder/'p/export/engineering_pin_text_result.json')
    graph=read(folder/'p/export/engineering_graph_proposals.json');records=read(folder/'p/export/completed_component_records.json')['records']
    original=copy.deepcopy((baseline,model,graph,records,raw))
    rgb=np.asarray(Image.open(LIVE.parent/f'pcb_open_gt_in/{case}.png').convert('RGB'))
    digest=hashlib.sha256((LIVE.parent/f'pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    assert graph['image_sha256']==digest
    before_counts=(new_calls,seeded,reused)
    def reader(image,box,scale,rotation,recognition_only,view):
        global engine,new_calls,seeded,reused
        patch=np.ascontiguousarray(image[box[1]:box[3],box[0]:box[2]])
        key=hashlib.sha256(patch.tobytes()+str((patch.shape,box,scale,rotation,recognition_only)).encode()).hexdigest()
        dest=cache/(key+'.json')
        previous=WORK/'pin_geometry_implementation_20261020/ocr_cache'/(key+'.json')
        if not dest.exists() and previous.exists():write(dest,read(previous))
        if dest.exists():
            reused+=1;return [{**t,'view':view} for t in read(dest)]
        if engine is None:engine,_=create_engine({'onnx_threads':4})
        texts=recognize(engine,image,box,scale,rotation,recognition_only,view)
        new_calls+=1;write(dest,texts);return texts
    # Exercise the same wrapper used by the live batch entrypoint.
    result,audit=agent.finalize_pin_vertical_repair(baseline,folder/'p',rgb,digest,reader=reader)
    agent.validate_target(result)
    assert original==(baseline,model,graph,records,raw), 'Input mutation'
    assert result['components']==baseline['components']
    assert baseline['nets'].keys()<=result['nets'].keys()
    for nid,net in baseline['nets'].items():assert net['edges']==result['nets'][nid]['edges']
    changes={(r['owner'],'pin_'+r['number']):r for r in audit['changes']}
    for owner,pins in baseline['pins'].items():
        for key,pin in pins.items():
            if (owner,key) in changes:
                assert pin==changes[owner,key]['before'] and result['pins'][owner][key]==changes[owner,key]['after']
            else:assert result['pins'][owner][key]==pin
    for change in audit['network_changes']:
        nid=change['network_id']
        def members(net):return sorted(v.strip() for v in net['hyperGraph'].strip('()').split(',') if v.strip())
        assert members(baseline['nets'][nid])==change['before']
        assert members(result['nets'][nid])==change['after']
    assert set(result['nets'])-set(baseline['nets'])=={r['network_id'] for r in audit['restored_networks']}
    write(out/f'{case}_Pred.json',result);write(out/f'{case}_audit.json',audit)
    if args.deployed:assert result==read(ROOT/f'staged/{case}_Pred.json')
    row={'case':case,'changed_fields':len(audit['changes']),'selected_components':len({p['owner'] for p in audit['groups']}),
         'network_changes':len(audit['network_changes']),'restored_networks':len(audit['restored_networks']),'new_ocr_calls':new_calls-before_counts[0],
         'seeded_ocr_views':seeded-before_counts[1],'reused_ocr_views':reused-before_counts[2],
         'budget_exhausted':audit['budget_exhausted']}
    stats.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)

# Freeze all predictions before loading any reference labels.
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
def score(pred,truth):
    pn,tn=logical_nets(pred),logical_nets(truth)
    connectivity=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
    pin=metrics.score_pins(pred,truth,20)[0];wire=score_wire(pred,truth)
    components=metrics.score_components(pred,truth)[0]
    total=100*(.3*components['f1']+.25*pin['f1']+.35*connectivity+.1*wire['f1'])
    return {'pin':pin,'connectivity':connectivity,'wire':wire,'score_100':total}
for row in stats:
    case=row['case'];baseline=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/{case}_Pred.json');pred=read(out/f'{case}_Pred.json')
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    row.update(before=score(baseline,truth),after=score(pred,truth))
    row['gained']=[];row['lost']=[]
    for c,comp in truth['components'].items():
        if not metrics.is_ic(comp['type']):continue
        for key,ref in truth['pins'].get(c,{}).items():
            def correct(doc):
                p=doc['pins'].get(c,{}).get(key)
                return p is not None and p['pinname']==ref['pinname'] and np.hypot(p['point']['x']-ref['point']['x'],p['point']['y']-ref['point']['y'])<=20
            a,b=correct(baseline),correct(pred)
            if b and not a:row['gained'].append(c+'.'+key[4:])
            if a and not b:row['lost'].append(c+'.'+key[4:])
    print('SCORE',json.dumps(row,ensure_ascii=False),flush=True)
report={'mode':mode,'cases':stats,'new_local_ocr_calls':new_calls,'seeded_views':seeded,'reused_views':reused,
        'external_api_calls':0,'full_model_runs':0,'predictions_frozen_before_reference_reads':True,
        'gained_pins':sum(len(r['gained']) for r in stats),'lost_pins':sum(len(r['lost']) for r in stats),
        'regressed_pages':[r['case'] for r in stats if r['after']['score_100']+1e-7<r['before']['score_100']],
        'macro_before':sum(r['before']['score_100'] for r in stats)/len(stats),
        'macro_after':sum(r['after']['score_100'] for r in stats)/len(stats)}
write(ROOT/f'{mode}_cached_check.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False),flush=True)
