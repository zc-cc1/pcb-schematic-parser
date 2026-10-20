"""Replay the staged/deployed entrypoint on 13 cached pages, then score separately."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
parser=argparse.ArgumentParser()
parser.add_argument('--deployed',action='store_true')
args=parser.parse_args()
LIVE=Path('D:/pcb5_submission')
sys.dont_write_bytecode=True
sys.path.insert(0,str(LIVE))
sys.path.insert(0,str(LIVE/'src'))
if not args.deployed:sys.path.insert(0,str(ROOT/'runtime'))
import pcb_parser
if not args.deployed:pcb_parser.__path__.insert(0,str(ROOT/'runtime/src/pcb_parser'))
import agent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
import platform_metrics_v3 as aggregate
assert aggregate.PIN_TOLERANCE_PX == 20

mode='deployed' if args.deployed else 'staged'
out=ROOT/mode
out.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
summaries=[]
for case in map(str,range(1,14)):
    baseline=read(LIVE.parent/f'pcb_open_gt_v37dots_b/final_results/{case}_Pred.json')
    raw=read(LIVE.parent/f'pcb_open_gt_v37dots_b/component_results/{case}_Pred.json')
    pre=copy.deepcopy(baseline)
    # Reconstruct only the coordinate state immediately before pin_lead_snap;
    # all final fields, component boxes and power-wire exports stay frozen.
    for c,pins in raw['pins'].items():
        if baseline['components'].get(c,{}).get('type') not in {'ic','box','connector','raw_block','block'}:continue
        for key,pin in pins.items():
            if key in pre['pins'].get(c,{}):pre['pins'][c][key]['point']=copy.deepcopy(pin['point'])
    rgb=np.asarray(Image.open(LIVE.parent/f'pcb_open_gt_in/{case}.png').convert('RGB'))
    disabled=copy.deepcopy(pre)
    audit={}
    with patch.dict(os.environ,{'PCB_PIN_SIDE_CONSENSUS':'0'}):
        agent.pin_lead_snap(disabled,18,rgb,'color',side_audit=audit)
    assert disabled==baseline, f'Legacy replay differs on page {case}'
    assert audit['enabled'] is False
    enabled=copy.deepcopy(pre)
    with patch.dict(os.environ,{'PCB_PIN_SIDE_CONSENSUS':'1'}):
        agent.pin_lead_snap(enabled,18,rgb,'color',side_audit=audit)
    agent.validate_target(enabled)
    assert enabled['components']==baseline['components']
    assert enabled['nets']==baseline['nets']
    stripped=copy.deepcopy(enabled)
    for c,pins in stripped['pins'].items():
        for key,pin in pins.items():pin['point']=copy.deepcopy(baseline['pins'][c][key]['point'])
    assert stripped==baseline, 'Only pin point values may change'
    if case in {'1','11'}:
        small=read(WORK/f'pin_cases_1_11_20261020/side_consistency/{case}_Pred.json')
        assert small==enabled, 'Staged rule differs from the frozen small experiment'
    write(out/f'{case}_Pred.json',enabled)
    write(out/f'{case}_pin_side_consensus.json',audit)
    if args.deployed:
        assert read(ROOT/f'staged/{case}_Pred.json')==enabled
    summaries.append({'case':case,'legacy_replay_identical':True,
                      'only_points_changed':True,'modified_pin_count':audit['modified_pin_count']})

# No reference labels were opened during generation above.
for row in summaries:
    case=row['case']
    baseline=read(LIVE.parent/f'pcb_open_gt_v37dots_b/final_results/{case}_Pred.json')
    pred=read(out/f'{case}_Pred.json')
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    before=aggregate.score_case(baseline,truth)
    after=aggregate.score_case(pred,truth)
    assert before['pin_f1']==metrics.score_pins(baseline,truth,20)[0]
    assert after['pin_f1']==metrics.score_pins(pred,truth,20)[0]
    for field in ['component_f1','connectivity_f1','wire_f1']:
        assert before[field]==after[field]
    row.update(baseline=before['pin_f1'],current=after['pin_f1'],
               baseline_total=before['final_score'],current_total=after['final_score'])
    row['gained']=[];row['lost']=[]
    for c,component in truth['components'].items():
        if not metrics.is_ic(component['type']):continue
        for key,ref in truth['pins'].get(c,{}).items():
            def correct(doc):
                p=doc['pins'].get(c,{}).get(key)
                return p is not None and p['pinname']==ref['pinname'] and np.hypot(
                    p['point']['x']-ref['point']['x'],p['point']['y']-ref['point']['y'])<=20
            old,new=correct(baseline),correct(pred)
            if new and not old:row['gained'].append(f'{c}.{key}')
            if old and not new:row['lost'].append(f'{c}.{key}')
    print(json.dumps(row,ensure_ascii=False))
result={'mode':mode,'pages':summaries,'pin_tolerance_px':20,'new_model_calls':0,'no_production_reference_reads':True,
        'regressed_pages':[r['case'] for r in summaries if r['current']['f1']<r['baseline']['f1']],
        'baseline_macro_total':sum(r['baseline_total'] for r in summaries)/len(summaries),
        'current_macro_total':sum(r['current_total'] for r in summaries)/len(summaries),
        'gained_pin_count':sum(len(r['gained']) for r in summaries),
        'lost_pin_count':sum(len(r['lost']) for r in summaries)}
write(ROOT/f'{mode}_cached_check.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='pages'},ensure_ascii=False))
