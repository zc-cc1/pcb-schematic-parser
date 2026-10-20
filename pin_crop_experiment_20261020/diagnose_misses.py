"""Post-hoc error attribution only; never called by crop/adoption code."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
result=read(ROOT/'results.json')
counts=Counter();rows=[]
for case_result in result['per_case']:
    case=case_result['case']
    base=read(ROOT/f'baseline/{case}_Pred.json')
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    for owner,component in truth['components'].items():
        if not metrics.is_ic(component.get('type')):continue
        for key,pin in truth.get('pins',{}).get(owner,{}).items():
            predicted=base['pins'].get(owner,{}).get(key)
            if predicted is None:kind='missing_exported_number_key'
            elif str(predicted['pinname'])!=str(pin['pinname']):kind='wrong_name_at_existing_number'
            elif math.dist(tuple(predicted['point'][v] for v in ['x','y']),tuple(pin['point'][v] for v in ['x','y']))>20:
                kind='name_number_correct_but_point_over_20px'
            else:continue
            counts[kind]+=1
            rows.append({'case':case,'owner':owner,'key':key,'name':pin['pinname'],'kind':kind})
fn=sum(r['variants']['baseline']['metrics']['pin']['fn'] for r in result['per_case'])
assert sum(counts.values())==fn
payload={'gt_pin_fn':fn,'categories':dict(counts),'rows':rows,
         'post_hoc_only':True,'used_by_prediction_code':False}
(ROOT/'miss_categories.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'gt_pin_fn':fn,'categories':dict(counts)},ensure_ascii=False))
