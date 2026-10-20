"""Read-only, post-hoc IC pin audit with model-candidate evidence."""
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PRED=Path('D:/pcb_open_gt_v37dots_b/final_results')
sys.path.insert(0,'D:/pcb5_submission')
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import agent
import platform_metrics as metrics

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
distance=lambda a,b:math.hypot(a[0]-b[0],a[1]-b[1])
by_hash={}
for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    by_hash[graph['image_sha256']]=folder
rows=[];totals=Counter();counts=Counter();coverage=[];candidates_by_case={}
for case in map(str,range(1,14)):
    pred=read(PRED/f'{case}_Pred.json')
    frozen=read(WORK/f'wire_contact_implementation_20261020/predictions/{case}_Pred.json')
    truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    # Power-only v2 additions do not alter scored IC containers.
    ic_owners={c for doc in [pred,frozen,truth] for c,item in doc['components'].items()
               if metrics.is_ic(item.get('type'))}
    same=all(pred['pins'].get(c,{})==frozen['pins'].get(c,{}) for c in ic_owners)
    scored,_=metrics.score_pins(pred,truth,20)
    totals.update({k:scored[k] for k in ['tp','fp','fn']})
    image=Path(f'D:/pcb_open_gt_in/{case}.png')
    folder=by_hash[sha(image)]
    model=read(folder/'p/export/engineering_pin_text_result.json')
    import cv2
    bgr=cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
    height=bgr.shape[0]
    primary=[p for p in model['pin_candidates'] if not p.get('text_hypothesis_primary_candidate_id')]
    snap={'components':copy.deepcopy(pred['components']),'pins':{},'nets':{}}
    for pin in primary:
        owner=pin.get('owner')
        if owner not in pred['components']:continue
        x,y=pin['point']
        snap['pins'].setdefault(owner,{})[pin['candidate_id']]={'pinname':pin.get('pinname') or '',
                                                               'point':{'x':x,'y':height-y}}
    agent.pin_lead_snap(snap,18,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),'color')
    candidate_points=[]
    for pin in primary:
        raw=(float(pin['point'][0]),float(height-pin['point'][1]))
        projected=snap['pins'].get(pin.get('owner'),{}).get(pin['candidate_id'],{}).get('point')
        projected=(projected['x'],projected['y']) if projected else raw
        candidate_points.append((raw,projected))
    gt=[]
    for owner,component in truth['components'].items():
        if metrics.is_ic(component.get('type')):
            for key,pin in truth['pins'].get(owner,{}).items():gt.append((owner,key,pin))
    # Reserve candidates jointly across every GT pin, so one noisy point cannot
    # explain multiple missing pins in a dense row. Dummy columns allow misses.
    matrix=np.full((len(gt),len(primary)+len(gt)),1e6,dtype=float)
    raw_dist=np.zeros((len(gt),len(primary)))
    for i,(owner,key,pin) in enumerate(gt):
        p=(pin['point']['x'],pin['point']['y'])
        for j,((raw,projected),candidate) in enumerate(zip(candidate_points,primary)):
            d=min(distance(p,raw),distance(p,projected))
            raw_dist[i,j]=d
            if d<=20:
                matrix[i,j]=d+(0 if candidate.get('owner')==owner else .1)
        matrix[i,len(primary)+i]=1000
    ir,jr=linear_sum_assignment(matrix)
    assigned={int(i):int(j) for i,j in zip(ir,jr) if j<len(primary) and matrix[i,j]<1000}
    per=Counter()
    for i,(owner,key,reference) in enumerate(gt):
        expected_number=key[4:]
        existing=pred['pins'].get(owner,{}).get(key)
        existing_dist=distance((existing['point']['x'],existing['point']['y']),
                               (reference['point']['x'],reference['point']['y'])) if existing else None
        name_ok=existing is not None and str(existing['pinname'])==str(reference['pinname'])
        if existing and name_ok and existing_dist<=20:continue
        stage='missing_number_key' if not existing else 'wrong_name' if not name_ok else 'wrong_position'
        j=assigned.get(i)
        model_pin=primary[j] if j is not None else None
        nearest_index=int(raw_dist[i].argmin()) if len(primary) else None
        nearest=primary[nearest_index] if nearest_index is not None else None
        nearest_dist=float(raw_dist[i,nearest_index]) if nearest_index is not None else None
        if stage=='wrong_position':category='exported_position_error'
        elif stage=='wrong_name':category='exported_name_or_number_mismatch'
        elif model_pin is None:
            category='no_primary_candidate_within_40px' if nearest_dist is None or nearest_dist>40 else 'candidate_offset_or_crowded_assignment'
        elif model_pin.get('owner')!=owner or model_pin.get('owner_status')!='assigned':
            category='candidate_owner_unresolved_or_wrong'
        elif model_pin.get('number') is None or model_pin.get('pinname') is None:
            category='candidate_fields_incomplete'
        elif str(model_pin.get('number'))!=expected_number:
            category='candidate_number_mismatch'
        elif str(model_pin.get('pinname'))!=str(reference['pinname']):
            category='candidate_name_mismatch'
        else:category='candidate_complete_but_not_exported'
        def brief(p,d):
            if p is None:return None
            return {'id':p['candidate_id'],'owner':p.get('owner'),'owner_status':p.get('owner_status'),
                    'number':p.get('number'),'pinname':p.get('pinname'),
                    'number_status':p.get('number_status'),'name_status':p.get('pinname_status'),
                    'point_top_left':p['point'],'distance_px':round(d,2)}
        near_exports=sorted([{'key':k,'name':p['pinname'],'distance_px':round(distance(
            (p['point']['x'],p['point']['y']),(reference['point']['x'],reference['point']['y'])),2)}
            for k,p in pred['pins'].get(owner,{}).items()],key=lambda p:p['distance_px'])[:2]
        row={'case':case,'component':owner,'key':key,'name':reference['pinname'],
             'gt_point':reference['point'],'category':category,'export_stage':stage,
             'exported':existing,'export_distance_px':round(existing_dist,2) if existing_dist is not None else None,
             'candidate':brief(model_pin,float(raw_dist[i,j])) if j is not None else None,
             'nearest_candidate':brief(nearest,nearest_dist),'nearest_exported_same_owner':near_exports}
        rows.append(row);counts[category]+=1;per[category]+=1
    coverage.append({'case':case,'score':scored,'frozen_ic_pins_identical':same,
                     'raw_candidates':len(model['pin_candidates']),'primary_candidates':len(primary),
                     'miss_categories':dict(per),'image_sha256':sha(image),'model_cache':str(folder)})
    candidates_by_case[case]=len(primary)
assert len(rows)==totals['fn']
result={'prediction_root':str(PRED),'tolerance_px':20,'cases':13,'totals':dict(totals),
        'gt_ic_pins':totals['tp']+totals['fn'],'categories':dict(counts),'per_case':coverage,'misses':rows,
        'frozen_ic_identical_cases':sum(c['frozen_ic_pins_identical'] for c in coverage),
        'new_model_inference_calls':0,'used_gt_only_for_diagnosis':True,
        'raw_candidate_note':'Same-image retained cache; distance uses raw or existing tip projection; primary candidates assigned one-to-one at 20px. No candidate within 40px is a suspect detection miss, not a visual proof.'}
(ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['per_case','misses','raw_candidate_note']},ensure_ascii=False))
for row in rows:
    print(json.dumps({k:row[k] for k in ['case','component','key','name','category','candidate','nearest_candidate']},ensure_ascii=False))
