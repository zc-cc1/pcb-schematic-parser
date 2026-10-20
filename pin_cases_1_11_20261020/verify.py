"""Replay postprocessing on copies and extract diagnostic crops; no inference or deployment."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, 'D:/pcb5_submission')
sys.path.insert(0, str(WORK/'pcb_parser_competition/scripts'))
import agent
import platform_metrics as metrics

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
rows = read(ROOT/'trace.json')
result = []
for row in rows:
    case = row['case']
    pred = read(f'D:/pcb_open_gt_v37dots_b/final_results/{case}_Pred.json')
    before = read(f'D:/pcb_open_gt_v37dots_b/component_results/{case}_Pred.json')
    before['components'] = copy.deepcopy(pred['components'])
    truth = read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    image = Image.open(f'D:/pcb_open_gt_in/{case}.png').convert('RGB')
    rgb = np.asarray(image)
    replay = copy.deepcopy(before)
    agent.pin_lead_snap(replay, 18, rgb, 'color')
    match = all(replay['pins'].get(c) == pred['pins'].get(c)
                for c, v in pred['components'].items() if metrics.is_ic(v['type']))
    selected = []
    for detail in row['details']:
        f = detail['failure']
        c, key = f['component'], f['key']
        if f['export_stage'] != 'wrong_position':
            continue
        box = pred['components'][c]['bbox']
        p = before['pins'][c][key]['point']
        gt = f['gt_point']
        d0 = ((p['x']-gt['x'])**2+(p['y']-gt['y'])**2)**.5
        x1,y1,x2,y2 = box
        height=image.height
        measured = {}
        for k, v in before['pins'][c].items():
            px,py=v['point']['x'],v['point']['y']
            sides = {'left':x1-px if px<x1 else None, 'right':px-x2 if px>x2 else None,
                     'bottom':y1-py if py<y1 else None, 'top':py-y2 if py>y2 else None}
            sides={k:v for k,v in sides.items() if v is not None}
            if sides:
                side=max(sides,key=sides.get)
                measured[k]={'side':side,'length':agent.measure_lead(rgb,image.width,image.height,
                    (x1,height-y2,x2,height-y1),side,(px,height-py))}
        selected.append({'component':c,'pin':key,'name':f['name'],'box':box,
                         'before_point':p,'before_distance':round(d0,3),
                         'final_point':pred['pins'][c][key]['point'],
                         'final_distance':f['export_distance_px'],'component_measurements':measured})
    # Fixed, image-only diagnostic crops, not inference inputs.
    for c in row['components']:
        box=row['components'][c]['record']['bbox_top_left']
        margin=55
        crop=image.crop((max(0,int(box[0])-margin),max(0,int(box[1])-margin),
                         min(image.width,int(box[2])+margin),min(image.height,int(box[3])+margin)))
        crop.resize((crop.width*2,crop.height*2)).save(ROOT/f'{case}_{c}_context.png')
    result.append({'case':case,'snap_replay_identical_scored_pins':match,
                   'baseline_score':metrics.score_pins(pred,truth,20)[0],
                   'without_snap_diagnostic_score':metrics.score_pins(before,truth,20)[0],
                   'position_failures':selected})
(ROOT/'verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in result:
    print(json.dumps(row,ensure_ascii=False))
