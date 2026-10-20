"""Image-only same-side length consensus on frozen predictions. GT used only afterward."""
import copy
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent/'side_consistency'
ROOT.mkdir(exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission')
import agent

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def generate(case):
    baseline=read(f'D:/pcb_open_gt_v37dots_b/final_results/{case}_Pred.json')
    raw=read(f'D:/pcb_open_gt_v37dots_b/component_results/{case}_Pred.json')
    image=np.asarray(Image.open(f'D:/pcb_open_gt_in/{case}.png').convert('RGB'))
    height,width=image.shape[:2]
    output=copy.deepcopy(baseline)
    audit=[]
    for component,pins in raw['pins'].items():
        comp=baseline['components'].get(component,{})
        if comp.get('type') not in {'ic','box','connector','raw_block','block'}:continue
        x1,y1,x2,y2=comp['bbox']
        groups=defaultdict(list)
        for key,pin in pins.items():
            if key not in output['pins'].get(component,{}):continue
            p=pin['point'];x,y=p['x'],p['y']
            distances={'left':abs(x-x1),'right':abs(x-x2),'top':abs(y-y2),'bottom':abs(y-y1)}
            side=min(distances,key=distances.get)
            # Distant or central points cannot vote for a body-edge lead.
            if distances[side]>60:continue
            length=agent.measure_lead(image,width,height,(x1,height-y2,x2,height-y1),side,(x,height-y))
            groups[side].append({'key':key,'raw_point':p,'length':length})
        for side,rows in groups.items():
            valid=[r for r in rows if r['length'] is not None and 1<=r['length']<=60]
            clusters=[]
            for r in valid:
                cluster=[q for q in valid if abs(q['length']-r['length'])<=3]
                mid=statistics.median(q['length'] for q in cluster)
                clusters.append((len(cluster),-sum(abs(q['length']-mid) for q in cluster),-mid,cluster))
            cluster=max(clusters,key=lambda t:t[:3])[3] if clusters else []
            # A repeated row cannot manufacture consensus. Majority of all side pins required.
            axis='y' if side in {'left','right'} else 'x'
            tangents=sorted(r['raw_point'][axis] for r in cluster)
            distinct=[]
            for t in tangents:
                if not distinct or t-distinct[-1]>6:distinct.append(t)
            accepted=len(distinct)>=3 and len(cluster)>len(rows)/2
            length=statistics.median(r['length'] for r in cluster) if accepted else None
            entry={'component':component,'side':side,'measurements':rows,
                   'accepted':accepted,'consensus_length':length,'changes':[]}
            if accepted:
                for r in rows:
                    point=output['pins'][component][r['key']]['point']
                    new=copy.deepcopy(point)
                    if side=='left':new['x']=x1-length
                    elif side=='right':new['x']=x2+length
                    elif side=='top':new['y']=y2+length
                    else:new['y']=y1-length
                    # Guard large corrections; avoid changing tangent, name, number or membership.
                    delta=((point['x']-new['x'])**2+(point['y']-new['y'])**2)**.5
                    if 0<delta<=25:
                        entry['changes'].append({'key':r['key'],'before':copy.deepcopy(point),'after':new})
                        point.update(new)
            audit.append(entry)
    assert baseline['components']==output['components'] and baseline['nets']==output['nets']
    assert all(set(pins)==set(output['pins'][c]) for c,pins in baseline['pins'].items())
    write(ROOT/f'{case}_Pred.json',output)
    write(ROOT/f'{case}_geometry_audit.json',audit)

# Freeze all experimental outputs before opening reference labels.
for case in ['1','11']:generate(case)

sys.path.insert(0,str(ROOT.parents[1]/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
results=[]
for case in ['1','11']:
    baseline=read(f'D:/pcb_open_gt_v37dots_b/final_results/{case}_Pred.json')
    pred=read(ROOT/f'{case}_Pred.json')
    truth=read(next((ROOT.parents[1]/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
    def correct(p,c,k):
        if c not in p['pins'] or k not in p['pins'][c]:return False
        a=p['pins'][c][k];b=truth['pins'][c][k]
        return a['pinname']==b['pinname'] and np.hypot(a['point']['x']-b['point']['x'],a['point']['y']-b['point']['y'])<=20
    gained=[];lost=[]
    for c,comp in truth['components'].items():
        if not metrics.is_ic(comp['type']):continue
        for k in truth['pins'].get(c,{}):
            old,new=correct(baseline,c,k),correct(pred,c,k)
            if new and not old:gained.append(f'{c}.{k}')
            if old and not new:lost.append(f'{c}.{k}')
    result={'case':case,'baseline':metrics.score_pins(baseline,truth,20)[0],
            'experimental':metrics.score_pins(pred,truth,20)[0],
            'gained':gained,'lost':lost,
            'changed_points':sum(len(e['changes']) for e in read(ROOT/f'{case}_geometry_audit.json')),
            'components_nets_and_keys_unchanged':True}
    results.append(result)
    print(json.dumps(result,ensure_ascii=False))
write(ROOT/'results.json',results)
