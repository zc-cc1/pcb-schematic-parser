"""Freeze OCR-driven page-one predictions, then score with references separately."""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission');sys.path.insert(0,'D:/pcb5_submission/src')
import agent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
folder=Path(trace['folder'])
model=read(folder/'p/export/engineering_pin_text_result.json')
graph=read(folder/'p/export/engineering_graph_proposals.json')
baseline=read(WORK/'pin_position_implementation_20261020/deployed/1_Pred.json')
raw=read('D:/pcb_open_gt_v37dots_b/component_results/1_Pred.json')
rgb=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'));height=rgb.shape[0]
pmap={p['candidate_id']:p for p in model['pin_candidates']}
requested=[p for p in model['pin_candidates'] if
           (p.get('owner')=='U3' and p.get('pinname')=='VM') or
           (p.get('owner')=='USB1' and p.get('pinname') in {'DN1','SBU1'}) or
           (p.get('owner')=='USB2' and p['candidate_id'] in {'PIN_CAND_0107','PIN_CAND_0135','PIN_CAND_0155'})]
groups=[]
for pin in requested:
    c=pin['owner'];x,y=pin['point'];a,b,d,e=baseline['components'][c]['bbox'];box=[a,height-e,d,height-b]
    side=min({'left':abs(x-box[0]),'right':abs(x-box[2]),'top':abs(y-box[1]),'bottom':abs(y-box[3])},
             key=lambda s:{'left':abs(x-box[0]),'right':abs(x-box[2]),'top':abs(y-box[1]),'bottom':abs(y-box[3])}[s])
    members=[pin]
    # Explicit suppressed text hypotheses belong to their canonical point.
    members += [p for p in model['pin_candidates'] if p.get('text_hypothesis_primary_candidate_id')==pin['candidate_id']]
    # Collinear duplicate detections on a vertical lead can carry useful text.
    if side in {'top','bottom'}:
        for p in model['pin_candidates']:
            if p['candidate_id']==pin['candidate_id'] or p.get('owner')!=c:continue
            if abs(p['point'][0]-x)<=6 and abs(p['point'][1]-y)<=45:members.append(p)
    numbers={str(p['number']) for p in members if p.get('number') is not None}
    names={str(p['pinname']) for p in members if p.get('pinname') is not None}
    groups.append({'id':pin['candidate_id'],'owner':c,'point':pin['point'],'side':side,'box':box,
                   'members':sorted({p['candidate_id'] for p in members}),
                   'number':next(iter(numbers)) if len(numbers)==1 else None,
                   'name':next(iter(names)) if len(names)==1 else None,
                   'original_number':pin.get('number'),'original_name':pin.get('pinname')})

whole=[]
for owner in {'U3','USB1','USB2'}:
    whole.extend(t for path in (WORK/'pin_crop_experiment_20261020/ocr').glob(f'1_{owner}_*.json')
                 for t in [{**t,'owner':owner,'source':'component_crop'} for t in read(path)['texts']])
strips=[{**t,'owner':view['roi']['owner']} for view in read(ROOT/'collection.json')['views'] for t in view['texts']]
glyphs=[{**t,'owner':row['owner'],'target_candidate_id':row['candidate_id']}
        for row in read(ROOT/'glyph_collection.json')['rows'] for t in row['texts']]

def number_layout(owner,side):
    box=baseline['components'][owner]['bbox'];edge=box[0] if side=='left' else box[2]
    column=[(t['bbox'][0]+t['bbox'][2])/2 for t in whole if t['owner']==owner and t['text'].isdigit()
            and t['confidence']>=.98 and abs((t['bbox'][0]+t['bbox'][2])/2-edge)<30]
    offsets=[]
    for p in model['pin_candidates']:
        if p.get('owner')!=owner or not p.get('number') or p.get('number_source')=='library_pin_name':continue
        for t in whole:
            cy=(t['bbox'][1]+t['bbox'][3])/2
            if t['owner']==owner and t['text']==p['number'] and t['confidence']>=.98 and abs(cy-p['point'][1])<=16:
                offsets.append(cy-p['point'][1])
    return statistics.median(column) if column else edge, statistics.median(offsets) if offsets else 0

def vote(tokens):
    views=defaultdict(set);evidence=defaultdict(list)
    for t in tokens:
        if t['confidence']>=.98:
            views[t['text']].add(t['view']);evidence[t['text']].append(t)
    rank=sorted(views,key=lambda text:(-len(views[text]),text))
    # High OCR confidence cannot replace repeated evidence or resolve a tie.
    if not rank or len(views[rank[0]])<2:return None,{'reason':'less_than_two_views','support':{k:len(v) for k,v in views.items()}}
    if len(rank)>1 and len(views[rank[1]])>=2:return None,{'reason':'conflicting_spelling','support':{k:len(v) for k,v in views.items()}}
    return rank[0],{'reason':'accepted','support':{k:len(v) for k,v in views.items()},'readings':evidence[rank[0]]}

def preferred_rotation(owner,side):
    column,offset=number_layout(owner,side)
    support=defaultdict(set)
    for p in model['pin_candidates']:
        if p.get('owner')!=owner or not p.get('number') or p.get('number_source')=='library_pin_name':continue
        for t in whole:
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if t['owner']!=owner or t['confidence']<.98 or t['text']!=p['number']:continue
            if abs(cx-column)>12 or abs(cy-(p['point'][1]+offset))>8:continue
            rotation=90 if t['view'].endswith('_r90') else 0
            support[rotation].add((p['candidate_id'],t['view']))
    rank=sorted(support,key=lambda r:(-len(support[r]),r))
    return rank[0] if rank and len(support[rank[0]])>=3 and (len(rank)==1 or len(support[rank[0]])>len(support[rank[1]])) else None

def recover(texts,orientation_check=False):
    result=copy.deepcopy(groups)
    for g in result:
        owner,side=g['owner'],g['side'];x,y=g['point'];box=g['box'];number_evidence=[]
        if side in {'left','right'}:
            column,offset=number_layout(owner,side)
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                if t['owner']!=owner or not t['text'].isdigit():continue
                if t.get('target_candidate_id') and t['target_candidate_id']!=g['id']:continue
                if abs(cx-column)<=12 and abs(cy-(y+offset))<=8:number_evidence.append(t)
        else:
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                inside=box[0]<cx<box[2] and box[1]<cy<box[3]
                edge=box[1] if side=='top' else box[3]
                if t['owner']==owner and t['text'].isdigit() and not inside and abs(cx-x)<=15 and abs(cy-edge)<=45:
                    number_evidence.append(t)
        g['rotation_policy']=None
        spellings={t['text'] for t in number_evidence if t['confidence']>=.98}
        if orientation_check and side in {'left','right'} and spellings=={'6','9'}:
            preferred=preferred_rotation(owner,side)
            if preferred is not None:
                number_evidence=[t for t in number_evidence if (90 if t['view'].endswith('_r90') else 0)==preferred]
                g['rotation_policy']={'preferred':preferred,'reason':'existing_component_number_observations'}
        number,audit=vote(number_evidence);g['number_audit']=audit
        if number is not None:g['number']=number
        if not g['name'] and side in {'top','bottom'} and g['number']:
            inner=[]
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                edge=box[1] if side=='top' else box[3]
                if t['owner']==owner and t['text']==g['number'] and box[0]<cx<box[2] and box[1]<cy<box[3] and abs(cx-x)<=15 and abs(cy-edge)<=35:
                    inner.append(t)
            name,audit=vote(inner);g['name_audit']=audit
            if name is not None:g['name']=name
    # Numbers claimed outside the selected physical groups remain reserved.
    selected={cid for g in result for cid in g['members']}
    reserved={(p.get('owner'),str(p['number'])) for p in model['pin_candidates']
              if p['candidate_id'] not in selected and p.get('number') is not None}
    counts=defaultdict(int)
    for g in result:
        if g['number'] is not None:counts[g['owner'],g['number']]+=1
    for g in result:
        if g['number'] is not None and ((g['owner'],g['number']) in reserved or counts[g['owner'],g['number']]>1):
            g['withheld']='number_conflict';g['number']=None
    return result

def export(recovered):
    out=copy.deepcopy(baseline);snap=copy.deepcopy(raw)
    snap['components']=copy.deepcopy(baseline['components']);snap['nets']={}
    changes=[]
    for g in recovered:
        if g['number'] is None or g['name'] is None:continue
        if g['number']==g['original_number'] and g['name']==g['original_name']:continue
        c,key=g['owner'],'pin_'+g['number'];x,y=g['point']
        snap['pins'].setdefault(c,{})[key]={'pinname':g['name'],'point':{'x':x,'y':height-y}}
        changes.append(g)
    agent.pin_lead_snap(snap,18,rgb,'color')
    for g in changes:
        c,key=g['owner'],'pin_'+g['number'];out['pins'].setdefault(c,{})[key]=copy.deepcopy(snap['pins'][c][key])
    agent.validate_target(out)
    assert out['components']==baseline['components'] and out['nets']==baseline['nets']
    return out,changes

variants={'baseline':(copy.deepcopy(baseline),[],[])}
for name,texts in [('merge_only',[]),('existing_crop_binding',whole),
                   ('local_strip_binding',whole+strips),('isolated_digit_binding',whole+strips+glyphs),
                   ('orientation_checked_binding',whole+strips+glyphs)]:
    recovered=recover(texts,orientation_check=name=='orientation_checked_binding');prediction,changes=export(recovered)
    variants[name]=(prediction,recovered,changes)
    folder_out=ROOT/name;folder_out.mkdir(exist_ok=True)
    write(folder_out/'1_Pred.json',prediction);write(folder_out/'binding_audit.json',recovered)

# Freeze predictions before any reference label is opened.
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/1').glob('*.json')))
results={}
for name,(pred,recovered,changes) in variants.items():
    gained=[];lost=[]
    for c,comp in truth['components'].items():
        if not metrics.is_ic(comp['type']):continue
        for key,ref in truth['pins'].get(c,{}).items():
            def correct(doc):
                p=doc['pins'].get(c,{}).get(key)
                return p is not None and p['pinname']==ref['pinname'] and math.hypot(
                    p['point']['x']-ref['point']['x'],p['point']['y']-ref['point']['y'])<=20
            a,b=correct(baseline),correct(pred)
            if b and not a:gained.append(c+'.'+key[4:])
            if a and not b:lost.append(c+'.'+key[4:])
    results[name]={'pin':metrics.score_pins(pred,truth,20)[0],'gained':gained,'lost':lost,
                   'changes':[{'candidate_id':g['id'],'owner':g['owner'],'number':g['number'],'name':g['name']} for g in changes],
                   'components_and_nets_unchanged':True,'scope':'pin_fields_and_selected_positions_only'}
    print(name,json.dumps(results[name],ensure_ascii=False))
write(ROOT/'results.json',{'variants':results,'production_modified':False,'gt_used_before_prediction_freeze':False,
                          'external_api_calls':0,'net_membership_validation_pending':True})
