"""Recover visible headers missed by the frozen generic component binder."""
from collections import defaultdict
from copy import deepcopy
import math
import re

import numpy as np

from .component_text import center, gap, iou, choose_global


def observed_words(texts):
    output=[];seen=set()
    for row in texts:
        observations=[row]+row.get('observations',[])
        for obs in observations:
            text=str(obs['text']);box=list(map(float,obs.get('bbox',row['bbox'])));confidence=float(obs['confidence'])
            view=obs.get('view',row.get('view','external'))
            if not all(math.isfinite(v) for v in box+[confidence]) or len(box)!=4 or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('Invalid OCR header evidence')
            for word in re.finditer(r'\S+',text):
                b=box.copy();approximate=word.group()!=text
                if approximate:
                    if b[3]-b[1]>1.8*(b[2]-b[0]):continue
                    width=b[2]-b[0];b[0],b[2]=b[0]+width*word.start()/len(text),b[0]+width*word.end()/len(text)
                key=(word.group(),tuple(round(v,2) for v in b),view,round(confidence,5))
                if key in seen:continue
                seen.add(key);output.append({'text':word.group(),'bbox':b,'confidence':confidence,'view':view,
                                             'source_text':text,'approximate_split_geometry':approximate})
    return output


def header_position(box,record,pins,cfg):
    """Learned side tips bound the title zone, including boxes containing titles."""
    db=record['bbox_top_left'];cx,cy=center(box)
    if gap(box,db)>cfg['max_header_gap_px'] or not db[0]-20<=cx<=db[2]+20:return None
    inset=min(cfg['max_header_inset_px'],(db[3]-db[1])*cfg['max_header_inset_fraction'])
    side_tips=[p for p in pins if min(abs(p['point'][0]-db[0]),abs(p['point'][0]-db[2]))
               <=min(abs(p['point'][1]-db[1]),abs(p['point'][1]-db[3]))]
    top=min(db[1]+inset,min((p['point'][1] for p in side_tips),default=db[1]+inset)-cfg['pin_header_guard_px'])
    bottom=max(db[3]-inset,max((p['point'][1] for p in side_tips),default=db[3]-inset)+cfg['pin_header_guard_px'])
    if cy<=top:return 'top'
    if cy>=bottom:return 'bottom'
    return None


def spatial_observations(items):
    groups=[]
    for item in items:
        for group in groups:
            if all(iou(item['bbox'],old['bbox'])>=.4 for old in group):group.append(item);break
        else:groups.append([item])
    return groups


def recover_component_identities(ownership,texts,records,cfg):
    result=deepcopy(records);words=observed_words(texts);owned=defaultdict(list)
    for pin in ownership['pin_candidates']:
        if pin['owner_status']=='assigned':owned[pin['owner_detection_index']].append(pin)
    existing={r['designator'] for r in records if r.get('designator')}
    occupied=[r['fields'][field]['bbox'] for r in records for field in ['designator','Name','value'] if field in r.get('fields',{})]
    pattern=re.compile(r'^('+ '|'.join(sorted(cfg['reference_prefixes'],key=len,reverse=True))+r')\d{1,4}(?:[A-Za-z])?$',re.I)
    groups=defaultdict(list)
    for word in words:
        m=pattern.fullmatch(word['text'])
        if not m or word['text'] in existing or word['confidence']<cfg['view_support_min_confidence']:continue
        if any(iou(word['bbox'],box)>=.45 for box in occupied):continue
        groups[word['text']].append({**word,'prefix':m[1].upper()})
    groups={text:items for text,items in groups.items() if len({w['view'] for w in items})>=cfg['min_distinct_views']
            and max(w['confidence'] for w in items)>=cfg['min_ocr_confidence']}
    active=[r for r in result if not r.get('designator') and r['type'] in cfg['component_types']
            and r.get('duplicate_of_detection_index') is None and len(owned[r['detection_index']])>=2]
    keys=sorted(groups);costs=np.full((len(active),len(keys)),1e6);selected={};debug={'candidates':[],'rejected_matches':[],'accepted':[]}
    for i,record in enumerate(active):
        index=record['detection_index']
        for j,text in enumerate(keys):
            options=[]
            for cluster in spatial_observations(groups[text]):
                valid=[]
                for word in cluster:
                    if word['prefix'] in {'H','LD'} and len(owned[index])<cfg['minimum_pins_special_prefix']:continue
                    pos=header_position(word['bbox'],record,owned[index],cfg)
                    if pos is not None:valid.append({**word,'header_position':pos})
                if len({w['view'] for w in valid})<cfg['min_distinct_views']:continue
                for word in valid:
                    db=record['bbox_top_left'];cost=gap(word['bbox'],db)+.04*abs(center(word['bbox'])[0]-db[0])+10*(1-word['confidence'])
                    if word['confidence']>=cfg['min_ocr_confidence']:options.append((cost,{**word,'local_observations':valid}))
            if options:
                costs[i,j],selected[i,j]=min(options,key=lambda r:r[0])
                debug['candidates'].append({'detection_index':index,'text':text,'cost':float(costs[i,j]),'view_count':len({w['view'] for w in groups[text]})})
    debug['withheld_duplicates']=[]
    for j,text in enumerate(keys):
        anchors=[selected[i,j]['bbox'] for i in range(len(active)) if (i,j) in selected]
        if any(iou(a,b)<.1 and math.dist(center(a),center(b))>20 for a in anchors for b in anchors):
            costs[:,j]=1e6;debug['withheld_duplicates'].append({'text':text,'reason':'repeated_header_designator_source_order_unobservable'})
    assignments=choose_global(costs,cfg['unmatched_cost'],cfg['assignment_margin'],debug['rejected_matches'],'header_designator')
    for i,(j,cost,margin) in assignments.items():
        record=active[i];text=keys[j];word=selected[i,j]
        record.update(component_id=text,designator=text,identity_status='observed_header_recovery')
        record['fields']['designator']={**word,'source':'learned_tip_header_region_OCR','cost':cost,
                                         'global_assignment_margin':margin,'observations':word['local_observations']}
        debug['accepted'].append({'detection_index':record['detection_index'],'designator':text,'evidence':record['fields']['designator']})
        # A model string printed in the same title zone can be recovered too.
        # Function/pin strings with underscores and title references are excluded.
        if record.get('Name') is None:
            models=defaultdict(list)
            for token in words:
                if (token['confidence']<cfg['min_ocr_confidence'] or pattern.fullmatch(token['text'])
                    or not re.fullmatch(r'[A-Za-z][A-Za-z0-9.+/\-]{3,}',token['text']) or not re.search(r'\d',token['text'])):continue
                if any(iou(token['bbox'],b)>=.45 for b in occupied):continue
                pos=header_position(token['bbox'],record,owned[record['detection_index']],cfg)
                if pos is not None:models[token['text']].append(token)
            ranked=[]
            for name,all_items in models.items():
                for items in spatial_observations(all_items):
                    if len({w['view'] for w in items})<cfg['min_distinct_views']:continue
                    score,best=min([(gap(w['bbox'],record['bbox_top_left'])+.02*math.dist(center(w['bbox']),center(word['bbox'])),w) for w in items],key=lambda r:r[0])
                    ranked.append((score,name,best,items))
            ranked.sort(key=lambda r:r[0])
            if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0]>=cfg['assignment_margin']):
                score,name,best,items=ranked[0];record['Name']=name
                record['fields']['Name']={**best,'source':'observed_IC_header_model','observations':items,'cost':score}
                occupied.append(best['bbox'])
        occupied.append(word['bbox'])
    if len({r['component_id'] for r in result})!=len(result):raise ValueError('Recovered identities collide')
    debug.update(new_identities=len(debug['accepted']),new_model_names=sum(r.get('Name') is not None and old.get('Name') is None for r,old in zip(result,records)),
                 policy='Unresolved IC/header records only; visible spellings and learned tips; no source CAD IDs or inferred text')
    return result,debug
