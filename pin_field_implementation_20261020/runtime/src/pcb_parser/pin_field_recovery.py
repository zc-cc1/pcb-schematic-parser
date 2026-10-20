"""Recover numeric IC pin fields from current-image OCR and model candidates.

No reference labels, case names, expected pin numbers or component identities
are encoded here. Missing geometry and uncertain ownership are left unchanged.
"""
from collections import defaultdict
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

from .component_text import restore_polygon

DEFAULT = {'max_components': 6, 'max_groups': 32, 'max_ocr_calls': 160,
           'min_confidence': .98, 'min_views': 2, 'scales': [3., 5.],
           'component_scales': [1., 3.], 'rotations': [0, 90]}
IC_TYPES = {'box', 'ic', 'connector', 'raw_block', 'block'}


def _side(point, box):
    x,y=point;a,b,c,d=box
    return min({'left':abs(x-a),'right':abs(x-c),'top':abs(y-b),'bottom':abs(y-d)},
               key=lambda s:{'left':abs(x-a),'right':abs(x-c),'top':abs(y-b),'bottom':abs(y-d)}[s])


def _outward(point, side, box):
    x,y=point
    return {'left':box[0]-x,'right':x-box[2],'top':box[1]-y,'bottom':y-box[3]}[side]


def build_groups(target, document, height):
    by_owner=defaultdict(list)
    for pin in document['pin_candidates']:
        if pin.get('owner_status')=='assigned' and target['components'].get(pin.get('owner'),{}).get('type') in IC_TYPES:
            by_owner[pin['owner']].append(pin)
    groups=[]
    for owner,pins in sorted(by_owner.items()):
        a,b,c,d=target['components'][owner]['bbox'];box=[a,height-d,c,height-b]
        pmap={p['candidate_id']:p for p in pins};parent={cid:cid for cid in pmap}
        def root(cid):
            while parent[cid]!=cid:cid=parent[cid]
            return cid
        def join(a,b):
            a,b=root(a),root(b)
            if a!=b:parent[max(a,b)]=min(a,b)
        for p in pins:
            primary=p.get('text_hypothesis_primary_candidate_id')
            if primary in pmap:join(p['candidate_id'],primary)
        for i,p in enumerate(pins):
            side=_side(p['point'],box)
            if side not in {'top','bottom'}:continue
            for q in pins[i+1:]:
                if _side(q['point'],box)!=side:continue
                if abs(p['point'][0]-q['point'][0])<=6 and abs(p['point'][1]-q['point'][1])<=45:
                    # Merge only unnamed duplicate detections; never collapse two names.
                    if not p.get('pinname') and not q.get('pinname'):join(p['candidate_id'],q['candidate_id'])
        sets=defaultdict(list)
        for p in pins:sets[root(p['candidate_id'])].append(p)
        for members in sets.values():
            primaries=[p for p in members if not p.get('text_hypothesis_primary_candidate_id')]
            if not primaries:continue
            primary=min(primaries,key=lambda p:p['candidate_id']);side=_side(primary['point'],box)
            if side in {'top','bottom'}:
                primary=max(primaries,key=lambda p:(_outward(p['point'],side,box),p.get('confidence',0)))
            numbers={str(p['number']) for p in members if p.get('number') is not None}
            names={str(p['pinname']) for p in members if p.get('pinname') is not None}
            groups.append({'id':primary['candidate_id'],'owner':owner,'point':primary['point'],
                           'side':side,'box':box,'members':sorted(p['candidate_id'] for p in members),
                           'number':next(iter(numbers)) if len(numbers)==1 else None,
                           'name':next(iter(names)) if len(names)==1 else None,
                           'original_number':primary.get('number'),'original_name':primary.get('pinname'),
                           'library_number':primary.get('number_source')=='library_pin_name',
                           'ambiguous':len(numbers)>1 or len(names)>1})
    return groups


def select_groups(groups, cfg):
    eligible=[]
    for g in groups:
        if g['ambiguous'] or g['number'] is not None and not g['number'].isdigit():continue
        repeated=g['library_number'] and g['name'] and sum(
            q['owner']==g['owner'] and q['name']==g['name'] for q in groups)>1
        missing=g['name'] and not g['number'] or g['number'] and not g['name'] and g['side'] in {'top','bottom'}
        merged=(g['number'] is not None and g['name'] is not None and
                (g['original_number']!=g['number'] or g['original_name']!=g['name']))
        if repeated or missing or merged:eligible.append(g)
    owners=[];selected=[]
    for g in sorted(eligible,key=lambda g:(g['owner'],g['id'])):
        if g['owner'] not in owners:
            if len(owners)>=cfg['max_components']:continue
            owners.append(g['owner'])
        if len(selected)<cfg['max_groups']:selected.append(deepcopy(g))
    return selected


def recognize(engine, rgb, box, scale, rotation, recognition_only, view):
    """Read one image patch; locations are always restored to the source image."""
    x1,y1,x2,y2=box;crop=rgb[y1:y2,x1:x2]
    rotated=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE) if rotation else crop
    patch=cv2.resize(rotated,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    result=engine(np.ascontiguousarray(patch[:,:,::-1]),use_det=not recognition_only,
                  use_cls=not recognition_only,use_rec=True)
    output=[]
    if recognition_only:
        for text,confidence in zip(result.txts or [],result.scores or []):
            output.append({'text':str(text),'confidence':float(confidence),'bbox':list(box),'view':view})
    elif result.boxes is not None:
        for polygon,text,confidence in zip(result.boxes,result.txts,result.scores):
            polygon=np.asarray(restore_polygon(np.asarray(polygon,dtype=float),box,scale,rotation))
            a=polygon.min(axis=0);b=polygon.max(axis=0)
            if a[0]<=x1+1 or a[1]<=y1+1 or b[0]>=x2-1 or b[1]>=y2-1:continue
            output.append({'text':str(text),'confidence':float(confidence),
                           'bbox':[float(a[0]),float(a[1]),float(b[0]),float(b[1])],'view':view})
    return output


def layout(owner, side, whole, document, target, threshold):
    edge=target['components'][owner]['bbox'][0 if side=='left' else 2]
    column=[(t['bbox'][0]+t['bbox'][2])/2 for t in whole if t['owner']==owner and t['text'].isdigit()
            and t['confidence']>=threshold and abs((t['bbox'][0]+t['bbox'][2])/2-edge)<30]
    offsets=[];support=set()
    for p in document['pin_candidates']:
        if p.get('owner')!=owner or not p.get('number') or p.get('number_source')=='library_pin_name':continue
        for t in whole:
            cy=(t['bbox'][1]+t['bbox'][3])/2
            if t['owner']==owner and t['text']==str(p['number']) and t['confidence']>=threshold and abs(cy-p['point'][1])<=16:
                offsets.append(cy-p['point'][1]);support.add((p['candidate_id'],t['view']))
        # Original observed glyphs and fresh crops can corroborate the same
        # printed number even when it is the component's only numbered pin.
        for t in p.get('text_candidates',{}).get('number',[]):
            if str(t.get('text'))!=str(p['number']) or t.get('confidence',0)<threshold or not t.get('source_view'):continue
            cy=(t['bbox'][1]+t['bbox'][3])/2
            if abs(cy-p['point'][1])<=16:
                offsets.append(cy-p['point'][1]);support.add((p['candidate_id'],t['source_view']))
    if not column or len(support)<2:return None
    return statistics.median(column),statistics.median(offsets)


def _vote(tokens, cfg):
    views=defaultdict(set);evidence=defaultdict(list)
    for t in tokens:
        if t['confidence']>=cfg['min_confidence']:
            views[t['text']].add(t['view']);evidence[t['text']].append(t)
    rank=sorted(views,key=lambda s:(-len(views[s]),s))
    support={k:len(v) for k,v in views.items()}
    if not rank or len(views[rank[0]])<cfg['min_views']:return None,{'reason':'insufficient_views','support':support}
    if len(rank)>1 and len(views[rank[1]])>=cfg['min_views']:return None,{'reason':'conflicting_spelling','support':support}
    return rank[0],{'reason':'accepted','support':support,'readings':evidence[rank[0]]}


def collect(rgb, groups, records, document, target, reader, cfg):
    height,width=rgb.shape[:2];texts=[];whole=[];calls=0
    by_owner={r['component_id']:r for r in records}
    def read(image,box,scale,rotation,recognition_only,view,owner,cid=None):
        nonlocal calls
        if calls>=cfg['max_ocr_calls']:return []
        calls+=1
        out=reader(image,box,scale,rotation,recognition_only,view)
        return [{**t,'owner':owner,'rotation':rotation,**({'target_candidate_id':cid} if cid else {})} for t in out]
    for owner in sorted({g['owner'] for g in groups}):
        box=by_owner.get(owner,{}).get('bbox_top_left')
        if not box:continue
        box=[max(0,math.floor(box[0]-32)),max(0,math.floor(box[1]-32)),
             min(width,math.ceil(box[2]+32)),min(height,math.ceil(box[3]+32))]
        for scale in cfg['component_scales']:
            for rotation in cfg['rotations']:
                whole.extend(read(rgb,box,scale,rotation,False,f'component_{owner}_s{scale}_r{rotation}',owner))
    texts.extend(whole)
    for g in groups:
        x,y=g['point'];side=g['side'];owner=g['owner'];box=g['box']
        if side in {'top','bottom'}:
            if g['name'] is not None:continue
            edge=box[1] if side=='top' else box[3]
            roi=[max(0,int(x-22)),max(0,int(edge-40)),min(width,int(x+22)+1),min(height,int(edge+40)+1)]
            for scale in cfg['scales']:
                for rotation in cfg['rotations']:
                    texts.extend(read(rgb,roi,scale,rotation,False,f'vertical_{g["id"]}_s{scale}_r{rotation}',owner))
            continue
        if g['number'] and not g['library_number']:continue
        learned=layout(owner,side,whole,document,target,cfg['min_confidence'])
        if learned is None:continue
        column,offset=learned;expected=y+offset
        x1=max(0,int(column-13));x2=min(width,int(column+14));y1=max(0,int(expected-13));y2=min(height,int(expected+14))
        patch=rgb[y1:y2,x1:x2].copy()
        red,green,blue=[patch[:,:,i].astype(int) for i in range(3)]
        neutral=(abs(red-green)<35)&(abs(red-blue)<35)&(abs(green-blue)<35)
        mask=(((red+green+blue)/3<205)&neutral).astype(np.uint8)
        mask=mask & (1-cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,13),np.uint8)))
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,count):
            a,b,w,h,area=map(int,stats[i]);cx,cy=centroids[i]
            if not(2<=w<=20 and 6<=h<=22 and area>=8):continue
            if abs(x1+cx-column)>8 or abs(y1+cy-expected)>9:continue
            if a==0 or b==0 or a+w>=patch.shape[1] or b+h>=patch.shape[0]:continue
            isolated=np.full((h+16,w+16,3),255,dtype=np.uint8)
            chosen=labels[b:b+h,a:a+w]==i
            isolated[8:8+h,8:8+w][chosen]=patch[b:b+h,a:a+w][chosen]
            source_box=[x1+a,y1+b,x1+a+w,y1+b+h]
            for scale in cfg['scales']:
                readings=read(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,
                              f'glyph_{g["id"]}_{i}_s{scale}_r0',owner,g['id'])
                texts.extend({**t,'bbox':source_box} for t in readings)
    return whole,texts,{'ocr_calls':calls,'budget_exhausted':calls>=cfg['max_ocr_calls']}


def bind(groups, all_groups, whole, texts, document, target, cfg):
    selected=deepcopy(groups)
    for g in selected:
        owner,side=g['owner'],g['side'];x,y=g['point'];box=g['box'];evidence=[]
        learned=layout(owner,side,whole,document,target,cfg['min_confidence']) if side in {'left','right'} else None
        for t in texts:
            if t['owner']!=owner or not t['text'].isdigit():continue
            if t.get('target_candidate_id') and t['target_candidate_id']!=g['id']:continue
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if side in {'left','right'}:
                if learned is None:continue
                column,offset=learned;distance=abs(cy-y-offset)
                if abs(cx-column)>12 or distance>8:continue
                rivals=[abs(cy-q['point'][1]-offset) for q in all_groups if q['owner']==owner and q['side']==side and q['id']!=g['id']]
            else:
                inside=box[0]<cx<box[2] and box[1]<cy<box[3];edge=box[1] if side=='top' else box[3]
                distance=abs(cx-x)
                if inside or distance>15 or abs(cy-edge)>45:continue
                rivals=[abs(cx-q['point'][0]) for q in all_groups if q['owner']==owner and q['side']==side and q['id']!=g['id']]
            if rivals and min(rivals)-distance<3:continue
            evidence.append(t)
        spellings={t['text'] for t in evidence if t['confidence']>=cfg['min_confidence']}
        if spellings=={'6','9'} and learned is not None:
            support=defaultdict(set);column,offset=learned
            for p in document['pin_candidates']:
                if p.get('owner')!=owner or not p.get('number') or p.get('number_source')=='library_pin_name':continue
                for t in whole:
                    cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                    if t['owner']==owner and t['text']==str(p['number']) and t['confidence']>=cfg['min_confidence'] and abs(cx-column)<=12 and abs(cy-p['point'][1]-offset)<=8:
                        support[t['rotation']].add((p['candidate_id'],t['view']))
            order=sorted(support,key=lambda r:(-len(support[r]),r))
            if order and len(support[order[0]])>=3 and (len(order)==1 or len(support[order[0]])>len(support[order[1]])):
                g['preferred_rotation']=order[0];evidence=[t for t in evidence if t['rotation']==order[0]]
        number,audit=_vote(evidence,cfg);g['number_audit']=audit
        limit=max(4,round(1.5*sum(q['owner']==owner for q in all_groups)))
        if number is not None and int(number)<=limit and (not g['number'] or g['library_number']):g['number']=number
        if not g['name'] and g['number'] and side in {'top','bottom'}:
            inner=[]
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2;edge=box[1] if side=='top' else box[3]
                if t['owner']==owner and t['text']==g['number'] and box[0]<cx<box[2] and box[1]<cy<box[3] and abs(cx-x)<=15 and abs(cy-edge)<=35:
                    rivals=[abs(cx-q['point'][0]) for q in all_groups if q['owner']==owner and q['side']==side and q['id']!=g['id']]
                    if not rivals or min(rivals)-abs(cx-x)>=3:inner.append(t)
            name,audit=_vote(inner,cfg);g['name_audit']=audit
            if name is not None:g['name']=name
    members={cid for g in selected for cid in g['members']}
    reserved={(p.get('owner'),str(p['number'])) for p in document['pin_candidates'] if p['candidate_id'] not in members and p.get('number') is not None}
    counts=defaultdict(int)
    for g in selected:
        if g['number'] is not None:counts[g['owner'],g['number']]+=1
    for g in selected:
        key=(g['owner'],g['number'])
        if g['number'] is not None and (key in reserved or counts[key]>1):g['withheld']='number_conflict'
        if g['library_number'] and g['original_number'] and g['number']!=g['original_number']:
            partner=next((q for q in selected if q['id']!=g['id'] and q['owner']==g['owner'] and q['number']==g['original_number'] and q['name'] and q.get('number_audit',{}).get('reason')=='accepted'),None)
            if partner is None:g['withheld']='incomplete_renumber_transaction'
    return selected


def sync_existing_networks(out, graph, changes):
    contacts={a['candidate_id']:a for a in graph.get('pin_attachments',[])};audit=[]
    for net in graph.get('networks',[]):
        nid=net['network_id'];relevant=[g for g in changes if g['id'] in net.get('pin_candidate_ids',[])]
        if not relevant:continue
        if nid not in out['nets']:
            audit.append({'network_id':nid,'status':'withheld_new_net','reason':'no_existing_exported_network'});continue
        refs={v.strip() for v in out['nets'][nid]['hyperGraph'].strip('()').split(',') if v.strip()};before=sorted(refs)
        for g in relevant:
            a=contacts.get(g['id'],{});ref=f"{g['owner']}.{g['number']}"
            if a.get('attachment_status')!='attached' or a.get('network_id')!=nid:continue
            if a.get('point')!=g['point']:continue
            if any(ref in {v.strip() for v in other['hyperGraph'].strip('()').split(',')} for other_id,other in out['nets'].items() if other_id!=nid):continue
            refs.add(ref)
        if sorted(refs)!=before:
            out['nets'][nid]['hyperGraph']='('+','.join(sorted(refs))+')'
            audit.append({'network_id':nid,'before':before,'after':sorted(refs)})
    return audit


def recover_pin_fields(target, document, graph, records, rgb, *, reader, position_corrector,
                       raw_pins, cfg=None):
    cfg={**DEFAULT,**(cfg or {})};height=rgb.shape[0]
    all_groups=build_groups(target,document,height);groups=select_groups(all_groups,cfg)
    audit={'enabled':True,'policy':'image_numeric_pin_recovery_v1','selected_groups':len(groups),
           'changes':[],'network_changes':[],'ocr_calls':0,'budget_exhausted':False}
    if not groups:return deepcopy(target),audit
    whole,texts,usage=collect(rgb,groups,records,document,target,reader,cfg);audit.update(usage)
    recovered=bind(groups,all_groups,whole,texts,document,target,cfg)
    audit['groups']=recovered
    complete=[g for g in recovered if not g.get('withheld') and g['number'] is not None and g['name'] is not None]
    changes=[g for g in complete if g['number']!=g['original_number'] or g['name']!=g['original_name']]
    contacts={a['candidate_id']:a for a in graph.get('pin_attachments',[])}
    out=deepcopy(target)
    for g in list(changes):
        c,key=g['owner'],'pin_'+g['number']
        # Replacing an existing key is allowed only as a complete observed
        # one-to-one transaction, never as an incidental OCR overwrite.
        if key in out['pins'].get(c,{}) and g['original_number']!=g['number']:
            partner=next((q for q in complete if q['owner']==c and q['original_number']==g['number'] and q['number']!=q['original_number']),None)
            if partner is None:g['withheld']='existing_key_not_in_transaction';changes.remove(g);continue
    # A library-derived reassignment must remain a complete transaction in
    # both the pin map and its existing physical network. Otherwise withhold
    # the whole pair instead of leaving a stale reference in another net.
    for g in list(changes):
        if not g['library_number'] or g['original_number'] is None or g['original_number']==g['number']:continue
        partner=next((q for q in changes if q['owner']==g['owner'] and q['number']==g['original_number'] and q['id']!=g['id']),None)
        a=contacts.get(g['id'],{});b=contacts.get(partner['id'],{}) if partner else {}
        if partner is None or a.get('attachment_status')!='attached' or b.get('attachment_status')!='attached' or a.get('network_id')!=b.get('network_id') or a.get('network_id') not in out['nets']:
            g['withheld']='unproven_renumber_network_transaction';changes.remove(g)
            if partner in changes:partner['withheld']='unproven_renumber_network_transaction';changes.remove(partner)
    snap={'components':deepcopy(target['components']),'pins':deepcopy(raw_pins),'nets':{}}
    for g in changes:
        c,key=g['owner'],'pin_'+g['number']
        x,y=g['point'];snap['pins'].setdefault(c,{})[key]={'pinname':g['name'],'point':{'x':float(x),'y':float(height-y)}}
    if changes:
        position_corrector(snap)
        for g in changes:
            c,key=g['owner'],'pin_'+g['number'];out['pins'].setdefault(c,{})[key]=deepcopy(snap['pins'][c][key])
        audit['network_changes']=sync_existing_networks(out,graph,changes)
    audit['changes']=changes
    return out,audit
