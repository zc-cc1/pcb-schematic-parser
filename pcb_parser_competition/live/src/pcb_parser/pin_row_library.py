"""Observed horizontal row reconciliation and exact-model library completion."""
from collections import defaultdict
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES,LeadEvidence,_members,_segment_distance,side_of,vote
from .pin_nc_recovery import crossed_lead
from .pin_vertical_repair import attached

DEFAULT={'min_confidence':.98,'min_views':2,'max_components':4,'max_ocr_calls':64,
         'max_extra_calls_per_component':12,'max_model_distance_px':32}


def norm(text):return ''.join(c.casefold() for c in str(text) if c.isalnum())


def strict_vote(tokens,cfg):
    value,support=vote(tokens,cfg)
    if value is not None and any(t['confidence']>=cfg['min_confidence'] and t['text']!=value for t in tokens):return None,support
    return value,support


def pin_table(record):
    """Preserve ambiguity in the source, rather than choosing the first alias."""
    numbers=defaultdict(set)
    for p in record.get('pins',[]):
        number=str(p.get('number') or '').strip();name=str(p.get('name') or '').strip()
        if number and name:numbers[number].add(name)
    return dict(numbers)


def field_tokens(whole,lead,frame,field):
    selected=[];left=lead['side']=='left';row=lead['row']
    for t in whole:
        cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
        inward=cx-frame[0] if left else frame[2]-cx
        outward=frame[0]-cx if left else cx-frame[2]
        if field=='name':valid=0<inward<.5*(frame[2]-frame[0]) and abs(cy-row)<=12 and any(c.isalpha() for c in t['text'])
        else:valid=0<=outward<=52 and -16<=cy-row<=2 and t['text'].isalnum() and any(c.isdigit() for c in t['text'])
        if valid:selected.append(t)
    return selected


def prepare_text(rgb,tokens,row,grouped):
    selected=[t for t in tokens if t.get('confidence',0)>=.75]
    if not selected:return None
    bounds=[statistics.median(t['bbox'][i] for t in selected) for i in range(4)]
    box=[math.floor(bounds[0]-2),math.floor(bounds[1]-2),math.ceil(bounds[2]+2),math.ceil(bounds[3]+2)]
    a,b,c,d=box
    if min(a,b)<0 or c>rgb.shape[1] or d>rgb.shape[0] or c<=a or d<=b:return None
    patch=rgb[b:d,a:c];v=patch.astype(np.int16)
    mask=((v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<230)).astype(np.uint8)
    if grouped:
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);accepted=[]
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i]);cx,cy=centroids[i]
            if 2<=h<=22 and area>=3 and abs(b+cy-row)<=10:
                if x==0 or y==0 or x+w==mask.shape[1] or y+h==mask.shape[0]:return None
                accepted.append(i)
        if not accepted:return None
        chosen=np.isin(labels,accepted);ys,xs=np.where(chosen)
        x1,y1,x2,y2=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
        if not 6<=y2-y1<=22 or not 2<=x2-x1<=64:return None
        isolated=np.full_like(patch,255);isolated[chosen]=patch[chosen]
        isolated=isolated[y1:y2,x1:x2];pad=4
    else:
        isolated=np.full_like(patch,255);isolated[mask.astype(bool)]=patch[mask.astype(bool)];pad=8
    return cv2.copyMakeBorder(isolated,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255)),box


def existing_route(owner,lead,target,document,graph,evidence):
    """NC marks and floating contacts never acquire network membership."""
    if crossed_lead(evidence,lead):return None,'visible_no_connect_cross'
    x,y=map(round,lead['point']);h,w=evidence.green.shape
    if not evidence.green[max(0,y-8):min(h,y+9),max(0,x-8):min(w,x+9)].any():return None,'no_visible_green_contact'
    edge_map={e['edge_id']:e for e in graph['edges']};nets={n['network_id']:n for n in graph['networks']};choices=set()
    for p in document['pin_candidates']:
        if p.get('owner')!=owner or p.get('owner_status')!='assigned' or abs(p['point'][1]-lead['row'])>6:continue
        outward=lead['edge']-p['point'][0] if lead['side']=='left' else p['point'][0]-lead['edge']
        if not 0<=outward<=lead['length']+8:continue
        a=attached(p,graph)
        if not a or a['network_id'] not in target['nets']:continue
        source=nets[a['network_id']]
        if source.get('unresolved_junction_ids'):continue
        distances=[]
        for eid in source.get('edge_ids',[]):
            pts=edge_map.get(eid,{}).get('polyline',[])
            distances.extend(_segment_distance(lead['point'],a,b) for a,b in zip(pts,pts[1:]))
        if distances and min(distances)<=8:choices.add(source['network_id'])
    if len(choices)!=1:return None,'no_unique_existing_observed_route'
    return next(iter(choices)),None


def reconcile_rows_with_library(target,document,graph,records,library,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};out=deepcopy(target);height,width=rgb.shape[:2]
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT);by_owner={r['component_id']:r for r in records}
    index=defaultdict(list)
    for model,record in library.items():
        table=pin_table(record)
        if len(table)>=3 and len(norm(model))>=4:index[norm(model)].append((model,table))
    audit={'enabled':True,'policy':'observed_rows_exact_library_v1','groups':[],'changes':[],
           'component_changes':[],'network_changes':[],'network_withheld':[],
           'ocr_calls':0,'budget_exhausted':False}
    def read(image,box,scale,rotation,rec,view):
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,rec,view)
    for owner,component in sorted(target['components'].items()):
        if (component['type'] not in IC_TYPES or owner not in by_owner
                or component.get('Name') not in [None,'',owner]):continue
        a,b,c,d=component['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if not found:continue
        frame=found[0];leads=evidence.leads(frame,'left')+evidence.leads(frame,'right')
        if not 3<=len(leads)<=16:continue
        if len(audit['groups'])>=cfg['max_components'] or audit['ocr_calls']+4+cfg['max_extra_calls_per_component']>cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;break
        original_box=by_owner[owner]['bbox_top_left']
        crop=[max(0,math.floor(original_box[0]-32)),max(0,math.floor(original_box[1]-32)),min(width,math.ceil(original_box[2]+32)),min(height,math.ceil(original_box[3]+32))]
        whole=[]
        for scale in [1.,3.]:
            for rotation in [0,90]:whole.extend(read(rgb,crop,scale,rotation,False,f'row_component_{owner}_s{scale}_r{rotation}'))
        model_tokens=[]
        for t in whole:
            key=norm(t['text']);cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if key==norm(owner) or key not in index or frame[0]<cx<frame[2] and frame[1]<cy<frame[3]:continue
            dx=max(frame[0]-cx,0,cx-frame[2]);dy=max(frame[1]-cy,0,cy-frame[3])
            if math.hypot(dx,dy)>cfg['max_model_distance_px']:continue
            for model,table in index[key]:
                if len(table)==len(leads):model_tokens.append({**t,'text':model})
        model,model_support=vote(model_tokens,cfg)
        if model is None:continue
        table=pin_table(library[model]);by_name=defaultdict(set)
        for number,names in table.items():
            for name in names:by_name[name].add(number)
        group={'owner':owner,'model':model,'model_support':model_support,'model_readings':model_tokens,'rows':[]}
        audit['groups'].append(group);extra=0;budget_failed=False
        def confirm(tokens,lead,field):
            nonlocal extra,budget_failed
            original=list(tokens);fresh=[]
            value,support=strict_vote(tokens,cfg)
            if value is not None:return value,support,original
            for grouped in [False,True]:
                prepared=prepare_text(rgb,original,lead['row'],grouped)
                if not prepared:continue
                if extra+2>cfg['max_extra_calls_per_component'] or audit['ocr_calls']+2>cfg['max_ocr_calls']:
                    audit['budget_exhausted']=True;budget_failed=True;return None,support,original+fresh
                image,box=prepared;extra+=2
                for scale in [3.,5.]:fresh.extend(read(image,[0,0,image.shape[1],image.shape[0]],scale,0,True,f"row_confirm_{owner}_{lead['side']}_{lead['row']}_{field}_{grouped}_s{scale}"))
                value,support=strict_vote(original+fresh,cfg)
                if value is not None:return value,support,original+fresh
            return None,support,original+fresh
        old=target['pins'].get(owner,{});rows=[];valid=True
        for lead in leads:
            current=[(key,pin) for key,pin in old.items() if side_of([pin['point']['x'],height-pin['point']['y']],frame)==lead['side']
                     and abs(height-pin['point']['y']-lead['row'])<=10]
            if len(current)>1:valid=False;break
            names=field_tokens(whole,lead,frame,'name');numbers=field_tokens(whole,lead,frame,'number')
            row={**lead,'old_key':current[0][0] if current else None,'old_record':deepcopy(current[0][1]) if current else None}
            if current:
                key,pin=current[0];number=key[4:];name=pin['pinname']
                if number in table and name in table[number]:
                    row.update(number=number,name=name,method='preserved_existing_record');rows.append(row);continue
                # An existing strong observed number is never overruled by a library.
                source=[p for p in document['pin_candidates'] if p.get('owner')==owner and str(p.get('number'))==number and p.get('pinname')==name]
                if len(source)!=1:valid=False;break
                source_votes=[{**t,'view':t.get('source_view')} for t in source[0].get('text_candidates',{}).get('number',[]) if t.get('source_view')]
                if any(t['text']==number and t['confidence']>=cfg['min_confidence'] for t in source_votes):valid=False;break
                observed_name,ns,nread=confirm(names,lead,'name')
                observed_number,ds,dread=confirm(numbers,lead,'number')
                if (observed_name!=name or observed_number is None or observed_number not in table
                        or name not in table[observed_number] or len(by_name[name])!=1
                        or 'pin_'+observed_number in old):valid=False;break
                memberships=[nid for nid,net in target['nets'].items() if owner+'.'+number in _members(net)]
                attachment=attached(source[0],graph)
                if len(memberships)>1 or memberships and (not attachment or memberships!=[attachment['network_id']]):valid=False;break
                row.update(number=observed_number,name=name,method='fresh_observed_rekey',number_support=ds,name_support=ns,
                           number_readings=dread,name_readings=nread,old_networks=memberships)
            else:
                candidates=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                            and side_of(p['point'],frame)==lead['side'] and abs(p['point'][1]-lead['row'])<=6
                            and 0<=(lead['edge']-p['point'][0] if lead['side']=='left' else p['point'][0]-lead['edge'])<=lead['length']+8]
                if not candidates:valid=False;break
                name,ns,nread=confirm(names,lead,'name')
                number,ds,dread=confirm(numbers,lead,'number') if numbers else (None,{},[])
                if name and number:
                    if number not in table or name not in table[number]:valid=False;break
                    method='fresh_observed_number_and_name'
                elif name and not numbers and len(by_name[name])==1:
                    number=next(iter(by_name[name]));method='exact_model_unique_name'
                else:method='unresolved'
                row.update(number=number,name=name,method=method,number_support=ds,name_support=ns,number_readings=dread,name_readings=nread)
            rows.append(row)
        group['rows']=rows
        if not valid or budget_failed or {r['old_key'] for r in rows if r['old_key']}!=set(old):
            group['status']='conflicting_or_incomplete_rows';continue
        # A fully unnumbered component can infer exactly one unprinted contact
        # only after every other physical lead has a unique observed name.
        unresolved=[r for r in rows if r['number'] is None or r['name'] is None]
        if unresolved:
            complete=[r for r in rows if r not in unresolved]
            remaining=set(table)-{r['number'] for r in complete}
            if (old or len(unresolved)!=1 or len(remaining)!=1 or len(complete)<4
                    or any(r['method']!='exact_model_unique_name' for r in complete)
                    or unresolved[0]['name_readings'] or unresolved[0]['number_readings']):
                group['status']='insufficient_observed_rows';continue
            missing=unresolved[0];number=next(iter(remaining))
            if len(table[number])!=1:group['status']='ambiguous_remaining_library_name';continue
            same=[r for r in rows if r['side']==missing['side']]
            other=[r for r in rows if r['side']!=missing['side']]
            if len(same)!=1 or len(other)!=len(complete):group['status']='unproven_single_contact_layout';continue
            missing.update(number=number,name=next(iter(table[number])),method='unique_remaining_library_contact')
        if len({r['number'] for r in rows})!=len(rows) or {r['number'] for r in rows}!=set(table):
            group['status']='nonunique_or_incomplete_library_mapping';continue
        changes=[r for r in rows if r['method']!='preserved_existing_record']
        if not changes:group['status']='verified_without_changes';continue
        planned=deepcopy(out);net_changes=[];withheld=[];pin_changes=[]
        for row in changes:
            number=row['number'];key='pin_'+number;oldkey=row['old_key']
            if key in planned['pins'].get(owner,{}) and key!=oldkey:valid=False;break
            if oldkey:
                pin=planned['pins'][owner].pop(oldkey);oldref=owner+'.'+oldkey[4:];newref=owner+'.'+number
                for nid,net in planned['nets'].items():
                    before=sorted(_members(net))
                    if oldref in before:
                        after=sorted((set(before)-{oldref})|{newref});net['hyperGraph']='('+','.join(after)+')'
                        net_changes.append({'network_id':nid,'before':before,'after':after,'reason':'observed_rekey'})
                if not row['old_networks']:withheld.append({'member':newref,'reason':'source_model_route_not_exported'})
                planned['pins'][owner][key]=pin
            else:
                ref=owner+'.'+number
                if any(ref in _members(n) for n in planned['nets'].values()):valid=False;break
                x,y=row['point'];pin={'pinname':row['name'],'point':{'x':float(x),'y':float(height-y)}}
                planned['pins'].setdefault(owner,{})[key]=pin
                nid,reason=existing_route(owner,row,planned,document,graph,evidence)
                if nid:
                    before=sorted(_members(planned['nets'][nid]));after=sorted(set(before)|{ref})
                    planned['nets'][nid]['hyperGraph']='('+','.join(after)+')'
                    net_changes.append({'network_id':nid,'before':before,'after':after,'reason':'observed_existing_attachment'})
                else:withheld.append({'member':ref,'reason':reason})
            pin_changes.append({'owner':owner,'number':number,'old_key':oldkey,'method':row['method'],
                                'before':row['old_record'],'after':deepcopy(planned['pins'][owner][key])})
        if not valid:group['status']='reserved_export_identity';continue
        old_name=planned['components'][owner].get('Name');planned['components'][owner]['Name']=model
        out=planned;group['status']='complete_observed_library_reconciliation'
        audit['changes'].extend(pin_changes);audit['network_changes'].extend(net_changes);audit['network_withheld'].extend(withheld)
        if old_name!=model:audit['component_changes'].append({'owner':owner,'field':'Name','before':old_name,'after':model})
    consolidated={}
    for change in audit['network_changes']:
        nid=change['network_id']
        if nid not in consolidated:consolidated[nid]={**deepcopy(change),'steps':[deepcopy(change)]}
        else:
            combined=consolidated[nid]
            if combined['after']!=change['before']:raise ValueError('Network audit chain differs')
            combined['after']=change['after'];combined['steps'].append(deepcopy(change))
    audit['network_changes']=list(consolidated.values())
    return out,audit
