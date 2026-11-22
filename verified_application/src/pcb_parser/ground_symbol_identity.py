"""Separate overlapping physical ground combs sharing one source detection.

Multiple pins alone do not prove multiple symbols: each row needs a distinct
short tooth and a jointly explained long/middle tooth profile plus its lead.
"""
from copy import deepcopy
import math
import numpy as np
from .local_symbol_contacts import _masks,_ground_pair_proof,_refs

DEFAULT={'enabled':False,'maximum_new_symbols':8}


def _runs(values):
    groups=[]
    for value in values:
        if not groups or value-groups[-1][-1]>1:groups.append([])
        groups[-1].append(int(value))
    return groups


def ground_comb_group(rgb,record,points):
    if (record.get('type')!='gnd' or record.get('detection_confidence',0)<.7
            or not 2<=len(points)<=4):return None
    ordered=sorted([list(map(float,p)) for p in points],key=lambda p:p[1])
    if max(p[0] for p in ordered)-min(p[0] for p in ordered)>3:return None
    if any(not 12<=q[1]-p[1]<=60 for p,q in zip(ordered,ordered[1:])):return None
    pairs=[_ground_pair_proof(rgb,record,p) for p in ordered]
    if any(p is None for p in pairs):return None
    columns=np.mean([p['bar_columns'] for p in pairs],axis=0)
    if any(max(abs(a-b) for a,b in zip(p['bar_columns'],columns))>1 for p in pairs):return None
    red,_,_=_masks(rgb);h,w=red.shape;a,b,c,d=record['bbox_top_left']
    top,bottom=max(0,math.floor(b)-12),min(h,math.ceil(d)+13)
    row_masks=[];halves=[]
    for column in columns:
        x=round(column)
        rows=red[top:bottom,max(0,x-1):min(w,x+2)].any(1)
        positions=np.flatnonzero(rows)+top
        if not len(positions):return None
        first=ordered[0][1]-positions[0];last=positions[-1]-ordered[-1][1]
        if min(first,last)<3 or abs(first-last)>3:return None
        half=float((first+last)/2);halves.append(half);row_masks.append(rows)
        expected=np.zeros(bottom-top,bool)
        for _,y in ordered:
            lo=max(top,round(y-half));hi=min(bottom,round(y+half)+1)
            expected[lo-top:hi-top]=True
        if np.count_nonzero(rows^expected)>4:return None
    if not (12<=halves[0]<=32 and 7<=halves[1]<=22 and 3<=halves[2]<=9
            and all(p-q>=4 for p,q in zip(halves,halves[1:]))):return None
    shortest=_runs(np.flatnonzero(row_masks[-1])+top)
    if len(shortest)!=len(ordered):return None
    if any(not 5<=len(run)<=19 or abs(float(np.mean(run))-p[1])>2 for run,p in zip(shortest,ordered)):return None
    symbols=[{'point':point,'body':[float(columns[0]-2),float(point[1]-halves[0]),
                                    float(columns[-1]+2),float(point[1]+halves[0])],
              'short_tooth_interval':[run[0],run[-1]]} for point,run in zip(ordered,shortest)]
    return {'source_record':record['component_id'],'source_confidence':record['detection_confidence'],
            'bar_columns':columns.tolist(),'bar_half_lengths':halves,'symbols':symbols}


def split_ground_identities(target,rgb,graph,records,image_sha256,*,cfg=None):
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_ground_identity_setting')
    if type(config['maximum_new_symbols']) is not int or not 0<=config['maximum_new_symbols']<=32:raise ValueError('invalid_ground_identity_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    for document in (graph,records):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('coordinate_origin')!='top_left' or document.get('prediction_only') is not True):
            raise ValueError('ground_identity_evidence_differs_from_image')
    events=[];added=0
    for record in records['records']:
        owner=record['component_id'];comp=out['components'].get(owner,{})
        old=out['pins'].get(owner,{})
        if comp.get('type')!='gnd' or comp.get('value')!='GND' or not 2<=len(old)<=4:continue
        contacts={key:[p['point']['x'],h-p['point']['y']] for key,p in old.items()}
        if len({tuple(p) for p in contacts.values()})!=len(contacts):continue
        proof=ground_comb_group(rgb,record,list(contacts.values()))
        if proof is None:continue
        plans=[];aliases={};blocked=False
        for index,symbol in enumerate(proof['symbols'],1):
            key=min(contacts,key=lambda k:math.dist(contacts[k],symbol['point']))
            dest=owner if index==1 else owner+'_symbol_'+str(index)
            a,b,c,d=symbol['body'];new_comp={'type':'gnd','Name':'GND','value':'GND','bbox':[a,h-d,c,h-b]}
            new_pin=deepcopy(old[key])
            if dest!=owner and dest in out['components']:
                existing=out['components'][dest];pins=out['pins'].get(dest,{})
                if (existing.get('type')!='gnd' or existing.get('value')!='GND' or set(pins)!={'pin_1'}
                        or math.dist([pins['pin_1']['point']['x'],h-pins['pin_1']['point']['y']],symbol['point'])>1
                        or max(abs(x-y) for x,y in zip(existing['bbox'],new_comp['bbox']))>2):blocked=True;break
            # Another physical identity at this contact would make a new child
            # duplicate it. A separate adjudication is needed in that case.
            if any(other not in {owner,dest} and c.get('type')=='gnd'
                   and any(math.dist([p['point']['x'],h-p['point']['y']],symbol['point'])<=2 for p in out['pins'].get(other,{}).values())
                   for other,c in out['components'].items()):blocked=True;break
            plans.append((dest,new_comp,new_pin));aliases[owner+'.'+key[4:]]=dest+'.1'
        if blocked:continue
        for net in out['nets'].values():
            refs={aliases.get(ref,ref) for ref in _refs(net)}
            net['hyperGraph']='('+','.join(sorted(refs))+')'
        for dest,component,pin in plans:
            if dest not in out['components']:added+=1
            out['components'][dest]=component;out['pins'][dest]={'pin_1':pin}
        events.append({'rule':'separate_observed_ground_combs','owner':owner,'aliases':aliases,
                       'identities':[p[0] for p in plans],**proof})
    if added>config['maximum_new_symbols']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[]}
    for owner,comp in target['components'].items():
        if comp['type']!='gnd' and (out['components'][owner]!=comp or out['pins'].get(owner)!=target['pins'].get(owner)):
            raise ValueError('ground_identity_changed_unrelated_component')
    for nid,net in target['nets'].items():
        if out['nets'][nid]['edges']!=net['edges']:raise ValueError('ground_identity_changed_wire_geometry')
    for net in out['nets'].values():
        for ref in _refs(net):
            owner,number=ref.rsplit('.',1)
            if 'pin_'+number not in out['pins'].get(owner,{}):raise ValueError('ground_identity_dangling_member')
    return out,{'enabled':True,'status':'applied' if out!=target else 'no_effect','changes':events,
                'new_symbols':added,'reference_read':False}
