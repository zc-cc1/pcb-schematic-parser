"""Recover locally observed contacts omitted by multi-member graph export.

Use plate geometry and continuous local ink. Never union pixel components over
the entire page: undotted crossings must retain their existing network split.
"""
from copy import deepcopy
import math,re
import cv2
import numpy as np
from .local_symbol_contacts import _masks,_pin,_refs
from .oriented_port_contacts import _dot_centres
from .capacitor_plate_axis import capacitor_axis_proof
from .terminal_topology import connected_corridor
from .numeric_power_caption import outward_wire,NAMED_SUPPLIES
from .structural_patterns import StructuralDraft,NC,RAILS
from .wire_false_positive_cleanup import _coverage

DEFAULT={'enabled':False,'maximum_axis_corrections':8,'maximum_attachments':32,'maximum_new_edges':16}


def _color(rgb):
    c=rgb.astype(np.int16);r,g,b=[c[:,:,i] for i in range(3)]
    return (np.maximum(r,g)>b+30)&(np.maximum(r,g)>70)


def _projection(point,edge,height):
    a,b=[np.array([p['x'],height-p['y']],float) for p in (edge[0],edge[-1])]
    v=b-a;length=float(v@v)
    q=a+v*np.clip(float((point-a)@v)/length,0,1) if length else a
    return q,float(np.linalg.norm(q-point))


def local_attachment(out,ref,rgb,*,maximum=12,point=None,include_nearby=True):
    h,w=rgb.shape[:2];owner,number=ref.rsplit('.',1)
    pin=out['pins'][owner]['pin_'+number]
    if str(pin.get('pinname','')).strip().upper() in NC:return None
    p=pin['point'];point=np.array(point if point is not None else [p['x'],h-p['y']],float)
    color=_color(rgb);red,green,_=_masks(rgb)
    def visible(q):
        x,y=map(round,q)
        if not (9<=x<w-9 and 9<=y<h-9):return False
        arms=sum(green[y+dy-1:y+dy+2,x+dx-1:x+dx+2].any()
                 for dx,dy in [(-8,0),(8,0),(0,-8),(0,8)])
        if arms==4 and not any(math.dist(q,dot)<=5 for dot in _dot_centres(red)):return False
        return connected_corridor(color,point,q)
    nearby=set()
    for other,ps in out['pins'].items():
        if other==owner or not include_nearby:continue
        for key,p in ps.items():
            if str(p.get('pinname','')).strip().upper() in NC:continue
            q=[p['point']['x'],h-p['point']['y']]
            if math.dist(point,q)<=8 and visible(q):nearby.add(other+'.'+key[4:])
    candidates={}
    for nid,net in out['nets'].items():
        if _refs(net)&nearby:candidates[nid]=(0.,point)
        for edge in net['edges'].values():
            q,distance=_projection(point,edge,h)
            if distance<=maximum and visible(q) and (nid not in candidates or distance<candidates[nid][0]):
                candidates[nid]=(distance,q)
    order=sorted(candidates,key=lambda k:(candidates[k][0],k))
    if len(order)>1 and candidates[order[1]][0]-candidates[order[0]][0]<3:return None
    if order:
        nid=order[0];net=out['nets'][nid];net['hyperGraph']='('+','.join(sorted(_refs(net)|{ref}))+')'
        contact=candidates[nid][1].tolist()
    elif nearby:
        # A unique directly touching identity can form a no-wire contact net.
        if len(nearby)!=1:return None
        nid='OBSERVED_TERMINAL_'+owner+'_'+number
        while nid in out['nets']:nid+='x'
        out['nets'][nid]={'hyperGraph':'('+','.join(sorted(nearby|{ref}))+')','edges':{}}
        contact=point.tolist()
    else:return None
    return {'network_id':nid,'member':ref,'contact':contact,'local_distance':float(math.dist(point,contact))}


def _outside_bodies(out,owner,a,b,height,*,endpoint_margin=2):
    samples=np.linspace(a,b,max(2,math.ceil(math.dist(a,b)/2)+1))
    for p in samples[endpoint_margin:len(samples)-endpoint_margin]:
        for other,c in out['components'].items():
            if other==owner:continue
            l,bb,r,t=c['bbox']
            if l+1<p[0]<r-1 and height-t+1<p[1]<height-bb-1:return False
    return True


def _missing_named_supplies(out,rgb,records,pins,ocr):
    """Export a measured supply pin only together with its unique local net.

    The caption stage must have confirmed the value first. Source OCR still
    needs exclusive ownership; a supply name alone never establishes a wire.
    """
    from .pin_numeric_connector import observations
    from .component_caption_fields import _supply_caption_owned,_inside_other_body
    h,w=rgb.shape[:2];words=observations(ocr);events=[]
    color=_color(rgb);red,_,_=_masks(rgb);dots=_dot_centres(red)
    occupied=set().union(*(_refs(n) for n in out['nets'].values())) if out['nets'] else set()
    for record in records['records']:
        owner=record['component_id'];comp=out['components'].get(owner,{})
        if (record.get('type') not in {'vcc','vbus_in','vbus_out'} or record.get('detection_confidence',0)<.8
                or comp.get('type') not in RAILS-{'gnd'} or comp.get('value') not in NAMED_SUPPLIES
                or out['pins'].get(owner) or any(ref.rsplit('.',1)[0]==owner for ref in occupied)):continue
        a,b,c,d=record['bbox_top_left']
        nearby=[t for t in words if t['text'] in NAMED_SUPPLIES and t['confidence']>=.8
                and abs((t['bbox'][0]+t['bbox'][2])/2-(a+c)/2)<45
                and abs((t['bbox'][1]+t['bbox'][3])/2-(b+d)/2)<30]
        if (not nearby or {t['text'] for t in nearby}!={comp['value']}
                or any(not _supply_caption_owned(record,records['records'],t['bbox'])
                       or _inside_other_body(out,owner,t['bbox'],h) for t in nearby)):continue
        owned=[p for p in pins['pin_candidates'] if p.get('owner')==owner]
        if len(owned)!=1:continue
        source=owned[0];point=source['point']
        if (source.get('owner_status')!='assigned' or source.get('confidence',0)<.9
                or source.get('text_hypothesis_primary_candidate_id')
                or not all(math.isfinite(v) for v in point)
                or not (a-3<=point[0]<=c+3 and b-3<=point[1]<=d+3)):continue
        proof=outward_wire(rgb,record['bbox_top_left'],point)
        if proof is None:continue
        cap=list(point);cap[proof['axis']]=proof['cap_coordinate']
        if (not connected_corridor(color,point,cap)
                or not _outside_bodies(out,owner,cap,point,h,endpoint_margin=0)):continue
        trial=deepcopy(out);trial['pins'][owner]={'pin_1':_pin(point,h,'1')}
        event=local_attachment(trial,owner+'.1',rgb,maximum=12,include_nearby=False)
        if (event is None or event['network_id'] not in out['nets']
                or not out['nets'][event['network_id']]['edges']
                or not _outside_bodies(out,owner,point,event['contact'],h,endpoint_margin=0)):continue
        # Close to the symbol, one crossing arm may still be a red lead.
        # Counting only green arms would turn that undotted crossing into a net.
        x,y=map(round,event['contact'])
        if not (9<=x<w-9 and 9<=y<h-9):continue
        arms=sum(color[y+dy-1:y+dy+2,x+dx-1:x+dx+2].any()
                 for dx,dy in [(-8,0),(8,0),(0,-8),(0,8)])
        if arms==4 and not any(math.dist(event['contact'],dot)<=5 for dot in dots):continue
        refs=_refs(out['nets'][event['network_id']])
        conflicting=[out['components'].get(ref.rsplit('.',1)[0],{}) for ref in refs]
        if any(c.get('type') in RAILS and c.get('value') in NAMED_SUPPLIES
               and c['value']!=comp['value'] for c in conflicting):continue
        out['pins'][owner]=trial['pins'][owner]
        out['nets'][event['network_id']]['hyperGraph']=trial['nets'][event['network_id']]['hyperGraph']
        occupied.add(owner+'.1')
        events.append({'rule':'observed_named_supply_contact','owner':owner,
                       'source_candidate':source.get('candidate_id'),'source_point':list(point),
                       'caption':comp['value'],'caption_observations':nearby,'outward_wire':proof,**event})
    return events


def recover_terminal_memberships(target,rgb,graph,records,pins,ocr,image_sha256,*,cfg=None,wire_cfg=None):
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_terminal_membership_setting')
    if any(type(config[k]) is not int or not 0<=config[k]<=128 for k in DEFAULT if k!='enabled'):raise ValueError('invalid_terminal_membership_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    for document in (graph,records,pins,ocr):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('coordinate_origin')!='top_left' or document.get('prediction_only') is not True):
            raise ValueError('terminal_membership_evidence_differs_from_image')
    events=[];rmap={r['component_id']:r for r in records['records']}
    draft=StructuralDraft(out,rgb,records,pins,ocr,wire_cfg or {},{})
    color=_color(rgb);red,green,_=_masks(rgb);dots=_dot_centres(red)
    shapes={o:p for o,r in rmap.items() if o in out['components'] and (p:=capacitor_axis_proof(rgb,r))}
    corrections=0
    for owner,proof in shapes.items():
        old=out['pins'].get(owner,{})
        if set(old)!={'pin_1','pin_2'} or rmap[owner].get('detection_confidence',0)<.7:continue
        before=[draft.point(old['pin_'+str(i)]) for i in (1,2)];axis=proof['axis'];cross=1-axis
        if abs(before[1][axis]-before[0][axis])>=.25*abs(before[1][cross]-before[0][cross]):continue
        pair=draft.terminal_pair(owner,axis_override=axis)
        if pair is None:continue
        _,points=pair
        if max(abs(p[cross]-proof['center'][cross]) for p in points)>4:continue
        if any(not connected_corridor(color,p,bar['center']) for p,bar in zip(points,proof['bars'])):continue
        # Snap to a uniquely observed existing IC contact, never change the IC.
        for i,p in enumerate(points):
            nearby=[draft.point(pin) for other,ps in out['pins'].items() if out['components'][other]['type'] in {'box','ic','pin'}
                    for pin in ps.values() if math.dist(p,draft.point(pin))<=8 and connected_corridor(color,p,draft.point(pin))]
            if len(nearby)==1:points[i]=nearby[0]
        for net in out['nets'].values():net['hyperGraph']='('+','.join(sorted(r for r in _refs(net) if r.rsplit('.',1)[0]!=owner))+')'
        out['pins'][owner]={'pin_'+str(i):_pin(p,h,str(i)) for i,p in enumerate(points,1)};corrections+=1
        events.append({'rule':'observed_capacitor_plate_axis','owner':owner,'before':before,'terminals':points,**proof})
    supply_events=_missing_named_supplies(out,rgb,records,pins,ocr);events.extend(supply_events)
    occupied=set().union(*(_refs(n) for n in out['nets'].values())) if out['nets'] else set()
    pending=[o+'.'+key[4:] for o,ps in out['pins'].items() if out['components'][o]['type'] in RAILS|{'c','box','ic','pin'}
             for key in ps if o+'.'+key[4:] not in occupied and str(ps[key].get('pinname','')).strip().upper() not in NC]
    attachments=len(supply_events);new_edges=0
    for ref in sorted(pending):
        owner,number=ref.rsplit('.',1);comp=out['components'][owner];record=rmap.get(owner)
        if record is None:continue
        p=out['pins'][owner]['pin_'+number]['point'];point=np.array([p['x'],h-p['y']],float)
        event=local_attachment(out,ref,rgb,maximum=12)
        if event:
            attachments+=1;events.append({'rule':'local_unbound_terminal','owner':owner,**event});continue
        ray=None
        if comp['type']=='c' and owner in shapes:
            shape=shapes[owner];axis=shape['axis'];delta=point[axis]-shape['center'][axis]
            if abs(delta)>10:ray=(axis,1 if delta>0 else -1)
        elif comp['type'] in RAILS-{'gnd'}:
            path=outward_wire(rgb,record['bbox_top_left'],point)
            if path:ray=(path['axis'],path['direction'])
        if ray is None:continue
        axis,direction=ray;cross=1-axis
        targets=sorted((direction*(dot[axis]-point[axis]),dot) for dot in dots
                       if abs(dot[cross]-point[cross])<=3 and 8<=direction*(dot[axis]-point[axis])<=96)
        if not targets:continue
        _,dot=targets[0]
        if not connected_corridor(color,point,dot) or not _outside_bodies(out,owner,point,dot,h):continue
        samples=np.rint(np.linspace(point,dot,max(2,math.ceil(math.dist(point,dot))+1))).astype(int)
        green_points=[np.array([x,y],float) for x,y in samples if green[y-1:y+2,x-1:x+2].any()]
        if len(green_points)<(5 if comp['type'] in RAILS else 8):continue
        event=local_attachment(out,ref,rgb,maximum=5,point=dot)
        if event is None:continue
        attachments+=1;event.update(rule='observed_axial_terminal_to_junction',owner=owner,source_point=point.tolist(),junction=dot.tolist(),green_samples=len(green_points))
        net=out['nets'][event['network_id']]
        start=np.rint(green_points[0]);end=np.rint(dot)
        edge=[{'x':float(q[0]),'y':float(h-q[1])} for q in [start,end]]
        if math.dist(start,end)>=8 and _coverage(edge,list(net['edges'].values()))<.95:
            index=1+max((int(k[5:]) for k in net['edges'] if k.startswith('edge_') and k[5:].isdigit()),default=0)
            key='edge_'+str(index)
            while key in net['edges']:key+='x'
            net['edges'][key]=edge;new_edges+=1;event['added_edge']={'edge_id':key,'edge':edge}
        events.append(event)
    if corrections>config['maximum_axis_corrections'] or attachments>config['maximum_attachments'] or new_edges>config['maximum_new_edges']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[]}
    for owner,c in target['components'].items():
        if out['components'][owner]!=c:raise ValueError('terminal_membership_changed_component')
        if c['type'] in {'box','ic','switch','pin'} and out['pins'].get(owner)!=target['pins'].get(owner):raise ValueError('terminal_membership_changed_protected_pin')
    for nid,net in target['nets'].items():
        for key,edge in net['edges'].items():
            if out['nets'][nid]['edges'].get(key)!=edge:raise ValueError('terminal_membership_changed_existing_wire')
    return out,{'enabled':True,'status':'applied' if out!=target else 'no_effect','changes':events,
                'axis_corrections':corrections,'attachments':attachments,'new_edges':new_edges,'reference_read':False}
