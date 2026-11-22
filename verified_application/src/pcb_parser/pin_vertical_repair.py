"""Reconcile duplicated vertical identities using measured leads and OCR."""
from collections import Counter
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES,LeadEvidence,_members,_segment_distance,side_of,vote

DEFAULT={'min_confidence':.98,'min_views':2,'max_components':2,'max_ocr_calls':48,
         'min_lead_px':24,'max_lead_px':60,'max_length_spread_px':4}


def vertical_leads(evidence,frame,side,cfg):
    edge=frame[1] if side=='top' else frame[3];sign=-1 if side=='top' else 1;columns=[]
    for x in range(math.ceil(frame[0])+6,math.floor(frame[2])-5):
        if sum(evidence.hit(x,edge+sign*d) for d in range(3,11))<6:continue
        last=gap=0
        for d in range(1,66):
            if evidence.hit(x,edge+sign*d):last=d;gap=0
            else:gap+=1
            if gap>2:break
        if cfg['min_lead_px']<=last<cfg['max_lead_px']:columns.append((x,last))
    clusters=[]
    for item in columns:
        if not clusters or item[0]-clusters[-1][-1][0]>1:clusters.append([])
        clusters[-1].append(item)
    return [{'side':side,'edge':edge,'point':[statistics.median(p[0] for p in g),edge+sign*statistics.median(p[1] for p in g)],
             'length':statistics.median(p[1] for p in g)} for g in clusters if len(g)<=7]


def neutral_patch(rgb,box):
    a,b,c,d=box
    if min(a,b)<0 or c>rgb.shape[1] or d>rgb.shape[0] or c<=a or d<=b:return None
    patch=rgb[b:d,a:c];v=patch.astype(np.int16)
    chosen=(v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<230)
    isolated=np.full_like(patch,255);isolated[chosen]=patch[chosen]
    return cv2.copyMakeBorder(isolated,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))


def attached(candidate,graph):
    matches=[a for a in graph.get('pin_attachments',[]) if a['candidate_id']==candidate['candidate_id']]
    if len(matches)!=1:return None
    a=matches[0];net=next((n for n in graph['networks'] if n['network_id']==a.get('network_id')),None)
    if (a.get('attachment_status')!='attached' or a.get('point')!=candidate['point'] or net is None
            or candidate['candidate_id'] not in net.get('pin_candidate_ids',[])):return None
    return a


def source_texts(document,owner,lead,frame,field,cfg):
    tokens=[];x=lead['point'][0];sign=-1 if lead['side']=='top' else 1;mid=(frame[1]+frame[3])/2
    for p in document['pin_candidates']:
        if p.get('owner')!=owner:continue
        for t in p.get('text_candidates',{}).get(field,[]):
            if (t.get('confidence',0)<cfg['min_confidence'] or not t.get('source_view')
                    or t.get('approximate_split_geometry')):continue
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if field=='number':valid=t['text'].isdigit() and 0<sign*(cy-lead['edge'])<24 and abs(cx-x)<15
            else:valid=(any(c.isalpha() for c in t['text']) and frame[0]<cx<frame[2] and frame[1]<cy<frame[3]
                        and abs(cx-x)<12 and (cy<mid if lead['side']=='top' else cy>mid))
            if valid:tokens.append({**t,'view':t['source_view']})
    return tokens


def lead_route(owner,lead,frame,document,graph,evidence):
    """Only accept an attached model route that reaches the measured endpoint."""
    edge_map={e['edge_id']:e for e in graph['edges']};nets={n['network_id']:n for n in graph['networks']}
    sign=-1 if lead['side']=='top' else 1;possible=[]
    for p in document['pin_candidates']:
        if p.get('owner')!=owner or p.get('owner_status')!='assigned':continue
        if (abs(p['point'][0]-lead['point'][0])>4 or side_of(p['point'],frame)!=lead['side']
                or not 1<sign*(p['point'][1]-lead['edge'])<=lead['length']+6):continue
        a=attached(p,graph)
        if not a:continue
        source=nets[a['network_id']];segments=[]
        for eid in source['edge_ids']:
            points=edge_map.get(eid,{}).get('polyline',[]);segments.extend(zip(points,points[1:]))
        if not segments or min(_segment_distance(lead['point'],a,b) for a,b in segments)>3:continue
        possible.append((p,a))
    if len({a['network_id'] for p,a in possible})!=1:return None
    # The visible continuation outside the measured red lead must be green.
    x,y=lead['point'];h,w=evidence.green.shape;hits=0
    for d in range(3,11):
        xx,yy=round(x),round(y+sign*d)
        if 1<=xx<w-1 and 1<=yy<h-1:hits+=bool(evidence.green[yy-1:yy+2,xx-1:xx+2].any())
    if hits<6:return None
    return {'network_id':possible[0][1]['network_id'],'candidate_ids':sorted(p['candidate_id'] for p,a in possible)}


def restore_direct_ground_branch(owner,row,out,document,graph,evidence):
    """Copy one observed two-contact ground path; never synthesize a wire."""
    source=next(n for n in graph['networks'] if n['network_id']==row['route']['network_id'])
    if source.get('unresolved_junction_ids') or len(source.get('physical_network_ids',[]))!=1:return None
    physical=next((n for n in graph.get('physical_networks',[]) if n['network_id']==source['physical_network_ids'][0]),None)
    if (physical is None or physical.get('unresolved_junction_ids') or len(physical.get('edge_ids',[]))!=1
            or set(physical['edge_ids'])!=set(source['edge_ids']) or len(physical.get('pin_candidate_ids',[]))!=2
            or set(physical['pin_candidate_ids'])!=set(source['pin_candidate_ids'])):return None
    pmap={p['candidate_id']:p for p in document['pin_candidates']};points=[];refs=[];pending=[]
    for cid in physical['pin_candidate_ids']:
        p=pmap[cid];a=attached(p,graph)
        if not a or a.get('physical_network_id')!=physical['network_id'] or a['network_id']!=source['network_id']:return None
        if p.get('owner')==owner:
            if cid not in row['route']['candidate_ids'] or math.dist(p['point'],row['point'])>6:return None
            refs.append(owner+'.'+row['number'])
        else:
            comp=out['components'].get(p.get('owner'),{})
            if (comp.get('type')!='gnd' or str(comp.get('value','')).upper()!='GND' or p.get('number') is None
                    or p.get('owner_status')!='assigned'):return None
            ref=p['owner']+'.'+str(p['number'])
            if any(ref in _members(net) for net in out['nets'].values()):return None
            key='pin_'+str(p['number']);old=out['pins'].get(p['owner'],{}).get(key)
            point={'x':float(p['point'][0]),'y':float(evidence.height-p['point'][1])}
            if old is not None and math.dist([old['point']['x'],evidence.height-old['point']['y']],p['point'])>3:return None
            if old is None:pending.append((p['owner'],key,{'pinname':p.get('pinname') or '', 'point':point}))
            refs.append(ref)
        points.append(p['point'])
    if len(set(refs))!=2 or sum(ref.startswith(owner+'.') for ref in refs)!=1:return None
    edge=next((e for e in graph['edges'] if e['edge_id']==physical['edge_ids'][0]),None)
    if not edge or len(edge.get('polyline',[]))<2:return None
    poly=edge['polyline'];endpoints=[poly[0],poly[-1]]
    if min(max(math.dist(points[i],endpoints[i]) for i in range(2)),max(math.dist(points[i],endpoints[1-i]) for i in range(2)))>3:return None
    hits=[];mask=evidence.green;h,w=mask.shape
    for a,b in zip(poly,poly[1:]):
        for t in np.linspace(0,1,max(2,math.ceil(math.dist(a,b)))):
            x,y=round(a[0]+t*(b[0]-a[0])),round(a[1]+t*(b[1]-a[1]))
            hits.append(1<=x<w-1 and 1<=y<h-1 and bool(mask[y-1:y+2,x-1:x+2].any()))
    if not hits or sum(hits)/len(hits)<.9:return None
    nid='RESTORED_'+physical['network_id']
    if nid in out['nets']:return None
    edges={f'edge_{i+1}':[{'x':float(a[0]),'y':float(evidence.height-a[1])},
                           {'x':float(b[0]),'y':float(evidence.height-b[1])}] for i,(a,b) in enumerate(zip(poly,poly[1:]))}
    return {'network_id':nid,'net':{'hyperGraph':'('+','.join(sorted(refs))+')','edges':edges},
            'pending_pins':pending,'source_network_id':source['network_id'],'source_edge_ids':physical['edge_ids']}


def repair_vertical_triplet(target,document,graph,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};out=deepcopy(target);evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT);height,width=rgb.shape[:2]
    audit={'enabled':True,'policy':'observed_vertical_triplet_v1','groups':[],'changes':[],
           'network_changes':[],'restored_networks':[],'ocr_calls':0,'budget_exhausted':False}
    def read(image,box,scale,view):
        audit['ocr_calls']+=1
        return reader(image,box,scale,0,True,view)
    for owner,component in sorted(target['components'].items()):
        old=target['pins'].get(owner,{})
        if component['type'] not in IC_TYPES or len(old)!=3:continue
        owned=[p for p in document['pin_candidates'] if p.get('owner')==owner]
        if not any(p.get('number_source')=='library_pin_name' for p in owned):continue
        oldpts=[[p['point']['x'],height-p['point']['y']] for p in old.values()]
        if min(math.dist(a,b) for i,a in enumerate(oldpts) for b in oldpts[i+1:])>4:continue
        a,b,c,d=component['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if not found:continue
        frame=found[0];leads=vertical_leads(evidence,frame,'top',cfg)+vertical_leads(evidence,frame,'bottom',cfg)
        if len(leads)!=3 or sorted(Counter(l['side'] for l in leads).values())!=[1,2]:continue
        if max(l['length'] for l in leads)-min(l['length'] for l in leads)>cfg['max_length_spread_px']:continue
        if any(abs(a['point'][0]-b['point'][0])<28 for i,a in enumerate(leads) for b in leads[i+1:]):continue
        # Account for every existing identity and its exact original attachment.
        old_sources={};valid=True
        for key,pin in old.items():
            candidates=[p for p in owned if str(p.get('number'))==key[4:] and p.get('pinname')==pin['pinname']
                        and p.get('owner_status')=='assigned' and abs(p['point'][0]-pin['point']['x'])<=3
                        and math.dist(p['point'],[pin['point']['x'],height-pin['point']['y']])<=25]
            if len(candidates)!=1:valid=False;break
            attachment=attached(candidates[0],graph);ref=owner+'.'+key[4:]
            current=[nid for nid,net in target['nets'].items() if ref in _members(net)]
            if not attachment or current!=[attachment['network_id']]:valid=False;break
            old_sources[ref]=attachment['network_id']
        if not valid:continue
        rows=[]
        for lead in leads:
            route=lead_route(owner,lead,frame,document,graph,evidence)
            names=source_texts(document,owner,lead,frame,'pinname',cfg);name,support=vote(names,cfg)
            if not route or not name:valid=False;break
            selected=[t for t in names if t['text']==name]
            bounds=[statistics.median(t['bbox'][i] for t in selected) for i in range(4)]
            namebox=[math.floor(bounds[0]-2),math.floor(bounds[1]-2),math.ceil(bounds[2]+2),math.ceil(bounds[3]+2)]
            numbers=source_texts(document,owner,lead,frame,'number',cfg)
            anchor=lead
            if not numbers:
                anchors=[(l,source_texts(document,owner,l,frame,'number',cfg)) for l in leads if l['side']==lead['side']]
                anchors=[v for v in anchors if vote(v[1],cfg)[0] is not None]
                if len(anchors)!=1:valid=False;break
                anchor,numbers=anchors[0]
            value,support=vote(numbers,cfg)
            if value is None:valid=False;break
            dx=statistics.median((t['bbox'][0]+t['bbox'][2])/2-anchor['point'][0] for t in numbers)
            dy=statistics.median((t['bbox'][1]+t['bbox'][3])/2-lead['edge'] for t in numbers)
            x=lead['point'][0]+dx;y=lead['edge']+dy
            box=[math.floor(x-9),math.floor(y-9),math.ceil(x+9),math.ceil(y+9)]
            if neutral_patch(rgb,box) is None or neutral_patch(rgb,namebox) is None:valid=False;break
            rows.append({**lead,'route':route,'source_name':name,'source_name_support':vote(names,cfg)[1],
                         'name_crop':namebox,'number_crop':box,'source_number':value if anchor is lead else None})
        if not valid:continue
        if len(audit['groups'])>=cfg['max_components'] or audit['ocr_calls']+24>cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;break
        group={'owner':owner,'frame':frame,'rows':rows};audit['groups'].append(group)
        for row in rows:
            patch=neutral_patch(rgb,row['number_crop']);rotations={}
            for rotation,code in [(90,cv2.ROTATE_90_CLOCKWISE),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
                oriented=cv2.rotate(patch,code);tokens=[]
                for scale in [3.,5.]:tokens.extend(read(oriented,[0,0,oriented.shape[1],oriented.shape[0]],scale,f"vertical_glyph_{owner}_{row['side']}_{row['point'][0]}_r{rotation}_s{scale}"))
                number,support=vote([t for t in tokens if t['text'].isdigit()],cfg)
                rotations[rotation]={'number':number,'support':support,'readings':tokens}
            supported=[r for r,v in rotations.items() if v['number'] is not None and (row['source_number'] is None or v['number']==row['source_number'])]
            row['number_readings']=rotations
            if len(supported)!=1:valid=False;continue
            rotation=supported[0];row['rotation']=rotation;row['number']=rotations[rotation]['number']
            patch=rgb[row['name_crop'][1]:row['name_crop'][3],row['name_crop'][0]:row['name_crop'][2]]
            fresh=[]
            for mode in ['raw','neutral']:
                isolated=(cv2.copyMakeBorder(patch,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255)) if mode=='raw' else neutral_patch(rgb,row['name_crop']))
                oriented=cv2.rotate(isolated,cv2.ROTATE_90_CLOCKWISE if rotation==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                for scale in [3.,5.]:fresh.extend(read(oriented,[0,0,oriented.shape[1],oriented.shape[0]],scale,f"vertical_tight_{owner}_{row['side']}_{row['point'][0]}_{mode}_s{scale}"))
            # Corroborate identity without replacing observed source spelling
            # when OCR changes only capitalization (e.g. Vin -> vin).
            label,support=vote([{**t,'text':t['text'].casefold()} for t in fresh],cfg)
            strong={t['text'].casefold() for t in fresh if t['confidence']>=cfg['min_confidence']}
            row.update(name=row['source_name'],name_support=support,name_readings=fresh)
            if label!=row['source_name'].casefold() or strong!={label}:valid=False
        old_numbers={key[4:] for key in old}
        if (not valid or {r.get('number') for r in rows}!=old_numbers or len({r.get('rotation') for r in rows})!=1):
            group['status']='incomplete_or_conflicting_observation';continue
        planned=deepcopy(out);network_plan=[];restore_plan=[];all_refs=set(old_sources)
        for nid,net in planned['nets'].items():
            before=sorted(_members(net));after=set(before)-all_refs
            after.update(owner+'.'+r['number'] for r in rows if r['route']['network_id']==nid)
            if sorted(after)!=before:network_plan.append({'network_id':nid,'before':before,'after':sorted(after)})
            if not after:valid=False;break
            if sorted(after)!=before:net['hyperGraph']='('+','.join(sorted(after))+')'
        if not valid:group['status']='would_empty_existing_net';continue
        for row in rows:
            nid=row['route']['network_id']
            if nid not in out['nets']:
                branch=restore_direct_ground_branch(owner,row,planned,document,graph,evidence)
                if branch is None:valid=False;break
                planned['nets'][branch['network_id']]=branch['net']
                for o,key,pin in branch['pending_pins']:planned['pins'].setdefault(o,{})[key]=pin
                restore_plan.append({k:v for k,v in branch.items() if k not in {'net','pending_pins'}})
            key='pin_'+row['number'];updated=deepcopy(old[key]);x,y=row['point']
            updated.update(pinname=row['name'],point={'x':float(x),'y':float(height-y)})
            planned['pins'][owner][key]=updated
        if not valid:group['status']='unproven_missing_ground_branch';continue
        for key in old:
            ref=owner+'.'+key[4:]
            if sum(ref in _members(n) for n in planned['nets'].values())!=1:valid=False
        if not valid:group['status']='nonunique_result_membership';continue
        out=planned;group['status']='complete_observed_vertical_repair'
        group['preserved_unresolved_junctions']={n['network_id']:n['unresolved_junction_ids'] for n in graph['networks']
            if n['network_id'] in {r['route']['network_id'] for r in rows} and n.get('unresolved_junction_ids')}
        for row in rows:
            key='pin_'+row['number']
            if out['pins'][owner][key]!=old[key]:audit['changes'].append({'owner':owner,'number':row['number'],'before':old[key],'after':out['pins'][owner][key]})
        audit['network_changes'].extend(network_plan);audit['restored_networks'].extend(restore_plan)
    return out,audit
