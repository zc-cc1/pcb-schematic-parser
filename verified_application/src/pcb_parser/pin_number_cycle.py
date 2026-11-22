"""Corroborate a three-pin bottom row and atomically permute its identities."""
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES, LeadEvidence, _members, side_of, vote

DEFAULT={'max_components':2,'max_ocr_calls':24,'min_confidence':.98,'min_views':2,
         'min_candidate_confidence':.9,'min_spacing_px':16}


def exported_attachment(candidate,target,graph):
    matches=[a for a in graph.get('pin_attachments',[]) if a['candidate_id']==candidate['candidate_id']]
    if len(matches)!=1:return None
    attachment=matches[0];nid=attachment.get('network_id')
    source=next((n for n in graph['networks'] if n['network_id']==nid),None)
    if (attachment.get('attachment_status')!='attached' or attachment.get('point')!=candidate['point']
            or source is None or candidate['candidate_id'] not in source.get('pin_candidate_ids',[])):return None
    ref=candidate['owner']+'.'+str(candidate['number'])
    memberships=[n for n,net in target['nets'].items() if ref in _members(net)]
    if len(memberships)!=1:return None
    exported=memberships[0]
    if exported==nid:return exported
    physical=next((n for n in graph.get('physical_networks',[])
                   if n['network_id']==attachment.get('physical_network_id')),None)
    if (physical is None or physical.get('unresolved_junction_ids')
            or candidate['candidate_id'] not in physical.get('pin_candidate_ids',[])
            or physical['network_id'] not in source.get('physical_network_ids',[])
            or not set(physical.get('edge_ids',[]))<=set(source.get('edge_ids',[]))
            or exported!='RESTORED_'+physical['network_id']):return None
    return exported


def number_patch(rgb,box):
    patch=rgb[box[1]:box[3],box[0]:box[2]];colors=patch.astype(np.int16)
    mask=(colors.max(axis=2)-colors.min(axis=2)<35)&(colors.mean(axis=2)<205)
    isolated=np.full_like(patch,255);isolated[mask]=patch[mask]
    return cv2.copyMakeBorder(isolated,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))


def reconcile_bottom_number_cycles(target,document,graph,records,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};height,width=rgb.shape[:2]
    out=deepcopy(target);evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT)
    by_owner={r['component_id']:r for r in records};pins=document['pin_candidates'];selected=0
    audit={'enabled':True,'policy':'bottom_number_cycle_v1','groups':[],'changes':[],
           'network_changes':[],'ocr_calls':0,'budget_exhausted':False}
    def read(image,box,scale,rotation,recognition_only,view):
        if audit['ocr_calls']>=cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;return []
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,recognition_only,view)
    for owner,component in sorted(target['components'].items()):
        if component['type'] not in IC_TYPES or owner not in by_owner:continue
        a,b,c,d=component['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if not found:continue
        frame=found[0]
        same=[p for p in pins if p.get('owner')==owner and p.get('owner_status')=='assigned'
              and not p.get('text_hypothesis_primary_candidate_id') and side_of(p['point'],frame)=='bottom'
              and str(p.get('number','')).isdigit() and p.get('pinname')
              and any(ch.isalpha() for ch in p['pinname'])]
        if len(same)!=3 or not any(p.get('number_source')=='library_pin_name' for p in same):continue
        same.sort(key=lambda p:p['point'][0]);old_numbers={str(p['number']) for p in same}
        if len(old_numbers)!=3 or any(same[i+1]['point'][0]-same[i]['point'][0]<cfg['min_spacing_px'] for i in range(2)):continue
        if any(p.get('confidence',0)<cfg['min_candidate_confidence'] for p in same):continue
        if any(sum(q.get('owner')==owner and str(q.get('number'))==str(p['number']) and not q.get('text_hypothesis_primary_candidate_id') for q in pins)!=1 for p in same):continue
        current=[];nets=[];valid=True
        for p in same:
            pin=out['pins'].get(owner,{}).get('pin_'+str(p['number']))
            if pin is None or pin['pinname']!=p['pinname']:valid=False;break
            point=[pin['point']['x'],height-pin['point']['y']]
            if (side_of(point,frame)!='bottom' or abs(point[0]-p['point'][0])>3
                    or not 8<=point[1]-frame[3]<=60
                    or sum(evidence.hit(point[0],frame[3]+d) for d in range(3,11))<6):valid=False;break
            nid=exported_attachment(p,out,graph)
            if nid is None:valid=False;break
            current.append(pin);nets.append(nid)
        if not valid:continue
        anchors=[];offsets=[]
        for p in same:
            if p.get('number_source')=='library_pin_name':continue
            tokens=[]
            for t in p.get('text_candidates',{}).get('number',[]):
                x=(t['bbox'][0]+t['bbox'][2])/2;y=(t['bbox'][1]+t['bbox'][3])/2
                if (str(t.get('text'))==str(p['number']) and t.get('confidence',0)>=cfg['min_confidence']
                        and t.get('source_view') and not t.get('approximate_split_geometry')
                        and 0<y-frame[3]<20 and abs(x-p['point'][0])<15):
                    tokens.append({**t,'view':t['source_view']})
            value,_=vote(tokens,cfg)
            if value==str(p['number']):
                anchors.append(p)
                offsets.extend([((t['bbox'][0]+t['bbox'][2])/2-p['point'][0],(t['bbox'][1]+t['bbox'][3])/2-frame[3]) for t in tokens])
        if not anchors:continue
        if selected>=cfg['max_components']:
            audit['budget_exhausted']=True;break
        selected+=1
        source_ids={a['network_id'] for a in graph['pin_attachments'] if a['candidate_id'] in {p['candidate_id'] for p in same}}
        preserved_junctions={n['network_id']:n['unresolved_junction_ids'] for n in graph['networks']
                             if n['network_id'] in source_ids and n.get('unresolved_junction_ids')}
        group={'owner':owner,'rows':[],'anchor_candidates':[p['candidate_id'] for p in anchors],
               'preserved_unresolved_junctions':preserved_junctions};audit['groups'].append(group)
        dx=statistics.median(v[0] for v in offsets);dy=statistics.median(v[1] for v in offsets)
        patches={};number_cache={}
        for p in same:
            x=p['point'][0]+dx;y=frame[3]+dy
            box=[max(0,math.floor(x-8)),max(0,math.floor(y-8)),min(width,math.ceil(x+8)),min(height,math.ceil(y+8))]
            patches[p['candidate_id']]=(number_patch(rgb,box),box)
        def read_number(p,rotation):
            key=(p['candidate_id'],rotation)
            if key not in number_cache:
                patch,box=patches[p['candidate_id']]
                oriented=cv2.rotate(patch,cv2.ROTATE_90_CLOCKWISE if rotation==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                tokens=[]
                for scale in [3.,5.]:tokens.extend(read(oriented,[0,0,oriented.shape[1],oriented.shape[0]],scale,0,True,f"cycle_glyph_{p['candidate_id']}_r{rotation}_s{scale}"))
                value,support=vote([t for t in tokens if t['text'].isdigit()],cfg)
                number_cache[key]={'number':value,'support':support,'readings':tokens,'crop':box}
            return number_cache[key]
        orientations=[]
        for anchor in anchors:
            supported=[r for r in [90,270] if read_number(anchor,r)['number']==str(anchor['number'])]
            if len(supported)==1:orientations.append(supported[0])
        if not orientations or len(set(orientations))!=1:
            group['status']='orientation_not_uniquely_corroborated';continue
        rotation=orientations[0];group['glyph_rotation']=rotation
        box=by_owner[owner]['bbox_top_left'];crop=[max(0,math.floor(box[0]-32)),max(0,math.floor(box[1]-32)),min(width,math.ceil(box[2]+32)),min(height,math.ceil(box[3]+32))]
        whole=[]
        for scale in [1.,3.]:
            for r in [0,90]:whole.extend(read(rgb,crop,scale,r,False,f'cycle_component_{owner}_s{scale}_r{r}'))
        for p,nid in zip(same,nets):
            reading=read_number(p,rotation);names=[];x=p['point'][0]
            for t in whole:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2;distance=abs(cx-x)
                if not(frame[0]<cx<frame[2] and frame[1]<cy<frame[3] and distance<=12 and any(ch.isalpha() for ch in t['text'])):continue
                rivals=[abs(cx-q['point'][0]) for q in same if q['candidate_id']!=p['candidate_id']]
                if min(rivals)-distance>=3:names.append(t)
            name,support=vote(names,cfg)
            group['rows'].append({'owner':owner,'candidate_id':p['candidate_id'],'old_number':str(p['number']),
                                  'number':reading['number'],'name':name,'expected_name':p['pinname'],
                                  'number_support':reading['support'],'number_readings':reading['readings'],
                                  'number_crop':reading['crop'],'name_support':support,'network_id':nid})
        rows=group['rows']
        if (any(not r['number'] or r['name']!=r['expected_name'] for r in rows)
                or {r['number'] for r in rows}!=old_numbers):
            group['status']='not_a_fully_observed_closed_permutation';continue
        if any(next(r for r in rows if r['candidate_id']==p['candidate_id'])['number']!=str(p['number']) for p in anchors):
            group['status']='observed_anchor_would_change';continue
        changes=[r for r in rows if r['old_number']!=r['number']]
        if not changes:group['status']='verified_unchanged';continue
        # Preserve physical records, and substitute every reference from one
        # immutable snapshot so a swap cannot collapse two refs into one.
        saved={r['old_number']:deepcopy(out['pins'][owner]['pin_'+r['old_number']]) for r in rows}
        mapping={owner+'.'+r['old_number']:owner+'.'+r['number'] for r in changes}
        replacements={nid:sorted({mapping.get(ref,ref) for ref in _members(net)}) for nid,net in out['nets'].items()}
        for r in rows:out['pins'][owner]['pin_'+r['number']]=saved[r['old_number']]
        for nid,after in replacements.items():
            before=sorted(_members(out['nets'][nid]))
            if before!=after:
                out['nets'][nid]['hyperGraph']='('+','.join(after)+')'
                audit['network_changes'].append({'network_id':nid,'before':before,'after':after})
        audit['changes'].extend(changes);group['status']='closed_number_and_network_permutation'
    return out,audit
