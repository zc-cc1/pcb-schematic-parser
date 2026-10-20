"""Relocate unsupported library-elimination pins using observed text and leads.

No case identities, component identities, target fields or reference coordinates
are encoded. Existing attached pins and directly observed fields are protected.
"""
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES, LeadEvidence, _members, side_of, sync_networks, vote

DEFAULT = {'max_candidates': 2, 'max_ocr_calls': 12, 'min_confidence': .98,
           'min_views': 2, 'max_old_confidence': .8, 'max_old_wire_probability': .2,
           'min_same_side_pins': 3, 'min_displacement_px': 25}


def unsupported_candidate(candidate, target, graph, height, cfg):
    owner=candidate.get('owner');number=str(candidate.get('number'));name=candidate.get('pinname')
    if not (candidate.get('owner_status')=='assigned' and number.isdigit() and name
            and target['components'].get(owner,{}).get('type') in IC_TYPES
            and candidate.get('number_source')=='library_elimination'
            and candidate.get('pinname_source')=='library_elimination'
            and not candidate.get('text_evidence') and not candidate.get('text_hypothesis_primary_candidate_id')
            and candidate.get('confidence',1)<cfg['max_old_confidence']
            and candidate.get('nearby_wire_probability_max',1)<cfg['max_old_wire_probability']):return False
    if any(t.get('confidence',0)>=cfg['min_confidence'] for kind in ['number','pinname']
           for t in candidate.get('text_candidates',{}).get(kind,[])):return False
    if any(candidate['candidate_id'] in n.get('pin_candidate_ids',[]) for n in graph.get('networks',[])):return False
    old=target['pins'].get(owner,{}).get('pin_'+number)
    if old is None or old['pinname']!=name:return False
    point=[old['point']['x'],height-old['point']['y']]
    if math.dist(point,candidate['point'])>2:return False
    ref=owner+'.'+number
    if any(ref in _members(n) for n in target['nets'].values()):return False
    attachments=[a for a in graph.get('pin_attachments',[]) if a['candidate_id']==candidate['candidate_id']]
    if len(attachments)!=1 or attachments[0].get('attachment_status')!='no_wire_within_radius':return False
    if attachments[0].get('point')!=candidate['point'] or attachments[0].get('network_id') is not None:return False
    return True


def isolate_number(rgb, box):
    patch=rgb[box[1]:box[3],box[0]:box[2]];colors=patch.astype(np.int16)
    mask=((colors.max(axis=2)-colors.min(axis=2)<35)&(colors.mean(axis=2)<205)).astype(np.uint8)
    mask&=1-cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,13),np.uint8))
    image=np.full_like(patch,255);image[mask>0]=patch[mask>0]
    return cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))


def relocate_unsupported_pins(target, document, graph, records, rgb, *, reader, cfg=None):
    cfg={**DEFAULT,**(cfg or {})};height,width=rgb.shape[:2]
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT);out=deepcopy(target)
    audit={'enabled':True,'policy':'unsupported_pin_relocation_v1','proposals':[],
           'changes':[],'network_changes':[],'ocr_calls':0,'budget_exhausted':False}
    by_owner={r['component_id']:r for r in records};candidates=document['pin_candidates']
    def read(image,box,scale,rotation,recognition_only,view):
        if audit['ocr_calls']>=cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;return []
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,recognition_only,view)
    for candidate in sorted(candidates,key=lambda p:p['candidate_id']):
        if not unsupported_candidate(candidate,out,graph,height,cfg):continue
        owner=candidate['owner'];number=str(candidate['number']);name=candidate['pinname'];key='pin_'+number
        if owner not in by_owner:continue
        if sum(p.get('owner')==owner and str(p.get('number'))==number for p in candidates)!=1:continue
        a,b,c,d=out['components'][owner]['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if found is None:continue
        box=found[0];x,y=candidate['point']
        if not(box[0]-2<=x<=box[2]+2 and box[1]-2<=y<=box[3]+2):continue
        if evidence.green[max(0,round(y)-4):round(y)+5,max(0,round(x)-4):round(x)+5].any():continue
        # This version validates only left-facing rows; other layouts abstain.
        known=[(k,p,[p['point']['x'],height-p['point']['y']]) for k,p in out['pins'].get(owner,{}).items() if k!=key]
        same=[item for item in known if side_of(item[2],box)=='left']
        if len(same)<cfg['min_same_side_pins']:continue
        rows=[p for p in evidence.leads(box,'left') if not any(abs(pt[1]-p['row'])<=6 for k,pin,pt in same)]
        if len(rows)!=1:continue
        proposal=rows[0];row=proposal['row'];new_point=proposal['point']
        if math.dist([x,y],new_point)<cfg['min_displacement_px']:continue
        observed=[]
        for p in candidates:
            if (p.get('owner')!=owner or not p.get('number') or 'library' in p.get('number_source','')
                    or not any(math.dist(p['point'],pt)<=25 for k,pin,pt in same)):continue
            for t in p.get('text_candidates',{}).get('number',[]):
                if str(t.get('text'))==str(p['number']) and t.get('confidence',0)>=cfg['min_confidence']:
                    observed.append(((t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2-p['point'][1]))
        if not observed:continue
        if len(audit['proposals'])>=cfg['max_candidates']:
            audit['budget_exhausted']=True;break
        column=statistics.median(v[0] for v in observed);offset=statistics.median(v[1] for v in observed)
        expected=row+offset
        number_box=[max(0,math.floor(column-11)),max(0,math.floor(expected-12)),min(width,math.ceil(column+11)),min(height,math.ceil(expected+12))]
        isolated=isolate_number(rgb,number_box);texts=[]
        for scale in [3.,5.]:
            texts.extend(read(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'relocation_number_{owner}_s{scale}'))
        observed_number,number_support=vote([t for t in texts if t['text'].isdigit()],cfg)
        entry={'owner':owner,'number':number,'name':name,'source_candidate_id':candidate['candidate_id'],
               'before':deepcopy(out['pins'][owner][key]['point']),'old_point':[x,y],'point':new_point,
               'observed_number':observed_number,'number_support':number_support,'number_crop':number_box}
        audit['proposals'].append(entry)
        if observed_number!=number:entry['status']='number_not_confirmed';continue
        old_box=by_owner[owner]['bbox_top_left']
        component_box=[max(0,math.floor(old_box[0]-32)),max(0,math.floor(old_box[1]-32)),
                       min(width,math.ceil(old_box[2]+32)),min(height,math.ceil(old_box[3]+32))]
        names=[]
        for scale in [2.,3.]:
            for rotation in [0,90]:
                observations=read(rgb,component_box,scale,rotation,False,f'relocation_component_{owner}_s{scale}_r{rotation}')
                names.extend(t for t in observations if box[0]<(t['bbox'][0]+t['bbox'][2])/2<box[2]
                             and box[1]<(t['bbox'][1]+t['bbox'][3])/2<box[3]
                             and abs((t['bbox'][1]+t['bbox'][3])/2-row)<=12
                             and any(ch.isalpha() for ch in t['text']))
        observed_name,name_support=vote(names,cfg);entry.update(observed_name=observed_name,name_support=name_support,component_crop=component_box)
        if observed_name!=name:entry['status']='name_not_confirmed';continue
        out['pins'][owner][key]['point']={'x':float(new_point[0]),'y':float(height-new_point[1])}
        entry['after']=deepcopy(out['pins'][owner][key]['point']);entry['status']='relocated_by_observed_text_and_lead'
        audit['changes'].append(entry)
        # The old candidate's attachment is invalid at the new point. Re-evaluate
        # from visible wire geometry, and allow only an already-exported net.
        audit['network_changes'].extend(sync_networks(out,graph,[{
            'owner':owner,'number':number,'name':name,'point':new_point,'status':entry['status']}],evidence))
    return out,audit
