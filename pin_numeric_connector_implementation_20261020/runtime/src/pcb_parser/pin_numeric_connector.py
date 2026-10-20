"""Recover independently observed numeric connector rows and local supply contacts.

Two existing digit-named pins calibrate a text column and lead length. Every
missing identity still requires its own multi-view text and fresh glyph reads.
No numbering sequence, component model, reference labels or wire continuation
is inferred. A proven T supply may form a local two-contact logical net only.
"""
from collections import defaultdict
from copy import deepcopy
import math
import statistics

import numpy as np

from .pin_geometry_recovery import LeadEvidence, DEFAULT as GEOMETRY_DEFAULT, _segment_distance

DEFAULT = {'min_confidence': .98, 'min_views': 2, 'max_components': 4,
           'max_ocr_calls': 24, 'max_rows': 16, 'scales': [3., 5.]}
POWER_NAMES = {'VIN','VOUT','VCC','VDD','VBAT','VBUS','AVDD','DVDD','VSS'}


def observations(document):
    values = []
    seen = set()
    for token in document.get('texts', []):
        for t in [token, *token.get('observations', [])]:
            box = t.get('bbox'); view = t.get('view')
            if not view or not box or len(box) != 4 or not all(math.isfinite(v) for v in box):continue
            key = (str(t.get('text')), view, tuple(box))
            if key in seen:continue
            seen.add(key)
            values.append({'text': str(t['text']), 'confidence': float(t.get('confidence',0)),
                           'bbox': list(box), 'view': view})
    return values


def vote(tokens, cfg):
    support = defaultdict(set)
    for token in tokens:
        if token.get('confidence',0) >= cfg['min_confidence']:
            support[str(token['text'])].add(token['view'])
    if len(support) != 1:return None
    text = next(iter(support))
    return text if len(support[text]) >= cfg['min_views'] else None


def isolate(rgb, box, color):
    h,w = rgb.shape[:2]
    a,b,c,d = max(0,math.floor(box[0])),max(0,math.floor(box[1])),min(w,math.ceil(box[2])),min(h,math.ceil(box[3]))
    if a>=c or b>=d:return None
    patch = rgb[b:d,a:c]
    r,g,blue = [patch[:,:,i].astype(int) for i in range(3)]
    if color == 'neutral':
        mask = (abs(r-g)<35)&(abs(r-blue)<35)&(abs(g-blue)<35)&((r+g+blue)/3<230)
    else:
        mask = (blue>r+35)&(blue>g+35)&(r<180)&(g<180)
    yy,xx = np.where(mask)
    if len(xx)<10:return None
    x1,x2,y1,y2 = min(xx),max(xx),min(yy),max(yy)
    if not 6<=y2-y1+1<=32 or x2-x1+1>55:return None
    source = [a+int(x1),b+int(y1),a+int(x2)+1,b+int(y2)+1]
    isolated = np.full((y2-y1+9,x2-x1+9,3),255,np.uint8)
    isolated[4:-4,4:-4][mask[y1:y2+1,x1:x2+1]]=0
    return isolated,source


def power_t(evidence, point):
    """Find a narrow red stem ending in one symmetric red cap, without a stem beyond it."""
    x,y = point
    sites=[]
    for direction in [-1,1]:
        found=[]
        for cx in range(round(x)-3,round(x)+4):
            for length in range(18,49):
                cy=round(y)+direction*length
                if not 10<=cy<evidence.height-10:continue
                stem=[evidence.hit(cx,round(y)+direction*d) for d in range(9,length-2)]
                if not stem or sum(stem)/len(stem)<.94:continue
                if any(evidence.hit(cx,cy+direction*d) for d in range(5,10)):continue
                left=right=0
                while left<28 and evidence.hit(cx-left-1,cy):left+=1
                while right<28 and evidence.hit(cx+right+1,cy):right+=1
                if not 10<=left<=24 or not 10<=right<=24 or abs(left-right)>4:continue
                found.append((cx,cy,left,right))
        if not found:continue
        cx=statistics.median(v[0] for v in found);cy=statistics.median(v[1] for v in found)
        if max(v[1] for v in found)-min(v[1] for v in found)>5:continue
        left=statistics.median(v[2] for v in found);right=statistics.median(v[3] for v in found)
        if abs(cx-x)>3:continue
        sites.append({'point':[cx,y], 'cap_y':cy, 'direction':direction,
                      'bbox':[cx-left,min(y,cy),cx+right,max(y,cy)]})
    return sites[0] if len(sites)==1 else None


def recover_numeric_connectors(target, document, graph, ocr_document, rgb, *, reader, cfg=None):
    cfg={**DEFAULT,**(cfg or {})}
    out=deepcopy(target);height,width=rgb.shape[:2]
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT);tokens=observations(ocr_document)
    audit={'enabled':True,'policy':'observed_numeric_connector_v1','groups':[], 'changes':[],
           'component_changes':[], 'network_changes':[], 'network_withheld':[], 'ocr_calls':0,'budget_exhausted':False}
    def confirm(box,color,view):
        isolated=isolate(rgb,box,color)
        if isolated is None:return None,[],None
        patch,source_box=isolated
        readings=[]
        for scale in cfg['scales']:
            if audit['ocr_calls']>=cfg['max_ocr_calls']:
                audit['budget_exhausted']=True;return None,readings,source_box
            audit['ocr_calls']+=1
            values=reader(patch,[0,0,patch.shape[1],patch.shape[0]],scale,0,True,view+f'_s{scale}')
            readings.extend({**t,'bbox':source_box} for t in values)
        return vote(readings,cfg),readings,source_box
    owners=0
    for owner,comp in sorted(target['components'].items()):
        if comp.get('type') not in {'box','ic'}:continue
        existing=target['pins'].get(owner,{})
        if len(existing)<2 or not all(k.startswith('pin_') and k[4:].isdigit() and p.get('pinname')==k[4:] for k,p in existing.items()):continue
        a,b,c,d=comp['bbox'];frame_result=evidence.frame([a,height-d,c,height-b])
        if frame_result is None:continue
        frame=frame_result[0];left=evidence.leads(frame,'left');right=evidence.leads(frame,'right')
        if bool(left)==bool(right):continue
        leads=left or right;side='left' if left else 'right';edge=frame[0] if left else frame[2];sign=-1 if left else 1
        if not len(existing)<len(leads)<=cfg['max_rows']:continue
        pitch=min(q['row']-p['row'] for p,q in zip(leads,leads[1:]))
        if pitch<16:continue
        digit_tokens=[]
        for t in tokens:
            x1,y1,x2,y2=t['bbox'];cx=(x1+x2)/2
            if not t['text'].isdigit() or len(t['text'])>3 or t['confidence']<cfg['min_confidence']:continue
            if not frame[0]+2<=x1<x2<=frame[2]-2 or not frame[1]+2<=y1<y2<=frame[3]-2:continue
            # OCR boxes may extend slightly beyond one row even for one glyph;
            # merged multi-row strings remain too tall and fail this gate.
            if not 5<=y2-y1<=min(35,1.2*pitch):continue
            if (side=='left' and cx>frame[0]+.55*(frame[2]-frame[0])) or (side=='right' and cx<frame[2]-.55*(frame[2]-frame[0])):continue
            digit_tokens.append(t)
        anchors=[];used=set()
        for key,pin in existing.items():
            px,py=pin['point']['x'],height-pin['point']['y']
            matches=[i for i,l in enumerate(leads) if abs(l['row']-py)<=6 and 8<=sign*(px-edge)<=55]
            if len(matches)!=1 or matches[0] in used:break
            index=matches[0];used.add(index)
            possible=[t for t in digit_tokens if t['text']==key[4:] and abs((t['bbox'][1]+t['bbox'][3])/2-leads[index]['row'])<=15]
            if vote(possible,cfg)!=key[4:]:break
            anchors.append({'key':key,'index':index,'length':sign*(px-edge),
                            'column':statistics.median((t['bbox'][0]+t['bbox'][2])/2 for t in possible),
                            'offset':statistics.median((t['bbox'][1]+t['bbox'][3])/2-leads[index]['row'] for t in possible)})
        if len(anchors)!=len(existing):continue
        if any(max(a[k] for a in anchors)-min(a[k] for a in anchors)>limit for k,limit in [('length',4),('column',6),('offset',6)]):continue
        column=statistics.median(a['column'] for a in anchors);offset=statistics.median(a['offset'] for a in anchors)
        lead_length=statistics.median(a['length'] for a in anchors)
        missing=[]
        for index,lead in enumerate(leads):
            if index in used:continue
            candidates=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                        and not p.get('text_hypothesis_primary_candidate_id') and abs(p['point'][1]-lead['row'])<=6
                        and min(edge,lead['point'][0])-3<=p['point'][0]<=max(edge,lead['point'][0])+3]
            if len(candidates)!=1 or candidates[0].get('number') is not None or candidates[0].get('pinname') is not None:break
            possible=[t for t in digit_tokens if abs((t['bbox'][0]+t['bbox'][2])/2-column)<=10
                      and abs((t['bbox'][1]+t['bbox'][3])/2-lead['row']-offset)<=8]
            number=vote(possible,cfg)
            if number is None or 'pin_'+number in existing:break
            point=[edge+sign*lead_length,lead['row']]
            if abs(lead['length']-lead_length)>10 or math.dist(point,candidates[0]['point'])>12 or not evidence.hit(*point):break
            missing.append({'number':number,'point':point,'candidate':candidates[0], 'tokens':possible})
        if len(missing)!=len(leads)-len(existing) or len({p['number'] for p in missing})!=len(missing):continue
        if owners>=cfg['max_components']:break
        owners+=1
        group={'owner':owner,'anchors':anchors,'frame':frame,'rows':[],'status':'pending'}
        audit['groups'].append(group)
        for proposal in missing:
            token=max(proposal['tokens'],key=lambda t:t['confidence'])
            text,readings,box=confirm(token['bbox'],'neutral',f'connector_{owner}_{proposal["number"]}')
            group['rows'].append({**{k:v for k,v in proposal.items() if k!='candidate'},
                                  'candidate_id':proposal['candidate']['candidate_id'],'fresh_readings':readings,'glyph_box':box})
            if text!=proposal['number']:break
        else:
            group['status']='accepted'
        if group['status']!='accepted':group['status']='fresh_read_disagrees';continue
        for proposal in missing:
            number=proposal['number'];point=proposal['point'];cid=proposal['candidate']['candidate_id'];ref=owner+'.'+number
            new_pin={'pinname':number,'point':{'x':point[0],'y':height-point[1]}}
            out['pins'].setdefault(owner,{})['pin_'+number]=new_pin
            audit['changes'].append({'owner':owner,'number':number,'source_candidate_id':cid,
                                     'method':'observed_numeric_name_and_consensus_position','after':new_pin})
            attachment=[a for a in graph.get('pin_attachments',[]) if a.get('candidate_id')==cid]
            source_net=next((n for n in graph.get('networks',[]) if len(attachment)==1 and n['network_id']==attachment[0].get('network_id')),None)
            destination=(source_net['network_id'] if source_net and not source_net.get('unresolved_junction_ids')
                         and attachment[0].get('attachment_status')=='attached' and attachment[0].get('point')==proposal['candidate']['point']
                         and cid in source_net.get('pin_candidate_ids',[]) and source_net['network_id'] in out['nets'] else None)
            if destination:
                edge_ids=set(source_net.get('edge_ids',[]))
                segments=[(a,b) for e in graph.get('edges',[]) if e['edge_id'] in edge_ids
                          for a,b in zip(e.get('polyline',[]),e.get('polyline',[])[1:])]
                px,py=map(round,point)
                green=evidence.green[max(0,py-5):py+6,max(0,px-5):px+6].any()
                if not green or not segments or min(_segment_distance(point,a,b) for a,b in segments)>8:destination=None
            existing_refs=[nid for nid,n in out['nets'].items() if ref in n['hyperGraph'].strip('()').split(',')]
            if existing_refs:destination=None
            supply=None;site=power_t(evidence,point)
            if site:
                sx,sy=site['point'];cap=site['cap_y'];direction=site['direction']
                roi=[sx-35,cap+3,sx+35,cap+33] if direction==1 else [sx-35,cap-33,sx+35,cap-3]
                power,reads,label_box=confirm(roi,'blue',f'connector_supply_{owner}_{number}')
                group['rows'][next(i for i,p in enumerate(missing) if p is proposal)]['supply']={'site':site,'readings':reads,'label_box':label_box,'text':power}
                already=any(c.get('type') in {'v','gnd'} and math.dist([(c['bbox'][0]+c['bbox'][2])/2,height-(c['bbox'][1]+c['bbox'][3])/2],[(site['bbox'][0]+site['bbox'][2])/2,(site['bbox'][1]+site['bbox'][3])/2])<35 for c in out['components'].values())
                key='RECOVERED_SUPPLY_'+cid
                if (power in POWER_NAMES and not already and key not in out['components'] and label_box
                        and abs((label_box[0]+label_box[2])/2-sx)<=10):
                    a,b,c,d=site['bbox'];supply=key+'.1'
                    new_component={'type':'v','Name':None,'value':power,'bbox':[a,height-d,c,height-b]}
                    out['components'][key]=new_component
                    out['pins'][key]={'pin_1':{'pinname':'Pin1','point':{'x':sx,'y':height-sy}}}
                    audit['component_changes'].append({'owner':key,'method':'observed_t_symbol_and_fresh_power_text','after':new_component})
            if destination:
                before=sorted(v for v in out['nets'][destination]['hyperGraph'].strip('()').split(',') if v)
                after=sorted(set(before)|{ref}|({supply} if supply else set()))
                out['nets'][destination]['hyperGraph']='('+','.join(after)+')'
                audit['network_changes'].append({'network_id':destination,'before':before,'after':after,'reason':'original_unique_attachment'})
            elif supply and not existing_refs:
                nid='OBSERVED_CONTACT_'+cid
                if nid not in out['nets']:
                    members=sorted([ref,supply]);out['nets'][nid]={'hyperGraph':'('+','.join(members)+')','edges':{}}
                    audit['network_changes'].append({'network_id':nid,'before':None,'after':members,'reason':'same_observed_t_contact_only'})
                audit['network_withheld'].append({'member':ref,'reason':'upstream_route_not_reconstructed','source_attachment_status':attachment[0].get('attachment_status') if len(attachment)==1 else None})
            else:
                audit['network_withheld'].append({'member':ref,'reason':'no_unique_original_exported_attachment'})
    return out,audit
