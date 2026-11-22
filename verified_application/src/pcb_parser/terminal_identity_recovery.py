"""Recover identities and ambiguous ground contacts from local source evidence."""
import math
import re
import cv2
import numpy as np


def isolate_caption(rgb, box):
    """Keep dark blue ink including strokes blended with a crossing green wire."""
    h,w=rgb.shape[:2];a,b,c,d=map(round,box)
    a,b,c,d=max(0,a),max(0,b),min(w,c),min(h,d)
    if a>=c or b>=d:return None
    patch=rgb[b:d,a:c]
    if not patch.size:return None
    r,g,blue=[patch[:,:,i].astype(np.int16) for i in range(3)]
    mask=(blue>r+5)&(blue>.7*g)&(r<150)&(g<180)
    ys,xs=np.where(mask)
    if not len(xs):return None
    x1,x2,y1,y2=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    if not (6<=y2-y1<=24 and 6<=x2-x1<=55):return None
    ink=np.full_like(patch,255);ink[mask]=0
    return cv2.copyMakeBorder(ink[y1:y2,x1:x2],4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255)),[a+x1,b+y1,a+x2,b+y2]


def bar_profile(red, box, axis, count, point=None):
    h,w=red.shape;a,b,c,d=map(round,box);a,b,c,d=max(0,a-2),max(0,b-2),min(w,c+3),min(h,d+3)
    patch=red[b:d,a:c].astype('uint8')
    kernel=np.ones((14,1) if axis==0 else (1,14),np.uint8)
    opened=cv2.morphologyEx(patch,cv2.MORPH_OPEN,kernel)
    _,_,stats,centres=cv2.connectedComponentsWithStats(opened,8)
    bars=[]
    for s,p in zip(stats[1:],centres[1:]):
        x,y,bw,bh,area=map(int,s);length=(bh,bw)[axis];thickness=(bw,bh)[axis]
        if length>=14 and thickness<=4 and area>=length:
            bars.append({'center':[float(p[0]+a),float(p[1]+b)],'length':length})
    if len(bars)!=count:return None
    bars.sort(key=lambda v:v['center'][axis]);positions=[v['center'][axis] for v in bars]
    if any(not 3<=q-p<=14 for p,q in zip(positions,positions[1:])):return None
    cross=[v['center'][1-axis] for v in bars]
    if max(cross)-min(cross)>3:return None
    lengths=[v['length'] for v in bars]
    if count==2:
        if min(lengths)/max(lengths)<.85:return None
    elif count==3:
        if point is None:return None
        ordered=sorted(bars,key=lambda v:abs(point[axis]-v['center'][axis]))
        lengths=[v['length'] for v in ordered]
        if not all(p-q>=4 for p,q in zip(lengths,lengths[1:])):return None
        if abs(point[1-axis]-ordered[0]['center'][1-axis])>3:return None
        if not 2<abs(point[axis]-ordered[0]['center'][axis])<=24:return None
    return {'axis':axis,'bars':bars}


def ground_proof(draft, record, candidate):
    if record.get('type')!='gnd' or record.get('detection_confidence',0)<.8:return None
    owner=record['component_id']
    if (candidate.get('owner_status')!='ambiguous_owner' or candidate.get('owner')
            or candidate.get('confidence',0)<.9 or candidate.get('text_hypothesis_primary_candidate_id')):return None
    ids={v['component_id'] for v in candidate.get('owner_candidates',[])}
    if owner not in ids or len(ids)<2:return None
    opponents=[draft.rmap.get(key) for key in ids-{owner}]
    if any(not r or r['type']!='capacitor' or r.get('designator') or r.get('Name')
           or r['component_id'] in draft.out['components']
           or r.get('detection_confidence',0)>record['detection_confidence']-.05 for r in opponents):return None
    box=record['bbox_top_left'];a,b,c,d=box;point=candidate['point']
    center=[(a+c)/2,(b+d)/2];axis=int(abs(point[1]-center[1])>abs(point[0]-center[0]))
    shape=bar_profile(draft.red,box,axis,3,point)
    if not shape:return None
    direction=1 if point[axis]>center[axis] else -1
    path=None
    for distance in range(36):
        q=list(point);q[axis]+=direction*distance;x,y=map(round,q)
        if not (1<=x<draft.w-1 and 1<=y<draft.h-1):return None
        if not (draft.red|draft.green)[y-1:y+2,x-1:x+2].any():return None
        if draft.green[y-1:y+2,x-1:x+2].any():
            path={'source_point':point,'green_point':[x,y],'distance_px':distance};break
    if not path:return None
    words=[t for t in draft.words if str(t['text']).strip().upper()=='GND' and t['confidence']>=.98
           and a-40<=(t['bbox'][0]+t['bbox'][2])/2<=c+40
           and b-40<=(t['bbox'][1]+t['bbox'][3])/2<=d+40]
    if len({t['view'] for t in words})<2:return None
    return {'shape':shape,'outward_wire':path,'caption_observations':words,
            'excluded_records':sorted(ids-{owner})}


def capacitor_candidates(draft, record):
    owner=record['component_id']
    if (record['type']!='capacitor' or record.get('designator') or record.get('Name')
            or record.get('duplicate_of_detection_index') is not None
            or record.get('detection_confidence',0)<.8):return None
    a,b,c,d=record['bbox_top_left'];center=[(a+c)/2,(b+d)/2]
    if any(math.dist(center,[(draft.box(v)[0]+draft.box(v)[2])/2,
                            (draft.box(v)[1]+draft.box(v)[3])/2])<8
           for v in draft.out['components'].values()):return None
    candidates=[p for p in draft.pin_document['pin_candidates'] if p.get('owner')==owner
                and p.get('owner_status')=='assigned' and not p.get('text_hypothesis_primary_candidate_id')]
    if len(candidates)!=2 or {str(p.get('number')) for p in candidates}!={'1','2'}:return None
    if any(p.get('confidence',0)<.9 for p in candidates):return None
    first,second=[p['point'] for p in candidates];axis=int(abs(first[1]-second[1])>abs(first[0]-second[0]))
    if abs(first[axis]-second[axis])<10 or abs(first[1-axis]-second[1-axis])>3:return None
    if (first[axis]-center[axis])*(second[axis]-center[axis])>=0:return None
    shape=bar_profile(draft.red,record['bbox_top_left'],axis,2)
    return (candidates,shape) if shape else None


def recover_capacitors(draft):
    claimed=set()
    for record in draft.records:
        proof=capacitor_candidates(draft,record)
        if proof is None:continue
        candidates,shape=proof;a,b,c,d=record['bbox_top_left']
        region=[a-8,b-30,c+8,b+3];isolated=isolate_caption(draft.rgb,region)
        if isolated is None:continue
        readings=[]
        for scale in (3.,5.):
            if draft.reader is None or draft.ocr_calls>=draft.cfg.get('maximum_caption_views',8):break
            draft.ocr_calls+=1
            readings+=draft.reader(draft.rgb,region,scale,'capacitor_identity_caption')
        votes={}
        for t in readings:
            name=str(t.get('text','')).strip().upper()
            if re.fullmatch(r'C[1-9]\d*',name) and t.get('confidence',0)>=.98 and t.get('view'):
                votes.setdefault(name,set()).add(t['view'])
        if len(votes)!=1 or len(next(iter(votes.values())))<2:
            draft.withheld.append({'rule':'unidentified_capacitor_identity','source_record':record['component_id'],
                                   'reason':'no_unique_multiview_designator','caption_observations':readings});continue
        owner=next(iter(votes))
        if owner in draft.out['components'] or any(r.get('designator')==owner for r in draft.records):continue
        # Source endpoint numbering is retained; no number comes from a missing-ID sequence.
        draft.out['components'][owner]={'type':'c','Name':owner,'value':record.get('value') or '',
                                        'bbox':[a,draft.h-d,c,draft.h-b]}
        draft.rmap[owner]=record;draft.source_points[owner]=[p['point'] for p in candidates]
        pair=draft.terminal_pair(owner,record)
        if pair is None:
            del draft.out['components'][owner];del draft.rmap[owner];del draft.source_points[owner];continue
        axis,terminals=pair;ordered=sorted(candidates,key=lambda p:p['point'][axis])
        # Every extension from the observed terminal to its green contact is conductive.
        valid=True
        for p,q in zip(ordered,terminals):
            samples=np.rint(np.linspace(p['point'],q,max(2,math.ceil(math.dist(p['point'],q))+1))).astype(int)
            if any(not (draft.red|draft.green)[max(0,y-1):y+2,max(0,x-1):x+2].any() for x,y in samples):valid=False
        if not valid:
            del draft.out['components'][owner];del draft.rmap[owner];del draft.source_points[owner];continue
        draft.out['pins'][owner]={'pin_'+str(p['number']):draft.export_pin(q,str(p['number'])) for p,q in zip(ordered,terminals)}
        claimed.add(owner)
        draft.events.append({'rule':'unidentified_capacitor_identity','owner':owner,'source_record':record['component_id'],
            'source_candidates':[p['candidate_id'] for p in candidates],'shape':shape,
            'caption_observations':readings,'views':sorted(votes[owner]),'terminals':terminals})
    return claimed


def recover_grounds(draft):
    for owner,comp in sorted(draft.out['components'].items()):
        if comp['type']!='gnd' or draft.out['pins'].get(owner) or owner not in draft.rmap:continue
        choices=[]
        for candidate in draft.pin_document['pin_candidates']:
            proof=ground_proof(draft,draft.rmap[owner],candidate)
            if proof:choices.append((candidate,proof))
        if len(choices)!=1:continue
        candidate,proof=choices[0]
        draft.out['pins'][owner]={'pin_1':draft.export_pin(candidate['point'])}
        draft.events.append({'rule':'ground_owner_arbitration','owner':owner,'candidate':candidate['candidate_id'],**proof})
