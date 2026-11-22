"""Recover a source-confirmed filled LED whose caption was fused with a value."""
import math
import re

import cv2
import numpy as np

from .terminal_identity_recovery import isolate_caption


def single_led_proof(draft,record):
    if (record['type']!='led' or record.get('designator') or record.get('Name')
            or record.get('detection_confidence',0)<.8):return None
    candidates=[p for p in draft.pin_document['pin_candidates'] if p.get('owner')==record['component_id']
                and p.get('owner_status')=='assigned' and not p.get('text_hypothesis_primary_candidate_id')]
    if len(candidates)!=2 or {str(p.get('number')) for p in candidates}!={'1','2'}:return None
    if min(p.get('confidence',0) for p in candidates)<.6 or max(p.get('confidence',0) for p in candidates)<.9:return None
    candidates.sort(key=lambda p:p['point'][1])
    if abs(candidates[0]['point'][0]-candidates[1]['point'][0])>3:return None
    a,b,c,d=record['bbox_top_left']
    x1,y1=max(0,math.floor(a)),max(0,math.floor(b))
    x2,y2=min(draft.w,math.ceil(c)),min(draft.h,math.ceil(d))
    opened=cv2.morphologyEx(draft.red[y1:y2,x1:x2].astype('uint8'),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    triangles=[];small=[]
    for contour in contours:
        area=cv2.contourArea(contour);xx,yy,ww,hh=cv2.boundingRect(contour)
        if 25<=area<=120:
            small.append({'center':np.array([x1+xx+ww/2,y1+yy+hh/2]),
                          'box':[x1+xx-3,y1+yy-3,x1+xx+ww+2,y1+yy+hh+2]})
        if not 150<=area<=1200 or not .65<=ww/hh<=1.8:continue
        triangle_area,vertices=cv2.minEnclosingTriangle(contour)
        if triangle_area<=0 or area/triangle_area<.75:continue
        vertices=vertices.reshape(-1,2)+[x1,y1]
        ordered=vertices[np.argsort(vertices[:,1])]
        if abs(ordered[1,1]-ordered[2,1])>4:continue
        if abs(ordered[0,0]-candidates[0]['point'][0])>4:continue
        if not 10<ordered[2,1]-ordered[0,1]<55:continue
        triangles.append(vertices)
    if len(triangles)!=1:return None
    vertices=triangles[0];center=vertices.mean(0)
    if not candidates[0]['point'][1]<center[1]<candidates[1]['point'][1]:return None
    arrows=[p for p in small if 10<center[0]-p['center'][0]<60 and 5<center[1]-p['center'][1]<55]
    if len(arrows)!=2:return None
    low=vertices.min(0)-2;high=vertices.max(0)+2
    box=[float(min(low[0],*(p['box'][0] for p in arrows))),float(min(low[1],*(p['box'][1] for p in arrows))),
         float(max(high[0],*(p['box'][2] for p in arrows))),float(max(high[1],*(p['box'][3] for p in arrows)))]
    right=vertices[:,0].max();top=vertices[:,1].min();bottom=vertices[:,1].max()
    xx1,yy1=max(0,math.floor(right+4)),max(0,math.floor(top-18))
    xx2,yy2=min(draft.w,math.ceil(right+140)),min(draft.h,math.ceil(bottom+5))
    patch=draft.rgb[yy1:yy2,xx1:xx2].astype(np.int16);r,g,blue=[patch[:,:,i] for i in range(3)]
    mask=(blue>r+35)&(blue>g+35)&(r<180)&(g<180)
    columns=np.where(mask.any(axis=0))[0];groups=[]
    for column in columns:
        if not groups or column-groups[-1][-1]>6:groups.append([])
        groups[-1].append(int(column))
    if not groups or not 15<=groups[0][-1]-groups[0][0]+1<=55:return None
    group=groups[0];ys=np.where(mask[:,group[0]:group[-1]+1].any(axis=1))[0]
    if not len(ys) or not 6<=ys[-1]-ys[0]+1<=24:return None
    caption=[xx1+group[0]-2,yy1+int(ys[0])-2,xx1+group[-1]+3,yy1+int(ys[-1])+3]
    terminals=[]
    for candidate,sign in zip(candidates,(-1,1)):
        point=list(candidate['point']);contact=point.copy()
        for step in range(36):
            q=[point[0],point[1]+sign*step];x,y=map(round,q)
            if not(2<=x<draft.w-2 and 2<=y<draft.h-2) or not (draft.red|draft.green)[y-1:y+2,x-1:x+2].any():break
            if draft.green[y-1:y+2,x-1:x+2].any():contact=q;break
        terminals.append(contact)
    return {'vertices':vertices.tolist(),'body_box':box,'caption_box':caption,
            'source_candidates':[p['candidate_id'] for p in candidates],
            'numbering':[str(p['number']) for p in candidates],'terminals':terminals}


def recover_single_leds(draft):
    claimed=set()
    for record in draft.records:
        if record['component_id'] in draft.out['components']:continue
        proof=single_led_proof(draft,record)
        if proof is None:continue
        box=proof['caption_box']
        if isolate_caption(draft.rgb,box) is None:continue
        readings=[]
        for scale in (3.,5.):
            if draft.reader is None or draft.ocr_calls>=draft.cfg.get('maximum_caption_views',8):break
            draft.ocr_calls+=1
            readings+=draft.reader(draft.rgb,box,scale,'single_led_identity_caption')
        votes={}
        for token in readings:
            name=str(token.get('text','')).strip().upper()
            if re.fullmatch(r'LED[1-9]\d*',name) and token.get('confidence',0)>=.98 and token.get('view'):
                votes.setdefault(name,set()).add(token['view'])
        if len(votes)!=1 or len(next(iter(votes.values())))<2:
            draft.withheld.append({'rule':'single_filled_led_identity','source_record':record['component_id'],
                                   'reason':'no_unique_multiview_designator','caption_observations':readings})
            continue
        owner=next(iter(votes))
        if owner in draft.out['components'] or any(r.get('designator')==owner for r in draft.records):continue
        a,b,c,d=proof['body_box']
        draft.out['components'][owner]={'type':'led','Name':owner,'value':None,'bbox':[a,draft.h-d,c,draft.h-b]}
        draft.out['pins'][owner]={'pin_'+number:draft.export_pin(point,number)
                                 for number,point in zip(proof['numbering'],proof['terminals'])}
        draft.events.append({'rule':'single_filled_led_identity','owner':owner,
                             'source_record':record['component_id'],'caption_observations':readings,
                             'views':sorted(votes[owner]),**proof})
        claimed.add(owner)
    return claimed
