"""Observe numeric voltage captions next to a model-confirmed supply symbol.

No IC pin name, other symbol's rail value, page or reference label is used to
infer voltage. OCR must independently read the local blue caption three times.
"""
import re

import numpy as np

from .pin_numeric_connector import isolate

NAMED_SUPPLIES={'VCC','VDD','VBAT','VBUS','VIN','VOUT'}


def numeric_voltage(text):
    value=re.sub(r'\s+','',str(text)).replace('−','-').replace('＋','+').upper()
    return value if re.fullmatch(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)V',value) else None


def outward_wire(rgb,box,point):
    """A source terminal must lead outward continuously to an observed green wire."""
    h,w=rgb.shape[:2];a,b,c,d=box;cx,cy=(a+c)/2,(b+d)/2
    delta=[point[0]-cx,point[1]-cy];axis=int(abs(delta[1])>abs(delta[0]))
    if abs(delta[axis])<2:return None
    direction=1 if delta[axis]>0 else -1
    colors=rgb.astype(np.int16);r,g,blue=[colors[:,:,i] for i in range(3)]
    red=(r>g+35)&(r>blue+35)&(g<150)&(blue<150)
    green=(g>r+20)&(g>blue+20)&(g>60)
    conductive=red|green
    # A supply T must have one crossbar behind the terminal, rather than just
    # an arbitrary red lead with voltage text nearby.
    transverse=1-axis;caps=[]
    for coordinate in range(max(1,round(box[axis])),min((w,h)[axis]-1,round(box[axis+2])+1)):
        if not 1 < direction*(point[axis]-coordinate) <= 24:continue
        p=[round(point[0]),round(point[1])];p[axis]=coordinate
        if not red[max(0,p[1]-1):min(h,p[1]+2),max(0,p[0]-1):min(w,p[0]+2)].any():continue
        lengths=[]
        for sign in (-1,1):
            length=0
            for step in range(1,25):
                q=p.copy();q[transverse]+=sign*step;x,y=q
                if not(1<=x<w-1 and 1<=y<h-1) or not red[y-1:y+2,x-1:x+2].any():break
                length=step
            lengths.append(length)
        if min(lengths)>=5 and abs(lengths[0]-lengths[1])<=5:caps.append(coordinate)
    groups=[]
    for value in caps:
        if not groups or value-groups[-1][-1]>2:groups.append([])
        groups[-1].append(value)
    if len(groups)!=1:return None
    cap=float(np.median(groups[0]))
    for distance in range(36):
        p=list(point);p[axis]+=direction*distance;x,y=map(round,p)
        if not(1<=x<w-1 and 1<=y<h-1):return None
        if not conductive[y-1:y+2,x-1:x+2].any():return None
        if green[y-1:y+2,x-1:x+2].any():
            return {'source_point':list(point),'green_point':[x,y],
                    'distance_px':distance,'axis':axis,'direction':direction,'cap_coordinate':cap}
    return None


def read_numeric_supply(rgb,box,point,words,*,reader,remaining_budget):
    """Return evidence and call count. Failure never supplies an inferred value."""
    result={'status':'withheld','calls':0,'observations':[]}
    a,b,c,d=box;cx,cy=(a+c)/2,(b+d)/2
    nearby=[t for t in words if t.get('confidence',0)>=.9 and
            abs((t['bbox'][0]+t['bbox'][2])/2-cx)<45 and abs((t['bbox'][1]+t['bbox'][3])/2-cy)<25]
    if any(str(t['text']).strip().upper() in NAMED_SUPPLIES for t in nearby):
        return {**result,'reason':'named_supply_handled_by_existing_rule'}
    seeds=[{**t,'text':numeric_voltage(t['text'])} for t in nearby if numeric_voltage(t['text'])]
    identities={t['text'] for t in seeds}
    if len(identities)>1:return {**result,'reason':'conflicting_voltage_captions'}
    path=outward_wire(rgb,box,point)
    if path is None:return {**result,'reason':'no_continuous_outward_wire'}
    isolated=isolate(rgb,[a-18,b-12,c+18,d+12],'blue')
    if isolated is None:return {**result,'reason':'no_isolated_blue_caption'}
    patch,caption_box=isolated
    result.update(observations=seeds,wire=path,caption_box=caption_box)
    def vote():
        support={}
        for token in result['observations']:
            name=numeric_voltage(token['text'])
            if name and token.get('confidence',0)>=.9 and token.get('view'):
                support.setdefault(name,set()).add(token['view'])
        if len(support)>1:return 'conflict',None
        if support:
            name=next(iter(support))
            if len(support[name])>=3:return 'accepted',name
        return 'insufficient',None
    orientation=90 if caption_box[3]-caption_box[1]>caption_box[2]-caption_box[0] else 0
    # The view token is geometry-neutral. The reader identifies the actual crop
    # hash, rotation and scale, so two symbols cannot share invented OCR votes.
    for rotation in (orientation,(orientation+180)%360,(orientation+90)%360):
        for scale in (3.,5.,4.):
            status,name=vote()
            if status=='conflict':return {**result,'reason':'conflicting_voltage_captions'}
            if status=='accepted':return {**result,'status':'accepted','caption':name}
            if reader is None or result['calls']>=remaining_budget:
                return {**result,'reason':'insufficient_independent_caption_views'}
            result['calls']+=1
            values=reader(rgb,caption_box,scale,'numeric_supply_caption',rotation=rotation)
            for t in values:
                normalized=numeric_voltage(t.get('text',''))
                if normalized and t.get('confidence',0)>=.9 and t.get('view'):
                    result['observations'].append({**t,'text':normalized})
                elif str(t.get('text','')).strip().upper() in NAMED_SUPPLIES and t.get('confidence',0)>=.9:
                    return {**result,'reason':'named_supply_not_a_numeric_voltage'}
    status,name=vote()
    if status=='accepted':return {**result,'status':'accepted','caption':name}
    return {**result,'reason':'conflicting_voltage_captions' if status=='conflict' else 'insufficient_independent_caption_views'}
