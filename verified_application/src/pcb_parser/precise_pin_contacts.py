"""Conservative image-only refinement of existing IC pin contact points.

No OCR, field inference, topology edits, model calls, file reads or reference data.
All output edits are confined to existing point x/y fields. Image is RGB top-left;
target coordinates are bottom-left. The caller must bind the precise image and
prediction snapshot with make_binding before invocation.
"""
from copy import deepcopy
import hashlib
import json
import math
import statistics
import numpy as np

IC_TYPES={'box','ic','connector','block','raw_block'}

def make_binding(target,rgb):
    return {'origin':'top_left','prediction_only':True,'size':[rgb.shape[1],rgb.shape[0]],
            'image_rgb_sha256':hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
            'target_sha256':hashlib.sha256(json.dumps(target,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

def _frame(red,box):
    h,w=red.shape;a,b,c,d=box;pos=[]
    if min(c-a,d-b)<40:return None
    for vertical,edge in [(True,a),(False,b),(True,c),(False,d)]:
        rows=[]
        for v in range(round(edge)-22,round(edge)+23):
            if vertical:
                if not 1<=v<w-1:continue
                patch=red[max(0,round(b)+7):min(h,round(d)-7),v-1:v+2];axis=1
            else:
                if not 1<=v<h-1:continue
                patch=red[v-1:v+2,max(0,round(a)+7):min(w,round(c)-7)];axis=0
            if patch.size:rows.append((float(patch.any(axis=axis).mean()),v))
        if not rows:return None
        best=max(s for s,v in rows)
        if best<.94:return None
        plateau=[v for s,v in rows if s>=best-.005]
        if max(plateau)-min(plateau)>6:return None
        pos.append(statistics.median(plateau))
    if not .5*(c-a)<=pos[2]-pos[0]<=1.5*(c-a) or not .5*(d-b)<=pos[3]-pos[1]<=1.5*(d-b):return None
    return pos

def _side(point,box):
    x,y=point;a,b,c,d=box
    ds={'left':a-x,'right':x-c,'top':b-y,'bottom':y-d}
    outside={s:v for s,v in ds.items() if v>0}
    return max(outside,key=outside.get) if outside else min(ds,key=lambda s:abs(ds[s]))

def _evidence(red,green,box,side,point):
    h,w=red.shape;vertical=side in {'top','bottom'};step=-1 if side in {'left','top'} else 1
    edge={'left':box[0],'right':box[2],'top':box[1],'bottom':box[3]}[side]
    tangent=point[0] if vertical else point[1]
    def xy(d,t):return (t,edge+step*d) if vertical else (edge+step*d,t)
    def hit(mask,d,t,radius=0):
        x,y=xy(d,t);x,y=math.floor(x+.5),math.floor(y+.5)
        if x<radius or y<radius or x>=w-radius or y>=h-radius:return False
        return bool(mask[y-radius:y+radius+1,x-radius:x+radius+1].any())
    def connected(d):
        # Exact centerline occupancy: dilation must never bridge a white gap.
        return all(hit(red,q,t) or hit(green,q,t) for q in range(0,math.ceil(d)+1))
    # A body-connected thin stem must persist for at least twelve pixels.
    rows=[]
    lo=box[0]+6 if vertical else box[1]+6;hi=box[2]-6 if vertical else box[3]-6
    for t in range(round(tangent)-12,round(tangent)+13):
        if not lo<=t<=hi:continue
        if sum(hit(red,d,t) for d in range(3,15))>=11:rows.append(t)
    groups=[]
    for t in rows:
        if not groups or t-groups[-1][-1]>1:groups.append([])
        groups[-1].append(t)
    groups=[g for g in groups if len(g)<=5]
    if len(groups)!=1:return None
    t=statistics.median(groups[0])
    red_run=-1
    for q in range(0,65):
        if not hit(red,q,t):break
        red_run=q
    # Inspect the complete local profile, not a fixed-width crossing window.
    # Once green is seen, any later red on this same stem axis is ambiguous.
    # This also rejects thick green overlays and distant symbol continuations.
    green_distances=[q for q in range(0,65) if hit(green,q,t)]
    if green_distances:
        first_green=min(green_distances)
        if any(hit(red,q,t) for q in range(first_green+1,65)):
            return {'point':None,'kind':'green_crossing','tangent':t,'edge':edge,'red_run':red_run}
        # Bilateral green transverse arms without a red disc are a crossing,
        # even if the overlay obscures the red for the rest of the scan.
        if any(hit(green,q,t-6,1) and hit(green,q,t+6,1) for q in green_distances):
            return {'point':None,'kind':'green_crossing','tangent':t,'edge':edge,'red_run':red_run}
    # Red junction discs hide the green intersection. Require green wire on
    # both transverse arms and a continuous red stem through the disc center.
    junction=[]
    for d in range(10,65):
        if sum(hit(green,d,t+k,1) for k in [-11,-9,9,11])<4 or not connected(d):continue
        if sum(hit(red,q,t,1) for q in range(3,d+1))<.94*max(1,d-2):continue
        # Disc has visibly wider red support than the thin stem.
        if not (hit(red,d,t-3) and hit(red,d,t+3)):continue
        junction.append(d)
    if junction and max(junction)-min(junction)<=8:
        d=statistics.median(junction);p=list(xy(d,t));axis=1 if vertical else 0
        p[axis]=math.floor(p[axis]+.5)
        return {'point':p,'kind':'red_disc_green_center','tangent':t,'length':d,'edge':edge}
    # A green crossing with red resuming beyond it is ambiguous. It must
    # never become an anchor or borrow another lead's contact length.
    for d in range(12,65):
        if hit(green,d,t) and hit(red,d-4,t) and hit(red,d+4,t):
            return {'point':None,'kind':'green_crossing','tangent':t,'edge':edge,'obstacle_distance':d,'red_run':red_run}
    # A one-sided green elbow is also an explicit contact, provided the red
    # stem terminates there. Green crossings with red beyond are rejected.
    elbows=[]
    for d in range(12,65):
        branch=max(sum(hit(green,d,t+sgn*k,1) for k in range(3,10)) for sgn in [-1,1])
        if branch<6 or not connected(d):continue
        if sum(hit(red,q,t,1) for q in range(3,d-1))<.94*max(1,d-4):continue
        if any(hit(red,q,t) for q in range(d+4,d+9)):continue
        if not any(hit(red,q,t,1) for q in range(d-3,d+1)):continue
        elbows.append(d)
    if elbows and max(elbows)-min(elbows)<=5:
        d=statistics.median(elbows);return {'point':list(xy(d,t)),'kind':'red_green_contact','tangent':t,'length':d,'edge':edge}
    # Green continuation must follow the red endpoint; no red continuation
    # beyond a crossing is allowed. A missing green transition is left open.
    for d in range(12,65):
        if not hit(green,d,t,1) or not connected(d):continue
        if sum(hit(green,q,t,1) for q in range(d,d+6))<5:continue
        if sum(hit(red,q,t,1) for q in range(3,d))<.94*max(1,d-3):continue
        if any(hit(red,q,t) for q in range(d+4,d+9)):return {'point':None,'kind':'green_crossing','tangent':t,'edge':edge,'obstacle_distance':d,'red_run':red_run}
        return {'point':list(xy(d,t)),'kind':'red_green_contact','tangent':t,'length':d,'edge':edge}
    if any(hit(green,d,t) for d in range(12,65)):
        return {'point':None,'kind':'disconnected_or_ambiguous_green','tangent':t,'edge':edge}

    return {'point':None,'kind':'red_continuation','tangent':t,'edge':edge,'red_run':red_run}

def refine_pin_contacts(target,rgb,*,binding,enabled=False,max_pins=500):
    """Return a deep copy and audit. Reject stale or cross-image evidence.

    Limits: unique body frame; one thin body-connected stem within 12 px;
    movement at most 16 px; visible contact or >=3 independent same-side
    contacts agreeing within 3 px. Edits within the 5 px uncertainty radius are deliberately retained.
    Budget exhaustion withholds the entire pass (no order-dependent partial edits).
    """
    out=deepcopy(target);audit={'policy':'precise_existing_pin_contacts_v1','changes':[], 'rejected':[], 'frames':{},'enabled':False}
    if enabled is not True:return out,audit
    if type(max_pins) is not int or not 0<=max_pins<=4096:audit['rejected'].append('invalid_pin_budget');return out,audit
    if not isinstance(rgb,np.ndarray) or rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:return out,audit
    if binding!=make_binding(target,rgb):audit['rejected'].append('unbound_or_stale_evidence');return out,audit
    n=sum(len(ps) for c,ps in target.get('pins',{}).items() if str(target.get('components',{}).get(c,{}).get('type','')).strip().lower() in IC_TYPES)
    if n>max_pins:audit['rejected'].append('pin_budget');return out,audit
    audit['binding']=make_binding(target,rgb);audit['pin_budget']=max_pins;audit['pins_considered']=n
    audit['enabled']=True;h,w=rgb.shape[:2];colors=rgb.astype(np.int16);r,g,b=[colors[:,:,i] for i in range(3)]
    red=(r>g+35)&(r>b+35)&(g<150)&(b<150);green=(g>r+20)&(g>b+20)&(g>60)
    for owner,component in sorted(target.get('components',{}).items()):
        if str(component.get('type','')).strip().lower() not in IC_TYPES:continue
        box=component.get('bbox',[])
        if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box):continue
        a,b,c,d=box;frame=_frame(red,[a,h-d,c,h-b])
        if frame is None:continue
        audit['frames'][owner]=frame;rows=[]
        for key,pin in sorted(target.get('pins',{}).get(owner,{}).items()):
            p=pin.get('point',{});x,y=p.get('x'),p.get('y')
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [x,y]):continue
            pt=[x,h-y];side=_side(pt,frame)
            edge={'left':frame[0],'right':frame[2],'top':frame[1],'bottom':frame[3]}[side]
            distance=abs((pt[0] if side in {'left','right'} else pt[1])-edge)
            if not 3<=distance<=75:continue
            evidence=_evidence(red,green,frame,side,pt)
            if evidence:rows.append({'key':key,'before':pt,'side':side,**evidence})
        for row in rows:
            new=row['point'];kind=row['kind'];side=row['side']
            if new is None:
                if kind!='red_continuation':continue
                anchors=[q for q in rows if q['side']==side and q['point'] is not None and q['kind']=='red_green_contact']
                if len(anchors)<3:continue
                lengths=[q['length'] for q in anchors];length=statistics.median(lengths)
                if max(lengths)-min(lengths)>3 or length<12 or row.get('red_run',0)<math.ceil(length):continue
                if row.get('obstacle_distance',float('inf'))<=length+4:continue
                # A continuation can borrow normal length, never another row.
                t=row['tangent'];edge=row['edge'];step=-1 if side in {'left','top'} else 1
                new=[edge+step*length,t] if side in {'left','right'} else [t,edge+step*length]
                kind='same_side_visible_contact_consensus'
            delta=math.dist(row['before'],new)
            if not 5<delta<=16:continue
            if kind=='red_green_contact':
                tangent_delta=abs(new[1]-row['before'][1]) if side in {'left','right'} else abs(new[0]-row['before'][0])
                if tangent_delta<=4:continue
            if kind=='same_side_visible_contact_consensus' and delta<=6:continue
            # A correction must stay much closer to its original pin than any
            # other exported pin in that component; preserve pin ownership.
            other=[q['before'] for q in rows if q['key']!=row['key']]
            if other and min(math.dist(q,new) for q in other)<=delta+3:continue
            before=out['pins'][owner][row['key']]['point'];after={'x':float(new[0]),'y':float(h-new[1])}
            audit['changes'].append({'owner':owner,'key':row['key'],'kind':kind,'before':dict(before),'after':after,'distance':delta})
            before.update(after)
    return out,audit








