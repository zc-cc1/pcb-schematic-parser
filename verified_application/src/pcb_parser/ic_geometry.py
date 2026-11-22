"""Image-supported rectangular IC frames, independent of OCR and references."""
from copy import deepcopy
import math

import cv2
import numpy as np


POLICY = 'ic_observed_four_sided_rectangle_v1'


def _iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1.,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def _segments(mask, vertical, minimum, thickness):
    count,_,stats,_ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    rows=[]
    for x,y,w,h,area in stats[1:]:
        length = h if vertical else w
        stroke = w if vertical else h
        if length < minimum or stroke > thickness:
            continue
        rows.append({'position':float(x+(w-1)/2 if vertical else y+(h-1)/2),
                     'start':int(y if vertical else x),'end':int(y+h-1 if vertical else x+w-1),
                     'length':int(length)})
    return rows


def locate_ic_rectangle(rgb, box):
    """Find four supported frame sides near the detector's proposal, or abstain."""
    height,width=rgb.shape[:2]
    if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError('Invalid IC detection box')
    x1,y1,x2,y2=map(float,box);w,h=x2-x1,y2-y1
    # Tiny generic "box" symbols can include pin stubs in their annotation.
    # Their internal rectangle is not enough evidence to redefine the box.
    if min(w,h)<40:
        return None,{'reason':'symbol_too_small_for_four_side_evidence'}
    padx,pady=max(10.,.25*w),max(10.,.25*h)
    ox,oy=max(0,math.floor(x1-padx)),max(0,math.floor(y1-pady))
    ex,ey=min(width,math.ceil(x2+padx)),min(height,math.ceil(y2+pady))
    if ex-ox<12 or ey-oy<12:
        return None,{'reason':'insufficient_image_context'}
    ink=(rgb[oy:ey,ox:ex].min(axis=2)<180).astype(np.uint8)
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(7,round(.35*h)),1),np.uint8))
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(7,round(.35*w))),np.uint8))
    vs=_segments(vertical,True,.35*h,max(4,.05*w));hs=_segments(horizontal,False,.35*w,max(4,.05*h))
    def near(rows,edge,span,offset):
        candidates=[r for r in rows if abs(r['position']+offset-edge)<=max(10.,.35*span)]
        return sorted(candidates,key=lambda r:(abs(r['position']+offset-edge),-r['length']))[:6]
    left,right=near(vs,x1,w,ox),near(vs,x2,w,ox)
    top,bottom=near(hs,y1,h,oy),near(hs,y2,h,oy)
    base={'roi_top_left':[ox,oy,ex,ey],'side_candidate_counts':dict(left=len(left),right=len(right),top=len(top),bottom=len(bottom))}
    if not all([left,right,top,bottom]):
        return None,{**base,'reason':'four_straight_sides_not_observed'}
    # A one-pixel normal tolerance handles antialiasing and two-pixel strokes;
    # support must still extend along almost the full length of every side.
    vmask=cv2.dilate(ink,np.ones((1,3),np.uint8))
    hmask=cv2.dilate(ink,np.ones((3,1),np.uint8))
    vint=np.pad(vmask.cumsum(axis=0,dtype=np.int32),((1,0),(0,0)))
    hint=np.pad(hmask.cumsum(axis=1,dtype=np.int32),((0,0),(1,0)))
    candidates=[]
    for a in left:
        for b in right:
            l,r=round(a['position']),round(b['position'])
            nw=r-l
            if not .28*w<=nw<=1.4*w:continue
            for c in top:
                for d in bottom:
                    t,bot=round(c['position']),round(d['position'])
                    nh=bot-t
                    if not .28*h<=nh<=1.4*h:continue
                    q=[float(l+ox),float(t+oy),float(r+ox),float(bot+oy)]
                    if _iou(q,box)<.25:continue
                    support=[float(vint[bot+1,l]-vint[t,l])/(nh+1),
                             float(vint[bot+1,r]-vint[t,r])/(nh+1),
                             float(hint[t,r+1]-hint[t,l])/(nw+1),
                             float(hint[bot,r+1]-hint[bot,l])/(nw+1)]
                    if min(support)<.97:continue
                    def span_agrees(segment,start,end):
                        overlap=max(0,min(segment['end'],end)-max(segment['start'],start)+1)
                        return overlap/max(end-start+1,segment['length'])>=.85
                    if not (span_agrees(a,t,bot) and span_agrees(b,t,bot)
                            and span_agrees(c,l,r) and span_agrees(d,l,r)):
                        continue
                    if not all(ink[max(0,y-2):y+3,max(0,x-2):x+3].any() for x,y in [(l,t),(r,t),(r,bot),(l,bot)]):continue
                    movement=(abs(q[0]-x1)/w+abs(q[2]-x2)/w+abs(q[1]-y1)/h+abs(q[3]-y2)/h)/4
                    shift=math.hypot(((q[0]+q[2])-(x1+x2))/(2*w),((q[1]+q[3])-(y1+y2))/(2*h))
                    score=.6*min(support)+.4*sum(support)/4-.10*movement-.06*shift
                    candidates.append({'bbox_top_left':q,'support':support,'score':score})
    candidates.sort(key=lambda c:-c['score'])
    if not candidates:
        return None,{**base,'reason':'rectangle_sides_incomplete'}
    best=candidates[0]
    distinct=next((c for c in candidates[1:] if max(abs(a-b) for a,b in zip(c['bbox_top_left'],best['bbox_top_left']))>4),None)
    audit={**base,'best':best,'alternative':distinct,'candidate_count':len(candidates)}
    if distinct and best['score']-distinct['score']<.03:
        return None,{**audit,'reason':'competing_rectangular_frames'}
    return best['bbox_top_left'],{**audit,'reason':'four_sided_frame_observed'}


def refine_ic_boxes(rgb,payload):
    """Modify Target IC bbox only; preserve names, identities, pins and nets."""
    height,width=rgb.shape[:2]
    result=deepcopy(payload);entries=[]
    for key,c in result['components'].items():
        if c['type'] not in {'ic','box'}:continue
        before=c['bbox'][:];b=[before[0],height-before[3],before[2],height-before[1]]
        found,evidence=locate_ic_rectangle(rgb,b)
        if found is not None:
            a,t,r,bot=found;c['bbox']=[a,height-bot,r,height-t]
        entries.append({'component_id':key,'before_bbox_bottom_left':before,
                        'after_bbox_bottom_left':c['bbox'][:],
                        'changed':any(abs(a-b)>1e-6 for a,b in zip(before,c['bbox'])),
                        'evidence':evidence})
    return result,{'policy':POLICY,'size':[width,height],'entries':entries,
                   'changed_IC_boxes':sum(e['changed'] for e in entries),
                   'input_contract':'Current PNG and detector boxes only; no reference coordinates or part lookup.'}
