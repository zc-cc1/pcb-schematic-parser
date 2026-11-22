"""Observe missing resistor contacts from a rectangle and symmetric stems."""
import math
import re

import cv2
import numpy as np


def resistor_contact_proof(rgb,record):
    if (record.get('type')!='resistor' or record.get('detection_confidence',0)<.8
            or not re.fullmatch(r'R[1-9]\d*',str(record.get('designator','')))):
        return None
    colors=np.asarray(rgb).astype(np.int16);r,g,b=[colors[:,:,i] for i in range(3)]
    red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
    green=(g>r+20)&(g>b+20)&(g>60);h,w=red.shape
    a,b,c,d=record['bbox_top_left']
    left,top=max(0,math.floor(a)-4),max(0,math.floor(b)-4)
    right,bottom=min(w,math.ceil(c)+4),min(h,math.ceil(d)+4)
    contours,hierarchy=cv2.findContours(red[top:bottom,left:right].astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    holes=[]
    for contour,info in zip(contours,hierarchy[0]):
        poly=cv2.approxPolyDP(contour,1.5,True)
        x,y,ww,hh=cv2.boundingRect(contour);area=cv2.contourArea(contour)
        if info[3]<0 or not 4<=len(poly)<=6 or not 80<=area<=3000:continue
        if area/max(1,(ww-1)*(hh-1))<.9 or not 1.7<=max(ww,hh)/min(ww,hh)<=6:continue
        x,y=x+left,y+top
        if not (x-2<=(a+c)/2<=x+ww+2 and y-2<=(b+d)/2<=y+hh+2):continue
        holes.append([x-1,y-1,x+ww,y+hh])
    if len(holes)!=1:return None
    body=holes[0];axis=int(body[3]-body[1]>body[2]-body[0]);cross=1-axis
    center=np.array([(body[0]+body[2])/2,(body[1]+body[3])/2])
    thick=cv2.distanceTransform(red.astype('uint8'),cv2.DIST_L2,5)>=3.2
    _,_,stats,centres=cv2.connectedComponentsWithStats(thick.astype('uint8'),8)
    dots=[p for s,p in zip(stats[1:],centres[1:]) if 2<=s[2]<=12 and 2<=s[3]<=12]
    def hit(mask,p):
        x,y=map(round,p)
        return 2<=x<w-2 and 2<=y<h-2 and mask[y-1:y+2,x-1:x+2].any()
    terminals=[];lengths=[];kinds=[]
    for direction,face in [(-1,body[axis]),(1,body[axis+2])]:
        dot_options=[p for p in dots if abs(p[cross]-center[cross])<=3 and 6<=direction*(p[axis]-face)<=45]
        end=None;kind=None
        for distance in range(1,47):
            point=center.copy();point[axis]=face+direction*distance
            if any(math.dist(point,p)<=1.5 for p in dot_options):
                end=min(dot_options,key=lambda p:math.dist(point,p));kind='junction';break
            if hit(green,point) and distance>=6:
                end=point;kind='color_transition';break
            if not hit(red,point):
                point[axis]-=direction
                arms=[]
                for sign in (-1,1):
                    q=point.copy();q[cross]+=sign*6
                    arms.append(hit(red,q) or hit(green,q))
                if distance>=7 and any(arms):end=point;kind='bend'
                break
        if end is None:return None
        terminals.append(list(map(float,end)));lengths.append(abs(float(end[axis]-face)));kinds.append(kind)
    if min(lengths)<6 or max(lengths)>45 or abs(lengths[0]-lengths[1])>4:return None
    return {'body':body,'axis':axis,'terminals':terminals,'stem_lengths':lengths,'end_kinds':kinds}


def recover_resistor_contacts(draft):
    for owner,comp in sorted(draft.out['components'].items()):
        if comp['type']!='r' or draft.out['pins'].get(owner):continue
        record=draft.rmap.get(owner)
        proof=resistor_contact_proof(draft.rgb,record) if record else None
        if proof is None:continue
        draft.out['pins'][owner]={'pin_'+str(index):draft.export_pin(point,str(index))
                                 for index,point in enumerate(proof['terminals'],1)}
        draft.events.append({'rule':'observed_resistor_terminals','owner':owner,
                             'source_record':record['component_id'],'numbering':['1','2'],**proof})
