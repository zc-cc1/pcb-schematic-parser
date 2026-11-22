"""Identify a capacitor axis from two disconnected, parallel red plates."""
import math,re
import cv2
import numpy as np
from .local_symbol_contacts import _masks
from .terminal_identity_recovery import bar_profile


def capacitor_axis_proof(rgb,record):
    if (record.get('type')!='capacitor' or record.get('detection_confidence',0)<.55
            or not re.fullmatch(r'C[1-9]\d*',str(record.get('designator','')))):return None
    red,_,_=_masks(rgb);h,w=red.shape;a,b,c,d=record['bbox_top_left'];found=[]
    left,top=max(0,math.floor(a)-2),max(0,math.floor(b)-2)
    right,bottom=min(w,math.ceil(c)+3),min(h,math.ceil(d)+3)
    _,labels=cv2.connectedComponents(red[top:bottom,left:right].astype('uint8'),8)
    for axis in (0,1):
        shape=bar_profile(red,record['bbox_top_left'],axis,2)
        if not shape:continue
        ids=[]
        for bar in shape['bars']:
            x,y=map(round,bar['center']);ids.append(int(labels[y-top,x-left]))
        if 0 in ids or len(set(ids))!=2:continue
        center=np.mean([v['center'] for v in shape['bars']],axis=0)
        found.append({**shape,'center':center.tolist(),'source_record':record['component_id']})
    return found[0] if len(found)==1 else None
