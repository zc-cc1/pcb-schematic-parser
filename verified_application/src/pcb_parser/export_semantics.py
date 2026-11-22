"""Normalize final component semantics using current-image evidence only.

Run after all topology recovery: those stages still consume legacy port captions.
The legacy profile is an explicit annotation convention, never selected by case.
"""
from copy import deepcopy
import math
import cv2
import numpy as np
from .component_text import gap, center, inside
from .ic_name_selection import category, _same_glyph
from .pin_numeric_connector import observations

MODEL_NAME_TYPES = {'led', 'switch', 'ic', 'box', 'connector'}
PORT_TYPES = {'net_input', 'net_output', 'net_bidirection'}


def _box(component, height):
    a,b,c,d = component['bbox']
    if not all(math.isfinite(v) for v in (a,b,c,d)) or a>=c or b>=d:
        raise ValueError('invalid_component_box')
    return [a,height-d,c,height-b]


def _port_direction(red, box, pins, height):
    if len(pins)!=1:
        return None
    p=next(iter(pins.values()))['point'];point=np.array([p['x'],height-p['y']],dtype=float)
    if not np.isfinite(point).all():
        return None
    h,w=red.shape;a,b,c,d=box
    left,top=max(0,math.floor(a)-12),max(0,math.floor(b)-12)
    right,bottom=min(w,math.ceil(c)+13),min(h,math.ceil(d)+13)
    if left>=right or top>=bottom:
        return None
    patch=cv2.morphologyEx(red[top:bottom,left:right],cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(patch,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    shapes=[]
    for contour in contours:
        if not 180<=cv2.contourArea(contour)<=8000:
            continue
        vertices=cv2.approxPolyDP(contour,3.,True).reshape(-1,2)+[left,top]
        if len(vertices)!=5 or not cv2.isContourConvex(vertices.astype(np.int32)):
            continue
        low,high=vertices.min(0),vertices.max(0)
        axis=int(high[1]-low[1]>high[0]-low[0]);cross=1-axis
        if not 1.4<=(high[axis]-low[axis])/max(1,high[cross]-low[cross])<=5.8:
            continue
        minimum=sum(abs(vertices[:,axis]-low[axis])<=3)
        maximum=sum(abs(vertices[:,axis]-high[axis])<=3)
        if sorted([minimum,maximum])!=[1,2]:
            continue
        # Reject nearby flags; the polygon must belong to this component box.
        inter=max(0,min(c,high[0])-max(a,low[0]))*max(0,min(d,high[1])-max(b,low[1]))
        if inter/max(1,float(np.prod(high-low)))<.65:
            continue
        mid=(low+high)/2
        if abs(point[cross]-mid[cross])>max(4.,(high[cross]-low[cross])*.3):
            continue
        if point[axis]<=low[axis]+4:
            side=-1;distance=low[axis]-point[axis]
        elif point[axis]>=high[axis]-4:
            side=1;distance=point[axis]-high[axis]
        else:
            continue
        if distance>max(50.,high[axis]-low[axis]):
            continue
        tip=-1 if minimum==1 else 1
        shapes.append({'type':'net_input' if side==tip else 'net_output',
                       'vertices':vertices.tolist(),'terminal':point.tolist()})
    if not shapes or len({p['type'] for p in shapes})!=1:
        return None
    # Inner and outer contours may describe one flag, but two physical flags
    # must never constitute evidence for a single component.
    if any(not _same_glyph([*np.min(p['vertices'],axis=0),*np.max(p['vertices'],axis=0)],
                           [*np.min(shapes[0]['vertices'],axis=0),*np.max(shapes[0]['vertices'],axis=0)]) for p in shapes[1:]):
        return None
    return shapes[0]


def _switch_model(owner,box,words,claimed,component_boxes):
    refs=[t for t in words if t['text']==owner and t['confidence']>=.98 and gap(t['bbox'],box)<=60]
    if not refs or len({t['view'] for t in refs})<2:
        return None
    # Repeated OCR of one glyph is support; two distinct printed references
    # with the same identity are ambiguous ownership.
    if any(not _same_glyph(t['bbox'],refs[0]['bbox']) for t in refs[1:]):
        return None
    eligible=[]
    for token in words:
        if token['confidence']<.98 or category(token['text'],owner)!='model':
            continue
        tb=token['bbox']
        if gap(tb,box)>65 or any(other!=owner and _same_glyph(tb,b) for other,b in claimed):
            continue
        if any(other!=owner and inside(center(tb),b,2) for other,b in component_boxes.items()):
            continue
        # Captions form a local text stack with the printed reference, on any
        # side of the symbol. This does not infer a model from its spelling.
        if not any(abs(tb[0]-r['bbox'][0])<=max(8.,(r['bbox'][3]-r['bbox'][1])*.7)
                   and 3<=abs(center(tb)[1]-center(r['bbox'])[1])<=45 for r in refs):
            continue
        # An unclaimed caption may still belong to another printed reference.
        # Refuse it if a closer/equally close reference shares this text stack.
        own_distance=min(abs(center(tb)[1]-center(r['bbox'])[1]) for r in refs)
        competitors=[t for t in words if t['text']!=owner and t['confidence']>=.98
                     and category(t['text'])=='excluded_other_reference'
                     and abs(tb[0]-t['bbox'][0])<=max(8.,(t['bbox'][3]-t['bbox'][1])*.7)
                     and abs(center(tb)[1]-center(t['bbox'])[1])<=own_distance+3]
        if competitors:
            continue
        eligible.append(token)
    if not eligible or len({t['text'] for t in eligible})!=1:
        return None
    if len({t['view'] for t in eligible})<2:
        return None
    if any(not _same_glyph(t['bbox'],eligible[0]['bbox']) for t in eligible[1:]):
        return None
    return max(eligible,key=lambda t:t['confidence'])


def normalize_export_semantics(target,rgb,records,ocr,image_sha256,profile='observed'):
    if profile not in {'observed','legacy'}:
        raise ValueError('unknown_export_semantics_profile')
    out=deepcopy(target)
    audit={'profile':profile,'changes':[],'skipped':[],'reference_read':False}
    if profile=='legacy':
        return out,audit
    h,w=rgb.shape[:2]
    for document in (records,ocr):
        if (document.get('image_sha256')!=image_sha256 or not isinstance(image_sha256,str) or len(image_sha256)!=64
                or document.get('size')!=[w,h] or document.get('coordinate_origin')!='top_left'
                or document.get('prediction_only') is not True):
            raise ValueError('export_semantics_evidence_differs_from_image')
    words=[t for t in observations(ocr) if math.isfinite(t['confidence']) and 0<=t['confidence']<=1]
    claimed=[(r['component_id'],v['bbox']) for r in records.get('records',[])
             for v in r.get('fields',{}).values() if isinstance(v,dict) and v.get('bbox')]
    boxes={}
    for owner,component in out['components'].items():
        try:boxes[owner]=_box(component,h)
        except (KeyError,TypeError,ValueError):pass
    r,g,b=[rgb[:,:,i].astype(np.int16) for i in range(3)]
    red=((r>g+35)&(r>b+35)&(g<150)&(b<150)).astype(np.uint8)
    proposals={}
    for owner,component in out['components'].items():
        if owner not in boxes:continue
        before=deepcopy(component)
        try:
            if component['type'] in PORT_TYPES:
                proof=_port_direction(red,boxes[owner],out['pins'].get(owner,{}),h)
                if proof:
                    component.update(type=proof['type'],value=None)
                    if component!=before:audit['changes'].append({'owner':owner,'rule':'flag_tip_relative_to_terminal','before':before,'after':deepcopy(component),'evidence':proof})
            elif component['type'] in MODEL_NAME_TYPES:
                if component.get('Name')==owner:
                    component['Name']=None
                if component['type']=='switch' and component.get('Name') is None:
                    proposal=_switch_model(owner,boxes[owner],words,claimed,boxes)
                    if proposal:proposals[owner]=proposal
                if component!=before:audit['changes'].append({'owner':owner,'rule':'designator_is_not_model','before':before,'after':deepcopy(component)})
        except (KeyError,TypeError,ValueError,cv2.error) as error:
            out['components'][owner]=before
            audit['skipped'].append({'owner':owner,'reason':str(error)})
    for owner,proposal in proposals.items():
        if any(other!=owner and _same_glyph(proposal['bbox'],p['bbox']) for other,p in proposals.items()):
            audit['skipped'].append({'owner':owner,'reason':'model_caption_has_multiple_owners'});continue
        out['components'][owner]['Name']=proposal['text']
        audit['changes'].append({'owner':owner,'rule':'observed_switch_model','evidence':proposal})
    return out,audit
