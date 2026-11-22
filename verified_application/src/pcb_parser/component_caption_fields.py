"""Recover missing component fields from owned, independently read captions.

Only Name/value may change. Pin, net and geometry payloads are immutable.
"""
from copy import deepcopy
from collections import defaultdict
import math,re
import cv2
import numpy as np
from .pin_numeric_connector import observations
from .numeric_power_caption import outward_wire,NAMED_SUPPLIES

DEFAULT={'enabled':False,'maximum_changes':24,'maximum_ocr_calls':32}
DESIGNATOR=re.compile(r'(?:U|USB|J|H|P|R|C|L|LED|D|SW|S|Q|CN|TP|X|Y|F|K|VR|RN)\d+[A-Z]?',re.I)


def caption_patch(rgb,box):
    h,w=rgb.shape[:2];a,b,c,d=map(round,box);a,b,c,d=max(0,a),max(0,b),min(w,c),min(h,d)
    if a>=c or b>=d:return None
    patch=rgb[b:d,a:c].astype(np.int16);r,g,blue=[patch[:,:,i] for i in range(3)]
    mask=(blue>r+5)&(blue>.7*g)&(r<150)&(g<180)
    yy,xx=np.where(mask)
    if not len(xx):return None
    left,right,top,bottom=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    if not 5<=bottom-top<=45 or not 6<=right-left<=250:return None
    ink=np.full((bottom-top,right-left,3),255,np.uint8);ink[mask[top:bottom,left:right]]=0
    return cv2.copyMakeBorder(ink,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))


def literal_caption_patch(rgb,box):
    """Preserve faint punctuation; only the red symbol is removed."""
    h,w=rgb.shape[:2];a,b,c,d=map(round,box);a,b,c,d=max(0,a),max(0,b),min(w,c),min(h,d)
    if not (6<=c-a<=250 and 5<=d-b<=45):return None
    patch=rgb[b:d,a:c].copy();r,g,blue=[patch[:,:,i].astype(np.int16) for i in range(3)]
    patch[(r>g+35)&(r>blue+35)]=255
    return patch


def _anchor(record,words):
    owner=record['component_id'];a,b,c,d=record['bbox_top_left']
    tokens=[t for t in words if t['text']==owner and t['confidence']>=.98 and a-50<=(t['bbox'][0]+t['bbox'][2])/2<=c+50
            and b-60<=(t['bbox'][1]+t['bbox'][3])/2<=d+35]
    if len({t['view'] for t in tokens})<2:return None
    return np.median([t['bbox'] for t in tokens],axis=0).tolist()


def _model_text(text):
    value=re.sub(r'\s+',' ',str(text)).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._()+/°\-]{2,60}',value):return None
    if not re.search(r'[A-Za-z]',value) or not re.search(r'\d',value) or DESIGNATOR.fullmatch(value):return None
    if re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?\s*[numµpkGM]?(?:V|A|H|F|W|R)',value,re.I):return None
    return value


def _position(target,record,anchor,box,height):
    comp=target['components'].get(record['component_id'],{})
    if comp.get('type') not in {'box','ic'}:return None
    a,b,c,d=comp['bbox'];b,d=height-d,height-b;x1,y1,x2,y2=box
    if anchor is not None and abs(x1-anchor[0])>40:return None
    below=d-8<=y1<=d+45 and a-45<=x1<=c+45
    right=c-8<=x1<=c+45 and b-35<=y1<=d+35 and (anchor is None or anchor[1]-5<=y1<=anchor[3]+40)
    if not below and not right:return None
    return min(abs(y1-d) if below else 1e5,abs(x1-c) if right else 1e5)+(.25*abs(x1-anchor[0]) if anchor is not None else 0)


def _inside_other_body(target,owner,box,height):
    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
    return any(other!=owner and c.get('type') in {'box','ic'} and c['bbox'][0]<=x<=c['bbox'][2]
               and height-c['bbox'][3]<=y<=height-c['bbox'][1] for other,c in target['components'].items())


def _supply_caption_owned(record,records,box):
    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2;ranks=[]
    for r in records:
        if r.get('type') not in {'vcc','vbus_in','vbus_out'}:continue
        a,b,c,d=r['bbox_top_left'];dx,dy=x-(a+c)/2,y-(b+d)/2
        if abs(dx)<45 and abs(dy)<30:ranks.append((math.hypot(dx,dy),r['component_id']))
    ranks.sort()
    return bool(ranks and ranks[0][1]==record['component_id'] and (len(ranks)==1 or ranks[1][0]-ranks[0][0]>=12))


def model_name_proof(target,record,words,height,records):
    owner=record['component_id'];comp=target['components'].get(owner,{})
    if record.get('type')!='ic' or record.get('detection_confidence',0)<.8 or comp.get('type') not in {'box','ic'}:return None
    anchor=_anchor(record,words)
    if anchor is None:return None
    current=comp.get('Name');votes=defaultdict(list)
    for t in words:
        text=_model_text(t['text'])
        if text is None or t['confidence']<.95 or _position(target,record,anchor,t['bbox'],height) is None:continue
        if _inside_other_body(target,owner,t['bbox'],height):continue
        if current not in {None,'',owner} and not (text.startswith(str(current)) and len(text)>len(str(current))):continue
        ranks=[]
        for r in records:
            if r.get('type')!='ic' or r['component_id'] not in target['components']:continue
            aa=_anchor(r,words)
            score=_position(target,r,aa,t['bbox'],height)
            if score is not None:ranks.append((score,r['component_id']))
        ranks.sort()
        if not ranks or ranks[0][1]!=owner or len(ranks)>1 and ranks[1][0]-ranks[0][0]<15:continue
        votes[text].append(t)
    qualified={name:ts for name,ts in votes.items() if len({t['view'] for t in ts})>=3}
    if not qualified:return None
    longest=max(qualified,key=len)
    if any(not longest.startswith(name) for name in qualified):return None
    tokens=qualified[longest]
    if len(qualified)>1 and max(t['confidence'] for t in tokens)<.99:return None
    return {'text':longest,'caption_box':np.median([t['bbox'] for t in tokens],axis=0).tolist(),
            'source_record':owner,'designator_box':anchor,'observations':tokens,
            'needs_local_confirmation':max(t['confidence'] for t in tokens)<.98}


def _read_vote(rgb,box,reader,counter,maximum,*,scales=(3.,5.),minimum=.95):
    if reader is None or counter[0]+len(scales)>maximum:return None,[]
    values=[]
    for scale in scales:
        counter[0]+=1;values.extend(reader(rgb,box,scale,'component_field_caption'))
    votes=defaultdict(set)
    for t in values:
        text=re.sub(r'\s+',' ',str(t.get('text',''))).strip()
        if t.get('confidence',0)>=minimum and t.get('view'):votes[text].add(t['view'])
    names=[name for name,views in votes.items() if len(views)>=len(scales)]
    return (names[0] if len(names)==1 and len(votes)==1 else None),values


def recover_component_caption_fields(target,rgb,graph,records,pins,ocr,image_sha256,*,reader=None,cfg=None):
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_component_caption_setting')
    if any(type(config[k]) is not int or not 0<=config[k]<=128 for k in DEFAULT if k!='enabled'):raise ValueError('invalid_component_caption_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    for document in (graph,records,pins,ocr):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('coordinate_origin')!='top_left' or document.get('prediction_only') is not True):
            raise ValueError('component_caption_evidence_differs_from_image')
    words=observations(ocr);events=[];withheld=[];calls=[0]
    for record in records['records']:
        owner=record['component_id'];comp=out['components'].get(owner)
        if comp is None:continue
        proof=model_name_proof(out,record,words,h,records['records'])
        if proof:
            if proof['needs_local_confirmation']:
                a,b,c,d=proof['caption_box'];text,reads=_read_vote(rgb,[a-3,b-3,c+3,d+3],reader,calls,config['maximum_ocr_calls'])
                if text!=proof['text']:
                    withheld.append({'owner':owner,'reason':'model_crop_not_confirmed','observations':reads});continue
                proof['local_observations']=reads
            before=comp.get('Name');comp['Name']=proof['text'];events.append({'rule':'owned_model_caption','owner':owner,'before':before,'after':comp['Name'],**proof})
        current=comp.get('value')
        if (isinstance(current,str) and re.fullmatch(r'[+-]?\d+\.\d+V',current)
                and record.get('type')=='vcc' and record.get('detection_confidence',0)>=.7):
            a,b,c,d=record['bbox_top_left']
            nearby_voltages=[t['text'] for t in words if t['confidence']>=.85 and re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?V',t['text'])
                             and abs((t['bbox'][0]+t['bbox'][2])/2-(a+c)/2)<45 and abs((t['bbox'][1]+t['bbox'][3])/2-(b+d)/2)<30]
            seeds=[t for t in words if ',' in t['text'] and t['text'].replace(',','.')==current and t['confidence']>=.85
                   and abs((t['bbox'][0]+t['bbox'][2])/2-(a+c)/2)<45 and abs((t['bbox'][1]+t['bbox'][3])/2-(b+d)/2)<30
                   and _supply_caption_owned(record,records['records'],t['bbox'])]
            owned=[p for p in pins['pin_candidates'] if p.get('owner')==owner and not p.get('text_hypothesis_primary_candidate_id')
                   and p.get('confidence',0)>=.8 and outward_wire(rgb,record['bbox_top_left'],p['point'])]
            if (seeds and all(t.replace(',','.')==current for t in nearby_voltages)
                    and len(owned)==1 and reader is not None and calls[0]+3<=config['maximum_ocr_calls']):
                # Keep punctuation descenders: choose the largest observed crop,
                # not the tighter tile crop with the highest token confidence.
                box=max(seeds,key=lambda t:(t['bbox'][2]-t['bbox'][0])*(t['bbox'][3]-t['bbox'][1]))['bbox'];reads=[]
                for scale in (3.,5.,4.):calls[0]+=1;reads.extend(reader(rgb,box,scale,'component_literal_caption'))
                accepted=[t for t in reads if t.get('confidence',0)>=.98 and t.get('view')]
                texts={str(t['text']).strip() for t in accepted}
                if (len(texts)==1 and len(accepted)==len(reads)==3 and len({t['view'] for t in accepted})>=3
                        and (literal:=next(iter(texts)))!=current and literal.replace(',','.')==current and ',' in literal):
                    comp['value']=literal;events.append({'rule':'observed_voltage_separator','owner':owner,'before':current,'after':literal,'caption_box':box,'observations':reads})
        if comp.get('value') not in {None,''}:continue
        if record.get('type')=='inductor' and comp.get('type')=='l' and record.get('detection_confidence',0)>=.8:
            anchor=_anchor(record,words);a,b,c,d=record['bbox_top_left']
            if anchor is None or c-a<1.8*(d-b):continue
            l,bb,r,t=comp['bbox'];bottom=h-bb
            box=[anchor[0]-3,bottom+2,min(w,c+15),min(h,bottom+45)]
            text,reads=_read_vote(rgb,box,reader,calls,config['maximum_ocr_calls'])
            if text and re.fullmatch(r'\d+(?:[.,]\d+)?(?:[numµ]?H)',text,re.I):
                comp['value']=text;events.append({'rule':'owned_inductor_value','owner':owner,'after':text,'caption_box':box,'observations':reads})
        if record.get('type') in {'vcc','vbus_in','vbus_out'} and comp.get('type') in {'v','vcc','vdd'} and record.get('detection_confidence',0)>=.8:
            owned=[p for p in pins['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                   and not p.get('text_hypothesis_primary_candidate_id') and p.get('confidence',0)>=.9]
            paths=[path for p in owned if (path:=outward_wire(rgb,record['bbox_top_left'],p['point']))]
            if len(paths)!=1:continue
            a,b,c,d=record['bbox_top_left'];seeds=[t for t in words if t['text'] in NAMED_SUPPLIES and t['confidence']>=.8
                 and abs((t['bbox'][0]+t['bbox'][2])/2-(a+c)/2)<45 and abs((t['bbox'][1]+t['bbox'][3])/2-(b+d)/2)<30
                 and _supply_caption_owned(record,records['records'],t['bbox'])]
            if len({t['text'] for t in seeds})!=1:continue
            box=np.median([t['bbox'] for t in seeds],axis=0).tolist();a,b,c,d=box
            text,reads=_read_vote(rgb,[a-3,b-3,c+3,d+3],reader,calls,config['maximum_ocr_calls'],scales=(3.,5.,4.),minimum=.93)
            if text==seeds[0]['text'] and max(t.get('confidence',0) for t in reads)>=.95:
                comp['value']=text;events.append({'rule':'owned_supply_value','owner':owner,'after':text,'caption_box':box,'outward_wire':paths[0],'observations':reads})
    if len(events)>config['maximum_changes']:return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],'ocr_calls':calls[0]}
    if out['pins']!=target['pins'] or out['nets']!=target['nets']:raise ValueError('component_caption_changed_electrical_output')
    for owner,c in target['components'].items():
        if {k:v for k,v in out['components'][owner].items() if k not in {'Name','value'}}!={k:v for k,v in c.items() if k not in {'Name','value'}}:raise ValueError('component_caption_changed_geometry_or_type')
    return out,{'enabled':True,'status':'applied' if out!=target else 'no_effect','changes':events,'withheld':withheld,
                'ocr_calls':calls[0],'reference_read':False}
