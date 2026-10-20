"""Read missing NC identities from crossed short leads, without network edits."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .pin_geometry_recovery import DEFAULT as GEOMETRY_DEFAULT
from .pin_geometry_recovery import IC_TYPES, LeadEvidence, _members, side_of, vote

DEFAULT={'min_confidence':.98,'min_views':2,'max_components':3,
         'max_proposals':8,'max_ocr_calls':48,'min_row_spacing_px':18}


def crossed_lead(evidence,lead):
    """Require both green diagonals at the endpoint, and no outgoing wire."""
    x,y=lead['point'];mask=evidence.green;h,w=mask.shape
    def hit(a,b):
        a,b=round(a),round(b)
        return 1<=a<w-1 and 1<=b<h-1 and bool(mask[b-1:b+2,a-1:a+2].any())
    offsets=[-8,-7,-6,-5,-4,4,5,6,7,8]
    # Raster lead endpoints and the centre of the X may differ by a few pixels.
    # Keep the diagonal support threshold while searching that small tolerance.
    if not any(all(sum(hit(x+dx+d,y+dy+slope*d) for d in offsets)>=8 for slope in [-1,1])
               for dx in range(-3,4) for dy in range(-2,3)):return False
    sign=-1 if lead['side']=='left' else 1
    if sum(hit(x+sign*d,y) for d in range(13,41))>3:return False
    return True


def neutral_group(rgb,box,row):
    """Retain antialias pixels and disconnected parts of the entire number."""
    x1,y1,x2,y2=box
    if min(x1,y1)<0 or x2>rgb.shape[1] or y2>rgb.shape[0] or x2<=x1 or y2<=y1:return None
    patch=rgb[y1:y2,x1:x2];v=patch.astype(np.int16)
    mask=((v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<230)).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);accepted=[]
    for i in range(1,count):
        a,b,w,h,area=map(int,stats[i]);cx,cy=centroids[i]
        if 1<=w<=24 and 2<=h<=17 and area>=2 and -14<=y1+cy-row<=0:
            if a==0 or b==0 or a+w==patch.shape[1] or b+h==patch.shape[0]:return None
            accepted.append(i)
    if not accepted:return None
    chosen=np.isin(labels,accepted);ys,xs=np.where(chosen)
    a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    if not 6<=d-b<=17 or not 2<=c-a<=28:return None
    # No single-character OCR or inferred prefix: keep the complete row group.
    isolated=np.full_like(patch,255);isolated[chosen]=patch[chosen]
    result=cv2.copyMakeBorder(isolated[b:d,a:c],8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
    return result,[x1+a,y1+b,x1+c,y1+d],len(accepted)


def source_anchors(owner,side,frame,leads,target,document,cfg):
    """Calibrate the above-lead digit band using existing observed identities."""
    anchors=[]
    for p in document['pin_candidates']:
        if (p.get('owner')!=owner or p.get('owner_status')!='assigned'
                or p.get('text_hypothesis_primary_candidate_id')
                or p.get('number_source')=='library_pin_name'
                or not str(p.get('number','')).isdigit()):continue
        pin=target['pins'].get(owner,{}).get('pin_'+str(p['number']))
        if not pin or pin['pinname']!=p.get('pinname') or side_of(p['point'],frame)!=side:continue
        near=[lead for lead in leads if abs(lead['row']-p['point'][1])<=5]
        if len(near)!=1:continue
        row=near[0]['row'];edge=near[0]['edge'];tokens=[]
        for t in p.get('text_candidates',{}).get('number',[]):
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            outward=edge-cx if side=='left' else cx-edge
            if (str(t.get('text'))==str(p['number']) and t.get('source_view')
                    and not t.get('approximate_split_geometry')
                    and 0<=outward<=32 and -14<=cy-row<=0):
                tokens.append({**t,'view':t['source_view']})
        number,support=vote(tokens,cfg)
        if number==str(p['number']):anchors.append({'candidate_id':p['candidate_id'],'number':number,'support':support})
    return anchors


def recover_crossed_nc(target,document,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT)
    audit={'enabled':True,'policy':'crossed_nc_full_number_v1','groups':[],
           'changes':[],'network_changes':[],'ocr_calls':0,'budget_exhausted':False}
    selected=0;proposed=[]
    def read(image,box,scale,rec,view):
        audit['ocr_calls']+=1
        return reader(image,box,scale,0,rec,view)
    for owner,comp in sorted(target['components'].items()):
        if comp['type'] not in IC_TYPES:continue
        a,b,c,d=comp['bbox'];found=evidence.frame([a,h-d,c,h-b])
        if not found:continue
        frame=found[0];owner_selected=False
        for side in ['left','right']:
            leads=evidence.leads(frame,side)
            known=[(p['point']['x'],h-p['point']['y']) for p in target['pins'].get(owner,{}).values()]
            pending=[lead for lead in leads if 12<=lead['length']<=24
                     and not any(side_of(point,frame)==side and abs(point[1]-lead['row'])<=6 for point in known)
                     and all(abs(q['row']-lead['row'])>=cfg['min_row_spacing_px'] for q in leads if q is not lead)
                     and crossed_lead(evidence,lead)]
            if not pending:continue
            anchors=source_anchors(owner,side,frame,leads,target,document,cfg)
            if len({v['number'] for v in anchors})<2:continue
            if not owner_selected:
                if selected>=cfg['max_components']:audit['budget_exhausted']=True;break
                selected+=1;owner_selected=True
            for lead in pending:
                if len(audit['groups'])>=cfg['max_proposals'] or audit['ocr_calls']+6>cfg['max_ocr_calls']:
                    audit['budget_exhausted']=True;break
                row=lead['row'];edge=lead['edge'];left=side=='left'
                number_box=[math.floor(edge-32 if left else edge-2),math.floor(row-15),
                            math.ceil(edge+2 if left else edge+32),math.ceil(row+2)]
                group=neutral_group(rgb,number_box,row)
                if not group:continue
                name_box=[math.ceil(edge+3 if left else edge-60),math.floor(row-1),
                          math.ceil(edge+60 if left else edge-3),math.ceil(row+19)]
                row_box=[math.floor(edge-40 if left else edge-65),math.floor(row-17),
                         math.ceil(edge+65 if left else edge+40),math.ceil(row+19)]
                if any(min(box[:2])<0 or box[2]>w or box[3]>h for box in [name_box,row_box]):continue
                isolated,glyph_box,parts=group
                entry={'owner':owner,**lead,'anchors':anchors,'number_crop':number_box,
                       'glyph_bbox':glyph_box,'glyph_parts':parts,'name_crop':name_box,'row_crop':row_box}
                audit['groups'].append(entry)
                names=[];numbers=[];whole=[]
                for scale in [3.,5.]:whole.extend(read(rgb,row_box,scale,False,f'nc_row_{owner}_{row}_s{scale}'))
                for t in whole:
                    cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                    outward=edge-cx if left else cx-edge
                    if t['text'].isdigit() and -3<=outward<=32 and -14<=cy-row<=1:numbers.append(t)
                for scale in [3.,5.]:numbers.extend(read(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,True,f'nc_group_{owner}_{row}_s{scale}'))
                patch=rgb[name_box[1]:name_box[3],name_box[0]:name_box[2]];v=patch.astype(np.int16)
                mask=(v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<205)
                name_image=np.full_like(patch,255);name_image[mask]=patch[mask]
                name_image=cv2.copyMakeBorder(name_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
                for scale in [3.,5.]:names.extend(read(name_image,[0,0,name_image.shape[1],name_image.shape[0]],scale,True,f'nc_name_{owner}_{row}_s{scale}'))
                number,ns=vote([t for t in numbers if t['text'].isdigit()],cfg);name,ps=vote(names,cfg)
                entry.update(number=number,name=name,number_support=ns,name_support=ps,number_readings=numbers,name_readings=names)
                # Even one strong conflicting number is unsafe; never select a
                # majority through contradictory high-confidence text.
                strong={t['text'] for t in numbers if t['text'].isdigit() and t['confidence']>=cfg['min_confidence']}
                if number is None or strong!={number} or name!='NC':entry['status']='insufficient_or_conflicting_text';continue
                if not 1<=int(number)<=max(4,2*len(leads)):entry['status']='number_outside_observed_capacity';continue
                ref=owner+'.'+number
                if 'pin_'+number in target['pins'].get(owner,{}):entry['status']='existing_number';continue
                if any(ref in _members(net) for net in target['nets'].values()):entry['status']='existing_network_reference';continue
                candidates=[p for p in document['pin_candidates'] if p.get('owner')==owner
                            and str(p.get('number'))==number and not p.get('text_hypothesis_primary_candidate_id')]
                if any(abs(p['point'][1]-row)>6 or p.get('pinname') not in [None,'','NC'] for p in candidates):
                    entry['status']='conflicting_model_identity';continue
                entry['status']='ready';proposed.append(entry)
    multiplicity=Counter((r['owner'],r['number']) for r in proposed)
    for r in proposed:
        if multiplicity[r['owner'],r['number']]!=1:r['status']='duplicate_proposed_number';continue
        x,y=r['point'];out['pins'].setdefault(r['owner'],{})['pin_'+r['number']]={'pinname':'NC','point':{'x':x,'y':h-y}}
        r['status']='observed_crossed_nc';audit['changes'].append(deepcopy(r))
    return out,audit
