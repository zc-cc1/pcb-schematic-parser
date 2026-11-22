"""Image-supported switch/one-contact terminal refinement, without reference labels.

Classification is gated by observed designators and actual symbol structure.
Every added/rekeyed number needs its own OCR; unreadable contacts stay unresolved.
"""
from collections import defaultdict
from copy import deepcopy
import math
import re
import statistics

import cv2
import numpy as np
from .pin_geometry_recovery import LeadEvidence, DEFAULT as GEOMETRY_DEFAULT, _members, _segment_distance
from .pin_vertical_repair import vertical_leads
from .pin_numeric_connector import observations, vote

DEFAULT={'min_confidence':.98,'min_views':2,'max_components':4,'max_ocr_calls':96,'scales':[3.,5.]}


def small_frame(evidence,box):
    a,b,c,d=box;h,w=evidence.red.shape
    x1,y1=max(0,math.floor(a-20)),max(0,math.floor(b-20))
    x2,y2=min(w,math.ceil(c+20)),min(h,math.ceil(d+20))
    if x1>=x2 or y1>=y2:return None
    mask=evidence.red[y1:y2,x1:x2].astype('uint8')
    horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,14),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(horizontal,8)
    bars=[]
    for x,y,width,height,area in stats[1:]:
        if 22<=width<=90 and height<=5:bars.append([x1+float(x),y1+float(y)+(height-1)/2,x1+float(x+width-1)])
    frames=[]
    for i,first in enumerate(bars):
        for second in bars[i+1:]:
            top,bottom=sorted([first,second],key=lambda v:v[1])
            left=(top[0]+bottom[0])/2;right=(top[2]+bottom[2])/2;height=bottom[1]-top[1]
            if not 10<=height<=38 or not 1.4<=(right-left)/height<=4:continue
            if max(abs(top[0]-bottom[0]),abs(top[2]-bottom[2]))>3:continue
            if max(abs(left-a),abs(right-c),abs(top[1]-b),abs(bottom[1]-d))>22:continue
            if any(sum(evidence.hit(x,y) for y in range(round(top[1])+2,round(bottom[1])-1))/max(1,round(height)-3)<.94 for x in [left,right]):continue
            interior=evidence.red[math.ceil(top[1])+3:math.floor(bottom[1])-2,math.ceil(left)+3:math.floor(right)-2]
            if not interior.size or interior.mean()>.01:continue
            frame=[left,top[1],right,bottom[1]]
            if not any(math.dist(frame,f)<4 for f in frames):frames.append(frame)
    return frames[0] if len(frames)==1 else None


def contact_rows(evidence,frame):
    a,b,c,d=frame
    mask=evidence.red[math.ceil(b)+7:math.floor(d)-6,math.ceil(a)+7:math.floor(c)-6].astype('uint8')
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    holes=[]
    if hierarchy is None:return []
    for contour,info in zip(contours,hierarchy[0]):
        area=cv2.contourArea(contour);x,y,w,h=cv2.boundingRect(contour)
        if info[3]>=0 and 2<=area<=65 and 2<=w<=14 and 2<=h<=14:
            holes.append([a+7+x+w/2,b+7+y+h/2,float(area)])
    groups=[]
    for hole in sorted(holes,key=lambda p:p[1]):
        if not groups or abs(hole[1]-statistics.median(p[1] for p in groups[-1]))>5:groups.append([])
        groups[-1].append(hole)
    return [g for g in groups if len(g)>=3 and max(p[0] for p in g)-min(p[0] for p in g)>=20]


def all_leads(evidence,frame):
    cfg={'min_lead_px':8,'max_lead_px':64}
    return (evidence.leads(frame,'left')+evidence.leads(frame,'right')+
            vertical_leads(evidence,frame,'top',cfg)+vertical_leads(evidence,frame,'bottom',cfg))


def text_patch(rgb,box,*,binary=False,bridge=None):
    h,w=rgb.shape[:2];a,b,c,d=map(int,box)
    if min(a,b)<0 or c>w or d>h or a>=c or b>=d:return None
    raw=rgb[b:d,a:c];v=raw.astype(np.int16)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<230)).astype('uint8')
    if bridge is not None:
        # Repair at most the observed three-pixel red frame band. No strokes
        # outside that band, glyph templates or expected identities are drawn.
        closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
        lo=max(0,round(bridge)-b-1);hi=min(mask.shape[0],lo+4)
        mask[lo:hi]=closed[lo:hi]
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    mask=np.isin(labels,[i for i in range(1,n) if stats[i,4]>=4])
    yy,xx=np.where(mask)
    if not len(xx):return None
    x1,x2,y1,y2=min(xx),max(xx),min(yy),max(yy)
    if x2-x1>42 or y2-y1>45:return None
    output=np.full((y2-y1+9,x2-x1+9,3),255,np.uint8)
    chosen=mask[y1:y2+1,x1:x2+1]
    output[4:-4,4:-4][chosen]=0 if binary or bridge is not None else raw[y1:y2+1,x1:x2+1][chosen]
    return output


def source_numbers(tokens,lead):
    side=lead['side'];edge=lead['edge'];x,y=lead['point'];found=[]
    for token in tokens:
        if not token['text'].isdigit() or int(token['text'])<=0:continue
        a,b,c,d=token['bbox'];cx,cy=(a+c)/2,(b+d)/2
        if side in {'top','bottom'}:
            valid=x-22<=cx<=x+3 and -4<=(-1 if side=='top' else 1)*(cy-edge)<=30 and d-b<=30
        else:
            valid=y-22<=cy<=y+3 and -3<=(-1 if side=='left' else 1)*(cx-edge)<=30 and c-a<=35
        if valid:found.append(token)
    return found


def recover_component_kinds(target,document,graph,ocr_document,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};out=deepcopy(target);evidence=LeadEvidence(rgb,GEOMETRY_DEFAULT)
    height,width=rgb.shape[:2];tokens=observations(ocr_document)
    audit={'enabled':True,'policy':'observed_component_kind_v1','groups':[],'component_changes':[],
           'changes':[],'network_changes':[],'network_withheld':[],'unresolved_contacts':[], 'ocr_calls':0,'budget_exhausted':False}
    def reads(patch,rotation,view):
        if patch is None:return []
        if rotation:patch=cv2.rotate(patch,cv2.ROTATE_90_CLOCKWISE if rotation==90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        values=[]
        for scale in cfg['scales']:
            if audit['ocr_calls']>=cfg['max_ocr_calls']:
                audit['budget_exhausted']=True;return values
            audit['ocr_calls']+=1
            values.extend(reader(patch,[0,0,patch.shape[1],patch.shape[0]],scale,0,True,view+f'_r{rotation}_s{scale}'))
        return values
    count=0
    for owner,comp in sorted(target['components'].items()):
        if comp.get('type') not in {'box','ic'} or comp.get('Name') not in [None,'',owner]:continue
        switch=bool(re.fullmatch(r'(?:SW|S)\d+',owner));terminal=bool(re.fullmatch(r'(?:H|J|P)\d+',owner))
        if not switch and not terminal:continue
        a,b,c,d=comp['bbox'];box=[a,height-d,c,height-b]
        designators=[t for t in tokens if t['text']==owner and abs((t['bbox'][0]+t['bbox'][2])/2-(a+c)/2)<max(50,(c-a)/2+20)
                     and box[1]-50<=(t['bbox'][1]+t['bbox'][3])/2<=box[3]+35]
        if vote(designators,cfg)!=owner:continue
        old=target['pins'].get(owner,{})
        if terminal:
            if set(old)!={'pin_1'} or old['pin_1'].get('pinname')!='1':continue
            frame=small_frame(evidence,box)
            if frame is None:continue
            leads=all_leads(evidence,frame)
            if len(leads)!=1 or leads[0]['side'] not in {'left','right'}:continue
            lead=leads[0];x,y=lead['point']
            owned=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                   and abs(p['point'][1]-y)<=5 and min(x,lead['edge'])-3<=p['point'][0]<=max(x,lead['edge'])+3
                   and str(p.get('number'))=='1' and str(p.get('pinname'))=='1']
            if len(owned)!=1:continue
            labels=[t for t in tokens if t['text']=='1' and frame[0]<=(t['bbox'][0]+t['bbox'][2])/2<=frame[2]
                    and frame[1]<=(t['bbox'][1]+t['bbox'][3])/2<=frame[3]]
            if vote(labels,cfg)!='1':continue
            if count>=cfg['max_components']:break
            count+=1;label=max(labels,key=lambda t:t['confidence']);observed=reads(text_patch(rgb,label['bbox']),0,f'terminal_{owner}')
            group={'owner':owner,'kind':'terminal','frame':frame,'lead':lead,'readings':observed};audit['groups'].append(group)
            if vote(observed,cfg)!='1':group['status']='unconfirmed_digit';continue
            point={'x':float(x),'y':float(height-y)}
            out['components'][owner]={**comp,'type':'pin','bbox':[frame[0],height-frame[3],frame[2],height-frame[1]]}
            out['pins'][owner]['pin_1']={**old['pin_1'],'point':point}
            audit['component_changes'].append({'owner':owner,'before':comp,'after':out['components'][owner]})
            audit['changes'].append({'owner':owner,'old_key':'pin_1','number':'1','before':old['pin_1'],'after':out['pins'][owner]['pin_1'],'source_candidate_id':owned[0]['candidate_id']})
            group['status']='accepted'
            # Same identity: preserve all existing memberships. No new route is
            # inferred merely because the measured terminal point moved.
            if not any(owner+'.1' in _members(n) for n in out['nets'].values()):audit['network_withheld'].append({'member':owner+'.1','reason':'no_existing_exported_membership'})
            continue
        found=evidence.frame(box)
        if found is None:continue
        frame=found[0];contacts=contact_rows(evidence,frame)
        if len(contacts)<2:continue
        leads=all_leads(evidence,frame);sides={s:sum(l['side']==s for l in leads) for s in ['left','right','top','bottom']}
        if not 8<=len(leads)<=32 or min(sides.values())<1 or sides['top']<3 or sides['bottom']<3:continue
        lengths=[l['length'] for l in leads];length=statistics.median(lengths)
        if sum(abs(v-length)<=4 for v in lengths)<.6*len(lengths):continue
        rows=[]
        for lead in leads:
            side=lead['side'];x,y=lead['point'];edge=lead['edge']
            if side in {'left','right'}:
                point=[edge+(-1 if side=='left' else 1)*length,y]
                roi=[int(edge-28 if side=='left' else edge+1),int(y-21),int(edge-1 if side=='left' else edge+28),int(y-1)]
            else:
                point=[x,edge+(-1 if side=='top' else 1)*length]
                roi=[int(x-20),int(edge-28 if side=='top' else edge-6),int(x-1 if side=='top' else x+2),int(edge+12 if side=='top' else edge+28)]
            candidates=[p for p in document['pin_candidates'] if p.get('owner')==owner and p.get('owner_status')=='assigned'
                        and ((abs(p['point'][1]-y)<=6 and min(edge,lead['point'][0])-4<=p['point'][0]<=max(edge,lead['point'][0])+4) if side in {'left','right'}
                             else (abs(p['point'][0]-x)<=4 and min(edge,lead['point'][1])-5<=p['point'][1]<=max(edge,lead['point'][1])+5))]
            inward=[point[0]+(2 if side=='left' else -2 if side=='right' else 0),
                    point[1]+(2 if side=='top' else -2 if side=='bottom' else 0)]
            if not candidates or not evidence.hit(*inward):break
            rows.append({**lead,'point':point,'crop':roi,'candidate_ids':[p['candidate_id'] for p in candidates],
                         'source_tokens':source_numbers(tokens,lead)})
        if len(rows)!=len(leads):continue
        if count>=cfg['max_components']:break
        count+=1;group={'owner':owner,'kind':'switch','frame':frame,'contact_rows':contacts,'rows':rows,'status':'pending'};audit['groups'].append(group)
        # Read both orientations of independently corroborated printed numbers
        # to establish the direction; do not use number order as an oracle.
        anchors=[r for r in rows if r['side'] in {'top','bottom'} and vote(r['source_tokens'],cfg) is not None]
        rotations={90:0,270:0};memo={}
        for row in anchors[:4]:
            expected=vote(row['source_tokens'],cfg)
            for rotation in [90,270]:
                values=reads(text_patch(rgb,row['crop']),rotation,f'switch_{owner}_{row["side"]}_{row["point"][0]}_gray')
                memo[(id(row),rotation,'gray')]=values
                if vote(values,cfg)==expected:rotations[rotation]+=1
        preferred=sorted(rotations,key=lambda r:-rotations[r])
        if rotations[preferred[0]]<3 or rotations[preferred[0]]<=rotations[preferred[1]]:
            group.update(status='no_unique_text_orientation',orientation_support=rotations);continue
        direction=preferred[0];group['orientation_support']=rotations;group['rotation']=direction
        for row in rows:
            rotation=direction if row['side'] in {'top','bottom'} else 0
            values=memo.get((id(row),rotation,'gray'))
            if values is None:values=reads(text_patch(rgb,row['crop']),rotation,f'switch_{owner}_{row["side"]}_{row["point"][0]}_{row["point"][1]}_gray')
            number=vote(values,cfg)
            row['readings']={'gray':values}
            if number is None or not number.isdigit() or int(number)<=0:
                values=reads(text_patch(rgb,row['crop'],binary=True),rotation,f'switch_{owner}_{row["side"]}_{row["point"][0]}_{row["point"][1]}_binary')
                row['readings']['binary']=values;number=vote(values,cfg)
            if (number is None or not number.isdigit() or int(number)<=0) and row['side']=='top':
                x=row['point'][0];edge=row['edge'];roi=[int(x-20),int(edge-25),int(x-1),int(edge+12)]
                values=reads(text_patch(rgb,roi,bridge=edge),rotation,f'switch_{owner}_top_{x}_frame_band')
                row['readings']['frame_band']=values;number=vote(values,cfg)
            row['number']=number if number and number.isdigit() and int(number)>0 else None
        accepted=[r for r in rows if r['number']]
        if len(accepted)<.7*len(rows) or len({r['number'] for r in accepted})!=len(accepted):group['status']='insufficient_or_conflicting_identities';continue
        # Every old physical record needs one confirmed destination; ambiguity
        # aborts the transaction rather than silently discarding an old pin.
        matched={};valid=True
        for key,pin in old.items():
            p=[pin['point']['x'],height-pin['point']['y']]
            candidates=[r for r in accepted if math.dist(p,r['point'])<=7]
            if len(candidates)!=1:valid=False;break
            row=candidates[0]
            if id(row) in matched:valid=False;break
            matched[id(row)]=key
        if not valid:group['status']='unmatched_existing_identity';continue
        new_pins={};rekeys={};pending=[]
        for row in accepted:
            number=row['number'];key='pin_'+number;old_key=matched.get(id(row))
            if old_key:
                new={**old[old_key],'pinname':number}
                if old_key!=key:rekeys[owner+'.'+old_key[4:]]=owner+'.'+number
            else:new={'pinname':number,'point':{'x':float(row['point'][0]),'y':float(height-row['point'][1])}}
            new_pins[key]=new
            if old_key is None or new!=old[old_key] or key!=old_key:
                pending.append({'owner':owner,'old_key':old_key,'number':number,'before':old.get(old_key),'after':new,'candidate_ids':row['candidate_ids']})
        # Rekey membership by physical record in one pass. A newly created old
        # number does not inherit the old physical contact's network.
        for nid,net in out['nets'].items():
            before=sorted(_members(net));after=sorted({rekeys.get(v,v) for v in before})
            if before!=after:
                net['hyperGraph']='('+','.join(after)+')';audit['network_changes'].append({'network_id':nid,'before':before,'after':after,'reason':'physical_identity_rekey'})
        out['pins'][owner]=new_pins;out['components'][owner]={**comp,'type':'switch'}
        audit['component_changes'].append({'owner':owner,'before':comp,'after':out['components'][owner]});audit['changes'].extend(pending)
        for row in rows:
            if not row['number']:audit['unresolved_contacts'].append({'owner':owner,'side':row['side'],'point':row['point'],'reason':'no_independently_confirmed_number'})
        for change in pending:
            ref=owner+'.'+change['number']
            if not any(ref in _members(n) for n in out['nets'].values()):audit['network_withheld'].append({'member':ref,'reason':'new_identity_without_verified_exported_route'})
        group['status']='accepted_with_unresolved_contacts' if len(accepted)<len(rows) else 'accepted'
    return out,audit
