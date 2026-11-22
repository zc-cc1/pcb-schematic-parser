"""Recover locally observed symbol contacts and membership, preserving wires.

This is a final export repair for overlapping horizontal LED symbols and
caption-confirmed four-contact open switches. No reference data is used.
"""
from collections import defaultdict
from copy import deepcopy
import math,re
import cv2
import numpy as np
from .pin_numeric_connector import observations
from .red_resistor_contacts import resistor_contact_proof
from .terminal_identity_recovery import bar_profile

DEFAULT={'enabled':False,'maximum_groups':4,'maximum_ocr_calls':24}


def _masks(rgb):
    r,g,b=[rgb[:,:,i].astype(np.int16) for i in range(3)]
    return ((r>g+35)&(r>b+35)&(g<150)&(b<150),
            (g>r+20)&(g>b+20)&(g>60),
            (np.maximum(r,g)>b+30)&(np.maximum(r,g)>70))


def _groups(values):
    groups=[]
    for value in values:
        if not groups or value-groups[-1][-1]>1:groups.append([])
        groups[-1].append(int(value))
    return groups


def parallel_led_proof(rgb,record,words):
    if record.get('type')!='led' or record.get('detection_confidence',0)<.85:return None
    red,green,color=_masks(rgb);h,w=red.shape
    a,b,c,d=record['bbox_top_left'];left,top,right,bottom=map(int,[math.floor(a),math.floor(b),math.ceil(c),math.ceil(d)])
    if min(left,top)<25 or right+90>=w or bottom+25>=h:return None
    rows=_groups(np.flatnonzero(red[top:bottom,right+2:right+31].sum(1)>=20)+top)
    if not 2<=len(rows)<=4:return None
    ys=[round(float(np.median(row))) for row in rows]
    if min(np.diff(ys))<12 or max(np.diff(ys))>45:return None
    patch=red[top:bottom,left:right].astype('uint8')
    contours,hierarchy=cv2.findContours(patch,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    holes=[cv2.contourArea(q) for q,info in zip(contours,hierarchy[0]) if info[3]>=0 and 80<=cv2.contourArea(q)<=350] if hierarchy is not None else []
    if len(holes)<len(rows):return None
    opened=cv2.morphologyEx(patch,cv2.MORPH_OPEN,np.ones((14,1),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(opened,8)
    columns=sorted({int(round(left+x+(ww-1)/2)) for x,y,ww,hh,area in stats[1:] if ww<=4 and hh>=14})
    columns=[v for i,v in enumerate(columns) if not i or v-columns[i-1]>3]
    if len(columns)!=2 or not 12<=columns[1]-columns[0]<=35:return None
    interior=np.zeros_like(patch)
    for contour,info in zip(contours,hierarchy[0]):
        if info[3]>=0 and 80<=cv2.contourArea(contour)<=350:cv2.drawContours(interior,[contour],-1,1,-1)
    for y in ys:
        band=interior[max(0,y-top-9):min(bottom-top,y-top+10)]
        if band[:,columns[1]-left-3].sum()<band[:,columns[0]-left+3].sum()+4:
            return None  # Right-facing/ambiguous cathodes need a separate proof.
    supports=defaultdict(list)
    for token in words:
        if (re.fullmatch(r'LED[1-9]\d*',str(token.get('text',''))) and token.get('confidence',0)>=.98
                and token['bbox'][0]>=a-22 and top-35<token['bbox'][1]<bottom+25):supports[token['text']].append(token)
    names=[name for name,ts in supports.items() if len({t['view'] for t in ts})>=2]
    names.sort(key=lambda name:float(np.median([t['bbox'][1] for t in supports[name]])))
    if len(names)!=len(ys) or record['component_id'] not in names:return None
    symbols=[]
    for name,y in zip(names,ys):
        contacts=[]
        for start,direction in [(columns[0]-6,-1),(columns[1]+3,1)]:
            for x in range(start,start+direction*85,direction):
                if not(2<=x<w-2):break
                if green[y-1:y+2,x-1:x+2].any():
                    yy,xx=np.where(green[y-1:y+2,x-1:x+2])
                    contacts.append([float(x-1+np.median(xx)),float(y-1+np.median(yy))]);break
                if not color[y-1:y+2,x-1:x+2].any():break
        if len(contacts)!=2 or abs(contacts[0][1]-contacts[1][1])>1:return None
        row=float(np.mean([p[1] for p in contacts]));height=columns[1]-columns[0]
        caption_center=np.median([[(t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2] for t in supports[name]],axis=0)
        models=[t for t in words if record.get('Name') and t.get('text')==record['Name']
                and t.get('confidence',0)>=.98
                and math.dist(caption_center,[(t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2])<=75]
        model_name=record['Name'] if len({t['view'] for t in models})>=2 else None
        # Frame the observed diode/nearby emission-stroke area, not its leads.
        frame=[float(max(left,columns[0]-height)),row-height*1.75,float(columns[1]+1),row+height*.85]
        symbols.append({'owner':name,'row':row,'contacts':contacts,'body':frame,'model_name':model_name,
                        'caption_views':sorted({t['view'] for t in supports[name]})})
    return {'source_record':record['component_id'],'columns':columns,'symbols':symbols}


def read_contact_numbers(rgb,shape,reader,budget):
    if reader is None or budget<8:return None
    rows=[];numbers=[];calls=0
    for index,(x,y) in enumerate(shape['contacts']):
        box=list(map(round,[x-18,y+4,x-1,y+28] if index<2 else [x-18,y-28,x-1,y-3]))
        accepted=None;observed=[]
        for rotation in (90,270):
            if calls+2>budget:return None
            for scale in (3.,5.):
                calls+=1;observed.extend(reader(rgb,box,scale,'symbol_contact_digit',rotation=rotation))
            votes=defaultdict(set)
            for t in observed:
                if str(t.get('text','')) in {'1','2','3','4'} and t.get('confidence',0)>=.98 and t.get('view'):
                    votes[str(t['text'])].add(t['view'])
            if len(votes)>1:return None
            if len(votes)==1 and len(next(iter(votes.values())))>=2:accepted=next(iter(votes));break
        if accepted is None:return None
        numbers.append(accepted);rows.append({'point':[x,y],'crop':box,'number':accepted,'observations':observed})
    if len(set(numbers))!=4:return None
    return {'numbers':numbers,'rows':rows,'calls':calls}


def switch_shape(rgb,caption):
    red,green,color=_masks(rgb);h,w=red.shape;x1,y1,x2,y2=caption['bbox']
    a,b,c,d=round(x1)-70,round(y1)-30,round(x1),round(y2)+100
    if min(a,b)<0 or c>w or d>h:return None
    opened=cv2.morphologyEx(red[b:d,a:c].astype('uint8'),cv2.MORPH_OPEN,np.ones((80,1),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(opened,8)
    stems=sorted([[float(a+x+(ww-1)/2),int(b+y),int(b+y+hh-1)] for x,y,ww,hh,area in stats[1:]
                  if ww<=3 and 90<=hh<=160])
    if len(stems)!=2 or not 20<=stems[1][0]-stems[0][0]<=65:return None
    if max(abs(stems[0][i]-stems[1][i]) for i in (1,2))>5:return None
    # The two floating left terminals have grey endpoint markers. Their actual
    # centres establish terminal rows; no inferred numbering order is used.
    gray=(rgb.max(2).astype(int)-rgb.min(2).astype(int)<25)&(rgb.mean(2)<210)
    ends=[];left=stems[0][0];marker_x=[]
    for y in stems[0][1:]:
        xx,yy=round(left),round(y);patch=gray[yy-7:yy+8,xx-4:xx+5].astype('uint8')
        _,_,ss,cc=cv2.connectedComponentsWithStats(patch,8)
        markers=[p+[xx-4,yy-7] for s,p in zip(ss[1:],cc[1:]) if 2<=s[2]<=7 and 2<=s[3]<=7 and 5<=s[4]<=40]
        if len(markers)!=1 or abs(markers[0][0]-left)>2:return None
        ends.append(float(markers[0][1]));marker_x.append(float(markers[0][0]))
    aa,bb,cc,dd=round(stems[0][0])+4,round(ends[0])+20,round(stems[1][0])-3,round(ends[1])-20
    contours,hierarchy=cv2.findContours(red[bb:dd,aa:cc].astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    holes=[]
    if hierarchy is not None:
        for q,info in zip(contours,hierarchy[0]):
            area=cv2.contourArea(q);x,y,ww,hh=cv2.boundingRect(q)
            if info[3]>=0 and 2<=area<=30 and max(ww,hh)<=9:holes.append([aa+x+ww/2,bb+y+hh/2])
    if len(holes)!=2 or abs(holes[0][1]-holes[1][1])>4:return None
    columns=[float(np.median(marker_x)),float(math.ceil(stems[1][0]))]
    contacts=[[x,end] for end in ends for x in columns]
    return {'contacts':contacts,'stems':stems,'contact_holes':holes,
            'body':[stems[0][0]-1,ends[0]+19,stems[1][0]+1,ends[1]-19]}


def _refs(net):return {r for r in str(net.get('hyperGraph','')).strip('()').split(',') if r}
def _pin(point,h,name=''):return {'pinname':name,'point':{'x':float(point[0]),'y':float(h-point[1])}}


def _ground_pair_proof(rgb,record,point):
    red,_,color=_masks(rgb);x,y=map(round,point);a,b,c,d=record['bbox_top_left']
    if record.get('type')!='gnd' or record.get('detection_confidence',0)<.7 or not a-8<x<c:return None
    if not b-3<=y<=d+3:return None
    patch=red[max(0,y-4):y+5,x+4:min(red.shape[1],round(c)+4)]
    columns=_groups(np.flatnonzero(patch.sum(0)>=7)+x+4)
    centers=[float(np.median(g)) for g in columns]
    if len(centers)!=3 or any(not 4<=q-p<=10 for p,q in zip(centers,centers[1:])):return None
    if not all(color[y-1:y+2,xx-1:xx+2].any() for xx in range(x,round(centers[0]))):return None
    return {'ground':record['component_id'],'point':list(point),'bar_columns':centers}


def _route(out,before,rgb,changed,explicit):
    red,green,color=_masks(rgb);h,w=red.shape;mask=color.copy()
    for comp in out['components'].values():
        l,b,r,t=comp['bbox'];a,bb,c,d=round(l),round(h-t),round(r),round(h-b)
        # Refined symbol frames may have a nearby external green conductor in
        # their antialias margin. Suppress red frame remnants in that margin,
        # while keeping the observed green pixels outside the actual body.
        ys=slice(max(0,bb-3),min(h,d+4));xs=slice(max(0,a-3),min(w,c+4))
        mask[ys,xs]&=green[ys,xs]
        mask[max(0,bb):min(h,d+1),max(0,a):min(w,c+1)]=False
    blue=(rgb[:,:,2].astype(int)>rgb[:,:,0].astype(int)+20)&(rgb[:,:,2].astype(int)>rgb[:,:,1].astype(int)+20)
    closed=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
    mask|=closed.astype(bool)&blue
    _,labels=cv2.connectedComponents(mask.astype('uint8'),8);groups=defaultdict(set);binding={}
    for owner,ps in out['pins'].items():
        for key,p in ps.items():
            x,y=round(p['point']['x']),round(h-p['point']['y']);ranks={}
            for yy in range(max(0,y-5),min(h,y+6)):
                for xx in range(max(0,x-5),min(w,x+6)):
                    cid=int(labels[yy,xx]);distance=math.hypot(xx-x,yy-y)
                    if cid and distance<=5:ranks[cid]=min(distance,ranks.get(cid,99))
            order=sorted(ranks,key=lambda k:(ranks[k],k))
            if order and (len(order)==1 or ranks[order[1]]-ranks[order[0]]>=2):
                ref=owner+'.'+key[4:];groups[order[0]].add(ref);binding[ref]=order[0]
    # A terminal entering an independently verified ground comb has its own
    # symbol contact, even where the comb's far red bar overlaps another wire.
    for i,pair in enumerate(explicit,1):
        for refs in groups.values():refs-=pair
        groups[-i]=set(pair)
        for ref in pair:binding[ref]=-i
    dirty={r for n in before['nets'].values() if any(s.rsplit('.',1)[0] in changed for s in _refs(n)) for r in _refs(n)}
    dirty|={r for r in binding if r.rsplit('.',1)[0] in changed}
    selected={binding[r] for r in dirty if r in binding};used=set();events=[]
    for cid in sorted(selected):
        refs=groups[cid]
        if len(refs)<2:continue
        candidates=[(len(_refs(n)&refs),len(n['edges']),nid) for nid,n in out['nets'].items()
                    if nid not in used and _refs(n)&refs]
        nid=max(candidates)[-1] if candidates else 'LOCAL_CONTACT_'+str(abs(cid))
        while nid in used:nid+='x'
        for other,net in out['nets'].items():
            if other!=nid:net['hyperGraph']='('+','.join(sorted(_refs(net)-refs))+')'
        net=out['nets'].get(nid,{'edges':{}});net['hyperGraph']='('+','.join(sorted(refs))+')';out['nets'][nid]=net;used.add(nid)
        events.append({'network_id':nid,'members':sorted(refs),'pixel_group':int(cid)})
    for net in out['nets'].values():
        kept={r for r in _refs(net) if r.rsplit('.',1)[0] not in changed
              or 'pin_'+r.rsplit('.',1)[1] in out['pins'].get(r.rsplit('.',1)[0],{})}
        net['hyperGraph']='('+','.join(sorted(kept))+')'
    out['nets']={nid:n for nid,n in out['nets'].items() if _refs(n) or n['edges']}
    return events


def recover_local_symbol_contacts(target,rgb,graph,records,pins,ocr,image_sha256,*,reader=None,cfg=None):
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_local_symbol_setting')
    if any(type(config[k]) is not int or not 0<=config[k]<=128 for k in ('maximum_groups','maximum_ocr_calls')):raise ValueError('invalid_local_symbol_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    for document in (graph,records,pins,ocr):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('prediction_only') is not True or document.get('coordinate_origin')!='top_left'):
            raise ValueError('local_symbol_evidence_differs_from_image')
    words=observations(ocr);red,green,color=_masks(rgb);events=[];changed=set();explicit=[];calls=0
    for record in records['records']:
        proof=parallel_led_proof(rgb,record,words)
        if proof is None:continue
        if any(s['owner'] in out['components'] and out['components'][s['owner']]['type']!='led' for s in proof['symbols']):continue
        for symbol in proof['symbols']:
            owner=symbol['owner'];a,b,c,d=symbol['body']
            model=symbol['model_name'] or out['components'].get(owner,{}).get('Name')
            out['components'][owner]={'type':'led','Name':model,'value':None,'bbox':[a,h-d,c,h-b]}
            out['pins'][owner]={'pin_2':_pin(symbol['contacts'][0],h,'K'),'pin_1':_pin(symbol['contacts'][1],h,'A')};changed.add(owner)
            grounds=[p for r in records['records'] if r['component_id'] in out['components']
                     and (p:=_ground_pair_proof(rgb,r,symbol['contacts'][1]))]
            if len(grounds)==1:
                gp=grounds[0];go=gp['ground'];key='pin_'+str(1+sum(1 for e in explicit if any(ref.startswith(go+'.') for ref in e)))
                if go not in changed:out['pins'][go]={}
                out['pins'][go][key]=_pin(gp['point'],h);changed.add(go);explicit.append({owner+'.1',go+'.'+key[4:]})
                events.append({'rule':'observed_ground_comb_contact',**gp})
        events.append({'rule':'parallel_horizontal_leds',**proof})
    caption_groups=defaultdict(list)
    for token in words:
        if re.fullmatch(r'(?:SW|S)[1-9]\d*',str(token.get('text',''))) and token.get('confidence',0)>=.98:caption_groups[token['text']].append(token)
    for owner,tokens in sorted(caption_groups.items()):
        if owner in out['components'] or len({t['view'] for t in tokens})<2:continue
        shape=switch_shape(rgb,max(tokens,key=lambda t:t['confidence']))
        if shape is None:continue
        def counted_reader(*args,**kwargs):
            nonlocal calls
            if calls>=config['maximum_ocr_calls']:return []
            calls+=1
            return reader(*args,**kwargs) if reader else []
        numbered=read_contact_numbers(rgb,shape,counted_reader if reader else None,config['maximum_ocr_calls']-calls)
        if numbered is None:continue
        a,b,c,d=shape['body']
        out['components'][owner]={'type':'switch','Name':owner,'value':None,'bbox':[a,h-d,c,h-b]}
        out['pins'][owner]={'pin_'+num:_pin(point,h,num) for num,point in zip(numbered['numbers'],shape['contacts'])};changed.add(owner)
        events.append({'rule':'observed_four_contact_switch','owner':owner,**shape,**numbered})
        # Only repair a resistor whose existing numbering is invalid for a
        # two-terminal device, with a fresh symmetric rectangle/stem proof.
        for record in records['records']:
            ro=record['component_id']
            if out['components'].get(ro,{}).get('type')!='r' or not set(out['pins'].get(ro,{}))-{'pin_1','pin_2'}:continue
            proof=resistor_contact_proof(rgb,record)
            if proof is None:continue
            out['pins'][ro]={'pin_'+str(i):_pin(p,h,str(i)) for i,p in enumerate(proof['terminals'],1)};changed.add(ro)
            events.append({'rule':'invalid_passive_numbering_from_observed_stems','owner':ro,**proof})
        # A lower switch stem may lead directly to a source-confirmed ground.
        for point in shape['contacts'][2:]:
            for record in records['records']:
                if record['type']!='gnd' or record.get('detection_confidence',0)<.65:continue
                a,b,c,d=record['bbox_top_left']
                if not abs(point[0]-(a+c)/2)<=4 or not 0<b-point[1]<30:continue
                if bar_profile(red,[a,b,c,d],1,3,point) is None:continue
                if not all(color[y-1:y+2,round(point[0])-1:round(point[0])+2].any() for y in range(round(point[1]),round(b))):continue
                go=record['component_id'];out['pins'][go]={'pin_1':_pin(point,h)};changed.add(go)
                events.append({'rule':'observed_switch_ground_stem','owner':go,'point':point})
    if not changed:return deepcopy(target),{'enabled':True,'status':'no_effect','changes':[],'ocr_calls':calls}
    if sum(e['rule'] in {'parallel_horizontal_leds','observed_four_contact_switch'} for e in events)>config['maximum_groups']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],'ocr_calls':calls}
    routes=_route(out,target,rgb,changed,explicit)
    # Wire geometry and every protected component/pin must remain byte-for-byte
    # unchanged. The new stage only edits supported symbols and memberships.
    for owner,c in target['components'].items():
        if c['type'] in {'box','ic','switch','pin'} and (out['components'].get(owner)!=c or out['pins'].get(owner)!=target['pins'].get(owner)):
            raise ValueError('local_symbol_changed_protected_component')
    protected={ref for net in target['nets'].values() for ref in _refs(net)
               if target['components'].get(ref.rsplit('.',1)[0],{}).get('type') in {'box','ic','switch','pin'}}
    remaining=set().union(*(_refs(n) for n in out['nets'].values())) if out['nets'] else set()
    if not protected<=remaining:
        return deepcopy(target),{'enabled':True,'status':'withheld_unproven_protected_contacts','changes':[],
                                 'ocr_calls':calls,'missing_members':sorted(protected-remaining)}
    for nid,n in target['nets'].items():
        if n['edges'] and out['nets'].get(nid,{}).get('edges')!=n['edges']:raise ValueError('local_symbol_changed_wire_geometry')
    if any(nid not in target['nets'] and n['edges'] for nid,n in out['nets'].items()):raise ValueError('local_symbol_created_unobserved_wire')
    return out,{'enabled':True,'status':'applied' if out!=target else 'no_effect','changes':events,'routes':routes,'ocr_calls':calls,'reference_read':False}
