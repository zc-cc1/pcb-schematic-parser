"""Recover independently observed oriented net ports and short red contacts.

Printed duplicate captions remain separate ordinary components. Suffixes are
stable namespace keys, not OCR-read digits or global electrical rail aliases.
"""
from copy import deepcopy
from collections import defaultdict
import math,re
import cv2
import numpy as np
from .local_symbol_contacts import _masks,_pin,_refs
from .pin_numeric_connector import observations
from .red_resistor_contacts import resistor_contact_proof

DEFAULT={'enabled':False,'maximum_ports':16,'maximum_contacts':8}


def _dot_centres(red):
    thick=cv2.distanceTransform(red.astype('uint8'),cv2.DIST_L2,5)>=3.2
    _,_,stats,centres=cv2.connectedComponentsWithStats(thick.astype('uint8'),8)
    return [p for s,p in zip(stats[1:],centres[1:]) if 2<=s[2]<=12 and 2<=s[3]<=12]


def _gray_end(rgb,point):
    h,w=rgb.shape[:2];x,y=map(round,point)
    if not(6<x<w-6 and 6<y<h-6):return None
    patch=rgb[y-5:y+6,x-5:x+6].astype(int)
    gray=(patch.max(2)-patch.min(2)<25)&(patch.mean(2)>75)&(patch.mean(2)<220)
    _,_,stats,centres=cv2.connectedComponentsWithStats(gray.astype('uint8'),8)
    found=[p+[x-5,y-5] for s,p in zip(stats[1:],centres[1:]) if 2<=s[2]<=9 and 2<=s[3]<=9 and 5<=s[4]<=65]
    found=[p for p in found if math.dist(p,point)<=5]
    return list(map(float,found[0])) if len(found)==1 else None


def port_proof(rgb,record,words):
    if record.get('type') not in {'vbus_in','vbus_out','vbus_bidirectional'} or record.get('detection_confidence',0)<.45:return None
    red,green,color=_masks(rgb);h,w=red.shape;a,b,c,d=record['bbox_top_left']
    left,top=max(0,math.floor(a)-12),max(0,math.floor(b)-12)
    right,bottom=min(w,math.ceil(c)+13),min(h,math.ceil(d)+13)
    contours,hierarchy=cv2.findContours(red[top:bottom,left:right].astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return None
    shapes=[]
    for contour,info in zip(contours,hierarchy[0]):
        if info[3]<0 or not 250<=cv2.contourArea(contour)<=6000:continue
        vertices=cv2.approxPolyDP(contour,3.,True).reshape(-1,2)+[left,top]
        if len(vertices)!=5 or not cv2.isContourConvex(vertices.astype(np.int32)):continue
        low,high=vertices.min(0),vertices.max(0);axis=int(high[1]-low[1]>high[0]-low[0]);cross=1-axis
        if not 1.6<=(high[axis]-low[axis])/max(1,high[cross]-low[cross])<=5:continue
        minimum=sum(abs(vertices[:,axis]-low[axis])<=3);maximum=sum(abs(vertices[:,axis]-high[axis])<=3)
        if sorted([minimum,maximum])!=[1,2]:continue
        shapes.append((vertices,axis))
    if len(shapes)!=1:return None
    vertices,axis=shapes[0];cross=1-axis;low=vertices.min(0).astype(float)-2;high=vertices.max(0).astype(float)+2
    center=(low+high)/2;caption_votes=defaultdict(list)
    for token in words:
        text=str(token.get('text','')).strip()
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]{1,15}',text) or token.get('confidence',0)<.98
                or re.fullmatch(r'(?:LED|[RCLDUQJH])[1-9]\d*',text)):continue
        box=token['bbox'];tc=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
        if abs(tc[cross]-center[cross])>max(15,(high[cross]-low[cross])*.6):continue
        gap=max(low[axis]-box[axis+2],box[axis]-high[axis],0)
        if gap>35 or low[axis]-3<tc[axis]<high[axis]+3:continue
        caption_votes[text].append(token)
    qualified={text:ts for text,ts in caption_votes.items() if len({t['view'] for t in ts})>=2}
    if len(qualified)!=1:return None
    caption=next(iter(qualified));views=sorted({t['view'] for t in qualified[caption]})
    if record.get('detection_confidence',0)<.7 and len(views)<3:return None
    dots=_dot_centres(red);options=[]
    def hit(mask,p):
        x,y=map(round,p)
        return 1<=x<w-1 and 1<=y<h-1 and mask[y-1:y+2,x-1:x+2].any()
    for direction,face in [(-1,low[axis]),(1,high[axis])]:
        anchor=center.copy();anchor[axis]=face
        nearby=[dot for dot in dots if abs(dot[cross]-center[cross])<=3 and 5<direction*(dot[axis]-face)<=45]
        for distance in range(1,48):
            point=anchor.copy();point[axis]+=direction*distance
            if min(point)<6 or point[0]>=w-6 or point[1]>=h-6:break
            if nearby and min(math.dist(point,p) for p in nearby)<=1.5:
                end=min(nearby,key=lambda p:math.dist(point,p));options.append((list(map(float,end)),'observed_junction'));break
            if hit(green,point) and distance>=5:
                x,y=map(round,point);yy,xx=np.where(green[y-1:y+2,x-1:x+2])
                options.append(([float(x-1+np.median(xx)),float(y-1+np.median(yy))],'green_transition'));break
            if not hit(color,point):
                end=_gray_end(rgb,point)
                if distance>=7 and end is not None:options.append((end,'floating_gray_terminal'))
                break
            if distance>=7:
                forward=point.copy();forward[axis]+=direction*6
                arms=[]
                for sign in (-1,1):
                    q=point.copy();q[cross]+=sign*6;arms.append(hit(color,q))
                if any(arms) and not hit(color,forward):
                    options.append((list(map(float,point)),'observed_red_bend'));break
    if len(options)!=1:return None
    point,kind=options[0]
    return {'source_record':record['component_id'],'caption':caption,'views':views,'axis':axis,
            'vertices':vertices.tolist(),'frame':[float(low[0]),float(low[1]),float(high[0]),float(high[1])],
            'point':point,'contact_kind':kind,'source_confidence':record['detection_confidence']}


def _overlap(a,b):
    l=max(a[0],b[0]);t=max(a[1],b[1]);r=min(a[2],b[2]);d=min(a[3],b[3])
    inter=max(0,r-l)*max(0,d-t)
    return inter/max(1,min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])))


def attach_contact(out,owner,number,rgb):
    """Add only the new contact to a uniquely adjacent existing contact/net.

    Full pixel connected components cannot distinguish undotted crossings.
    Preserve existing groups and require local continuous ink to an endpoint.
    """
    h,w=rgb.shape[:2];_,_,color=_masks(rgb)
    p=out['pins'][owner]['pin_'+number]['point'];point=np.array([p['x'],h-p['y']])
    def visible(q,limit):
        distance=math.dist(point,q)
        if distance>limit:return False
        for pos in np.linspace(point,q,max(2,int(distance*2)+1)):
            x,y=map(round,pos)
            if not(1<=x<w-1 and 1<=y<h-1) or not color[y-1:y+2,x-1:x+2].any():return False
        return True
    nearby={other+'.'+key[4:] for other,ps in out['pins'].items() if other!=owner
            for key,p in ps.items() if visible([p['point']['x'],h-p['point']['y']],6)}
    candidates=[]
    for nid,net in out['nets'].items():
        if _refs(net)&nearby or any(visible([p['x'],h-p['y']],12)
                                  for edge in net['edges'].values() for p in [edge[0],edge[-1]]):candidates.append(nid)
    if len(candidates)>1:return None
    ref=owner+'.'+number
    if candidates:
        nid=candidates[0];net=out['nets'][nid];net['hyperGraph']='('+','.join(sorted(_refs(net)|nearby|{ref}))+')'
    elif nearby:
        nid='ORIENTED_CONTACT_'+owner+'_'+number
        while nid in out['nets']:nid+='x'
        out['nets'][nid]={'hyperGraph':'('+','.join(sorted(nearby|{ref}))+')','edges':{}}
    else:return None
    return {'network_id':nid,'members':sorted(_refs(out['nets'][nid]))}


def clear_shared_contact_slivers(out,refs,point,rgb):
    """Suppress tiny segments wholly inside a coincident-contact tolerance.

    A segment extending beyond the observed contact neighborhood is retained.
    The caller has proved both physical contacts are within six pixels.
    """
    h,w=rgb.shape[:2];_,_,color=_masks(rgb);removed=[]
    for nid,net in out['nets'].items():
        if _refs(net)!=refs:continue
        for key,edge in list(net['edges'].items()):
            pts=np.array([[p['x'],h-p['y']] for p in edge])
            if len(pts)!=2 or math.dist(*pts)>8 or max(math.dist(p,point) for p in pts)>4:continue
            samples=np.rint(np.linspace(*pts,20)).astype(int)
            if any(not(1<=x<w-1 and 1<=y<h-1) or not color[y-1:y+2,x-1:x+2].any() for x,y in samples):continue
            del net['edges'][key];removed.append({'network_id':nid,'edge_id':key,'edge':edge})
    return removed


def recover_oriented_port_contacts(target,rgb,graph,records,pins,ocr,image_sha256,*,cfg=None):
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target);h,w=rgb.shape[:2]
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_oriented_port_setting')
    if any(type(config[k]) is not int or not 0<=config[k]<=128 for k in ('maximum_ports','maximum_contacts')):raise ValueError('invalid_oriented_port_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    for document in (graph,records,pins,ocr):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('coordinate_origin')!='top_left' or document.get('prediction_only') is not True):
            raise ValueError('oriented_port_evidence_differs_from_image')
    words=observations(ocr);proofs=[]
    for record in sorted(records['records'],key=lambda r:-r.get('detection_confidence',0)):
        proof=port_proof(rgb,record,words)
        if proof and not any(_overlap(proof['frame'],p['frame'])>=.7 for p in proofs):proofs.append(proof)
    changed=set();events=[]
    # Unprinted duplicate-label suffixes are namespace keys. Keep all existing
    # physical identities; allocate new ones right-to-left, then top-to-bottom.
    for proof in sorted(proofs,key=lambda p:(-sum(p['frame'][::2])/2,sum(p['frame'][1::2])/2)):
        matches=[]
        for owner,comp in out['components'].items():
            if comp['type']!='net_bidirection' or comp.get('value')!=proof['caption']:continue
            a,b,c,d=comp['bbox'];box=[a,h-d,c,h-b]
            if _overlap(box,proof['frame'])>=.5:matches.append(owner)
        if len(matches)>1:continue
        if matches:
            owner=matches[0]
            if out['pins'].get(owner):continue
        else:
            base=proof['caption'];owner=base;number=1
            while owner in out['components']:number+=1;owner=base+'_'+str(number)
            a,b,c,d=proof['frame'];out['components'][owner]={'type':'net_bidirection','Name':owner,'value':base,'bbox':[a,h-d,c,h-b]}
        name='OUT' if proof['caption'].upper().endswith('OUT') else 'IN'
        out['pins'][owner]={'pin_1':_pin(proof['point'],h,name)};changed.add(owner)
        events.append({'rule':'observed_oriented_port_contact','owner':owner,**proof})
    if len(events)>config['maximum_ports']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[]}
    if not proofs:return deepcopy(target),{'enabled':True,'status':'no_effect','changes':[]}
    completed=0;removed=[]
    for record in records['records']:
        owner=record['component_id'];comp=out['components'].get(owner,{})
        old=out['pins'].get(owner,{})
        if comp.get('type')!='r' or not 1<=len(old)<=2 or not set(old)<={'pin_1','pin_2'}:continue
        proof=resistor_contact_proof(rgb,record)
        if proof is None:continue
        key=sorted(old)[0];point=old[key]['point'];source=[point['x'],h-point['y']]
        ranks=sorted((math.dist(source,p),i) for i,p in enumerate(proof['terminals']))
        if ranks[0][0]>5 or ranks[1][0]-ranks[0][0]<8:continue
        missing=proof['terminals'][1-ranks[0][1]]
        near=[other for other,ps in out['pins'].items() if other!=owner and out['components'].get(other,{}).get('type')=='led'
              and any(math.dist(missing,[p['point']['x'],h-p['point']['y']])<=6 for p in ps.values())]
        if len(near)!=1:continue
        number='2' if key=='pin_1' else '1'
        if len(old)==2:
            p=old['pin_'+number]['point']
            if math.dist(missing,[p['x'],h-p['y']])>5:continue
        out['pins'][owner][key]['point']={'x':float(proof['terminals'][ranks[0][1]][0]),'y':float(h-proof['terminals'][ranks[0][1]][1])}
        out['pins'][owner]['pin_'+number]=_pin(missing,h,number);changed.add(owner);completed+=1
        events.append({'rule':'single_missing_resistor_contact','owner':owner,'number':number,'adjacent_owner':near[0],**proof})
        adjacent=[near[0]+'.'+key[4:] for key,p in out['pins'][near[0]].items()
                  if math.dist(missing,[p['point']['x'],h-p['point']['y']])<=6]
        if len(adjacent)==1:removed+=clear_shared_contact_slivers(out,{owner+'.'+number,adjacent[0]},missing,rgb)
    if completed>config['maximum_contacts']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[]}
    new_contacts=[(owner,key[4:]) for owner in sorted(changed) for key in out['pins'][owner]
                  if key not in target['pins'].get(owner,{})]
    routes=[event for owner,num in new_contacts if (event:=attach_contact(out,owner,num,rgb))]
    for owner,c in target['components'].items():
        if c['type'] in {'box','ic','switch','pin'} and (out['components'].get(owner)!=c or out['pins'].get(owner)!=target['pins'].get(owner)):
            raise ValueError('oriented_port_changed_protected_component')
    protected={ref for n in target['nets'].values() for ref in _refs(n) if target['components'].get(ref.rsplit('.',1)[0],{}).get('type') in {'box','ic','switch','pin'}}
    remaining=set().union(*(_refs(n) for n in out['nets'].values())) if out['nets'] else set()
    if not protected<=remaining:
        return deepcopy(target),{'enabled':True,'status':'withheld_protected_contacts','changes':[]}
    expected=deepcopy(target)
    for event in removed:del expected['nets'][event['network_id']]['edges'][event['edge_id']]
    if any(n['edges'] and out['nets'].get(nid,{}).get('edges')!=n['edges'] for nid,n in expected['nets'].items()):
        raise ValueError('oriented_port_changed_wire_geometry')
    return out,{'enabled':True,'status':'applied' if out!=target else 'no_effect','changes':events,'routes':routes,'removed_symbol_slivers':removed,'reference_read':False}
