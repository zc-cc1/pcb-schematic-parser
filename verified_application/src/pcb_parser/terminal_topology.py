"""Observe short mixed-color ground branches and isolated no-connect markers."""
import math
import cv2
import numpy as np


def refs(net):
    return {s.strip() for s in str(net.get('hyperGraph') or '').strip('()').split(',') if s.strip()}


def conductive(draft):
    r,g,b=[draft.rgb[:,:,i].astype(np.int16) for i in range(3)]
    return (np.maximum(r,g)>b+30)&(np.maximum(r,g)>70)


def connected_corridor(color, a, b):
    """The actual ink must stay connected; nearby pixels cannot bridge white gaps."""
    left=max(0,math.floor(min(a[0],b[0]))-3);top=max(0,math.floor(min(a[1],b[1]))-3)
    right=min(color.shape[1],math.ceil(max(a[0],b[0]))+4);bottom=min(color.shape[0],math.ceil(max(a[1],b[1]))+4)
    corridor=np.zeros((bottom-top,right-left),np.uint8)
    first=[round(a[0])-left,round(a[1])-top];last=[round(b[0])-left,round(b[1])-top]
    cv2.line(corridor,tuple(first),tuple(last),1,5)
    _,labels=cv2.connectedComponents((color[top:bottom,left:right]&corridor.astype(bool)).astype('uint8'),8)
    def ids(point):
        x,y=point
        return set(np.unique(labels[max(0,y-2):y+3,max(0,x-2):x+3]))-{0}
    return bool(ids(first)&ids(last))


def ground_branch_proof(draft, first, second):
    owner,key=first.rsplit('.',1);go,gkey=second.rsplit('.',1)
    comp=draft.out['components'][owner];ground=draft.out['components'][go]
    if comp['type']!='c' or ground['type']!='gnd':return None
    if set(draft.out['pins'].get(owner,{}))!={'pin_1','pin_2'}:return None
    if len(draft.out['pins'].get(go,{}))!=1:return None
    a=draft.point(draft.out['pins'][owner]['pin_'+key]);b=draft.point(draft.out['pins'][go]['pin_'+gkey])
    distance=math.dist(a,b);axis=int(abs(a[1]-b[1])>abs(a[0]-b[0]));cross=1-axis
    if not 8<=distance<=48 or abs(a[cross]-b[cross])>3:return None
    record=draft.rmap.get(go)
    if not record or record['type']!='gnd' or record.get('detection_confidence',0)<.7:return None
    source=[p for p in draft.pin_document['pin_candidates'] if p.get('owner')==go
            and p.get('owner_status')=='assigned' and not p.get('text_hypothesis_primary_candidate_id')
            and p.get('confidence',0)>=.8 and math.dist(p['point'],b)<1]
    if len(source)!=1:return None
    for o,p,q in [(owner,a,b),(go,b,a)]:
        x1,y1,x2,y2=draft.box(draft.out['components'][o]);center=[(x1+x2)/2,(y1+y2)/2]
        if np.dot(np.subtract(p,center),np.subtract(q,p))<=0:return None
    samples=np.rint(np.linspace(a,b,math.ceil(distance)+1)).astype(int)
    color=conductive(draft)
    if any(not(2<=x<draft.w-2 and 2<=y<draft.h-2) for x,y in samples):return None
    if any(not color[y-2:y+3,x-2:x+3].any() for x,y in samples):return None
    if not connected_corridor(color,a,b):return None
    # Demand a real green run and a red ground lead, not an arbitrary colored label.
    green_support=[bool(draft.green[y-1:y+2,x-1:x+2].any()) for x,y in samples]
    if sum(green_support)<5:return None
    x,y=map(round,b)
    if not draft.red[y-2:y+3,x-2:x+3].any():return None
    # A branch/junction or intervening component requires the normal graph stage.
    for x,y in samples[4:-4]:
        for sign in (-1,1):
            q=[x,y];q[cross]+=sign*6;xx,yy=q
            if color[max(0,yy-1):yy+2,max(0,xx-1):xx+2].any():return None
        for o,c in draft.out['components'].items():
            if o in {owner,go}:continue
            x1,y1,x2,y2=draft.box(c)
            if x1-2<=x<=x2+2 and y1-2<=y<=y2+2:return None
    return {'members':[first,second],'points':[a,b],'axis':axis,
            'source_candidate':source[0]['candidate_id'],'colored_support':1.,
            'connected_ink':True,'green_samples':sum(green_support),'distance_px':distance}


def recover_ground_branches(draft):
    occupied=set().union(*(refs(n) for n in draft.out['nets'].values())) if draft.out['nets'] else set()
    caps=[o+'.'+k[4:] for o,ps in draft.out['pins'].items() if draft.out['components'][o]['type']=='c'
          for k in ps if o+'.'+k[4:] not in occupied]
    grounds=[o+'.'+k[4:] for o,ps in draft.out['pins'].items() if draft.out['components'][o]['type']=='gnd'
             for k in ps if o+'.'+k[4:] not in occupied]
    choices={}
    for cap in caps:
        values=[p for ground in grounds if (p:=ground_branch_proof(draft,cap,ground))]
        if len(values)==1:choices[cap]=values[0]
    counts={g:sum(p['members'][1]==g for p in choices.values()) for g in grounds}
    for cap,proof in sorted(choices.items()):
        ground=proof['members'][1]
        if counts[ground]!=1:continue
        nid='MIXED_GROUND_'+cap.replace('.','_')
        if nid in draft.out['nets']:continue
        a,b=proof['points']
        draft.out['nets'][nid]={'hyperGraph':'('+','.join(sorted(proof['members']))+')',
            'edges':{'edge_1':[{'x':float(p[0]),'y':float(draft.h-p[1])} for p in (a,b)]}}
        draft.events.append({'rule':'mixed_color_ground_branch','network_id':nid,**proof})


def no_connect_marker(draft, ref):
    owner,key=ref.rsplit('.',1);comp=draft.out['components'][owner]
    if comp['type'] not in {'ic','box'}:return None
    p=draft.point(draft.out['pins'][owner]['pin_'+key]);x,y=map(round,p)
    if not(20<=x<draft.w-20 and 20<=y<draft.h-20):return None
    box=draft.box(comp);center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
    axis=int(abs(p[1]-center[1])>abs(p[0]-center[0]));cross=1-axis
    direction=1 if p[axis]>center[axis] else -1;boundary=box[axis+2] if direction>0 else box[axis]
    if not 5<=direction*(p[axis]-boundary)<=65:return None
    roi=draft.green[y-18:y+19,x-18:x+19].astype('uint8')
    _,labels,stats,_=cv2.connectedComponentsWithStats(roi,8);found=[]
    for label,(a,b,w,h,area) in enumerate(stats[1:],1):
        a,b,w,h,area=map(int,(a,b,w,h,area))
        if not (9<=w<=25 and 9<=h<=25 and .75<=w/h<=1.33):continue
        if a==0 or b==0 or a+w>=37 or b+h>=37:continue
        cx,cy=a+(w-1)/2,b+(h-1)/2
        if math.dist([cx,cy],[18,18])>5:return None
        yy,xx=np.where(labels==label);dx=xx-cx;dy=yy-cy
        diagonal=np.minimum(abs(dx-dy),abs(dx+dy))<=2.6
        if diagonal.mean()<.9:continue
        if any(np.count_nonzero((dx*sx>=3)&(dy*sy>=3))<5 for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]):continue
        if np.count_nonzero(((abs(dx)<1.5)&(abs(dy)>5))|((abs(dy)<1.5)&(abs(dx)>5)))>2:continue
        marker=[x-18+cx,y-18+cy]
        if abs(marker[cross]-p[cross])>3:continue
        # The X is at the end of the red IC lead, with white space beyond it.
        start=p.copy();start[axis]=boundary+direction*2;end=p.copy();end[axis]=marker[axis]-direction*5
        points=np.rint(np.linspace(start,end,max(2,math.ceil(math.dist(start,end))+1))).astype(int)
        if sum(draft.red[yy-2:yy+3,xx-2:xx+3].any() for xx,yy in points)/len(points)<.95:continue
        colors=conductive(draft);extent=(w,h)[axis]/2
        beyond=[]
        for step in range(math.ceil(extent)+3,math.ceil(extent)+11):
            q=marker.copy();q[axis]+=direction*step;xx,yy=map(round,q)
            if not(2<=xx<draft.w-2 and 2<=yy<draft.h-2):return None
            beyond.append(bool(colors[yy-1:yy+2,xx-1:xx+2].any()))
        if any(beyond):continue
        found.append({'member':ref,'point':p,'marker_center':marker,
                      'marker_box':[x-18+a,y-18+b,x-18+a+w,y-18+b+h],
                      'axis':axis,'direction':direction,'lead_boundary':boundary,
                      'diagonal_fraction':float(diagonal.mean()),'green_pixels':area})
    return found[0] if len(found)==1 else None


def isolated_nc_net(draft,net,proof):
    if refs(net)!={proof['member']} or not net.get('edges'):return False
    axis=proof['axis'];cross=1-axis;direction=proof['direction'];p=proof['point'];boundary=proof['lead_boundary']
    for edge in net['edges'].values():
        if not isinstance(edge,list) or len(edge)<2:return False
        for v in edge:
            q=[v['x'],draft.h-v['y']]
            if abs(q[cross]-p[cross])>3 or direction*(q[axis]-boundary)<-2 or direction*(q[axis]-p[axis])>4:return False
    return True


def prune_no_connect_nets(draft):
    candidates={next(iter(refs(n))) for n in draft.out['nets'].values() if len(refs(n))==1}
    for ref in sorted(candidates):
        proof=no_connect_marker(draft,ref)
        if not proof:continue
        related=[(nid,n) for nid,n in draft.out['nets'].items() if ref in refs(n)]
        if not related or not all(isolated_nc_net(draft,n,proof) for _,n in related):continue
        ids=sorted(nid for nid,_ in related)
        for nid in ids:del draft.out['nets'][nid]
        draft.events.append({'rule':'image_no_connect_marker','network_ids':ids,'members':[ref],**proof})
