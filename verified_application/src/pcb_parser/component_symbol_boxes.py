"""Recover symbol frames from image bars and emission arrows, without changing contacts."""
from copy import deepcopy
from itertools import combinations
import math
import cv2
import numpy as np

DEFAULT = {'enabled': False, 'maximum_changes': 64}


def _patch(mask, box, margin=6):
    h,w=mask.shape
    a,b,c,d=box
    x1,y1=max(0,math.floor(a)-margin),max(0,math.floor(b)-margin)
    x2,y2=min(w,math.ceil(c)+margin),min(h,math.ceil(d)+margin)
    return mask[y1:y2,x1:x2],x1,y1


def _components(mask):
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [(labels==i,stats[i],centers[i]) for i in range(1,count)]


def _ground_frame(red, record, contact=None):
    patch,ox,oy=_patch(red,record['bbox_top_left'],margin=12)
    proofs=[]
    for vertical in (True,False):
        # Work in coordinates where all comb teeth are vertical.
        image=patch if vertical else patch.T
        opened=cv2.morphologyEx(image.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,1),np.uint8))
        bars=[]
        for _,stats,_ in _components(opened):
            x,y,w,h,area=map(int,stats)
            if 1<=w<=5 and 5<=h<=100 and h>=w*1.5:
                bars.append({'x':x,'y':y,'w':w,'h':h,'center':y+(h-1)/2})
        bars.sort(key=lambda p:p['x'])
        for count in (4,3):
            for chosen in combinations(bars,count):
                for ordered in (chosen,chosen[::-1]):
                    positions=[p['x']+(p['w']-1)/2 for p in ordered]
                    gaps=np.abs(np.diff(positions));lengths=[p['h'] for p in ordered]
                    if not (len(gaps) and min(gaps)>=3 and max(gaps)<=13 and max(gaps)-min(gaps)<=2.5):continue
                    if not (lengths[0]>=15 and all(a-b>=3 for a,b in zip(lengths,lengths[1:]))):continue
                    if max(p['center'] for p in ordered)-min(p['center'] for p in ordered)>2.5:continue
                    if not (lengths[-1]<=lengths[0]*.65 and lengths[-1]>=3):continue
                    # Four teeth provide full support. A three-tooth symbol is
                    # allowed only when its observed lengths extrapolate to a
                    # short final tooth or to the natural end of the comb.
                    centers=np.array([p['center'] for p in ordered]);center=float(np.median(centers))
                    first=ordered[0];direction=1 if positions[-1]>positions[0] else -1
                    step=float(np.median(gaps))
                    left=min(p['x'] for p in ordered);right=max(p['x']+p['w'] for p in ordered)
                    top=min(p['y'] for p in ordered);bottom=max(p['y']+p['h'] for p in ordered)
                    extra=None
                    if count==3:
                        nextx=round(positions[-1]+direction*step)
                        if 1<=nextx<image.shape[1]-1:
                            col=image[:,nextx-1:nextx+2].any(1);yy=round(center)
                            if 0<=yy<len(col) and col[yy]:
                                lo=hi=yy
                                while lo>0 and col[lo-1]:lo-=1
                                while hi+1<len(col) and col[hi+1]:hi+=1
                                if 2<=hi-lo+1<lengths[-1]-1 and abs((lo+hi)/2-center)<=2:
                                    extra=[nextx-1,lo,nextx+2,hi+1]
                                    left=min(left,extra[0]);right=max(right,extra[2]);top=min(top,lo);bottom=max(bottom,hi+1)
                    # The lead must visibly leave the longest tooth, opposite
                    # the shorter teeth. Follow only this transverse red run.
                    cy=round(center);base=round(positions[0]);lead=base
                    line=image[max(0,cy-1):min(image.shape[0],cy+2)].any(0)
                    while 0<=lead-direction<image.shape[1] and line[lead-direction]:lead-=direction
                    lead_length=abs(lead-base)
                    if not 3<=lead_length<=80:continue
                    # A one-pixel break is not silently bridged.
                    if not line[min(lead,base):max(lead,base)+1].all():continue
                    # The red run can continue into a wire or junction. The
                    # existing contact anchors a search for a filled junction.
                    # Its observed center bounds a symbol lead; a plain wire
                    # contact never expands the comb frame.
                    contact_lead=None
                    if contact is not None:
                        px,py=(contact[0]-ox,contact[1]-oy) if vertical else (contact[1]-oy,contact[0]-ox)
                        distance=(base-px)*direction
                        pxi,pyi=round(px),round(py)
                        nodes=[]
                        if abs(py-center)<=4 and 0<=distance<=lead_length+10:
                            for q in range(max(2,pxi-10),min(image.shape[1]-2,pxi+11)):
                                if (base-q)*direction<5 or not line[q]:continue
                                junction=image[max(0,pyi-2):min(image.shape[0],pyi+3),q-2:q+3]
                                if junction.shape!=(5,5) or np.count_nonzero(junction)<22:continue
                                column=image[max(0,pyi-10):min(image.shape[0],pyi+11),q]
                                span=int(np.count_nonzero(column))
                                if 6<=span<=20:nodes.append((q,span))
                        if nodes:
                            widest=max(v for _,v in nodes)
                            contact_lead=float(np.mean([q for q,v in nodes if v>=widest-1]))
                    if contact_lead is not None:
                        left=min(left,contact_lead);right=max(right,contact_lead+1)
                    box=[left,top,right,bottom]
                    if not vertical:box=[top,left,bottom,right]
                    box=[float(box[0]+ox),float(box[1]+oy),float(box[2]+ox),float(box[3]+oy)]
                    proofs.append({'box_top_left':box,'bar_axis':'vertical' if vertical else 'horizontal',
                                   'bar_count':count+(extra is not None),'bar_lengths':lengths,
                                   'lead_length':lead_length,'contact_included':contact_lead is not None,'bar_centers':[[float(p['x']+(p['w']-1)/2+ox),float(p['center']+oy)] if vertical else
                                                  [float(p['center']+ox),float(p['x']+(p['w']-1)/2+oy)] for p in ordered]})
            if any(p['bar_count']==4 for p in proofs):break
    if not proofs:return None
    proofs.sort(key=lambda p:(-p['bar_count'],-max(p['bar_lengths'])))
    best=proofs[0]
    # Ambiguous independent combs in one source frame are never merged.
    if any(max(abs(a-b) for a,b in zip(p['box_top_left'],best['box_top_left']))>4 and p['bar_count']==best['bar_count'] for p in proofs[1:]):return None
    return best


def _runs(values):
    result=[]
    for value in values:
        if not result or value-result[-1][-1]>1:result.append([])
        result[-1].append(int(value))
    return result


def _led_frame(red,record):
    patch,ox,oy=_patch(red,record['bbox_top_left'])
    components=_components(patch);proofs=[]
    for vertical in (True,False):
        bodies=[];arrows=[]
        for pixels,stats,center in components:
            x,y,w,h,area=map(int,stats)
            # An arrow needs a diagonal shaft and a filled head, not a caption
            # character or an arbitrary nearby red speck.
            if 6<=w<=25 and 6<=h<=25 and .65<=w/h<=1.5 and 18<=area<=250 and .15<=area/(w*h)<=.7:
                yy,xx=np.nonzero(pixels);corr=np.corrcoef(xx,yy)[0,1]
                covariance,vectors=np.linalg.eigh(np.cov(np.array([xx,yy])))
                coordinates=np.stack([xx,yy],axis=1)@vectors
                along=coordinates[:,1];across=coordinates[:,0];middle=(along.min()+along.max())/2
                widths=[float(np.ptp(across[part])) for part in (along<=middle,along>middle) if np.count_nonzero(part)>=3]
                headed=(len(widths)==2 and max(widths)>=4 and max(widths)>=1.4*max(min(widths),.5))
                if headed and abs(corr)>=.35 and covariance[-1]/max(covariance[0],.1)>=1.7 and x>0 and y>0 and x+w<patch.shape[1] and y+h<patch.shape[0]:
                    arrows.append({'box':[x,y,x+w,y+h],'center':center,'corr':float(corr)})
            image=pixels if vertical else pixels.T
            spans=[]
            for row in image:
                indexes=np.flatnonzero(row);spans.append(int(indexes[-1]-indexes[0]+1) if len(indexes) else 0)
            if not spans:continue
            widest=max(spans)
            if not 12<=widest<=65:continue
            for run in _runs(np.flatnonzero(np.array(spans)>=max(6,widest*.24))):
                if not 9<=len(run)<=65 or not .45<=widest/len(run)<=2.5:continue
                if max(spans[i] for i in run)<1.5*min(spans[i] for i in run):continue
                region=image[run[0]:run[-1]+1];yy,xx=np.nonzero(region)
                if not len(xx):continue
                box=[int(xx.min()),run[0],int(xx.max())+1,run[-1]+1]
                if not vertical:box=[box[1],box[0],box[3],box[2]]
                bodies.append(box)
        for body in bodies:
            bx=(body[0]+body[2])/2;by=(body[1]+body[3])/2
            for pair in combinations(arrows,2):
                first,second=pair;delta=first['center']-second['center']
                if not (5<=abs(delta[0])<=22 and 5<=abs(delta[1])<=22 and .65<=abs(delta[0]/delta[1])<=1.55):continue
                if first['corr']*second['corr']<0:continue
                centers=[p['center'] for p in pair]
                sideways=[p[0]-bx if vertical else p[1]-by for p in centers]
                if sideways[0]*sideways[1]<=0 or not all(10<=abs(v)<=65 for v in sideways):continue
                if any(math.dist(p,[bx,by])>70 for p in centers):continue
                if any(not (p['box'][2]<=body[0] or p['box'][0]>=body[2] or p['box'][3]<=body[1] or p['box'][1]>=body[3]) for p in pair):continue
                boxes=[body]+[p['box'] for p in pair]
                box=[min(p[0] for p in boxes)+ox,min(p[1] for p in boxes)+oy,max(p[2] for p in boxes)+ox,max(p[3] for p in boxes)+oy]
                proofs.append({'box_top_left':list(map(float,box)),'arrow_count':2,'body_axis':'vertical' if vertical else 'horizontal',
                               'body_box_top_left':[float(body[0]+ox),float(body[1]+oy),float(body[2]+ox),float(body[3]+oy)],
                               'arrow_boxes_top_left':[[float(p['box'][0]+ox),float(p['box'][1]+oy),float(p['box'][2]+ox),float(p['box'][3]+oy)] for p in pair],
                               'distance':sum(math.dist(p,[bx,by]) for p in centers)})
    if not proofs:return None
    proofs.sort(key=lambda p:p['distance']);proof=proofs[0];proof.pop('distance')
    return proof


def recover_component_symbol_boxes(target,rgb,graph,records,image_sha256,*,cfg=None):
    """Return a copied prediction and provenance audit; only component bbox changes.

    Evidence coordinates use top-left; exported boxes use bottom-left. Geometry
    is accepted only for an existing LED/ground with a matching confident source.
    """
    config={**DEFAULT,**(cfg or {})};out=deepcopy(target)
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:raise ValueError('invalid_symbol_box_setting')
    if type(config['maximum_changes']) is not int or not 0<=config['maximum_changes']<=256:raise ValueError('invalid_symbol_box_budget')
    if not config['enabled']:return out,{'enabled':False,'status':'disabled','changes':[]}
    if not isinstance(rgb,np.ndarray) or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('invalid_symbol_box_image')
    h,w=rgb.shape[:2]
    for document in (graph,records):
        if (document.get('image_sha256')!=image_sha256 or document.get('size')!=[w,h]
                or document.get('coordinate_origin')!='top_left' or document.get('prediction_only') is not True):
            raise ValueError('symbol_box_evidence_differs_from_image')
    r,g,b=rgb.astype(np.int16).transpose(2,0,1)
    red=(r>g+40)&(r>b+40)&(g<170)&(b<170)
    events=[]
    for record in records.get('records',[]):
        owner=record.get('component_id');comp=out.get('components',{}).get(owner)
        if comp is None or comp.get('type') not in ('gnd','led') or record.get('type')!=comp['type']:continue
        if record.get('detection_confidence',0)<(.7 if comp['type']=='gnd' else .6):continue
        box=record.get('bbox_top_left')
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or not (box[0]<box[2] and box[1]<box[3]):continue
        # A source covering another existing physical identity cannot redefine
        # either child's frame. This also protects previously split combs.
        if comp['type']=='gnd' and any(other!=owner and item.get('type')==comp['type'] and
               min(box[2],item['bbox'][2])-max(box[0],item['bbox'][0])>0 and
               min(box[3],h-item['bbox'][1])-max(box[1],h-item['bbox'][3])>0
               for other,item in out['components'].items()):continue
        old=comp.get('bbox')
        if not old or len(old)!=4:continue
        # Full LED frames already include the arrows. Restrict this recovery to
        # compact fallback frames, keeping independently reconstructed frames.
        if comp['type']=='led' and max(old[2]-old[0],old[3]-old[1])>25:continue
        pins=list(out.get('pins',{}).get(owner,{}).values())
        contact=None
        if len(pins)==1:
            point=pins[0].get('point',{})
            if isinstance(point.get('x'),(int,float)) and isinstance(point.get('y'),(int,float)):
                contact=[float(point['x']),float(h-point['y'])]
        proof=_ground_frame(red,record,contact) if comp['type']=='gnd' else _led_frame(red,record)
        if proof is None:continue
        if comp['type']=='gnd' and max(proof['bar_lengths'])<2.5*max(old[2]-old[0],old[3]-old[1]):continue
        a,b,c,d=proof['box_top_left'];new=[a,float(h-d),c,float(h-b)]
        if new==old:continue
        out['components'][owner]['bbox']=new
        events.append({'owner':owner,'rule':'observed_ground_comb_frame' if comp['type']=='gnd' else 'observed_led_arrow_frame',
                       'source_record':owner,'source_confidence':record['detection_confidence'],'old_bbox':deepcopy(old),'bbox':new,**proof})
    audit={'enabled':True,'status':'applied' if events else 'no_effect','changes':events,'reference_read':False,
           'image_sha256':image_sha256,'size':[w,h],'coordinate_origin':'top_left','prediction_only':True}
    if len(events)>config['maximum_changes']:
        audit.update(status='withheld_budget',changes=[]);return deepcopy(target),audit
    return out,audit
