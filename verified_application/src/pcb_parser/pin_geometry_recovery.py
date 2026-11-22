"""Image-supported missing IC leads and shared ground-contact correction.

Consumes current-image predictions only. Does not infer numbering order,
change existing fields, create networks, or draw wire segments.
"""
from collections import defaultdict
from copy import deepcopy
import math
import statistics

import cv2
import numpy as np

IC_TYPES = {'ic', 'box', 'connector', 'block', 'raw_block'}
DEFAULT = {'min_frame_side': 40, 'frame_search_px': 22, 'frame_support': .94,
           'max_components': 6, 'max_proposals': 32, 'max_ocr_calls': 96,
           'min_confidence': .98, 'min_views': 2}


def side_of(point, box):
    x,y=point;a,b,c,d=box
    outside={'left':a-x,'right':x-c,'top':b-y,'bottom':y-d}
    outward={s:v for s,v in outside.items() if v>0}
    return max(outward,key=outward.get) if outward else min(outside,key=lambda s:abs(outside[s]))


def vote(tokens, cfg):
    support=defaultdict(set)
    for t in tokens:
        if t['confidence']>=cfg['min_confidence']:support[t['text']].add(t['view'])
    rank=sorted(support,key=lambda k:(-len(support[k]),k));counts={k:len(v) for k,v in support.items()}
    if not rank or len(support[rank[0]])<cfg['min_views']:return None,counts
    if len(rank)>1 and len(support[rank[1]])>=cfg['min_views']:return None,counts
    return rank[0],counts


class LeadEvidence:
    def __init__(self, rgb, cfg):
        self.rgb=rgb;self.cfg=cfg;self.height,self.width=rgb.shape[:2]
        colors=np.asarray(rgb,dtype=np.int16);r,g,b=[colors[:,:,i] for i in range(3)]
        self.red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
        self.green=(g>r+20)&(g>b+20)&(g>60)

    def hit(self,x,y):
        x,y=round(x),round(y)
        return 0<=x<self.width and 0<=y<self.height and bool(self.red[max(0,y-1):y+2,max(0,x-1):x+2].any())

    def frame(self,box):
        a,b,c,d=box
        if not all(math.isfinite(v) for v in box) or min(c-a,d-b)<self.cfg['min_frame_side']:return None
        positions={};scores={};radius=self.cfg['frame_search_px']
        for side,edge in [('left',a),('right',c),('top',b),('bottom',d)]:
            rows=[]
            for v in range(round(edge)-radius,round(edge)+radius+1):
                if side in {'left','right'}:
                    if not 0<=v<self.width:continue
                    patch=self.red[max(0,int(b)+7):min(self.height,int(d)-7),max(0,v-1):min(self.width,v+2)]
                    axis=1
                else:
                    if not 0<=v<self.height:continue
                    patch=self.red[max(0,v-1):min(self.height,v+2),max(0,int(a)+7):min(self.width,int(c)-7)];axis=0
                if patch.size:rows.append((float(patch.any(axis=axis).mean()),v))
            if not rows:return None
            best=max(v[0] for v in rows)
            if best<self.cfg['frame_support']:return None
            plateau=[v for score,v in rows if score>=best-.005]
            if max(plateau)-min(plateau)>6:return None
            positions[side]=statistics.median(plateau);scores[side]=best
        result=[positions[s] for s in ['left','top','right','bottom']]
        if not .5*(c-a)<=result[2]-result[0]<=1.5*(c-a) or not .5*(d-b)<=result[3]-result[1]<=1.5*(d-b):return None
        return result,scores

    def leads(self,box,side):
        edge=box[0] if side=='left' else box[2];step=-1 if side=='left' else 1;rows=[]
        for y in range(math.ceil(box[1])+6,math.floor(box[3])-5):
            if sum(self.hit(edge+step*d,y) for d in range(3,11))<6:continue
            last=0;gap=0
            for d in range(1,66):
                if self.hit(edge+step*d,y):last=d;gap=0
                else:gap+=1
                if gap>2:break
            if 8<=last<64:rows.append((y,last))
        clusters=[]
        for row in rows:
            if not clusters or row[0]-clusters[-1][-1][0]>1:clusters.append([])
            clusters[-1].append(row)
        return [{'row':statistics.median(v[0] for v in cluster),
                 'length':statistics.median(v[1] for v in cluster),'side':side,'edge':edge,
                 'point':[edge+step*statistics.median(v[1] for v in cluster),statistics.median(v[0] for v in cluster)]}
                for cluster in clusters if len(cluster)<=7]

    def ground_anchors(self,point,side,box,target):
        edge=box[1] if side=='top' else box[3];step=-1 if side=='top' else 1;anchors=[]
        for owner,comp in sorted(target['components'].items()):
            if comp['type']!='gnd':continue
            for key,pin in sorted(target['pins'].get(owner,{}).items()):
                x,y=pin['point']['x'],self.height-pin['point']['y'];distance=step*(y-edge)
                if abs(x-point[0])>3 or not 12<=distance<=80:continue
                samples=[self.hit(point[0],edge+step*d) for d in range(3,max(4,round(distance)+1))]
                if sum(samples)/len(samples)>=.94:anchors.append((owner,key,pin))
        return anchors


def _members(net):
    return {v.strip() for v in net['hyperGraph'].strip('()').split(',') if v.strip()}


def _segment_distance(point,a,b):
    vx,vy=b[0]-a[0],b[1]-a[1];den=vx*vx+vy*vy
    t=max(0,min(1,((point[0]-a[0])*vx+(point[1]-a[1])*vy)/den)) if den else 0
    return math.hypot(a[0]+t*vx-point[0],a[1]+t*vy-point[1])


def sync_networks(out, graph, changes, evidence):
    """Attach only to a unique existing model net, preserving every wire."""
    attachments={a['candidate_id']:a for a in graph.get('pin_attachments',[])}
    net_edges=defaultdict(list);nets={n['network_id']:n for n in graph.get('networks',[])}
    edge_map={e['edge_id']:e for e in graph.get('edges',[])}
    for nid,net in nets.items():
        for eid in net.get('edge_ids',[]):
            poly=edge_map.get(eid,{}).get('polyline',[])
            net_edges[nid].extend(zip(poly,poly[1:]))
    audit=[]
    for change in changes:
        ref=f"{change['owner']}.{change['number']}";destinations=[]
        current=[nid for nid,n in out['nets'].items() if ref in _members(n)]
        if current:continue
        if change['status']=='unique_collinear_ground_contact':
            ground_ref=change['ground_ref']
            destinations=[nid for nid,n in out['nets'].items() if ground_ref in _members(n)]
        elif change.get('source_candidate_id'):
            a=attachments.get(change['source_candidate_id'],{})
            if (a.get('attachment_status')=='attached' and a.get('point')==change['point']
                    and change['source_candidate_id'] in nets.get(a.get('network_id'),{}).get('pin_candidate_ids',[])):
                destinations=[a['network_id']]
        else:
            if str(change.get('name','')).upper() in {'NC','N.C.','DNC','N/C'}:
                audit.append({'member':ref,'status':'withheld','reason':'no_connect_field_without_model_attachment'});continue
            x,y=map(round,change['point'])
            if not evidence.green[max(0,y-4):y+5,max(0,x-4):x+5].any():
                audit.append({'member':ref,'status':'withheld','reason':'no_visible_green_wire'});continue
            nearby=[]
            for nid,segments in net_edges.items():
                if segments:
                    distance=min(_segment_distance(change['point'],a,b) for a,b in segments)
                    if distance<=3:nearby.append((distance,nid))
            nearby.sort()
            if len(nearby)==1 or len(nearby)>1 and nearby[1][0]-nearby[0][0]>=3:
                destinations=[nearby[0][1]]
        if len(destinations)!=1 or destinations[0] not in out['nets']:
            audit.append({'member':ref,'status':'withheld','reason':'no_unique_existing_exported_net','candidate_networks':destinations});continue
        nid=destinations[0]
        if nets.get(nid,{}).get('unresolved_junction_ids'):
            audit.append({'member':ref,'status':'withheld','reason':'unresolved_junctions','network_id':nid});continue
        before=sorted(_members(out['nets'][nid]));after=sorted(set(before)|{ref})
        out['nets'][nid]['hyperGraph']='('+','.join(after)+')'
        audit.append({'member':ref,'network_id':nid,'before':before,'after':after})
    return audit


def recover_image_leads(target,document,graph,records,rgb,*,reader,cfg=None):
    cfg={**DEFAULT,**(cfg or {})};evidence=LeadEvidence(rgb,cfg);height,width=rgb.shape[:2]
    out=deepcopy(target);audit={'enabled':True,'policy':'image_lead_geometry_v1','frames':{},
                               'proposals':[],'changes':[],'network_changes':[],'ocr_calls':0,'budget_exhausted':False}
    by_owner={r['component_id']:r for r in records};ocr_owners=0
    def read(image,box,scale,rotation,recognition_only,view):
        if audit['ocr_calls']>=cfg['max_ocr_calls']:
            audit['budget_exhausted']=True;return []
        audit['ocr_calls']+=1
        return reader(image,box,scale,rotation,recognition_only,view)
    def number_glyph(column,expected,owner):
        x1=max(0,int(column-13));x2=min(width,int(column+14));y1=max(0,int(expected-13));y2=min(height,int(expected+14))
        patch=rgb[y1:y2,x1:x2];colors=np.asarray(patch,dtype=np.int16);rr,gg,bb=[colors[:,:,i] for i in range(3)]
        mask=(((rr+gg+bb)/3<205)&(abs(rr-gg)<35)&(abs(rr-bb)<35)&(abs(gg-bb)<35)).astype(np.uint8)
        mask=mask&(1-cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,13),np.uint8)))
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);tokens=[]
        for i in range(1,count):
            a,b,w,h,area=map(int,stats[i]);cx,cy=centroids[i]
            if not(2<=w<=20 and 6<=h<=22 and area>=8):continue
            if abs(x1+cx-column)>8 or abs(y1+cy-expected)>9:continue
            if a==0 or b==0 or a+w>=patch.shape[1] or b+h>=patch.shape[0]:continue
            isolated=np.full((h+16,w+16,3),255,dtype=np.uint8);chosen=labels[b:b+h,a:a+w]==i
            isolated[8:8+h,8:8+w][chosen]=patch[b:b+h,a:a+w][chosen]
            for scale in [3.,5.]:
                readings=read(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'lead_glyph_{owner}_{expected}_{i}_s{scale}')
                tokens.extend({**t,'bbox':[x1+a,y1+b,x1+a+w,y1+b+h]} for t in readings)
        return tokens
    for owner,component in sorted(target['components'].items()):
        if component['type'] not in IC_TYPES:continue
        a,b,c,d=component['bbox'];found=evidence.frame([a,height-d,c,height-b])
        if found is None:continue
        box,scores=found;audit['frames'][owner]={'box':box,'scores':scores}
        existing=[(key,p,[p['point']['x'],height-p['point']['y']]) for key,p in target['pins'].get(owner,{}).items()]
        pending=[]
        for side in ['left','right']:
            known=[item for item in existing if side_of(item[2],box)==side]
            pending.extend(p for p in evidence.leads(box,side) if not any(abs(pt[1]-p['row'])<=6 for key,pin,pt in known))
        texts=[]
        if pending and owner in by_owner and ocr_owners<cfg['max_components']:
            ocr_owners+=1;old=by_owner[owner]['bbox_top_left']
            crop=[max(0,math.floor(old[0]-32)),max(0,math.floor(old[1]-32)),min(width,math.ceil(old[2]+32)),min(height,math.ceil(old[3]+32))]
            for scale in [1.,3.]:
                for rotation in [0,90]:texts.extend(read(rgb,crop,scale,rotation,False,f'lead_component_{owner}_s{scale}_r{rotation}'))
        for proposal in pending:
            if len(audit['proposals'])>=cfg['max_proposals']:break
            side=proposal['side'];row=proposal['row'];edge=proposal['edge']
            known=[item for item in existing if side_of(item[2],box)==side]
            def number_band(cx):return -30<=cx-edge<=4 if side=='left' else -4<=cx-edge<=30
            numbers=[t for t in texts if t['text'].isdigit() and t['confidence']>=cfg['min_confidence'] and number_band((t['bbox'][0]+t['bbox'][2])/2)]
            offsets=[];columns=[]
            for key,pin,pt in known:
                for t in numbers:
                    cy=(t['bbox'][1]+t['bbox'][3])/2
                    if t['text']==key[4:] and abs(cy-pt[1])<=16:
                        offsets.append(cy-pt[1]);columns.append((t['bbox'][0]+t['bbox'][2])/2)
            if not offsets:continue
            offset=statistics.median(offsets);column=statistics.median(columns);names=[]
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                inward=cx-box[0] if side=='left' else box[2]-cx
                if not any(ch.isalpha() for ch in t['text']) or not box[0]<cx<box[2] or not box[1]<cy<box[3]:continue
                if 3<=inward<=.55*(box[2]-box[0]) and abs(cy-row)<=12:names.append(t)
            name,name_support=vote(names,cfg)
            candidates=[t for t in numbers if abs((t['bbox'][1]+t['bbox'][3])/2-row-offset)<=7]
            number,number_support=vote(candidates,cfg)
            if name and number is None:
                glyphs=number_glyph(column,row+offset,owner)
                number,number_support=vote([t for t in glyphs if t['text'].isdigit()],cfg)
            entry={'owner':owner,**proposal,'name':name,'number':number,'name_support':name_support,'number_support':number_support,'number_offset':offset}
            if not name or number is None:entry['status']='insufficient_fields';audit['proposals'].append(entry);continue
            limit=max(4,round(1.5*(len(existing)+len(pending))))
            if int(number)>limit:entry['status']='number_beyond_observed_capacity';audit['proposals'].append(entry);continue
            if 'pin_'+number in out['pins'].get(owner,{}):entry['status']='existing_number';audit['proposals'].append(entry);continue
            eligible=[]
            for p in document['pin_candidates']:
                if p.get('owner_status')!='ambiguous_owner':continue
                if not any(o['component_id']==owner and not o.get('warning') for o in p.get('owner_candidates',[])):continue
                x,y=p['point'];outward=box[0]-x if side=='left' else x-box[2]
                if abs(y-row)<=3 and 6<=outward<=proposal['length']+3:eligible.append(p)
            if len(eligible)==1:
                entry['point']=eligible[0]['point'];entry['source_candidate_id']=eligible[0]['candidate_id'];entry['status']='owner_resolved_by_lead_and_text'
            elif not eligible:entry['status']='image_lead_endpoint'
            else:entry['status']='ambiguous_candidates';audit['proposals'].append(entry);continue
            x,y=entry['point'];out['pins'].setdefault(owner,{})['pin_'+number]={'pinname':name,'point':{'x':float(x),'y':float(height-y)}}
            audit['proposals'].append(entry);audit['changes'].append(entry)
        for side in ['top','bottom']:
            pins=[item for item in existing if side_of(item[2],box)==side]
            if len(pins)!=1:continue
            key,pin,point=pins[0];anchors=evidence.ground_anchors(point,side,box,target)
            if len(anchors)!=1:continue
            ground,ground_key,q=anchors[0];before=out['pins'][owner][key]['point']
            if before==q['point']:continue
            out['pins'][owner][key]['point']=deepcopy(q['point'])
            audit['changes'].append({'owner':owner,'number':key[4:],'name':pin['pinname'],'status':'unique_collinear_ground_contact',
                                     'ground_ref':ground+'.'+ground_key[4:],'before':before,'after':q['point']})
    audit['network_changes']=sync_networks(out,graph,audit['changes'],evidence)
    return out,audit
