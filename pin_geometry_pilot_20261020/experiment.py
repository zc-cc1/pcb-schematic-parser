"""Page-one image-supported lead recovery pilot; reference-free generation."""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,'D:/pcb5_submission/src')
from pcb_parser.pin_text_ocr import create_engine
from pcb_parser.pin_field_recovery import recognize
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
baseline=read(WORK/'pin_field_implementation_20261020/deployed/1_Pred.json')
trace=next(r for r in read(WORK/'pin_cases_1_11_20261020/trace.json') if r['case']=='1')
model=read(Path(trace['folder'])/'p/export/engineering_pin_text_result.json')
image=np.asarray(Image.open('D:/pcb_open_gt_in/1.png').convert('RGB'));height,width=image.shape[:2]
r,g,b=[image[:,:,i].astype(int) for i in range(3)]
red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
green=(g>r+20)&(g>b+20)&(g>60)
whole=[{**t,'owner':v['owner'],'rotation':v['rotation']} for v in read(ROOT/'collection.json')['views'] for t in v['texts']]
engine=None;new_calls=0;digit_reads=[]

def side(point,box):
    x,y=point;a,b,c,d=box
    distances={'left':abs(x-a),'right':abs(x-c),'top':abs(y-b),'bottom':abs(y-d)}
    return min(distances,key=distances.get)

def frame(box):
    a,b,c,d=box;positions={};scores={}
    for s,edge in [('left',a),('right',c),('top',b),('bottom',d)]:
        rows=[]
        for v in range(round(edge)-22,round(edge)+23):
            patch=red[max(0,int(b)+7):min(height,int(d)-7),max(0,v-1):min(width,v+2)] if s in {'left','right'} else red[max(0,v-1):min(height,v+2),max(0,int(a)+7):min(width,int(c)-7)]
            if patch.size:rows.append((float(patch.any(axis=1 if s in {'left','right'} else 0).mean()),v))
        best=max(v[0] for v in rows)
        if best<.94:return None
        plateau=[v for score,v in rows if score>=best-.005]
        if max(plateau)-min(plateau)>6:return None
        positions[s]=statistics.median(plateau);scores[s]=best
    output=[positions[s] for s in ['left','top','right','bottom']]
    if output[2]-output[0]<.5*(c-a) or output[3]-output[1]<.5*(d-b):return None
    return output,scores

def red_hit(x,y):
    x,y=round(x),round(y)
    return 0<=x<width and 0<=y<height and red[max(0,y-1):y+2,max(0,x-1):x+2].any()

def leads(box,s):
    edge=box[0] if s=='left' else box[2];step=-1 if s=='left' else 1;rows=[]
    for y in range(math.ceil(box[1])+6,math.floor(box[3])-5):
        if sum(red_hit(edge+step*d,y) for d in range(3,11))<6:continue
        last=0;gap=0
        for d in range(1,66):
            if red_hit(edge+step*d,y):last=d;gap=0
            else:gap+=1
            if gap>2:break
        if 8<=last<64:rows.append((y,last))
    clusters=[]
    for row in rows:
        if not clusters or row[0]-clusters[-1][-1][0]>1:clusters.append([])
        clusters[-1].append(row)
    return [{'row':statistics.median(v[0] for v in cluster),
             'length':statistics.median(v[1] for v in cluster),'side':s,
             'edge':edge,'point':[edge+step*statistics.median(v[1] for v in cluster),statistics.median(v[0] for v in cluster)]}
            for cluster in clusters if len(cluster)<=7]

def vote(tokens):
    support=defaultdict(set)
    for t in tokens:
        if t['confidence']>=.98:support[t['text']].add(t['view'])
    rank=sorted(support,key=lambda k:(-len(support[k]),k))
    if not rank or len(support[rank[0]])<2:return None,{k:len(v) for k,v in support.items()}
    if len(rank)>1 and len(support[rank[1]])>=2:return None,{k:len(v) for k,v in support.items()}
    return rank[0],{k:len(v) for k,v in support.items()}

def isolated_number(owner,column,expected):
    global engine,new_calls
    x1=max(0,int(column-13));x2=min(width,int(column+14));y1=max(0,int(expected-13));y2=min(height,int(expected+14))
    patch=image[y1:y2,x1:x2];rr,gg,bb=[patch[:,:,i].astype(int) for i in range(3)]
    mask=(((rr+gg+bb)/3<205)&(abs(rr-gg)<35)&(abs(rr-bb)<35)&(abs(gg-bb)<35)).astype(np.uint8)
    mask=mask&(1-cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,13),np.uint8)))
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);out=[]
    for i in range(1,count):
        a,b,w,h,area=map(int,stats[i]);cx,cy=centroids[i]
        if not(2<=w<=20 and 6<=h<=22 and area>=8):continue
        if abs(x1+cx-column)>8 or abs(y1+cy-expected)>9:continue
        if a==0 or b==0 or a+w>=patch.shape[1] or b+h>=patch.shape[0]:continue
        isolated=np.full((h+16,w+16,3),255,dtype=np.uint8);chosen=labels[b:b+h,a:a+w]==i
        isolated[8:8+h,8:8+w][chosen]=patch[b:b+h,a:a+w][chosen]
        if engine is None:engine,_=create_engine({'onnx_threads':4})
        for scale in [3.,5.]:
            out.extend(recognize(engine,isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'isolated_{owner}_{expected}_{i}_s{scale}'))
            new_calls+=1
    digit_reads.append({'owner':owner,'column':column,'row':expected,'readings':out})
    return out

output=copy.deepcopy(baseline);audit=[];frames={};proposals=[]
# This is a scoped pilot on the three already diagnosed components. No pin
# numbers, reference coordinates, or missing-row indices are supplied.
for owner in ['U1','U2','USB1']:
    a,b,c,d=baseline['components'][owner]['bbox'];found=frame([a,height-d,c,height-b])
    if found is None:continue
    box,scores=found;frames[owner]={'box':box,'scores':scores}
    texts=[t for t in whole if t['owner']==owner]
    existing=[(key,p,[p['point']['x'],height-p['point']['y']]) for key,p in baseline['pins'].get(owner,{}).items()]
    for s in ['left','right']:
        row_proposals=leads(box,s)
        known=[(key,p,pt) for key,p,pt in existing if side(pt,box)==s]
        edge=box[0] if s=='left' else box[2]
        def in_number_band(cx):return -30<=cx-edge<=4 if s=='left' else -4<=cx-edge<=30
        number_texts=[t for t in texts if t['text'].isdigit() and t['confidence']>=.98 and in_number_band((t['bbox'][0]+t['bbox'][2])/2)]
        offsets=[];columns=[]
        for key,p,pt in known:
            for t in number_texts:
                cy=(t['bbox'][1]+t['bbox'][3])/2
                if t['text']==key[4:] and abs(cy-pt[1])<=16:
                    offsets.append(cy-pt[1]);columns.append((t['bbox'][0]+t['bbox'][2])/2)
        if not offsets:continue
        offset=statistics.median(offsets);column=statistics.median(columns)
        for proposal in row_proposals:
            row=proposal['row']
            if any(abs(pt[1]-row)<=6 for key,p,pt in known):continue
            name_texts=[]
            for t in texts:
                cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
                inward=cx-box[0] if s=='left' else box[2]-cx
                if not any(ch.isalpha() for ch in t['text']) or not box[0]<cx<box[2] or not box[1]<cy<box[3]:continue
                if 3<=inward<=.55*(box[2]-box[0]) and abs(cy-row)<=12:name_texts.append(t)
            name,name_support=vote(name_texts)
            ns=[t for t in number_texts if abs((t['bbox'][1]+t['bbox'][3])/2-row-offset)<=7]
            number,number_support=vote(ns)
            if name and number is None:
                readings=isolated_number(owner,column,row+offset)
                number,number_support=vote([t for t in readings if t['text'].isdigit()])
            entry={'owner':owner,**proposal,'name':name,'number':number,
                   'name_support':name_support,'number_support':number_support,'number_offset':offset}
            if not name or number is None:entry['status']='insufficient_fields';proposals.append(entry);continue
            key='pin_'+number
            if key in output['pins'].get(owner,{}):entry['status']='existing_number';proposals.append(entry);continue
            # Prefer an existing ambiguous tip only when the current component
            # is explicitly among its eligible owners and it lies on this lead.
            candidates=[]
            for p in model['pin_candidates']:
                if p.get('owner_status')!='ambiguous_owner':continue
                if not any(o['component_id']==owner and not o.get('warning') for o in p['owner_candidates']):continue
                x,y=p['point'];outward=box[0]-x if s=='left' else x-box[2]
                if abs(y-row)<=3 and 6<=outward<=proposal['length']+3:candidates.append(p)
            if len(candidates)==1:
                entry['point']=candidates[0]['point'];entry['source_candidate_id']=candidates[0]['candidate_id']
                entry['status']='owner_resolved_by_lead_and_text'
            elif not candidates:entry['status']='image_lead_endpoint'
            else:entry['status']='ambiguous_candidates';proposals.append(entry);continue
            x,y=entry['point'];output['pins'].setdefault(owner,{})[key]={'pinname':name,'point':{'x':float(x),'y':float(height-y)}}
            proposals.append(entry);audit.append(entry)
    # A sole vertical IC lead may share its visible red line with a ground
    # symbol. An existing unique ground contact is a better joint anchor than
    # a component-wide side-length median.
    for s in ['top','bottom']:
        pins=[(key,p,pt) for key,p,pt in existing if side(pt,box)==s]
        if len(pins)!=1:continue
        key,p,pt=pins[0];anchors=[];edge=box[1] if s=='top' else box[3];step=-1 if s=='top' else 1
        for ground,comp in baseline['components'].items():
            if comp['type']!='gnd':continue
            for q in baseline['pins'].get(ground,{}).values():
                x,y=q['point']['x'],height-q['point']['y'];distance=step*(y-edge)
                if abs(x-pt[0])>3 or not 12<=distance<=80:continue
                samples=[red_hit(pt[0],edge+step*d) for d in range(3,max(4,round(distance)+1))]
                if sum(samples)/len(samples)<.94:continue
                anchors.append((ground,q))
        if len(anchors)==1:
            ground,q=anchors[0];before=copy.deepcopy(output['pins'][owner][key]['point'])
            output['pins'][owner][key]['point']=copy.deepcopy(q['point'])
            audit.append({'owner':owner,'number':key[4:],'name':p['pinname'],
                          'status':'unique_collinear_ground_contact','ground':ground,
                          'before':before,'after':q['point']})

assert output['components']==baseline['components'] and output['nets']==baseline['nets']
write(ROOT/'1_Pred.json',output)
write(ROOT/'generation_audit.json',{'frames':frames,'proposals':proposals,'changes':audit,
    'digit_reads':digit_reads,'new_local_glyph_calls':new_calls,'gt_used':False})

# Reference labels enter only after all geometry and fields are frozen.
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/1').glob('*.json')))
gained=[];lost=[];remaining=[]
for c,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,ref in truth['pins'].get(c,{}).items():
        def correct(doc):
            p=doc['pins'].get(c,{}).get(key)
            return p is not None and p['pinname']==ref['pinname'] and math.hypot(p['point']['x']-ref['point']['x'],p['point']['y']-ref['point']['y'])<=20
        a,b=correct(baseline),correct(output)
        if b and not a:gained.append(c+'.'+key[4:])
        if a and not b:lost.append(c+'.'+key[4:])
        if not b:remaining.append(c+'.'+key[4:])
result={'before':metrics.score_pins(baseline,truth,20)[0],'after':metrics.score_pins(output,truth,20)[0],
        'gained':gained,'lost':lost,'remaining':remaining,'changes':audit,
        'network_sync_not_tested':True,'production_modified':False}
write(ROOT/'results.json',result)
print(json.dumps(result,ensure_ascii=False))
