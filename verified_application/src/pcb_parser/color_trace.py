"""Experimental graph from explicitly colored current-image conductors."""
from collections import defaultdict
import math
import cv2,networkx as nx,numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .wire_graph import compress_skeleton,junction_graph,detect_junction_dots

def make_graph(rgb,pins,cfg,keepouts=None):
    c=rgb.astype(np.int16);r,g,b=[c[:,:,i] for i in range(3)]
    green=(g>r+20)&(g>b+20)&(g>60);red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
    # Antialiased red/green transitions can be brown, with nearly equal R/G.
    # Accept those only inside observed node discs or verified short lead paths.
    colored=(np.maximum(r,g)>b+30)&(np.maximum(r,g)>70)
    mask=green.copy();dots=detect_junction_dots(rgb,cfg)
    # Junction discs connected to a red component lead are not standalone CCs.
    # Local thickness recovers their centres without using any reference point.
    thick=cv2.distanceTransform(red.astype('uint8'),cv2.DIST_L2,5)>=3.2
    n,labels,stats,centres=cv2.connectedComponentsWithStats(thick.astype('uint8'),8)
    for p in centres[1:]:
        if all(math.dist(p,q)>5 for q in dots):dots.append(p.tolist())
    accepted=[]
    for x,y in dots:
        x,y=round(x),round(y)
        if sum(bool(green[max(0,y+dy-2):max(0,min(mask.shape[0],y+dy+3)),
                           max(0,x+dx-2):max(0,min(mask.shape[1],x+dx+3))].any())
               for dx,dy in [(8,0),(-8,0),(0,8),(0,-8)])<1:continue
        y1,y2=max(0,y-7),min(mask.shape[0],y+8);x1,x2=max(0,x-7),min(mask.shape[1],x+8)
        yy,xx=np.mgrid[y1:y2,x1:x2];mask[y1:y2,x1:x2]|=colored[y1:y2,x1:x2]&((xx-x)**2+(yy-y)**2<=49)
        accepted.append([x,y])
    blue=(b>r+20)&(b>g+20)
    closed=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((1,15),np.uint8))|cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((15,1),np.uint8))
    added=(closed>0)&~mask;text_occlusion=np.zeros_like(mask)
    near_blue=cv2.dilate(blue.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    n,lab,st,_=cv2.connectedComponentsWithStats(added.astype('uint8'),8)
    for i,(x,y,ww,hh,area) in enumerate(st[1:],1):
        if min(ww,hh)<=4 and max(ww,hh)<=15 and area<=60 and near_blue[lab==i].any():text_occlusion[lab==i]=True
    mask|=text_occlusion
    bridges=[];conductive=colored
    for ref,point in pins.items():
        x,y=map(round,point)
        if green[max(0,y-2):y+3,max(0,x-2):x+3].any():continue
        options=[]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            for d in range(1,36):
                xx,yy=x+dx*d,y+dy*d
                if not(1<=xx<mask.shape[1]-1 and 1<=yy<mask.shape[0]-1):break
                if keepouts is not None and keepouts[yy,xx]:break
                if not conductive[yy-1:yy+2,xx-1:xx+2].any():break
                if mask[yy-1:yy+2,xx-1:xx+2].any():options.append((d,[xx,yy]));break
        if not options:continue
        options.sort()
        if len(options)>1 and options[1][0]-options[0][0]<3:continue
        d,end=options[0];cv2.line(mask.view('uint8'),(x,y),tuple(end),1,1);bridges.append({'member':ref,'point':point,'green_point':end,'length':d})
    prob=mask.astype(float);skel=skeletonize(mask,method='zhang');observed=compress_skeleton(skel,prob,cfg)
    graph,uncertain=junction_graph(observed,cfg)
    connected=[]
    for j in uncertain:
        if any(math.dist(j['point'],p)<=5 for p in accepted):
            virtual=j['virtual_nodes'];node=j['junction_id'];graph.add_node(node,point=j['point'],kind='image_red_dot')
            for v in virtual:
                for u,w,key,data in list(graph.edges(v,keys=True,data=True)):
                    other=w if u==v else u;graph.add_edge(node,other,key=key,**data)
                graph.remove_node(v)
            connected.append(node)
    groups=sorted(nx.connected_components(graph),key=lambda nodes:min(nodes));nets={};samples=[];sample_ids=[]
    for i,nodes in enumerate(groups):
        nid=f'COLOR_{i+1:04d}';edges=[]
        for u,v,key,data in graph.edges(keys=True,data=True):
            if u in nodes:
                edges.append(data)
                for p in data['path']:samples.append(p);sample_ids.append(nid)
        if edges:nets[nid]={'nodes':list(nodes),'edges':edges}
    tree=cKDTree(samples) if samples else None
    def bind(point,radius=5):
        choices={}
        if tree:
            for idx in tree.query_ball_point(point,radius):
                nid=sample_ids[idx];distance=math.dist(point,samples[idx])
                if nid not in choices or distance<choices[nid]:choices[nid]=distance
        rank=sorted(choices.items(),key=lambda v:(v[1],v[0]))
        return rank[0][0] if rank and (len(rank)==1 or rank[1][1]-rank[0][1]>=2) else None
    return nets,bind,{'source':'current_RGB_green_conductors_and_small_red_junction_discs','dots':accepted,'blue_text_occlusion_pixels':int(text_occlusion.sum()),'red_dot_junctions_joined':connected,'pin_lead_extensions':bridges,'four_way_crossings':len(uncertain),'no_dot_crossings_kept_separate':len(uncertain)-len(connected)},mask
