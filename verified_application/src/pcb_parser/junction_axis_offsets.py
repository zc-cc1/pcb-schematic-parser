"""Preserve measured rail offsets inside a source-confirmed three-arm junction.

Tiny junction steps run after generic fragment suppression. Both the current
raster arms and the retained model paths must independently prove the offset.
No component names, page coordinates or reference annotations are consulted.
"""
from copy import deepcopy
import math
import numpy as np
from .wire_export_geometry import GeometryDraft, DEFAULT as GEOMETRY_DEFAULT

DEFAULT={'enabled':False,'maximum_changes':32}


def _observe_arm(draft,dot,axis,direction):
    levels=[]
    for distance in range(12,25):
        sample=np.array(dot,float);sample[axis]+=direction*distance
        x,y=round(sample[0]),round(draft.h-sample[1])
        if not(5<x<draft.w-5 and 5<y<draft.h-5):continue
        patch=draft.green[y-4:y+5,x] if axis==0 else draft.green[y,x-4:x+5]
        positions=np.flatnonzero(patch)
        if len(positions):
            levels.append(draft.h-(y-4+float(np.median(positions))) if axis==0
                          else x-4+float(np.median(positions)))
    if len(levels)<10 or np.ptp(levels)>1:return None
    return float(np.median(levels))


def _source_junction(graph,nid,dot,height):
    network=next((n for n in graph.get('networks',[]) if n['network_id']==nid),None)
    if not network:return None
    candidates=[n for n in graph.get('nodes',[]) if n.get('kind')=='junction'
                and n['node_id'] in network.get('node_ids',[])
                and math.dist([n['point'][0],height-n['point'][1]],dot)<=3]
    if len(candidates)!=1:return None
    node=candidates[0];allowed=set(network.get('edge_ids',[]))
    edges=[e for e in graph.get('edges',[]) if e['edge_id'] in allowed
           and node['node_id'] in {e['source'],e['target']}]
    if len(edges)!=3:return None
    arms={}
    for edge in edges:
        poly=edge.get('polyline',[])
        if len(poly)<2:return None
        if math.dist(poly[-1],node['point'])<math.dist(poly[0],node['point']):poly=list(reversed(poly))
        if math.dist(poly[0],node['point'])>3:return None
        q=next((p for p in poly[1:] if math.dist(p,node['point'])>=12),None)
        if q is None:return None
        vector=np.array([q[0]-node['point'][0],node['point'][1]-q[1]])
        axis=int(abs(vector[1])>abs(vector[0]));direction=1 if vector[axis]>0 else -1
        if abs(vector[1-axis])>4 or (axis,direction) in arms:return None
        arms[(axis,direction)]=float(height-q[1] if axis==0 else q[0])
    return node['node_id'],arms


def _refine(target,rgb,graph):
    draft=GeometryDraft(target,rgb,graph,GEOMETRY_DEFAULT);out=draft.out;events=[]
    for nid,net in out['nets'].items():
        for dot in draft.dots:
            source=_source_junction(graph,nid,dot,draft.h)
            if source is None:continue
            links=[]
            for eid,edge in net['edges'].items():
                if not isinstance(edge,list) or len(edge)!=2:continue
                ps=np.array([[p['x'],p['y']] for p in edge],float)
                index=int(math.dist(ps[1],dot)<math.dist(ps[0],dot))
                if math.dist(ps[index],dot)<=5 and math.dist(*ps)>30:
                    links.append((eid,index,ps[index],ps[1-index]))
            if len(links)!=3:continue
            arms={}
            for eid,index,point,other in links:
                vector=other-point;axis=int(abs(vector[1])>abs(vector[0]));direction=1 if vector[axis]>0 else -1
                if abs(vector[1-axis])>4 or (axis,direction) in arms:break
                level=_observe_arm(draft,dot,axis,direction)
                if level is None:break
                arms[(axis,direction)]=(eid,index,point,other,level)
            if (len(arms)!=3 or set(arms)!=set(source[1])
                    or any(abs(value[-1]-source[1][key])>.75 for key,value in arms.items())):continue
            axis=next((a for a in (0,1) if (a,-1) in arms and (a,1) in arms),None)
            if axis is None:continue
            pair=[arms[(axis,-1)],arms[(axis,1)]]
            if not 2<=abs(pair[0][-1]-pair[1][-1])<=4:continue
            common=np.median([link[2] for link in links],axis=0)
            if max(math.dist(point,common) for _,_,point,_ in links)>4:continue
            moving=max(pair,key=lambda v:abs(v[-1]-common[1-axis]));fixed=pair[1] if moving is pair[0] else pair[0]
            if abs(fixed[-1]-common[1-axis])>.75:continue
            eid,index,point,other,level=moving
            corrected=common.copy();corrected[1-axis]=level
            if not 1.5<=math.dist(corrected,common)<=4:continue
            if not draft.outside_bodies([point,corrected,common]) or not draft.ink_connected(corrected,common):continue
            if draft.support(other,corrected)<.98:continue
            # Repeated finalization may already contain the protected small step.
            def same_step(edge):
                ps=[[p['x'],p['y']] for p in edge]
                return min(max(math.dist(a,b) for a,b in zip(ps,ends))
                           for ends in ([corrected,common],[common,corrected]))<=.5
            if any(same_step(edge) for edge in net['edges'].values() if len(edge)==2):continue
            before=deepcopy(net['edges'][eid]);net['edges'][eid][index]={'x':float(corrected[0]),'y':float(corrected[1])}
            step=[{'x':float(p[0]),'y':float(p[1])} for p in (corrected,common)]
            key='junction_offset_'+str(len(events)+1)
            while key in net['edges']:key+='x'
            net['edges'][key]=step
            events.append({'rule':'observed_junction_axis_offset','network_id':nid,'source_junction':source[0],
                           'edge_id':eid,'before':before,'after':deepcopy(net['edges'][eid]),'step':step,
                           'observed_arm_levels':{str(k):v[-1] for k,v in arms.items()}})
    if events:
        for net in out['nets'].values():
            net['edges']={'edge_'+str(i):e for i,e in enumerate(net['edges'].values(),1)}
    return out,events


def refine_junction_axis_offsets(target,rgb,graph,image_sha256,*,cfg=None):
    config={**DEFAULT,**(cfg or {})}
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:
        raise ValueError('invalid_junction_axis_offset_setting')
    if type(config['maximum_changes']) is not int or not 0<=config['maximum_changes']<=128:
        raise ValueError('invalid_junction_axis_offset_budget')
    if not config['enabled']:return deepcopy(target),{'enabled':False,'status':'disabled','changes':[]}
    h,w=rgb.shape[:2]
    if (graph.get('image_sha256')!=image_sha256 or graph.get('size')!=[w,h]
            or graph.get('coordinate_origin')!='top_left' or graph.get('prediction_only') is not True):
        raise ValueError('junction_axis_offset_evidence_differs_from_image')
    out,events=_refine(target,rgb,graph)
    if len(events)>config['maximum_changes']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],'proposed_changes':len(events)}
    if out['components']!=target['components'] or out['pins']!=target['pins'] or set(out['nets'])!=set(target['nets']):
        raise ValueError('junction_axis_offset_modified_structure')
    for nid,net in out['nets'].items():
        if {k:v for k,v in net.items() if k!='edges'}!={k:v for k,v in target['nets'][nid].items() if k!='edges'}:
            raise ValueError('junction_axis_offset_modified_members')
    return out,{'enabled':True,'status':'applied' if events else 'no_effect','changes':events,'reference_read':False}
