"""Consolidate repeated exported spans over one observed conductor.

Only geometry moves between already existing network records. Electrical
identities and membership strings stay intact; no reference data is consulted.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .wire_export_geometry import GeometryDraft,DEFAULT as GEOMETRY_DEFAULT


DEFAULT={'enabled':False,'maximum_changes':32}


def _consolidate(target,rgb,dots):
    out=deepcopy(target);draft=GeometryDraft(target,rgb,{},GEOMETRY_DEFAULT);events=[]
    red,green,blue=[rgb[:,:,i].astype(np.int16) for i in range(3)]
    text=(blue>red+20)&(blue>green+20)
    # A short gap may contain overwritten blue glyph pixels, never white space.
    closed=(cv2.morphologyEx(draft.color.astype('uint8'),cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
            |cv2.morphologyEx(draft.color.astype('uint8'),cv2.MORPH_CLOSE,np.ones((5,1),np.uint8)))
    draft.color|=closed.astype(bool)&text
    segments=[]
    for nid,net in out['nets'].items():
        for eid,edge in net['edges'].items():
            if not isinstance(edge,list) or len(edge)!=2:continue
            points=np.asarray([[p['x'],p['y']] for p in edge],float)
            if not np.isfinite(points).all():continue
            vector=points[1]-points[0];axis=int(abs(vector[1])>abs(vector[0]))
            length=float(np.linalg.norm(vector))
            if length<20 or abs(vector[1-axis])>3:continue
            segments.append({'nid':nid,'eid':eid,'points':points,'axis':axis,'length':length})
    used=set()
    for index,seed in enumerate(segments):
        if index in used:continue
        group={index};changed=True
        while changed:
            changed=False
            for candidate_index,candidate in enumerate(segments):
                if candidate_index in used or candidate_index in group or candidate['axis']!=seed['axis']:continue
                axis=seed['axis'];cross=1-axis
                coords=np.concatenate([segments[k]['points'] for k in group]+[candidate['points']])
                if np.ptp(coords[:,cross])>3:continue
                low,high=min(coords[:,axis]),max(coords[:,axis])
                if any(low+6<dot[axis]<high-6 and abs(dot[cross]-float(np.mean(coords[:,cross])))<=5 for dot in dots):continue
                for member_index in sorted(group):
                    first,second=segments[member_index]['points'],candidate['points']
                    overlap=min(max(first[:,axis]),max(second[:,axis]))-max(min(first[:,axis]),min(second[:,axis]))
                    shared=min(math.dist(a,b) for a in first for b in second)<=2
                    if ((overlap>=25 and segments[member_index]['nid']!=candidate['nid'])
                            or (shared and segments[member_index]['nid']==candidate['nid'])):
                        group.add(candidate_index);changed=True;break
                if changed:break
        if len(group)<2 or len({segments[k]['nid'] for k in group})<2:continue
        axis=seed['axis'];coords=np.concatenate([segments[k]['points'] for k in sorted(group)])
        if len(group)<3 and any(math.dist(point,dot)<=5 for point in coords for dot in dots):continue
        first=coords[np.argmin(coords[:,axis])];last=coords[np.argmax(coords[:,axis])]
        span_length=math.dist(first,last);longest=max(segments[k]['length'] for k in group)
        # On replay the complete span may already exist alongside restored
        # fragments. Remove those duplicates before network reconciliation.
        if longest+2<span_length<longest+20:continue
        if len(group)<3 and span_length>longest+2:continue
        internal=[point for point in coords if min(math.dist(point,first),math.dist(point,last))>5]
        group_ids={(segments[k]['nid'],segments[k]['eid']) for k in group}
        blocked=any(math.dist([point['x'],point['y']],site)<=3
                    for nid,net in out['nets'].items() for eid,edge in net['edges'].items()
                    if (nid,eid) not in group_ids for point in edge for site in internal)
        if blocked or not draft.outside_bodies([first,last]):continue
        if draft.support(first,last)<.98 or not draft.ink_connected(first,last):continue
        winner=max(group,key=lambda k:(len(out['nets'][segments[k]['nid']]['hyperGraph'].split(',')),
                                       len(out['nets'][segments[k]['nid']]['edges']),segments[k]['length'],-k))
        retained=segments[winner]
        edge=[{'x':float(point[0]),'y':float(point[1])} for point in (first,last)]
        before=[{'network_id':segments[k]['nid'],'edge_id':segments[k]['eid'],
                 'edge':deepcopy(out['nets'][segments[k]['nid']]['edges'][segments[k]['eid']])}
                for k in sorted(group)]
        for member_index in group:
            item=segments[member_index]
            if member_index!=winner:del out['nets'][item['nid']]['edges'][item['eid']]
        out['nets'][retained['nid']]['edges'][retained['eid']]=edge
        events.append({'rule':'overlapping_collinear_span','retained_net':retained['nid'],
                       'retained_edge':retained['eid'],'before':before,'after':edge})
        used|=group
    if events:
        for net in out['nets'].values():
            net['edges']={'edge_'+str(index):edge for index,edge in enumerate(net['edges'].values(),1)}
    return out,events


def consolidate_wire_spans(target,rgb,graph,image_sha256,observed_junctions,*,cfg=None):
    config={**DEFAULT,**(cfg or {})}
    if set(config)!=set(DEFAULT) or type(config['enabled']) is not bool:
        raise ValueError('invalid_wire_span_consolidation_setting')
    if type(config['maximum_changes']) is not int or not 0<=config['maximum_changes']<=128:
        raise ValueError('invalid_wire_span_consolidation_budget')
    if not config['enabled']:
        return deepcopy(target),{'enabled':False,'status':'disabled','changes':[]}
    rgb=np.asarray(rgb);height,width=rgb.shape[:2]
    if (graph.get('image_sha256')!=image_sha256 or graph.get('size')!=[width,height]
            or graph.get('coordinate_origin')!='top_left' or graph.get('prediction_only') is not True):
        raise ValueError('wire_span_evidence_differs_from_image')
    if any(not isinstance(point,(list,tuple)) or len(point)!=2 or
           any(type(value) not in (int,float) or not math.isfinite(value) for value in point)
           for point in observed_junctions):
        raise ValueError('invalid_observed_junction')
    out,events=_consolidate(target,rgb,observed_junctions)
    if len(events)>config['maximum_changes']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],
                                 'proposed_changes':len(events)}
    if out['components']!=target['components'] or out['pins']!=target['pins'] or set(out['nets'])!=set(target['nets']):
        raise ValueError('wire_span_consolidation_modified_structure')
    for nid,net in out['nets'].items():
        if {k:v for k,v in net.items() if k!='edges'}!={k:v for k,v in target['nets'][nid].items() if k!='edges'}:
            raise ValueError('wire_span_consolidation_modified_net_members')
    return out,{'enabled':True,'status':'applied' if events else 'no_effect','changes':events,'reference_read':False}
