"""U-Net probability -> skeleton -> graph. No raw image segmentation or labels.

Component boxes and pin tips are model predictions. Four-way junctions remain
explicitly uncertain; the conservative graph keeps straight branches separate.
"""
from collections import Counter,defaultdict
import itertools
import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def guarded_mask(probability,pin_result,cfg):
    mask=probability>=cfg['wire_threshold']
    blocked=np.zeros(mask.shape,bool)
    protection=np.zeros(mask.shape,bool)
    height,width=mask.shape
    pins_by_owner=defaultdict(list)
    for pin in pin_result['pin_candidates']:
        if pin['owner_status']=='assigned':
            pins_by_owner[pin['owner_detection_index']].append(pin)
    if cfg['body_guard']:
        for component in pin_result['components']:
            x1,y1,x2,y2=component['bbox_top_left']
            inset=max(cfg['body_inset_min_px'],min(cfg['body_inset_max_px'],cfg['body_inset_fraction']*min(x2-x1,y2-y1)))
            a,b=max(0,math.ceil(x1+inset)),max(0,math.ceil(y1+inset))
            c,d=min(width,math.floor(x2-inset)+1),min(height,math.floor(y2-inset)+1)
            if a>=c or b>=d:
                continue
            blocked[b:d,a:c]=True
            for pin in pins_by_owner[component['detection_index']]:
                x,y=pin['point']
                radius=cfg['pin_protection_radius_px']
                ax,ay=max(0,math.floor(x-radius)),max(0,math.floor(y-radius))
                bx,by=min(width,math.ceil(x+radius)+1),min(height,math.ceil(y+radius)+1)
                yy,xx=np.mgrid[ay:by,ax:bx]
                region=(xx-x)**2+(yy-y)**2<=radius**2
                # Protection cannot bridge through the middle of a component.
                side=min([('l',abs(x-x1)),('r',abs(x-x2)),('t',abs(y-y1)),('b',abs(y-y2))],key=lambda v:v[1])[0]
                if side=='l': region &= xx<(x1+x2)/2-.5
                elif side=='r': region &= xx>(x1+x2)/2+.5
                elif side=='t': region &= yy<(y1+y2)/2-.5
                else: region &= yy>(y1+y2)/2+.5
                protection[ay:by,ax:bx] |= region
    excluded=blocked&~protection
    return mask&~excluded,excluded


def pixel_adjacency(skeleton):
    height,width=skeleton.shape
    pixels=set(zip(*np.where(skeleton)))
    adjacent={p:[] for p in pixels}
    for y,x in pixels:
        for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
            q=(y+dy,x+dx)
            if q not in pixels:
                continue
            # L bends must not form a diagonal shortcut and a false junction.
            if dy and dx and ((y,x+dx) in pixels or (y+dy,x) in pixels):
                continue
            adjacent[y,x].append(q)
            adjacent[q].append((y,x))
    return adjacent


def polyline(points,tolerance):
    points=np.asarray(points,float)
    if len(points)<=2:
        return points.tolist()
    begin,end=points[0],points[-1]
    vector=end-begin
    if np.dot(vector,vector):
        t=np.clip((points-begin)@vector/np.dot(vector,vector),0,1)
        distances=np.linalg.norm(points-(begin+t[:,None]*vector),axis=1)
    else:
        distances=np.linalg.norm(points-begin,axis=1)
    index=int(distances.argmax())
    if distances[index]<=tolerance:
        return points[[0,-1]].tolist()
    return polyline(points[:index+1],tolerance)[:-1]+polyline(points[index:],tolerance)


def compress_skeleton(skeleton,probability,cfg):
    adjacent=pixel_adjacency(skeleton)
    graph=nx.MultiGraph()
    junction_pixels={p for p,v in adjacent.items() if len(v)>=3}
    nodes={}
    remaining=set(junction_pixels)
    while remaining:
        start=min(remaining);stack=[start];cluster=[];remaining.remove(start)
        while stack:
            p=stack.pop();cluster.append(p)
            for q in adjacent[p]:
                if q in remaining:
                    remaining.remove(q);stack.append(q)
        name=f'J{len(graph)+1:05d}'
        point=[float(np.mean([p[1] for p in cluster])),float(np.mean([p[0] for p in cluster]))]
        graph.add_node(name,point=point,kind='junction',pixel_count=len(cluster))
        for p in cluster:nodes[p]=name
    for p in sorted(adjacent):
        if len(adjacent[p])<=1:
            name=f'N{len(graph)+1:05d}'
            graph.add_node(name,point=[int(p[1]),int(p[0])],kind='endpoint' if adjacent[p] else 'isolated_pixel',pixel_count=1)
            nodes[p]=name
    visited=set()
    def link(a,b):return tuple(sorted((a,b)))
    def trace(start,next_pixel):
        path=[start,next_pixel]
        visited.add(link(start,next_pixel))
        previous,current=start,next_pixel
        while current not in nodes:
            following=[q for q in adjacent[current] if q!=previous]
            if len(following)!=1:
                raise ValueError('Invalid degree-two skeleton chain')
            following=following[0]
            visited.add(link(current,following));path.append(following)
            previous,current=current,following
            if len(path)>len(adjacent)+1:
                raise ValueError('Skeleton trace did not terminate')
        source,target=nodes[start],nodes[current]
        xy=[[int(p[1]),int(p[0])] for p in path]
        edge_id=f'W{graph.number_of_edges()+1:05d}'
        length=float(sum(math.dist(a,b) for a,b in zip(xy,xy[1:])))
        graph.add_edge(source,target,key=edge_id,edge_id=edge_id,path=xy,polyline=polyline(xy,cfg['polyline_tolerance_px']),
                       length_px=length,mean_probability=float(np.mean([probability[p] for p in path])))
    for p in sorted(nodes):
        for q in adjacent[p]:
            if link(p,q) in visited:continue
            if q in nodes and nodes[p]==nodes[q]:
                visited.add(link(p,q));continue
            trace(p,q)
    # A pure closed loop has no endpoint or junction; preserve it as a self-loop.
    for p in sorted(adjacent):
        for q in adjacent[p]:
            if link(p,q) in visited:continue
            if p not in nodes:
                name=f'L{len(graph)+1:05d}'
                graph.add_node(name,point=[int(p[1]),int(p[0])],kind='cycle_anchor',pixel_count=1);nodes[p]=name
            trace(p,q)
    return graph


def incident_arms(graph,node,cfg):
    arms=[]
    for u,v,key,data in graph.edges(node,keys=True,data=True):
        path=data['path'] if node==u else list(reversed(data['path']))
        # NetworkX emits the queried node as u; use endpoint geometry instead.
        point=np.asarray(graph.nodes[node]['point'])
        if math.dist(path[-1],point)<math.dist(path[0],point):path=list(reversed(path))
        step=min(len(path)-1,cfg['junction_direction_steps'])
        vector=np.asarray(path[step],float)-np.asarray(path[0],float)
        norm=float(np.linalg.norm(vector))
        arms.append({'u':u,'v':v,'key':key,'data':data,'direction':(vector/norm).tolist() if norm else [0.,0.]})
    return arms


def junction_graph(observed,cfg):
    conservative=observed.copy()
    uncertain=[]
    for node in list(observed.nodes):
        if observed.nodes[node]['kind']!='junction':continue
        degree=observed.degree(node)
        if degree<4:continue
        arms=incident_arms(observed,node,cfg)
        candidate={'junction_id':node,'point':observed.nodes[node]['point'],'arms':len(arms),'degree':degree,
                   'status':'connected_or_crossing_model_not_available','observed_hypothesis':'all_arms_connected',
                   'virtual_nodes':[],'edge_ids':[a['key'] for a in arms]}
        uncertain.append(candidate)
        if cfg['junction_policy']=='observed_connected':continue
        if cfg['junction_policy']!='split_four_way':raise ValueError('Unknown junction policy')
        pairs=[]
        if len(arms)==4 and all(a['u']!=a['v'] for a in arms):
            options=[[(0,1),(2,3)],[(0,2),(1,3)],[(0,3),(1,2)]]
            pairs=min(options,key=lambda pairing:sum(1+np.dot(arms[i]['direction'],arms[j]['direction']) for i,j in pairing))
            max_dot=math.cos(math.radians(cfg['straight_pair_min_angle_degrees']))
            if any(np.dot(arms[i]['direction'],arms[j]['direction'])>max_dot for i,j in pairs):pairs=[]
        groups=pairs or [(i,) for i in range(len(arms))]
        candidate['conservative_hypothesis']='straight_pairs_separate' if pairs else 'all_arms_separate'
        # Rewrite existing edges in place. This also supports an edge joining
        # two uncertain nodes: the previously rewritten remote endpoint stays.
        for index,group in enumerate(groups):
            virtual=f'{node}_V{index+1}'
            conservative.add_node(virtual,point=observed.nodes[node]['point'],kind='unresolved_crossing_branch',junction_id=node)
            candidate['virtual_nodes'].append(virtual)
            for i in group:
                arm=arms[i]
                matches=list(conservative.edges(node,keys=True,data=True))
                found=next(((u,v,k,d) for u,v,k,d in matches if k==arm['key']),None)
                if found is None:continue
                u,v,key,data=found
                conservative.remove_edge(u,v,key)
                other=v if u==node else u
                if other==node:other=virtual
                conservative.add_edge(virtual,other,key=key,**data)
        conservative.remove_node(node)
    return conservative,uncertain


def gap_candidates(graph,probability,excluded,cfg):
    endpoints=[n for n in graph if graph.nodes[n]['kind']=='endpoint' and graph.degree(n)==1]
    if len(endpoints)<2:return []
    points=np.array([graph.nodes[n]['point'] for n in endpoints])
    tree=cKDTree(points)
    network={n:i for i,group in enumerate(nx.connected_components(graph)) for n in group}
    result=[]
    minimum=math.cos(math.radians(cfg['gap_max_angle_degrees']))
    for i,j in sorted(tree.query_pairs(cfg['gap_max_distance_px'])):
        a,b=endpoints[i],endpoints[j]
        if network[a]==network[b]:continue
        delta=points[j]-points[i];distance=float(np.linalg.norm(delta))
        if distance<1:continue
        direction=delta/distance
        da=np.asarray(incident_arms(graph,a,cfg)[0]['direction'])
        db=np.asarray(incident_arms(graph,b,cfg)[0]['direction'])
        if np.dot(-da,direction)<minimum or np.dot(-db,-direction)<minimum:continue
        samples=np.rint(np.linspace(points[i],points[j],max(2,math.ceil(distance)+1))).astype(int)
        if excluded[samples[:,1],samples[:,0]].any():continue
        values=probability[samples[:,1],samples[:,0]]
        result.append({'endpoints':[a,b],'points':[points[i].tolist(),points[j].tolist()],'distance_px':distance,
                       'probability_mean':float(values.mean()),'probability_min':float(values.min()),
                       'status':'gap_candidate_not_filled','automatically_connected':False})
    return result


def network_membership(graph,pin_result,cfg,junctions):
    groups=sorted(nx.connected_components(graph),key=lambda group:min(group))
    by_node={node:f'NET_CAND_{i+1:05d}' for i,group in enumerate(groups) for node in group}
    support=defaultdict(set)
    for u,v,key,data in graph.edges(keys=True,data=True):
        for point in data['path']:support[tuple(point)].add(by_node[u])
    for node,data in graph.nodes(data=True):support[tuple(data['point'])].add(by_node[node])
    points=np.array(list(support),float).reshape(-1,2)
    tree=cKDTree(points) if len(points) else None
    attachments=[]
    nets=[]
    for i,group in enumerate(groups):
        nid=f'NET_CAND_{i+1:05d}'
        edge_ids=[key for u,v,key in graph.edges(keys=True) if u in group]
        ambiguous={graph.nodes[n].get('junction_id') for n in group if graph.nodes[n].get('junction_id')}
        # Observed-connected variant still must mark its junctions uncertain.
        ambiguous.update(j['junction_id'] for j in junctions if j['junction_id'] in group)
        nets.append({'network_id':nid,'node_ids':sorted(group),'edge_ids':edge_ids,'pin_candidate_ids':[],
                     'official_pin_references':[],'unresolved_junction_ids':sorted(ambiguous),
                     'status':'requires_junction_review' if ambiguous else 'model_mask_connectivity_proposal'})
    by_id={n['network_id']:n for n in nets}
    for pin in pin_result['pin_candidates']:
        point=pin['point']
        choices={}
        if tree is not None:
            for index in tree.query_ball_point(point,cfg['pin_attach_radius_px']):
                distance=math.dist(point,points[index])
                for nid in support[tuple(points[index])]:
                    if nid not in choices or distance<choices[nid]['distance_px']:
                        choices[nid]={'network_id':nid,'distance_px':distance,'wire_projection':points[index].tolist()}
        ranked=sorted(choices.values(),key=lambda r:(r['distance_px'],r['network_id']))
        status='no_wire_within_radius';chosen=None
        if ranked:
            if len(ranked)>1 and ranked[1]['distance_px']-ranked[0]['distance_px']<cfg['attachment_ambiguity_margin_px']:
                status='ambiguous_nearby_networks'
            else:
                status='attached';chosen=ranked[0]
        attachment={'candidate_id':pin['candidate_id'],'point':point.copy(),'owner':pin['owner'],'owner_status':pin['owner_status'],
                    'number':pin.get('number'),'pinname':pin.get('pinname'),'attachment_status':status,
                    'network_id':chosen['network_id'] if chosen else None,'wire_projection':chosen['wire_projection'] if chosen else None,
                    'wire_gap_px':chosen['distance_px'] if chosen else None,'alternatives':ranked[:4]}
        attachments.append(attachment)
        if chosen:
            net=by_id[chosen['network_id']]
            net['pin_candidate_ids'].append(pin['candidate_id'])
            component=next((c for c in pin_result['components'] if c['detection_index']==pin.get('owner_detection_index')),None)
            if pin['owner_status']=='assigned' and component and component.get('designator') and pin.get('number') is not None:
                net['official_pin_references'].append(f"{pin['owner']}.{pin['number']}")
    return nets,attachments


def build_wire_graph(probability,pin_result,cfg):
    if probability.ndim!=2 or list(probability.shape[::-1])!=pin_result['size'] or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():
        raise ValueError('Expected finite original-size U-Net wire probabilities in [0,1]')
    if pin_result.get('coordinate_origin')!='top_left':raise ValueError('Expected original top-left coordinates')
    ids=[p['candidate_id'] for p in pin_result['pin_candidates']]
    if len(ids)!=len(set(ids)):raise ValueError('Pin candidate IDs must be unique')
    width,height=pin_result['size']
    for pin in pin_result['pin_candidates']:
        x,y=pin['point']
        if not np.isfinite([x,y]).all() or not 0<=x<width or not 0<=y<height:raise ValueError('Pin outside input image')
    mask,excluded=guarded_mask(probability,pin_result,cfg)
    skeleton=skeletonize(mask,method='zhang')
    observed=compress_skeleton(skeleton,probability,cfg)
    graph,junctions=junction_graph(observed,cfg)
    gaps=gap_candidates(graph,probability,excluded,cfg)
    nets,attachments=network_membership(graph,pin_result,cfg,junctions)
    # Mark nets touching a suspected gap, without joining them.
    node_network={node:n['network_id'] for n in nets for node in n['node_ids']}
    for gap in gaps:
        gap['network_ids']=[node_network[n] for n in gap['endpoints']]
    result={'coordinate_origin':'top_left','size':pin_result['size'],'status':'provisional_physical_networks_logical_labels_pending',
            'nodes':[{'node_id':n,**data} for n,data in graph.nodes(data=True)],
            'edges':[{'source':u,'target':v,**{k:value for k,value in data.items() if k!='path'}} for u,v,data in graph.edges(data=True)],
            'junction_candidates':junctions,'gap_candidates':gaps,'networks':nets,'pin_attachments':attachments,
            'summary':{'wire_foreground_pixels':int((probability>=cfg['wire_threshold']).sum()),'body_excluded_foreground_pixels':int((excluded&(probability>=cfg['wire_threshold'])).sum()),
                       'skeleton_pixels':int(skeleton.sum()),'graph_nodes':len(graph),'graph_edges':graph.number_of_edges(),
                       'observed_components':nx.number_connected_components(observed),'network_proposals':len(nets),
                       'multi_pin_network_proposals':sum(len(n['pin_candidate_ids'])>=2 for n in nets),'attached_pins':sum(a['attachment_status']=='attached' for a in attachments),
                       'attachment_status_counts':dict(Counter(a['attachment_status'] for a in attachments)),
                       'uncertain_junctions':len(junctions),'gap_candidates_not_filled':len(gaps),'pin_candidates_preserved':len(attachments)==len(ids),
                       'pin_coordinate_changes':0,'automatic_gap_connections':0,'logical_label_merges':0},
            'raw_image_segmentation_used':False,'junction_classifier_used':False,'official_NetF1':None}
    return result,{'guarded_wire_mask':mask,'body_excluded_mask':excluded,'skeleton':skeleton}
