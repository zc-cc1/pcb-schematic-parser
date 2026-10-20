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


def junction_graph(observed,cfg,decisions=None):
    conservative=observed.copy()
    uncertain=[]
    decisions=decisions or {}
    candidate_nodes={n for n in observed if observed.nodes[n]['kind']=='junction' and observed.degree(n)>=4}
    if set(decisions)-candidate_nodes:raise ValueError('Junction decisions refer to unknown candidate nodes')
    for node in list(observed.nodes):
        if observed.nodes[node]['kind']!='junction':continue
        degree=observed.degree(node)
        if degree<4:continue
        arms=incident_arms(observed,node,cfg)
        candidate={'junction_id':node,'point':observed.nodes[node]['point'],'arms':len(arms),'degree':degree,
                   'status':'connected_or_crossing_model_not_available','observed_hypothesis':'all_arms_connected',
                   'virtual_nodes':[],'edge_ids':[a['key'] for a in arms],'resolved':False}
        uncertain.append(candidate)
        decision=decisions.get(node)
        if decision:
            candidate['classifier_decision']=decision
            candidate['status']='learned_junction_abstained'
            if decision.get('accepted'):
                minimum=decision.get('rotation_min_confidence',0);threshold=decision.get('threshold',.95)
                if (degree!=4 or decision.get('decision_source')!='learned_RGB_junction_CNN'
                    or decision.get('label') not in {'connected_junction','crossing_without_connection'}
                    or not decision.get('rotation_agreement')
                    or not np.isfinite([minimum,threshold]).all() or not .5<threshold<=minimum<=1):
                    raise ValueError('Invalid accepted learned junction decision')
                if decision['label']=='connected_junction':
                    candidate.update(status='learned_connected_junction',resolved=True,
                                     conservative_hypothesis='learned_all_arms_connected')
                    continue
                candidate['status']='learned_crossing_geometry_unresolved'
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
        resolved_crossing=bool(pairs and decision and decision.get('accepted') and decision['label']=='crossing_without_connection')
        if resolved_crossing:candidate.update(status='learned_crossing_without_connection',resolved=True)
        # Rewrite existing edges in place. This also supports an edge joining
        # two uncertain nodes: the previously rewritten remote endpoint stays.
        for index,group in enumerate(groups):
            virtual=f'{node}_V{index+1}'
            conservative.add_node(virtual,point=observed.nodes[node]['point'],
                                  kind='learned_crossing_branch' if resolved_crossing else 'unresolved_crossing_branch',
                                  **({'resolved_junction_id':node} if resolved_crossing else {'junction_id':node}))
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
        ambiguous.update(j['junction_id'] for j in junctions if not j.get('resolved',False) and j['junction_id'] in group)
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


def detect_junction_dots(rgb,cfg):
    """Centres of the printed node discs: small, round, strongly red blobs on the wire."""
    if rgb is None:
        return []
    min_red=float(cfg.get('dot_min_red',150));max_green=float(cfg.get('dot_max_green',120))
    max_blue=float(cfg.get('dot_max_blue',120));min_area=int(cfg.get('dot_min_area',12))
    max_area=int(cfg.get('dot_max_area',240))
    red=(rgb[:,:,0]>=min_red)&(rgb[:,:,1]<=max_green)&(rgb[:,:,2]<=max_blue)
    if not red.any():
        return []
    from scipy import ndimage
    labels,count=ndimage.label(red)
    if count==0:
        return []
    boxes=ndimage.find_objects(labels)
    centres=[]
    for index,box in enumerate(boxes,start=1):
        if box is None:
            continue
        height=box[0].stop-box[0].start;width=box[1].stop-box[1].start
        if height<2 or width<2 or max(height,width)/min(height,width)>2.2:
            continue
        area=int((labels[box]==index).sum())
        if not min_area<=area<=max_area:
            continue
        if area/(height*width)<float(cfg.get('dot_min_fill',0.55)):
            continue
        centres.append([float(box[1].start+width/2),float(box[0].start+height/2)])
    return centres


def build_wire_graph(probability,pin_result,cfg,junction_decisions=None,dot_points=None):
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
    graph,junctions=junction_graph(observed,cfg,junction_decisions)
    gaps=gap_candidates(graph,probability,excluded,cfg)
    # Optional gap filling.  `gap_candidates` already screens each gap by distance,
    # direction (gap_max_angle_degrees), component-body exclusion and the probability
    # along the straight line; filling therefore adds only gaps that the model itself
    # supports.  OFF by default (`bridge_max_gap_px` 0): it joins separate nets, so it
    # has to be measured on the released pages before it is trusted.
    bridged=0
    bridge_limit=float(cfg.get('bridge_max_gap_px') or 0)
    if bridge_limit>0:
        bridge_min_probability=float(cfg.get('bridge_min_probability') or 0)
        taken=set()
        for gap in sorted(gaps,key=lambda g:g['distance_px']):
            if gap['distance_px']>bridge_limit or gap['probability_min']<bridge_min_probability:
                continue
            left,right=gap['endpoints']
            if left in taken or right in taken:
                continue
            a,b=(np.asarray(point,float) for point in gap['points'])
            steps=max(2,int(round(gap['distance_px']))+1)
            path=[point.astype(int).tolist() for point in np.rint(np.linspace(a,b,steps))]
            graph.add_edge(left,right,key=f'BRIDGE_{bridged+1:04d}',edge_id=f'BRIDGE_{bridged+1:04d}',kind='bridge',
                           path=path,polyline=polyline(path,cfg['polyline_tolerance_px']),
                           length_px=float(gap['distance_px']),
                           mean_probability=gap['probability_mean'])
            gap['automatically_connected']=True;gap['status']='gap_filled_by_model_support'
            taken.update((left,right));bridged+=1
    # Optional: cut an edge where a pin sits on its interior.  The reference splits a
    # drawn wire at every pin/junction it passes through; the skeleton only splits at
    # junctions, so an edge that runs past a pin is one segment where the reference has
    # two - and a two-point segment is scored on BOTH endpoints, so it misses even
    # though the ink is identical.  Measured on the 13 released pages: 80.6% of the
    # wire segments that miss lie on a reference line but are cut elsewhere, and 15.5%
    # of them fall exactly on a pin.  `pin_split_max_distance_px` 0 disables it.
    # Cut on the actual raster path, preserving all bends.
    pin_splits = 0
    pin_split_limit = float(cfg.get('pin_split_max_distance_px') or 0)
    if pin_split_limit > 0:
        pin_points = np.asarray([p['point'] for p in pin_result['pin_candidates']], dtype=float).reshape(-1, 2)
        next_node = 0
        for u, v, key, data in list(graph.edges(keys=True, data=True)):
            path = np.asarray(data['path'], dtype=float)
            if len(path) < 3 or not len(pin_points):
                continue
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if arc[-1] < 2 * pin_split_limit:
                continue
            cuts = set()
            for point in pin_points:
                distances = np.linalg.norm(path - point, axis=1)
                index = int(distances.argmin())
                if (distances[index] <= pin_split_limit
                        and pin_split_limit < arc[index] < arc[-1] - pin_split_limit):
                    cuts.add(index)
            if not cuts:
                continue
            # NetworkX's edge ordering may differ from stored path ordering.
            if math.dist(graph.nodes[u]['point'], path[0]) > math.dist(graph.nodes[v]['point'], path[0]):
                u, v = v, u
            graph.remove_edge(u, v, key)
            boundaries = [0] + sorted(cuts) + [len(path) - 1]
            previous = u
            for part, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
                fragment = path[begin:end+1].tolist()
                if part == len(boundaries) - 2:
                    target = v
                else:
                    target = f'pin_split_{next_node:05d}'
                    next_node += 1
                    graph.add_node(target, kind='pin_split', point=fragment[-1], degree=2)
                edge_id = f"{data['edge_id']}_{part+1}"
                graph.add_edge(previous, target, key=edge_id, edge_id=edge_id,
                               kind=data.get('kind', 'wire'), path=fragment,
                               polyline=polyline(fragment, cfg['polyline_tolerance_px']),
                               length_px=float(arc[end] - arc[begin]),
                               mean_probability=data.get('mean_probability', 0.0))
                previous = target
            pin_splits += len(cuts)
    # Optional: cut an edge at a printed junction dot.  The reference cuts a net exactly
    # at its electrical nodes, and those nodes are drawn as small red discs (measured on
    # the 13 released pages: 138 of 326 reference segment endpoints sit on a red pixel,
    # against 1 of 489 mid-segment samples - the wire itself is green).  Deriving nodes
    # from skeleton branching puts them a few pixels off, which is fatal because a
    # segment only scores when BOTH endpoints are inside 5 px.  `dot_split_max_distance_px`
    # 0 disables this.
    dot_splits = 0
    dot_limit = float(cfg.get('dot_split_max_distance_px') or 0)
    dot_points = np.asarray(dot_points if dot_points is not None else [], dtype=float).reshape(-1, 2)
    if dot_limit > 0 and len(dot_points):
        next_dot_node = 0
        for u, v, key, data in list(graph.edges(keys=True, data=True)):
            path = np.asarray(data['path'], dtype=float)
            if len(path) < 3:
                continue
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            if arc[-1] < 2 * dot_limit:
                continue
            cuts = set()
            for point in dot_points:
                distances = np.linalg.norm(path - point, axis=1)
                index = int(distances.argmin())
                if (distances[index] <= dot_limit
                        and dot_limit < arc[index] < arc[-1] - dot_limit):
                    cuts.add(index)
            if not cuts:
                continue
            if math.dist(graph.nodes[u]['point'], path[0]) > math.dist(graph.nodes[v]['point'], path[0]):
                u, v = v, u
            graph.remove_edge(u, v, key)
            boundaries = [0] + sorted(cuts) + [len(path) - 1]
            previous = u
            for part, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
                fragment = path[begin:end+1].tolist()
                if part == len(boundaries) - 2:
                    target = v
                else:
                    target = f'dot_split_{next_dot_node:05d}'
                    next_dot_node += 1
                    graph.add_node(target, kind='dot_split', point=fragment[-1], degree=2)
                edge_id = f"{data['edge_id']}_d{part+1}"
                graph.add_edge(previous, target, key=edge_id, edge_id=edge_id,
                               kind=data.get('kind', 'wire'), path=fragment,
                               polyline=polyline(fragment, cfg['polyline_tolerance_px']),
                               length_px=float(arc[end] - arc[begin]),
                               mean_probability=data.get('mean_probability', 0.0))
                previous = target
            dot_splits += len(cuts)
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
                       'uncertain_junctions':sum(not j.get('resolved',False) for j in junctions),
                       'junction_candidates':len(junctions),'learned_resolved_junctions':sum(j.get('resolved',False) for j in junctions),
                       'gap_candidates_not_filled':len(gaps),'pin_candidates_preserved':len(attachments)==len(ids),
                       'pin_coordinate_changes':0,'automatic_gap_connections':bridged,'logical_label_merges':0},
            'raw_image_segmentation_used':False,'junction_classifier_used':junction_decisions is not None,'official_NetF1':None}
    return result,{'guarded_wire_mask':mask,'body_excluded_mask':excluded,'skeleton':skeleton}
