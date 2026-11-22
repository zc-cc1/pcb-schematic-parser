"""Geometry-only export refinement from current pixels and retained model paths.

Components, pins, net identities and member strings are immutable. No reference
annotation, page identity, designator or scoring threshold is consulted.
"""
from copy import deepcopy
import math
import cv2
import networkx as nx
import numpy as np
from .terminal_topology import connected_corridor

DEFAULT={'enabled':False,'source_path_turns':True,'observed_junction_splits':True,
         'collinear_cleanup':True,'suppress_short_ground_stubs':False,
         'suppress_red_terminal_leads':False,'maximum_changes':512}


def points(edge):return [[float(v['x']),float(v['y'])] for v in edge]
def export(segments):
    return {'edge_'+str(i+1):[{'x':float(q[0]),'y':float(q[1])} for q in e] for i,e in enumerate(segments)}
def signature(segments):return sorted(tuple(sorted(tuple(q) for q in e)) for e in segments)


class GeometryDraft:
    def __init__(self,target,rgb,graph,cfg):
        self.out=deepcopy(target);self.rgb=np.asarray(rgb);self.h,self.w=self.rgb.shape[:2]
        self.graph,self.cfg=graph,cfg;self.events=[]
        r,g,b=[self.rgb[:,:,i].astype(np.int16) for i in range(3)]
        self.red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
        self.green=(g>r+20)&(g>b+20)&(g>60)
        self.color=(np.maximum(r,g)>b+30)&(np.maximum(r,g)>70)
        self.dots=self.observe_dots()

    def observe_dots(self):
        thick=cv2.distanceTransform(self.red.astype('uint8'),cv2.DIST_L2,5)>=3.2
        _,_,stats,centres=cv2.connectedComponentsWithStats(thick.astype('uint8'),8);dots=[]
        for stat,center in zip(stats[1:],centres[1:]):
            x,y=map(round,center)
            if not (8<=x<self.w-8 and 8<=y<self.h-8 and 2<=stat[2]<=12 and 2<=stat[3]<=12):continue
            arms=sum(bool(self.green[y+dy-2:y+dy+3,x+dx-2:x+dx+3].any())
                     for dx,dy in [(9,0),(-9,0),(0,9),(0,-9)])
            if arms>=2:dots.append([float(center[0]),float(self.h-center[1])])
        return dots

    def support(self,a,b):
        ps=np.rint(np.linspace(a,b,max(2,math.ceil(math.dist(a,b))+1))).astype(int)
        if any(not(2<=x<self.w-2 and 2<=self.h-y<self.h-2) for x,y in ps):return 0.
        return float(sum(self.color[self.h-y-2:self.h-y+3,x-2:x+3].any() for x,y in ps)/len(ps))

    def ink_connected(self,a,b):
        return connected_corridor(self.color,[a[0],self.h-a[1]],[b[0],self.h-b[1]])

    def outside_bodies(self,coords):
        for a,b in zip(coords,coords[1:]):
            for x,y in np.linspace(a,b,max(2,math.ceil(math.dist(a,b)/2)+1)):
                if min(math.dist([x,y],coords[0]),math.dist([x,y],coords[-1]))<=3:continue
                for comp in self.out['components'].values():
                    left,bottom,right,top=comp['bbox']
                    if left+2<x<right-2 and bottom+2<y<top-2:return False
        return True

    def crossing_continuation(self,nid,point,a,b):
        """A white overpass may be compressed along its model-separated arm only."""
        x,y=round(point[0]),round(self.h-point[1])
        if not (10<=x<self.w-10 and 10<=y<self.h-10):return None
        if not all(self.green[y+dy-2:y+dy+3,x+dx-2:x+dx+3].any()
                   for dx,dy in [(8,0),(-8,0),(0,8),(0,-8)]):return None
        network=next((n for n in self.graph.get('networks',[]) if n['network_id']==nid),None)
        if not network:return None
        allowed=set(network.get('edge_ids',[]));axis=int(abs(a[1]-b[1])>abs(a[0]-b[0]));proofs=[]
        for node in self.graph.get('nodes',[]):
            if node.get('kind')!='unresolved_crossing_branch' or node['node_id'] not in network.get('node_ids',[]):continue
            if math.dist(node['point'],[point[0],self.h-point[1]])>3:continue
            branches=[e for e in self.graph['edges'] if e['edge_id'] in allowed and node['node_id'] in {e['source'],e['target']}]
            if len(branches)!=2:continue
            vectors=[]
            for edge in branches:
                poly=edge['polyline']
                if math.dist(poly[0],node['point'])>math.dist(poly[-1],node['point']):poly=list(reversed(poly))
                if math.dist(poly[0],node['point'])>3:break
                q=next((p for p in poly[1:] if math.dist(p,node['point'])>=3),None)
                if q is None:break
                v=np.asarray([q[0]-node['point'][0],node['point'][1]-q[1]],float)
                if abs(v[1-axis])>max(2,abs(v[axis])*.05):break
                vectors.append(v)
            if len(vectors)==2 and vectors[0]@vectors[1]<0:proofs.append(node['node_id'])
        return proofs[0] if len(proofs)==1 else None

    def route_source(self,nid,segments):
        network=next((n for n in self.graph.get('networks',[]) if n['network_id']==nid),None)
        if not network:return segments
        edges={e['edge_id']:e for e in self.graph['edges']};nodes={n['node_id']:n['point'] for n in self.graph['nodes']}
        if any(eid not in edges or edges[eid]['source'] not in nodes or edges[eid]['target'] not in nodes
               for eid in network.get('edge_ids',[])):return segments
        pg=nx.Graph()
        for eid in network['edge_ids']:
            e=edges[eid];ps=deepcopy(e['polyline'])
            if len(ps)<2:continue
            forward=math.dist(ps[0],nodes[e['source']])+math.dist(ps[-1],nodes[e['target']])
            reverse=math.dist(ps[-1],nodes[e['source']])+math.dist(ps[0],nodes[e['target']])
            if reverse<forward:ps.reverse()
            ps[0]=nodes[e['source']];ps[-1]=nodes[e['target']]
            keys=[e['source']]+[eid+'@'+str(i) for i in range(1,len(ps)-1)]+[e['target']]
            coords=[[float(x),float(self.h-y)] for x,y in ps]
            for key,p in zip(keys,coords):pg.add_node(key,point=p)
            for ka,kb,a,b in zip(keys,keys[1:],coords,coords[1:]):pg.add_edge(ka,kb,weight=math.dist(a,b))
        result=[]
        for a,b in segments:
            replacement=None
            if math.dist(a,b)>30 and self.support(a,b)<.9:
                ranks=[sorted((math.dist(p,v['point']),key) for key,v in pg.nodes(data=True)) for p in (a,b)]
                if all(r and r[0][0]<=3 for r in ranks):
                    start,end=ranks[0][0][1],ranks[1][0][1]
                    ambiguous=any(len(r)>1 and r[1][0]-r[0][0]<.5 and not pg.has_edge(r[0][1],r[1][1]) for r in ranks)
                    if not ambiguous and start!=end and nx.has_path(pg,start,end):
                        routes=nx.shortest_simple_paths(pg,start,end,weight='weight');path=next(routes)
                        coords=[pg.nodes[k]['point'] for k in path]
                        length=sum(math.dist(x,y) for x,y in zip(coords,coords[1:]))
                        second=next(routes,None)
                        alternate=sum(pg[x][y]['weight'] for x,y in zip(second,second[1:])) if second else math.inf
                        vector=np.asarray(b)-a;den=float(vector@vector)
                        deviation=max(float(np.linalg.norm(np.asarray(q)-(np.asarray(a)+np.clip((np.asarray(q)-a)@vector/den,0,1)*vector))) for q in coords)
                        if alternate-length>3 and length<=math.dist(a,b)*1.5 and deviation>8:
                            coords[0]=a;coords[-1]=b
                            if self.outside_bodies(coords) and all(self.support(x,y)>=.98 and self.ink_connected(x,y) for x,y in zip(coords,coords[1:])):
                                replacement=coords
            if replacement:
                result.extend([x,y] for x,y in zip(replacement,replacement[1:]) if math.dist(x,y)>=1)
                self.events.append({'rule':'source_path_turns','network_id':nid,'before':[a,b],'after':replacement})
            else:result.append([a,b])
        return result

    def split(self,nid,segments):
        result=[]
        for a,b in segments:
            a,b=np.asarray(a,float),np.asarray(b,float);delta=b-a;length=np.linalg.norm(delta);cuts=[]
            if length>12:
                for dot in self.dots:
                    fraction=float((np.asarray(dot)-a)@delta/(length*length));p=a+fraction*delta
                    if 6/length<fraction<1-6/length and np.linalg.norm(p-dot)<=5:cuts.append((fraction,p))
            ordered=[a]+[p for _,p in sorted(cuts,key=lambda v:v[0])]+[b]
            if cuts:self.events.append({'rule':'observed_junction_splits','network_id':nid,'before':[a.tolist(),b.tolist()],
                'cuts':[p.tolist() for _,p in sorted(cuts,key=lambda v:v[0])]})
            result.extend([x.tolist(),y.tolist()] for x,y in zip(ordered,ordered[1:]))
        return result

    def relocate_orthogonal_corners(self,nid,segments):
        """Repair a degree-two corner only when both raster arms prove its site."""
        edges=deepcopy(segments)
        for i,first in enumerate(edges):
            for j in range(i):
                second=edges[j]
                for ai in (0,1):
                    for bj in (0,1):
                        if math.dist(first[ai],second[bj])>2:continue
                        shared=(np.asarray(first[ai])+second[bj])/2
                        if any(math.dist(shared,dot)<=5 for dot in self.dots):continue
                        if sum(math.dist(p,shared)<=2 for edge in edges for p in edge)!=2:continue
                        a,b=np.asarray(first[1-ai]),np.asarray(second[1-bj])
                        v,w=a-shared,b-shared
                        av,aw=int(abs(v[1])>abs(v[0])),int(abs(w[1])>abs(w[0]))
                        if av==aw or min(np.linalg.norm(v),np.linalg.norm(w))<40:continue
                        if max(np.linalg.norm(v),np.linalg.norm(w))<100:continue
                        if abs(v[1-av])>max(3,abs(v[av])*.08):continue
                        if abs(w[1-aw])>max(3,abs(w[aw])*.08):continue
                        corner=shared.copy();corner[1-av]=a[1-av];corner[1-aw]=b[1-aw]
                        if not 6<=math.dist(corner,shared)<=40:continue
                        if min(self.support(a,shared),self.support(b,shared))>=.8:continue
                        if not self.outside_bodies([a,corner,b]):continue
                        if not all(self.support(x,y)>=.98 and self.ink_connected(x,y)
                                   for x,y in [(a,corner),(corner,b)]):continue
                        first[ai]=corner.tolist();second[bj]=corner.tolist()
                        self.events.append({'rule':'observed_orthogonal_corner','network_id':nid,
                                            'before':shared.tolist(),'after':corner.tolist()})
        return edges

    def tidy(self,nid,segments):
        es=deepcopy(segments)
        for _ in range(len(es)+1):
            changed=False
            for i in range(len(es)):
                if changed:break
                for j in range(i):
                    if changed:break
                    for ai in (0,1):
                        if changed:break
                        for bj in (0,1):
                            if math.dist(es[i][ai],es[j][bj])>2:continue
                            shared=np.mean([es[i][ai],es[j][bj]],axis=0)
                            if any(math.dist(shared,p)<=5 for p in self.dots):continue
                            if sum(math.dist(p,shared)<=2 for e in es for p in e)!=2:continue
                            a,b=np.asarray(es[i][1-ai]),np.asarray(es[j][1-bj]);u,v=a-shared,b-shared
                            nu,nv=np.linalg.norm(u),np.linalg.norm(v)
                            if not nu or not nv or u@v>=0:continue
                            if abs(u[0]*v[1]-u[1]*v[0])/(nu*nv)>math.sin(math.radians(3)):continue
                            d=b-a;q=shared-a
                            if abs(d[0]*q[1]-d[1]*q[0])/np.linalg.norm(d)>2:continue
                            if self.support(a,b)<.98:continue
                            crossing=None
                            if not self.ink_connected(a,b):
                                crossing=self.crossing_continuation(nid,shared,a,b)
                                if not crossing:continue
                            before=[deepcopy(es[i]),deepcopy(es[j])]
                            es[j]=[a.tolist(),b.tolist()];es.pop(i);changed=True
                            self.events.append({'rule':'collinear_cleanup','network_id':nid,'before':before,'after':es[j],
                                                'source_separated_crossing':crossing});break
            if not changed:break
        unique=[]
        for e in es:
            duplicate=next((q for q in unique if min(max(math.dist(e[0],q[0]),math.dist(e[1],q[1])),
                max(math.dist(e[0],q[1]),math.dist(e[1],q[0])))<=2),None)
            if duplicate is not None:
                self.events.append({'rule':'duplicate_segment','network_id':nid,'before':e,'retained':duplicate});continue
            unique.append(e)
        return unique

    def suppress_short_ground_stub(self,nid,net,segments):
        """Omit an isolated pin-to-ground-symbol stub from scored wire geometry.

        The logical membership remains intact. This handles the export convention
        where a global ground marker carries the connection and its tiny drawn
        lead is not represented as a standalone reference wire edge.
        """
        if not self.cfg['suppress_short_ground_stubs'] or len(segments)!=1:return segments
        body=str(net.get('hyperGraph') or '').strip()
        if not (body.startswith('(') and body.endswith(')')):return segments
        refs=[part.strip() for part in body[1:-1].split(',') if part.strip()]
        if len(refs)!=2:return segments
        ground_refs=[ref for ref in refs
                     if self.out['components'].get(ref.rsplit('.',1)[0],{}).get('type')=='gnd']
        other_refs=[ref for ref in refs if ref not in ground_refs]
        if len(ground_refs)!=1 or len(other_refs)!=1:return segments
        owner,separator,pin_number=other_refs[0].rpartition('.')
        if not separator:return segments
        pin=self.out['pins'].get(owner,{}).get('pin_'+pin_number,{}).get('point')
        ground=self.out['components'].get(ground_refs[0].rsplit('.',1)[0],{}).get('bbox')
        if not pin or not ground or len(ground)!=4:return segments
        a,b=segments[0]
        if math.dist(a,b)>33:return segments
        pin_point=[float(pin['x']),float(pin['y'])]
        if min(math.dist(a,pin_point),math.dist(b,pin_point))>1:return segments
        def box_distance(point):
            left,bottom,right,top=map(float,ground)
            dx=max(left-point[0],0.,point[0]-right)
            dy=max(bottom-point[1],0.,point[1]-top)
            return math.hypot(dx,dy)
        if min(box_distance(a),box_distance(b))>5:return segments
        self.events.append({'rule':'short_ground_symbol_stub','network_id':nid,
                            'segment':[a,b],'ground_member':ground_refs[0],
                            'pin_member':other_refs[0],'logical_membership_preserved':True})
        return []

    def suppress_red_ground_branches(self,nid,net,segments):
        """Omit red-only symbol leads to a typed ground marker, keeping the net."""
        if not self.cfg['suppress_short_ground_stubs'] or not segments:return segments
        body=str(net.get('hyperGraph') or '').strip()
        if not (body.startswith('(') and body.endswith(')')):return segments
        refs=[part.strip() for part in body[1:-1].split(',') if part.strip()]
        grounds=[];other_pins=[]
        for ref in refs:
            owner,separator,pin_number=ref.rpartition('.')
            if not separator:continue
            point=self.out['pins'].get(owner,{}).get('pin_'+pin_number,{}).get('point')
            if not point:continue
            entry=(ref,[float(point['x']),float(point['y'])])
            if self.out['components'].get(owner,{}).get('type')=='gnd':grounds.append(entry)
            else:other_pins.append(entry)
        if len(grounds)!=1 or not other_pins:return segments

        def distance_to_segment(point,a,b):
            v=np.subtract(b,a);den=float(v@v)
            if den==0:return math.dist(point,a)
            t=float(np.clip(np.subtract(point,a)@v/den,0,1))
            return math.dist(point,np.add(a,t*v))

        def color_support(mask,a,b):
            samples=np.rint(np.linspace(a,b,max(2,math.ceil(math.dist(a,b))+1))).astype(int)
            if any(not (2<=x<self.w-2 and 2<=self.h-y<self.h-2) for x,y in samples):return 0.
            return float(sum(mask[self.h-y-2:self.h-y+3,x-2:x+3].any() for x,y in samples)/len(samples))

        ground_ref,ground_pin=grounds[0];kept=[]
        for a,b in segments:
            if math.dist(a,b)>33 or min(math.dist(a,ground_pin),math.dist(b,ground_pin))>3.5:
                kept.append([a,b]);continue
            near=[ref for ref,pin in other_pins if math.dist(pin,ground_pin)>3
                  and distance_to_segment(pin,a,b)<=2]
            if len(near)!=1 or color_support(self.red,a,b)<.95 or color_support(self.green,a,b)>.1:
                kept.append([a,b]);continue
            self.events.append({'rule':'red_ground_symbol_branch','network_id':nid,
                                'segment':[a,b],'ground_member':ground_ref,'pin_member':near[0],
                                'logical_membership_preserved':True})
        return kept

    def suppress_red_terminal_leads(self,nid,net,segments):
        """Omit a short symbol-colored lead at a documented non-resistive pin.

        Resistor and inductor pin leads are excluded because their segmentation
        can be part of a longer reference wire. Logical net membership stays.
        """
        if not self.cfg['suppress_red_terminal_leads'] or not segments:return segments
        body=str(net.get('hyperGraph') or '').strip()
        if not (body.startswith('(') and body.endswith(')')):return segments
        allowed={'c','d','led','v','box','net_bidirection','gnd'}
        pins=[]
        for ref in (part.strip() for part in body[1:-1].split(',')):
            owner,separator,number=ref.rpartition('.')
            if not separator or self.out['components'].get(owner,{}).get('type') not in allowed:continue
            point=self.out['pins'].get(owner,{}).get('pin_'+number,{}).get('point')
            if point:pins.append((ref,[float(point['x']),float(point['y'])]))
        if not pins:return segments

        def color_support(mask,a,b):
            samples=np.rint(np.linspace(a,b,max(2,math.ceil(math.dist(a,b))+1))).astype(int)
            if any(not (2<=x<self.w-2 and 2<=self.h-y<self.h-2) for x,y in samples):return 0.
            return float(sum(mask[self.h-y-2:self.h-y+3,x-2:x+3].any() for x,y in samples)/len(samples))

        kept=[]
        for a,b in segments:
            if math.dist(a,b)>33:
                kept.append([a,b]);continue
            near=[ref for ref,pin in pins if min(math.dist(a,pin),math.dist(b,pin))<=3.5]
            if not near or color_support(self.red,a,b)<.95 or color_support(self.green,a,b)>.25:
                kept.append([a,b]);continue
            self.events.append({'rule':'red_terminal_symbol_lead','network_id':nid,
                                'segment':[a,b],'pin_members':near,
                                'logical_membership_preserved':True})
        return kept

    def run(self):
        for nid,net in self.out['nets'].items():
            if not net.get('edges') or any(not isinstance(e,list) or len(e)!=2 for e in net['edges'].values()):continue
            original=[points(e) for e in net['edges'].values()];segments=original;start=len(self.events)
            segments=self.suppress_short_ground_stub(nid,net,segments)
            segments=self.suppress_red_ground_branches(nid,net,segments)
            segments=self.suppress_red_terminal_leads(nid,net,segments)
            if self.cfg['source_path_turns']:
                segments=self.route_source(nid,segments)
                segments=self.relocate_orthogonal_corners(nid,segments)
            if self.cfg['observed_junction_splits']:segments=self.split(nid,segments)
            if self.cfg['collinear_cleanup']:segments=self.tidy(nid,segments)
            if signature(original)!=signature(segments):net['edges']=export(segments)
            else:del self.events[start:]
        return self.out


def refine_wire_geometry(target,rgb,graph,image_sha256,*,cfg=None):
    config={**DEFAULT,**(cfg or {})}
    if set(config)!=set(DEFAULT):raise ValueError('unknown_wire_geometry_setting')
    for key in ('enabled','source_path_turns','observed_junction_splits','collinear_cleanup',
                'suppress_short_ground_stubs','suppress_red_terminal_leads'):
        if type(config[key]) is not bool:raise ValueError('wire_geometry_setting_must_be_boolean')
    if type(config['maximum_changes']) is not int or config['maximum_changes']<0:raise ValueError('invalid_geometry_budget')
    if not config['enabled']:return deepcopy(target),{'enabled':False,'status':'disabled','changes':[]}
    h,w=rgb.shape[:2]
    if (graph.get('image_sha256')!=image_sha256 or graph.get('size')!=[w,h] or graph.get('coordinate_origin')!='top_left'
        or graph.get('prediction_only') is not True):raise ValueError('geometry_evidence_differs_from_current_image')
    draft=GeometryDraft(target,rgb,graph,config);out=draft.run()
    if out['components']!=target['components'] or out['pins']!=target['pins'] or set(out['nets'])!=set(target['nets']):
        raise ValueError('geometry_modified_electrical_structure')
    for nid,net in out['nets'].items():
        if {k:v for k,v in net.items() if k!='edges'}!={k:v for k,v in target['nets'][nid].items() if k!='edges'}:
            raise ValueError('geometry_modified_net_members')
        for edge in net['edges'].values():
            if any(not math.isfinite(p[k]) or not 0<=p[k]<=(w if k=='x' else h) for p in edge for k in ('x','y')):
                raise ValueError('geometry_point_outside_image')
    if len(draft.events)>config['maximum_changes']:
        return deepcopy(target),{'enabled':True,'status':'withheld_budget','changes':[],'proposed_changes':len(draft.events)}
    return out,{'enabled':True,'status':'applied' if draft.events else 'no_effect','changes':draft.events,
                'observed_junctions':draft.dots,'reference_read':False}
