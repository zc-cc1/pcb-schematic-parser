"""OCR-to-physical-network binding and auditable ordinary-label proposals.

Uses model-produced boxes, pin points, OCR and wire graphs only. Never extracts
raw-image lines or reads CAD/Golden. Physical edges remain unchanged.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import re

import networkx as nx
import numpy as np

from .component_text import REF,value_kind,bbox,center,gap,iou


def power_name(text,cfg):
    return any(re.fullmatch(pattern,text,re.I) for pattern in cfg['power_name_patterns'])


def label_text(text,cfg):
    text=text.strip()
    if not text or len(text)>40 or not re.search(r'[A-Za-z]',text):return None
    if not re.fullmatch(r'[A-Za-z0-9_+./~!#()\[\] -]+',text):return None
    if text.upper() in {'NC','N/C','DNC','NO CONNECT','PAGE','SCHEMATIC','HEADER','POWER SUPPLY'}:return None
    if REF.fullmatch(text) or re.fullmatch(r'(?:RP|RN|CN|K|H|EC)\d+',text,re.I) or value_kind(text) is not None:return None
    if text.lower() in {'key','cap','res','cap_0402','r_0402','header 2','header 3'}:return None
    if len(text.split())>3:return None
    return text


def within(point,box):return box[0]<=point[0]<=box[2] and box[1]<=point[1]<=box[3]


def point_box_gap(point,box):return gap([*point,*point],box)


def observations(texts,size):
    result=[];seen=set();width,height=size
    for row in texts:
        rb=row.get('bbox') or bbox(row['polygon'])
        for observation in [{'text':row['text'],'confidence':row['confidence'],'bbox':rb,'view':row.get('view')}]+row.get('observations',[]):
            box=list(map(float,observation.get('bbox',rb)));score=float(observation['confidence'])
            if len(box)!=4 or not np.isfinite(box+[score]).all() or not 0<=score<=1 or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:
                raise ValueError('OCR geometry/confidence outside original image')
            key=(str(observation['text']),tuple(round(x,2) for x in box),observation.get('view',row.get('view')))
            if key in seen:continue
            seen.add(key)
            result.append({'text':str(observation['text']),'confidence':score,'bbox':box,
                           'source_text_id':row['text_id'],'source_view':observation.get('view',row.get('view'))})
    # A location may have several observed spellings. Equal names at different
    # locations remain separate sites; overlapping OCR duplicates do not merge.
    groups=nx.Graph();groups.add_nodes_from(range(len(result)))
    for i,a in enumerate(result):
        for j in range(i):
            b=result[j]
            ba,bb=a['bbox'],b['bbox']
            intersection=max(0,min(ba[2],bb[2])-max(ba[0],bb[0]))*max(0,min(ba[3],bb[3])-max(ba[1],bb[1]))
            minimum_area=min((ba[2]-ba[0])*(ba[3]-ba[1]),(bb[2]-bb[0])*(bb[3]-bb[1]))
            partial=(a['text'].strip() in b['text'].strip() or b['text'].strip() in a['text'].strip()) and intersection/minimum_area>=.8
            if iou(ba,bb)>=.45 or partial:groups.add_edge(i,j)
    return [[result[i] for i in sorted(group)] for group in nx.connected_components(groups)]


def endpoint_anchors(graph,cfg):
    nodes={n['node_id']:n for n in graph['nodes']}
    by_node=defaultdict(list)
    for edge in graph['edges']:
        by_node[edge['source']].append(edge);by_node[edge['target']].append(edge)
    memberships={node:n['network_id'] for n in graph['networks'] for node in n['node_ids']}
    result=[]
    for node,edges in by_node.items():
        if nodes[node]['kind']!='endpoint' or len(edges)!=1:continue
        edge=edges[0]
        if edge['length_px']<cfg['endpoint_min_wire_length_px']:continue
        line=edge['polyline']
        point=nodes[node]['point']
        if math.dist(line[-1],point)<math.dist(line[0],point):line=list(reversed(line))
        if len(line)<2:continue
        delta=np.asarray(line[0],float)-np.asarray(line[1],float);norm=float(np.linalg.norm(delta))
        if not norm:continue
        result.append({'point':point,'outward_direction':(delta/norm).tolist(),'network_id':memberships[node],
                       'node_id':node,'edge_id':edge['edge_id']})
    return result


def symbol_matches(box,pin_result,graph,cfg):
    attachments={p['candidate_id']:p for p in graph['pin_attachments']}
    result=[]
    for component in pin_result['components']:
        if component['type'] not in cfg['label_component_types']+cfg['power_component_types']:continue
        cb=component['bbox_top_left'];distance=gap(box,cb)
        if distance>cfg['symbol_text_max_gap_px']:continue
        owned=[p for p in pin_result['pin_candidates'] if p['owner_status']=='assigned' and p.get('owner_detection_index')==component['detection_index']]
        attached=[attachments[p['candidate_id']] for p in owned if attachments[p['candidate_id']]['attachment_status']=='attached']
        nets={p['network_id'] for p in attached}
        # Do not guess which of multiple possible electrical contacts is a tip.
        unique_points={tuple(p['point']) for p in attached}
        if not nets:continue
        if component['type']=='net_short':
            # The detector class is a two-sided short/diamond symbol. Associate
            # only when two learned tip clusters sit on opposite box sides and
            # a nearby actual component establishes the external text side.
            clusters=[]
            for p in attached:
                for cluster in clusters:
                    if math.dist(p['point'],cluster[0]['point'])<=cfg['net_short_tip_cluster_distance_px']:
                        cluster.append(p);break
                else:clusters.append([p])
            if len(clusters)!=2:continue
            horizontal=cb[2]-cb[0]>=cb[3]-cb[1];normal=0 if horizontal else 1;tangential=1-normal
            tips=[c[0]['point'] for c in clusters];mid=center(cb)
            if (tips[0][normal]-mid[normal])*(tips[1][normal]-mid[normal])>=0:continue
            if any(min(abs(t[normal]-cb[normal]),abs(t[normal]-cb[normal+2]))>6 for t in tips):continue
            neighbors=[c for c in pin_result['components'] if c['type'] not in cfg['label_component_types']+cfg['power_component_types']
                       and gap(cb,c['bbox_top_left'])<=cfg['net_short_context_component_max_gap_px']]
            if not neighbors:continue
            nearest=min(neighbors,key=lambda c:gap(cb,c['bbox_top_left']))
            body=center(nearest['bbox_top_left']);text=center(box)
            if (text[normal]-mid[normal])*(body[normal]-mid[normal])>=0:continue
            # Horizontal labels usually sit immediately above the green lead;
            # the bottom of the OCR glyph box is the useful baseline cue.
            target=box[3]-1 if horizontal else center(box)[0]
            tangent=abs(target-float(np.mean([t[tangential] for t in tips])))
            if tangent>4.5:continue
            anchor=min(attached,key=lambda p:math.dist(p['point'],body))
            result.append({'cost':distance+.7*tangent,'network_id':anchor['network_id'],'network_ids':sorted(nets),
                           'point':anchor['point'],'points':[p['point'] for p in attached],
                           'evidence_kind':'detected_two_tip_net_short_external_text',
                           'component_id':component['component_id'],'detection_index':component['detection_index'],
                           'component_type':component['type'],'candidate_id':anchor['candidate_id'],
                           'owned_candidate_ids':[p['candidate_id'] for p in attached],
                           'topology_caveat':'Two-sided detector symbol assumed locally conductive; ordinary-label proposal, not certified pin numbering'})
            continue
        if len(nets)!=1 or len(unique_points)!=1:continue
        anchor=attached[0]
        x,y=anchor['point'];cx,cy=center(box)
        side=min([('l',abs(x-cb[0])),('r',abs(x-cb[2])),('t',abs(y-cb[1])),('b',abs(y-cb[3]))],key=lambda r:r[1])[0]
        axis=1 if side in {'l','r'} else 0
        tangent=abs([cx,cy][axis]-[x,y][axis])
        limit=cfg['symbol_text_tangent_limit_px']
        for neighbor in pin_result['components']:
            if neighbor['detection_index']==component['detection_index'] or neighbor['type'] not in cfg['label_component_types']:continue
            other=center(neighbor['bbox_top_left']);this=center(cb)
            if abs(other[1-axis]-this[1-axis])>12:continue
            spacing=abs(other[axis]-this[axis])
            if spacing>3:limit=min(limit,max(3,spacing*.45))
        if tangent>limit:continue
        inward={'l':cx-x,'r':x-cx,'t':cy-y,'b':y-cy}[side]
        if inward<-2:continue  # text on the attached wire side is a different role
        result.append({'cost':distance+.5*tangent+.08*math.dist(center(box),center(cb)),
                       'network_id':anchor['network_id'],'network_ids':sorted(nets),'point':anchor['point'],'evidence_kind':'detected_symbol_and_owned_pin',
                       'component_id':component['component_id'],'detection_index':component['detection_index'],
                       'component_type':component['type'],'candidate_id':anchor['candidate_id']})
    return result


def endpoint_matches(box,anchors,pin_result,cfg):
    result=[]
    horizontal=box[2]-box[0]>=box[3]-box[1]
    axis=0 if horizontal else 1;other=1-axis
    limit=math.cos(math.radians(cfg['endpoint_max_tangent_angle_degrees']))
    for anchor in anchors:
        point=anchor['point'];direction=anchor['outward_direction']
        if within(point,box):continue  # glyph skeleton or overlapping label is ambiguous
        distance=point_box_gap(point,box)
        if distance>cfg['endpoint_text_max_gap_px'] or abs(direction[axis])<limit:continue
        # Text beside the endpoint, or immediately above a horizontal wire.
        tangent=max(box[other]-point[other],point[other]-box[other+2],0.)
        if tangent>cfg['endpoint_text_tangent_limit_px']:continue
        delta=center(box)[axis]-point[axis]
        if delta*direction[axis]<0:continue
        if any(c['type'] not in cfg['label_component_types'] and within(point,c['bbox_top_left']) for c in pin_result['components']):continue
        result.append({'cost':distance+.5*tangent,'network_id':anchor['network_id'],'point':point,
                       'node_id':anchor['node_id'],'edge_id':anchor['edge_id'],'evidence_kind':'model_wire_endpoint_next_to_OCR'})
    return result


def bind_network_labels(pin_result,graph,texts,component_records,cfg):
    if graph['size']!=pin_result['size'] or graph['coordinate_origin']!='top_left':raise ValueError('Graph and pin coordinate contracts differ')
    if graph.get('image_sha256')!=pin_result.get('image_sha256'):raise ValueError('Graph and pin images differ')
    blocked=[r['fields'][f]['bbox'] for r in component_records for f in ['designator','Name','value'] if f in r.get('fields',{})]
    blocked += [p['text_evidence'][field]['bbox'] for p in pin_result['pin_candidates'] for field in ['number','pinname'] if field in p.get('text_evidence',{})]
    anchors=endpoint_anchors(graph,cfg);sites=[]
    for group in observations(texts,pin_result['size']):
        eligible=[]
        for token in group:
            label=label_text(token['text'],cfg)
            if label and token['confidence']>=cfg['min_ocr_confidence']:eligible.append({**token,'label':label})
        if not eligible:continue
        # A cropped substring is not evidence for replacing a complete word.
        eligible=[t for t in eligible if not any(t['label']!=o['label'] and t['label'] in o['label'] and
                    len(t['label'])<len(o['label']) and o['confidence']>=t['confidence']-.08 for o in eligible)]
        ranked=sorted(eligible,key=lambda r:-r['confidence']);chosen=ranked[0]
        views=defaultdict(set)
        for token in ranked:views[token['label']].add(token['source_view'])
        vote_order=sorted(views,key=lambda label:(-len(views[label]),-max(t['confidence'] for t in ranked if t['label']==label)))
        vote_supported=len(views[vote_order[0]])>=cfg['ocr_vote_min_distinct_views'] and (
            len(vote_order)==1 or len(views[vote_order[0]])-len(views[vote_order[1]])>=cfg['ocr_vote_min_view_lead'])
        if vote_supported:chosen=max((t for t in ranked if t['label']==vote_order[0]),key=lambda t:t['confidence'])
        site={'site_id':f'LABEL_SITE_{len(sites)+1:05d}',**chosen,'coordinate_origin':'top_left',
              'image_sha256':pin_result.get('image_sha256'),
              'observed_spellings':sorted({r['label'] for r in eligible}),'alternatives':ranked[:6],
              'network_id':None,'point':None,'merge_eligible':False,
              'distinct_view_votes':{name:len(v) for name,v in views.items()},'spelling_selected_by_observed_view_votes':vote_supported}
        sites.append(site);box=chosen['bbox'];text_point=center(box)
        if any(iou(box,b)>=cfg['text_role_overlap_iou'] for b in blocked):
            site['status']='already_assigned_component_or_pin_text';continue
        if any(c['type'] not in cfg['label_component_types']+cfg['power_component_types'] and within(text_point,c['bbox_top_left']) for c in pin_result['components']):
            site['status']='inside_nonlabel_component';continue
        if not vote_supported and any(r['label']!=chosen['label'] and chosen['confidence']-r['confidence']<cfg['spelling_conflict_margin'] for r in ranked):
            site['status']='OCR_spelling_conflict';continue
        options=symbol_matches(box,pin_result,graph,cfg)
        # Symbol and endpoint evidence can agree; rank symbol evidence first,
        # but retain distinct networks to expose nearby conflicting assignments.
        for option in endpoint_matches(box,anchors,pin_result,cfg):options.append({**option,'cost':option['cost']+5})
        by_net={}
        for option in sorted(options,key=lambda o:o['cost']):by_net.setdefault(option['network_id'],option)
        options=sorted(by_net.values(),key=lambda o:o['cost']);site['attachment_candidates']=options[:5]
        if not options:site['status']='no_unambiguous_model_wire_or_symbol_anchor';continue
        if len(options)>1 and options[1]['cost']-options[0]['cost']<cfg['network_attachment_margin_px']:
            site['status']='ambiguous_nearby_networks';continue
        best=options[0];site.update(best)
        site.setdefault('network_ids',[site['network_id']])
        is_power=power_name(site['label'],cfg) or best.get('component_type') in cfg['power_component_types']
        if best.get('component_type') in cfg['power_component_types'] and not power_name(site['label'],cfg):
            site.update(status='power_symbol_text_role_unresolved',merge_eligible=False);continue
        site.update(status='power_semantics_pending' if is_power else 'bound_ordinary_label',
                    merge_eligible=not is_power,matching_rule='exact_observed_spelling_outer_whitespace_trim_only')
        if not is_power and best['evidence_kind']=='model_wire_endpoint_next_to_OCR' and cfg['symbol_evidence_required_for_merge']:
            site.update(status='endpoint_text_role_unverified',merge_eligible=False)
    # One drawing location cannot become two occurrences through overlapping
    # OCR views or duplicated detector boxes.
    kept=[]
    for site in sorted(sites,key=lambda r:-r['confidence']):
        if site['point'] is not None and any(other['point'] is not None and site['label']==other['label'] and
             math.dist(site['point'],other['point'])<=cfg['duplicate_site_distance_px'] and iou(site['bbox'],other['bbox'])>.1 for other in kept):
            site['status']='duplicate_physical_label_site';site['merge_eligible']=False
        else:kept.append(site)
    by_symbol=defaultdict(list)
    for site in sites:
        if site['merge_eligible'] and site.get('detection_index') is not None:by_symbol[site['detection_index']].append(site)
    for rows in by_symbol.values():
        ordered=sorted(rows,key=lambda r:r['cost'])
        different=[r for r in ordered[1:] if r['label']!=ordered[0]['label']]
        ambiguous=bool(different and different[0]['cost']-ordered[0]['cost']<cfg['network_attachment_margin_px'])
        for site in ordered:
            if ambiguous or site['label']!=ordered[0]['label']:
                site.update(status='ambiguous_symbol_text_roles' if ambiguous else 'symbol_text_used_by_better_label',merge_eligible=False)
    return sites


def merge_ordinary_labels(graph,sites,cfg):
    physical={n['network_id']:n for n in graph['networks']}
    if len({s['site_id'] for s in sites})!=len(sites):raise ValueError('Duplicate label site IDs')
    graph_nx=nx.Graph();graph_nx.add_nodes_from(physical)
    by_name=defaultdict(list);names_on_net=defaultdict(set)
    for site in sites:
        if site['merge_eligible']:
            if site.get('image_sha256',graph.get('image_sha256'))!=graph.get('image_sha256'):raise ValueError('Cross-image label site')
            if not set(site.get('network_ids',[site['network_id']]))<=set(physical):raise ValueError('Unknown physical network in label site')
            by_name[site['label']].append(site)
            qualified=site['evidence_kind'] in {'detected_symbol_and_owned_pin','detected_two_tip_net_short_external_text'}
            if qualified or not cfg['symbol_evidence_required_for_merge']:
                for nid in site.get('network_ids',[site['network_id']]):names_on_net[nid].add(site['label'])
    gap_nets={nid for g in graph['gap_candidates'] for nid in g['network_ids']}
    decisions=[]
    for label,rows in sorted(by_name.items()):
        nets=sorted({nid for s in rows for nid in s.get('network_ids',[s['network_id']])});reasons=[]
        if power_name(label,cfg) or any(s.get('component_type') in cfg['power_component_types'] for s in rows):reasons.append('power_semantics_not_implemented')
        if cfg['symbol_evidence_required_for_merge'] and any(s['evidence_kind'] not in {'detected_symbol_and_owned_pin','detected_two_tip_net_short_external_text'} for s in rows):
            reasons.append('label_symbol_evidence_missing')
        if len(rows)<2:reasons.append('single_label_occurrence')
        if len(nets)<2:reasons.append('already_same_physical_network')
        if any(len(names_on_net[n])>1 for n in nets):reasons.append('multiple_distinct_labels_on_one_network')
        if not cfg['merge_networks_with_unresolved_junctions'] and any(physical[n]['unresolved_junction_ids'] for n in nets):reasons.append('junction_semantics_unresolved')
        if not cfg['merge_networks_touching_gap_candidates'] and set(nets)&gap_nets:reasons.append('suspected_wire_disconnection')
        if not cfg['enable_ordinary_label_merges']:reasons.append('automatic_merge_disabled')
        decision={'label':label,'site_ids':[s['site_id'] for s in rows],'physical_network_ids':nets,
                  'status':'withheld' if reasons else 'logical_membership_proposal_applied','withheld_reasons':reasons,
                  'scope_image_sha256':graph.get('image_sha256'),'physical_wire_added':False}
        decisions.append(decision)
        if not reasons:
            for nid in nets[1:]:graph_nx.add_edge(nets[0],nid,label=label)
    applied={s['site_id'] for d in decisions if not d['withheld_reasons'] for s in by_name[d['label']]}
    logical=deepcopy(graph);logical['physical_networks']=deepcopy(graph['networks']);logical['networks']=[];mapping={}
    for index,group in enumerate(sorted(nx.connected_components(graph_nx),key=lambda g:min(g))):
        old=[physical[n] for n in sorted(group)];nid=f'LOG_NET_CAND_{index+1:05d}'
        points=[{'site_id':s['site_id'],'label':s['label'],'point':s['point'],'component_id':s.get('component_id'),
                 'candidate_id':s.get('candidate_id'),'owned_candidate_ids':s.get('owned_candidate_ids',[s.get('candidate_id')]),
                 'points':s.get('points',[s['point']]),'evidence_kind':s['evidence_kind']} for s in sites if s['site_id'] in applied and s['network_id'] in group]
        net={'network_id':nid,'physical_network_ids':sorted(group),'node_ids':sorted({x for n in old for x in n['node_ids']}),
             'edge_ids':sorted({x for n in old for x in n['edge_ids']}),'pin_candidate_ids':sorted({x for n in old for x in n['pin_candidate_ids']}),
             'official_pin_references':sorted({x for n in old for x in n['official_pin_references']}),
             'unresolved_junction_ids':sorted({x for n in old for x in n['unresolved_junction_ids']}),
             'logical_label_names':sorted({p['label'] for p in points}),'logical_label_sites':points,
             'flywire':len(group)>1,'status':'logical_membership_proposal_not_certified' if len(group)>1 else old[0]['status']}
        logical['networks'].append(net)
        for p in group:mapping[p]=nid
    for attachment in logical['pin_attachments']:
        attachment['physical_network_id']=attachment['network_id']
        if attachment['network_id'] is not None:attachment['network_id']=mapping[attachment['network_id']]
        for alternative in attachment['alternatives']:
            alternative['physical_network_id']=alternative['network_id'];alternative['network_id']=mapping[alternative['network_id']]
    for gap in logical['gap_candidates']:
        gap['physical_network_ids']=gap['network_ids'];gap['network_ids']=[mapping[n] for n in gap['network_ids']]
    applied_decisions=[d for d in decisions if not d['withheld_reasons']]
    logical.update(status='ordinary_logical_label_proposals_power_semantics_pending',label_sites=sites,label_merge_decisions=decisions,
                   original_physical_edge_geometry_preserved=True,power_label_merges=0,cross_image_merges=0,fuzzy_label_matches=0)
    logical['summary'].update(physical_network_proposals=len(physical),network_proposals=len(logical['networks']),
         multi_pin_network_proposals=sum(len(n['pin_candidate_ids'])>=2 for n in logical['networks']),
         logical_label_merges=len(applied_decisions),logical_networks_merged=sum(len(d['physical_network_ids'])-1 for d in applied_decisions),
         flywire_network_proposals=sum(n['flywire'] for n in logical['networks']),
         label_site_status_counts=dict(Counter(s['status'] for s in sites)),physical_wire_added=0)
    return logical


def build_logical_networks(pin_result,graph,texts,records,cfg):
    sites=bind_network_labels(pin_result,graph,texts,records,cfg)
    return merge_ordinary_labels(graph,sites,cfg)
