"""Exact observed power-caption connectivity proposals on image-model graphs.

Preserves every learned tip and physical edge; invents no source component key,
pin number, pinname, long-distance line, or ordinary flywire geometry.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import re

import networkx as nx

from .component_text import iou
from .pin_text_v2 import tokens_with_views
from .pin_output_rules import power_network_guard


def qualified_name(name, kind, cfg):
    pattern = cfg['ground_name_pattern'] if kind == 'gnd' else cfg['supply_name_pattern'] if kind == 'vcc' else None
    return bool(pattern and re.fullmatch(pattern, name, re.I))


def bind_power_sites(pins, graph, records, texts, cfg):
    owned = defaultdict(list)
    for pin in pins['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            owned[pin['owner_detection_index']].append(pin)
    pmap = {p['candidate_id']:p for p in pins['pin_candidates']}
    attachments = {p['candidate_id']:p for p in graph['pin_attachments']}
    flat = [o for o in tokens_with_views(texts) if o['source_text']==o['text'] and not o.get('approximate_split_geometry')]
    sites, queue = [], []
    by_index = defaultdict(list)
    for site in graph.get('label_sites',[]):
        if site.get('component_type') in {'vcc','gnd'} and site.get('evidence_kind') == 'detected_symbol_and_owned_pin':
            by_index[site['detection_index']].append(site)
    for record in records:
        if record['type'] not in {'gnd','vcc'}:
            continue
        index = record['detection_index']
        candidates = []
        reasons = []
        if len(owned[index]) != 1:
            reasons.append('single_owned_power_tip_not_available')
        for site in by_index[index]:
            if site.get('image_sha256') != pins.get('image_sha256'):
                raise ValueError('Power caption came from another image')
            cid = site.get('candidate_id')
            if cid not in pmap or cid not in attachments:
                raise ValueError('Unknown power caption tip')
            pin, attachment = pmap[cid], attachments[cid]
            if pin.get('owner_detection_index') != index or pin['point'] != site['point'] or pin['point'] != attachment['point']:
                raise ValueError('Power caption ownership or coordinates differ')
            failed = list(reasons)
            label = site['label']
            if not qualified_name(label, record['type'], cfg):
                failed.append('power_name_incompatible_with_detected_symbol')
            readings = [o for o in flat if o['confidence'] >= cfg['min_OCR_confidence'] and o.get('source_view')
                        and iou(o['bbox'],site['bbox']) >= cfg['same_glyph_iou']]
            votes = defaultdict(set)
            for o in readings:
                votes[o['text']].add(o['source_view'])
            views = sorted(votes[label])
            if len(views) < cfg['min_distinct_views']:
                failed.append('insufficient_strong_same_glyph_views')
            if any(text != label and len(v) >= cfg['conflicting_spelling_min_views'] for text,v in votes.items()):
                failed.append('conflicting_supported_power_caption_spellings')
            if attachment['attachment_status'] != 'attached' or attachment['network_id'] is None:
                failed.append('power_tip_wire_attachment_unresolved')
            network = next((n for n in graph['networks'] if n['network_id'] == attachment['network_id']),None)
            if not network or cid not in network['pin_candidate_ids']:
                failed.append('power_tip_absent_from_network_members')
            proposal = {'site_id':site['site_id'], 'label':label, 'component_type':record['type'],
                        'component_id':record['component_id'], 'detection_index':index, 'candidate_id':cid,
                        'point':pin['point'], 'bbox':site['bbox'], 'input_network_id':attachment['network_id'],
                        'confidence':site['confidence'], 'strong_OCR_views':views, 'observed_spellings':{k:len(v) for k,v in votes.items()},
                        'withheld_reasons':failed, 'image_sha256':pins.get('image_sha256'),
                        'canonical_component_key_inferred':False, 'pin_fields_completed':False}
            candidates.append(proposal)
        accepted = [s for s in candidates if not s['withheld_reasons']]
        if len({s['label'] for s in accepted}) > 1:
            for s in accepted:s['withheld_reasons'].append('multiple_power_caption_roles_on_one_symbol')
            accepted = []
        if accepted:
            best = max(accepted,key=lambda s:(len(s['strong_OCR_views']),s['confidence']))
            sites.append(best)
        if not candidates:
            reasons.append('no_model_bound_observed_power_caption')
        queue.append({'detection_index':index,'component_id':record['component_id'],'type':record['type'],
                      'bbox_top_left':record['bbox_top_left'],'owned_tip_ids':[p['candidate_id'] for p in owned[index]],
                      'bound_site_id':sites[-1]['site_id'] if accepted else None,
                      'review_reasons':sorted(set(reasons+[r for s in candidates for r in s['withheld_reasons']])),
                      'caption_candidates':candidates,'canonical_identity_still_unresolved':not bool(record.get('designator'))})
    return sites, queue


def propose_power_networks(pins, graph, records, texts, cfg, guard_cfg):
    if (cfg['name_matching'] != 'exact_literal_no_aliases' or cfg['allow_incomplete_topology_merges']
        or cfg['allow_unrepresented_power_near_wire'] or cfg['allow_ordinary_flywire_in_power_merge'] or cfg['change_pin_fields']):
        raise ValueError('Unsupported power merge policy')
    if graph.get('image_sha256') != pins.get('image_sha256') or graph['size'] != pins['size']:
        raise ValueError('Power graph differs from input image')
    if graph.get('power_logic_summary'):
        raise ValueError('Power stage must start from the frozen ordinary-label graph')
    sites, queue = bind_power_sites(pins,graph,records,texts,cfg)
    nets = {n['network_id']:n for n in graph['networks']}
    guard = power_network_guard(pins,graph,records,guard_cfg)
    blocked = {b['network_id'] for b in guard['blocks']}
    gaps = {nid for g in graph['gap_candidates'] for nid in g['network_ids']}
    by_name, names_on_net = defaultdict(list), defaultdict(set)
    for site in sites:
        by_name[site['label']].append(site)
        names_on_net[site['input_network_id']].add(site['label'])
    connections = nx.Graph();connections.add_nodes_from(nets)
    decisions = []
    for label, group in sorted(by_name.items()):
        members = sorted({s['input_network_id'] for s in group})
        reasons = []
        if len(group)<2:reasons.append('single_observed_power_occurrence')
        if len(members)<2:reasons.append('already_same_input_network')
        if len({s['component_type'] for s in group})>1:reasons.append('same_name_on_mixed_power_symbol_types')
        if any(len(names_on_net[n])>1 for n in members):reasons.append('input_network_has_distinct_power_names')
        if any(nets[n]['unresolved_junction_ids'] for n in members):reasons.append('junction_semantics_unresolved')
        if set(members)&gaps:reasons.append('suspected_wire_disconnection')
        if set(members)&blocked:reasons.append('unrepresented_power_symbol_near_member_wire')
        if any(nets[n].get('flywire') for n in members):reasons.append('ordinary_flywire_network_in_power_group')
        decision = {'label':label,'site_ids':[s['site_id'] for s in group], 'input_network_ids':members,
                    'withheld_reasons':reasons,'status':'withheld' if reasons else 'power_logical_connectivity_proposal',
                    'image_sha256':pins.get('image_sha256'),'creates_ordinary_flywire':False,'physical_wire_added':False}
        decisions.append(decision)
        if not reasons:
            for nid in members[1:]:connections.add_edge(members[0],nid,label=label)
    result = deepcopy(graph);result['networks'] = []; mapping = {}
    for i,group in enumerate(sorted(nx.connected_components(connections),key=lambda g:min(g))):
        old = [nets[n] for n in sorted(group)]
        nid = next(iter(group)) if len(group)==1 else f'PWR_NET_CAND_{i+1:05d}'
        for n in group:mapping[n]=nid
        if len(group)==1:
            result['networks'].append(deepcopy(old[0]));continue
        merged = {'network_id':nid,'input_network_ids':sorted(group),
                  'physical_network_ids':sorted({x for n in old for x in n.get('physical_network_ids',[n['network_id']])}),
                  **{k:sorted({x for n in old for x in n[k]}) for k in ['node_ids','edge_ids','pin_candidate_ids','reference_pin_references','unresolved_junction_ids']},
                  'logical_label_names':sorted({x for n in old for x in n.get('logical_label_names',[])}),
                  'logical_label_sites':[s for n in old for s in n.get('logical_label_sites',[])],
                  'power_names':sorted({s['label'] for s in sites if s['input_network_id'] in group}),
                  'power_site_ids':[s['site_id'] for s in sites if s['input_network_id'] in group],
                  'power_logical_merge':True,'flywire':False,'status':'power_membership_proposal_not_certified'}
        result['networks'].append(merged)
    for attachment in result['pin_attachments']:
        attachment['network_id'] = mapping[attachment['network_id']] if attachment['network_id'] is not None else None
        for alternative in attachment['alternatives']:
            alternative['network_id'] = mapping[alternative['network_id']]
    for gap in result['gap_candidates']:
        gap['network_ids'] = [mapping[n] for n in gap['network_ids']]
    for site in sites:site['proposal_network_id']=mapping[site['input_network_id']]
    applied = [d for d in decisions if not d['withheld_reasons']]
    summary = {'observed_power_sites':len(sites),'power_groups_applied':len(applied),
               'input_network_groups_joined':sum(len(d['input_network_ids'])-1 for d in applied),
               'power_components_in_review':len(queue),'power_components_without_qualified_caption':sum(q['bound_site_id'] is None for q in queue),
               'network_proposals_before':len(graph['networks']),'network_proposals_after':len(result['networks']),
               'multi_pin_networks_before':sum(len(n['pin_candidate_ids'])>=2 for n in graph['networks']),
               'multi_pin_networks_after':sum(len(n['pin_candidate_ids'])>=2 for n in result['networks']),
               'decision_withheld_reason_counts':dict(Counter(r for d in decisions for r in d['withheld_reasons'])),
               'pin_coordinates_changed':0,'pin_fields_changed':0,'physical_edges_added':0,
               'power_ordinary_flywires_created':0,'default_pipeline_adopted':False}
    result.update(power_sites=sites,power_merge_decisions=decisions,input_to_power_network=mapping,power_logic_summary=summary)
    result['summary'].update(network_proposals=len(result['networks']),multi_pin_network_proposals=summary['multi_pin_networks_after'],
                             power_label_merges=len(applied))
    return result, {'summary':summary,'decisions':decisions,'component_review_queue':queue,'unrepresented_power_guard':guard,
                    'source_annotations_used':False,'reference_NetF1':None}
