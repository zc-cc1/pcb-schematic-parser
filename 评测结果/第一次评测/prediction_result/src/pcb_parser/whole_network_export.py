"""Observed interface identities and atomic, complete electrical-net export.

All model members remain in the graph/audit. One contact represented by two
tip hypotheses is not an electrical network joining two distinct contacts.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import re

from .component_identity_recovery import observed_words, spatial_observations
from .pin_output_rules import sync_fields
from .nearby_tip_fields import reconcile_nearby_tip_fields
from .submission_fields import validate_fields


def recover_interface_titles(pins, graph, records, texts, cfg):
    validate_fields(pins, graph, records)
    if cfg['header_identity_policy'] != 'unique_observed_title_with_matching_PIN_count_footer':
        raise ValueError('Unsupported interface identity policy')
    if cfg['header_min_views'] < 3 or cfg['footer_min_views'] < 2:
        raise ValueError('Repeated header and footer OCR required')
    words = [w for w in observed_words(texts) if w['text'] == w['source_text']
             and not w['approximate_split_geometry'] and w['confidence'] >= cfg['header_min_confidence']]
    existing = {r['designator'] for r in records if r.get('designator')}
    proposals = []
    for record in records:
        if (record.get('designator') or record['type'] not in cfg['interface_component_types']
                or record.get('duplicate_of_detection_index') is not None):
            continue
        index = record['detection_index'];box = record['bbox_top_left']
        owned = [p for p in pins['pin_candidates'] if p['owner_status'] == 'assigned' and p['owner_detection_index'] == index]
        numbers = {p['number'] for p in owned if p.get('number') and re.fullmatch(r'[1-9]\d*', p['number'])
                   and p.get('number_status') in {'observed_OCR_text', 'observed_targeted_local_OCR_missing_field_fill'}}
        if len(numbers) < cfg['minimum_observed_numbers']:
            continue
        groups = defaultdict(list)
        for word in words:
            b = word['bbox'];cx = (b[0]+b[2])/2;cy = (b[1]+b[3])/2
            if (re.fullmatch(r'[A-Z][A-Z_\-]{1,19}', word['text'])
                    and box[0]-cfg['horizontal_padding_px'] <= cx <= box[2]+cfg['horizontal_padding_px']
                    and box[1]-cfg['header_max_gap_px'] <= cy <= box[1]+cfg['header_max_inset_px']):
                groups[word['text']].append(word)
        for title, candidates in groups.items():
            for cluster in spatial_observations(candidates):
                views = sorted({w['view'] for w in cluster})
                if len(views) < cfg['header_min_views']:
                    continue
                footers = defaultdict(list)
                for w in words:
                    m = re.fullmatch(re.escape(title)+r'-(\d{1,3})PINS?', w['text'])
                    b = w['bbox'];cx = (b[0]+b[2])/2;cy = (b[1]+b[3])/2
                    if (m and box[0]-cfg['horizontal_padding_px'] <= cx <= box[2]+cfg['horizontal_padding_px']
                            and abs(cy-box[3]) <= cfg['footer_max_gap_px']):
                        footers[(w['text'], int(m[1]))].append(w)
                for (footer, count), readings in footers.items():
                    for footer_cluster in spatial_observations(readings):
                        footer_views = sorted({w['view'] for w in footer_cluster})
                        if len(footer_views) < cfg['footer_min_views']:
                            continue
                        reasons = []
                        if title in existing:reasons.append('identifier_already_used')
                        title_locations = [g for g in spatial_observations([w for w in words if w['text'] == title])
                                           if len({w['view'] for w in g}) >= cfg['header_min_views']]
                        if len(title_locations) > 1:reasons.append('repeated_title_glyph_source_order_unknown')
                        if count < 2 or any(int(n) > count for n in numbers):reasons.append('observed_numbers_conflict_with_footer')
                        proposals.append({'detection_index': index, 'old_component_id': record['component_id'],
                                          'observed_title': title, 'observed_footer': footer, 'footer_pin_count': count,
                                          'observed_numbers': sorted(numbers, key=int), 'title_views': views,
                                          'footer_views': footer_views, 'title_observations': cluster,
                                          'footer_observations': footer_cluster, 'withheld_reasons': reasons})
    # Multiple glyph locations or owners never receive invented suffixes.
    counts = Counter(p['observed_title'] for p in proposals)
    per_owner = Counter(p['detection_index'] for p in proposals)
    for proposal in proposals:
        if counts[proposal['observed_title']] > 1:proposal['withheld_reasons'].append('repeated_or_competing_interface_title')
        if per_owner[proposal['detection_index']] > 1:proposal['withheld_reasons'].append('multiple_interface_titles_on_component')
    updated, new_records = deepcopy(pins), deepcopy(records)
    changes = []
    for proposal in proposals:
        if proposal['withheld_reasons']:continue
        index, title = proposal['detection_index'], proposal['observed_title']
        record = next(r for r in new_records if r['detection_index'] == index)
        record.update(component_id=title, designator=title, identity_status='observed_unique_interface_title')
        record.setdefault('fields', {})['designator'] = {
            **max(proposal['title_observations'], key=lambda w: w['confidence']),
            'source': 'observed_unique_title_with_matching_PIN_footer', 'observations': proposal['title_observations'],
            'generated_designator_or_suffix': False, 'canonical_source_ID_known': False}
        if record.get('Name') is None:
            record['Name'] = proposal['observed_footer']
            record['fields']['Name'] = {**max(proposal['footer_observations'], key=lambda w: w['confidence']),
                                        'source': 'observed_interface_footer', 'observations': proposal['footer_observations']}
        component = next(c for c in updated['components'] if c['detection_index'] == index)
        component.update(component_id=title, designator=title, identity_status='observed_unique_interface_title')
        for pin in updated['pin_candidates']:
            if pin['owner_status'] == 'assigned' and pin['owner_detection_index'] == index:pin['owner'] = title
        changes.append(deepcopy(proposal))
    updated_graph = sync_fields(updated, graph)
    pmap = {p['candidate_id']: p for p in updated['pin_candidates']}
    rmap = {r['detection_index']: r for r in new_records}
    for net in updated_graph['networks']:
        # Derived references are rebuilt from every member after identity changes.
        net['official_pin_references'] = sorted({f"{p['owner']}.{p['number']}" for cid in net['pin_candidate_ids']
            for p in [pmap[cid]] if p['owner_status'] == 'assigned' and p.get('number') is not None
            and rmap[p['owner_detection_index']].get('designator')})
    validate_fields(updated, updated_graph, new_records)
    return updated, updated_graph, new_records, {'changes': changes,
        'withheld': [p for p in proposals if p['withheld_reasons']], 'new_identities': len(changes),
        'new_pin_numbers_or_names': 0, 'generated_designators_or_suffixes': 0}


def net_export_guard(pins, graph, records, cfg, nearby_cfg):
    """Geometry/provenance gates, with every original member retained in audit."""
    validate_fields(pins, graph, records)
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    aliases, proofs = {}, []
    work = deepcopy(pins)
    targets = {}
    for p in work['pin_candidates']:
        e = p.get('text_evidence', {}).get('pinname', {})
        link = e.get('field_link', {})
        if (e.get('source') == 'nearby_tip_same_physical_lead_reconciliation'
                and link.get('candidate_id') == p['candidate_id']
                and e.get('observed_at_candidate_id') == link.get('donor_candidate_id')):
            targets[p['candidate_id']] = link['donor_candidate_id']
            p['pinname'] = None;p['text_evidence'].pop('pinname')
    if targets:
        _, _, verified = reconcile_nearby_tip_fields(work, sync_fields(work, graph), records, nearby_cfg)
        for link in verified['changes']:
            target, donor = link['candidate_id'], link['donor_candidate_id']
            if targets.get(target) == donor and pmap[target]['pinname'] == pmap[donor]['pinname']:
                aliases[donor] = target;proofs.append(link)
    edges = {e['edge_id']: e for e in graph['edges']}
    node_points = {n['node_id']: n['point'] for n in graph.get('nodes', [])}
    blocks, nets = [], []
    for net in graph['networks']:
        members = net['pin_candidate_ids']
        if not members:continue
        contacts = sorted({aliases.get(cid, cid) for cid in members})
        reasons = []
        if len(members) >= 2 and len(contacts) < 2:reasons.append('single_distinct_contact_fragment')
        degree = Counter(node for eid in net['edge_ids'] for node in [edges[eid]['source'], edges[eid]['target']])
        # NetworkX edge source/target order need not match the stored polyline
        # order. The model graph node owns the endpoint coordinate.
        endpoints = {node: node_points.get(node, point) for eid in net['edge_ids'] for node, point in
                     [(edges[eid]['source'], edges[eid]['polyline'][0]), (edges[eid]['target'], edges[eid]['polyline'][-1])]
                     if degree[node] == 1}
        missing = []
        for node, point in endpoints.items():
            for record in records:
                if (record['type'] not in cfg['endpoint_component_types'] or record.get('duplicate_of_detection_index') is not None
                        or record.get('detection_confidence', 0) < cfg['endpoint_min_detection_confidence']):continue
                x1, y1, x2, y2 = record['bbox_top_left'];margin = cfg['endpoint_component_margin_px']
                if not x1-margin <= point[0] <= x2+margin or not y1-margin <= point[1] <= y2+margin:continue
                if any(pmap[cid]['owner_status'] == 'assigned' and pmap[cid]['owner_detection_index'] == record['detection_index']
                       and math.dist(pmap[cid]['point'], point) <= cfg['endpoint_owned_tip_radius_px'] for cid in members):continue
                missing.append({'node_id': node, 'point': point, 'detection_index': record['detection_index'], 'component_id': record['component_id']})
        if missing:reasons.append('wire_endpoint_near_component_without_member')
        if reasons:blocks.append({'network_id': net['network_id'], 'reasons': reasons})
        nets.append({'network_id': net['network_id'], 'original_member_ids': list(members),
                     'canonical_contact_ids': contacts,
                     'member_to_contact': {cid: aliases.get(cid, cid) for cid in members},
                     'unrepresented_component_endpoints': missing, 'withheld_reasons': reasons})
    return {'blocks': blocks, 'networks': nets, 'verified_alias_proofs': proofs,
            'original_members_removed': 0, 'wire_geometry_changed': False,
            'summary': {'verified_duplicate_tip_pairs': len(proofs), 'blocked_networks': len(blocks),
                        'blocker_counts': dict(Counter(reason for b in blocks for reason in b['reasons']))}}


def apply_net_export_guard(payload, coverage, guard, pins, graph):
    out, cov = deepcopy(payload), deepcopy(coverage)
    for block in guard['blocks']:
        nid = block['network_id'];out['nets'].pop(nid, None)
        row = next((r for r in cov['withheld_networks'] if r['network_id'] == nid), None)
        if row is None:row = {'network_id': nid, 'reasons': []};cov['withheld_networks'].append(row)
        row['reasons'] = sorted(set(row['reasons']+block['reasons']))
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    edge_map = {e['edge_id']: e for e in graph['edges']}
    audited_nets = {n['network_id']: n for n in guard['networks']}
    proofs = []
    for net in graph['networks']:
        nid = net['network_id'];audit = audited_nets.get(nid)
        if not audit or audit['withheld_reasons']:continue
        contact_map = audit['member_to_contact']
        if nid not in out['nets']:
            blocked = next((r for r in cov['withheld_networks'] if r['network_id'] == nid), None)
            # Only an unresolved redundant tip may be resolved by a proven
            # physical-contact alias. Other gates remain effective.
            if (not blocked or set(blocked['reasons']) != {'contains_pin_with_unresolved_identity_number_or_name'}
                    or net.get('flywire') or net.get('power_logical_merge')
                    or len(audit['canonical_contact_ids']) < 2):continue
            contacts = [pmap[cid] for cid in audit['canonical_contact_ids']]
            if any('pin_'+str(p.get('number')) not in out['pins'].get(p['owner'], {}) for p in contacts):continue
            refs = sorted({f"{p['owner']}.{p['number']}" for p in contacts})
            if len(refs) < 2:continue
            segments = {};height = pins['size'][1]
            for eid in net['edge_ids']:
                for a, b in zip(edge_map[eid]['polyline'], edge_map[eid]['polyline'][1:]):
                    segments[f'edge_{len(segments)+1}'] = [{'x': a[0], 'y': height-a[1]}, {'x': b[0], 'y': height-b[1]}]
            out['nets'][nid] = {'hyperGraph': '('+','.join(refs)+')', 'edges': segments}
            cov['withheld_networks'] = [r for r in cov['withheld_networks'] if r['network_id'] != nid]
        if nid not in out['nets']:continue
        refs = []
        for cid in net['pin_candidate_ids']:
            p = pmap[contact_map[cid]];key = 'pin_'+str(p.get('number'))
            if key not in out['pins'].get(p['owner'], {}):raise ValueError('Whole-net export omitted an unresolved member')
            refs.append(f"{p['owner']}.{p['number']}")
        declared = out['nets'][net['network_id']]['hyperGraph'][1:-1].split(',')
        if len(set(refs)) < 2 or set(refs) != set(declared):raise ValueError('Whole-net export is incomplete or a self-loop fragment')
        proofs.append({'network_id': net['network_id'], 'all_original_member_ids': list(net['pin_candidate_ids']),
                       'member_to_contact': dict(contact_map),
                       'all_pin_references': refs, 'every_member_resolves': True, 'distinct_contact_count': len(set(refs)),
                       'model_edge_ids': list(net['edge_ids']), 'graph_geometry_preserved': True})
    cov.update(exported_nets=len(out['nets']), exported_network_ids=sorted(out['nets']),
               network_reason_counts=dict(Counter(reason for n in cov['withheld_networks'] for reason in n['reasons'])),
               atomic_complete_network_proofs=proofs, whole_network_guard=guard['summary'],
               complete_competition_JSON=False, official_evaluator_verified=False)
    return out, cov
