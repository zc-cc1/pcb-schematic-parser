"""Image-predicted identities, explicit format rules and export blockers.

Source CAD component/pin ordering is not observable in a PNG. Repeated-label
suffixes are therefore review proposals, never accepted canonical identities.
Missing pin names remain missing even if the local schema accepts empty strings.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math


def validate_fields(pins, graph, records):
    by_index = {r['detection_index']: r for r in records}
    components = {c['detection_index']: c for c in pins['components']}
    if len(by_index) != len(records) or by_index.keys() != components.keys():
        raise ValueError('Component index contract mismatch')
    for index, record in by_index.items():
        component = components[index]
        if (record['component_id'] != component['component_id'] or record['type'] != component['type']
                or record['bbox_top_left'] != component['bbox_top_left']):
            raise ValueError('Component record/owner mismatch')
    ids = [r['component_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate component IDs')
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    attachments = {p['candidate_id']: p for p in graph['pin_attachments']}
    if len(pmap) != len(pins['pin_candidates']) or len(attachments) != len(graph['pin_attachments']) or pmap.keys() != attachments.keys():
        raise ValueError('Duplicate or missing pin IDs')
    width, height = pins['size']
    for cid, pin in pmap.items():
        point = pin['point']
        if (len(point) != 2 or not all(math.isfinite(v) for v in point)
                or not 0 <= point[0] <= width or not 0 <= point[1] <= height
                or point != attachments[cid]['point']):
            raise ValueError('Invalid/changed pin point')
        if pin['owner_status'] == 'assigned':
            owner = by_index.get(pin['owner_detection_index'])
            if owner is None or owner['component_id'] != pin['owner']:
                raise ValueError('Pin owner identity mismatch')
    net_ids = [n['network_id'] for n in graph['networks']]
    if len(set(net_ids)) != len(net_ids):
        raise ValueError('Duplicate network IDs')
    net_pins = set()
    for net in graph['networks']:
        members = net['pin_candidate_ids']
        if len(set(members)) != len(members) or any(cid not in pmap for cid in members):
            raise ValueError('Unknown/duplicate network pin')
        if net_pins.intersection(members):
            raise ValueError('Pin belongs to multiple networks')
        net_pins.update(members)


def complete_fields(pins, graph, records, cfg):
    """Return copies; never access reference labels or alter geometry/topology."""
    validate_fields(pins, graph, records)
    result, updated_graph, updated_records = deepcopy(pins), deepcopy(graph), deepcopy(records)
    by_index = {r['detection_index']: r for r in updated_records}
    components = {c['detection_index']: c for c in result['components']}
    owned = defaultdict(list)
    for pin in result['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            owned[pin['owner_detection_index']].append(pin)
    strong = [s for s in graph.get('label_sites', []) if s.get('status') == 'bound_ordinary_label'
              and s.get('evidence_kind') in {'detected_symbol_and_owned_pin', 'detected_two_tip_net_short_external_text'}
              and s.get('component_type') in cfg['label_component_types']]
    per_component = defaultdict(list)
    for site in strong:
        index = site.get('detection_index')
        if index not in by_index or by_index[index]['component_id'] != site['component_id']:
            raise ValueError('Label component mismatch')
        expected = {p['candidate_id'] for p in owned[index]}
        claimed = set(site.get('owned_candidate_ids') or [site.get('candidate_id')])
        if not claimed or None in claimed or not claimed.issubset(expected):
            raise ValueError('Label pin ownership mismatch')
        per_component[index].append(site)
    # Count even unaccepted strong sites: rejecting one duplicate must not turn
    # the other into an apparently unique unsuffixed canonical component key.
    labels = defaultdict(list)
    for index, sites in per_component.items():
        for label in sorted({s['label'] for s in sites}):
            labels[label].append(index)
    existing = {r['designator']: r['detection_index'] for r in updated_records if r.get('designator')}
    if len(existing) != sum(bool(r.get('designator')) for r in updated_records):
        raise ValueError('Duplicate recognized designator')
    proposals = []
    accepted = 0
    auto_numbers = 0
    for index, sites in sorted(per_component.items()):
        record = by_index[index]
        site = max(sites, key=lambda s: s['confidence'])
        raw = site['label']
        reasons = []
        if len({s['label'] for s in sites}) > 1:
            reasons.append('conflicting_label_spellings_on_component')
        if site['confidence'] < cfg['identity_min_confidence']:
            reasons.append('low_OCR_confidence')
        votes = site.get('distinct_view_votes', {}).get(raw, 0)
        if votes < cfg['identity_min_distinct_views']:
            reasons.append('insufficient_existing_OCR_views')
        if len(raw) < 2:
            reasons.append('single_character_label_role_unverified')
        if any(c in raw for c in '.,()'):
            reasons.append('label_not_safe_as_hypergraph_component_key')
        if record['type'] == 'net_short' and not cfg['allow_two_tip_label_identity']:
            reasons.append('two_tip_label_identity_semantics_pending')
        duplicates = sorted(set(labels[raw]), key=lambda i: (by_index[i]['bbox_top_left'][1], by_index[i]['bbox_top_left'][0], i))
        proposal_key = raw
        if len(duplicates) > 1:
            proposal_key = f'{raw}_{duplicates.index(index)+1}'
            reasons.append('duplicate_suffix_source_component_order_unobservable')
        # This is an ambiguity gate only. It never changes a spelling, creates
        # a shared key, or merges a network. Missing underscores must not make
        # a second physical label appear to be a unique canonical identity.
        skeleton = ''.join(c for c in raw.casefold() if c not in ' _')
        if any(other != raw and ''.join(c for c in other.casefold() if c not in ' _') == skeleton
               and any(j != index for j in indexes) for other,indexes in labels.items()):
            reasons.append('similar_label_spelling_may_hide_duplicate_identity')
        if raw in existing and existing[raw] != index:
            reasons.append('label_key_collides_with_recognized_designator')
        original_id = record['component_id']
        identity_accepted = not reasons and not record.get('designator')
        if identity_accepted:
            record.update(component_id=raw, designator=raw, Name=raw,
                          identity_status='observed_unique_label_name',
                          fields={**record.get('fields', {}), 'designator': {
                              'text': raw, 'bbox': site['bbox'], 'confidence': site['confidence'],
                              'source': 'ordinary_label_symbol_OCR', 'site_id': site['site_id'],
                              'source_component_order_available': False}})
            components[index].update(component_id=raw, designator=raw, identity_status='observed_unique_label_name')
            existing[raw] = index
            accepted += 1
        # Automatic numbering is a limited output convention stated in p18(c).
        # Only a model-owned single-tip label gets 1; IC/passive/two-tip ordering
        # is never inferred from coordinates or from missing OCR.
        number_changes = []
        if (record['type'] in cfg['single_pin_label_types'] and len(owned[index]) == 1
                and len({s['label'] for s in sites}) == 1):
            pin = owned[index][0]
            if pin.get('number') is None and not pin.get('text_candidates', {}).get('number'):
                pin['number'] = cfg['single_tip_auto_number']
                pin['number_status'] = 'PDF_single_tip_label_format_auto_number'
                pin.setdefault('text_evidence', {})['number'] = {
                    'source': 'PDF_0803Rev1_page18_c_single_tip_label_rule',
                    'observed_number_glyph': False, 'site_id': site['site_id'],
                    'reason': 'One assigned model tip on an ordinary single-tip label; no number text available',
                    'completeness_of_detector_and_pin_prediction_verified': False}
                auto_numbers += 1
                number_changes.append(pin['candidate_id'])
        proposals.append({'detection_index': index, 'original_component_id': original_id,
                          'site_id': site['site_id'], 'raw_label_name': raw,
                          'proposed_component_key': proposal_key,
                          'accepted_component_key': record['component_id'] if record.get('designator') else None,
                          'new_identity_accepted': identity_accepted, 'withheld_reasons': reasons,
                          'existing_OCR_view_count': votes, 'bbox_top_left': record['bbox_top_left'],
                          'auto_numbered_candidate_ids': number_changes,
                          'pinname_inferred_from_label': False,
                          'duplicate_order_rule': cfg['duplicate_suffix_policy'] if len(duplicates) > 1 else None})
    # Identity remapping is based on the unchanged detection index, including
    # alternate owner candidates. Graph membership uses unchanged candidate IDs.
    for pin in result['pin_candidates']:
        if pin['owner_status'] == 'assigned':
            pin['owner'] = by_index[pin['owner_detection_index']]['component_id']
        for candidate in pin.get('owner_candidates', []):
            candidate['component_id'] = by_index[candidate['detection_index']]['component_id']
    pmap = {p['candidate_id']: p for p in result['pin_candidates']}
    for pin in updated_graph['pin_attachments']:
        source = pmap[pin['candidate_id']]
        for field in ['owner', 'owner_status', 'number', 'pinname']:
            pin[field] = source.get(field)
    summary = {'new_label_identities': accepted, 'label_identity_proposals': len(proposals),
               'single_tip_label_auto_numbers': auto_numbers,
               'observed_number_glyphs_added': 0, 'pin_names_guessed': 0,
               'repeated_label_suffixes_accepted': 0, 'coordinate_changes': 0,
               'identity_withheld_reason_counts': dict(Counter(reason for p in proposals for reason in p['withheld_reasons']))}
    result['submission_field_summary'] = summary
    validate_fields(result, updated_graph, updated_records)
    if graph['edges'] != updated_graph['edges'] or graph['networks'] != updated_graph['networks']:
        raise ValueError('Field completion altered topology')
    return result, updated_graph, updated_records, {'summary': summary, 'components': proposals}


def network_field_queue(pins, graph, records, coverage):
    pmap = {p['candidate_id']: p for p in pins['pin_candidates']}
    cmap = {r['component_id']: r for r in records}
    pending = {p['candidate_id']: p['reasons'] for p in coverage['pending_pins']}
    gaps = {n for g in graph['gap_candidates'] for n in g['network_ids']}
    export_reasons = defaultdict(set)
    for row in coverage.get('withheld_networks', []):
        export_reasons[row['network_id']].update(row['reasons'])
    exported = set(coverage.get('exported_network_ids', []))
    rows = []
    for net in graph['networks']:
        if len(net['pin_candidate_ids']) < 2:
            continue
        members = []
        for cid in net['pin_candidate_ids']:
            pin = pmap[cid]
            component = cmap.get(pin['owner'])
            members.append({'candidate_id': cid, 'component_id': pin['owner'],
                            'owner_detection_index': pin.get('owner_detection_index'),
                            'component_type': component['type'] if component else None,
                            'component_bbox_top_left': component['bbox_top_left'] if component else None,
                            'point': pin['point'], 'number': pin.get('number'), 'pinname': pin.get('pinname'),
                            'number_status': pin.get('number_status'), 'pinname_status': pin.get('pinname_status'),
                            'missing_fields': pending.get(cid, [])})
        blockers = []
        if net['unresolved_junction_ids']:
            blockers.append('junction_semantics_unresolved')
        if net['network_id'] in gaps:
            blockers.append('suspected_disconnection')
        # Power proposals and omitted-member gates are real export blockers,
        # even when physical junction/gap checks happen to be clear.
        if net.get('power_logical_merge'):
            export_reasons[net['network_id']].add('power_logical_membership_provisional')
        reasons = sorted(export_reasons[net['network_id']])
        non_field_blockers = sorted(set(reasons+blockers)-{
            'contains_pin_with_unresolved_identity_number_or_name'})
        tasks = {}
        for pin in members:
            for field in pin['missing_fields']:
                if field == 'component_identity_unresolved' and pin['owner_detection_index'] is not None:
                    key = f"component:{pin['owner_detection_index']}:identity"
                else:
                    key = f"pin:{pin['candidate_id']}:{field}"
                task = tasks.setdefault(key, {'task_id': key, 'field': field,
                    'owner_detection_index': pin['owner_detection_index'], 'component_id': pin['component_id'],
                    'component_type': pin['component_type'], 'affected_candidate_ids': []})
                task['affected_candidate_ids'].append(pin['candidate_id'])
        counts = Counter(r for p in members for r in p['missing_fields'])
        rows.append({'network_id': net['network_id'], 'flywire': bool(net.get('flywire')),
                     'pin_count': len(members), 'pins_with_missing_fields': sum(bool(p['missing_fields']) for p in members),
                     'missing_field_counts': dict(counts), 'missing_field_total': sum(counts.values()),
                     'topology_blockers': blockers, 'pin_candidates': members,
                     'export_blockers': reasons, 'non_field_export_blockers': non_field_blockers,
                     'completion_tasks': list(tasks.values()), 'completion_task_count': len(tasks),
                     'field_only_candidate': bool(counts) and not non_field_blockers,
                     'exported': net['network_id'] in exported})
    rows.sort(key=lambda r: (bool(r['non_field_export_blockers']), r['completion_task_count'],
                            r['missing_field_total'], r['pin_count'], r['network_id']))
    return {'networks': rows, 'multi_pin_networks': len(rows),
            'topology_clear_but_field_incomplete': sum(not r['topology_blockers'] and r['missing_field_total'] > 0 for r in rows),
            'field_only_incomplete_networks': sum(r['field_only_candidate'] for r in rows),
            'non_field_export_blocker_counts': dict(Counter(b for r in rows for b in r['non_field_export_blockers'])),
            'purpose': 'Prioritize missing identity/number/pinname review; topology-clear is not certified correct',
            'official_NetF1': None}
