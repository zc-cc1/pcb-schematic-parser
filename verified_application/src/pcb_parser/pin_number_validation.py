"""Withhold package-like passive numbers without rewriting their OCR evidence.

This is a pilot role filter, not a correction to the observed string. It does
not assign replacements, infer ordering, or change physical network members.
"""
from collections import Counter
from copy import deepcopy

from .submission_fields import validate_fields
from .pin_output_rules import sync_fields


def withhold_package_numbers(pins, graph, records, cfg):
    validate_fields(pins, graph, records)
    if cfg.get('require_observed_OCR_number') is not True:
        raise ValueError('Package filter requires observed-number evidence')
    allowed_types = set(cfg['component_types'])
    literals = set(cfg['package_literals'])
    if (not allowed_types <= {'resistor', 'capacitor', 'inductor'}
            or any(not isinstance(s, str) or len(s) != 4 or not s.isascii() or not s.isdigit() for s in literals)):
        raise ValueError('Invalid package role filter')
    result = deepcopy(pins)
    by_index = {r['detection_index']: r for r in records}
    changes = []
    for pin in result['pin_candidates']:
        if pin['owner_status'] != 'assigned':
            continue
        record = by_index[pin['owner_detection_index']]
        number = pin.get('number')
        evidence = pin.get('text_evidence', {}).get('number', {})
        if (record['type'] not in allowed_types or number not in literals
                or pin.get('number_status') != 'observed_OCR_text' or evidence.get('text') != number):
            continue
        row = {'candidate_id': pin['candidate_id'], 'detection_index': pin['owner_detection_index'],
               'component_id': pin['owner'], 'component_type': record['type'], 'point': pin['point'],
               'original_number': number, 'original_number_status': pin['number_status'],
               'original_evidence': deepcopy(evidence), 'replacement_number': None,
               'reason': 'observed_literal_is_supported_passive_package_code',
               'role_filter_provisional': True, 'OCR_string_corrected': False}
        pin['number'] = None
        pin['number_status'] = 'package_like_OCR_number_withheld'
        pin.setdefault('text_evidence', {}).pop('number', None)
        # Do not retain a downstream name generated solely from this number.
        # A separately observed name is preserved.
        name_ev = pin.get('text_evidence', {}).get('pinname', {})
        if (pin.get('pinname') == number and name_ev.get('provisional')
                and pin.get('pinname_status') == 'candidate_PDF_example_number_as_passive_name'):
            row['dependent_candidate_pinname_withheld'] = pin['pinname']
            row['original_dependent_pinname_evidence'] = deepcopy(name_ev)
            pin['pinname'] = None
            pin['pinname_status'] = 'dependent_package_number_name_withheld'
            pin['text_evidence'].pop('pinname', None)
        pin.setdefault('withheld_number_observations', []).append(deepcopy(row))
        changes.append(row)
    updated_graph = sync_fields(result, graph)
    # Derived reference strings must not continue to advertise a withdrawn
    # number. Network membership and all physical geometry remain unchanged.
    withdrawn_refs = {f"{r['component_id']}.{r['original_number']}" for r in changes}
    reference_changes = []
    for net in updated_graph['networks']:
        if 'reference_pin_references' not in net:
            continue
        old_refs = net['reference_pin_references']
        new_refs = [ref for ref in old_refs if ref not in withdrawn_refs]
        if new_refs != old_refs:
            reference_changes.append({'network_id': net['network_id'], 'before': old_refs, 'after': new_refs})
            net['reference_pin_references'] = new_refs
    validate_fields(result, updated_graph, records)
    topology = lambda g: [{k: v for k, v in n.items() if k != 'reference_pin_references'} for n in g['networks']]
    if graph['edges'] != updated_graph['edges'] or topology(graph) != topology(updated_graph):
        raise ValueError('Package role filter changed physical topology')
    stable = lambda p: {k: p.get(k) for k in ['candidate_id', 'point', 'owner', 'owner_status', 'owner_detection_index']}
    if [stable(p) for p in pins['pin_candidates']] != [stable(p) for p in result['pin_candidates']]:
        raise ValueError('Package role filter changed localization or ownership')
    return result, updated_graph, {'changes': changes, 'derived_reference_changes': reference_changes, 'summary': {
        'package_like_numbers_withheld': len(changes),
        'withheld_by_literal': dict(Counter(r['original_number'] for r in changes)),
        'withheld_by_component_type': dict(Counter(r['component_type'] for r in changes)),
        'replacement_numbers_invented': 0, 'tip_coordinates_changed': 0,
        'network_members_changed': 0, 'default_pipeline_adopted': False}}
