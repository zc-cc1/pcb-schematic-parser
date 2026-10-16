"""Offline explanations of draft matches; never imported by PNG inference."""
from math import dist

from evaluate_pin_text import match

MATCH_RADIUS_PX = 5


def expected_field(reference, field):
    if field not in {'number', 'pinname'}:
        raise ValueError('Only number and pinname can be audited')
    return reference['expected_numbers'] if field == 'number' else reference['expected_name']


def field_agrees(value, reference, field):
    expected = expected_field(reference, field)
    return value in expected if field == 'number' else value == expected


def diagnose_fields(predicted, truth, owner_map):
    """Keep the existing 5px, same-owner, one-to-one matching contract.

    Nearby or same-text references are evidence for review, never extra matches.
    A prediction with a mapped owner but no reference pins is distinguished from
    an unmapped owner. A reference used by another prediction is recorded rather
    than counting both predictions as matched.
    """
    ids = [p['candidate_id'] for p in predicted]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs')
    assigned = [i for i, p in enumerate(predicted) if p.get('owner_status') == 'assigned']
    pairs = [(assigned[i], j) for i, j in match([predicted[i] for i in assigned], truth, owner_map)]
    mapped = dict(pairs)
    reserved = {j: predicted[i]['candidate_id'] for i, j in pairs}
    rows = []
    for i, pin in enumerate(predicted):
        owner = owner_map.get(pin.get('owner_detection_index'))
        same_owner = [(j, r) for j, r in enumerate(truth)
                      if owner is not None and r['component_source_id'] == owner]
        nearest = sorted(same_owner, key=lambda item: (dist(pin['point'], item[1]['point']), item[0]))
        eligible = [(j, r) for j, r in nearest if dist(pin['point'], r['point']) <= MATCH_RADIUS_PX]
        for field in ['number', 'pinname']:
            value = pin.get(field)
            if value is None:
                continue

            def describe(j, reference):
                return {'reference_index': j, 'pin_source_id': reference.get('pin_source_id'),
                        'component_source_id': reference['component_source_id'],
                        'point': list(reference['point']), 'distance_px': dist(pin['point'], reference['point']),
                        'expected_numbers': list(reference['expected_numbers']),
                        'expected_name': reference['expected_name'],
                        'reserved_by_candidate_id': reserved.get(j)}

            matched = describe(mapped[i], truth[mapped[i]]) if i in mapped else None
            if matched is not None:
                reference = truth[mapped[i]]
                reason = ('reference_field_missing' if not expected_field(reference, field) else
                          'agrees_with_draft' if field_agrees(value, reference, field) else
                          'field_disagrees_with_draft')
            elif pin.get('owner_status') != 'assigned':
                reason = 'prediction_owner_unresolved'
            elif owner is None:
                reason = 'detected_owner_not_matched_to_draft'
            elif not same_owner:
                reason = 'mapped_owner_has_no_reference_pins'
            elif eligible:
                reason = 'one_to_one_reference_competition'
            else:
                reason = 'same_owner_point_outside_5px'
            same_text = next(((j, r) for j, r in nearest if field_agrees(value, r, field)), None)
            rows.append({'candidate_id': pin['candidate_id'], 'owner': pin.get('owner'),
                         'owner_status': pin.get('owner_status'), 'mapped_component_source_id': owner,
                         'point': list(pin['point']), 'field': field, 'text': value, 'reason': reason,
                         'field_status': pin.get(field + '_status'),
                         'matched_reference': matched,
                         'nearest_same_owner_references': [describe(j, r) for j, r in nearest[:2]],
                         'eligible_references': [describe(j, r) for j, r in eligible],
                         'nearest_same_text_reference': describe(*same_text) if same_text else None,
                         'matching_radius_px': MATCH_RADIUS_PX,
                         'additional_match_count': 0, 'reference_change_applied': False})
    return rows
