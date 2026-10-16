"""Exact observed part/package and observed row text constrain physical numbers.

No symbol ordering, CAD/source annotations, or canonical name generation.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import re

import numpy as np

from .component_text import choose_global, iou
from .pin_text import contexts
from .pin_text_v2 import tokens_with_views, glyph_groups, tip_hypotheses, cost_v2


def validate_library(library):
    if library.get('schema_version') != 1:
        raise ValueError('Unknown library schema')
    models = set()
    for part in library['parts']:
        if part['model'] in models or not part.get('package'):
            raise ValueError('Duplicate model or missing package')
        models.add(part['model'])
        pins = part['pins']
        if {p['number'] for p in pins} != {str(i) for i in range(1, part['pin_count']+1)} or len(pins) != part['pin_count']:
            raise ValueError('Library pin numbers must be complete and unique')
        aliases = [a for p in pins for a in p['observed_primary_tokens']]
        if len(aliases) != len(set(aliases)):
            raise ValueError('Primary token cannot refer to multiple physical pins')
        if not part['source'].get('url', '').startswith('https://www.st.com/') or len(part['source'].get('sha256', '')) != 64:
            raise ValueError('Public source provenance missing')


def number_for_name(text, part):
    """Match a documented primary GPIO token, not an approximate OCR spelling."""
    if not isinstance(text, str):
        return None
    if re.fullmatch(r'P[A-D]\d{1,2}(?:[/_][A-Za-z0-9_+./~-]+)?', text):
        primary = re.match(r'P[A-D]\d{1,2}', text)[0]
    else:
        primary = text
    return next((p['number'] for p in part['pins'] if primary in p['observed_primary_tokens']), None)


def supported_model(record, part, cfg):
    if record.get('Name') != part['model'] or record.get('type') != 'ic' or record.get('duplicate_of_detection_index') is not None:
        return False
    evidence = record.get('fields', {}).get('Name', {})
    if evidence.get('text') != part['model'] or evidence.get('confidence', 0) < cfg['model_min_confidence']:
        return False
    views = {o.get('view') for o in evidence.get('observations', [])
             if o.get('text') == part['model'] and o.get('confidence', 0) >= cfg['model_min_confidence']
             and o.get('view') and iou(o['bbox'], evidence['bbox']) >= .4}
    return len(views) >= cfg['model_min_views']


def bind_library_pins(original, texts, records, library, cfg, text_cfg):
    validate_library(library)
    result = deepcopy(original)
    working, secondary = tip_hypotheses(original, text_cfg)
    context = contexts(working, text_cfg)
    groups = glyph_groups(tokens_with_views(texts))
    targets = []
    for r in records:
        part = next((p for p in library['parts'] if supported_model(r, p, cfg)), None)
        if part:
            targets.append((r, part))
    changes, withheld = [], []
    components = []
    def change(pin, field, value, status, evidence):
        before = pin.get(field)
        if before == value:
            return
        changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'field': field,
                        'before': before, 'after': value, 'reason': status, 'evidence': evidence})
        pin[field] = value
        pin[field+'_status'] = status
        if value is None:
            pin.setdefault('text_evidence', {}).pop(field, None)
        else:
            pin.setdefault('text_evidence', {})[field] = evidence

    for record, part in targets:
        detection = record['detection_index']
        pins = [p for p in result['pin_candidates'] if p.get('owner_detection_index') == detection and p['owner_status'] == 'assigned']
        legal = {p['number'] for p in part['pins']}
        for pin in pins:
            if pin.get('number') is not None and pin['number'] not in legal:
                change(pin, 'number', None, 'invalid_number_for_exact_package_withheld',
                       {'source': 'public_component_library_constraint', 'model': part['model'], 'package': part['package'],
                        'old_evidence': deepcopy(pin.get('text_evidence', {}).get('number'))})
        active = [p for p in pins if p['candidate_id'] in context and p['candidate_id'] not in secondary]
        costs = np.full((len(active), part['pin_count']), 1e6)
        choices = {}
        occupied = defaultdict(set)
        for pin in pins:
            if pin.get('number') in legal:
                occupied[pin['number']].add(pin['candidate_id'])
        blocked = [r['fields'][f]['bbox'] for r in records for f in ['Name', 'designator', 'value'] if f in r.get('fields', {})]
        for i, pin in enumerate(active):
            ctx = context[pin['candidate_id']]
            box = ctx['component']['bbox_top_left']
            options = []
            for group in groups:
                spellings = defaultdict(list)
                for token in group:
                    b = token['bbox']
                    if (token['confidence'] < cfg['name_min_confidence'] or token.get('approximate_split_geometry')
                        or token['source_text'] != token['text'] or not box[0] <= b[0] <= b[2] <= box[2]
                        or not box[1] <= b[1] <= b[3] <= box[3]):
                        continue
                    number = number_for_name(token['text'], part)
                    score = cost_v2(token, ctx, 'pinname', text_cfg, blocked)
                    if number is not None and score is not None:
                        spellings[token['text']].append((score, token, number))
                votes = sorted([(len({t['source_view'] for _, t, _ in rows}), text) for text, rows in spellings.items()], reverse=True)
                if not votes or votes[0][0] < cfg['name_min_views']:
                    continue
                if len(votes) > 1 and votes[0][0]-votes[1][0] < cfg['spelling_vote_lead']:
                    withheld.append({'candidate_id': pin['candidate_id'], 'reason': 'conflicting_observed_full_spellings', 'votes': votes})
                    continue
                rows = spellings[votes[0][1]]
                score, token, number = min(rows, key=lambda r: r[0])
                if pin.get('number') is not None and pin['number'] != number:
                    withheld.append({'candidate_id': pin['candidate_id'], 'reason': 'existing_number_disagrees_with_name',
                                     'number': pin['number'], 'observed_name': token['text'], 'library_number': number})
                    continue
                if occupied[number]-{pin['candidate_id']}:
                    withheld.append({'candidate_id': pin['candidate_id'], 'reason': 'physical_number_already_claimed', 'number': number})
                    continue
                old_name = pin.get('pinname')
                if old_name is not None and old_name != token['text'] and old_name not in token['text']:
                    withheld.append({'candidate_id': pin['candidate_id'], 'reason': 'existing_literal_name_disagrees',
                                     'existing_name': old_name, 'observed_name': token['text']})
                    continue
                options.append((score, token, number, rows))
            for score, token, number, rows in options:
                j = int(number)-1
                if score < costs[i,j]:
                    costs[i,j] = score
                    choices[i,j] = (token, rows)
        rejected = []
        assignments = choose_global(costs, cfg['unmatched_cost'], cfg['assignment_margin'], rejected, 'library_pin')
        withheld.extend(rejected)
        for i, (j, score, margin) in assignments.items():
            pin = active[i]
            token, rows = choices[i,j]
            evidence = {**token, 'cost': score, 'global_assignment_margin': margin,
                        'observed_support_views': sorted({t['source_view'] for _,t,_ in rows}),
                        'alternatives': [t for _,t,_ in rows], 'source': 'observed_OCR_exact_package_constraint',
                        'model': part['model'], 'package': part['package'], 'source_document': part['source']}
            change(pin, 'pinname', token['text'], 'observed_full_name_library_validated', evidence)
            if pin.get('number') is None:
                change(pin, 'number', str(j+1), 'public_library_from_observed_pinname',
                       {'source': 'public_datasheet_pin_mapping', 'model': part['model'], 'package': part['package'],
                        'bbox': token['bbox'], 'bbox_role': 'observed_pinname_glyph_not_number_glyph',
                        'observed_pinname': token['text'], 'name_evidence': evidence,
                        'source_document': part['source'], 'number_observed_in_image': False})
            pin['library_validation'] = {'model': part['model'], 'package': part['package'], 'number': str(j+1),
                                         'observed_name': token['text'], 'status': 'observed_name_and_number_agree'}
        components.append({'component_id': record['component_id'], 'detection_index': detection,
                           'model': part['model'], 'package': part['package'], 'candidates': len(pins),
                           'validated_name_number_pairs': len(assignments)})
    summary = {'supported_components': len(targets), 'numbers_added_from_library': sum(c['field']=='number' and c['after'] is not None for c in changes),
               'invalid_numbers_withheld': sum(c['field']=='number' and c['after'] is None for c in changes),
               'missing_names_added': sum(c['field']=='pinname' and c['before'] is None for c in changes),
               'partial_names_completed': sum(c['field']=='pinname' and c['before'] is not None for c in changes),
               'number_observed_OCR_count': sum(p.get('number') is not None and p.get('number_status') != 'public_library_from_observed_pinname' for p in result['pin_candidates']),
               'number_available_count': sum(p.get('number') is not None for p in result['pin_candidates']),
               'name_available_count': sum(p.get('pinname') is not None for p in result['pin_candidates']),
               'number_and_name_available_count': sum(p.get('number') is not None and p.get('pinname') is not None for p in result['pin_candidates']),
               'coordinate_changes': 0, 'ownership_changes': 0, 'trained': False, 'default_pipeline_adopted': False}
    result['library_pin_summary'] = summary
    result['pin_text_summary'].update(number_observed=summary['number_observed_OCR_count'],
        number_available=summary['number_available_count'], name_observed=summary['name_available_count'],
        numbers_derived_from_public_library=summary['numbers_added_from_library'],
        number_status_counts=dict(Counter(p['number_status'] for p in result['pin_candidates'])),
        name_status_counts=dict(Counter(p['pinname_status'] for p in result['pin_candidates'])),
        number_and_name_observed=sum(p.get('number') is not None and p.get('pinname') is not None and p.get('number_status') != 'public_library_from_observed_pinname' for p in result['pin_candidates']),
        number_and_name_available=summary['number_and_name_available_count'])
    return result, {'summary': summary, 'components': components, 'changes': changes, 'withheld': withheld,
                    'secondary_tip_hypotheses': secondary, 'original_reference_annotations_used': False}
