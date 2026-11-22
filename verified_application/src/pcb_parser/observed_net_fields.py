"""Fill missing observed fields without changing model contacts or topology."""
from collections import defaultdict
from copy import deepcopy
import re

import numpy as np

from .component_text import center, gap, iou, choose_global
from .component_identity_recovery import observed_words, header_position, spatial_observations
from .engineering_network_export import rebuild_references
from .pin_text_v2 import contexts, tokens_with_views, glyph_groups, cost_v2
from .submission_fields import validate_fields


OBSERVED = {'observed_OCR_text', 'observed_targeted_local_OCR_missing_field_fill'}


def claimed_boxes(pins, records):
    boxes = [v['bbox'] for r in records for k, v in r.get('fields', {}).items()
             if k in {'designator', 'Name', 'value'} and v.get('bbox')]
    boxes += [v['bbox'] for p in pins['pin_candidates'] for k, v in p.get('text_evidence', {}).items()
              if k in {'number', 'pinname'} and p.get(k) is not None and v.get('bbox')]
    return boxes


def correct_body_half(token, ctx, guard):
    """The complete glyph box must be in the corresponding half of the body."""
    box, body = token['bbox'], ctx['component']['bbox_top_left']
    if not (body[0] <= box[0] < box[2] <= body[2] and body[1] <= box[1] < box[3] <= body[3]):return False
    axis = 0 if ctx['side'] in {'left', 'right'} else 1
    mid = (body[axis]+body[axis+2])/2
    return box[axis+2] <= mid-guard if ctx['side'] in {'left', 'top'} else box[axis] >= mid+guard


def fill_side_names(pins, texts, records, cfg, binding_cfg):
    result = deepcopy(pins);ctx = contexts(pins, binding_cfg);occupied = claimed_boxes(pins, records)
    tokens = [t for t in tokens_with_views(texts) if not t.get('approximate_split_geometry')
              and t['text'] == t['source_text'] and not any(iou(t['bbox'], b) >= .4 for b in occupied)]
    groups = glyph_groups(tokens)
    known = {r['detection_index'] for r in records if r.get('designator')}
    # Pins missing numbers still compete for a glyph, although they cannot
    # receive a new name here. This prevents moving evidence to a nearby row.
    active = [p for p in result['pin_candidates'] if p['candidate_id'] in ctx
              and p['owner_detection_index'] in known and p.get('pinname') is None
              and not p.get('text_hypothesis_primary_candidate_id')
              and ctx[p['candidate_id']]['component']['type'] in binding_cfg['name_component_types']]
    costs = np.full((len(active), len(groups)), 1e6);chosen = {};candidates = [];rejected = []
    for i, pin in enumerate(active):
        context = ctx[pin['candidate_id']]
        for j, group in enumerate(groups):
            ranked = sorted([(score, t) for t in group
                if correct_body_half(t, context, cfg['body_half_guard_px'])
                and (score := cost_v2(t, context, 'pinname', binding_cfg, [])) is not None], key=lambda r: r[0])
            if not ranked:continue
            votes = defaultdict(set)
            for _, t in ranked:
                if t['confidence'] >= cfg['name_min_confidence']:votes[t['text']].add(t['source_view'])
            supported = [name for name, views in votes.items() if len(views) >= cfg['name_min_views']]
            conflicting = {t['text'] for _, t in ranked if t['confidence'] >= cfg['conflict_min_confidence']
                and len({other['source_view'] for _, other in ranked if other['text'] == t['text']
                    and other['confidence'] >= cfg['conflict_min_confidence']}) >= cfg['conflict_min_views']}
            reasons = []
            if len(supported) != 1:reasons.append('insufficient_or_conflicting_repeated_complete_spelling')
            elif conflicting - {supported[0]}:reasons.append('competing_observed_spelling')
            candidates.append({'candidate_id': pin['candidate_id'], 'observations': [t for _, t in ranked], 'reasons': reasons})
            if reasons:continue
            name = supported[0];observations = [t for _, t in ranked if t['text'] == name]
            score, best = next((score, t) for score, t in ranked if t['text'] == name and t['confidence'] >= cfg['name_min_confidence'])
            costs[i, j] = score;chosen[i, j] = (best, observations, sorted(votes[name]))
    assigned = choose_global(costs, binding_cfg['unmatched_cost'], cfg['name_assignment_margin'], rejected, 'side_pinname')
    changes = []
    for i, (j, score, margin) in assigned.items():
        pin = active[i]
        if pin.get('number') is None or pin.get('number_status') not in OBSERVED:continue
        token, observations, views = chosen[i, j]
        evidence = {**token, 'source': 'repeated_complete_OCR_glyph_in_owned_body_half', 'cost': score,
            'global_assignment_margin': margin, 'observations': observations, 'distinct_views': views,
            'side': ctx[pin['candidate_id']]['side'], 'body_half_guard_px': cfg['body_half_guard_px'],
            'observed_OCR_glyph': True, 'reference_field_verified': False, 'views_are_independent': False}
        pin.update(pinname=token['text'], pinname_status='observed_body_side_OCR_missing_name_fill')
        pin.setdefault('text_evidence', {})['pinname'] = evidence
        changes.append({'candidate_id': pin['candidate_id'], 'owner': pin['owner'], 'number': pin['number'],
                        'pinname': pin['pinname'], 'evidence': evidence})
    return result, {'changes': changes, 'candidates': candidates, 'rejected_assignments': rejected}


def recover_literal_headers(pins, texts, records, cfg):
    """Accept a unique visible title even when its prefix differs from the class.

    A separate repeated footer and at least four observed pin numbers are
    required. The title is a local ID, never a canonical CAD ordering claim.
    """
    result = deepcopy(records);occupied = claimed_boxes(pins, records);owned = defaultdict(list)
    for p in pins['pin_candidates']:
        if p['owner_status'] == 'assigned':owned[p['owner_detection_index']].append(p)
    words = [w for w in observed_words(texts) if not w['approximate_split_geometry']
        and w['text'] == w['source_text'] and w['confidence'] >= cfg['header_min_confidence']
        and not any(iou(w['bbox'], b) >= .4 for b in occupied)]
    groups = defaultdict(list)
    for word in words:groups[word['text']].append(word)
    all_clusters = {text: spatial_observations(items) for text, items in groups.items()}
    clusters = {text: [g for g in options if len({w['view'] for w in g}) >= cfg['header_min_views']]
                for text, options in all_clusters.items()}
    existing = {r['component_id'] for r in records} | {r.get('designator') for r in records}
    titles = {text: groups for text, groups in clusters.items() if re.fullmatch(r'[A-Z]{1,3}[1-9]\d{0,3}[A-Z]?', text)
              and text not in existing and len(groups) == 1 and len(all_clusters[text]) == 1}
    active = [];reviews = [];selected = {};observed_fields = {}
    for r in result:
        if r.get('designator') or r['type'] not in {'ic', 'opamp', 'block'}:continue
        ps = owned[r['detection_index']]
        numbers = {p['number'] for p in ps if p.get('number') is not None and p.get('number_status') in OBSERVED
                   and p.get('text_evidence', {}).get('number', {}).get('text') == p['number']}
        names = {p['pinname'] for p in ps if p.get('pinname') is not None and p.get('pinname_status') in OBSERVED
                 and p.get('text_evidence', {}).get('pinname', {}).get('text') == p['pinname']}
        if (r.get('duplicate_of_detection_index') is not None or r.get('detection_confidence', 0) < cfg['minimum_detection_confidence']
            or len(numbers) < cfg['minimum_observed_numbers'] or len(names) < cfg['minimum_observed_names']):continue
        active.append(r);observed_fields[r['detection_index']] = (numbers, names)
    keys = sorted(titles);costs = np.full((len(active), len(keys)), 1e6)
    for i, r in enumerate(active):
        db = r['bbox_top_left'];ps = owned[r['detection_index']]
        numbers, names = observed_fields[r['detection_index']]
        def zone(w, expected):
            # A title/footer box must not extend into an observed pin row.
            border = min(p['point'][1] for p in ps)-cfg['header_geometry']['pin_header_guard_px'] if expected == 'top' else max(p['point'][1] for p in ps)+cfg['header_geometry']['pin_header_guard_px']
            return (header_position(w['bbox'], r, ps, cfg['header_geometry']) == expected
                and (w['bbox'][3] <= border if expected == 'top' else w['bbox'][1] >= border)
                and db[0]-cfg['header_left_padding_px'] <= center(w['bbox'])[0] <= db[0]+cfg['header_left_fraction']*(db[2]-db[0]))
        footers = []
        for text, options in clusters.items():
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9.+/\-]{2,}', text) or text in titles or text in names:continue
            for group in options:
                valid = [w for w in group if zone(w, 'bottom')]
                if len({w['view'] for w in valid}) >= cfg['header_min_views']:footers.append((text, valid))
        for j, title in enumerate(keys):
            valid = [w for w in titles[title][0] if zone(w, 'top')]
            if len({w['view'] for w in valid}) < cfg['header_min_views']:continue
            reasons = [] if len(footers) == 1 else ['footer_missing_or_ambiguous']
            reviews.append({'detection_index': r['detection_index'], 'title': title, 'title_observations': valid,
                'footer_candidates': footers, 'observed_numbers': sorted(numbers), 'observed_names': sorted(names), 'reasons': reasons})
            if reasons:continue
            best = min(valid, key=lambda w: gap(w['bbox'], db)+.04*abs(center(w['bbox'])[0]-db[0])+10*(1-w['confidence']))
            costs[i, j] = gap(best['bbox'], db)+.04*abs(center(best['bbox'])[0]-db[0])+10*(1-best['confidence'])
            selected[i, j] = (best, valid, footers[0], sorted(numbers), sorted(names))
    rejected = [];assignments = choose_global(costs, cfg['header_unmatched_cost'], cfg['header_assignment_margin'], rejected, 'literal_header')
    changes = []
    for i, (j, score, margin) in assignments.items():
        r = active[i];title = keys[j];best, readings, (footer, footer_readings), numbers, names = selected[i, j]
        old_id = r['component_id'];evidence = {**best, 'source': 'repeated_literal_top_title_with_separate_footer_and_observed_pins',
            'observations': readings, 'cost': score, 'global_assignment_margin': margin,
            'observed_numbers': numbers, 'observed_names': names, 'reference_identity_verified': False,
            'type_prefix_agreement_required': False, 'canonical_CAD_identity_inferred': False, 'views_are_independent': False}
        r.update(component_id=title, designator=title, identity_status='observed_literal_header_local_ID')
        r.setdefault('fields', {})['designator'] = evidence
        if r.get('Name') is None:
            r['Name'] = footer;r['fields']['Name'] = {**footer_readings[0], 'observations': footer_readings,
                'source': 'repeated_literal_footer', 'reference_field_verified': False}
        changes.append({'detection_index': r['detection_index'], 'old_component_id': old_id, 'component_id': title,
                        'footer': footer, 'evidence': evidence})
    return result, {'changes': changes, 'candidates': reviews, 'rejected_assignments': rejected}


def complete_observed_net_fields(pins, graph, texts, records, cfg, binding_cfg):
    validate_fields(pins, graph, records)
    if cfg['profile'] != 'engineering_observed_fields_only' or cfg['reference_identity_verified']:raise ValueError('Unsupported profile')
    updated, name_audit = fill_side_names(pins, texts, records, cfg, binding_cfg)
    updated_records, header_audit = recover_literal_headers(updated, texts, records, cfg)
    rmap = {r['detection_index']: r for r in updated_records}
    for c in updated['components']:
        r = rmap[c['detection_index']]
        if c['component_id'] != r['component_id']:
            c.update(component_id=r['component_id'], designator=r['designator'], identity_status=r['identity_status'])
    for p in updated['pin_candidates']:
        if p['owner_status'] == 'assigned':p['owner'] = rmap[p['owner_detection_index']]['component_id']
    g = rebuild_references(updated, graph, updated_records);validate_fields(updated, g, updated_records)
    for old, new in zip(pins['pin_candidates'], updated['pin_candidates']):
        if any(old.get(k) != new.get(k) for k in ['candidate_id', 'point', 'owner_detection_index', 'owner_status', 'number']):raise ValueError('Contact or number changed')
        if old.get('pinname') is not None and old['pinname'] != new['pinname']:raise ValueError('Existing name overwritten')
    for k in ['edges', 'nodes', 'physical_networks', 'gap_candidates']:
        if graph.get(k) != g.get(k):raise ValueError('Physical graph changed')
    if [(n['network_id'], n['pin_candidate_ids']) for n in graph['networks']] != [(n['network_id'], n['pin_candidate_ids']) for n in g['networks']]:raise ValueError('Raw net members changed')
    audit = {'side_names': name_audit, 'literal_headers': header_audit, 'summary': {
        'observed_pin_names_added': len(name_audit['changes']), 'observed_component_ids_added': len(header_audit['changes']),
        'observed_footer_names_added': sum(r.get('Name') is None and s.get('Name') is not None for r, s in zip(records, updated_records)),
        'new_pin_numbers': 0, 'points_or_edges_changed': 0, 'raw_members_removed': 0, 'new_OCR_model_runs': 0}}
    return updated, g, updated_records, audit
