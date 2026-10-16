import argparse
from copy import deepcopy
import math
from pathlib import Path
import re

from run_observed_net_fields import ROOT, read, save


def geometry_bijection(before, after, tolerance=.05):
    """Match run-local confidence-ordered IDs using unique model positions.

    This is reproducibility checking only. Neither run's files are rewritten.
    """
    if len(before) != len(after):raise ValueError('Tip count differs')
    matches = []
    for p in before:
        candidates = [q for q in after if math.dist(p['point'], q['point']) <= tolerance]
        if len(candidates) != 1:raise ValueError('Model contact lacks unique numerical match')
        matches.append(candidates[0])
    if len({p['candidate_id'] for p in matches}) != len(after):raise ValueError('Contact match is not a bijection')
    return matches


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s24_observed_net_fields')
    p.add_argument('--fresh', type=Path, default=ROOT/'runs/s24_runtime');p.add_argument('--cases', nargs='+', default=['0036', '0018']);a = p.parse_args()
    results = []
    for case in a.cases:
        replay = a.output/f'predictions/{case}';fresh = a.fresh/case
        left, right = read(replay/'summary.json'), read(fresh/'summary.json')
        if right['used_external_model_caches']:raise ValueError('Verification must start from PNG')
        for key in ['image_sha256', 'size', 'observed_net_fields_config_sha256', 'observed_binding_config_sha256', 'observed_fields',
            'engineering_export', 'baseline_export', 'previous_engineering_export', 'new_complete_network_ids', 'completion']:
            if left[key] != right[key]:raise ValueError('Fresh summary differs: '+key)
        p1, p2 = [read(f/'export/engineering_pin_text_result.json')['pin_candidates'] for f in [replay, fresh]]
        p2 = geometry_bijection(p1, p2)
        cid_map = {q['candidate_id']: p['candidate_id'] for p, q in zip(p1, p2)}
        reordered = [{'replay_candidate_id': p['candidate_id'], 'fresh_candidate_id': q['candidate_id']}
                     for p, q in zip(p1, p2) if p['candidate_id'] != q['candidate_id']]
        maximum = 0
        for old, new in zip(p1, p2):
            for key in ['owner', 'owner_detection_index', 'owner_status', 'number', 'pinname', 'number_status', 'pinname_status']:
                if old.get(key) != new.get(key):raise ValueError('Fresh pin field differs: '+key)
            maximum = max(maximum, math.dist(old['point'], new['point']))
        if maximum > .05:raise ValueError('Fresh model numerical variation exceeds 0.05px')
        x, y = [read(f/'export/engineering_submission.json') for f in [replay, fresh]]
        if x['components'] != y['components'] or x['nets'] != y['nets']:raise ValueError('Whole nets/paths differ')
        left_graph, right_graph = [read(f/'export/engineering_graph_proposals.json') for f in [replay, fresh]]
        model_variations = []
        for key in ['physical_networks', 'networks']:
            for old, new in zip(left_graph[key], right_graph[key]):
                if (old['network_id'] != new['network_id'] or old['edge_ids'] != new['edge_ids']
                    or set(old['pin_candidate_ids']) != {cid_map[c] for c in new['pin_candidate_ids']}):
                    if key == 'networks' and (old['network_id'] in x['nets'] or new['network_id'] in y['nets']):raise ValueError('Exported model topology differs')
                    model_variations.append({'kind': key, 'replay': old, 'fresh': new, 'exported': False})
            if len(left_graph[key]) != len(right_graph[key]):raise ValueError('Network count differs')
        proofs = read(replay/'export/engineering_export_coverage.json')['atomic_complete_network_proofs']
        exported_edges = {eid for proof in proofs for eid in proof['model_edge_ids']}
        if len(left_graph['edges']) != len(right_graph['edges']):raise ValueError('Model edge count differs')
        for old, new in zip(left_graph['edges'], right_graph['edges']):
            if old['edge_id'] != new['edge_id'] or old['polyline'] != new['polyline']:
                if old['edge_id'] in exported_edges or new['edge_id'] in exported_edges:raise ValueError('Exported model path differs')
                model_variations.append({'kind': 'withheld_model_edge', 'replay_edge_id': old['edge_id'], 'fresh_edge_id': new['edge_id'],
                    'replay_polyline': old['polyline'], 'fresh_polyline': new['polyline'], 'exported': False})
        for key in ['atomic_complete_network_proofs', 'exported_network_ids']:
            if read(replay/'export/engineering_export_coverage.json')[key] != read(fresh/'export/engineering_export_coverage.json')[key]:raise ValueError('Raw member proof differs')
        # Compare actual new observed spellings and their glyph boxes/support.
        u, v = [read(f/'observed_fields_audit.json') for f in [replay, fresh]]
        evidence_delta = 0
        for key in ['side_names', 'literal_headers']:
            old_changes, new_changes = deepcopy(u[key]['changes']), deepcopy(v[key]['changes'])
            if len(old_changes) != len(new_changes):raise ValueError('Fresh observed field count differs')
            for old, new in zip(old_changes, new_changes):
                for field in ['cost', 'global_assignment_margin']:
                    evidence_delta = max(evidence_delta, abs(old['evidence'].pop(field)-new['evidence'].pop(field)))
            if old_changes != new_changes:raise ValueError('Fresh observed spelling/box/support differs')
        if evidence_delta > .05:raise ValueError('Observed assignment score variation exceeds numerical tolerance')
        results.append({'case_id': case, 'fresh_entrypoint_wall_seconds': right['entrypoint_wall_seconds'],
            'fresh_all_tip_max_delta_px': maximum, 'new_fields_member_proofs_and_exported_wire_paths_match': True,
            'assignment_score_max_numeric_delta': evidence_delta,
            'run_local_candidate_ID_reordering': reordered, 'contact_comparison': 'unique_bijection_within_0.05px_no_output_rewriting',
            'all_raw_model_paths_and_members_identical': not model_variations,
            'withheld_model_variations': model_variations,
            'fresh_run_used_external_model_caches': False})
    log = (ROOT/'runs/s24_regression.log').read_text(encoding='utf-8');m = re.search(r'Ran (\d+) tests', log)
    if not m or not log.rstrip().endswith('OK'):raise ValueError('Passing regression log required')
    result = {'cases': results, 'regression_tests_passed': int(m[1]), 'numeric_reproduction_tolerance_px': .05,
        'assignment_score_numerical_tolerance': .05,
        'evaluation_radius_px_unchanged': 5, 'official_PinF1': None, 'official_NetF1': None}
    save(a.output/'runtime_verification.json', result);print(result)


if __name__ == '__main__':main()
