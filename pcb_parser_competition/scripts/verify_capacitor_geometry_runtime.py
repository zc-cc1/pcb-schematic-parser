"""Fresh PNG versus frozen-cache reproducibility of S23 geometry and whole nets."""
import argparse
import math
from pathlib import Path
import re

from run_capacitor_tip_geometry import ROOT, read, save


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s23_capacitor_geometry')
    p.add_argument('--fresh', type=Path, default=ROOT/'runs/s23_runtime/0036');p.add_argument('--case', default='0036');a = p.parse_args()
    replay = a.output/f'predictions/{a.case}';old, fresh = read(replay/'summary.json'), read(a.fresh/'summary.json')
    if fresh['used_external_model_stage'] or fresh['used_external_probability_caches']:raise ValueError('Verification must start from PNG')
    for key in ['image_sha256', 'size', 'capacitor_geometry_config_sha256', 'geometry', 'engineering_export', 'baseline_export', 'completion', 'network_guard']:
        if old[key] != fresh[key]:raise ValueError('Fresh runtime summary changed: '+key)
    before = read(replay/'export/engineering_pin_text_result.json')['pin_candidates'];after = read(a.fresh/'export/engineering_pin_text_result.json')['pin_candidates']
    if len(before) != len(after):raise ValueError('Fresh tip count changed')
    maximum = refined_maximum = 0
    for p, q in zip(before, after):
        for key in ['candidate_id', 'owner', 'owner_detection_index', 'owner_status', 'number', 'pinname', 'geometry_source']:
            if p.get(key) != q.get(key):raise ValueError('Fresh semantic fields or routing changed')
        distance = math.dist(p['point'], q['point']);maximum = max(maximum, distance)
        if p.get('geometry_source'):refined_maximum = max(refined_maximum, distance)
    if maximum > .05:raise ValueError('Fresh model numeric variation exceeds 0.05px')
    for filename in ['engineering_submission.json', 'engineering_export_coverage.json']:
        left, right = read(replay/'export'/filename), read(a.fresh/'export'/filename)
        keys = ['components', 'nets'] if filename == 'engineering_submission.json' else ['atomic_complete_network_proofs', 'exported_network_ids']
        for key in keys:
            if left[key] != right[key]:raise ValueError('Fresh whole net differs: '+key)
    changes = [read(b/'geometry_audit.json')['changes'] for b in [replay, a.fresh]]
    for x, y in zip(*changes):
        for key in ['candidate_id', 'component_id', 'node_id', 'edge_id', 'physical_network_id', 'supported_wire_path']:
            if x[key] != y[key]:raise ValueError('Fresh geometry support differs: '+key)
    log = (ROOT/'runs/s23_regression.log').read_text(encoding='utf-8');match = re.search(r'Ran (\d+) tests', log)
    if not match or not log.rstrip().endswith('OK'):raise ValueError('Passing regression log required')
    result = {'case_id': a.case, 'fresh_entrypoint_wall_seconds': fresh['entrypoint_wall_seconds'],
        'fresh_all_tip_max_delta_px': maximum, 'fresh_refined_tip_max_delta_px': refined_maximum,
        'numeric_reproduction_tolerance_px': .05, 'evaluation_radius_px_unchanged': 5,
        'fresh_fields_members_model_paths_and_exported_wire_geometry_match': True,
        'fresh_run_used_external_model_caches': False, 'regression_tests_passed': int(match[1]),
        'official_PinF1': None, 'official_NetF1': None}
    save(a.output/'runtime_verification.json', result);print(result)


if __name__ == '__main__':main()
