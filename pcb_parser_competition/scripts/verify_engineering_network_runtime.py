"""Compare fresh PNG inference with the S22 model-cache replay, without labels."""
import argparse
import math
from pathlib import Path
import re

from run_engineering_networks import ROOT, read, save


def main():
    p = argparse.ArgumentParser();p.add_argument('--case', default='0036')
    p.add_argument('--fresh', type=Path, default=ROOT/'runs/s22_runtime/0036')
    p.add_argument('--output', type=Path, default=ROOT/'runs/s22_engineering_networks');a = p.parse_args()
    replay = a.output/f'predictions/{a.case}'
    old, new = read(replay/'summary.json'), read(a.fresh/'summary.json')
    if new['used_external_model_caches']:raise ValueError('Fresh verification must start from the PNG')
    for key in ['image_sha256', 'size', 'engineering_config_sha256', 'profile', 'engineering_fields',
                'baseline_export', 'engineering_export', 'new_complete_network_ids', 'completion', 'network_guard']:
        if old[key] != new[key]:raise ValueError('Fresh PNG summary changed: '+key)
    ps = [read(b/'engineering_pin_text_result.json')['pin_candidates'] for b in [replay, a.fresh]]
    if len(ps[0]) != len(ps[1]):raise ValueError('Fresh PNG tip count changed')
    maximum = 0
    for before, after in zip(*ps):
        for key in ['candidate_id', 'owner', 'owner_status', 'owner_detection_index', 'number', 'pinname']:
            if before.get(key) != after.get(key):raise ValueError('Fresh PNG fields changed: '+key)
        maximum = max(maximum, math.dist(before['point'], after['point']))
    if maximum > .05:raise ValueError('Fresh model numeric variation exceeds 0.05px')
    outputs = [read(b/'engineering_submission.json') for b in [replay, a.fresh]]
    for key in ['components', 'nets']:
        if outputs[0][key] != outputs[1][key]:raise ValueError('Fresh PNG complete net changed: '+key)
    if outputs[0]['pins'].keys() != outputs[1]['pins'].keys():raise ValueError('Fresh exported owners differ')
    for owner, pins in outputs[0]['pins'].items():
        other = outputs[1]['pins'][owner]
        if pins.keys() != other.keys():raise ValueError('Fresh exported pin numbers differ')
        for key, pin in pins.items():
            if pin['pinname'] != other[key]['pinname']:raise ValueError('Fresh exported pin name differs')
            if math.dist(list(pin['point'].values()), list(other[key]['point'].values())) > .05:raise ValueError('Fresh exported point differs')
    covs = [read(b/'engineering_export_coverage.json') for b in [replay, a.fresh]]
    if covs[0]['atomic_complete_network_proofs'] != covs[1]['atomic_complete_network_proofs']:raise ValueError('Fresh raw member proof differs')
    log = (ROOT/'runs/s22_regression.log').read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests', log)
    if not match or not log.rstrip().endswith('OK'):raise ValueError('Successful regression log missing')
    save(a.output/'runtime_verification.json', {'case_id': a.case, 'fresh_entrypoint_wall_seconds': new['entrypoint_wall_seconds'],
        'fresh_tip_max_delta_px': maximum, 'model_numeric_tolerance_px': .05, 'evaluation_radius_px_unchanged': 5,
        'fresh_fields_components_complete_nets_edges_and_raw_member_proofs_match': True,
        'used_external_model_caches_for_fresh_run': new['used_external_model_caches'],
        'regression_tests_passed': int(match[1]), 'official_numbering_verified': False, 'official_NetF1': None})
    print({'runtime_seconds': new['entrypoint_wall_seconds'], 'max_tip_delta_px': maximum, 'regression_tests': int(match[1])})


if __name__ == '__main__':main()
