"""Compare S21 raw-PNG whole-network output with frozen replay."""
import argparse
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from run_whole_networks import ROOT, read, save, sha


def main():
    p = argparse.ArgumentParser();p.add_argument('--case-id', default='0042');p.add_argument('--fresh-output', type=Path, required=True)
    p.add_argument('--replay-root', type=Path, default=ROOT/'runs/s21_whole_networks');a = p.parse_args()
    folder = a.replay_root/f'predictions/{a.case_id}'
    fresh, replay = read(a.fresh_output/'summary.json'), read(folder/'summary.json')
    if fresh.get('used_external_model_caches') is not False:raise ValueError('Expected fresh PNG without supplied model caches')
    for key in ['image_sha256', 'size', 'whole_network_config_sha256', 'identity_recovery', 'network_guard', 'baseline_export', 'proposal_export', 'completion']:
        if fresh[key] != replay[key]:raise ValueError('Fresh/replay differs: '+key)
    before, after = [read(path/'completed_pin_text_result.json')['pin_candidates'] for path in [folder, a.fresh_output]]
    if len(before) != len(after):raise ValueError('Pin count differs')
    distance = np.linalg.norm(np.array([p['point'] for p in before])[:, None]-np.array([p['point'] for p in after])[None, :], axis=2)
    rows, cols = linear_sum_assignment(distance);delta = float(distance[rows, cols].max()) if len(rows) else 0
    # Full-page CPU/GPU quadratic-peak decoding can differ slightly on an
    # unused/unassigned hypothesis. Whole-net member tips retain the 0.05px
    # check below; all hypotheses use a separate subpixel 0.1px bound.
    if delta > .1:raise ValueError('Whole-page tip numeric tolerance exceeded')
    mapping = {after[int(j)]['candidate_id']: before[int(i)]['candidate_id'] for i, j in zip(rows, cols)}
    deltas = {before[int(i)]['candidate_id']: float(distance[i, j]) for i, j in zip(rows, cols)}
    whole_delta = 0
    for i, j in zip(rows, cols):
        if any(before[int(i)].get(k) != after[int(j)].get(k) for k in ['owner', 'owner_detection_index', 'owner_status', 'number', 'pinname']):
            raise ValueError('Fresh pin field mismatch')
    for profile, name in [('conservative', 'partial_submission.json'), ('candidate', 'submission_candidate.json')]:
        x, y = [read(path/f'proposal_export/{name}') for path in [folder, a.fresh_output]]
        if x['nets'] != y['nets'] or x['components'] != y['components']:raise ValueError('Complete net references/geometry/components differ')
        rc, fc = [read(path/f'proposal_export/{profile}_export_coverage.json') for path in [folder, a.fresh_output]]
        for proof in rc['atomic_complete_network_proofs']:
            whole_delta = max(whole_delta, *(deltas[cid] for cid in proof['all_original_member_ids']))
        normalize = lambda cov, m: sorted((proof['network_id'], tuple(sorted(m.get(cid, cid) for cid in proof['all_original_member_ids'])),
                                          tuple(sorted(proof['all_pin_references'])), proof['distinct_contact_count']) for proof in cov['atomic_complete_network_proofs'])
        if normalize(rc, {}) != normalize(fc, mapping):raise ValueError('Whole-member proof differs')
    if whole_delta > .05:raise ValueError('Exported whole-net tip numeric tolerance exceeded')
    log_path = ROOT/'runs/s21_regression.log';log = log_path.read_text(encoding='utf-8-sig');tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not re.search(r'\bOK\s*$', log) or 'test_verified_alias_preserves_all_members_in_complete_two_contact_net' not in log:
        raise ValueError('Missing whole-net regression record')
    result = {'fresh_case_id': a.case_id, 'fresh_PNG_passed': True, 'supplied_prediction_caches_used': False,
              'complete_network_members_references_geometry_and_fields_match_replay': True,
              'fresh_tip_max_delta_px': delta, 'tip_numeric_tolerance_px': .1,
              'fresh_whole_net_tip_max_delta_px': whole_delta, 'whole_net_tip_numeric_tolerance_px': .05,
              'fresh_entrypoint_wall_seconds': fresh['entrypoint_wall_seconds'], 'regression_tests_passed': int(tests[1]),
              'fresh_summary_sha256': sha(a.fresh_output/'summary.json'), 'replay_summary_sha256': sha(folder/'summary.json'),
              'regression_log_sha256': sha(log_path), 'default_pipeline_adopted': False,
              'official_complete_networks_certified': 0, 'official_NetF1': None}
    save(a.replay_root/'runtime_verification.json', result);print(result, flush=True)


if __name__ == '__main__':main()
