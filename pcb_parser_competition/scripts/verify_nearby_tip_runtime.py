"""Verify the final S20 raw PNG entry point against development replay."""
import argparse
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from run_nearby_tip_fields import ROOT, read, save, sha


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--case-id', default='0018')
    parser.add_argument('--fresh-output', type=Path, required=True)
    parser.add_argument('--replay-root', type=Path, default=ROOT/'runs/s20_nearby_tip_fields');args = parser.parse_args()
    replay_folder = args.replay_root/f'predictions/{args.case_id}'
    replay, fresh = read(replay_folder/'summary.json'), read(args.fresh_output/'summary.json')
    if fresh.get('used_external_model_caches') is not False:
        raise ValueError('Expected raw PNG entrypoint without supplied stage caches')
    for key in ['image_sha256', 'size', 'nearby_tip_config_sha256', 'reconciliation', 'baseline_export', 'proposal_export', 'completion']:
        if replay[key] != fresh[key]:raise ValueError('Fresh/replay summary differs: '+key)
    before, after = read(replay_folder/'completed_pin_text_result.json'), read(args.fresh_output/'completed_pin_text_result.json')
    old, new = before['pin_candidates'], after['pin_candidates']
    if len(old) != len(new):raise ValueError('Fresh tip count differs')
    distance = np.linalg.norm(np.array([p['point'] for p in old])[:, None]-np.array([p['point'] for p in new])[None, :], axis=2)
    rows, cols = linear_sum_assignment(distance);delta = float(distance[rows, cols].max()) if len(rows) else 0
    if delta > .05:raise ValueError('Fresh tip numeric tolerance exceeded')
    mapping = {new[int(j)]['candidate_id']: old[int(i)]['candidate_id'] for i, j in zip(rows, cols)}
    for i, j in zip(rows, cols):
        if any(old[int(i)].get(k) != new[int(j)].get(k) for k in ['owner', 'owner_detection_index', 'owner_status', 'number', 'pinname']):
            raise ValueError('Fresh pin fields differ')
    graphs = [read(path/'graph_proposals.json') for path in [replay_folder, args.fresh_output]]
    normalize = lambda g, m: sorted((tuple(sorted(m.get(cid, cid) for cid in n['pin_candidate_ids'])), tuple(sorted(n.get('official_pin_references', [])))) for n in g['networks'])
    if normalize(graphs[0], {}) != normalize(graphs[1], mapping):raise ValueError('Fresh network member/reference sets differ')
    ra, fa = [read(path/'field_link_audit.json') for path in [replay_folder, args.fresh_output]]
    simplify = lambda audit, m: sorted((m.get(row['candidate_id'], row['candidate_id']),
                                       m.get(row['donor_candidate_id'], row['donor_candidate_id']), row['owner'], row['number'], row['pinname'], row['component_side'],
                                       tuple(row['number_support_views']), tuple(row['name_support_views'])) for row in audit['changes'])
    if simplify(ra, {}) != simplify(fa, mapping):raise ValueError('Fresh reconciled field evidence differs')
    log_path = ROOT/'runs/s20_regression.log';log = log_path.read_text(encoding='utf-8-sig');tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not re.search(r'\bOK\s*$', log) or 'test_missing_secondary_side_metadata_uses_geometry' not in log:
        raise ValueError('Missing regression record for fresh-side geometry gate')
    result = {'fresh_case_id': args.case_id, 'fresh_PNG_passed': True, 'supplied_prediction_caches_used': False,
              'fresh_fields_links_exports_and_network_members_match_replay': True, 'fresh_tip_max_delta_px': delta,
              'tip_numeric_tolerance_px': .05, 'fresh_entrypoint_wall_seconds': fresh['entrypoint_wall_seconds'],
              'regression_tests_passed': int(tests[1]), 'regression_log_sha256': sha(log_path),
              'fresh_summary_sha256': sha(args.fresh_output/'summary.json'), 'replay_summary_sha256': sha(replay_folder/'summary.json'),
              'default_pipeline_adopted': False, 'official_PinF1': None, 'official_NetF1': None}
    save(args.replay_root/'runtime_verification.json', result);print(result, flush=True)


if __name__ == '__main__':main()
