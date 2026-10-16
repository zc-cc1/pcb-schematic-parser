"""Compare a fresh PNG execution to the frozen S16 development replay."""
import argparse
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from run_power_caption_ocr import ROOT, read, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-id', default='0018')
    p.add_argument('--fresh-output', type=Path, required=True)
    p.add_argument('--replay-root', type=Path, default=ROOT/'runs/s16_power_caption_ocr')
    a = p.parse_args()
    folder = a.replay_root/f'predictions/{a.case_id}'
    fresh, replay = read(a.fresh_output/'summary.json'), read(folder/'summary.json')
    if fresh.get('used_external_model_caches') is not False:
        raise ValueError('Expected raw PNG execution without supplied prediction caches')
    for key in ['before_power', 'power', 'recovery', 'baseline_export', 'proposal_export']:
        if fresh[key] != replay[key]:
            raise ValueError('Fresh/replay summary differs: '+key)
    if fresh['image_sha256'] != replay['image_sha256'] or fresh['OCR_models'] != replay['OCR_models']:
        raise ValueError('Fresh/replay image or OCR weights differ')
    new, old = read(a.fresh_output/'power_graph_proposals.json'), read(folder/'power_graph_proposals.json')
    after, before = new['pin_attachments'], old['pin_attachments']
    if len(after) != len(before):
        raise ValueError('Fresh pin count differs')
    distance = np.linalg.norm(np.array([p['point'] for p in before])[:, None]
                              -np.array([p['point'] for p in after])[None, :], axis=2)
    rows, cols = linear_sum_assignment(distance)
    max_delta = float(distance[rows, cols].max()) if len(rows) else 0.0
    if max_delta > .05:
        raise ValueError('Fresh pin coordinates exceed CPU/GPU tolerance')
    remap = {after[int(j)]['candidate_id']: before[int(i)]['candidate_id'] for i, j in zip(rows, cols)}
    for i, j in zip(rows, cols):
        if any(before[int(i)].get(k) != after[int(j)].get(k)
               for k in ['owner', 'owner_detection_index', 'owner_status', 'number', 'pinname']):
            raise ValueError('Fresh pin fields differ')
    normalized = lambda graph, mapping: sorted((tuple(n.get('power_names', [])),
                        tuple(sorted(mapping.get(cid, cid) for cid in n['pin_candidate_ids']))) for n in graph['networks'])
    if normalized(new, remap) != normalized(old, {}):
        raise ValueError('Fresh network membership differs')
    log = (ROOT/'runs/s16_regression.log').read_text(encoding='utf-8-sig')
    tests = re.search(r'Ran (\d+) tests', log)
    if not tests or '\nOK' not in log or 'test_detected_boxes_restore_to_original_pixels' not in log:
        raise ValueError('Expected successful full regression record')
    result = {'fresh_case_id': a.case_id, 'fresh_PNG_passed': True, 'supplied_prediction_caches_used': False,
              'fresh_summary_fields_and_network_members_match_replay': True,
              'fresh_tip_max_delta_px': max_delta, 'tip_numeric_tolerance_px': .05,
              'fresh_local_stage_wall_seconds': fresh['wall_seconds'],
              'fresh_entrypoint_wall_seconds': fresh['entrypoint_wall_seconds'],
              'regression_tests_passed': int(tests.group(1)),
              'regression_log_sha256': sha(ROOT/'runs/s16_regression.log'),
              'fresh_summary_sha256': sha(a.fresh_output/'summary.json'),
              'replay_summary_sha256': sha(folder/'summary.json'), 'official_NetF1': None,
              'default_pipeline_adopted': False}
    save(a.replay_root/'runtime_verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
