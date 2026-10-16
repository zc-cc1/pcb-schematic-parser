"""Check raw PNG S18 inference against frozen development replay."""
import argparse
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from run_targeted_pin_ocr import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--case-id',default='0018');p.add_argument('--fresh-output',type=Path,required=True)
    p.add_argument('--replay-root',type=Path,default=ROOT/'runs/s18_targeted_pin_ocr');a=p.parse_args()
    folder=a.replay_root/f'predictions/{a.case_id}'
    fresh,replay=read(a.fresh_output/'summary.json'),read(folder/'summary.json')
    if fresh.get('used_external_model_caches') is not False:raise ValueError('Expected raw PNG without supplied caches')
    for key in ['recovery','roi_count','baseline_export','proposal_export','completion']:
        if fresh[key]!=replay[key]:raise ValueError('Fresh/replay summary differs: '+key)
    if fresh['image_sha256']!=replay['image_sha256'] or fresh['targeted_OCR_config_sha256']!=replay['targeted_OCR_config_sha256']:
        raise ValueError('Fresh image/config differs')
    old,new=read(folder/'graph_proposals.json'),read(a.fresh_output/'graph_proposals.json')
    before,after=old['pin_attachments'],new['pin_attachments']
    if len(before)!=len(after):raise ValueError('Fresh pin count differs')
    distances=np.linalg.norm(np.array([p['point'] for p in before])[:,None]-np.array([p['point'] for p in after])[None,:],axis=2)
    rows,cols=linear_sum_assignment(distances);delta=float(distances[rows,cols].max()) if len(rows) else 0.0
    if delta>.05:raise ValueError('Fresh tip numeric tolerance exceeded')
    mapping={after[int(j)]['candidate_id']:before[int(i)]['candidate_id'] for i,j in zip(rows,cols)}
    for i,j in zip(rows,cols):
        if any(before[int(i)].get(k)!=after[int(j)].get(k) for k in ['owner','owner_detection_index','owner_status','number','pinname']):
            raise ValueError('Fresh pin field differs')
    normalize=lambda graph,m:sorted((tuple(sorted(m.get(cid,cid) for cid in n['pin_candidate_ids'])),
                                     tuple(sorted(n.get('official_pin_references',[])))) for n in graph['networks'])
    if normalize(old,{})!=normalize(new,mapping):raise ValueError('Fresh network/reference membership differs')
    log=(ROOT/'runs/s18_regression.log').read_text(encoding='utf-8-sig');tests=re.search(r'Ran (\d+) tests',log)
    if not tests or '\nOK' not in log or 'test_global_views_alone_cannot_trigger' not in log:
        raise ValueError('Expected full regression record')
    result={'fresh_case_id':a.case_id,'fresh_PNG_passed':True,'supplied_prediction_caches_used':False,
        'fresh_fields_and_network_members_match_replay':True,'fresh_tip_max_delta_px':delta,'tip_numeric_tolerance_px':.05,
        'fresh_entrypoint_wall_seconds':fresh['entrypoint_wall_seconds'],'regression_tests_passed':int(tests.group(1)),
        'regression_log_sha256':sha(ROOT/'runs/s18_regression.log'),'fresh_summary_sha256':sha(a.fresh_output/'summary.json'),
        'replay_summary_sha256':sha(folder/'summary.json'),'default_pipeline_adopted':False,'official_PinF1':None,'official_NetF1':None}
    save(a.replay_root/'runtime_verification.json',result);print(result,flush=True)


if __name__=='__main__':main()
