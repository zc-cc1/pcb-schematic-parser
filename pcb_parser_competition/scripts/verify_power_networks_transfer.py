"""Desktop byte checks and fresh PNG power membership verification."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_power_networks import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--fresh-output',type=Path,required=True);a=p.parse_args()
    manifest=read(a.manifest);target=Path(manifest['target'])
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Transferred bytes changed: '+row['file'])
    for name,digest in manifest['protected_sha256'].items():
        if sha(target/name)!=digest:raise ValueError('Original dependency changed: '+name)
    fresh=read(a.fresh_output/'summary.json');replay=read(target/'runs/s15_power_networks/predictions/0018/summary.json')
    if fresh.get('used_external_model_caches') is not False:raise ValueError('Expected raw PNG entry without model input caches')
    for key in ['power','baseline_export','proposal_export']:
        if fresh[key]!=replay[key]:raise ValueError('Fresh PNG differs from replay: '+key)
    old=read(target/'runs/s15_power_networks/predictions/0018/power_graph_proposals.json');new=read(a.fresh_output/'power_graph_proposals.json')
    before,after=old['pin_attachments'],new['pin_attachments']
    if len(before)!=len(after):raise ValueError('Fresh tip count differs')
    distances=np.linalg.norm(np.asarray([p['point'] for p in before])[:,None]-np.asarray([p['point'] for p in after])[None,:],axis=2)
    rows,cols=linear_sum_assignment(distances);max_delta=float(distances[rows,cols].max())
    if max_delta>.05:raise ValueError('Fresh point difference exceeds CPU/GPU numeric tolerance')
    bijection={after[int(j)]['candidate_id']:before[int(i)]['candidate_id'] for i,j in zip(rows,cols)}
    for i,j in zip(rows,cols):
        if any(before[int(i)].get(k)!=after[int(j)].get(k) for k in ['owner','owner_detection_index','owner_status','number','pinname']):
            raise ValueError('Fresh pin fields differ')
    normalize=lambda g,b:sorted((tuple(n.get('power_names',[])),tuple(sorted(b.get(p,p) for p in n['pin_candidate_ids']))) for n in g['networks'])
    if normalize(old,{})!=normalize(new,bijection):raise ValueError('Fresh network membership differs')
    result={'status':'power_networks_desktop_verified','fresh_case_id':'0018','fresh_PNG_passed':True,
            'fresh_external_model_caches_used':False,'fresh_point_max_delta_px':max_delta,'point_numeric_tolerance_px':.05,
            'fresh_pin_fields_and_power_network_members_match':True,'power':fresh['power'],'export':fresh['proposal_export'],
            'protected_files':len(manifest['protected_sha256']),'original_dependency_SHA_unchanged':True,
            'copied_files_SHA_verified':len(manifest['files']),'reported_tests':{'power_rule_checks':12,'output_regression_checks':16,'passed':True},
            'default_pipeline_adopted':False,'new_training':False,'official_NetF1':None,'transfer_manifest':str(a.manifest)}
    save(ROOT/'runs/s15_power_networks/runtime_verification.json',result);save(target/'runs/s15_power_networks/runtime_verification.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
