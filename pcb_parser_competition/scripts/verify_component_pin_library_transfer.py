"""Check desktop dependency hashes and a fresh PNG runtime result."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_component_pin_library import ROOT, read, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--fresh-output',type=Path,required=True)
    a=p.parse_args();manifest=read(a.manifest);target=Path(manifest['target'])
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:
            raise ValueError('Transferred file changed: '+row['file'])
    for name,digest in manifest['protected_sha256'].items():
        if sha(target/name)!=digest:
            raise ValueError('Frozen adopted upstream changed: '+name)
    fresh=read(a.fresh_output/'summary.json')
    if fresh.get('used_external_model_caches') is not False:
        raise ValueError('Expected fresh PNG inference')
    replay=read(target/'runs/s14_component_library/predictions/0018/summary.json')
    for key in ['library','components','conservative','candidate']:
        if fresh[key]!=replay[key]:
            raise ValueError('Fresh PNG differs from replay: '+key)
    before=read(target/'runs/s14_component_library/predictions/0018/pin_text/pin_text_result.json')['pin_candidates']
    after=read(a.fresh_output/'pin_text/pin_text_result.json')['pin_candidates']
    if len(before)!=len(after):
        raise ValueError('Fresh PNG candidate count differs')
    # The decoder orders temporary IDs by confidence. Small CPU/GPU numeric
    # differences can swap near-tied ranks; compare a spatial bijection instead.
    distance=np.linalg.norm(np.asarray([p['point'] for p in before])[:,None]-np.asarray([p['point'] for p in after])[None,:],axis=2)
    rows,cols=linear_sum_assignment(distance)
    tolerance=0.05
    if float(distance[rows,cols].max())>tolerance:
        raise ValueError('Fresh PNG coordinates differ beyond cross-device numeric tolerance')
    renumbered=0
    for i,j in zip(rows,cols):
        old,new=before[int(i)],after[int(j)]
        if any(old.get(k)!=new.get(k) for k in ['owner','owner_detection_index','owner_status','number','pinname','number_status','pinname_status']):
            raise ValueError('Fresh PNG pin fields differ')
        renumbered+=old['candidate_id']!=new['candidate_id']
    result={'status':'public_component_pin_library_target_verified','fresh_PNG_passed':True,
            'fresh_external_model_caches_used':False,'fresh_case_id':'0018',
            'fresh_fields_equal_after_geometric_matching':True,'fresh_match_tolerance_px':tolerance,
            'fresh_point_max_difference_px':float(distance[rows,cols].max()),'candidate_id_renumbered_count':renumbered,
            'library':fresh['library'],'conservative':fresh['conservative'],'candidate':fresh['candidate'],
            'frozen_upstream_sha256_unchanged':True,'protected_files':len(manifest['protected_sha256']),
            'copied_files_SHA_verified':len(manifest['files']), 'transfer_manifest':str(a.manifest),
            'default_pipeline_adopted':False,'new_training':False,'official_PinF1':None,'official_NetF1':None}
    save(ROOT/'runs/s14_component_library/runtime_verification.json',result)
    save(target/'runs/s14_component_library/runtime_verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
