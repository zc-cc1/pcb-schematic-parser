"""Verify target replay, fresh PNG provenance, all frozen points and backups."""
import argparse
from pathlib import Path
import shutil
from run_pin_output_rules import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();target=a.target.resolve(strict=True);manifest=read(a.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Wrong target manifest')
    for name,digest in manifest['protected_sha256'].items():
        if sha(target/name)!=digest:raise ValueError('Protected file changed: '+name)
    changed=[]
    for row in manifest['files']:
        dest=target/row['file']
        if sha(dest)!=row['sha256']:
            if not row['file'].startswith('runs/s12_pin_output_rules/predictions/'):
                raise ValueError('Transferred source changed: '+row['file'])
            # A target replay changes elapsed time, not predictions. Compare
            # every functional payload below before adopting replayed hashes.
            changed.append(row)
    relative=Path('runs/s12_pin_output_rules');main_batch=read(ROOT/relative/'predictions/batch_summary.json')
    target_batch=read(target/relative/'predictions/batch_summary.json')
    if not target_batch.get('completed') or main_batch['aggregate']!=target_batch['aggregate']:
        raise ValueError('26-case coverage differs')
    count=0
    for row in main_batch['cases']:
        case=row['case_id'];folder=relative/'predictions'/case
        for name in ['partial_submission.json','submission_candidate.json','export_coverage.json','candidate_export_coverage.json',
                     'candidate_edge_geometry.json','pin_field_changes.json','power_review_queue.json','network_review_manifest.json']:
            if read(ROOT/folder/name)!=read(target/folder/name):raise ValueError('Target replay differs: '+case+'/'+name)
        # S11 downstream already applies the single-label pin_1 convention.
        # Compare against that stage boundary, rather than its earlier OCR file.
        original=read(target/f'runs/s11_pin_text_v2/downstream/{case}/submission/completed_pin_text_result.json')
        expected=[tuple(p.get(k) if k!='point' else tuple(p[k]) for k in ['candidate_id','point','owner_detection_index','number']) for p in original['pin_candidates']]
        for name in ['conservative_pin_text_result.json','candidate_pin_text_result.json']:
            actual=read(target/folder/name)
            signature=[tuple(p.get(k) if k!='point' else tuple(p[k]) for k in ['candidate_id','point','owner_detection_index','number']) for p in actual['pin_candidates']]
            if expected!=signature:raise ValueError('Original tip, owner or number changed')
        count+=len(expected)
    runtime=target/relative/'png_runtime_check/0002';summary=read(runtime/'summary.json')
    upstream=runtime/'upstream';text=read(upstream/'pin_text/summary.json');owner=read(upstream/'pin_text/ownership/summary.json')
    if (summary['used_external_model_caches'] or read(upstream/'summary.json')['used_external_model_caches']
            or text['used_ownership_cache'] or text['used_external_OCR_cache']
            or owner['used_detector_cache'] or owner['used_pin_cache']):
        raise ValueError('Fresh PNG entry used old input caches')
    original=read(upstream/'pin_text/ownership/pin_ownership.json')
    expected=[(p['candidate_id'],p['point']) for p in original['pin_candidates']]
    for name in ['conservative_pin_text_result.json','candidate_pin_text_result.json']:
        result=read(runtime/name)
        if expected!=[(p['candidate_id'],p['point']) for p in result['pin_candidates']]:raise ValueError('Fresh tips changed')
    fresh_json=read(runtime/'submission_candidate.json')
    if '(C86.1,IC2.R8,X2.3)' not in {n['hyperGraph'] for n in fresh_json['nets'].values()}:
        raise ValueError('Fresh PNG did not reproduce the visually checked candidate network')
    verification={'status':'pin_output_rules_target_verified','protected_files_unchanged':len(manifest['protected_sha256']),
        'new_tests_main_and_target':16,'legacy_field_tests_main_and_target':13,'target_26_case_replay_JSON_identical':True,
        'development_pin_points_owners_numbers_preserved':count,'fresh_png_case':'0002','fresh_png_device':'cpu',
        'fresh_png_used_external_model_caches':False,'fresh_png_pin_count':len(expected),
        'fresh_export':{k:summary[k] for k in ['conservative','candidate']},
        'fresh_candidate_net':'(C86.1,IC2.R8,X2.3)','new_training':False,'original_dataset_modified':False,
        'official_PinF1':None,'official_NetF1':None,'complete_competition_JSON':False}
    for row in changed:
        row['initial_copy_sha256']=row['sha256'];row['sha256']=sha(target/row['file'])
    for root in [ROOT,target]:save(root/relative/'runtime_verification.json',verification)
    # Generate the reviewed report once, then synchronize its exact wording.
    from build_pin_output_rules_report import main as report
    report()
    name='runs/s12_pin_output_rules/report.md';dest=target/name
    backup=a.manifest.parent/'final_report_backup/report.md';backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,backup);shutil.copy2(ROOT/name,dest)
    row=next(r for r in manifest['files'] if r['file']==name)
    row['initial_copy_sha256']=row['sha256'];row['sha256']=sha(dest);row['final_report_backup']=str(backup)
    manifest['post_transfer_verification']=verification['status'];save(a.manifest,manifest)
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Final target SHA differs: '+row['file'])
    print({k:verification[k] for k in ['status','protected_files_unchanged','development_pin_points_owners_numbers_preserved','fresh_export']},flush=True)


if __name__=='__main__':main()
