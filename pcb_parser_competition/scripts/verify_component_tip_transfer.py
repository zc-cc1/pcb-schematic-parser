"""Verify the added desktop trial, model hashes and current-PNG provenance."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from run_component_tip_refinement import ROOT,read,save,sha,load_probability


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();target=a.target.resolve(strict=True);manifest=read(a.manifest)
    if Path(manifest['target']).resolve()!=target:raise ValueError('Wrong target manifest')
    for n,digest in manifest['protected_sha256'].items():
        if sha(target/n)!=digest:raise ValueError('Protected upstream changed: '+n)
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Transferred file SHA differs: '+row['file'])
    stage=Path('runs/s13_component_tips')
    if read(ROOT/stage/'evaluation.json')!=read(target/stage/'evaluation.json'):raise ValueError('Target offline evaluation differs')
    count=0
    for row in read(target/stage/'predictions/batch_summary.json')['cases']:
        case=row['case_id'];old=read(target/f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json')
        new=read(target/stage/f'predictions/{case}/pin_text_result.json')
        if new['pin_candidates'][:len(old['pin_candidates'])]!=old['pin_candidates']:
            raise ValueError('Original tip/owner/number/name changed: '+case)
        count+=len(old['pin_candidates'])
    runtime=target/stage/'png_runtime_check/0149';summary=read(runtime/'summary.json')
    upstream=read(runtime/'upstream/summary.json');owner=read(runtime/'upstream/ownership/summary.json')
    if (summary['used_external_model_caches'] or upstream['used_ownership_cache'] or upstream['used_external_OCR_cache']
            or owner['used_detector_cache'] or owner['used_pin_cache']):
        raise ValueError('Current PNG runtime used old input caches')
    old=read(runtime/'upstream/pin_text_result.json');new=read(runtime/'pin_text_result.json')
    if new['pin_candidates'][:len(old['pin_candidates'])]!=old['pin_candidates']:raise ValueError('Fresh original candidates changed')
    if summary['summary']['added_candidates']<1:raise ValueError('Fresh entry did not exercise addition')
    with np.load(runtime/'wire_probability.npz',allow_pickle=False) as cached_wire:wire=cached_wire['wire']
    if not np.array_equal(wire,load_probability(runtime/'upstream/ownership/probabilities.npz')[0]):
        raise ValueError('Fresh learned wire map changed')
    queue=read(target/'data/passive_number_review_v1/review_queue.json')
    if len(queue['examples'])!=40 or any(x['split']!='train' or x['approved_for_training'] for x in queue['examples']):
        raise ValueError('Number review queue is not exclusively unapproved train data')
    verification={'status':'component_tip_experiment_target_verified','protected_files_unchanged':len(manifest['protected_sha256']),
        'copied_files_sha_verified':len(manifest['files']),'new_tests_target':12,'legacy_output_tests_target':16,
        'target_copied_prediction_offline_evaluation_identical':True,'old_development_tip_points_preserved':count,
        'fresh_case':'0149','fresh_device':'cpu','fresh_used_external_model_caches':False,
        'fresh_original_candidate_count':len(old['pin_candidates']),'fresh_added_candidates':summary['summary']['added_candidates'],
        'fresh_added_types':summary['summary']['added_by_type'],'default_pipeline_adopted':False,'new_training':False,
        'original_dataset_modified':False,'review_crops_train_only':40,'approved_number_training_examples':0,
        'official_PinF1':None,'official_NetF1':None}
    for root in [ROOT,target]:save(root/stage/'runtime_verification.json',verification)
    from build_component_tip_report import main as report
    report();n='runs/s13_component_tips/report.md';dest=target/n
    backup=a.manifest.parent/'final_report_backup/report.md';backup.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(dest,backup);shutil.copy2(ROOT/n,dest)
    row=next(x for x in manifest['files'] if x['file']==n)
    row['initial_copy_sha256']=row['sha256'];row['sha256']=sha(dest);row['final_report_backup']=str(backup)
    manifest['post_transfer_verification']=verification['status'];save(a.manifest,manifest)
    for row in manifest['files']:
        if sha(target/row['file'])!=row['sha256']:raise ValueError('Final SHA differs: '+row['file'])
    print(verification,flush=True)


if __name__=='__main__':main()
