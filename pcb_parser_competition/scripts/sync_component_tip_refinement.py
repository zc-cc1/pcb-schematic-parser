"""Authorized desktop transfer; add trial/review data without changing models."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from run_component_tip_refinement import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args();target=a.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_pin_output_rules.py').is_file():raise ValueError('Expected existing desktop project')
    names=['configs/component_tip_refinement_v1.json','src/pcb_parser/component_tip_refinement.py',
           'scripts/run_component_tip_refinement.py','scripts/run_component_tip_refinement.ps1','scripts/batch_component_tip_refinement.py',
           'scripts/evaluate_component_tip_refinement.py','scripts/audit_passive_number_training.py','scripts/build_component_tip_report.py',
           'tests/test_component_tip_refinement.py','docs/component_tip_refinement_stage_v1.md','docs/pin_number_questions.md',
           'docs/downstream_implementation_plan.md','data/reviews/component_tip_refinement_v1.json']
    for folder in ['runs/s13_component_tips','data/passive_number_review_v1']:
        names += [f.relative_to(ROOT).as_posix() for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in {'.json','.png','.md'}]
    names=sorted(set(names));protected=set()
    for folder,pattern in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        protected.update(f.relative_to(target).as_posix() for f in (target/folder).glob(pattern) if f.relative_to(target).as_posix() not in names)
    # Freeze only already adopted checkpoints; unrelated active training runs
    # can legitimately update their own weights while this transfer runs.
    protected.update(['runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
        'data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl','configs/annotation_overrides.json',
        'data/reviews/network_labels_logical_v1.json','runs/s12_pin_output_rules/audit.json'])
    hashes={n:sha(target/n) for n in sorted(protected) if (target/n).is_file()}
    for n in ['scripts/run_pin_text_v2.py','scripts/run_pin_output_rules.py','scripts/run_wire_graph.py',
              'src/pcb_parser/pin_ownership.py','src/pcb_parser/wire_graph.py','configs/pin_ownership_v1.json','configs/unet_inference_v2.json']:
        if sha(target/n)!=sha(ROOT/n):raise ValueError('Upstream dependency differs: '+n)
    for row in read(ROOT/'runs/s13_component_tips/predictions/batch_summary.json')['cases']:
        case=row['case_id']
        for n in [f'data/standardized/{case}/{case}.png',f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json',
                  f'runs/unet_inference_v2/cache/{case}/scale_1.0.npz',f'runs/unet_inference_v2/cache/{case}/scale_1.5.npz']:
            if not (target/n).is_file() and '/cache/' in n:
                names.append(n)
            elif sha(target/n)!=sha(ROOT/n):raise ValueError('Frozen image/model cache differs: '+n)
    if not (target/'runs/unet_inference_v2/split.json').is_file():names.append('runs/unet_inference_v2/split.json')
    names=sorted(set(names))
    archive=target/'data/local_imports'/('component_tip_refinement_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    files=[]
    for n in names:
        source,dest=ROOT/n,target/n
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(source):
            backup=archive/'backup'/n;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Copy SHA mismatch')
        files.append({'file':n,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n)!=digest for n,digest in hashes.items()):raise ValueError('Adopted upstream changed')
    mp=archive/'import_manifest.json';save(mp,{'target':str(target),'files':files,'protected_sha256':hashes,
          'status':'component_tip_refinement_experiment_transferred','new_training':False,'default_pipeline_adopted':False,
          'original_dataset_modified':False,'approved_number_training_examples':0})
    print({'files':len(files),'protected':len(hashes),'manifest':str(mp)},flush=True)


if __name__=='__main__':main()
