"""Authorized desktop transfer of the isolated power pilot, with backups."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from run_power_networks import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args();target=a.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_component_pin_library.py').is_file():raise ValueError('Expected existing desktop S14 project')
    names=['configs/power_networks_v1.json','src/pcb_parser/power_networks.py','scripts/run_power_networks.py',
           'scripts/run_power_networks.ps1','scripts/batch_power_networks.py','scripts/evaluate_power_networks.py',
           'scripts/build_power_network_report.py','tests/test_power_networks.py','docs/power_networks_stage_v1.md',
           'docs/downstream_implementation_plan.md','data/reviews/power_networks_v1.json']
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'runs/s15_power_networks').rglob('*') if f.is_file() and f.suffix in {'.json','.png','.md'}]
    names=sorted(set(names));protected=set()
    for folder,pattern in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        protected.update(f.relative_to(target).as_posix() for f in (target/folder).glob(pattern) if f.relative_to(target).as_posix() not in names)
    protected.update(['runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
        'data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl','data/manifest.jsonl',
        'configs/annotation_overrides.json','runs/s14_component_library/evaluation.json','runs/s14_component_library/runtime_verification.json',
        'data/reviews/component_pin_library_v1.json','runs/s9_network_labels_v1/reference_logical_connectivity.json'])
    protected_hash={n:sha(target/n) for n in sorted(protected) if (target/n).is_file()}
    for n in ['src/pcb_parser/network_labels.py','src/pcb_parser/pin_output_rules.py','scripts/run_component_pin_library.py',
        'scripts/run_pin_output_rules.py','scripts/run_network_labels.py','configs/component_pin_library_v1.json',
        'runs/s6_pin_text_v1/draft_field_references.json','runs/s9_network_labels_v1/reference_logical_connectivity.json']:
        if sha(target/n)!=sha(ROOT/n):raise ValueError('Frozen dependency differs: '+n)
    for row in read(ROOT/'runs/s15_power_networks/predictions/batch_summary.json')['cases']:
        case=row['case_id']
        for n in [f'data/standardized/{case}/{case}.png',f'runs/s14_component_library/predictions/{case}/pin_text/pin_text_result.json',
                  f'runs/s14_component_library/predictions/{case}/pin_text/texts.json',f'runs/s14_component_library/predictions/{case}/pin_text/component_records.json',
                  f'runs/s14_component_library/predictions/{case}/network/logical_graph_proposals.json']:
            if not (target/n).is_file() or sha(target/n)!=sha(ROOT/n):raise ValueError('Frozen model case differs: '+n)
    archive=target/'data/local_imports'/('power_networks_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    files=[]
    for n in names:
        src,dest=ROOT/n,target/n
        if target not in dest.resolve().parents:raise ValueError('Destination outside authorized target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(src):
            backup=archive/'backup'/n;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        if sha(dest)!=sha(src):raise ValueError('Transferred bytes differ')
        files.append({'file':n,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n)!=digest for n,digest in protected_hash.items()):raise ValueError('Original adopted inputs changed')
    manifest=archive/'import_manifest.json';save(manifest,{'target':str(target),'files':files,'protected_sha256':protected_hash,
         'status':'power_networks_trial_transferred','default_pipeline_adopted':False,'original_dataset_modified':False,'new_training':False})
    print({'manifest':str(manifest),'copied_files':len(files),'protected_files':len(protected_hash)},flush=True)


if __name__=='__main__':main()
