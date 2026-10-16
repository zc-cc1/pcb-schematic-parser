"""Authorized desktop transfer with backups and frozen adopted dependencies."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

from run_component_pin_library import ROOT, read, save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, required=True)
    a = p.parse_args()
    target = a.target.resolve(strict=True)
    if target == ROOT or not (target/'scripts/run_pin_output_rules.py').is_file():
        raise ValueError('Expected the existing authorized desktop project')
    names = ['configs/component_pin_library_v1.json','src/pcb_parser/component_pin_library.py',
             'scripts/run_component_pin_library.py','scripts/run_component_pin_library.ps1','scripts/batch_component_pin_library.py',
             'scripts/evaluate_component_pin_library.py','scripts/build_component_library_report.py','tests/test_component_pin_library.py',
             'docs/component_pin_library_stage_v1.md','docs/downstream_implementation_plan.md',
             'data/component_pin_library_v1/library.json','data/component_pin_library_v1/sources/stm32f103c8.pdf',
             'data/reviews/component_pin_library_v1.json']
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'runs/s14_component_library').rglob('*')
              if f.is_file() and f.suffix in {'.json','.png','.md'}]
    names = sorted(set(names))
    protected = set()
    for folder,pattern in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        protected.update(f.relative_to(target).as_posix() for f in (target/folder).glob(pattern) if f.relative_to(target).as_posix() not in names)
    protected.update(['runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
                      'data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl',
                      'configs/annotation_overrides.json','data/reviews/network_labels_logical_v1.json',
                      'runs/s12_pin_output_rules/audit.json','runs/s13_component_tips/evaluation.json'])
    hashes = {n:sha(target/n) for n in sorted(protected) if (target/n).is_file()}
    for n in ['scripts/run_pin_text_v2.py','scripts/run_pin_output_rules.py','scripts/run_wire_graph.py',
              'scripts/run_network_labels.py','src/pcb_parser/pin_text_v2.py','src/pcb_parser/wire_graph.py',
              'configs/unet_inference_v2.json','configs/pin_text_v2.json','runs/s6_pin_text_v1/draft_field_references.json']:
        if sha(target/n) != sha(ROOT/n):
            raise ValueError('Frozen dependency differs: '+n)
    for row in read(ROOT/'runs/s14_component_library/predictions/batch_summary.json')['cases']:
        case = row['case_id']
        for n in [f'data/standardized/{case}/{case}.png',f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json',
                  f'runs/s11_pin_text_v2/predictions/{case}/texts.json',f'runs/s11_pin_text_v2/predictions/{case}/component_records.json',
                  f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz',f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.json',
                  f'runs/s11_pin_text_v2/downstream/{case}/network/logical_graph_proposals.json']:
            if not (target/n).is_file() or sha(target/n) != sha(ROOT/n):
                raise ValueError('Frozen case model input differs: '+n)
    archive = target/'data/local_imports'/('component_pin_library_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    archive.mkdir(parents=True,exist_ok=False)
    files = []
    for n in names:
        source,dest = ROOT/n,target/n
        if target not in dest.resolve().parents:
            raise ValueError('Destination outside intended target')
        backup = None
        if dest.is_file() and sha(dest) != sha(source):
            backup = archive/'backup'/n;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest) != sha(source):
            raise ValueError('Transferred file bytes differ')
        files.append({'file':n,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n) != digest for n,digest in hashes.items()):
        raise ValueError('Frozen adopted dependency changed')
    manifest = archive/'import_manifest.json'
    save(manifest,{'target':str(target),'files':files,'protected_sha256':hashes,
         'default_pipeline_adopted':False,'new_training':False,'original_dataset_modified':False,
         'status':'public_component_pin_library_trial_transferred'})
    print({'manifest':str(manifest),'copied_files':len(files),'protected_files':len(hashes)},flush=True)


if __name__=='__main__':
    main()
