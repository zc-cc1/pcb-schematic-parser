"""Selective field-stage transfer with upstream hashes and recoverable backup."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from run_submission_fields import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args();target=a.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_network_labels.py').is_file():raise ValueError('Expected existing desktop project')
    names=['src/pcb_parser/submission_fields.py','src/pcb_parser/label_roi_ocr.py',
           'scripts/run_submission_fields.py','scripts/run_submission_fields.ps1','scripts/batch_submission_fields.py','scripts/audit_submission_fields.py',
           'configs/submission_fields_v1.json','tests/test_submission_fields.py',
           'docs/submission_fields_stage_v1.md','docs/downstream_implementation_plan.md']
    for file in (ROOT/'runs/s10_submission_fields_v1').rglob('*'):
        if file.is_file() and file.suffix in {'.json','.md','.png'}:names.append(file.relative_to(ROOT).as_posix())
    writable=set(names);protected=set()
    for folder,glob in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        for file in (target/folder).glob(glob):
            name=file.relative_to(target).as_posix()
            if name not in writable:protected.add(name)
    for file in (target/'runs/training').rglob('*.pt'):protected.add(file.relative_to(target).as_posix())
    protected.update(['data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl',
                      'data/topology_review_v1/annotations.json','data/reviews/network_labels_logical_v1.json',
                      'runs/s9_network_labels_v1/evaluation.json'])
    hashes={name:sha(target/name) for name in sorted(protected) if (target/name).is_file()}
    for name in ['scripts/run_network_labels.py','scripts/run_wire_graph.py','scripts/run_pin_text.py',
                 'src/pcb_parser/wire_graph_artifacts.py','src/pcb_parser/network_labels.py','src/pcb_parser/pin_text_ocr.py']:
        if sha(target/name)!=sha(ROOT/name):raise ValueError('Upstream dependency differs: '+name)
    for row in read(ROOT/'runs/s9_network_labels_v1/batch_summary.json')['cases']:
        case=row['case_id']
        for name in [f'data/standardized/{case}/{case}.png',f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json',
                     f'runs/s6_pin_text_v1/predictions/{case}/texts.json',f'runs/s9_network_labels_v1/predictions/{case}/logical_graph_proposals.json']:
            if sha(target/name)!=sha(ROOT/name):raise ValueError('Frozen model prediction differs: '+name)
    archive=target/'data/local_imports'/('submission_fields_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    copied=[]
    for name in names:
        source,dest=ROOT/name,target/name
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(source):
            backup=archive/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Transfer hash mismatch')
        copied.append({'file':name,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/name)!=h for name,h in hashes.items()):raise ValueError('Protected upstream file changed')
    manifest=archive/'import_manifest.json'
    save(manifest,{'target':str(target),'files':copied,'protected_sha256':hashes,'new_training':False,
                   'original_dataset_modified':False,'status':'submission_fields_transferred_upstream_preserved'})
    print({'copied_files':len(copied),'protected_files':len(hashes),'manifest':str(manifest)},flush=True)


if __name__=='__main__':main()
