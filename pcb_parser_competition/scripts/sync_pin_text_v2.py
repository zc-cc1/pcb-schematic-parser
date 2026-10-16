"""Transfer new modules and evidence without overwriting frozen upstream files."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from run_pin_text_v2 import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args();target=a.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_submission_fields.py').is_file():raise ValueError('Expected existing desktop project')
    names=['src/pcb_parser/component_identity_recovery.py','src/pcb_parser/pin_text_v2.py','configs/pin_text_v2.json',
           'scripts/run_pin_text_v2.py','scripts/batch_pin_text_v2.py','scripts/evaluate_pin_text_v2.py',
           'scripts/run_pin_semantics.py','scripts/run_pin_semantics.ps1','scripts/batch_pin_semantics.py','scripts/build_pin_text_v2_report.py',
           'tests/test_pin_text_v2.py','docs/pin_text_stage_v2.md','docs/downstream_implementation_plan.md']
    for file in (ROOT/'runs/s11_pin_text_v2').rglob('*'):
        if file.is_file() and file.suffix in {'.json','.md','.png'} and not {'geometry_only_experiment','diagnostics'}&set(file.parts):
            names.append(file.relative_to(ROOT).as_posix())
    protected=set()
    for folder,pattern in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        for file in (target/folder).glob(pattern):
            name=file.relative_to(target).as_posix()
            if name not in names:protected.add(name)
    for file in (target/'runs/training').rglob('*.pt'):protected.add(file.relative_to(target).as_posix())
    protected.update(['data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl',
                      'data/topology_review_v1/annotations.json','data/reviews/network_labels_logical_v1.json',
                      'runs/s9_network_labels_v1/evaluation.json','runs/s10_submission_fields_v1/audit.json'])
    hashes={n:sha(target/n) for n in sorted(protected) if (target/n).is_file()}
    for n in ['src/pcb_parser/component_text.py','src/pcb_parser/pin_text.py','src/pcb_parser/pin_text_ocr.py',
              'scripts/run_pin_text.py','scripts/run_wire_graph.py','scripts/run_network_labels.py','scripts/run_submission_fields.py']:
        if sha(target/n)!=sha(ROOT/n):raise ValueError('Upstream dependency differs: '+n)
    for row in read(ROOT/'runs/s11_pin_text_v2/predictions/batch_summary.json')['cases']:
        case=row['case_id']
        for n in [f'data/standardized/{case}/{case}.png',f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json',
                  f'runs/s6_pin_text_v1/predictions/{case}/texts.json',f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz']:
            if sha(target/n)!=sha(ROOT/n):raise ValueError('Frozen input differs: '+n)
    archive=target/'data/local_imports'/('pin_text_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    copied=[]
    for n in names:
        source,dest=ROOT/n,target/n
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(source):
            backup=archive/'backup'/n;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Transfer hash mismatch')
        copied.append({'file':n,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n)!=h for n,h in hashes.items()):raise ValueError('Protected upstream changed')
    manifest=archive/'import_manifest.json';save(manifest,{'target':str(target),'files':copied,'protected_sha256':hashes,
        'new_training':False,'dataset_modified':False,'status':'pin_text_v2_transferred_upstream_preserved'})
    print({'files':len(copied),'protected':len(hashes),'manifest':str(manifest)},flush=True)


if __name__=='__main__':main()
