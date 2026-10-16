"""Copy the added stage into the authorized desktop project with SHA backups."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from run_pin_output_rules import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);a=p.parse_args();target=a.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_pin_semantics.py').is_file():raise ValueError('Expected existing desktop project')
    names=['configs/pin_output_rules_v1.json','src/pcb_parser/pin_output_rules.py','scripts/run_pin_output_rules.py',
           'scripts/run_pin_output_rules.ps1','scripts/batch_pin_output_rules.py','scripts/build_pin_output_rules_report.py',
           'tests/test_pin_output_rules.py','docs/pin_output_rules_stage_v1.md','docs/downstream_implementation_plan.md',
           'data/reviews/pin_output_rules_v1.json']
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'runs/s12_pin_output_rules').rglob('*')
              if f.is_file() and f.suffix in {'.json','.png','.md'}]
    names=sorted(set(names));protected=set()
    for folder,pattern in [('src/pcb_parser','*.py'),('scripts','*.py'),('scripts','*.ps1'),('configs','*.json'),('weights','*.pt')]:
        protected.update(f.relative_to(target).as_posix() for f in (target/folder).glob(pattern)
                         if f.relative_to(target).as_posix() not in names)
    protected.update(f.relative_to(target).as_posix() for f in (target/'runs/training').rglob('*.pt'))
    protected.update(['data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl',
                      'data/reviews/network_labels_logical_v1.json','runs/s11_pin_text_v2/export_audit.json'])
    hashes={n:sha(target/n) for n in sorted(protected) if (target/n).is_file()}
    for n in ['src/pcb_parser/submission_fields.py','src/pcb_parser/wire_graph_artifacts.py','scripts/run_pin_semantics.py',
              'scripts/run_submission_fields.py','scripts/run_network_labels.py','configs/submission_fields_v1.json']:
        if sha(target/n)!=sha(ROOT/n):raise ValueError('Upstream dependency differs: '+n)
    for item in read(ROOT/'runs/s11_pin_text_v2/predictions/batch_summary.json')['cases']:
        case=item['case_id']
        for n in [f'data/standardized/{case}/{case}.png',f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json',
                  f'runs/s11_pin_text_v2/predictions/{case}/texts.json',f'runs/s11_pin_text_v2/downstream/{case}/network/logical_graph_proposals.json']:
            if sha(target/n)!=sha(ROOT/n):raise ValueError('Frozen input differs: '+n)
    archive=target/'data/local_imports'/('pin_output_rules_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    copied=[]
    for n in names:
        source,dest=ROOT/n,target/n
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(source):
            backup=archive/'backup'/n;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Transfer SHA mismatch')
        copied.append({'file':n,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n)!=h for n,h in hashes.items()):raise ValueError('Protected upstream changed')
    path=archive/'import_manifest.json';save(path,{'target':str(target),'files':copied,'protected_sha256':hashes,
        'new_training':False,'dataset_modified':False,'status':'pin_output_rules_transferred_upstream_preserved'})
    print({'files':len(copied),'protected':len(hashes),'manifest':str(path)},flush=True)


if __name__=='__main__':main()
