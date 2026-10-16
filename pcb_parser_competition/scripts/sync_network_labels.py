"""Selective transfer of the new stage; preserve every upstream model/config."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

from run_network_labels import ROOT,read,save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True);args=parser.parse_args();target=args.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_wire_graph.py').is_file():raise ValueError('Expected existing desktop project')
    names=['src/pcb_parser/network_labels.py','scripts/run_network_labels.py','scripts/run_network_labels.ps1',
           'scripts/batch_network_labels.py','scripts/evaluate_network_labels.py','configs/network_labels_v1.json',
           'tests/test_network_labels.py','data/reviews/network_labels_logical_v1.json','docs/network_labels_stage_v1.md','docs/downstream_implementation_plan.md']
    for p in (ROOT/'runs/s9_network_labels_v1').rglob('*'):
        if p.is_file() and p.suffix in {'.json','.md','.png'} and not {'initial_probe','revised_probe'}&set(p.parts):names.append(p.relative_to(ROOT).as_posix())
    writable=set(names)
    protected=[p.relative_to(target).as_posix() for p in (target/'configs').glob('*.json') if p.relative_to(target).as_posix() not in writable]
    protected+=[p.relative_to(target).as_posix() for p in (target/'weights').glob('*.pt')]
    protected+=['scripts/run_pin_text.py','scripts/run_pin_ownership.py','scripts/train_detector.py','scripts/train_unet.py','scripts/run_wire_graph.py',
                'src/pcb_parser/component_text.py','src/pcb_parser/pin_text.py','src/pcb_parser/pin_ownership.py','src/pcb_parser/unet_model.py',
                'src/pcb_parser/unet_inference.py','src/pcb_parser/wire_graph.py','src/pcb_parser/wire_graph_artifacts.py','src/pcb_parser/junction_classifier.py',
                'runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
                'data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl','data/topology_review_v1/annotations.json']
    expected={name:sha(target/name) for name in protected if (target/name).is_file()}
    for name in ['scripts/run_wire_graph.py','src/pcb_parser/component_text.py','src/pcb_parser/wire_graph.py','src/pcb_parser/wire_graph_artifacts.py']:
        if sha(target/name)!=sha(ROOT/name):raise ValueError('Upstream dependency differs: '+name)
    for row in read(ROOT/'runs/s9_network_labels_v1/batch_summary.json')['cases']:
        case=row['case_id']
        for name in [f'data/standardized/{case}/{case}.png',f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json',
                     f'runs/s6_pin_text_v1/predictions/{case}/texts.json',f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json']:
            if sha(target/name)!=sha(ROOT/name):raise ValueError('Frozen prediction differs: '+name)
    archive=target/'data/local_imports'/('network_labels_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in names:
        source,dest=ROOT/name,target/name
        if target not in dest.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if dest.is_file() and sha(dest)!=sha(source):
            backup=archive/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,backup)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha(dest)!=sha(source):raise ValueError('Transfer mismatch')
        rows.append({'file':name,'sha256':sha(dest),'backup':str(backup) if backup else None})
    if any(sha(target/n)!=h for n,h in expected.items()):raise ValueError('Protected files changed')
    manifest=archive/'import_manifest.json';save(manifest,{'files':rows,'protected_sha256':expected,'target':str(target),
         'status':'network_label_stage_transferred_upstream_unchanged','new_dependencies':[],'new_training':False,'original_dataset_modified':False})
    print({'copied_files':len(rows),'protected_files':len(expected),'manifest':str(manifest)})


if __name__=='__main__':main()
