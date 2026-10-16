"""Selective deployment with backups and upstream integrity guards."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil

from run_wire_graph import ROOT,read,save,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True);args=parser.parse_args()
    target=args.target.resolve(strict=True)
    if target==ROOT or not (target/'scripts/run_pin_text.py').is_file():raise ValueError('Expected the existing desktop PCB project')
    names=['scripts/prepare_junction_reviews.py','scripts/train_junction_classifier.py','scripts/freeze_junction_model.py',
           'scripts/batch_learned_wire_graph.py','scripts/evaluate_learned_wire_graph.py','scripts/run_learned_wire_graph.ps1',
           'scripts/run_wire_graph.py','src/pcb_parser/wire_graph.py','src/pcb_parser/wire_graph_artifacts.py',
           'src/pcb_parser/junction_classifier.py','src/pcb_parser/junction_training_data.py',
           'configs/junction_training_v1.json','configs/wire_graph_v2.json','weights/junction_classifier_v1.pt',
           'tests/test_junction_classifier.py','docs/junction_classifier_stage_v1.md','docs/downstream_implementation_plan.md']
    for folder in ['data/junction_classifier_v1','runs/training/junction_classifier_v1','runs/s8_junction_classifier_v1']:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.json','.md','.png','.pt'}]
    writable=set(names)
    protected_names=[p.relative_to(target).as_posix() for p in (target/'configs').glob('*.json') if p.relative_to(target).as_posix() not in writable]
    protected_names += [p.relative_to(target).as_posix() for p in (target/'weights').glob('*.pt') if p.relative_to(target).as_posix() not in writable]
    protected_names += ['scripts/train_detector.py','scripts/train_unet.py','scripts/run_baseline.py','scripts/run_pin_ownership.py','scripts/run_pin_text.py',
                        'src/pcb_parser/pin_text.py','src/pcb_parser/pin_ownership.py','src/pcb_parser/unet_model.py','src/pcb_parser/unet_inference.py',
                        'runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
                        'data/segmentation_unet_v1/manifest.jsonl','data/detection_v1/manifest.jsonl','data/topology_review_v1/annotations.json']
    protected={name:sha(target/name) for name in protected_names if (target/name).is_file()}
    for name in ['scripts/run_pin_text.py','src/pcb_parser/component_text.py','scripts/evaluate_pin_text.py','configs/wire_graph_v1.json']:
        if sha(target/name)!=sha(ROOT/name):raise ValueError('Upstream dependency differs: '+name)
    for row in read(ROOT/'data/junction_classifier_v1/reviewed_annotations.json')['records']:
        if sha(target/f"data/standardized/{row['case_id']}/{row['case_id']}.png")!=row['source_image_sha256']:raise ValueError('Target source image differs')
        if sha(target/'data/topology_review_v1'/row['crop_file'])!=row['crop_sha256']:raise ValueError('Target review crop differs')
    archive=target/'data/local_imports'/('junction_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    files=[]
    for name in names:
        source,destination=ROOT/name,target/name
        if target not in destination.resolve().parents:raise ValueError('Destination outside target')
        backup=None
        if destination.is_file() and sha(destination)!=sha(source):
            backup=archive/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,backup)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if sha(source)!=sha(destination):raise ValueError('Transfer mismatch: '+name)
        files.append({'file':name,'sha256':sha(source),'backup':str(backup) if backup else None})
    if any(sha(target/name)!=expected for name,expected in protected.items()):raise ValueError('Protected upstream files changed')
    path=archive/'import_manifest.json';save(path,{'status':'junction_pilot_transferred_v1_default_preserved','files':files,'protected_sha256':protected,
           'target':str(target),'new_dependencies':[],'upstream_training_started':False,'original_annotations_modified':False})
    print({'files':len(files),'protected_files':len(protected),'manifest':str(path)})


if __name__=='__main__':main()
