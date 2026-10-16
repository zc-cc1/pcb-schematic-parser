"""Selective wire-graph deployment. Preserve training models and labels."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True);args=parser.parse_args()
    target=args.target.resolve(strict=True)
    assert target!=ROOT and (target/'scripts/run_pin_text.py').is_file()
    protected_names=[p.relative_to(target).as_posix() for p in (target/'configs').glob('*.json')]
    protected_names+=[p.relative_to(target).as_posix() for p in (target/'weights').glob('*.pt')]
    protected_names+=['scripts/run_baseline.py','scripts/train_detector.py','scripts/train_unet.py','scripts/run_pin_ownership.py','scripts/run_pin_text.py',
                      'src/pcb_parser/pin_text.py','src/pcb_parser/pin_ownership.py','src/pcb_parser/unet_model.py','src/pcb_parser/unet_inference.py',
                      'runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
                      'data/segmentation_unet_v1/manifest.jsonl','data/detection_v1/manifest.jsonl']
    protected={name:sha(target/name) for name in protected_names if (target/name).is_file()}
    for name in ['scripts/run_pin_text.py','src/pcb_parser/component_text.py','scripts/evaluate_pin_text.py']:
        assert sha(target/name)==sha(ROOT/name), f'Upstream dependency differs: {name}'
    names=['src/pcb_parser/wire_graph.py','src/pcb_parser/wire_graph_artifacts.py','scripts/run_wire_graph.py','scripts/run_wire_graph.ps1',
           'scripts/batch_wire_graph.py','scripts/evaluate_wire_graph.py','scripts/build_topology_review.py',
           'configs/wire_graph_v1.json','requirements-topology.txt','tests/test_wire_graph.py','tests/test_wire_graph_artifacts.py',
           'docs/wire_graph_stage_v1.md']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'runs/s7_wire_graph_v1').rglob('*') if p.is_file() and p.suffix in {'.json','.md','.png','.npz'}]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'data/topology_review_v1').rglob('*') if p.is_file() and p.suffix in {'.json','.png'}]
    archive=target/'data/local_imports'/('wire_graph_'+datetime.now().strftime('%Y%m%d_%H%M%S'));archive.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in names:
        source,destination=ROOT/name,target/name
        assert target in destination.resolve().parents
        backup=None
        if destination.is_file() and sha(destination)!=sha(source):
            backup=archive/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,backup)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix=='.md':
            destination.write_text(source.read_text(encoding='utf-8').replace(ROOT.as_posix(),target.as_posix()),encoding='utf-8')
        else:
            shutil.copy2(source,destination);assert sha(source)==sha(destination)
        records.append({'file':name,'source_sha256':sha(source),'target_sha256':sha(destination),'backup':str(backup) if backup else None})
    assert all(sha(target/name)==expected for name,expected in protected.items())
    manifest={'status':'wire_graph_transferred_models_training_labels_preserved','target':str(target),'files':records,'protected_sha256':protected,
              'new_dependencies':['scikit-image==0.26.0','networkx==3.7'],'original_training_dataset_modified':False}
    path=archive/'import_manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'copied_files':len(records),'protected_files':len(protected),'manifest':str(path)},ensure_ascii=False))


if __name__=='__main__':main()
