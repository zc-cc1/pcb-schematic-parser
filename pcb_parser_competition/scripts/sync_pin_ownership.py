"""Copy only the new ownership module and its artifacts, with backups/hashes."""
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args()
    target=args.target.resolve(strict=True)
    assert target!=ROOT and (target/'scripts/run_unet.py').is_file()
    protected_names=['scripts/run_baseline.py','scripts/train_detector.py','scripts/train_unet.py',
                     'configs/baseline_v3_adopted.json','configs/unet_inference_v2.json','configs/unet_label_reviews.json',
                     'weights/detector_dataset300_v3_nano.pt','runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt']
    # Older training checkouts may not contain the workspace-only adopted config.
    protected={name:sha(target/name) for name in protected_names if (target/name).is_file()}
    detector_cfg=json.loads((ROOT/'configs/pin_ownership_detector_v3.json').read_text(encoding='utf-8'))
    unet_cfg=json.loads((ROOT/'configs/unet_inference_v2.json').read_text(encoding='utf-8'))
    assert sha(target/detector_cfg['weights'])==detector_cfg['weights_sha256'], 'Target detector checkpoint differs'
    assert sha(target/'runs/training/unet_wire_pin_v1/best.pt')==unet_cfg['checkpoint_sha256'], 'Target U-Net checkpoint differs'
    assert sha(target/'src/pcb_parser/fusion.py')==sha(ROOT/'src/pcb_parser/fusion.py'), 'Target detector fusion implementation differs'
    names=['src/pcb_parser/pin_ownership.py','src/pcb_parser/pin_ownership_visual.py',
           'scripts/run_pin_ownership.py','scripts/run_pin_ownership.ps1',
           'configs/pin_ownership_v1.json','configs/pin_ownership_detector_v3.json',
           'tests/test_pin_ownership.py','docs/pin_ownership_stage_v1.md',
           'runs/s5_pin_ownership_v1/report.md','runs/s5_pin_ownership_v1/evaluation.json','runs/s5_pin_ownership_v1/label_review_queue.json']
    for folder in (ROOT/'runs/s5_pin_ownership_v1').iterdir():
        if folder.is_dir() and len(folder.name)==4 and folder.name.isdigit():
            names.extend(p.relative_to(ROOT).as_posix() for p in folder.iterdir() if p.suffix in {'.json','.png'})
    archive=target/'data/local_imports'/('pin_ownership_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    archive.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in names:
        source,destination=ROOT/name,target/name
        assert target in destination.resolve().parents
        backup=None
        if destination.exists() and sha(destination)!=sha(source):
            backup=archive/'backup'/name
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination,backup)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('report.md'):
            destination.write_text(source.read_text(encoding='utf-8').replace(ROOT.as_posix(),target.as_posix()),encoding='utf-8')
        else:
            shutil.copy2(source,destination)
            assert sha(source)==sha(destination)
        records.append({'file':name,'source_sha256':sha(source),'target_sha256':sha(destination),'backup':str(backup) if backup else None})
    assert protected=={name:sha(target/name) for name in protected}
    manifest={'status':'ownership_copied_existing_models_and_training_preserved','target':str(target),'files':records,'protected_sha256':protected}
    (archive/'import_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'files':len(records),'manifest':str(archive/'import_manifest.json')},ensure_ascii=False))


if __name__=='__main__':
    main()
