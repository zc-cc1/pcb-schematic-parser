"""Selective pin-text deployment with hashes and recoverable file backups."""
import argparse
from datetime import datetime
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
    assert target!=ROOT and (target/'scripts/run_pin_ownership.py').is_file()
    protected_names=[p.relative_to(target).as_posix() for p in (target/'configs').glob('*.json')]
    protected_names+=['scripts/run_baseline.py','scripts/train_detector.py','scripts/train_unet.py','scripts/run_unet.py',
                      'scripts/run_pin_ownership.py','src/pcb_parser/pin_ownership.py','src/pcb_parser/unet_inference.py',
                      'weights/detector_dataset300_v3_nano.pt','runs/training/unet_wire_pin_v1/best.pt','runs/training/unet_wire_pin_v1/last.pt',
                      'data/detection_v1/manifest.jsonl','data/segmentation_unet_v1/manifest.jsonl']
    protected={name:sha(target/name) for name in protected_names if (target/name).is_file()}
    for name in ['scripts/run_pin_ownership.py','src/pcb_parser/pin_ownership.py']:
        assert sha(ROOT/name)==sha(target/name), f'Ownership dependency differs: {name}'
    if (target/'src/pcb_parser/component_text.py').exists():
        assert sha(ROOT/'src/pcb_parser/component_text.py')==sha(target/'src/pcb_parser/component_text.py'), 'Existing component OCR implementation differs; preserve it'
    names=['src/pcb_parser/pin_text.py','src/pcb_parser/pin_text_ocr.py','src/pcb_parser/component_text.py',
           'scripts/run_pin_text.py','scripts/run_pin_text.ps1','scripts/batch_pin_text.py','scripts/evaluate_pin_text.py',
           'configs/pin_text_v1.json','configs/pin_text_component_v2.json','tests/test_pin_text.py','docs/pin_text_stage_v1.md']
    # Exported reference artifacts are for offline scoring; inference does not
    # import the evaluator or consume these files.
    for file in (ROOT/'runs/s6_pin_text_v1').glob('*'):
        if file.is_file() and file.suffix in {'.json','.md'}:
            names.append(file.relative_to(ROOT).as_posix())
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'runs/s6_pin_text_v1/predictions').rglob('*') if p.is_file() and p.suffix in {'.json','.png'})
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'runs/s6_pin_text_v1/local_experiment').rglob('*') if p.is_file() and p.suffix in {'.json','.png','.md'})
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'runs/s6_pin_text_v1/global_OCR_inputs').rglob('*.json'))
    archive=target/'data/local_imports'/('pin_text_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    archive.mkdir(parents=True,exist_ok=False)
    copied=[]
    for name in names:
        source,destination=ROOT/name,target/name
        assert target in destination.resolve().parents
        backup=None
        if destination.exists() and sha(source)!=sha(destination):
            backup=archive/'backup'/name
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(destination,backup)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix=='.md':
            destination.write_text(source.read_text(encoding='utf-8').replace(ROOT.as_posix(),target.as_posix()),encoding='utf-8')
        else:
            shutil.copy2(source,destination)
            assert sha(source)==sha(destination)
        copied.append({'file':name,'source_sha256':sha(source),'target_sha256':sha(destination),'backup':str(backup) if backup else None})
    assert all(sha(target/name)==expected for name,expected in protected.items())
    manifest={'status':'pin_text_deployed_existing_models_training_labels_preserved','files':copied,'protected_sha256':protected,'target':str(target)}
    (archive/'import_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'copied_files':len(copied),'protected_files':len(protected),'manifest':str(archive/'import_manifest.json')},ensure_ascii=False))


if __name__=='__main__':
    main()
