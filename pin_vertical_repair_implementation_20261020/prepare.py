"""Stage the observed-evidence repair against the exact current release."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;TARGET=Path('D:/pcb5_submission')
source=(TARGET/'agent.py').read_text(encoding='utf-8')
wrapper='''def finalize_pin_vertical_repair(target, stage, image_rgb, image_sha256, reader=None):
    """Reconcile observed vertical identities and measured endpoints atomically."""
    if os.environ.get('PCB_PIN_VERTICAL_REPAIR', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0, 'restored_networks': []}
    from pcb_parser.pin_vertical_repair import repair_vertical_triplet
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    for document in [pins, graph]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Vertical repair artifacts differ from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Vertical repair dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_vertical_repair_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    engine = None
    fingerprints = None
    if reader is None:
        def reader(rgb, box, scale, rotation, recognition_only, view):
            nonlocal engine, fingerprints
            if engine is None:
                from pcb_parser.pin_text_ocr import create_engine
                engine, fingerprints = create_engine({'onnx_threads': 4})
            return recognize(engine, rgb, box, scale, rotation, recognition_only, view)
    recovered, audit = repair_vertical_triplet(target, pins, graph, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


'''
assert 'def finalize_pin_vertical_repair(' not in source
assert source.count('def images_in(directory):')==1
source=source.replace('def images_in(directory):',wrapper+'def images_in(directory):')
anchor="        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"
hook='''        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _VerticalImage
                    import numpy as _vertical_np
                    with _VerticalImage.open(image) as _vertical_image:
                        rgb = _vertical_np.asarray(_vertical_image.convert('RGB'))
                target, _vertical_audit = finalize_pin_vertical_repair(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_vertical_repair'] = {'enabled': _vertical_audit['enabled'],
                    'changed_pins': len(_vertical_audit['changes']),
                    'restored_networks': len(_vertical_audit['restored_networks']),
                    'changed_networks': sum('after' in item for item in _vertical_audit['network_changes']),
                    'ocr_calls': _vertical_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_vertical_repair.json', _vertical_audit)
            except Exception as _error:
                row['pin_vertical_repair_error'] = f'{type(_error).__name__}: {_error}'
'''
assert source.count(anchor)==1;source=source.replace(anchor,hook+anchor)
(ROOT/'runtime/agent.py').write_text(source,encoding='utf-8',newline='\n')
files=[]
for name in ['agent.py','src/pcb_parser/pin_vertical_repair.py','configs/pin_vertical_repair_v1.json','tests/test_pin_vertical_repair.py','docs/pin_vertical_repair.md']:
    path=TARGET/name;previous=path.read_bytes() if path.exists() else None
    if previous is not None:
        backup=ROOT/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.read_bytes()==previous
        else:backup.write_bytes(previous)
    staged=ROOT/'runtime'/name
    files.append({'path':name,'sha256':hashlib.sha256(staged.read_bytes()).hexdigest(),
                  'previous_sha256':hashlib.sha256(previous).hexdigest() if previous else None})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(TARGET),'files':files},indent=2)+'\n',encoding='utf-8')
print('Staged',len(files),'files.')
