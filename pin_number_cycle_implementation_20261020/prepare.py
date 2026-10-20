"""Stage the observed-evidence repair against the exact current release."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;TARGET=Path('D:/pcb5_submission')
source=(TARGET/'agent.py').read_text(encoding='utf-8')
wrapper='''def finalize_pin_number_cycle(target, stage, image_rgb, image_sha256, reader=None):
    """Corroborate and atomically permute bottom-row pin identities."""
    if os.environ.get('PCB_PIN_NUMBER_CYCLE', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_number_cycle import reconcile_bottom_number_cycles
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    for document in [pins, graph]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Pin number cycle artifacts differ from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Pin number cycle dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_number_cycle_v1.json'
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
    recovered, audit = reconcile_bottom_number_cycles(target, pins, graph, records, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


'''
assert 'def finalize_pin_number_cycle(' not in source
assert source.count('def images_in(directory):')==1
source=source.replace('def images_in(directory):',wrapper+'def images_in(directory):')
anchor="        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"
hook='''        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _CycleImage
                    import numpy as _cycle_np
                    with _CycleImage.open(image) as _cycle_image:
                        rgb = _cycle_np.asarray(_cycle_image.convert('RGB'))
                target, _cycle_audit = finalize_pin_number_cycle(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_number_cycle'] = {'enabled': _cycle_audit['enabled'],
                    'changed_pins': len(_cycle_audit['changes']),
                    'changed_networks': sum('after' in item for item in _cycle_audit['network_changes']),
                    'ocr_calls': _cycle_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_number_cycle.json', _cycle_audit)
            except Exception as _error:
                row['pin_number_cycle_error'] = f'{type(_error).__name__}: {_error}'
'''
assert source.count(anchor)==1;source=source.replace(anchor,hook+anchor)
(ROOT/'runtime/agent.py').write_text(source,encoding='utf-8',newline='\n')
files=[]
for name in ['agent.py','src/pcb_parser/pin_number_cycle.py','configs/pin_number_cycle_v1.json','tests/test_pin_number_cycle.py','docs/pin_number_cycle.md']:
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
