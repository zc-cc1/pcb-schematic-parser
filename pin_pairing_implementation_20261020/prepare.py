"""Build a narrow release against current live sources with exact backups."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;TARGET=Path('D:/pcb5_submission')
source=(TARGET/'agent.py').read_text(encoding='utf-8')
wrapper='''def finalize_pin_pairing(target, stage, image_rgb, image_sha256, reader=None):
    """Resolve shared-number conflicts as atomic pin/network transactions."""
    if os.environ.get('PCB_PIN_PAIRING', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'restored_networks': [], 'ocr_calls': 0}
    from pcb_parser.pin_pairing import reconcile_pin_pairs
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    for document in [pins, graph]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Pin pairing artifacts differ from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Pin pairing dimensions or coordinate system differ')
    config_path = ROOT/'configs/pin_pairing_v1.json'
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
    cleanup = cleanup_wire_edges if os.environ.get('PCB_WIRE_CLEANUP', '1').strip().lower() not in {'0', 'off', 'false'} else None
    recovered, audit = reconcile_pin_pairs(target, pins, graph, records, image_rgb, reader=reader, cleanup=cleanup, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


'''
assert 'def finalize_pin_pairing(' not in source
assert source.count('def images_in(directory):')==1
source=source.replace('def images_in(directory):',wrapper+'def images_in(directory):')
anchor="        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"
hook='''        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _PairingImage
                    import numpy as _pairing_np
                    with _PairingImage.open(image) as _pairing_image:
                        rgb = _pairing_np.asarray(_pairing_image.convert('RGB'))
                target, _pairing_audit = finalize_pin_pairing(target, stage, rgb, summary['image_sha256'])
                prediction = target
                row['pin_pairing'] = {'enabled': _pairing_audit['enabled'],
                    'changed_pins': len(_pairing_audit['changes']),
                    'changed_networks': sum('after' in item for item in _pairing_audit['network_changes']),
                    'restored_networks': len(_pairing_audit['restored_networks']),
                    'ocr_calls': _pairing_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_pairing.json', _pairing_audit)
            except Exception as _error:
                row['pin_pairing_error'] = f'{type(_error).__name__}: {_error}'
'''
assert source.count(anchor)==1;source=source.replace(anchor,hook+anchor)
(ROOT/'runtime/agent.py').write_text(source,encoding='utf-8',newline='\n')
files=[]
for name in ['agent.py','src/pcb_parser/pin_pairing.py','configs/pin_pairing_v1.json','tests/test_pin_pairing.py','docs/pin_pairing.md']:
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
