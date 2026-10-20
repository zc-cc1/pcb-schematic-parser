"""Stage the live entrypoint without replacing previous unrelated changes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;TARGET=Path('D:/pcb5_submission')
original=(TARGET/'agent.py').read_bytes();source=original.decode('utf-8').replace('\r\n','\n')
wrapper='''def finalize_pin_field_recovery(target, stage, image_rgb, image_sha256,
                                raw_pins, lead_px=18, tip_mode="color", reader=None):
    """Recover observed numeric pin fields, then sync proven existing nets."""
    if os.environ.get('PCB_PIN_FIELD_RECOVERY', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'network_changes': [], 'ocr_calls': 0}
    from pcb_parser.pin_field_recovery import recover_pin_fields, recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    records = json.loads((stage/'export/completed_component_records.json').read_text(encoding='utf-8'))['records']
    if graph.get('image_sha256') != image_sha256 or pins.get('image_sha256', image_sha256) != image_sha256:
        raise ValueError('Pin field recovery artifacts differ from the current image')
    if graph.get('coordinate_origin') != 'top_left' or graph.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
        raise ValueError('Pin field recovery image dimensions or coordinates differ')
    config_path = ROOT/'configs/pin_field_recovery_v1.json'
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
    def correct_positions(snapshot):
        if lead_px:
            pin_lead_snap(snapshot, float(lead_px), image_rgb, tip_mode)
    recovered, audit = recover_pin_fields(target, pins, graph, records, image_rgb,
        reader=reader, position_corrector=correct_positions, raw_pins=raw_pins, cfg=config)
    validate_target(recovered)
    audit['image_sha256'] = image_sha256
    audit['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    audit['ocr_models'] = fingerprints
    return recovered, audit


'''
anchor='def images_in(directory):'
assert source.count(anchor)==1 and 'def finalize_pin_field_recovery(' not in source
source=source.replace(anchor,wrapper+anchor)
anchor="        _lead = os.environ.get('PCB_PIN_LEAD_PX', '18')"
replacement="""        _field_raw_pins = None
        if output_format == 'target':
            from copy import deepcopy as _field_deepcopy
            _field_raw_pins = _field_deepcopy(target.get('pins') or {})
        _lead = os.environ.get('PCB_PIN_LEAD_PX', '18')"""
assert source.count(anchor)==1;source=source.replace(anchor,replacement)
anchor="        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"
replacement="""        if output_format == 'target':
            try:
                if rgb is None:
                    from PIL import Image as _FieldImage
                    import numpy as _field_np
                    with _FieldImage.open(image) as _field_image:
                        rgb = _field_np.asarray(_field_image.convert('RGB'))
                target, _field_audit = finalize_pin_field_recovery(
                    target, stage, rgb, summary['image_sha256'], _field_raw_pins,
                    _lead, os.environ.get('PCB_PIN_TIP_MODE', 'color').strip().lower())
                prediction = target
                row['pin_field_recovery'] = {'enabled': _field_audit['enabled'],
                    'changed_pins': len(_field_audit['changes']),
                    'changed_networks': sum('after' in item for item in _field_audit['network_changes']),
                    'ocr_calls': _field_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_field_recovery.json', _field_audit)
            except Exception as _error:
                row['pin_field_recovery_error'] = f'{type(_error).__name__}: {_error}'
        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"""
assert source.count(anchor)==1;source=source.replace(anchor,replacement)
(ROOT/'runtime/agent.py').write_text(source,encoding='utf-8',newline='\n')
paths=['agent.py','src/pcb_parser/pin_field_recovery.py','configs/pin_field_recovery_v1.json',
       'tests/test_pin_field_recovery.py','docs/pin_field_recovery.md']
files=[]
for name in paths:
    target=TARGET/name;previous=target.read_bytes() if target.exists() else None
    staged=ROOT/'runtime'/name
    if previous is not None:
        backup=ROOT/'backup'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        if backup.exists():assert backup.read_bytes()==previous
        else:backup.write_bytes(previous)
    files.append({'path':name,'sha256':hashlib.sha256(staged.read_bytes()).hexdigest(),
                  'previous_sha256':hashlib.sha256(previous).hexdigest() if previous else None})
(ROOT/'release_plan.json').write_text(json.dumps({'target':str(TARGET),'files':files},indent=2)+'\n',encoding='utf-8')
print('Staged',len(files),'files.')
