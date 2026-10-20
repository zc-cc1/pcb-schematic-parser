import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LIVE=Path('D:/pcb5_submission')
for folder in ['runtime/configs','runtime/tests','runtime/docs','backup']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
for p in (LIVE/'configs').glob('*.json'):(ROOT/'runtime/configs'/p.name).write_bytes(p.read_bytes())
library=ROOT/'runtime/data/component_library/BoxICLibrary.json';library.parent.mkdir(parents=True,exist_ok=True)
library.write_bytes((LIVE/'data/component_library/BoxICLibrary.json').read_bytes())
config={'min_confidence':.98,'min_views':2,'max_components':4,'max_ocr_calls':24,'max_rows':16,'scales':[3.,5.]}
(ROOT/'runtime/configs/pin_numeric_connector_v1.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
wrapper='''def finalize_pin_numeric_connector(target, stage, image_rgb, image_sha256, ocr_document, reader=None):
    """Bind observed digit-named connector rows and isolated local supply contacts."""
    if os.environ.get('PCB_PIN_NUMERIC_CONNECTOR', '1').strip().lower() in {'0', 'off', 'false'}:
        return target, {'enabled': False, 'changes': [], 'component_changes': [], 'network_changes': [], 'network_withheld': [], 'ocr_calls': 0}
    from pcb_parser.pin_numeric_connector import recover_numeric_connectors
    from pcb_parser.pin_field_recovery import recognize
    pins = json.loads((stage/'export/engineering_pin_text_result.json').read_text(encoding='utf-8'))
    graph = json.loads((stage/'export/engineering_graph_proposals.json').read_text(encoding='utf-8'))
    for document in [pins, graph, ocr_document]:
        if document.get('image_sha256') != image_sha256 or not document.get('prediction_only'):
            raise ValueError('Numeric connector evidence differs from current-image predictions')
        if document.get('coordinate_origin') != 'top_left' or document.get('size') != [image_rgb.shape[1], image_rgb.shape[0]]:
            raise ValueError('Numeric connector image dimensions or coordinates differ')
    config_path = ROOT/'configs/pin_numeric_connector_v1.json'
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
    recovered, audit = recover_numeric_connectors(target, pins, graph, ocr_document, image_rgb, reader=reader, cfg=config)
    validate_target(recovered)
    audit.update(image_sha256=image_sha256, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), ocr_models=fingerprints)
    return recovered, audit


'''
source=(LIVE/'agent.py').read_text(encoding='utf-8')
assert 'def finalize_pin_numeric_connector' not in source
source=source.replace('def images_in(directory):',wrapper+'def images_in(directory):',1)
marker="        write(output/'final_results'/f'{case_id}_Pred.json', prediction)"
hook='''        if output_format == 'target':
            try:
                target, _numeric_audit = finalize_pin_numeric_connector(target, stage, rgb, summary['image_sha256'], ocr_document)
                prediction = target
                row['pin_numeric_connector'] = {'enabled': _numeric_audit['enabled'],
                    'changed_pins': len(_numeric_audit['changes']),
                    'changed_components': len(_numeric_audit['component_changes']),
                    'changed_networks': len(_numeric_audit['network_changes']),
                    'ocr_calls': _numeric_audit['ocr_calls']}
                write(output/'logs'/f'{case_id}_pin_numeric_connector.json', _numeric_audit)
            except Exception as _error:
                row['pin_numeric_connector_error'] = f'{type(_error).__name__}: {_error}'
'''
assert source.count(marker)==1
source=source.replace(marker,hook+marker,1)
(ROOT/'runtime/agent.py').write_text(source,encoding='utf-8')
(ROOT/'backup/agent.py').write_bytes((LIVE/'agent.py').read_bytes())
print('Staged numeric connector hook; production files unchanged.')
