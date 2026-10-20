"""PNG -> frozen image models -> public exact-package pin constraints -> JSON."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image

from run_pin_text_v2 import ROOT, read, save, sha, validate_image
from run_submission_fields import fingerprint
from run_wire_graph import process as wire_process, validate_cache
from run_network_labels import process as labels_process
from run_pin_output_rules import process as export_process, validate_cache as validate_baseline
from pcb_parser.component_pin_library import bind_library_pins, validate_library


def load_library(cfg):
    path = ROOT/cfg['library_path']
    if sha(path) != cfg['library_sha256']:
        raise ValueError('Public pin library bytes changed')
    library = read(path)
    validate_library(library)
    for part in library['parts']:
        if sha(ROOT/part['source']['local_path']) != part['source']['sha256']:
            raise ValueError('Public source document bytes changed')
    return library, sha(path)


def process(image, original_folder, wire_cache, wire_metadata, output, cfg, baseline_network=None, device='cpu'):
    start = time.perf_counter()
    image_hash = sha(image)
    with Image.open(image) as source:
        size = list(source.size)
    original_path = original_folder/'pin_text_result.json'
    texts_path = original_folder/'texts.json'
    records_path = original_folder/'component_records.json'
    original, ocr, records = read(original_path), read(texts_path), read(records_path)
    for payload in [original, ocr]:
        validate_image(payload, image_hash, size)
    if not ocr.get('prediction_only') or records.get('image_sha256') != image_hash:
        raise ValueError('Expected original PNG model OCR/component records')
    if original['pin_text_config_sha256'] != fingerprint(read(ROOT/'configs/pin_text_v2.json')):
        raise ValueError('Pin-text cache configuration differs')
    if original['component_text_config_sha256'] != sha(ROOT/'configs/pin_text_component_v2.json'):
        raise ValueError('Component-text configuration differs')
    components = {c['detection_index']: c for c in original['components']}
    if len(records['records']) != len(components):
        raise ValueError('Component records incomplete')
    for r in records['records']:
        c = components[r['detection_index']]
        if r['component_id'] != c['component_id'] or r['bbox_top_left'] != c['bbox_top_left']:
            raise ValueError('Component records differ from input owners')
    inference = ROOT/'configs/unet_inference_v2.json'
    validate_cache(read(wire_metadata), wire_cache, image_hash, size,
                   read(inference)['checkpoint_sha256'], sha(inference))
    if baseline_network is not None:
        validate_baseline(image, baseline_network, original_path, texts_path)
    library, library_hash = load_library(cfg)
    # A connector prints its part number as one line ("TYPE-C 6P(073)") and the page OCR reads
    # that line whole, but the component-field binder only accepts single tokens, so the part
    # stayed unnamed and the library could not number its pins.  Bind such a line as the model.
    model_names = []
    try:
        if __import__('os').environ.get('PCB_MODEL_NAME_FROM_OCR', '0').strip().lower() in {'0','off','false',''}:
            raise RuntimeError('model-name binding disabled')
        from run_engineering_networks import bind_model_names
        model_names = bind_model_names(records['records'], ocr['texts'])
    except Exception as error:
        print(f'MODEL-NAME-ERROR {type(error).__name__}: {error}', flush=True)
    result, audit = bind_library_pins(original, ocr['texts'], records['records'], library, cfg, read(ROOT/'configs/pin_text_v2.json'))
    invariant = lambda payload: [(p['candidate_id'], p['point'], p.get('owner_detection_index'), p['owner_status'], p.get('owner')) for p in payload['pin_candidates']]
    if invariant(original) != invariant(result):
        raise ValueError('Library stage changed pin localization or ownership')
    allowed = {r['detection_index'] for r in audit['components']}
    if any(old != new for old, new in zip(original['pin_candidates'], result['pin_candidates']) if old.get('owner_detection_index') not in allowed):
        raise ValueError('Unsupported component fields changed')
    provenance = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left',
                  'library_sha256': library_hash, 'library_config_sha256': fingerprint(cfg),
                  'original_pin_input_sha256': sha(original_path), 'OCR_input_sha256': sha(texts_path),
                  'records_input_sha256': sha(records_path), 'prediction_only': True}
    result.update(**provenance)
    pin_folder = output/'pin_text'
    save(pin_folder/'pin_text_result.json', result)
    save(pin_folder/'summary.json', {'summary': result['pin_text_summary'], **provenance})
    for path in [texts_path, records_path]:
        shutil.copy2(path, pin_folder/path.name)
    if model_names:
        save(pin_folder/'component_records.json', records)
        provenance['records_input_sha256'] = sha(pin_folder/'component_records.json')
        save(pin_folder/'summary.json', {'summary': result['pin_text_summary'], **provenance,
                                        'model_names_bound': model_names})
    save(output/'library_binding_audit.json', {**provenance, **audit})
    wire = wire_process(image, pin_folder/'pin_text_result.json', wire_cache, output/'physical',
                        read(ROOT/'configs/wire_graph_v1.json'), wire_metadata, device)
    labels = labels_process(image, output/'physical/graph_proposals.json', pin_folder/'pin_text_result.json',
                            pin_folder/'texts.json', output/'network', read(ROOT/'configs/network_labels_v1.json'))
    exports = export_process(image, output/'network', pin_folder/'pin_text_result.json', pin_folder/'texts.json',
                             output/'output_rules', read(ROOT/'configs/pin_output_rules_v1.json'))
    physical_unchanged = None
    if baseline_network is not None:
        old, new = read(baseline_network/'logical_graph_proposals.json'), read(output/'network/logical_graph_proposals.json')
        attachments = lambda g: [{k:v for k,v in p.items() if k not in {'number','pinname'}} for p in g['pin_attachments']]
        physical_unchanged = old['edges'] == new['edges'] and attachments(old) == attachments(new)
        if not physical_unchanged or [n['pin_candidate_ids'] for n in old['networks']] != [n['pin_candidate_ids'] for n in new['networks']]:
            raise ValueError('Pin text library stage changed wire geometry or network members')
    summary = {**provenance, 'library': audit['summary'], 'components': audit['components'],
               'conservative': exports['conservative'], 'candidate': exports['candidate'],
               'physical_graph_unchanged_from_cached_baseline': physical_unchanged,
               'tip_geometry_and_ownership_preserved': True, 'unsupported_components_preserved': True,
               'input_contract': 'PNG and image-model predictions plus public manufacturer pin library; no dataset source annotations/CAD/Golden reads',
               'complete_competition_JSON': False, 'official_PinF1': None, 'official_NetF1': None,
               'new_training': False, 'default_pipeline_adopted': False, 'wall_seconds': time.perf_counter()-start}
    save(output/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--pin-folder', type=Path)
    p.add_argument('--wire-cache', type=Path)
    p.add_argument('--wire-metadata', type=Path)
    p.add_argument('--baseline-network', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'configs/component_pin_library_v1.json')
    a = p.parse_args()
    provided = [a.pin_folder, a.wire_cache, a.wire_metadata, a.baseline_network]
    if any(provided) and not all(provided):
        p.error('Cached mode needs pin folder, wire cache/metadata, and baseline network together')
    if all(provided):
        folder, cache, metadata, baseline = provided
    else:
        folder = a.output/'upstream'
        subprocess.run([sys.executable, str(ROOT/'scripts/run_pin_text_v2.py'), '--image', str(a.image),
                        '--output', str(folder), '--device', a.device], check=True)
        cache = folder/'ownership/probabilities.npz'
        owner = read(folder/'ownership/summary.json')
        metadata = a.output/'wire_input_metadata.json'
        save(metadata, {'image_sha256': owner['image_sha256'], 'size': owner['size'], 'coordinate_origin': 'top_left',
             'weights_sha256': owner['unet_weights_sha256'], 'inference_config_sha256': sha(ROOT/'configs/unet_inference_v2.json'),
             'probabilities_sha256': sha(cache), 'prediction_only': True, 'source': 'Current_PNG_U-Net_inference'})
        baseline = None
    summary = process(a.image, folder, cache, metadata, a.output, read(a.config), baseline, a.device)
    summary['used_external_model_caches'] = bool(all(provided))
    save(a.output/'summary.json', summary)
    print({k: summary[k] for k in ['library', 'conservative', 'candidate', 'used_external_model_caches']}, flush=True)


if __name__ == '__main__':
    main()
