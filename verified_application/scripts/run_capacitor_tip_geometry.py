"""PNG -> S21 -> local capacitor geometry -> audited engineering output."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from run_engineering_networks import ROOT, read, save, sha, fingerprint, process as engineering_process
from pcb_parser.capacitor_tip_geometry import refine_capacitor_geometry
from pcb_parser.unet_inference import decode_pins


def process(image, stage, output, cfg, probability_paths=None, cache_metadata=None, device='cpu'):
    if stage.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(stage.resolve()):raise ValueError('Geometry output overlaps frozen S21')
    begin = time.perf_counter()
    with Image.open(image) as source:rgb = np.asarray(source.convert('RGB'))
    size = [rgb.shape[1], rgb.shape[0]];image_hash = sha(image)
    inputs = [stage/'completed_pin_text_result.json', stage/'graph_proposals.json', stage/'completed_component_records.json',
        stage/'proposal_export/partial_submission.json', stage/'proposal_export/conservative_export_coverage.json']
    digests = {str(p.resolve()): sha(p) for p in inputs};pins, graph, rp = [read(p) for p in inputs[:3]]
    for value in [pins, graph]:
        if (value.get('image_sha256') != image_hash or value.get('size') != size or not value.get('prediction_only')
                or value.get('coordinate_origin') != 'top_left'
                or value.get('whole_network_config_sha256') != fingerprint(read(ROOT/'configs/whole_network_export_v1.json'))):raise ValueError('Expected current PNG/S21 outputs')
    if rp['image_sha256'] != image_hash:raise ValueError('Records differ from PNG')
    inference = read(ROOT/'configs/unet_inference_v2.json');weights = ROOT/'weights/unet_wire_pin_v1.pt'
    if sha(weights) != inference['checkpoint_sha256']:raise ValueError('U-Net checkpoint changed')
    if probability_paths:
        meta = read(cache_metadata)
        if (meta['image_sha256'] != image_hash or meta['size'] != size or meta['coordinate_origin'] != 'top_left'
                or meta['weights_sha256'] != inference['checkpoint_sha256'] or not meta.get('prediction_only')
                or meta['scales'] != cfg['scales'] or meta['probability_sha256'] != [sha(p) for p in probability_paths]):raise ValueError('Model probability provenance mismatch')
        arrays = []
        for path in probability_paths:
            with np.load(path, allow_pickle=False) as source:arrays.append(source['probability'])
    else:
        import torch
        from pcb_parser.unet_checkpoint import load_wire_pin_checkpoint
        from pcb_parser.unet_inference import predict_at_scale
        torch.set_num_threads(4)
        model, ck, _unet_info = load_wire_pin_checkpoint(weights, device)
        arrays = [predict_at_scale(model, rgb, device, ck['config']['tile_size'], ck['config']['stride'], scale) for scale in cfg['scales']]
        probability_paths = []
        for scale, array in zip(cfg['scales'], arrays):
            path = output/f'probabilities/scale_{scale:.1f}.npz';path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probability=array);probability_paths.append(path)
        meta = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left', 'weights_sha256': sha(weights),
            'scales': cfg['scales'], 'probability_sha256': [sha(p) for p in probability_paths], 'prediction_only': True, 'source': 'Current_PNG_frozen_U-Net_three_scales'}
    candidates = decode_pins(arrays[0][1], inference['pin_threshold'], inference['pin_peak_kernel'], inference['pin_refinement'], inference['pin_nms_radius'], inference['pin_candidate_limit'])
    if (len(candidates) != len(pins['pin_candidates']) or any(math.dist(p['point'], q['point']) > .1
            or abs(p['confidence']-q['confidence']) > .002 for p, q in zip(candidates, pins['pin_candidates']))):
        raise ValueError('Native probabilities do not reproduce frozen model tips')
    updated, g, audit = refine_capacitor_geometry(pins, graph, rp['records'], *arrays, cfg, read(ROOT/'configs/engineering_network_export_v1.json'))
    # Save a distinct model-only stage. The untouched conservative JSON is
    # carried forward; the downstream engineering branch uses the corrected graph.
    corrected = output/'corrected_model_stage'
    provenance = {'image_sha256': image_hash, 'size': size, 'prediction_only': True, 'coordinate_origin': 'top_left',
        'capacitor_geometry_config_sha256': fingerprint(cfg), 'original_S21_input_sha256': digests,
        'model_probability_provenance': meta, 'points_and_local_edge_geometry_may_change': True}
    save(corrected/'completed_pin_text_result.json', {**updated, **provenance});save(corrected/'graph_proposals.json', {**g, **provenance})
    save(corrected/'completed_component_records.json', rp)
    save(corrected/'proposal_export/partial_submission.json', read(inputs[3]))
    save(corrected/'proposal_export/conservative_export_coverage.json', read(inputs[4]))
    result = engineering_process(image, corrected, output/'export', read(ROOT/'configs/engineering_network_export_v1.json'))
    if any(sha(Path(p)) != digest for p, digest in digests.items()):raise ValueError('Frozen S21 changed')
    save(output/'geometry_audit.json', {**provenance, **audit});save(output/'probability_inputs.json', meta)
    summary = {**provenance, 'geometry': audit['summary'], 'engineering_export': result['engineering_export'],
        'baseline_export': result['baseline_export'], 'completion': result['completion'], 'network_guard': result['network_guard'],
        'new_training': False, 'default_pipeline_adopted': False, 'reference_PinF1': None, 'reference_NetF1': None,
        'used_external_probability_caches': cache_metadata is not None,
        'input_contract': 'PNG and frozen image-model/OCR outputs only; no annotations/CAD/Golden reads',
        'wall_seconds': time.perf_counter()-begin}
    save(output/'summary.json', summary);return summary


def main():
    begin = time.perf_counter();p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage-result', type=Path);p.add_argument('--device', default='cpu')
    p.add_argument('--config', type=Path, default=ROOT/'configs/capacitor_tip_geometry_v1.json');a = p.parse_args()
    stage = a.stage_result or a.output.parent/(a.output.name+'_stage21')
    if a.stage_result is None:
        subprocess.run([sys.executable, str(ROOT/'scripts/run_whole_networks.py'), '--image', str(a.image), '--output', str(stage), '--device', a.device], check=True)
    result = process(a.image, stage, a.output, read(a.config), device=a.device)
    result.update(used_external_model_stage=a.stage_result is not None, entrypoint_wall_seconds=time.perf_counter()-begin)
    save(a.output/'summary.json', result);print({k: result[k] for k in ['geometry', 'engineering_export']}, flush=True)


if __name__ == '__main__':main()
