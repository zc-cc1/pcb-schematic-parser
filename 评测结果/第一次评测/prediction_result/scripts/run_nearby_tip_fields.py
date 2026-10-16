"""PNG -> S18 -> same-lead OCR name reconciliation, with no reference reads."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from run_targeted_pin_ocr import ROOT, read, save, sha, export_fields, fingerprint
from pcb_parser.nearby_tip_fields import reconcile_nearby_tip_fields
from pcb_parser.submission_fields import network_field_queue, validate_fields
from pcb_parser.pin_output_rules import complete_pin_output_names


def process(image, stage, output, cfg):
    if output.resolve() == stage.resolve() or stage.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(stage.resolve()):
        raise ValueError('S20 output must be separate from S18 input')
    begin = time.perf_counter()
    with Image.open(image) as source:
        size = list(source.size)
    image_hash = sha(image)
    pin_path, graph_path = stage/'completed_pin_text_result.json', stage/'graph_proposals.json'
    record_path = stage/'completed_component_records.json'
    pins, graph, record_payload = read(pin_path), read(graph_path), read(record_path)
    for payload in [pins, graph]:
        if (payload.get('image_sha256') != image_hash or payload.get('size') != size
                or payload.get('coordinate_origin') != 'top_left' or not payload.get('prediction_only')
                or payload.get('targeted_OCR_config_sha256') != fingerprint(read(ROOT/'configs/targeted_pin_ocr_v1.json'))
                or payload.get('pin_role_config_sha256') != fingerprint(read(ROOT/'configs/pin_number_validation_v1.json'))):
            raise ValueError('S18 input differs from PNG/configuration')
    if record_payload.get('image_sha256') != image_hash:
        raise ValueError('Component records differ from PNG')
    records = record_payload['records']
    validate_fields(pins, graph, records)
    input_hashes = {str(path.resolve()): sha(path) for path in [pin_path, graph_path, record_path]}
    baseline = export_fields(pins, graph, records, output/'baseline_export', fields_completed=True)
    updated, new_graph, audit = reconcile_nearby_tip_fields(pins, graph, records, cfg)
    exports = export_fields(updated, new_graph, records, output/'proposal_export', fields_completed=True)
    strict, sg, candidate, cg, _ = complete_pin_output_names(updated, new_graph, records, read(ROOT/'configs/pin_output_rules_v1.json'))
    queues = {}
    for profile, payload, g in [('conservative', strict, sg), ('candidate', candidate, cg)]:
        queue = network_field_queue(payload, g, records, read(output/f'proposal_export/{profile}_export_coverage.json'))
        save(output/f'{profile}_completion_queue.json', queue)
        queues[profile] = {k: queue[k] for k in ['multi_pin_networks', 'field_only_incomplete_networks', 'non_field_export_blocker_counts']}
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError('S18 input bytes changed during reconciliation')
    provenance = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left',
                  'input_pin_sha256': sha(pin_path), 'input_graph_sha256': sha(graph_path),
                  'input_records_sha256': sha(record_path), 'nearby_tip_config_sha256': fingerprint(cfg),
                  'prediction_only': True}
    save(output/'field_link_audit.json', {**provenance, **audit})
    save(output/'completed_pin_text_result.json', {**strict, **provenance})
    save(output/'candidate_pin_text_result.json', {**candidate, **provenance})
    save(output/'graph_proposals.json', {**sg, **provenance})
    save(output/'completed_component_records.json', record_payload)
    result = {**provenance, 'reconciliation': audit['summary'], 'baseline_export': baseline, 'proposal_export': exports,
              'completion': queues, 'original_points_owners_fields_edges_and_members_preserved': True,
              'new_OCR_observations': 0, 'new_training': False, 'default_pipeline_adopted': False,
              'official_PinF1': None, 'official_NetF1': None, 'wall_seconds': time.perf_counter()-begin,
              'input_contract': 'PNG and S18 model/OCR predictions only; no annotation/CAD/Golden reads'}
    save(output/'summary.json', result)
    return result


def main():
    begin = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-result', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/nearby_tip_fields_v1.json')
    args = parser.parse_args()
    stage = args.stage_result or args.output.parent/(args.output.name+'_stage18')
    if args.stage_result is None:
        subprocess.run([sys.executable, str(ROOT/'scripts/run_targeted_pin_ocr.py'), '--image', str(args.image),
                        '--output', str(stage), '--device', args.device], check=True)
    result = process(args.image, stage, args.output, read(args.config))
    result.update(used_external_model_caches=args.stage_result is not None, entrypoint_wall_seconds=time.perf_counter()-begin)
    save(args.output/'summary.json', result)
    print({k: result[k] for k in ['reconciliation', 'baseline_export', 'proposal_export']}, flush=True)


if __name__ == '__main__':
    main()
