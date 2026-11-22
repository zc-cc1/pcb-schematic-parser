"""PNG -> S20 -> observed interface IDs -> atomic whole-network export."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from run_nearby_tip_fields import ROOT, read, save, sha, fingerprint, export_fields
from pcb_parser.whole_network_export import recover_interface_titles, net_export_guard, apply_net_export_guard
from pcb_parser.pin_output_rules import complete_pin_output_names, compress_candidate_edges
from pcb_parser.submission_fields import network_field_queue, validate_fields
from pcb_parser.dataset_converter import validate_output


def process(image, stage, texts_path, output, cfg):
    if (stage.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(stage.resolve())
            or texts_path.resolve().is_relative_to(output.resolve())):
        raise ValueError('Whole-network output overlaps frozen inputs')
    begin = time.perf_counter()
    with Image.open(image) as source:size = list(source.size)
    image_hash = sha(image)
    paths = [stage/'completed_pin_text_result.json', stage/'graph_proposals.json', stage/'completed_component_records.json', texts_path]
    digests = {str(p.resolve()): sha(p) for p in paths}
    pins, graph, records_payload, ocr = [read(p) for p in paths]
    nearby_cfg = read(ROOT/'configs/nearby_tip_fields_v1.json')
    for p in [pins, graph, ocr]:
        if (p.get('image_sha256') != image_hash or p.get('size') != size
                or p.get('coordinate_origin') != 'top_left' or not p.get('prediction_only')):
            raise ValueError('Model/OCR inputs differ from PNG')
    for p in [pins, graph]:
        if p.get('nearby_tip_config_sha256') != fingerprint(nearby_cfg):raise ValueError('Expected current S20 inputs')
    if records_payload.get('image_sha256') != image_hash:raise ValueError('Records differ from PNG')
    records = records_payload['records'];validate_fields(pins, graph, records)
    baseline = export_fields(pins, graph, records, output/'baseline_export', fields_completed=True)
    updated, new_graph, new_records, identity = recover_interface_titles(pins, graph, records, ocr['texts'], cfg)
    guard = net_export_guard(updated, new_graph, new_records, cfg, nearby_cfg)
    export_fields(updated, new_graph, new_records, output/'proposal_export', fields_completed=True)
    strict, sg, candidate, cg, _ = complete_pin_output_names(updated, new_graph, new_records, read(ROOT/'configs/pin_output_rules_v1.json'))
    exports = {};queues = {}
    for profile, data, g, filename in [('conservative', strict, sg, 'partial_submission.json'), ('candidate', candidate, cg, 'submission_candidate.json')]:
        payload, coverage = read(output/f'proposal_export/{filename}'), read(output/f'proposal_export/{profile}_export_coverage.json')
        payload, coverage = apply_net_export_guard(payload, coverage, guard, data, g)
        if profile == 'candidate':payload, _ = compress_candidate_edges(payload, g, size[1], 1)
        errors, bounds = validate_output(payload, *size)
        if errors or bounds:raise ValueError('Whole-network export violates local JSON contract')
        save(output/f'proposal_export/{filename}', payload);save(output/f'proposal_export/{profile}_export_coverage.json', coverage)
        queue = network_field_queue(data, g, new_records, coverage);save(output/f'{profile}_completion_queue.json', queue)
        if profile == 'conservative':save(output/'proposal_export/network_field_queue.json', queue)
        queues[profile] = {k: queue[k] for k in ['multi_pin_networks', 'field_only_incomplete_networks', 'non_field_export_blocker_counts']}
        exports[profile] = {k: coverage[k] for k in ['exported_components', 'exported_pins', 'exported_nets']}
    if any(sha(Path(p)) != digest for p, digest in digests.items()):raise ValueError('Frozen input changed')
    if any(p['point'] != q['point'] or p.get('number') != q.get('number') or p.get('pinname') != q.get('pinname')
           or p['owner_detection_index'] != q['owner_detection_index'] for p, q in zip(pins['pin_candidates'], updated['pin_candidates'])):
        raise ValueError('Whole-network stage changed protected pin fields')
    if graph['edges'] != new_graph['edges'] or [n['pin_candidate_ids'] for n in graph['networks']] != [n['pin_candidate_ids'] for n in new_graph['networks']]:
        raise ValueError('Whole-network stage changed model topology')
    provenance = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left', 'prediction_only': True,
                  'input_pin_sha256': digests[str(paths[0].resolve())], 'input_graph_sha256': digests[str(paths[1].resolve())],
                  'input_records_sha256': digests[str(paths[2].resolve())], 'input_OCR_sha256': digests[str(texts_path.resolve())],
                  'whole_network_config_sha256': fingerprint(cfg)}
    save(output/'completed_pin_text_result.json', {**strict, **provenance})
    save(output/'candidate_pin_text_result.json', {**candidate, **provenance})
    save(output/'completed_component_records.json', {'image_sha256': image_hash, 'records': new_records})
    save(output/'graph_proposals.json', {**sg, **provenance})
    save(output/'interface_identity_audit.json', {**provenance, **identity});save(output/'whole_network_audit.json', {**provenance, **guard})
    result = {**provenance, 'identity_recovery': {'new_identities': identity['new_identities'], 'generated_designators_or_suffixes': 0},
              'network_guard': guard['summary'], 'baseline_export': baseline, 'proposal_export': exports, 'completion': queues,
              'all_original_tip_points_numbers_names_owner_indexes_edges_and_members_preserved': True,
              'default_pipeline_adopted': False, 'complete_target_JSON': False, 'reference_PinF1': None, 'reference_NetF1': None,
              'input_contract': 'PNG and S20 model/OCR outputs only; no annotations/CAD/Golden reads',
              'wall_seconds': time.perf_counter()-begin}
    save(output/'summary.json', result);return result


def main():
    begin = time.perf_counter();parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True);parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-result', type=Path);parser.add_argument('--texts', type=Path);parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/whole_network_export_v1.json');args = parser.parse_args()
    if bool(args.stage_result) != bool(args.texts):parser.error('Cached mode needs stage result and OCR texts together')
    stage = args.stage_result
    if stage is None:
        stage = args.output.parent/(args.output.name+'_stage20')
        subprocess.run([sys.executable, str(ROOT/'scripts/run_nearby_tip_fields.py'), '--image', str(args.image),
                        '--output', str(stage), '--device', args.device], check=True)
        texts = stage.parent/(stage.name+'_stage18')/'stage17/stage16/upstream/pin_text/texts.json'
    else:texts = args.texts
    result = process(args.image, stage, texts, args.output, read(args.config))
    result.update(used_external_model_caches=args.stage_result is not None, entrypoint_wall_seconds=time.perf_counter()-begin)
    save(args.output/'summary.json', result);print({k: result[k] for k in ['identity_recovery', 'network_guard', 'proposal_export']}, flush=True)


if __name__ == '__main__':main()
