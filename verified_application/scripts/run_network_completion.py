"""S16 predictions -> passive number role filter and complete export triage.

Cached mode checks the original S14 model inputs against S16 provenance.
PNG mode executes the S16 image pipeline first. No paired annotations are read.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from run_power_caption_ocr import ROOT, read, save, sha, export_fields
from run_pin_output_rules import validate_cache
from run_submission_fields import fingerprint
from pcb_parser.submission_fields import complete_fields, network_field_queue
from pcb_parser.pin_output_rules import complete_pin_output_names
from pcb_parser.pin_number_validation import withhold_package_numbers


def process(image, stage_folder, upstream_folder, output, cfg):
    start = time.perf_counter()
    original_pin = upstream_folder/'pin_text/pin_text_result.json'
    original_text = upstream_folder/'pin_text/texts.json'
    rgb, original_graph, pins, records, graph_path = validate_cache(
        image, upstream_folder/'network', original_pin, original_text)
    upstream_summary = read(upstream_folder/'summary.json')
    if (upstream_summary.get('records_input_sha256') != sha(upstream_folder/'pin_text/component_records.json')
            or pins.get('library_config_sha256') != fingerprint(read(ROOT/'configs/component_pin_library_v1.json'))):
        raise ValueError('S14 component records or pin-library configuration changed')
    summary = read(stage_folder/'summary.json')
    stage_graph_path = stage_folder/'power_graph_proposals.json'
    graph = read(stage_graph_path)
    for payload in [summary, graph]:
        if (payload.get('image_sha256') != sha(image) or payload.get('size') != pins['size']
                or payload.get('coordinate_origin') != 'top_left' or not payload.get('prediction_only')
                or payload.get('input_pin_sha256') != sha(original_pin)
                or payload.get('input_OCR_sha256') != sha(original_text)
                or payload.get('input_graph_sha256') != sha(graph_path)
                or payload.get('power_config_sha256') != fingerprint(read(ROOT/'configs/power_networks_v1.json'))
                or payload.get('recovery_config_sha256') != fingerprint(read(ROOT/'configs/power_caption_ocr_v1.json'))):
            raise ValueError('S16 cache provenance differs from PNG/S14/configs')
    stable = lambda g: [{k: v for k, v in p.items() if k not in {'network_id', 'alternatives'}} for p in g['pin_attachments']]
    if graph['edges'] != original_graph['edges'] or stable(graph) != stable(original_graph):
        raise ValueError('S16 graph changed original pin fields or wire geometry')
    members = lambda g: sorted(p for n in g['networks'] for p in n['pin_candidate_ids'])
    if members(graph) != members(original_graph):
        raise ValueError('S16 graph omitted or duplicated members')
    fields, field_graph, completed_records, _ = complete_fields(
        pins, graph, records, read(ROOT/'configs/submission_fields_v1.json'))
    filtered, filtered_graph, audit = withhold_package_numbers(fields, field_graph, completed_records, cfg)
    baseline = export_fields(pins, graph, records, output/'baseline_export')
    exports = export_fields(filtered, filtered_graph, completed_records, output/'proposal_export', fields_completed=True)
    strict, strict_graph, candidate, candidate_graph, _ = complete_pin_output_names(
        filtered, filtered_graph, completed_records, read(ROOT/'configs/pin_output_rules_v1.json'))
    queues = {}
    for profile, data, g in [('conservative', strict, strict_graph), ('candidate', candidate, candidate_graph)]:
        coverage = read(output/f'proposal_export/{profile}_export_coverage.json')
        queues[profile] = network_field_queue(data, g, completed_records, coverage)
        save(output/f'{profile}_completion_queue.json', queues[profile])
    provenance = {'image_sha256': sha(image), 'size': pins['size'], 'coordinate_origin': 'top_left',
                  'input_pin_sha256': sha(original_pin), 'input_OCR_sha256': sha(original_text),
                  'input_records_sha256': sha(upstream_folder/'pin_text/component_records.json'),
                  'input_stage_graph_sha256': sha(stage_graph_path),
                  'pin_role_config_sha256': fingerprint(cfg), 'prediction_only': True}
    save(output/'completed_pin_text_result.json', {**strict, **provenance})
    save(output/'candidate_pin_text_result.json', {**candidate, **provenance})
    save(output/'completed_component_records.json', {'image_sha256': sha(image), 'records': completed_records})
    save(output/'pin_number_role_audit.json', {**provenance, **audit})
    save(output/'graph_proposals.json', {**strict_graph, **provenance})
    result = {**provenance, 'number_role_validation': audit['summary'], 'baseline_export': baseline,
              'proposal_export': exports, 'completion': {profile: {key: queues[profile][key] for key in
                  ['multi_pin_networks', 'field_only_incomplete_networks', 'non_field_export_blocker_counts']}
                  for profile in queues},
              'tip_points_ownership_and_all_network_members_preserved': True,
              'default_pipeline_adopted': False, 'reference_PinF1': None, 'reference_NetF1': None,
              'input_contract': 'PNG and integrity-checked S14/S16 image predictions; no annotations/CAD/Golden reads',
              'wall_seconds': time.perf_counter()-start}
    save(output/'summary.json', result)
    return result


def main():
    begin = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--stage-result', type=Path)
    p.add_argument('--upstream-result', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'configs/pin_number_validation_v1.json')
    a = p.parse_args()
    supplied = [a.stage_result, a.upstream_result]
    if any(supplied) and not all(supplied):
        p.error('Cached mode needs both S16 stage and S14 upstream result folders')
    if all(supplied):
        stage, upstream = supplied
    else:
        stage = a.output/'stage16'
        subprocess.run([sys.executable, str(ROOT/'scripts/run_power_caption_ocr.py'),
                        '--image', str(a.image), '--output', str(stage), '--device', a.device], check=True)
        upstream = stage/'upstream'
    result = process(a.image, stage, upstream, a.output, read(a.config))
    result.update(used_external_model_caches=bool(all(supplied)), entrypoint_wall_seconds=time.perf_counter()-begin)
    save(a.output/'summary.json', result)
    print({k: result[k] for k in ['number_role_validation', 'proposal_export', 'completion']}, flush=True)


if __name__ == '__main__':
    main()
