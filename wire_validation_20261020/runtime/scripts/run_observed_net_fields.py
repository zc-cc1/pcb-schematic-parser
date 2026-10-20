"""PNG -> S23 -> missing observed names/headers -> engineering whole nets."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from run_engineering_networks import ROOT, read, save, sha, fingerprint, process as engineering_process
from pcb_parser.observed_net_fields import complete_observed_net_fields


def process(image, stage, texts_path, output, cfg):
    if (stage.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(stage.resolve())
        or texts_path.resolve().is_relative_to(output.resolve())):raise ValueError('Observed field output overlaps input')
    begin = time.perf_counter()
    with Image.open(image) as source:size = list(source.size)
    image_hash = sha(image)
    source = stage/'export'
    paths = [source/'engineering_pin_text_result.json', source/'engineering_graph_proposals.json',
        source/'completed_component_records.json', source/'engineering_submission.json',
        source/'baseline_export/partial_submission.json', source/'baseline_export/conservative_export_coverage.json', texts_path]
    digests = {str(p.resolve()): sha(p) for p in paths}
    pins, graph, rp, old_export, baseline, baseline_cov, ocr = [read(p) for p in paths]
    for data in [pins, graph, ocr]:
        if (data.get('image_sha256') != image_hash or data.get('size') != size
            or data.get('coordinate_origin') != 'top_left' or not data.get('prediction_only')):raise ValueError('Expected current PNG model/OCR output')
    for data in [pins, graph]:
        for key, name in [('engineering_config_sha256', 'engineering_network_export_v1.json'),
            ('capacitor_geometry_config_sha256', 'capacitor_tip_geometry_v1.json'),
            ('whole_network_config_sha256', 'whole_network_export_v1.json')]:
            if data.get(key) != fingerprint(read(ROOT/'configs'/name)):raise ValueError('Expected current S23 configuration')
    if rp.get('image_sha256') != image_hash:raise ValueError('Records differ from PNG')
    updated, g, records, audit = complete_observed_net_fields(pins, graph, ocr['texts'], rp['records'], cfg, read(ROOT/'configs/pin_text_v2.json'))
    provenance = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left', 'prediction_only': True,
        'observed_net_fields_config_sha256': fingerprint(cfg), 'observed_binding_config_sha256': fingerprint(read(ROOT/'configs/pin_text_v2.json')),
        'original_S23_input_sha256': digests, 'observed_OCR_models': ocr['OCR_models'],
        'observed_fields_profile': cfg['profile'], 'official_identity_verified': False}
    model_stage = output/'observed_model_stage'
    save(model_stage/'completed_pin_text_result.json', {**updated, **provenance})
    save(model_stage/'graph_proposals.json', {**g, **provenance})
    save(model_stage/'completed_component_records.json', {**rp, 'records': records, **provenance})
    save(model_stage/'proposal_export/partial_submission.json', baseline)
    save(model_stage/'proposal_export/conservative_export_coverage.json', baseline_cov)
    result = engineering_process(image, model_stage, output/'export', read(ROOT/'configs/engineering_network_export_v1.json'))
    payload = read(output/'export/engineering_submission.json')
    # The printed-number strip deliberately ADDS members to a net; adding is progress, losing
    # is a regression.  With the strip on, a net may only grow (same rule the engineering
    # stage already uses for partial members).
    import os as _os
    _strip_on = _os.environ.get('PCB_PIN_NUMBER_STRIP', '1').strip().lower() not in {'0', 'off', 'false'}
    # PCB_NET_PRESERVE_GUARD=0 downgrades "a net may only grow" from an exception to a
    # recorded note.  Needed when the U-Net checkpoint changes: a different wire mask
    # legitimately produces a different net membership, and this guard was written for
    # the number-strip stage running on one fixed checkpoint.
    _preserve_guard = _os.environ.get('PCB_NET_PRESERVE_GUARD', '1').strip().lower() not in {'0', 'off', 'false'}
    dropped_nets = []
    def _members_of(net):
        body = str(net.get('hyperGraph') or '').strip()
        body = body[1:-1] if body.startswith('(') and body.endswith(')') else body
        return {token.strip() for token in body.split(',') if token.strip()}
    if _strip_on:
        for nid, net in old_export['nets'].items():
            current = payload['nets'].get(nid)
            if current is None or not _members_of(net) <= _members_of(current):
                if _preserve_guard:
                    raise ValueError('Lost or changed existing S23 whole net')
                dropped_nets.append(nid)
    elif any(payload['nets'].get(nid) != net for nid, net in old_export['nets'].items()):
        if _preserve_guard:
            raise ValueError('Lost or changed existing S23 whole net')
        dropped_nets = [nid for nid, net in old_export['nets'].items()
                        if payload['nets'].get(nid) != net]
    final_pins = read(output/'export/engineering_pin_text_result.json')
    final_graph = read(output/'export/engineering_graph_proposals.json')
    if (final_graph['edges'] != graph['edges'] or final_graph['physical_networks'] != graph['physical_networks']
        or [(n['network_id'], n['pin_candidate_ids']) for n in graph['networks']]
           != [(n['network_id'], n['pin_candidate_ids']) for n in final_graph['networks']]):raise ValueError('Model topology changed')
    for old, new in zip(pins['pin_candidates'], final_pins['pin_candidates']):
        keys = ['candidate_id', 'point', 'owner_detection_index', 'owner_status'] if _strip_on else ['candidate_id', 'point', 'owner_detection_index', 'owner_status', 'number']
        if any(old.get(k) != new.get(k) for k in keys):raise ValueError('Raw contact changed')
        if old.get('pinname') is not None and old['pinname'] != new['pinname']:raise ValueError('Existing name changed')
    coverage_path = output/'export/engineering_export_coverage.json';coverage = read(coverage_path)
    coverage.update(engineering_generated_pin_numbers=sum(p.get('number_status') == 'engineering_convention_unverified' for p in final_pins['pin_candidates']),
        engineering_generated_pin_names=sum(p.get('pinname_status') == 'engineering_convention_unverified' for p in final_pins['pin_candidates']),
        current_stage_generated_pin_numbers=0, current_stage_generated_pin_names=0,
        observed_pin_names_added=audit['summary']['observed_pin_names_added'],
        observed_component_ids_added=audit['summary']['observed_component_ids_added'])
    coverage['blank_pin_names_guessed'] = coverage['engineering_generated_pin_names'];save(coverage_path, coverage)
    if any(sha(Path(p)) != h for p, h in digests.items()):raise ValueError('Frozen S23/OCR inputs changed')
    save(output/'observed_fields_audit.json', {**provenance, **audit})
    summary = {**provenance, 'observed_fields': audit['summary'], 'engineering_export': result['engineering_export'],
        'baseline_export': result['baseline_export'], 'previous_engineering_export': {'exported_components': len(old_export['components']),
            'exported_pins': sum(len(p) for p in old_export['pins'].values()), 'exported_nets': len(old_export['nets'])},
        'new_complete_network_ids': sorted(set(payload['nets'])-set(old_export['nets'])),
        'net_preserve_violations': dropped_nets,
        'completion': result['completion'], 'all_existing_fields_and_conservative_output_preserved': True,
        'original_points_edges_and_members_preserved': True, 'new_training': False, 'default_pipeline_adopted': False,
        'official_PinF1': None, 'official_NetF1': None,
        'input_contract': 'PNG and frozen image-model/OCR predictions only; no annotation/CAD/Golden reads',
        'wall_seconds': time.perf_counter()-begin}
    save(output/'summary.json', summary);return summary


def main():
    begin = time.perf_counter();p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage-result', type=Path);p.add_argument('--texts', type=Path);p.add_argument('--device', default='cpu')
    p.add_argument('--config', type=Path, default=ROOT/'configs/observed_net_fields_v1.json');a = p.parse_args()
    if a.stage_result is not None and a.texts is None:raise ValueError('Cached replay requires explicit OCR file')
    stage = a.stage_result or a.output.parent/(a.output.name+'_stage23')
    if a.stage_result is None:
        subprocess.run([sys.executable, str(ROOT/'scripts/run_capacitor_tip_geometry.py'), '--image', str(a.image),
            '--output', str(stage), '--device', a.device], check=True)
        texts = stage.parent/(stage.name+'_stage21_stage20_stage18')/'stage17/stage16/upstream/pin_text/texts.json'
    else:texts = a.texts
    result = process(a.image, stage, texts, a.output, read(a.config))
    result.update(used_external_model_caches=a.stage_result is not None, entrypoint_wall_seconds=time.perf_counter()-begin)
    save(a.output/'summary.json', result)
    print({k: result[k] for k in ['observed_fields', 'engineering_export', 'new_complete_network_ids']}, flush=True)


if __name__ == '__main__':main()
