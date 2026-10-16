"""PNG -> S21 -> separate, audited engineering complete-net output."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from run_whole_networks import ROOT, read, save, sha, fingerprint
from run_network_labels import logical_partial_export
from run_pin_output_rules import power_network_guard, apply_power_export_gate
from pcb_parser.engineering_network_export import complete_engineering_fields
from pcb_parser.whole_network_export import net_export_guard, apply_net_export_guard
from pcb_parser.submission_fields import network_field_queue
from pcb_parser.dataset_converter import validate_output


def process(image, stage, output, cfg):
    if stage.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(stage.resolve()):
        raise ValueError('Engineering output overlaps frozen S21 inputs')
    begin = time.perf_counter()
    with Image.open(image) as source:size = list(source.size)
    image_hash = sha(image)
    paths = [stage/'completed_pin_text_result.json', stage/'graph_proposals.json',
             stage/'completed_component_records.json', stage/'proposal_export/partial_submission.json',
             stage/'proposal_export/conservative_export_coverage.json']
    digests = {str(p.resolve()): sha(p) for p in paths}
    pins, graph, rp, baseline, baseline_cov = [read(p) for p in paths]
    whole_cfg = read(ROOT/'configs/whole_network_export_v1.json');nearby_cfg = read(ROOT/'configs/nearby_tip_fields_v1.json')
    for payload in [pins, graph]:
        if (payload.get('image_sha256') != image_hash or payload.get('size') != size
                or payload.get('coordinate_origin') != 'top_left' or not payload.get('prediction_only')
                or payload.get('whole_network_config_sha256') != fingerprint(whole_cfg)
                or payload.get('nearby_tip_config_sha256') != fingerprint(nearby_cfg)):
            raise ValueError('Expected current PNG/S21 model outputs')
    if rp.get('image_sha256') != image_hash:raise ValueError('Records differ from PNG')
    records = rp['records']
    data, g, audit = complete_engineering_fields(pins, graph, records, cfg)
    payload, coverage = logical_partial_export(data, g, records)
    power = power_network_guard(data, g, records, read(ROOT/'configs/pin_output_rules_v1.json'))
    payload, coverage = apply_power_export_gate(payload, coverage, power)
    for net in g['networks']:
        if not net.get('power_logical_merge'):continue
        nid = net['network_id'];payload['nets'].pop(nid, None)
        row = next((n for n in coverage['withheld_networks'] if n['network_id'] == nid), None)
        if row is None:row = {'network_id': nid, 'reasons': []};coverage['withheld_networks'].append(row)
        row['reasons'] = sorted(set(row['reasons']+['power_logical_membership_provisional']))
    guard = net_export_guard(data, g, records, whole_cfg, nearby_cfg)
    payload, coverage = apply_net_export_guard(payload, coverage, guard, data, g)
    # A new convention must not alter or lose any established whole net.
    if any(payload['nets'].get(nid) != net for nid, net in baseline['nets'].items()):
        raise ValueError('Engineering convention changed an established whole net')
    for out in [baseline, payload]:
        errors, bounds = validate_output(out, *size)
        if errors or bounds:raise ValueError('Engineering export violates local JSON contract')
    for old, new in zip(pins['pin_candidates'], data['pin_candidates']):
        if old['candidate_id'] != new['candidate_id'] or old['point'] != new['point']:raise ValueError('Model tip moved or removed')
        for field in ['number', 'pinname']:
            if old.get(field) is not None and old[field] != new[field]:raise ValueError('Existing recognized field changed')
    if (graph['edges'] != g['edges'] or graph['physical_networks'] != g['physical_networks']
            or [(n['network_id'], n['pin_candidate_ids']) for n in graph['networks']]
                != [(n['network_id'], n['pin_candidate_ids']) for n in g['networks']]):
        raise ValueError('Engineering export changed model topology')
    if any(sha(Path(path)) != digest for path, digest in digests.items()):raise ValueError('Frozen S21 input changed')
    pmap = {p['candidate_id']: p for p in data['pin_candidates']};net_provenance = []
    for proof in coverage['atomic_complete_network_proofs']:
        members = []
        for cid in proof['all_original_member_ids']:
            pin = pmap[proof['member_to_contact'][cid]]
            members.append({'original_candidate_id': cid, 'contact_candidate_id': pin['candidate_id'],
                'pin_reference': f"{pin['owner']}.{pin['number']}",
                'number_source': pin.get('text_evidence', {}).get('number', {}).get('source', pin.get('number_status')),
                'name_source': pin.get('text_evidence', {}).get('pinname', {}).get('source', pin.get('pinname_status')),
                'engineering_fields': [f for f in ['number', 'pinname'] if pin.get(f+'_status') == 'engineering_convention_unverified']})
        net_provenance.append({**proof, 'members': members, 'contains_engineering_conventions': any(m['engineering_fields'] for m in members)})
    provenance = {'image_sha256': image_hash, 'size': size, 'coordinate_origin': 'top_left', 'prediction_only': True,
        'engineering_config_sha256': fingerprint(cfg), 'input_sha256': digests,
        'profile': 'engineering_only_provisional_numbering', 'official_numbering_verified': False}
    coverage.update(profile=provenance['profile'], engineering_generated_pin_numbers=audit['summary']['generated_pin_numbers'],
                    engineering_generated_pin_names=audit['summary']['generated_pin_names'],
                    blank_pin_names_guessed=audit['summary']['generated_pin_names'],
                    official_numbering_verified=False, official_evaluator_verified=False)
    save(output/'baseline_export/partial_submission.json', baseline)
    save(output/'baseline_export/conservative_export_coverage.json', baseline_cov)
    save(output/'engineering_submission.json', payload);save(output/'engineering_export_coverage.json', coverage)
    save(output/'engineering_pin_text_result.json', {**data, **provenance})
    save(output/'engineering_graph_proposals.json', {**g, **provenance})
    save(output/'completed_component_records.json', rp)
    save(output/'engineering_field_audit.json', {**provenance, **audit})
    save(output/'whole_network_audit.json', {**provenance, **guard})
    save(output/'field_provenance.json', {**provenance, 'networks': net_provenance})
    queue = network_field_queue(data, g, records, coverage);save(output/'engineering_completion_queue.json', queue)
    result = {**provenance, 'engineering_fields': audit['summary'],
        'baseline_export': {k: baseline_cov[k] for k in ['exported_components', 'exported_pins', 'exported_nets']},
        'engineering_export': {k: coverage[k] for k in ['exported_components', 'exported_pins', 'exported_nets']},
        'new_complete_network_ids': sorted(set(payload['nets'])-set(baseline['nets'])),
        'completion': {k: queue[k] for k in ['multi_pin_networks', 'field_only_incomplete_networks', 'non_field_export_blocker_counts']},
        'network_guard': guard['summary'], 'original_points_edges_and_members_preserved': True,
        'all_existing_numbers_names_and_conservative_output_preserved': True,
        'new_OCR_observations': 0, 'new_training': False, 'default_pipeline_adopted': False,
        'complete_competition_JSON': False, 'official_PinF1': None, 'official_NetF1': None,
        'input_contract': 'PNG and S21 model/OCR predictions only; no annotations/CAD/Golden reads',
        'wall_seconds': time.perf_counter()-begin}
    save(output/'summary.json', result);return result


def main():
    begin = time.perf_counter();p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage-result', type=Path);p.add_argument('--device', default='cpu')
    p.add_argument('--config', type=Path, default=ROOT/'configs/engineering_network_export_v1.json');a = p.parse_args()
    stage = a.stage_result or a.output.parent/(a.output.name+'_stage21')
    if a.stage_result is None:
        subprocess.run([sys.executable, str(ROOT/'scripts/run_whole_networks.py'), '--image', str(a.image),
                        '--output', str(stage), '--device', a.device], check=True)
    result = process(a.image, stage, a.output, read(a.config))
    result.update(used_external_model_caches=a.stage_result is not None, entrypoint_wall_seconds=time.perf_counter()-begin)
    save(a.output/'summary.json', result)
    print({k: result[k] for k in ['engineering_fields', 'baseline_export', 'engineering_export', 'new_complete_network_ids']}, flush=True)


if __name__ == '__main__':main()
