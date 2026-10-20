"""PNG -> frozen S14 image predictions -> localized power OCR -> S15 gates."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from run_power_networks import ROOT, read, save, sha, export_fields
from run_pin_output_rules import validate_cache
from run_submission_fields import fingerprint
from pcb_parser.power_networks import propose_power_networks
from pcb_parser.power_caption_ocr import recover_power_captions
from pcb_parser.pin_text_ocr import create_engine


def process(image, network_folder, pin_path, texts_path, output, cfg, engine=None, models=None):
    start = time.perf_counter()
    rgb, graph, pins, records, graph_path = validate_cache(image, network_folder, pin_path, texts_path)
    if pins.get('library_config_sha256') != fingerprint(read(ROOT/'configs/component_pin_library_v1.json')):
        raise ValueError('Expected frozen S14 input')
    power_cfg = read(ROOT/'configs/power_networks_v1.json')
    guard_cfg = read(ROOT/'configs/pin_output_rules_v1.json')
    ocr = read(texts_path)
    baseline, before = propose_power_networks(pins, graph, records, ocr['texts'], power_cfg, guard_cfg)
    if engine is None:
        engine, models = create_engine(cfg)
    enriched, texts, recovery = recover_power_captions(
        rgb, pins, graph, records, ocr['texts'], before, engine, cfg, power_cfg,
        read(ROOT/'configs/network_labels_v1.json'))
    if any(enriched[k] != graph[k] for k in graph if k != 'label_sites'):
        raise ValueError('Localized OCR changed frozen graph fields')
    if enriched['label_sites'][:len(graph['label_sites'])] != graph['label_sites']:
        raise ValueError('Original label sites changed')
    proposed, after = propose_power_networks(pins, enriched, records, texts, power_cfg, guard_cfg)
    stable = lambda g: [{k: v for k, v in p.items() if k not in {'network_id', 'alternatives'}} for p in g['pin_attachments']]
    if proposed['edges'] != graph['edges'] or stable(proposed) != stable(graph):
        raise ValueError('Power OCR changed physical geometry or pin fields')
    members = lambda g: sorted(p for n in g['networks'] for p in n['pin_candidate_ids'])
    if members(proposed) != members(graph):
        raise ValueError('Power OCR omitted/duplicated network members')
    provenance = {'image_sha256': sha(image), 'size': pins['size'], 'coordinate_origin': 'top_left',
                  'input_graph_sha256': sha(graph_path), 'input_pin_sha256': sha(pin_path),
                  'input_OCR_sha256': sha(texts_path), 'power_config_sha256': fingerprint(power_cfg),
                  'recovery_config_sha256': fingerprint(cfg), 'OCR_models': models, 'prediction_only': True}
    save(output/'recovered_texts.json', {**provenance, 'texts': texts})
    save(output/'power_caption_recovery.json', {**provenance, **recovery})
    save(output/'baseline_power_graph.json', {**baseline, **provenance})
    save(output/'power_graph_proposals.json', {**proposed, **provenance})
    save(output/'power_binding_audit.json', {**provenance, **after})
    old_export = export_fields(pins, baseline, records, output/'baseline_export')
    new_export = export_fields(pins, proposed, records, output/'proposal_export')
    summary = {**provenance, 'before_power': before['summary'], 'power': after['summary'],
               'recovery': {k: recovery[k] for k in ['roi_count', 'skipped_by_budget', 'localized_observations', 'new_power_caption_sites']},
               'baseline_export': old_export, 'proposal_export': new_export,
               'all_original_tips_fields_owners_and_physical_edges_preserved': True,
               'all_original_network_members_retained': True, 'default_pipeline_adopted': False,
               'input_contract': 'PNG and hash-checked S14 predictions only; no annotations/CAD/Golden reads',
               'new_training': False, 'official_PinF1': None, 'official_NetF1': None,
               'wall_seconds': time.perf_counter()-start}
    save(output/'summary.json', summary)
    return summary


def main():
    entry_start = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--network-folder', type=Path)
    p.add_argument('--pin-text', type=Path)
    p.add_argument('--texts', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'configs/power_caption_ocr_v1.json')
    a = p.parse_args()
    provided = [a.network_folder, a.pin_text, a.texts]
    if any(provided) and not all(provided):
        p.error('Cached mode needs network folder, pin text and OCR together')
    if all(provided):
        network, pins, texts = provided
    else:
        upstream = a.output/'upstream'
        subprocess.run([sys.executable, str(ROOT/'scripts/run_component_pin_library.py'),
                        '--image', str(a.image), '--output', str(upstream), '--device', a.device], check=True)
        network = upstream/'network'
        pins = upstream/'pin_text/pin_text_result.json'
        texts = upstream/'pin_text/texts.json'
    summary = process(a.image, network, pins, texts, a.output, read(a.config))
    summary['used_external_model_caches'] = bool(all(provided))
    summary['entrypoint_wall_seconds'] = time.perf_counter()-entry_start
    save(a.output/'summary.json', summary)
    print({k: summary[k] for k in ['before_power', 'power', 'recovery', 'proposal_export']}, flush=True)


if __name__ == '__main__':
    main()
