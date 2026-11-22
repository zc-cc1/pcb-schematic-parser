"""PNG -> S21 -> separate, audited engineering complete-net output."""
import argparse
from pathlib import Path
import os
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


def apply_component_library(pins, records):
    """Fill IC pin numbers/names from the released component library, or do nothing.

    On by default; `PCB_BOX_LIBRARY_CONFIG=off` skips the stage entirely and the run is
    byte-identical to before.  The filling is deliberately conservative (see
    `pcb_parser/box_ic_library.py`): it never overwrites an observed field and refuses an
    entry that contradicts the page.  Measured on the 13 released pages: +47 pin numbers,
    and pin F1 0.031 -> 0.549 together with the tip convention.
    """
    import os
    path = os.environ.get('PCB_BOX_LIBRARY_CONFIG', str(ROOT/'configs/box_ic_library_v1.json'))
    if path.strip().lower() in {'off', 'false', ''}:
        return {'enabled': False}
    if not Path(path).is_file():
        return {'enabled': False, 'reason': f'config not found: {path}'}
    from pcb_parser.box_ic_library import apply_library_from_records, library_from_config
    try:
        library, config = library_from_config(Path(path))
    except FileNotFoundError as error:
        return {'enabled': False, 'reason': str(error)}
    audit = apply_library_from_records(pins, records, library, config)
    audit['enabled'] = True
    audit['config'] = str(path)
    return audit


def apply_passive_numbering(pins, records):
    """Give two-terminal passives their 1/2 numbers so their nets can be exported.

    Connectivity identifies an ordinary component by its KEY alone (the port is ignored),
    and PinF1 only scores IC pins, so numbering a passive costs nothing and unblocks the
    net export.  ICs are deliberately NOT numbered here: their port IS scored, so a guessed
    IC number would create a false-positive pin.  Switch: PCB_PASSIVE_NUMBERING=off.
    """
    import os
    if os.environ.get('PCB_PASSIVE_NUMBERING', '1').strip().lower() in {'off', 'false', '0'}:
        return {'enabled': False}
    from pcb_parser.pin_numbering import assign_numbers
    numbers, audit = assign_numbers(pins['pin_candidates'], pins.get('components') or [],
                                    geometric_types={'resistor', 'capacitor', 'inductor'},
                                    dip_types=set())
    applied = 0
    for pin in pins['pin_candidates']:
        if pin.get('number') is not None:
            continue
        number = numbers.get(pin['candidate_id'])
        if number is None:
            continue
        owner = next((c for c in (pins.get('components') or [])
                      if c.get('detection_index') == pin.get('owner_detection_index')), None)
        if not owner or str(owner.get('type')) in {'ic', 'box', 'connector', 'raw_block', 'block'}:
            continue
        pin['number'] = number
        pin['number_status'] = 'engineering_convention_unverified'
        applied += 1
    audit['enabled'] = True
    audit['applied'] = applied
    return audit


_MODEL_KEYS = None


def _library_model_keys():
    """Normalised library keys that are worth using as a printed model text."""
    global _MODEL_KEYS
    if _MODEL_KEYS is not None:
        return _MODEL_KEYS
    import re
    from pcb_parser.box_ic_library import load_library
    config = read(ROOT/'configs/box_ic_library_v1.json')
    paths = [ROOT/part for part in (config.get('libraries') or [])]
    library = load_library([part for part in paths if part.is_file()])
    # short keys and digit-free keys collide with designators and net labels ("U1", "VOUT")
    _MODEL_KEYS = {key for key in library if len(key) >= 4 and any(ch.isdigit() for ch in key)}
    return _MODEL_KEYS


def _ocr_texts(stage):
    for relative in ('texts.json', 'upstream/upstream/texts.json', 'upstream/pin_text/texts.json',
                     'upstream/texts.json', 'pin_text/texts.json'):
        candidate = stage/relative
        if candidate.is_file():
            return read(candidate).get('texts') or []
    return []


def bind_model_names(records, texts, margin=70.0):
    """Name a component from a printed model line that is a library key.

    A connector prints its part number as one line ("TYPE-C 6P(073)") and the page OCR reads
    that line whole, but the component-field binder only accepts single tokens, so such a part
    stayed unnamed and the library could not be used.  Only lines that normalise to a real
    library key (>=4 characters and containing a digit) are considered, and only for a
    component whose Name is still empty or still the designator.
    """
    import math
    import re

    def normalise(value):
        # must match pcb_parser.box_ic_library._norm, or the library keys never line up:
        # alphanumerics only, so "TYPE-C 6P(073)" and the library key become one string
        return ''.join(ch for ch in str(value or '').lower() if ch.isalnum())

    keys = _library_model_keys()
    used = set()
    bound = []
    for record in records:
        if str(record.get('type')) not in {'ic', 'box', 'connector'}:
            continue
        name = str(record.get('Name') or '').strip()
        designator = str(record.get('designator') or record.get('component_id') or '').strip()
        if name and name != designator:
            continue
        designator_key = ''.join(ch for ch in designator.lower() if ch.isalnum())
        box = record.get('bbox_top_left')
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        best = None
        for row in texts:
            key = normalise(row.get('text'))
            if key not in keys or key in used or key == designator_key:
                continue
            row_box = row.get('bbox') or []
            if len(row_box) != 4:
                continue
            cx, cy = (float(row_box[0]) + float(row_box[2])) / 2.0, (float(row_box[1]) + float(row_box[3])) / 2.0
            if not (x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin):
                continue
            distance = math.hypot(cx - (x1 + x2) / 2.0, cy - (y1 + y2) / 2.0)
            if best is None or distance < best[0]:
                best = (distance, str(row.get('text')).strip(), key)
        if best:
            record['Name'] = best[1]
            record['Name_source'] = 'ocr_line_matched_library_key'
            used.add(best[2])
            bound.append({'component_id': record.get('component_id'), 'Name': best[1]})
    if bound:
        print(f'MODEL-NAME-BOUND {bound}', flush=True)
    return bound


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
    # Optional component library (off unless PCB_BOX_LIBRARY_CONFIG is set).  It fills pin
    # numbers/names for ICs whose model the records already resolved; without it an IC pin
    # needs both fields read by OCR and most of them never reach the export.
    # A printed part number that matches a library key names the component, so the library can
    # number its pins.  Connectors are the case that needed it: their model line is read whole
    # but the field binder only took single tokens, leaving Name empty.
    if os.environ.get('PCB_MODEL_NAME_FROM_OCR', '0').strip().lower() not in {'0', 'off', 'false'}:
        try:
            bind_model_names(records, _ocr_texts(stage))
        except Exception as _error:
            print(f'MODEL-NAME-ERROR {type(_error).__name__}: {_error}', flush=True)
    library_audit = apply_component_library(pins, records)
    numbering_audit = apply_passive_numbering(pins, records)
    # Printed pin NUMBER strip, centred on the pin tip.  Without a number the pin and its
    # net members are dropped entirely; the page OCR band looks at the body edge and misses
    # these numbers, which sit at the outer end of the lead.
    strip_audit = None
    try:
        import numpy as _strip_np
        from pcb_parser.pin_number_strip import read_numbers
        from pcb_parser.pin_text_ocr import create_engine as _strip_engine
        _strip_cfg = read(ROOT/'configs/pin_number_strip_v1.json')
        with Image.open(image) as _strip_image:
            _strip_rgb = _strip_np.asarray(_strip_image.convert('RGB'))
        _engine, _ = _strip_engine({'onnx_threads': _strip_cfg.get('onnx_threads', 4)})
        _strip_result = read_numbers(_strip_rgb, pins, records, _strip_cfg, _engine, graph=graph, scope=_strip_cfg.get('scope', 'unattached'))
        strip_audit = _strip_result['summary']
        save(output/'pin_number_strip_audit.json', _strip_result)
    except Exception as _strip_error:
        strip_audit = {'error': f'{type(_strip_error).__name__}: {_strip_error}'}
    # A pin whose only printed text is a bare digit (a header) uses that digit as its number
    # too; without it the pin cannot be expressed at all.
    if os.environ.get('PCB_DIGIT_NAME_AS_NUMBER', '0').strip().lower() not in {'0', 'off', 'false'}:
        digit_filled = 0
        for pin in pins.get('pin_candidates') or []:
            if pin.get('owner_status') != 'assigned':
                continue
            if str(pin.get('number') or '').strip():
                continue
            name = str(pin.get('pinname') or '').strip()
            if not name.isdigit() or len(name) > 3:
                continue
            pin['number'] = name
            pin['number_status'] = 'digit_pin_name'
            pin['number_source'] = 'printed_digit_used_as_number'
            digit_filled += 1
            for attachment in (graph.get('pin_attachments') or []):
                if attachment.get('candidate_id') == pin.get('candidate_id'):
                    attachment['number'] = name
                    break
        if digit_filled:
            print(f'DIGIT-NAME-AS-NUMBER {digit_filled}', flush=True)
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
    # A new convention must not lose an established whole net.  With partial members a
    # later stage may resolve MORE pins for the same net (a superset), which is progress,
    # not a regression; without the switch the old exact-equality rule still applies.
    partial_members = __import__('os').environ.get('PCB_NET_PARTIAL_MEMBERS', '1').strip().lower() \
        not in {'0', 'off', 'false'}
    for nid, net in baseline['nets'].items():
        current = payload['nets'].get(nid)
        if current is None:
            raise ValueError('Engineering convention lost an established whole net')
        if partial_members:
            before = set(str(net.get('hyperGraph') or '').strip('()').split(','))
            after = set(str(current.get('hyperGraph') or '').strip('()').split(','))
            if not before <= after:
                raise ValueError('Engineering convention dropped an established net member')
        elif current != net:
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
        'profile': 'engineering_only_provisional_numbering', 'reference_numbering_verified': False}
    coverage.update(profile=provenance['profile'], engineering_generated_pin_numbers=audit['summary']['generated_pin_numbers'],
                    engineering_generated_pin_names=audit['summary']['generated_pin_names'],
                    blank_pin_names_guessed=audit['summary']['generated_pin_names'],
                    reference_numbering_verified=False, reference_evaluator_verified=False)
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
        'component_library': library_audit, 'passive_numbering': numbering_audit,
        'baseline_export': {k: baseline_cov[k] for k in ['exported_components', 'exported_pins', 'exported_nets']},
        'engineering_export': {k: coverage[k] for k in ['exported_components', 'exported_pins', 'exported_nets']},
        'new_complete_network_ids': sorted(set(payload['nets'])-set(baseline['nets'])),
        'completion': {k: queue[k] for k in ['multi_pin_networks', 'field_only_incomplete_networks', 'non_field_export_blocker_counts']},
        'network_guard': guard['summary'], 'original_points_edges_and_members_preserved': True,
        'all_existing_numbers_names_and_conservative_output_preserved': True,
        'new_OCR_observations': 0, 'new_training': False, 'default_pipeline_adopted': False,
        'complete_target_JSON': False, 'reference_PinF1': None, 'reference_NetF1': None,
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

