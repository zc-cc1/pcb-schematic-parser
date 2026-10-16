"""Replay frozen PNG predictions; labels enter only this offline evaluator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'submission/platform_adapter'))
from pcb_parser.component_joint_pins import resolve_component_pins, export_joint_contacts
from adapter import validate_target


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def score(pred, truth, scope):
    def selected(data, predicted=False):
        output = {}
        for cid, ps in data['pins'].items():
            typ = truth['components'].get(cid, data['components'].get(cid, {})).get('type')
            if scope == 'ic' and typ not in {'ic', 'box'}:
                continue
            if scope == 'rcl' and typ not in {'r', 'c', 'l', 'resistor', 'capacitor', 'inductor'}:
                continue
            output.update({(cid, key): pin for key, pin in ps.items()})
        return output
    p, t = selected(pred, True), selected(truth)
    tp = 0
    errors = []
    for key, pin in p.items():
        target = t.get(key)
        if target is None:
            errors.append({'owner': key[0], 'pin': key[1], 'error': 'identity_not_in_reference'})
            continue
        distance = ((pin['point']['x']-target['point']['x'])**2+(pin['point']['y']-target['point']['y'])**2)**.5
        if pin['pinname'] == target['pinname'] and distance <= 20:
            tp += 1
        else:
            errors.append({'owner': key[0], 'pin': key[1], 'pred_name': pin['pinname'],
                           'reference_name': target['pinname'], 'distance_px': distance})
    return metrics(tp, len(p)-tp, len(t)-tp) | {'errors': errors}


def metrics(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': tp/(tp+fn) if tp+fn else 0.,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, default=ROOT/'runs/s24_observed_net_fields/predictions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/component_joint_pins_v1.json')
    args = parser.parse_args()
    cfg = read(args.config)
    library = read(ROOT/cfg['library_path'])
    protocol = read(ROOT/'configs/unet_inference_v2.json')
    cases = []
    for base in sorted(args.predictions.iterdir()):
        folder = base/'export'
        if not (folder/'engineering_pin_text_result.json').is_file():
            continue
        cid = base.name
        ocr_path = ROOT/'runs/s11_pin_text_v2/predictions'/cid/'texts.json'
        pins, graph = [read(folder/name) for name in ['engineering_pin_text_result.json', 'engineering_graph_proposals.json']]
        records = read(folder/'completed_component_records.json')['records']
        old = read(folder/'engineering_submission.json')
        ocr = read(ocr_path)
        if ocr['image_sha256'] != pins['image_sha256'] or graph['image_sha256'] != pins['image_sha256']:
            raise ValueError('Frozen prediction images differ')
        data, g, audit = resolve_component_pins(pins, graph, records, ocr['texts'], library, cfg)
        out, coverage = export_joint_contacts(data, g, records, old, cfg)
        validate_target(out)
        if pins['pin_candidates'] != read(folder/'engineering_pin_text_result.json')['pin_candidates']:
            raise ValueError('Input predictions changed')
        if graph['edges'] != g['edges']:
            raise ValueError('Wire geometry changed')
        dest = args.output/cid
        write(dest/'joint_submission.json', out)
        write(dest/'joint_pins.json', data)
        write(dest/'joint_audit.json', audit)
        write(dest/'joint_coverage.json', coverage)
        references = []
        for split in ['200_train_cases', '10_GTcase']:
            candidates = list((args.public_root/split).glob(cid+'*/*.json'))
            if not candidates:
                continue
            label = candidates[0]
            png = next(label.parent.glob('*.png'))
            source = ROOT/'data/standardized'/cid/(cid+'.png')
            with Image.open(png) as a, Image.open(source) as b:
                if a.size != b.size or a.convert('RGB').tobytes() != b.convert('RGB').tobytes():
                    raise ValueError('Public reference PNG differs from frozen model input: '+cid)
            truth = read(label)
            references.append({'split': split, 'path': str(label),
                               'sha256': hashlib.sha256(label.read_bytes()).hexdigest(),
                               'scores': {scope: {'baseline': score(old, truth, scope), 'joint': score(out, truth, scope)}
                                          for scope in ['ic', 'rcl', 'all']}})
        cases.append({'case': cid, 'group': 'check' if cid in protocol['acceptance_cases'] else 'development',
                      'baseline': {'pins': sum(map(len, old['pins'].values())), 'nets': len(old['nets'])},
                      'joint': {k: coverage[k] for k in ['exported_pins', 'exported_nets', 'exported_segments']},
                      'resolver': audit['summary'], 'references': references})
    aggregate = {}
    for split in ['200_train_cases', '10_GTcase']:
        for group in ['all', 'development', 'check']:
            selected = [(c, r) for c in cases if group == 'all' or c['group'] == group
                        for r in c['references'] if r['split'] == split]
            if not selected:
                continue
            aggregate[split+'/'+group] = {'cases': len(selected)}
            for scope in ['ic', 'rcl', 'all']:
                aggregate[split+'/'+group][scope] = {}
                for variant in ['baseline', 'joint']:
                    totals = Counter()
                    for _, r in selected:
                        totals.update({k: r['scores'][scope][variant][k] for k in ['tp', 'fp', 'fn']})
                    aggregate[split+'/'+group][scope][variant] = metrics(**totals)
    summary = {'protocol': 'offline exact owner/pin ID/name + Euclidean <=20px, counts pooled; not official scores',
               'limits': 'Frozen historical development/validation pages; Golden overlaps train images; versions scored separately.',
               'inference_reads_labels': False, 'wire_geometry_changed': False,
               'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
               'library_source_sha256': library['source_sha256'], 'cases': cases, 'aggregate': aggregate}
    write(args.output/'comparison.json', summary)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
