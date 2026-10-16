"""Read-only public-label audit and offline box-convention diagnostic.

Reference labels are read only by this analysis tool, never by image inference.
The geometry diagnostic uses GT for association and is not an official score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_audit(root, split):
    types = Counter()
    names = defaultdict(Counter)
    dimensions = defaultdict(Counter)
    exceptions = []
    issues = Counter()
    manifest = []
    for path in sorted((root/split).rglob('*.json')):
        data = read(path)
        png = next(path.parent.glob('*.png'))
        with Image.open(png) as image:
            size = list(image.size)
        manifest.append({'json': str(path), 'json_sha256': sha(path),
                         'png': str(png), 'png_sha256': sha(png), 'size': size})
        for cid, component in data['components'].items():
            typ, name = component['type'], component.get('Name')
            types[typ] += 1
            names[typ]['equals_key' if name == cid else 'null' if name is None else 'other'] += 1
            box = component['bbox']
            dimensions[typ][(round(box[2]-box[0], 3), round(box[3]-box[1], 3))] += 1
            if name != cid and typ != 'box':
                exceptions.append({'file': str(path), 'key': cid, 'component': component})
            if cid not in data['pins']:
                issues['missing_pin_group'] += 1
        refs = {f'{cid}.{pin[4:]}' for cid, pins in data['pins'].items() for pin in pins}
        for net in data['nets'].values():
            members = {s.strip() for s in net['hyperGraph'][1:-1].split(',') if s.strip()}
            issues['unresolved_net_refs'] += len(members-refs)
    return {'case_count': len(manifest), 'component_count': sum(types.values()),
            'type_counts': dict(types), 'name_counts': {k: dict(v) for k, v in names.items()},
            'bbox_dimensions': {k: [{'width': wh[0], 'height': wh[1], 'count': n}
                                    for wh, n in counts.most_common()]
                                for k, counts in dimensions.items()},
            'non_box_name_exceptions': exceptions, 'issues': dict(issues), 'manifest': manifest}


def geometry_diagnostic(root, cached):
    adapter_path = Path(__file__).resolve().parents[1]/'submission/platform_adapter/adapter.py'
    spec = importlib.util.spec_from_file_location('public_audit_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    mapping = {'resistor': 'r', 'capacitor': 'c', 'inductor': 'l', 'diode': 'd', 'led': 'led'}
    sizes = {'r': (8, 20), 'c': (16, 20), 'l': (6, 34), 'd': (14, 20), 'led': (14, 20)}
    rows, cases = [], []
    for case in read(cached)['cases']:
        folder = root/'200_train_cases'/case['case_id']
        if not folder.is_dir():
            continue
        local_png = Path(__file__).resolve().parents[1]/'data/standardized'/case['case_id']/(case['case_id']+'.png')
        public_png = next(folder.glob('*.png'))
        with Image.open(local_png) as a, Image.open(public_png) as b:
            if a.size != b.size or a.convert('RGB').tobytes() != b.convert('RGB').tobytes():
                raise ValueError('Cached prediction image differs from public PNG: '+case['case_id'])
        cases.append(case['case_id'])
        labels = read(next(folder.glob('*.json')))
        with Image.open(next(folder.glob('*.png'))) as image:
            height = image.height
        for source_type, typ in mapping.items():
            predictions = [p for p in case['predictions'] if p['class_name'] == source_type]
            truth = [(cid, c) for cid, c in labels['components'].items() if c['type'] == typ]
            if not predictions or not truth:
                continue
            boxes = [[p['bbox'][0], height-p['bbox'][3], p['bbox'][2], height-p['bbox'][1]]
                     for p in predictions]
            gt_boxes = [c['bbox'] for _, c in truth]
            centers = lambda bs: np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in bs])
            pc, gc = centers(boxes), centers(gt_boxes)
            cost = np.sqrt(((pc[:, None]-gc[None, :])**2).sum(axis=2))
            pred_indices, gt_indices = linear_sum_assignment(cost)
            for pi, gi in zip(pred_indices, gt_indices):
                if cost[pi, gi] > 30:
                    continue
                box = boxes[pi]
                cx, cy = pc[pi]
                proposed = adapter.to_target({'components': {'DIAGNOSTIC': {
                    'type': typ, 'Name': None, 'value': None, 'bbox': box}},
                    'pins': {}, 'nets': {}})['components']['DIAGNOSTIC']['bbox']
                rows.append({'case': case['case_id'], 'type': typ, 'gt_key': truth[gi][0],
                             'prediction_index_within_class': int(pi),
                             'center_distance_px': float(cost[pi, gi]),
                             'raw_bbox': box, 'proposed_bbox': proposed, 'gt_bbox': gt_boxes[gi],
                             'raw_max_error_px': max(abs(x-y) for x, y in zip(box, gt_boxes[gi])),
                             'proposed_max_error_px': max(abs(x-y) for x, y in zip(proposed, gt_boxes[gi]))})
    totals = {}
    for typ in sizes:
        selected = [r for r in rows if r['type'] == typ]
        if not selected:
            continue
        totals[typ] = {'paired_count': len(selected)}
        for prefix in ['raw', 'proposed']:
            values = [r[prefix+'_max_error_px'] for r in selected]
            totals[typ].update({prefix+'_le_5': int(sum(v <= 5 for v in values)),
                                prefix+'_le_20': int(sum(v <= 20 for v in values)),
                                prefix+'_mean_max_error_px': float(statistics.mean(values))})
    return {'purpose': 'Box convention diagnostic only, not recognition F1 or official rescoring',
            'association': 'same category, Hungarian center distance, retain distances <=30 px',
            'selection_limit': 'Missed and unpaired objects excluded; do not present as full detection accuracy',
            'ground_truth_used_for_association_only': True, 'proposed_box_uses_gt_coordinates': False,
            'all_paired_case_images_verified_pixel_identical': True,
            'projection_source': str(adapter_path), 'projection_source_sha256': sha(adapter_path),
            'prediction_cache': str(cached), 'cache_sha256': sha(cached),
            'cases': cases, 'totals': totals, 'pairs': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cached-detector', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'source_root': str(args.root), 'source_untouched': True, 'splits': {}}
    for split in ['200_train_cases', '10_GTcase']:
        report['splits'][split] = split_audit(args.root, split)
    library = read(args.root/'cpntLibrary')
    conflicts = []
    for model, entry in library.items():
        numbers = defaultdict(set)
        for pin in entry['pins']:
            if pin.get('number') and pin.get('name'):
                numbers[str(pin['number'])].add(pin['name'])
        ambiguous = {n: sorted(v) for n, v in numbers.items() if len(v) > 1}
        if ambiguous:
            conflicts.append({'model': model, 'ambiguous_number_count': len(ambiguous)})
    report['library'] = {'path': str(args.root/'cpntLibrary'), 'sha256': sha(args.root/'cpntLibrary'),
                         'model_count': len(library), 'models_with_conflicting_pin_names': len(conflicts),
                         'conflict_examples': conflicts[:10]}
    report['train_vs_golden_pairs'] = []
    for gt in (args.root/'10_GTcase').rglob('*.json'):
        cid = gt.parent.name.split('-')[0]
        train = next((args.root/'200_train_cases'/cid).glob('*.json'))
        gd, td = read(gt), read(train)
        gp, tp = next(gt.parent.glob('*.png')), next(train.parent.glob('*.png'))
        with Image.open(gp) as a, Image.open(tp) as b:
            same_pixels = a.size == b.size and a.convert('RGB').tobytes() == b.convert('RGB').tobytes()
        common = gd['components'].keys() & td['components'].keys()
        report['train_vs_golden_pairs'].append({'case': cid, 'same_pixels': same_pixels,
            'identical_image_bytes': sha(gp) == sha(tp),
            'gt_json': str(gt), 'train_json': str(train),
            'common_component_keys': len(common),
            'different_names_on_common_keys': sum(gd['components'][k]['Name'] != td['components'][k]['Name'] for k in common),
            'different_values_on_common_keys': sum(gd['components'][k]['value'] != td['components'][k]['value'] for k in common),
            'different_boxes_on_common_keys': sum(gd['components'][k]['bbox'] != td['components'][k]['bbox'] for k in common),
            'golden_only_keys': sorted(gd['components'].keys()-td['components'].keys()),
            'train_only_keys': sorted(td['components'].keys()-gd['components'].keys())})
    if args.cached_detector:
        geometry = geometry_diagnostic(args.root, args.cached_detector)
        (args.output/'geometry_diagnostic.json').write_text(json.dumps(geometry, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        report['geometry_summary'] = {k: v for k, v in geometry.items() if k != 'pairs'}
    (args.output/'dataset_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'case_counts': {k: v['case_count'] for k, v in report['splits'].items()},
                      'library': report['library'], 'geometry': report.get('geometry_summary', {}).get('totals'),
                      'output': str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
