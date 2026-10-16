import argparse
from collections import Counter
from pathlib import Path

from run_capacitor_tip_geometry import ROOT, read, save, sha, process


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s23_capacitor_geometry/predictions');a = p.parse_args()
    source = ROOT/'runs/s21_whole_networks/predictions';batch = read(source/'batch_summary.json')
    split = read(ROOT/'runs/unet_inference_v2/split.json');model_hash = read(ROOT/'configs/unet_inference_v2.json')['checkpoint_sha256']
    if not batch.get('completed') or split['source_checkpoint_sha256'] != model_hash:raise ValueError('Frozen inputs unavailable')
    if a.output.resolve().is_relative_to(source) or source.is_relative_to(a.output.resolve()):raise ValueError('Output overlaps S21')
    rows = [];save(a.output/'batch_summary.json', {'completed': False, 'cases': []})
    for item in batch['cases']:
        case = item['case_id'];image = ROOT/f'data/standardized/{case}/{case}.png'
        row = next(r for r in split['rows'] if r['case_id'] == case)
        if sha(image) != row['image_sha256']:raise ValueError('Frozen cache image mismatch')
        paths = [ROOT/f'runs/unet_inference_v2/cache/{case}/scale_{scale:.1f}.npz' for scale in [1, 1.5, 2]]
        meta = a.output.parent/f'inputs/{case}.json'
        save(meta, {'image_sha256': sha(image), 'size': [row['width'], row['height']], 'coordinate_origin': 'top_left',
            'weights_sha256': model_hash, 'prediction_only': True, 'scales': [1.0, 1.5, 2.0],
            'probability_sha256': [sha(path) for path in paths], 'source': 'Frozen_PNG_U-Net_v2_cache_with_native_tip_reproduction_check'})
        result = process(image, source/case, a.output/case, read(ROOT/'configs/capacitor_tip_geometry_v1.json'), paths, meta)
        old_export = read(ROOT/f'runs/s22_engineering_networks/predictions/{case}/engineering_submission.json')
        new_export = read(a.output/f'{case}/export/engineering_submission.json')
        if any(new_export['nets'].get(nid, {}).get('hyperGraph') != net['hyperGraph'] for nid, net in old_export['nets'].items()):
            raise ValueError('Geometry trial lost an established engineering complete net')
        rows.append({'case_id': case, **result});print({'case': case, 'geometry': result['geometry'], 'export': result['engineering_export']}, flush=True)
    aggregate = {'geometry': dict(sum((Counter({k: v for k, v in r['geometry'].items() if isinstance(v, int)}) for r in rows), Counter())),
        'engineering_export': {k: sum(r['engineering_export'][k] for r in rows) for k in ['exported_components', 'exported_pins', 'exported_nets']}}
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate, 'independent_test': False, 'official_PinF1': None, 'official_NetF1': None})
    print(aggregate, flush=True)


if __name__ == '__main__':main()
