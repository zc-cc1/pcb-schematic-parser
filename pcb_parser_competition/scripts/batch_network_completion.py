"""Frozen S16 replay; no reference fields used in the completion pipeline."""
import argparse
from pathlib import Path

from run_network_completion import ROOT, read, save, process


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'runs/s17_network_completion/predictions')
    a = p.parse_args()
    batch = read(ROOT/'runs/s16_power_caption_ocr/predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('Incomplete S16 batch')
    rows = []
    save(a.output/'batch_summary.json', {'completed': False, 'cases': rows})
    for row in batch['cases']:
        case = row['case_id']
        result = process(ROOT/f'data/standardized/{case}/{case}.png',
                         ROOT/f'runs/s16_power_caption_ocr/predictions/{case}',
                         ROOT/f'runs/s14_component_library/predictions/{case}',
                         a.output/case, read(ROOT/'configs/pin_number_validation_v1.json'))
        if result['baseline_export'] != row['proposal_export']:
            raise ValueError('Frozen S16 export changed')
        rows.append({'case_id': case, **result})
        save(a.output/'batch_summary.json', {'completed': False, 'cases': rows})
        print({'case': case, 'package_like_numbers_withheld': result['number_role_validation']['package_like_numbers_withheld']}, flush=True)
    aggregate = {profile: {key: sum(r['proposal_export'][profile][key] for r in rows)
                         for key in ['exported_components', 'exported_pins', 'exported_nets']}
                 for profile in ['conservative', 'candidate']}
    aggregate['package_like_numbers_withheld'] = sum(r['number_role_validation']['package_like_numbers_withheld'] for r in rows)
    aggregate['multi_pin_networks'] = sum(r['completion']['conservative']['multi_pin_networks'] for r in rows)
    aggregate['field_only_incomplete_networks'] = sum(r['completion']['conservative']['field_only_incomplete_networks'] for r in rows)
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate,
         'official_NetF1': None, 'independent_test': False, 'default_pipeline_adopted': False})
    print(aggregate, flush=True)


if __name__ == '__main__':
    main()
