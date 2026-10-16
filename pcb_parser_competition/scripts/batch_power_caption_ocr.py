"""Localized OCR replay on the existing development pages, without references."""
import argparse
from pathlib import Path

from run_power_caption_ocr import ROOT, read, save, process, create_engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'runs/s16_power_caption_ocr/predictions')
    p.add_argument('--case-id', action='append')
    a = p.parse_args()
    batch = read(ROOT/'runs/s14_component_library/predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('S14 batch incomplete')
    cfg = read(ROOT/'configs/power_caption_ocr_v1.json')
    engine, models = create_engine(cfg)
    rows = []
    known = {r['case_id'] for r in batch['cases']}
    if a.case_id and not set(a.case_id) <= known:
        p.error('Case not in frozen S14 development batch')
    save(a.output/'batch_summary.json', {'completed': False, 'cases': []})
    for row in batch['cases']:
        case = row['case_id']
        if a.case_id and case not in a.case_id:
            continue
        f = ROOT/f'runs/s14_component_library/predictions/{case}'
        result = process(ROOT/f'data/standardized/{case}/{case}.png', f/'network',
                         f/'pin_text/pin_text_result.json', f/'pin_text/texts.json', a.output/case, cfg, engine, models)
        frozen = read(ROOT/f'runs/s15_power_networks/predictions/{case}/summary.json')
        if result['before_power'] != frozen['power'] or result['baseline_export'] != frozen['proposal_export']:
            raise ValueError('Frozen S15 replay differs')
        rows.append({'case_id': case, **result})
        save(a.output/'batch_summary.json', {'completed': False, 'cases': rows})
        print({'case': case, 'ROIs': result['recovery']['roi_count'],
               'sites': [result['before_power']['observed_power_sites'], result['power']['observed_power_sites']],
               'groups': [result['before_power']['power_groups_applied'], result['power']['power_groups_applied']]}, flush=True)
    totals = {name: {key: sum(r[name][key] for r in rows) for key in
                    ['observed_power_sites', 'power_groups_applied', 'input_network_groups_joined',
                     'power_components_in_review', 'power_components_without_qualified_caption']}
              for name in ['before_power', 'power']}
    totals['recovery'] = {k: sum(r['recovery'][k] for r in rows) for k in rows[0]['recovery']}
    for profile in ['conservative', 'candidate']:
        totals[profile] = {k: sum(r['proposal_export'][profile][k] for r in rows)
                           for k in ['exported_components', 'exported_pins', 'exported_nets']}
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': totals,
         'independent_test': False, 'official_NetF1': None, 'default_pipeline_adopted': False})
    print(totals, flush=True)


if __name__ == '__main__':
    main()
