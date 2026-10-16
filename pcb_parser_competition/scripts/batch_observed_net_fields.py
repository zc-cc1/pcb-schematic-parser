import argparse
from collections import Counter
from pathlib import Path

from run_observed_net_fields import ROOT, read, save, process


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s24_observed_net_fields/predictions');a = p.parse_args()
    source = ROOT/'runs/s23_capacitor_geometry/predictions';batch = read(source/'batch_summary.json')
    if not batch.get('completed'):raise ValueError('S23 batch incomplete')
    if a.output.resolve().is_relative_to(source) or source.is_relative_to(a.output.resolve()):raise ValueError('Batch output overlaps S23')
    rows = [];save(a.output/'batch_summary.json', {'completed': False, 'cases': []})
    for item in batch['cases']:
        case = item['case_id']
        result = process(ROOT/f'data/standardized/{case}/{case}.png', source/case,
            ROOT/f'runs/s14_component_library/predictions/{case}/pin_text/texts.json', a.output/case,
            read(ROOT/'configs/observed_net_fields_v1.json'))
        rows.append({'case_id': case, **result})
        print({'case': case, 'fields': result['observed_fields'], 'export': result['engineering_export'], 'new_nets': result['new_complete_network_ids']}, flush=True)
    aggregate = {'observed_fields': dict(sum((Counter(r['observed_fields']) for r in rows), Counter())),
        'engineering_export': {k: sum(r['engineering_export'][k] for r in rows) for k in ['exported_components', 'exported_pins', 'exported_nets']},
        'new_complete_nets': sum(len(r['new_complete_network_ids']) for r in rows)}
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate,
        'independent_test': False, 'official_PinF1': None, 'official_NetF1': None});print(aggregate, flush=True)


if __name__ == '__main__':main()
