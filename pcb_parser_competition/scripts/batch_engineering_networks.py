import argparse
from collections import Counter
from pathlib import Path

from run_engineering_networks import ROOT, read, save, process


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s22_engineering_networks/predictions');a = p.parse_args()
    source = ROOT/'runs/s21_whole_networks/predictions'
    if source.is_relative_to(a.output.resolve()) or a.output.resolve().is_relative_to(source):raise ValueError('Output overlaps S21')
    batch = read(source/'batch_summary.json')
    if not batch.get('completed'):raise ValueError('S21 batch incomplete')
    cfg = read(ROOT/'configs/engineering_network_export_v1.json');rows = []
    save(a.output/'batch_summary.json', {'completed': False, 'cases': []})
    for item in batch['cases']:
        case = item['case_id'];result = process(ROOT/f'data/standardized/{case}/{case}.png', source/case, a.output/case, cfg)
        if result['baseline_export'] != item['proposal_export']['conservative']:raise ValueError('S21 baseline changed')
        rows.append({'case_id': case, **result});print({'case': case, 'engineering': result['engineering_export'], 'new_nets': result['new_complete_network_ids']}, flush=True)
    aggregate = {}
    for profile in ['baseline_export', 'engineering_export']:
        aggregate[profile] = {k: sum(r[profile][k] for r in rows) for k in ['exported_components', 'exported_pins', 'exported_nets']}
    aggregate['engineering_fields'] = dict(sum((Counter({k: v for k, v in r['engineering_fields'].items() if isinstance(v, int)}) for r in rows), Counter()))
    aggregate['field_only_incomplete_networks'] = sum(r['completion']['field_only_incomplete_networks'] for r in rows)
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate,
        'independent_test': False, 'official_numbering_verified': False, 'official_PinF1': None, 'official_NetF1': None})
    print(aggregate, flush=True)


if __name__ == '__main__':main()
