import argparse
from pathlib import Path

from run_whole_networks import ROOT, read, save, process


def main():
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, default=ROOT/'runs/s21_whole_networks/predictions');args = p.parse_args()
    source = ROOT/'runs/s20_nearby_tip_fields'
    if args.output.resolve().is_relative_to(source) or source.is_relative_to(args.output.resolve()):raise ValueError('Batch output overlaps S20')
    batch = read(source/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S20 batch incomplete')
    cfg = read(ROOT/'configs/whole_network_export_v1.json');rows = []
    save(args.output/'batch_summary.json', {'completed': False, 'cases': []})
    for item in batch['cases']:
        case = item['case_id'];texts = ROOT/f'runs/s14_component_library/predictions/{case}/pin_text/texts.json'
        result = process(ROOT/f'data/standardized/{case}/{case}.png', source/f'predictions/{case}', texts, args.output/case, cfg)
        if result['baseline_export'] != item['proposal_export']:raise ValueError('S20 baseline coverage changed')
        rows.append({'case_id': case, **result});print({'case': case, 'identity': result['identity_recovery'], 'export': result['proposal_export']['conservative']}, flush=True)
    aggregate = {'new_identities': sum(r['identity_recovery']['new_identities'] for r in rows)}
    for profile in ['conservative', 'candidate']:
        aggregate[profile] = {k: sum(r['proposal_export'][profile][k] for r in rows) for k in ['exported_components', 'exported_pins', 'exported_nets']}
    save(args.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate,
        'independent_test': False, 'official_PinF1': None, 'official_NetF1': None});print(aggregate, flush=True)


if __name__ == '__main__':main()
