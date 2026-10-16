"""Replay S18 predictions through the independent S20 field-link pilot."""
import argparse
from pathlib import Path

from run_nearby_tip_fields import ROOT, read, save, process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'runs/s20_nearby_tip_fields/predictions')
    args = parser.parse_args()
    source = ROOT/'runs/s18_targeted_pin_ocr'
    if args.output.resolve().is_relative_to(source) or source.is_relative_to(args.output.resolve()):
        raise ValueError('Batch output overlaps frozen S18 input')
    batch = read(source/'predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('S18 batch incomplete')
    cfg = read(ROOT/'configs/nearby_tip_fields_v1.json')
    rows = []
    save(args.output/'batch_summary.json', {'completed': False, 'cases': []})
    for item in batch['cases']:
        case = item['case_id']
        result = process(ROOT/f'data/standardized/{case}/{case}.png',
                         ROOT/f'runs/s18_targeted_pin_ocr/predictions/{case}', args.output/case, cfg)
        if result['baseline_export'] != item['proposal_export']:
            raise ValueError('S18 export coverage changed')
        rows.append({'case_id': case, **result})
        print({'case': case, 'reconciliation': result['reconciliation'], 'export': result['proposal_export']['conservative']}, flush=True)
    aggregate = {'names_reconciled': sum(r['reconciliation']['names_reconciled'] for r in rows)}
    for profile in ['conservative', 'candidate']:
        aggregate[profile] = {k: sum(r['proposal_export'][profile][k] for r in rows)
                              for k in ['exported_components', 'exported_pins', 'exported_nets']}
    save(args.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': aggregate,
                                           'independent_test': False, 'official_PinF1': None, 'official_NetF1': None})
    print(aggregate, flush=True)


if __name__ == '__main__':
    main()
