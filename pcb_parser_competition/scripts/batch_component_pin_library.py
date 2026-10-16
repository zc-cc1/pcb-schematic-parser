"""Frozen 26-page model replay, without reading reference annotation fields."""
import argparse
from collections import Counter
from pathlib import Path

from run_component_pin_library import ROOT, read, save, process


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'runs/s14_component_library/predictions')
    a = p.parse_args()
    upstream = read(ROOT/'runs/s11_pin_text_v2/predictions/batch_summary.json')
    if not upstream.get('completed'):
        raise ValueError('Incomplete frozen upstream batch')
    rows = []
    for item in upstream['cases']:
        case = item['case_id']
        source = ROOT/f'runs/s11_pin_text_v2/predictions/{case}'
        cache = ROOT/f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz'
        summary = process(ROOT/f'data/standardized/{case}/{case}.png', source, cache, cache.with_suffix('.json'),
                          a.output/case, read(ROOT/'configs/component_pin_library_v1.json'),
                          ROOT/f'runs/s11_pin_text_v2/downstream/{case}/network')
        rows.append({'case_id': case, **summary})
        save(a.output/'batch_summary.json', {'completed': False, 'cases': rows})
        print({'case': case, 'library': summary['library'], 'conservative': summary['conservative']}, flush=True)
    total = {}
    for key in ['conservative', 'candidate']:
        total[key] = {k: sum(r[key][k] for r in rows) for k in ['exported_components', 'exported_pins', 'exported_nets']}
    total['library'] = dict(sum((Counter(r['library']) for r in rows), Counter()))
    save(a.output/'batch_summary.json', {'completed': True, 'cases': rows, 'aggregate': total,
         'old_development_data': True, 'official_PinF1': None, 'official_NetF1': None, 'default_pipeline_adopted': False})
    print(total, flush=True)


if __name__ == '__main__':
    main()
