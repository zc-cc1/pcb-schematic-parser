"""Replay a frozen 26-page development batch; never read reference fields."""
import argparse
from pathlib import Path
from collections import Counter
from run_pin_output_rules import ROOT, read, save, process


def aggregate(rows):
    totals={}
    for profile in ['before','conservative','candidate']:
        value={k:sum(r[profile][k] for r in rows) for k in ['exported_components','exported_pins','exported_nets']}
        reasons=Counter()
        for row in rows:reasons.update(row[profile]['pin_reason_counts'])
        value['pin_reason_counts']=dict(reasons);totals[profile]=value
    totals['label_names_added']=sum(r['name_completion']['ordinary_label_names_added'] for r in rows)
    totals['candidate_passive_names_added']=sum(r['name_completion']['candidate_passive_names_added'] for r in rows)
    totals['candidate_nets_excluded_by_power_guard']=sum(len(r['power_guard_excluded_candidate_network_ids']) for r in rows)
    totals['candidate_nets']=[{'case_id':r['case_id'],'network_id':n} for r in rows
                             for n in r['candidate_exported_network_ids']]
    return totals


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s12_pin_output_rules/predictions');a=p.parse_args()
    batch=read(ROOT/'runs/s11_pin_text_v2/predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Upstream batch is incomplete')
    rows=[]
    for item in batch['cases']:
        case=item['case_id'];pin_folder=ROOT/f'runs/s11_pin_text_v2/predictions/{case}'
        s=process(ROOT/f'data/standardized/{case}/{case}.png',ROOT/f'runs/s11_pin_text_v2/downstream/{case}/network',
                  pin_folder/'pin_text_result.json',pin_folder/'texts.json',a.output/case,read(ROOT/'configs/pin_output_rules_v1.json'))
        rows.append({'case_id':case,**s});save(a.output/'batch_summary.json',{'completed':False,'cases':rows})
        print({'case':case,'conservative':s['conservative'],'candidate':s['candidate']},flush=True)
    summary={'completed':True,'cases':rows,'aggregate':aggregate(rows),'old_development_data':True,
             'official_PinF1':None,'official_NetF1':None,'accuracy_evaluation_performed':False}
    save(a.output/'batch_summary.json',summary)


if __name__=='__main__':main()
