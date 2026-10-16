"""Power proposals on the same frozen 26 development model predictions."""
import argparse
from pathlib import Path
from collections import Counter
from run_power_networks import ROOT,read,save,process


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s15_power_networks/predictions');a=p.parse_args()
    batch=read(ROOT/'runs/s14_component_library/predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S14 model batch incomplete')
    rows=[]
    for row in batch['cases']:
        case=row['case_id'];f=ROOT/f'runs/s14_component_library/predictions/{case}'
        summary=process(ROOT/f'data/standardized/{case}/{case}.png',f/'network',f/'pin_text/pin_text_result.json',
                        f/'pin_text/texts.json',a.output/case,read(ROOT/'configs/power_networks_v1.json'))
        for profile in ['conservative','candidate']:
            if summary['baseline_export'][profile] != {k:row[profile][k] for k in ['exported_components','exported_pins','exported_nets']}:
                raise ValueError('Frozen S14 export changed')
        rows.append({'case_id':case,**summary});save(a.output/'batch_summary.json',{'completed':False,'cases':rows})
        print({'case':case,'power_groups':summary['power']['power_groups_applied']},flush=True)
    totals={k:sum(r['power'][k] for r in rows) for k in ['observed_power_sites','power_groups_applied','input_network_groups_joined',
         'power_components_in_review','power_components_without_qualified_caption','network_proposals_before','network_proposals_after']}
    for profile in ['conservative','candidate']:
        totals[profile]={k:sum(r['proposal_export'][profile][k] for r in rows) for k in ['exported_components','exported_pins','exported_nets']}
    save(a.output/'batch_summary.json',{'completed':True,'cases':rows,'aggregate':totals,'official_NetF1':None,'independent_test':False})
    print(totals,flush=True)


if __name__=='__main__':main()
