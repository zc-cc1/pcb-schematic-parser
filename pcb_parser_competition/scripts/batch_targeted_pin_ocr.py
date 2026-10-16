"""Selective OCR on the same frozen S17 development image predictions."""
import argparse
from pathlib import Path
from collections import Counter

from run_targeted_pin_ocr import ROOT,read,save,process,create_engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s18_targeted_pin_ocr/predictions')
    p.add_argument('--case-id',action='append');a=p.parse_args()
    batch=read(ROOT/'runs/s17_network_completion/predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S17 batch incomplete')
    known={r['case_id'] for r in batch['cases']}
    if a.case_id and not set(a.case_id)<=known:p.error('Case absent from S17 batch')
    cfg=read(ROOT/'configs/targeted_pin_ocr_v1.json');engine,models=create_engine(cfg)
    rows=[];save(a.output/'batch_summary.json',{'completed':False,'cases':[]})
    for row in batch['cases']:
        case=row['case_id']
        if a.case_id and case not in a.case_id:continue
        result=process(ROOT/f'data/standardized/{case}/{case}.png',ROOT/f'runs/s17_network_completion/predictions/{case}',
                       a.output/case,cfg,engine,models)
        if result['baseline_export']!=row['proposal_export']:raise ValueError('S17 export coverage changed')
        rows.append({'case_id':case,**result});save(a.output/'batch_summary.json',{'completed':False,'cases':rows})
        print({'case':case,'ROIs':result['roi_count'],'recovery':result['recovery'],'export':result['proposal_export']['conservative']},flush=True)
    totals={k:sum(r['recovery'][k] for r in rows) for k in ['missing_numbers_added','missing_names_added','existing_fields_overwritten','tip_coordinates_changed','network_members_changed']}
    totals['roi_count']=sum(r['roi_count'] for r in rows)
    for profile in ['conservative','candidate']:
        totals[profile]={k:sum(r['proposal_export'][profile][k] for r in rows) for k in ['exported_components','exported_pins','exported_nets']}
    save(a.output/'batch_summary.json',{'completed':True,'cases':rows,'aggregate':totals,'independent_test':False,'official_PinF1':None,'official_NetF1':None})
    print(totals,flush=True)


if __name__=='__main__':main()
