"""Freeze the same 26 images, boxes, points and OCR to isolate text changes."""
import argparse
from pathlib import Path
from run_pin_text_v2 import ROOT,read,save,process


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s11_pin_text_v2/predictions')
    p.add_argument('--case',action='append');a=p.parse_args();rows=[];cfg=read(ROOT/'configs/pin_text_v2.json')
    folders=sorted(f for f in (ROOT/'runs/s6_pin_text_v1/predictions').iterdir() if f.is_dir() and len(f.name)==4 and f.name.isdigit())
    for folder in folders:
        case=folder.name
        if a.case and case not in a.case:continue
        summary=process(ROOT/f'data/standardized/{case}/{case}.png',folder/'pin_text_result.json',folder/'texts.json',a.output/case,cfg)
        summary['case_id']=case;rows.append(summary);save(a.output/'batch_summary.json',{'cases':rows,'config':cfg,'official_PinF1':None,'completed':False})
        print({'case':case,'numbers':summary['summary']['number_observed'],'names':summary['summary']['name_observed'],
               'new_identities':summary['summary']['new_header_identities'],'secondary_hypotheses':summary['summary']['secondary_tip_hypotheses_text_withheld']},flush=True)
    if a.case and {r['case_id'] for r in rows}!=set(a.case):raise ValueError('Unavailable case')
    save(a.output/'batch_summary.json',{'cases':rows,'config':cfg,'official_PinF1':None,'completed':True})


if __name__=='__main__':main()
