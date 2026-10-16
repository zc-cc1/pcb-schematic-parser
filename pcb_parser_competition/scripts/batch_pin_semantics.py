"""Downstream replay using the same learned wire probabilities and pin points."""
import argparse
from pathlib import Path
from run_pin_semantics import ROOT,read,save,downstream


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s11_pin_text_v2/downstream')
    p.add_argument('--case',action='append');a=p.parse_args();rows=[]
    batch=read(ROOT/'runs/s11_pin_text_v2/predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Pin text batch is incomplete')
    for row in batch['cases']:
        case=row['case_id']
        if a.case and case not in a.case:continue
        summary=downstream(ROOT/f'data/standardized/{case}/{case}.png',ROOT/f'runs/s11_pin_text_v2/predictions/{case}',
                           ROOT/f'runs/s7_wire_graph_v1/inputs/{case}/wire_probability.npz',None,a.output/case)
        # Text changes must not alter physical wire geometry or attachments.
        graph=read(a.output/f'{case}/physical/graph_proposals.json');old=read(ROOT/f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json')
        if graph['edges']!=old['edges'] or [(x['candidate_id'],x['point'],x['network_id']) for x in graph['pin_attachments']] != [(x['candidate_id'],x['point'],x['network_id']) for x in old['pin_attachments']]:
            raise ValueError('Text stage changed frozen physical topology')
        rows.append({'case_id':case,**summary});save(a.output/'batch_summary.json',{'cases':rows,'completed':False})
        print({'case':case,'export':summary['export']},flush=True)
    if a.case and set(a.case)!={r['case_id'] for r in rows}:raise ValueError('Unavailable case requested')
    save(a.output/'batch_summary.json',{'cases':rows,'completed':True})


if __name__=='__main__':main()
