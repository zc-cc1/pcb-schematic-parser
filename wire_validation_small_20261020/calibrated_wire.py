"""Isolated wire-scoring prototype checked against three historical platform cases.

Same-name power rails merge before network matching; both networks and line
segments use one-to-one assignment. This is a local reproduction, not official code.
"""
import copy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'wire_validation_20261020'
sys.path.insert(0,str(PREVIOUS))
from validate import read,metrics,TRUTH


def groups(document):
    components=document['components']
    rail=metrics.v2.rails(components)
    rails={(value,) for cid in components if (value:=rail(cid,components)) is not None}
    result=[{'vertices':{metrics.v2.vertex_key(t,components,rail) for t in metrics.old._members(net)},
             'edges':metrics.old._segments(net)} for net in document['nets'].values()]
    changed=True
    while changed:
        changed=False
        for i in range(len(result)):
            for j in range(i+1,len(result)):
                if result[i]['vertices'] & result[j]['vertices'] & rails:
                    result[i]['vertices']|=result[j]['vertices']
                    result[i]['edges']+=result[j]['edges']
                    result.pop(j);changed=True;break
            if changed:break
    # Deterministic ordering prevents JSON dictionary order affecting tie resolution.
    def key(net):
        vertices=tuple(sorted(tuple(map(str,v)) for v in net['vertices']))
        edges=tuple(sorted(min(tuple(map(tuple,e)),tuple(map(tuple,e[::-1]))) for e in net['edges']))
        return vertices,edges
    return sorted(result,key=key)


def score_wire(prediction,truth,tolerance=20.0):
    pn,tn=groups(prediction),groups(truth)
    pcount=sum(len(n['edges']) for n in pn);tcount=sum(len(n['edges']) for n in tn)
    similarity=np.zeros((len(pn),len(tn)))
    for i,p in enumerate(pn):
        for j,t in enumerate(tn):
            similarity[i,j]=2*len(p['vertices']&t['vertices'])/max(1,len(p['vertices'])+len(t['vertices']))
    rows,cols=linear_sum_assignment(-similarity)
    tp=0
    for i,j in zip(rows,cols):
        if similarity[i,j]<=0:continue
        ps,ts=np.asarray(pn[i]['edges']),np.asarray(tn[j]['edges'])
        if not ps.size or not ts.size:continue
        distance=np.minimum(np.linalg.norm(ps[:,None,:,:]-ts[None,:,:,:],axis=3).max(axis=2),
                            np.linalg.norm(ps[:,None,:,:]-ts[None,:,::-1,:],axis=3).max(axis=2))
        hit=(distance<=tolerance).astype(int)
        r,c=linear_sum_assignment(-hit);tp+=int(hit[r,c].sum())
    return metrics.f1_block(tp,pcount-tp,tcount-tp)


def main():
    cases=['5','6','13']
    platform=ROOT.parent/'评测结果/第9次测评-66.59/scoring_result'
    official={r['case']:r['wire_f1'] for r in read(platform/'summary.json')['cases']}
    calibration=[]
    for case in cases:
        prediction=read(platform/f'final_results/{case}_Pred.json')
        truth=read(next((TRUTH/case).glob('*.json')))
        local=score_wire(prediction,truth)['f1']
        # Existing f1_block rounds to four decimals; platform report retains six.
        calibration.append({'case':case,'platform':official[case],'local':local,
                            'within_report_rounding':abs(local-official[case])<=.000051})
    calibration_passed=all(r['within_report_rounding'] for r in calibration)
    source=read(ROOT/'small_set_results.json')
    rows=[]
    for case in cases:
        truth=read(next((TRUTH/case).glob('*.json')))
        row={'case':case,'variants':{}}
        for name in source['macro']:
            prediction=read(ROOT/name/f'{case}_Pred.json')
            wire=score_wire(prediction,truth)
            reverse=copy.deepcopy(prediction);reverse['nets']=dict(reversed(list(prediction['nets'].items())))
            assert score_wire(reverse,truth)==wire
            old=next(r for r in source['per_case'] if r['case']==case)['variants'][name]
            values={**old['metrics'],'wire_f1':wire['f1']}
            row['variants'][name]={'metrics':values,'wire_counts':wire,
                'score_100':100*sum(metrics.WEIGHTS[k]*v for k,v in values.items())}
        rows.append(row)
    macro={name:{'metrics':{k:sum(r['variants'][name]['metrics'][k] for r in rows)/len(rows)
                           for k in metrics.WEIGHTS},
                 'score_100':sum(r['variants'][name]['score_100'] for r in rows)/len(rows)}
           for name in source['macro']}
    result={'calibration':calibration,'calibration_passed':calibration_passed,'cases':cases,'macro':macro,'per_case':rows,
            'network_order_invariance_passed':True,
            'note':'Experimental diagnostic scorer. If calibration_passed is false, these values must not be represented as platform scores or used for deployment acceptance.'}
    (ROOT/'candidate_scorer_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'calibration':calibration,'calibration_passed':calibration_passed},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
