"""Compare paired full-pipeline runs; use GT exclusively in this scoring script."""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from validate import HERE, TRUTH, metrics, read, save, score


def geometry_counts(pred, truth):
    ps=[s for n in pred['nets'].values() for s in metrics.old._segments(n)]
    ts=[s for n in truth['nets'].values() for s in metrics.old._segments(n)]
    matrix=np.array([[metrics.old._close(p,t,20.0) for t in ts] for p in ps], dtype=int)
    hits=0
    if matrix.size:
        i,j=linear_sum_assignment(-matrix);hits=int(matrix[i,j].sum())
    return {'tp':hits,'fp':len(ps)-hits,'fn':len(ts)-hits,
            'f1':2*hits/max(1,len(ps)+len(ts))}


def robust_wire(pred, truth):
    pn=list(pred['nets'].values());tn=list(truth['nets'].values())
    hits=np.zeros((len(pn),len(tn)),dtype=int)
    for a,p in enumerate(pn):
        ps=metrics.old._segments(p)
        for b,t in enumerate(tn):
            ts=metrics.old._segments(t)
            matrix=np.array([[metrics.old._close(x,y,20.0) for y in ts] for x in ps],dtype=int)
            if matrix.size:
                i,j=linear_sum_assignment(-matrix);hits[a,b]=int(matrix[i,j].sum())
    tp=0
    if hits.size:
        i,j=linear_sum_assignment(-hits);tp=int(hits[i,j].sum())
    pred_count=sum(len(metrics.old._segments(n)) for n in pn)
    truth_count=sum(len(metrics.old._segments(n)) for n in tn)
    return 2*tp/max(1,pred_count+truth_count)


def main():
    baseline=HERE/'baseline';candidate=HERE/'candidate_path'
    results={'baseline':score(baseline),'candidate_path':score(candidate)}
    assert results['baseline']['cases']==13 and results['candidate_path']['cases']==13
    for name in ['baseline','candidate_path','control_replay']:
        decision=read(HERE/name/'decision_log.json')
        assert len(decision['cases'])==13 and all(r['success'] for r in decision['cases']), name
    rows=[]
    control_rows=[]
    totals={name:[] for name in ['baseline_geometry','candidate_geometry','baseline_robust','candidate_robust']}
    for folder in sorted(TRUTH.iterdir(),key=lambda p:int(p.name)):
        filename=f'{folder.name}_Pred.json'
        b=read(baseline/'final_results'/filename);c=read(candidate/'final_results'/filename)
        control=read(HERE/'control_replay/final_results'/filename)
        control_rows.append({'case':folder.name,'identical':b==control})
        truth=read(next(folder.glob('*.json')))
        bg,cg=geometry_counts(b,truth),geometry_counts(c,truth)
        br,cr=robust_wire(b,truth),robust_wire(c,truth)
        totals['baseline_geometry'].append(bg['f1']);totals['candidate_geometry'].append(cg['f1'])
        totals['baseline_robust'].append(br);totals['candidate_robust'].append(cr)
        bm,cm=metrics.score_case(b,truth),metrics.score_case(c,truth)
        rows.append({'case':folder.name,'baseline_wire':bm['wire_f1']['f1'],'candidate_wire':cm['wire_f1']['f1'],
            'wire_delta':cm['wire_f1']['f1']-bm['wire_f1']['f1'],
            'total_score_delta_100':100*(cm['final_score']-bm['final_score']),
            'baseline_geometry':bg,'candidate_geometry':cg,
            'baseline_robust':br,'candidate_robust':cr,
            'components_identical':b['components']==c['components'],
            'pins_identical':b['pins']==c['pins']})
    results['independent_checks']={k:sum(v)/len(v) for k,v in totals.items()}
    results['per_case_comparison']=rows
    results['metric_note']='Primary: local platform_metrics_v3, 20px, macro. Independent checks enforce one-to-one segment matches; not official scores.'
    neural_inputs={}
    for name in ['baseline','candidate_path','control_replay']:
        neural_inputs[name]={}
        for work in (HERE/name/'work').iterdir():
            early=work/'p_stage23_stage21_stage20_stage18/stage17/stage16/upstream/upstream'
            if not (early/'pin_text_result.json').exists():continue
            image_hash=read(early/'pin_text_result.json')['image_sha256']
            neural_inputs[name][image_hash]={str(p):hashlib.sha256((early/p).read_bytes()).hexdigest()
                for p in ['pin_text_result.json','component_records.json','texts.json','ownership/probabilities.npz']}
    input_identity=all(neural_inputs['baseline']==neural_inputs[name] for name in ['candidate_path','control_replay'])
    results['frozen_neural_inputs_identical']=input_identity
    assert all(r['identical'] for r in control_rows), 'Frozen original replay differs from PNG baseline'
    assert input_identity, 'Neural prediction caches differ between variants'
    save(HERE/'control_integrity.json',{'cases':len(control_rows),
        'identical_cases':sum(r['identical'] for r in control_rows),'per_case':control_rows,
        'frozen_neural_inputs_identical':input_identity,'neural_input_hashes':neural_inputs})
    save(HERE/'comparison.json',results)
    print(json.dumps({'baseline':{k:v for k,v in results['baseline'].items() if k!='per_case'},
                      'candidate_path':{k:v for k,v in results['candidate_path'].items() if k!='per_case'},
                      'independent_checks':results['independent_checks'],
                      'per_case_comparison':rows},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
