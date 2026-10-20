import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
sys.path.insert(0,'D:/pcb5_submission')
from adapter import validate_target
import platform_metrics_v3 as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
from contact_proposal import propose
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def conn(pred,truth):
    pn,tn=logical_nets(pred),logical_nets(truth)
    return 2*sum(score for _,_,score in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))


def calibrate():
    checked=0
    for edition in ['第9次测评-66.59','第8次测评-54.92']:
        archive=WORK/'评测结果'/edition/'scoring_result'
        report=next((archive/'report/submission').iterdir())
        for row in read(archive/'summary.json')['cases']:
            case=row['case']
            prediction=read(archive/f'final_results/{case}_Pred.json')
            truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
            expected=read(report/f'{case}.json')['connectivity']['f1']['f1']
            assert abs(conn(prediction,truth)-expected)<1e-6,(edition,case,conn(prediction,truth),expected)
            checked+=1
    return checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['5','9','10','13'])
    parser.add_argument('--report',default='contacts_small_results.json')
    args=parser.parse_args()
    checked=calibrate()
    work_by_hash={}
    for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
        graph=read(folder/'p/export/engineering_graph_proposals.json')
        work_by_hash[graph['image_sha256']]=folder
    rows=[]
    for case in args.cases:
        base=read(WORK/f'wire_implementation_20261020/cached_predictions/{case}_Pred.json')
        digest=hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
        folder=work_by_hash[digest]
        graph=read(folder/'p/export/engineering_graph_proposals.json')
        pins=read(folder/'p/export/engineering_pin_text_result.json')
        # Construct candidates before loading GT. No annotations enter propose().
        variants={'baseline':(base,[]),'unique_only':propose(base,graph,pins),
                  'unique_or_explicit_primary':propose(base,graph,pins,True)}
        truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT'/case).glob('*.json')))
        before=metrics.score_case(base,truth)
        row={'case':case,'variants':{}}
        for name,(prediction,audit) in variants.items():
            validate_target(prediction)
            assert prediction['components']==base['components']
            assert prediction['nets'].keys()==base['nets'].keys()
            assert all(prediction['nets'][k]['edges']==v['edges'] for k,v in base['nets'].items())
            assert all(prediction['pins'][owner][key]==pin for owner,p in base['pins'].items() for key,pin in p.items())
            scored=metrics.score_case(prediction,truth)
            assert all(scored[k]==before[k] for k in ['component_f1','pin_f1'])
            wire=score_wire(prediction,truth)
            connectivity=conn(prediction,truth)
            score=100*(.30*scored['component_f1']['f1']+.25*scored['pin_f1']['f1']+.35*connectivity+.10*wire['f1'])
            row['variants'][name]={'wire':wire,'connectivity':connectivity,'score_100':score,'contacts':audit}
            destination=ROOT/name
            destination.mkdir(exist_ok=True)
            (destination/f'{case}_Pred.json').write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows.append(row)
    summary={name:{'wire_f1':sum(r['variants'][name]['wire']['f1'] for r in rows)/len(rows),
                   'connectivity_f1':sum(r['variants'][name]['connectivity'] for r in rows)/len(rows),
                   'score_100':sum(r['variants'][name]['score_100'] for r in rows)/len(rows)}
             for name in rows[0]['variants']}
    result={'cases':args.cases,'summary':summary,'per_case':rows,
            'connectivity_calibration_cases':checked,'neural_inference_calls':0,'production_modified':False}
    (ROOT/args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'calibrated_conn_cases':checked,
                     'per_case':[{'case':r['case'],'variants':{n:{'wire':v['wire']['f1'],'conn':round(v['connectivity'],4),
                         'score':round(v['score_100'],3),'contacts':len(v['contacts'])} for n,v in r['variants'].items()}} for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
