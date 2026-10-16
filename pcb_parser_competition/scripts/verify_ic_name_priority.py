"""Replay current-image S24 OCR/records; reference labels are metrics only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.ic_name_selection import category, select_ic_names


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    out=ROOT/'dist/ic_name_priority_20261016'
    spec=importlib.util.spec_from_file_location('ic_priority_adapter',ROOT/'submission/platform_adapter/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    summaries=read(ROOT/'runs/s24_observed_net_fields/predictions/batch_summary.json')['cases']
    public=Path(read(ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json')['source_root'])
    pages=[];changes=[];counts={'ICs':0,'null_before':0,'null_after':0,'model_before':0,'model_changed':0}
    public_metrics={'paired_ICs':0,'before_exact_Name':0,'after_exact_Name':0}
    for summary in summaries:
        case=summary['case_id'];stage=ROOT/f'runs/s24_observed_net_fields/predictions/{case}'
        raw_path=stage/'export/engineering_submission.json';records_path=stage/'export/completed_component_records.json'
        ocr_path=next(Path(p) for p in summary['original_S23_input_sha256'] if p.endswith('texts.json'))
        assert sha(ocr_path)==summary['original_S23_input_sha256'][str(ocr_path)]
        raw,records,ocr=read(raw_path),read(records_path),read(ocr_path)
        image=ROOT/f'data/standardized/{case}/{case}.png'
        assert sha(image)==records['image_sha256']==ocr['image_sha256']==summary['image_sha256']
        before=adapter.to_target(raw);after,audit=select_ic_names(before,records,ocr)
        adapter.validate_target(after)
        control=deepcopy(after)
        for key,c in before['components'].items():
            if c['type'] not in {'box','ic'}:continue
            counts['ICs']+=1;counts['null_before']+=c['Name'] is None
            counts['null_after']+=after['components'][key]['Name'] is None
            if c['Name'] and category(c['Name'])=='model':
                counts['model_before']+=1;counts['model_changed']+=c['Name']!=after['components'][key]['Name']
            control['components'][key]['Name']=c['Name']
            if c['Name']!=after['components'][key]['Name']:
                changes.append({'case':case,'key':key,'before':c['Name'],'after':after['components'][key]['Name'],
                    'category':next(e['selected_category'] for e in audit['components'] if e['component_id']==key)})
        assert control==before,'Changed fields outside IC Name'
        gp=public/'200_train_cases'/case
        if gp.is_dir():
            from PIL import Image
            with Image.open(image) as a,Image.open(next(gp.glob('*.png'))) as b:
                assert a.size==b.size and a.convert('RGB').tobytes()==b.convert('RGB').tobytes()
            gt=read(next(gp.glob('*.json')))
            for key,c in before['components'].items():
                if c['type']=='box' and key in gt['components'] and gt['components'][key]['type']=='box':
                    public_metrics['paired_ICs']+=1
                    public_metrics['before_exact_Name']+=c['Name']==gt['components'][key]['Name']
                    public_metrics['after_exact_Name']+=after['components'][key]['Name']==gt['components'][key]['Name']
        pages.append({'case':case,'image_sha256':sha(image),'OCR_sha256':sha(ocr_path),
                      'records_sha256':sha(records_path),'audit':audit})
        if case in {'0036','0042','0170'}:
            save(out/'container-input'/f'{case}.json',{'case':case,'before':before,'records':records,'ocr':ocr,'expected':after})
    online=read(ROOT/'dist/ic_analysis_20261016/online_ICs.json')['online_ICs']
    nonempty=[row for row in online if row['prediction']['Name'] is not None]
    assert all(category(row['prediction']['Name'])=='model' for row in nonempty)
    result={'policy':['model','reference','description'],'replayed_pages':len(pages),'counts':counts,
            'changes':changes,'public_training_Name_metrics':public_metrics,
            'online_nonempty_Name_categories':{'count':len(nonempty),'all_model':True,
                'scope':'Category check only; online raw OCR/records unavailable, not an online replay.'},
            'pages':pages,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [
                ROOT/'src/pcb_parser/ic_name_selection.py',ROOT/'submission/platform_adapter/agent.py',
                ROOT/'submission/platform_adapter/adapter.py']},
            'model_inference_runs':0,'official_score':None,
            'limits':['Cached image predictions only; no new OCR accuracy claim.',
                      'Model-before-reference priority does not resolve mixed GT Name conventions.',
                      'Preserved existing nonempty models can still contain OCR errors.']}
    save(out/'verification.json',result)
    print(json.dumps({k:result[k] for k in ['replayed_pages','counts','changes','public_training_Name_metrics','online_nonempty_Name_categories']}))


if __name__=='__main__':main()
