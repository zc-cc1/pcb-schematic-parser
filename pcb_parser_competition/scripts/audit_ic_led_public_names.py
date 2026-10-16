"""Read-only IC/LED conventions audit across public and live result versions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    public=Path(read(ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json')['source_root'])
    result={'source_root':str(public),'source_untouched':True,'splits':{},'same_image_label_versions':[]}
    hashes={};all_rows=[];png_index={}
    for split in ['200_train_cases','10_GTcase']:
        names=defaultdict(Counter);dims=defaultdict(Counter);values=defaultdict(Counter);page_modes=Counter()
        rows=[]
        for path in sorted((public/split).rglob('*.json')):
            hashes[str(path)]=sha(path)
            doc=read(path);png=next(path.parent.glob('*.png'));png_hash=sha(png);png_index[png_hash]=str(png)
            boxes=[(cid,c) for cid,c in doc['components'].items() if c['type'] in {'box','ic'}]
            if boxes:
                equals=sum(c.get('Name')==cid for cid,c in boxes)
                page_modes['all_key' if equals==len(boxes) else 'all_other' if equals==0 else 'mixed']+=1
            for cid,c in doc['components'].items():
                if c['type'] not in {'box','ic','led'}:continue
                typ=c['type'];name=c.get('Name');b=c['bbox']
                role='null' if name is None else 'empty' if name=='' else 'equals_key' if name==cid else 'other'
                names[typ][role]+=1;dims[typ][(b[2]-b[0],b[3]-b[1])]+=1
                values[typ]['null' if c.get('value') is None else str(c['value'])]+=1
                row={'split':split,'file':str(path),'png':str(png),'png_sha256':png_hash,
                     'key':cid,'component':c,'Name_role':role}
                rows.append(row);all_rows.append(row)
        result['splits'][split]={
            'name_counts':{k:dict(v) for k,v in names.items()},
            'bbox_dimensions':{k:[{'width':wh[0],'height':wh[1],'count':n} for wh,n in counts.most_common()] for k,counts in dims.items()},
            'value_counts':{k:dict(v) for k,v in values.items()},'IC_page_Name_modes':dict(page_modes),'rows':rows}
    for gt_path in sorted((public/'10_GTcase').rglob('*.json')):
        case=gt_path.parent.name[:4]
        train=next((public/'200_train_cases'/case).glob('*.json'))
        a,b=read(train),read(gt_path)
        if sha(next(train.parent.glob('*.png')))!=sha(next(gt_path.parent.glob('*.png'))):continue
        for cid,c in b['components'].items():
            old=a['components'].get(cid)
            if c['type'] not in {'box','ic','led'} or old is None:continue
            result['same_image_label_versions'].append({'case':case,'key':cid,'identical_png_bytes':True,
                'train_file':str(train),'Golden_file':str(gt_path),'train_component':old,'Golden_component':c})
    audit=read(ROOT/'dist/fourth_eval_28_92_audit/audit.json');live=Path(audit['new_root'])
    live_led=[]
    for case in audit['cases']:
        pred=read(live/f"final_results/{case['case']}_Pred.json")
        report_path=live/f"report/submission/20261016T085457Z/{case['case']}.json"
        report=read(report_path);hashes[str(report_path)]=sha(report_path)
        errors={e['component_id']:e for e in report['component']['diagnostics']['error_examples'] if 'failed_checks' in e}
        assert len(errors)==report['component']['diagnostics']['matched_component_count']-report['component']['f1']['tp']
        for cid,c in pred['components'].items():
            if c['type']!='led':continue
            checks=errors.get(cid,{}).get('failed_checks',[])
            live_led.append({'case':case['case'],'key':cid,'prediction':c,'reported_failed_checks':checks,
                             'Name_flagged_as_mismatch':'Name' in checks})
    result['live_LED_evidence']=live_led
    result['exact_public_png_hash_matches']={case['case']:png_index.get(case['image_sha256']) for case in audit['cases']}
    result['limits']=['Other Name values are literal annotations, not automatically all valid model numbers.',
                     'No hidden online GT raw Name or original image is included in the live reports.',
                     'Same part model or same case number does not establish identical test image.',
                     'No IC/LED output rules or model weights changed by this audit.']
    output=ROOT/'dist/ic_led_public_confirmation_20261016';output.mkdir(exist_ok=True)
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'source_sha256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'Name_counts':{k:v['name_counts'] for k,v in result['splits'].items()},
                      'live_LED_Name_unflagged':sum(not x['Name_flagged_as_mismatch'] for x in live_led),
                      'live_LED_Name_flagged':sum(x['Name_flagged_as_mismatch'] for x in live_led)},ensure_ascii=True))


if __name__=='__main__':main()
