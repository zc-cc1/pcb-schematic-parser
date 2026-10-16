"""Public cached detector boxes -> image-only refinement -> separate diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from pcb_parser.ic_geometry import locate_ic_rectangle,refine_ic_boxes


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def metrics(rows,key):
    values=[row[key] for row in rows]
    return {'count':len(values),'within_5px':sum(v<=5 for v in values),'within_20px':sum(v<=20 for v in values),
            'mean_max_error_px':float(np.mean(values)),'max_error_px':max(values)}


def main():
    previous=read(ROOT/'dist/ic_analysis_20261016/public_geometry.json')['paired_predictions']
    public=Path(read(ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json')['source_root'])
    cache_path=ROOT/'runs/detector_comparison_v3/adopted_fused_cpu.json'
    cache=read(cache_path)
    manifest_path=ROOT/'data/detection_v1/manifest.jsonl'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==cache['manifest_sha256']
    manifest={r['case_id']:r for r in map(json.loads,manifest_path.read_text(encoding='utf-8').splitlines())}
    cached={r['case_id']:r for r in cache['cases']}
    rows=[];images={};start=time.perf_counter()
    for old in previous:
        case=old['case']
        if case not in images:
            image=next((public/'200_train_cases'/case).glob('*.png'))
            rgb=np.array(Image.open(image).convert('RGB'))
            local=ROOT/f'data/standardized/{case}/{case}.png'
            assert np.array_equal(rgb,np.array(Image.open(local).convert('RGB')))
            assert hashlib.sha256(local.read_bytes()).hexdigest()==manifest[case]['image_sha256']
            images[case]=rgb
        rgb=images[case];height=rgb.shape[0];b=old['raw_bbox'];proposal=[b[0],height-b[3],b[2],height-b[1]]
        assert any(b==[p['bbox'][0],height-p['bbox'][3],p['bbox'][2],height-p['bbox'][1]]
                   for p in cached[case]['predictions'] if p['class_name']=='ic')
        found,audit=locate_ic_rectangle(rgb,proposal)
        q=[found[0],height-found[3],found[2],height-found[1]] if found else b
        error=max(abs(a-b) for a,b in zip(q,old['gt_bbox']))
        rows.append({**old,'after_bbox':q,'after_error':error,'accepted':found is not None,'audit':audit})
    report={'before':metrics(rows,'max_error'),'after':metrics(rows,'after_error'),
            'accepted':sum(r['accepted'] for r in rows),'reasons':dict(Counter(r['audit']['reason'] for r in rows)),
            'regressed_at_20px':[r for r in rows if r['max_error']<=20<r['after_error']],
            'regressed_at_5px':[r for r in rows if r['max_error']<=5<r['after_error']],
            'seconds':time.perf_counter()-start,'rows':rows,
            'inference_source_sha256':hashlib.sha256((ROOT/'src/pcb_parser/ic_geometry.py').read_bytes()).hexdigest(),
            'cached_detector_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            'all_prediction_images_verified':True,
            'limits':'Paired public cached detections, development diagnostics only. GT used for association/metrics, never for refinement.'}
    out=ROOT/'dist/ic_bbox_fix_20261016';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fixtures=out/'compatibility-input';fixtures.mkdir(exist_ok=True)
    for case in ['0001','0074','0123']:
        rgb=images[case];height=rgb.shape[0]
        before={'components':{},'pins':{},'nets':{}}
        for i,p in enumerate(cached[case]['predictions']):
            if p['class_name']!='ic':continue
            a,t,r,bot=p['bbox']
            before['components'][f'DET_{i}']={'type':'box','Name':'KEEP_OBSERVED_NAME','value':None,'bbox':[a,height-bot,r,height-t]}
        expected,audit=refine_ic_boxes(rgb,before)
        image=ROOT/f'data/standardized/{case}/{case}.png'
        shutil.copy2(image,fixtures/f'{case}.png')
        (fixtures/f'{case}.json').write_text(json.dumps({'before':before,'expected':expected,
            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','regressed_at_20px','regressed_at_5px']}))
    print('regressed20',[(r['case'],r['key'],r['max_error'],r['after_error']) for r in report['regressed_at_20px']])
    print('regressed5',[(r['case'],r['key'],r['max_error'],r['after_error']) for r in report['regressed_at_5px']])


if __name__=='__main__':main()
