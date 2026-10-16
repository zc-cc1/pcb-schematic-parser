"""Image-only LED inference, followed by separate public-reference diagnostics."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'artifacts/ultralytics'))
os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np
from PIL import Image
import torch
from pcb_parser.led_geometry import POLICY,refine_led_boxes,yolo_predictor


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metric(rows,field):
    errors=[row[field] for row in rows]
    return {'count':len(errors),'within_5px':sum(v<=5 for v in errors),
            'within_20px':sum(v<=20 for v in errors),'mean_max_coordinate_error_px':float(np.mean(errors)),
            'max_coordinate_error_px':max(errors)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('led_adapter',ROOT/'submission/platform_adapter/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    previous=read(ROOT/'dist/official_public_dataset_audit_20261016/geometry_diagnostic.json')
    public=Path(read(ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json')['source_root'])
    assert sha(ROOT/previous['prediction_cache'])==previous['cache_sha256']
    cache=read(ROOT/previous['prediction_cache'])
    manifest_path=ROOT/'data/detection_v1/manifest.jsonl'
    assert sha(manifest_path)==cache['manifest_sha256']
    assert sha(ROOT/'weights/detector_dataset300_v3_nano.pt')==cache['weights_sha256']
    manifest={r['case_id']:r for r in map(json.loads,manifest_path.read_text(encoding='utf-8').splitlines())}
    cache={row['case_id']:row for row in cache['cases']}
    wanted=defaultdict(list)
    for row in previous['pairs']:
        if row['type']=='led':wanted[row['case']].append(row)
    predict=yolo_predictor(ROOT/'weights/detector_dataset300_v3_nano.pt',args.device)
    rows=[];stress=[];pages=[];start=time.perf_counter()
    for case,pairs in wanted.items():
        image=next((public/'200_train_cases'/case).glob('*.png'))
        rgb=np.array(Image.open(image).convert('RGB'));height=rgb.shape[0]
        local=ROOT/f'data/standardized/{case}/{case}.png'
        assert np.array_equal(rgb,np.array(Image.open(local).convert('RGB')))
        assert sha(local)==manifest[case]['image_sha256']
        raw={'components':{},'pins':{},'nets':{}}
        detections=[p for p in cache[case]['predictions'] if p['class_name']=='led']
        for i,p in enumerate(detections):
            a,b,c,d=p['bbox']
            raw['components'][f'DETECTION_{i}']={'type':'led','Name':None,'value':None,'bbox':[a,height-d,c,height-b]}
        old=adapter.to_target(raw)
        page_start=time.perf_counter();new,audit=refine_led_boxes(rgb,raw,old,predict);seconds=time.perf_counter()-page_start
        for pair in pairs:
            key=f"DETECTION_{pair['prediction_index_within_class']}"
            q=new['components'][key]['bbox'];gt=pair['gt_bbox']
            assert raw['components'][key]['bbox']==pair['raw_bbox']
            rows.append({'case':case,'public_reference_key':pair['gt_key'],'prediction_id':key,
                         'before_bbox':old['components'][key]['bbox'],'after_bbox':q,'reference_bbox':gt,
                         'before_error_px':pair['proposed_max_error_px'],'after_error_px':max(abs(a-b) for a,b in zip(q,gt))})
        pages.append({'case':case,'image_sha256':sha(image),'cached_detector_image_sha256':sha(local),
                      'seconds':seconds,'audit':audit})
        # Perturbed detector seeds are robustness examples, not new test images
        # or an official score. Keep all neighboring predicted LEDs in scope.
        if case in {'0016','0105','0107','0131','0132','0141','0147','0183'}:
            pair=pairs[0];key=f"DETECTION_{pair['prediction_index_within_class']}";gt=pair['gt_bbox']
            for dx,dy in [(12,0),(-24,0),(0,18),(0,-24)]:
                seeded=deepcopy(raw);b=seeded['components'][key]['bbox']
                seeded['components'][key]['bbox']=[b[0]+dx,b[1]-dy,b[2]+dx,b[3]-dy]
                initial=adapter.to_target(seeded)
                refined,details=refine_led_boxes(rgb,seeded,initial,predict)
                b0=initial['components'][key]['bbox'];b1=refined['components'][key]['bbox']
                entry=next(e for e in details['entries'] if e['component_id']==key)
                stress.append({'case':case,'prediction_id':key,'seed_shift_top_left':[dx,dy],
                               'before_error_px':max(abs(a-b) for a,b in zip(b0,gt)),
                               'after_error_px':max(abs(a-b) for a,b in zip(b1,gt)),
                               'local_reason':entry['local_detection']['reason'],'audit':entry})
        print(json.dumps({'case':case,'paired_LEDs':len(pairs),'seconds':round(seconds,3)}),flush=True)
    result={'policy':POLICY,'device':args.device,'seconds':time.perf_counter()-start,
            'baseline':{'before':metric(rows,'before_error_px'),'after':metric(rows,'after_error_px')},
            'perturbed_seed_stress':{'before':metric(stress,'before_error_px'),'after':metric(stress,'after_error_px')},
            'rows':rows,'stress_rows':stress,'pages':pages,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/pcb_parser/led_geometry.py',ROOT/'weights/detector_dataset300_v3_nano.pt',ROOT/'submission/platform_adapter/agent.py']},
            'limits':['Paired public detections only; missing/unpaired LEDs are excluded.',
                      'Development diagnostics, not independent recognition F1 or official rescoring.',
                      'The online failed PNGs and GT boxes are unavailable; their recovery is unverified.',
                      'Reference labels are read for metrics only; the inference API receives pixels and predictions.']}
    out=ROOT/'dist/led_bbox_fix_20261016/verification.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['device','seconds','baseline','perturbed_seed_stress']}))


if __name__=='__main__':main()
