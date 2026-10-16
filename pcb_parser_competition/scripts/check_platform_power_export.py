"""Offline power-export diagnostic against matching public images, not a scorer."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'submission/platform_adapter'/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    adapter=module('power_check_adapter','adapter.py')
    exporter=module('power_check_exporter','component_export.py')
    public=Path(read(ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json')['source_root'])/'200_train_cases'
    cases=[]
    for path in sorted((ROOT/'runs/s24_observed_net_fields/predictions').glob('*/export/completed_component_records.json')):
        case=path.parent.parent.name;folder=public/case
        if not folder.is_dir():continue
        public_png=next(folder.glob('*.png'));local_png=ROOT/f'data/standardized/{case}/{case}.png'
        with Image.open(public_png) as a,Image.open(local_png) as b:
            if a.size!=b.size or a.convert('RGB').tobytes()!=b.convert('RGB').tobytes():continue
        original=read(path.parent/'engineering_submission.json');records=read(path)
        summary=read(path.parent.parent/'summary.json')
        ocr_path=next(Path(p) for p in summary['original_S23_input_sha256'] if p.endswith('texts.json'))
        augmented,audit=exporter.export_power_components(original,records,read(ocr_path))
        assert all(augmented['components'][k]==v for k,v in original['components'].items())
        assert all(augmented['pins'][k]==v for k,v in original['pins'].items())
        assert augmented['nets']==original['nets']
        predicted=adapter.to_target(augmented)
        gt=read(next(folder.glob('*.json')))
        powers=lambda p:[(k,v) for k,v in p['components'].items() if v['type'] in {'gnd','v'}]
        truth=powers(gt);pred=powers(predicted)
        cost=np.full((len(truth),len(pred)),1e9)
        for i,(_,a) in enumerate(truth):
            for j,(_,b) in enumerate(pred):
                error=max(abs(x-y) for x,y in zip(a['bbox'],b['bbox']))
                if a['type']==b['type'] and a.get('value')==b.get('value') and error<=20:
                    cost[i,j]=error
        rows,cols=linear_sum_assignment(cost)
        pairs=[{'gt_key':truth[i][0],'pred_key':pred[j][0],'type':truth[i][1]['type'],
                'value':truth[i][1].get('value'),'max_bbox_error':float(cost[i,j])}
               for i,j in zip(rows,cols) if cost[i,j]<1e9]
        cases.append({'case':case,'gt_power':len(truth),'pred_power':len(pred),
                      'matched_power':len(pairs),'value_counts':audit['value_counts'],'matches':pairs})
    totals={k:sum(c[k] for c in cases) for k in ['gt_power','pred_power','matched_power']}
    result={'cases':cases,'totals':totals,'same_pixel_images_verified':True,
            'existing_ordinary_components_pins_and_networks_preserved':True,
            'diagnostic_only':True,'official_score':None,
            'limits':'Cached development pages with offline GT matching; not independent evaluation or official F1.'}
    output=ROOT/'dist/component_power_fix_20261016';output.mkdir(exist_ok=True)
    (output/'public_power_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'cases':len(cases),**totals}))


if __name__=='__main__':main()
