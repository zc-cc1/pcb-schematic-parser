import json
from pathlib import Path
from PIL import Image
import math

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
pred=read(ROOT/'deployed/2_Pred.json')
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/2').glob('*.json')))
image=Image.open('D:/pcb_open_gt_in/2.png');h=image.height
inventory=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='2')
folder=Path(inventory['model_cache'])
model=read(folder/'p/export/engineering_pin_text_result.json')
records=read(folder/'p/export/completed_component_records.json')['records']
raw=read('D:/pcb_open_gt_v37dots_b/component_results/2_Pred.json')
print('SIZE',image.size)
for label,data in [('PRED',pred),('GT',truth),('RAW',raw)]:
    print(label,json.dumps({'component':data['components']['U2'],'pins':data['pins']['U2']},ensure_ascii=False))
points=[[p['point']['x'],h-p['point']['y']] for p in pred['pins']['U2'].values()]
for p in model['pin_candidates']:
    if p.get('owner')=='U2' or p.get('owner_component_id')=='U2' or any(math.dist(p['point'],q)<8 for q in points):
        print('CANDIDATE',json.dumps(p,ensure_ascii=False))
print('RECORD',json.dumps([r for r in records if r['component_id']=='U2'],ensure_ascii=False))
print('INVENTORY',json.dumps(inventory,ensure_ascii=False))
