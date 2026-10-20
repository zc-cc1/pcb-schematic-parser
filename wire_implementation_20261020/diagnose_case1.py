import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
from platform_wire_metrics import score_wire_detailed,logical_nets
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/1').glob('*.json')))
data=read(ROOT/'wire_validation_small_20261020/remaining_ten_results.json')
row=next(r for r in data['per_case'] if r['case']=='1')
for name in row['variants']:
    prediction=read(ROOT/f'wire_validation_small_20261020/{name}/1_Pred.json')
    counts,detail=score_wire_detailed(prediction,truth)
    print(name,counts,'audit',row['variants'][name]['restored'])
    if name in {'baseline','restore_physical_contacts'}:
        print('pairs',detail['matched_net_pairs'])
        print('power groups',[(n.source_ids,sorted(n.vertices)) for n in logical_nets(prediction)
                               if any(v[0]=='rail' for v in n.vertices)])
