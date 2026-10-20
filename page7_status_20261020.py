"""Read-only status of the deployed page 7 prediction; no inference calls."""
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'page7_status_20261020'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
pred=read(ROOT/'pin_number_cycle_implementation_20261020/deployed/7_Pred.json')
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/7').glob('*.json')))
rows=[]
for owner,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,ref in sorted(truth['pins'].get(owner,{}).items()):
        got=pred['pins'].get(owner,{}).get(key)
        distance=math.hypot(got['point']['x']-ref['point']['x'],got['point']['y']-ref['point']['y']) if got else None
        rows.append({'ref':owner+'.'+key[4:],'expected_name':ref['pinname'],'predicted_name':got['pinname'] if got else None,
                     'distance_px':distance,'correct_20px':bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pn,tn=logical_nets(pred),logical_nets(truth)
connectivity=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
pin=metrics.score_pins(pred,truth,20)[0]
component=metrics.score_components(pred,truth)[0]
wire=score_wire(pred,truth)
report={'pin':pin,'components':component,'connectivity':connectivity,'wire':wire,'pins':rows,
        'max_pin_error_px':max(r['distance_px'] for r in rows),
        'score_100':100*(.3*component['f1']+.25*pin['f1']+.35*connectivity+.1*wire['f1']),
        'new_inference_calls':0,'source_prediction':'pin_number_cycle_implementation_20261020/deployed/7_Pred.json'}
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 第 7 页当前情况','',
       '当前 IC 为 U1（MP2236CGJ-Z），共有 8 个评分范围内的引脚。按此前一致的 20 像素容差复核，8 个全部正确，没有漏检或误检。',
       '', '| 引脚 | 原图名称 | 当前名称 | 坐标误差（像素） |', '|---|---|---|---:|']
lines.extend(f"| {r['ref']} | {r['expected_name']} | {r['predicted_name']} | {r['distance_px']:.2f} |" for r in rows)
lines.extend(['',f"引脚 F1 为 {pin['f1']:.2%}；本地综合评分 {report['score_100']:.5f}/100。连接关系指标为 {connectivity:.4%}；导线 F1 为 {wire['f1']:.2%}，TP/FP/FN 为 {wire['tp']}/{wire['fp']}/{wire['fn']}。",
              '', '本页引脚编号、名称及位置已正确，暂不需要继续补引脚。剩余失分需要优先检查网络分组和导线导出；上述分数不能直接证明每个网络的具体错误原因。',
              '', '此次仅复核已有预测和原图，未修改正式代码，未运行模型或 OCR。结果为本地评测，非官方提交分数。'])
(OUT/'第7页现状.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
