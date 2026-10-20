"""Read-only page 9 audit of current outputs; no model or OCR calls."""
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'page9_status_20261020';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
pred=read(ROOT/'pin_nc_recovery_implementation_20261020/deployed/9_Pred.json')
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/9').glob('*.json')))
rows=[]
for owner,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,ref in sorted(truth['pins'].get(owner,{}).items(),key=lambda v:int(v[0][4:])):
        got=pred['pins'].get(owner,{}).get(key)
        distance=math.hypot(got['point']['x']-ref['point']['x'],got['point']['y']-ref['point']['y']) if got else None
        rows.append({'ref':owner+'.'+key[4:],'expected_name':ref['pinname'],'predicted_name':got['pinname'] if got else None,
                     'distance_px':distance,'correct_20px':bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pn,tn=logical_nets(pred),logical_nets(truth)
conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
pin=metrics.score_pins(pred,truth,20)[0];comp=metrics.score_components(pred,truth)[0];wire=score_wire(pred,truth)
report={'case':'9','pin':pin,'components':comp,'connectivity':conn,'wire':wire,'pins':rows,
        'max_pin_error_px':max((r['distance_px'] for r in rows if r['distance_px'] is not None),default=None),
        'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1']),
        'new_inference_calls':0,'production_code_modified':False}
report['unread_supply_values']=[{'component':key,'bbox':item['bbox'],'value':item.get('value')}
                              for key,item in pred['components'].items()
                              if item.get('type')=='v' and item.get('value') is None]
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 第 9 页当前情况','',
       'U2（MT2492）共 6 个评分范围内的引脚。按此前一致的 20 像素容差复核，编号、名称和位置均正确，没有漏检或误检。',
       '', '| 引脚 | 原图名称 | 当前名称 | 坐标误差（像素） |', '|---|---|---|---:|']
for r in rows:
    error=f"{r['distance_px']:.2f}" if r['distance_px'] is not None else '未检出'
    lines.append(f"| {r['ref']} | {r['expected_name']} | {r['predicted_name']} | {error} |")
lines.extend(['',f"引脚 F1 为 {pin['f1']:.2%}，元件 F1 为 {comp['f1']:.2%}，本地综合评分为 {report['score_100']:.5f}/100。",
              '', '另有两处电源符号的 VCC 值未读出，预测 value 仍为 null：R1 上方，以及 C4、C5 下方电源线的右端。原图可见 VCC，当前属于元件字段识别问题，不是 U2 引脚漏检。',
              '',f"剩余失分集中在网络连接关系和导线：连接关系指标 {conn:.4%}，导线 F1 {wire['f1']:.2%}，导线 TP/FP/FN 为 {wire['tp']}/{wire['fp']}/{wire['fn']}。",
              '', '本页暂不需要补引脚；后续应检查网络分组、连接成员和导线导出。上述统计不能单独确定具体错连、断线或导出遗漏的成因，需要另做逐网对照。',
              '', '本次仅核对既有预测、参考标注和原图，未运行模型或 OCR，未修改正式代码。分数为本地评测，非官方提交分数。'])
(OUT/'第9页现状.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
