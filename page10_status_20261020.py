"""Read-only page 10 audit of retained predictions and their source evidence."""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'page10_status_20261020';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
pred=read(ROOT/'pin_nc_recovery_implementation_20261020/deployed/10_Pred.json')
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/10').glob('*.json')))
rows=[]
for owner,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,ref in sorted(truth['pins'].get(owner,{}).items(),key=lambda item:int(item[0][4:])):
        got=pred['pins'].get(owner,{}).get(key)
        distance=math.hypot(got['point']['x']-ref['point']['x'],got['point']['y']-ref['point']['y']) if got else None
        correct=bool(got and got['pinname']==ref['pinname'] and distance<=20)
        rows.append({'owner':owner,'ref':owner+'.'+key[4:],'expected_name':ref['pinname'],
                     'predicted':got,'expected_point':ref['point'],'error_px':distance,'correct_20px':correct})
pin=metrics.score_pins(pred,truth,20)[0];comp=metrics.score_components(pred,truth)[0];wire=score_wire(pred,truth)
pn,tn=logical_nets(pred),logical_nets(truth)
conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
per_owner={owner:{'correct':sum(r['correct_20px'] for r in rows if r['owner']==owner),
                  'total':sum(r['owner']==owner for r in rows)} for owner in sorted({r['owner'] for r in rows})}
report={'case':'10','pin':pin,'components':comp,'wire':wire,'connectivity':conn,
        'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1']),
        'per_owner':per_owner,'pins':rows,'new_inference_calls':0,'production_code_modified':False,
        'U2_network_membership':{nid:net['hyperGraph'] for nid,net in pred['nets'].items()
                                 if any(v.strip().startswith('U2.') for v in net['hyperGraph'].strip('()').split(','))}}
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Image.open('D:/pcb_open_gt_in/10.png').crop((1550,215,1795,450)).resize((490,470)).save(OUT/'U2原图放大.png')
text=f'''# 第 10 页当前情况

共 12 个 IC 引脚，正确 9 个；评分统计为 TP/FP/FN = {pin['tp']}/{pin['fp']}/{pin['fn']}，引脚 F1 {pin['f1']:.2%}，本地综合评分 {report['score_100']:.5f}/100。

U1（EG1192L）9 个引脚全部正确，包括此前几何与 OCR 补回的 U1.4 VIA。剩余问题全部集中在 U2（CJ78L05）。U2 三个编号键均有输出，但身份和坐标不正确，因此在评分中各形成误检及漏检，而非没有任何记录。

| 应有引脚 | 当前输出 | 问题 |
|---|---|---|
| U2.1 Vout | Vout，(1623.34,630) | 名称、编号正确；参考位置 (1624,609)，误差 21.01 像素，超过 20 像素门槛。 |
| U2.2 GND | Vin，(1702.19,630) | 2 号被填为 Vin，位置在右下方；实际 2 号 GND 在元件上方，参考位置 (1664,767)。 |
| U2.3 Vin | GND，(1703.55,630) | 3 号位置与错误的 2 号基本重叠在右下方；名称被填为 GND，实际应为 Vin，参考位置 (1703,609)。 |

## 原因证据

1. U2 在本图中的布局是上方 2/GND、左下 1/Vout、右下 3/Vin。原始候选 PIN_CAND_0053 位于上方 (1664.20,252.33，左上原点)，接近真实 2 号位置；其多视图数字 2 和 GND 文字置信度均约 0.9999，但状态为 ambiguous_text、text_used_by_other_pin，没有形成正确导出记录。
2. 右下 Vin 候选 PIN_CAND_0063 的原始数字证据其实是 3，最终却由 library_pin_name 补成 2；另一个同位置候选 PIN_CAND_0041 被输出成 3/GND。因此不能只做 2、3 键名交换，还须找回上方 GND 的物理候选，并按原图重新配对。
3. 左下 Vout 候选本来靠近元件底边；现有端点校正使用约 18 像素长度，导出 y=630。原图该元件底部红色引脚约 40 像素，参考端点 y=609。需要依据真实引脚长度校正，不能仅改字段。

建议先验证原图三组编号/名称的逐侧匹配、去重和真实端点恢复，再核对网络成员引用是否随物理引脚同步修复。第 6 页仅换编号的修复规则不直接覆盖本例，因为本例的上方 GND 候选被遗漏，底部又有重复候选和位置偏差。

连接关系指标 {conn:.4%}，导线 F1 {wire['f1']:.2%}。本次仅核对原图、预测及候选审计，未运行 OCR 或模型，未修改正式代码。分数为本地评测，非官方提交分数。

![U2 原图放大]({(OUT/'U2原图放大.png').as_posix()})
'''
(OUT/'第10页现状.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='pins'},ensure_ascii=False))
print(json.dumps({'incorrect':[r for r in rows if not r['correct_20px']]},ensure_ascii=False))
