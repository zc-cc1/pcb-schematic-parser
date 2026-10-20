"""Read-only status of page 11 after the deployed page-10 repair."""
import json
import math
import sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'page11_status_20261020';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
pred=read(ROOT/'pin_vertical_repair_implementation_20261020/deployed/11_Pred.json')
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/11').glob('*.json')))
rows=[];component_fields=[]
for owner,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    actual=pred['components'].get(owner,{})
    if actual.get('Name')!=comp['Name']:component_fields.append({'owner':owner,'actual_name':actual.get('Name'),'expected_name':comp['Name']})
    for key,ref in truth['pins'].get(owner,{}).items():
        got=pred['pins'].get(owner,{}).get(key)
        distance=math.hypot(got['point']['x']-ref['point']['x'],got['point']['y']-ref['point']['y']) if got else None
        rows.append({'owner':owner,'ref':owner+'.'+key[4:],'expected_name':ref['pinname'],'predicted':got,
                     'expected_point':ref['point'],'error_px':distance,
                     'correct_20px':bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pin=metrics.score_pins(pred,truth,20)[0];comp=metrics.score_components(pred,truth)[0];wire=score_wire(pred,truth)
pn,tn=logical_nets(pred),logical_nets(truth)
conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
owners={o:{'correct':sum(r['correct_20px'] for r in rows if r['owner']==o),'total':sum(r['owner']==o for r in rows)} for o in sorted({r['owner'] for r in rows})}
extras=[]
for owner in owners:
    for key,value in pred['pins'].get(owner,{}).items():
        if key not in truth['pins'].get(owner,{}):extras.append({'owner':owner,'key':key,'record':value})
report={'case':'11','pin':pin,'components':comp,'wire':wire,'connectivity':conn,
        'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1']),
        'per_owner':owners,'pins':rows,'extra_pin_keys':extras,'component_name_errors':component_fields,
        'new_inference_calls':0,'production_code_modified':False}
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=Image.open('D:/pcb_open_gt_in/11.png')
for label,box in [('U1',(900,325,1140,485)),('USB1',(75,245,325,435)),('USB2',(550,320,755,475))]:
    crop=source.crop(box);crop.resize((crop.width*2,crop.height*2)).save(OUT/f'{label}原图放大.png')
text=f'''# 第 11 页当前情况

当前正确 11/20 个 IC 引脚，评分 TP/FP/FN = {pin['tp']}/{pin['fp']}/{pin['fn']}，引脚 F1 为 {pin['f1']:.2%}。9 个未正确得分中，1 个是编号错误，8 个没有对应导出记录；不能将 9 个未得分都理解为图像中找不到引脚。

| 元件 | 正确 / 总数 | 当前剩余问题 |
|---|---:|---|
| U1（TP4333） | 7 / 8 | VDD 实际 6 号，被导出为 9 号。 |
| USB1（TYPE-C 6P(073)） | 4 / 7 | 缺 7/EH、A5/CC1、B5/CC2。 |
| USB2（AF 90° WJDG） | 0 / 5 | 缺 1/VCC、2/D-、3/D+、4/GND、5/SHLD。 |

## 可直接从原图继续验证的四个引脚

| 引脚 | 当前证据与阻塞点 |
|---|---|
| U1.6 VDD | 原位置和 VDD 名称已存在，原图打印 6；既有旋转 OCR 读作 9，置信度仅 0.95902，导出 pin_9 后未继续纠错。需局部确认方向，按真实数字重编号并同步网络引用。 |
| USB1.7 EH | 原图有 7、EH；左侧候选却被绑定为右侧编号 A5，且号码为空。原始文字证据中 EH 已高置信度读出，需要约束同侧文字配对。 |
| USB1.A5 CC1 | A5 编号被辅助候选占用，主候选缺编号/名称；正确 CC1 文字出现在既有证据中。需先合并同一物理引脚的主辅证据，支持字母加数字编号，再重审名称。 |
| USB1.B5 CC2 | B5 已由多个视图读出，CC2 也有多视图高置信度证据，但主辅候选分裂和占号使它未导出。不能只放宽去重或把 B5 截成 5。 |

此前 U1.2 BAT 和 U1.8 LED1 已补回，本次不再列为漏检；其当前记录均在 20 像素评分容差内。U1 的 OUT 有约 8 像素位置偏差，BAT 有约 9.5 像素偏差，仍正确得分，但后续端点优化可以复查。

## USB2 的五个引脚为何不同

原图打印了 VCC、D-、D+、GND 四个名称，但没有打印对应数字编号；左侧外壳接点既没有编号，也未印 SHLD。现有模型能找到这些接点，部分名称也已读取，因此不是完全没有几何候选。

二次 OCR 可以改善名称绑定，不能读出原图不存在的编号和 SHLD。若要补这五个，需要先根据可核验型号和符号布局确定连接器模板，再验证对应关系；不能使用参考答案直接填入。其型号 AF 90° WJDG 当前也未恢复，仍显示为 USB2。

## 其他失分及验证范围

U1、USB1、USB2 的模型名称目前分别仍为 U1、USB1、USB2，未恢复图中 TP4333、TYPE-C 6P(073)、AF 90° WJDG，这也造成元件字段失分。

当前本地综合评分 {report['score_100']:.5f}/100，元件 F1 {comp['f1']:.2%}，连接关系指标 {conn:.4%}，导线 F1 {wire['f1']:.2%}。本次仅核对原图、最新预测和已留存候选证据，没有修改正式代码，没有新增模型或 OCR 调用。分数为本地评测，非官方提交结果。

建议先验证 U1.6 和 USB1 的三个有明确原图文字的引脚；USB2 单独做可信模板验证。

![U1 原图放大]({(OUT/'U1原图放大.png').as_posix()})

![USB1 原图放大]({(OUT/'USB1原图放大.png').as_posix()})

![USB2 原图放大]({(OUT/'USB2原图放大.png').as_posix()})
'''
(OUT/'第11页现状.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='pins'},ensure_ascii=False))
print(json.dumps({'incorrect':[r['ref']+' '+r['expected_name'] for r in rows if not r['correct_20px']]},ensure_ascii=False))
