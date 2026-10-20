"""Audit the frozen latest page 12 output, without inference or code changes."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'page12_status_20261020'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets, _assign_nets, score_wire

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
source = ROOT/'pin_row_library_implementation_20261020/deployed/12_Pred.json'
pred = read(source)
truth = read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/12').glob('*.json')))
rows = []
for owner, comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):
        continue
    for key, ref in sorted(truth['pins'].get(owner, {}).items(), key=lambda kv: int(kv[0][4:])):
        got = pred['pins'].get(owner, {}).get(key)
        distance = math.dist([got['point'][a] for a in ['x','y']], [ref['point'][a] for a in ['x','y']]) if got else None
        rows.append({'ref': owner+'.'+key[4:], 'expected_name': ref['pinname'],
                     'predicted_name': got['pinname'] if got else None, 'distance_px': distance,
                     'correct_20px': bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pn, tn = logical_nets(pred), logical_nets(truth)
conn = 2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
pin = metrics.score_pins(pred,truth,20)[0]
components = metrics.score_components(pred,truth)[0]
wire = score_wire(pred,truth)
missing = [key for key,c in truth['components'].items() if metrics.ntype(c['type']) not in metrics.POWER_KIND and key not in pred['components']]
assert missing == ['VIN','LED1','VIN_2','VIN_3','VIN_4']
assert pred['components']['L1']['value'] is None and truth['components']['L1']['value']=='2.2uH'
report = {'case':'12', 'source_prediction':str(source), 'pin':pin, 'components':components,
          'connectivity':conn, 'wire':wire, 'pins':rows,
          'max_pin_error_px':max(r['distance_px'] for r in rows),
          'missing_ordinary_components':missing, 'missing_component_fields':[{'owner':'L1','field':'value','expected':'2.2uH','actual':None}],
          'score_100':100*(.3*components['f1']+.25*pin['f1']+.35*conn+.1*wire['f1']),
          'new_inference_calls':0, 'production_code_modified':False}
assert pin['tp']==9 and pin['fp']==pin['fn']==0
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines = ['# 第 12 页当前情况','',
         '当前最新输出来自第 11 页修复后的部署缓存。U1（SY6912AFCC）的 9 个 IC 引脚编号、名称和位置均正确，按此前一致的 20 像素容差，没有漏检或误检。','',
         '| 引脚 | 名称 | 坐标误差（像素） |','|---|---|---:|']
lines.extend(f"| {r['ref']} | {r['predicted_name']} | {r['distance_px']:.3f} |" for r in rows)
lines.extend(['',f"最大坐标误差为 {report['max_pin_error_px']:.3f} 像素，引脚 F1 为 100%。",'',
              '| 本地指标 | 当前结果 |','|---|---:|',
              f"| 元件 F1 | {components['f1']:.2%} |",
              f"| 连接关系指标 | {conn:.2%} |",
              f"| 导线 F1 | {wire['f1']:.2%} |",
              f"| 导线 TP / FP / FN | {wire['tp']} / {wire['fp']} / {wire['fn']} |",
              f"| 综合评分（100 分制） | {report['score_100']:.5f} |",'',
              '原图及当前输出可直接核对的外围问题：',
              '- LED1 未出现在最终元件表中。',
              '- 四处 VIN 端口均未导出：右上、R3 右侧、左下跳接点旁、LED1 下方。参考表中分别为 VIN、VIN_4、VIN_3、VIN_2。',
              '- L1 的 value 为 null，原图可见 2.2uH。',
              '', '本页暂不需要继续补 U1 引脚。下一步适合验证 VIN 端口与 LED1 的候选及导出链路，再处理 L1 数值和相关网络成员。仅凭本次评分不能确定每条导线的具体错误原因，也未将外围缺项的根因归结为 OCR 或检测模型。',
              '', '本次仅核对已有预测、参考标注和原图，没有调用 OCR、模型或外部 API，未修改正式代码。以上为本地评测，非官方提交分数；引脚 F1 满分不代表整页满分。'])
(OUT/'第12页现状.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
