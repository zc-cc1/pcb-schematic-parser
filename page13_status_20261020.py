"""Read-only audit of latest page 13 predictions and retained OCR evidence."""
import json
import math
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'page13_status_20261020'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets, _assign_nets, score_wire
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
source = ROOT/'pin_row_library_implementation_20261020/deployed/13_Pred.json'
pred = read(source)
truth = read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/13').glob('*.json')))
inventory = next(r for r in read(ROOT/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='13')
model = read(Path(inventory['model_cache'])/'p/export/engineering_pin_text_result.json')
candidate = next(r for r in model['pin_candidates'] if r['candidate_id']=='PIN_CAND_0046')
ocr = read('D:/pcb_open_gt_v37dots_b/logs/13_ocr_texts.json')
text = next(r for r in ocr['texts'] if r['text_id']=='text_0026')
assert candidate['owner']=='H1' and candidate['number'] is None and candidate['pinname'] is None
assert text['text']=='3' and text['confidence']>=.99
rows=[]
for owner,comp in truth['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,ref in sorted(truth['pins'].get(owner,{}).items(),key=lambda kv:int(kv[0][4:])):
        got=pred['pins'].get(owner,{}).get(key)
        distance=math.dist([got['point'][a] for a in ['x','y']], [ref['point'][a] for a in ['x','y']]) if got else None
        rows.append({'ref':owner+'.'+key[4:],'expected_name':ref['pinname'],
                     'predicted_name':got['pinname'] if got else None,'distance_px':distance,
                     'correct_20px':bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pin=metrics.score_pins(pred,truth,20)[0]
comp=metrics.score_components(pred,truth)[0]
pn,tn=logical_nets(pred),logical_nets(truth)
conn=2*sum(s for _,_,s in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
wire=score_wire(pred,truth)
missing=[r for r in rows if not r['correct_20px']]
assert [r['ref'] for r in missing]==['H1.3']
assert (pin['tp'],pin['fp'],pin['fn'])==(11,0,1)
report={'case':'13','source_prediction':str(source),'pin':pin,'components':comp,'connectivity':conn,'wire':wire,
        'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*conn+.1*wire['f1']),
        'pins':rows,'missing_pins':missing,'max_detected_pin_error_px':max(r['distance_px'] for r in rows if r['distance_px'] is not None),
        'missing_candidate':candidate,'existing_ocr_text_for_missing_pin':text,
        'missing_supply_symbol':{'value':'VIN','location':'H1 pin 3 junction'},
        'new_inference_calls':0,'production_code_modified':False}
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Image.open('D:/pcb_open_gt_in/13.png').convert('RGB').crop((1220,265,1370,450)).resize((450,555)).save(OUT/'H1局部.png')
lines=['# 第 13 页当前情况','',
       '按此前统一的 20 像素引脚容差复核：共 12 个评分范围内的引脚，11 个正确，漏检 1 个，误检 0 个。U1（RT7272BGSP）9 个全部正确；H1 的 1、2 号正确，3 号缺失。','',
       '| 引脚 | 应有名称 | 当前名称 | 坐标误差（像素） |','|---|---|---|---:|']
for r in rows:
    distance=f"{r['distance_px']:.3f}" if r['distance_px'] is not None else '未导出'
    lines.append(f"| {r['ref']} | {r['expected_name']} | {r['predicted_name'] or '缺失'} | {distance} |")
lines.extend(['','## H1.3 漏检原因','',
              '原图右侧三针连接器的最下方接点，位于 VIN 线上。原模型已有 PIN_CAND_0046，位置为左上原点坐标 (1252.812,381.908)，归属 H1、状态 assigned。其编号和名称均为空，状态 no_reliable_text，最终没有形成 pin_3 记录。',
              '', '但既有整图 OCR 缓存并非完全读不到该文字：text_0026 已读出“3”，置信度 0.99999；另一个视图为 0.99949，文字框在元件边框内。缓存还出现将纵向 1、2、3 合并为“123”的识别。可确定问题发生在文字与物理引脚的配对或后续字段接收阶段；本次未进一步声称某一条过滤条件已被验证为唯一根因。',
              '', '因此应优先验证同侧引脚行与内外两列数字的配对，在真实候选和多视图数字一致时补齐 3/3，并检查连接引用。没有证据时不能只凭 1、2 推定 3。H1 没有可识别的精确型号，不适合直接套用第 11 页的精确型号库补全规则。',
              '', '## 其他失分','',
              f"- 引脚 F1：{pin['f1']:.2%}；元件 F1：{comp['f1']:.2%}。",
              '- H1 的 3 号接点处 VIN 电源符号也没有导出；另一处 VIN（R5 下方）和上方 VOUT 已导出。',
              f"- 连接关系指标：{conn:.2%}；导线 F1：{wire['f1']:.2%}，TP/FP/FN = {wire['tp']}/{wire['fp']}/{wire['fn']}。",
              f"- 本地综合评分：{report['score_100']:.5f}/100。",
              '', '导线和网络仍有明显失分，补齐 H1.3 后仍需独立检查各条连接，不能由本次统计直接认定具体断线或错连原因。',
              '', '本次仅检查冻结预测、原图、参考标注和保留的模型/OCR 缓存，没有进行新的推理或修改正式代码。分数为本地复现，非官方提交结果。','',
              f"![H1 局部]({(OUT/'H1局部.png').as_posix()})"])
(OUT/'第13页现状.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
