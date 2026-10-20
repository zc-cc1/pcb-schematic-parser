"""Audit page 8 using retained outputs only."""
import json
import math
import sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'page8_status_20261020'
OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sys.path.insert(0,str(ROOT/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
from platform_wire_metrics import logical_nets,_assign_nets,score_wire
pred=read(ROOT/'pin_number_cycle_implementation_20261020/deployed/8_Pred.json')
truth=read(next((ROOT/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/8').glob('*.json')))
rows=[]
for key,ref in sorted(truth['pins']['U1'].items(),key=lambda kv:int(kv[0][4:])):
    got=pred['pins']['U1'].get(key)
    distance=math.dist(list(got['point'].values()),list(ref['point'].values())) if got else None
    rows.append({'ref':'U1.'+key[4:],'name':ref['pinname'],'detected':got is not None,
                 'predicted_name':got['pinname'] if got else None,'error_px':distance,
                 'correct_20px':bool(got and got['pinname']==ref['pinname'] and distance<=20)})
pin=metrics.score_pins(pred,truth,20)[0]
comp=metrics.score_components(pred,truth)[0]
wire=score_wire(pred,truth)
pn,tn=logical_nets(pred),logical_nets(truth)
connectivity=2*sum(v for _,_,v in _assign_nets(pn,tn))/max(1,len(pn)+len(tn))
result={'case':'8','pin':pin,'components':comp,'wire':wire,'connectivity':connectivity,
        'score_100':100*(.3*comp['f1']+.25*pin['f1']+.35*connectivity+.1*wire['f1']),
        'pins':rows,'new_inference_calls':0,'production_code_modified':False,
        'geometry_evidence':read(ROOT/'pin_geometry_implementation_20261020/deployed/8_audit.json')}
(OUT/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Image.open('D:/pcb_open_gt_in/8.png').crop((945,640,1065,780)).resize((600,700)).save(OUT/'NC引脚原图放大.png')
text='''# 第 8 页当前情况

U1（PAW3395DM-T6QU）共 16 个引脚，当前正确 14 个、漏检 2 个、误检 0 个，引脚 F1 为 93.33%。此前几何与 OCR 补充已经找回 U1.6 NC，当前缺失的是 U1.2 NC 和 U1.16 NC。

| 剩余漏检 | 原图位置 | 现有流程的阻塞点 |
|---|---|---|
| U1.2 NC | 左下方四个 NC 中的第二行 | 短引脚位置与 NC 名称已找到，但没有达到双视图、0.98 置信度要求的数字编号证据，审计状态为 insufficient_fields。 |
| U1.16 NC | 左下方四个 NC 中的第四行，NCS 上方 | 短引脚位置与 NC 名称已找到，但该行目前得到的编号投票是 6，与已有 U1.6 重复，审计状态为 existing_number，未补成 16。 |

这组 NC 引脚间距约 20 像素，左侧绿色不连接叉号紧贴短引脚和数字。同名 NC 不能区分身份。16 行目前读取/关联到了 6，可能包含漏掉数字 1 或相邻行文本串入，尚不能仅凭现有审计区分这两种原因。

建议只针对这两行做窄范围数字裁剪、屏蔽彩色叉号并放大，再配合逐行文本归属约束验证；不按同名 NC 或缺失编号集合直接猜补。由于 NC 表示不连接，补全引脚身份也不应自动新增网络连接。

本次仅核对原图、当前预测和既有 OCR 审计，没有执行新的推理或修改正式代码。
'''
text+=f"\n本地综合评分 {result['score_100']:.5f}/100，连接关系指标 {connectivity:.4%}，导线 F1 {wire['f1']:.2%}。均为本地评测，非官方提交分数。\n"
(OUT/'第8页现状.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'geometry_evidence','pins'}},ensure_ascii=False))
print(json.dumps({'missing':[r for r in rows if not r['correct_20px']]},ensure_ascii=False))
