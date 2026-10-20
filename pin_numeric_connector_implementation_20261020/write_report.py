"""Verify deployed release integrity and describe frozen page 13 results."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
report=read(ROOT/'deployed_cached_check.json');plan=read(ROOT/'release_plan.json')
assert len(report['cases'])==13 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
before=read(WORK/'pin_row_library_implementation_20261020/deployed/13_Pred.json')
after=read(ROOT/'deployed/13_Pred.json');audit=read(ROOT/'deployed/13_audit.json')
fresh=read(ROOT/'deployed/13_fresh_audit.json')
for key in ['groups','changes','component_changes','network_changes','network_withheld']:
    assert fresh[key]==audit[key]
assert fresh['ocr_calls']==4
unchanged=[]
for case in map(str,range(1,14)):
    assert read(ROOT/f'deployed/{case}_Pred.json')==read(ROOT/f'staged/{case}_Pred.json')
    if case!='13':
        assert read(ROOT/f'deployed/{case}_Pred.json')==read(WORK/f'pin_row_library_implementation_20261020/deployed/{case}_Pred.json')
        unchanged.append(case)
for owner,comp in before['components'].items():assert after['components'][owner]==comp
for owner,pins in before['pins'].items():
    for key,pin in pins.items():assert after['pins'][owner][key]==pin
for nid,net in before['nets'].items():assert after['nets'][nid]==net
assert set(after['components'])-set(before['components'])=={'RECOVERED_SUPPLY_PIN_CAND_0046'}
assert set(after['pins']['H1'])-set(before['pins']['H1'])=={'pin_3'}
assert set(after['nets'])-set(before['nets'])=={'OBSERVED_CONTACT_PIN_CAND_0046'}
assert after['nets']['OBSERVED_CONTACT_PIN_CAND_0046']=={'hyperGraph':'(H1.3,RECOVERED_SUPPLY_PIN_CAND_0046.1)','edges':{}}
entry=read(ROOT/'entrypoint_deployed_check.json');tests=read(ROOT/'tests_deployed.json')
assert entry['runtime']=='deployed' and len(entry['cases'])==10 and all(r['success'] for r in entry['cases'])
assert tests['runtime']=='deployed' and tests['success'] and tests['tests_run']==252
counts={side:{key:sum(r[side]['pin'][key] for r in report['cases']) for key in ['tp','fp','fn']} for side in ['before','after']}
assert counts=={'before':{'tp':190,'fp':6,'fn':1},'after':{'tp':191,'fp':6,'fn':0}}
page=next(r for r in report['cases'] if r['case']=='13')
assert page['after']['pin']=={'tp':12,'fp':0,'fn':0,'precision':1.,'recall':1.,'f1':1.}
assert all(r['ocr_calls']==0 for r in report['cases'] if r['case']!='13')
# Reference use is diagnostic only, after output predictions were frozen.
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/13').glob('*.json')))
point=lambda d:[d['pins']['H1']['pin_3']['point'][a] for a in ['x','y']]
error=math.dist(point(after),point(truth));assert error==1.5
verification={'release_hashes_verified':True,'unit_tests':tests,'new_tests':34,
              'fully_unchanged_pages':unchanged,'existing_pins_components_and_networks_preserved':True,
              'new_pin':after['pins']['H1']['pin_3'],'pin_error_px':error,
              'new_supply':after['components']['RECOVERED_SUPPLY_PIN_CAND_0046'],
              'only_new_network':after['nets']['OBSERVED_CONTACT_PIN_CAND_0046'],
              'upstream_route_not_reconstructed':True,'existing_wire_segments_preserved':True,
              'fresh_local_ocr_matches_cache':True,'local_ocr_calls_per_run':4,
              'external_api_calls':0,'full_model_runs':0,'aggregate_pin_metrics':counts,
              'predictions_frozen_before_reference_reads':report['predictions_frozen_before_reference_reads'],
              'entrypoint':entry}
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

image=Image.open('D:/pcb_open_gt_in/13.png').convert('RGB');box=(1220,265,1370,450)
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig,axes=plt.subplots(1,2,figsize=(8.5,6),dpi=160)
for ax,doc,title in zip(axes,[before,after],['修复前：3 号引脚与 VIN 未导出','修复后：引脚、VIN 及直接接触恢复']):
    ax.imshow(image.crop(box),extent=(1220,1370,450,265),interpolation='nearest')
    for key,pin in doc['pins']['H1'].items():
        ax.scatter([pin['point']['x']],[797-pin['point']['y']],s=95,facecolors='none',edgecolors='#168263',linewidths=2)
    if doc is before:
        ax.scatter([1260.5],[382],s=130,facecolors='none',edgecolors='#d22e39',linewidths=2)
        desc='红圈处存在候选和文字，但字段未配对\n原结果仅保留 H1 的 1、2 号'
    else:
        c=doc['components']['RECOVERED_SUPPLY_PIN_CAND_0046'];a,b,cx,d=c['bbox']
        ax.add_patch(Rectangle((a,797-d),cx-a,d-b,fill=False,edgecolor='#1676bd',linewidth=1.6))
        desc='3 / 3 端点误差 1.5 像素；蓝框为 VIN\n仅恢复两者的直接连接，上游线路仍待处理'
    ax.set_title(title,fontproperties=font,fontsize=12)
    ax.axis('off');ax.text(.5,-.05,desc,transform=ax.transAxes,ha='center',va='top',fontproperties=font,fontsize=10,linespacing=1.7)
fig.subplots_adjust(bottom=.18,wspace=.2)
fig.savefig(ROOT/'第13页H1修复对照.png',bbox_inches='tight',facecolor='white');plt.close(fig)

text=f'''# 第 13 页 H1.3 与 VIN 修复：落地验证报告

已部署到 `D:\\pcb5_submission`。H1 的 3 号引脚由原图独立文字证据补齐，旁边漏检的 VIN 符号也已恢复。U1 的 9 个引脚及 H1 原有 1、2 号记录不变。

| 本地指标 | 修复前 | 修复后 |
|---|---:|---:|
| 正确引脚 TP | 11 | 12 |
| 误检 / 漏检 FP / FN | 0 / 1 | 0 / 0 |
| 引脚 F1 | 95.65% | 100% |
| 元件 TP / FP / FN | 18 / 1 / 2 | 19 / 1 / 1 |
| 元件 F1 | 92.31% | 95.00% |
| 连接关系指标 | {page['before']['connectivity']:.6%} | {page['after']['connectivity']:.6%} |
| 导线 F1 | 30.95% | 30.95% |
| 综合评分（100 分制） | {page['before']['score_100']:.5f} | {page['after']['score_100']:.5f} |

以上为本地复现评测，非官方提交分数；引脚匹配沿用此前统一的 20 像素容差。新引脚实际误差仅 1.5 像素，5 像素容差下也能通过。引脚 F1 满分不代表整页满分。

## 修复依据

H1.3 已有真实模型候选 PIN_CAND_0046，但原字段 number、pinname 同时为空。缓存 OCR 的框内名称“3”具有两个独立视图，置信度分别为 0.99999、0.99949，未成功配对到该候选。旧字段补全主要选择已有一项身份字段的候选，没有覆盖这种双字段均空、名称本身为数字的情况。

新规则要求至少两个既有数字名称引脚与编号一致，再学习框内文字列、行偏移和同侧引脚长度。H1 的 1、2 号均支持 27.5 像素外伸距离；将该距离用于已找到的第三条真实引脚行，得到左下原点 `(1260.5,415)`。位置仍须落在实际红色引脚上，并接近独立模型候选。不会仅凭 1、2 的顺序推定下一编号。

对“3”单独裁剪再做两次本地 OCR，结果为 3（0.99998、0.99999），与原图两个视图一致。因此导出 `H1.pin_3`，名称为“3”。合并为“123”的多行 OCR 框不参与拆号。

VIN 符号此前连元件检测记录也没有，不属于仅补字段。新规则在该真实接点旁测得连续红色细竖线和对称端帽，再裁剪旁边蓝色文字，独立两次读到 VIN（均 0.99987）。测得符号框左下原点为 `(1243,386,1275,415)`，接触点为 `(1259,415)`，以稳定来源编号导出为 `RECOVERED_SUPPLY_PIN_CAND_0046`，没有使用元件库或参考坐标生成预测。

## 网络范围与剩余问题

新增局部逻辑网络 `OBSERVED_CONTACT_PIN_CAND_0046`，只包含 `(H1.3,RECOVERED_SUPPLY_PIN_CAND_0046.1)`，edges 为空。这两个接触点只相距 1.5 像素，且红色 T 符号直接连接到该物理引脚，局部连接有原图依据。

原模型对 H1.3 的状态为 `no_wire_within_radius`，所以未把它接入附近已有的上游网络。上游存在缺失线路和交点歧义，本次没有恢复它们，也没有添加推测线段。全部既有网络成员、网络编号和导线坐标逐项保持不变。导线 TP/FP/FN 仍为 13/22/36。

## 回归、测试与成本

- 新增 34 项测试，部署后完整 252 项全部通过；覆盖非连续编号、独立视图、冲突文字、候选缺失、错误归属、位置校正、T 形符号、预算、来源校验及不确定连接。
- 13 页使用同图缓存复核，其余 12 页完整 JSON 与上一部署版本完全一致；只有第 13 页的新规则触发 OCR，每次 4 次局部识别。
- 部署后实际做了 4 次新的本地 OCR，最终预测与缓存验证一致；重复执行不再做 OCR，也不再修改输出。
- 第 1、2、4、5、6、8、10、11、12、13 页实际导出入口检查通过，IC 引脚、新 VIN 和全部网络与验证输出一致。
- 13 页引脚 TP/FP/FN 从 190/6/1 变为 191/6/0，即评分范围内的漏检清零。第 1 页仍保留 6 个误检，未宣称全部页面引脚精度满分。
- 本地综合分逐页均值从 {report['macro_before']:.5f} 提升到 {report['macro_after']:.5f}。
- 预测先冻结后评分。未调用外部 API，未完整重跑任何页面模型。

部署五个文件：`agent.py`、`src/pcb_parser/pin_numeric_connector.py`、配置、测试、说明。部署后 SHA-256 全部核验通过，原入口备份于本目录 `backup/`。默认启用，设置 `PCB_PIN_NUMERIC_CONNECTOR=0` 可关闭。

证据：`verification.json`、`deployed_cached_check.json`、`deployed_fresh_check.json`、`tests_deployed.json`、`entrypoint_deployed_check.json`、`deployed/13_audit.json`。13 页最新预测在 `deployed/`，实际导出结果在 `cached_entrypoint/`。

![第 13 页 H1 修复对照]({(ROOT/'第13页H1修复对照.png').as_posix()})
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps({'release_verified':True,'tests':252,'unchanged_pages':len(unchanged),'counts':counts,'H1_3_error_px':error,'report':str(ROOT/'落地验证报告.md')},ensure_ascii=True))
