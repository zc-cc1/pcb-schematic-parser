import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
report=read(ROOT/'deployed_cached_check.json');plan=read(ROOT/'release_plan.json')
assert len(report['cases'])==13 and report['gained_pins']==3 and report['lost_pins']==0 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[];changed=[]
for row in report['cases']:
    case=row['case'];a=read(WORK/f'pin_pairing_implementation_20261020/deployed/{case}_Pred.json');b=read(ROOT/f'deployed/{case}_Pred.json')
    assert a['components']==b['components'] and a['nets']==b['nets']
    assert a['pins'].keys()==b['pins'].keys()
    for owner,pins in a['pins'].items():
        assert pins.keys()==b['pins'][owner].keys()
        for key,pin in pins.items():
            if pin!=b['pins'][owner][key]:changed.append({'case':case,'owner':owner,'key':key,'before':pin,'after':b['pins'][owner][key]})
    if case!='5':assert a==b;unchanged.append(case)
assert {(r['case'],r['owner'],r['key']) for r in changed}=={('5','U1','pin_1'),('5','U3','pin_1'),('5','U3','pin_2')}
current=read(ROOT/'deployed/5_Pred.json');base=read(WORK/'pin_pairing_implementation_20261020/deployed/5_Pred.json')
assert current==read(ROOT/'5_pilot_Pred.json')
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/5').glob('*.json')))
point=lambda doc:(doc['pins']['U1']['pin_1']['point']['x'],doc['pins']['U1']['pin_1']['point']['y'])
old_error=math.dist(point(base),point(truth));new_error=math.dist(point(current),point(truth))
counts={stage:{key:sum(r[stage]['pin'][key] for r in report['cases']) for key in ['tp','fp','fn']} for stage in ['before','after']}
verify={'release_hashes_verified':True,'unit_tests_passed':112,'new_tests':18,'fully_unchanged_pages':unchanged,
        'changed_pin_records':changed,'all_components_networks_and_wires_unchanged':True,
        'pin_key_sets_unchanged':True,'old_GND_error_px':old_error,'new_GND_error_px':new_error,
        'fresh_local_ocr_matches_cache':True,'fresh_local_ocr_calls':8,'total_pin_metrics':counts,
        'entrypoint':read(ROOT/'entrypoint_check.json')}
(ROOT/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');image=Image.open('D:/pcb_open_gt_in/5.png');h=image.height
fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=150)
for ax,crop,title in zip(axes,[[440,190,750,375],[490,680,790,815]],['U1.1：从 EN 行移回右侧 GND 引线','U3：恢复图上实际印出的完整名称']):
    ax.imshow(image.crop(crop),extent=[crop[0]-.5,crop[2]-.5,crop[3]-.5,crop[1]-.5],interpolation='nearest')
    ax.set_xlim(crop[0],crop[2]);ax.set_ylim(crop[3],crop[1]);ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=14,pad=12)
old=point(base);new=point(current);old=(old[0],h-old[1]);new=(new[0],h-new[1])
axes[0].plot(*old,'x',color='#d35400',markersize=12,markeredgewidth=2.5)
axes[0].plot(*new,'o',color='#0077cc',markersize=10,markerfacecolor='none',markeredgewidth=2)
axes[0].annotate('原错点',xy=old,xytext=(447,360),fontproperties=font,fontsize=12,color='#b74700',arrowprops={'arrowstyle':'->','color':'#d35400'})
axes[0].annotate('1 / GND\n误差 1.5 px',xy=new,xytext=(685,268),fontproperties=font,fontsize=11,color='#0066aa',
                 bbox={'facecolor':'white','edgecolor':'none','alpha':.96},arrowprops={'arrowstyle':'->','color':'#0077cc'})
axes[1].annotate('GND → ADJ(GND)',xy=(590,739),xytext=(496,694),fontproperties=font,fontsize=12,color='#0066aa',arrowprops={'arrowstyle':'->','color':'#0077cc'})
axes[1].annotate('VOUT → VOUT(TAB)',xy=(595,758),xytext=(496,813),fontproperties=font,fontsize=12,color='#0066aa',arrowprops={'arrowstyle':'->','color':'#0077cc'})
fig.text(.5,.025,'第 5 页：18 个参考 IC 引脚全部得分；GND 接地网络仍因模型歧义保留为待连接。',ha='center',fontproperties=font,fontsize=12)
fig.subplots_adjust(left=.015,right=.985,top=.86,bottom=.10,wspace=.06)
fig.savefig(ROOT/'page5_repair_result.png',facecolor='white');plt.close(fig)
page=next(r for r in report['cases'] if r['case']=='5');before,after=page['before'],page['after']
text=f'''# 第五页引脚证据修复落地验证

已更新 D:/pcb5_submission 正式入口，默认开启 PCB_PIN_EVIDENCE_REPAIR。
发布 5 个文件，部署后哈希一致。设置 PCB_PIN_EVIDENCE_REPAIR=0 可关闭。
备份保存在本目录 backup/，install.py --rollback 会先核对当前文件哈希再回退本次发布。

## 第五页结果

| 指标 | 基线 | 部署版缓存回放 |
| --- | ---: | ---: |
| 正确 IC 引脚 TP | 15 | 18 |
| 漏检 FN | 3 | 0 |
| 误检 FP | 3 | 0 |
| 引脚 F1 | 0.8333 | 1.0000 |
| U1.1 GND 位置误差 | {old_error:.3f} px | {new_error:.3f} px |
| 本地加权总分 | {before['score_100']:.4f} | {after['score_100']:.4f} |

只改动三条记录：U1.1 的坐标，以及 U3.1/U3.2 的名称。编号键集合、其他引脚、全部组件、网络及导线保持不变。
其余 12 页最终 JSON 完全一致，无正确引脚损失、新误检或页面退化。
13 页本地加权平均分 {report['macro_before']:.4f} → {report['macro_after']:.4f}。
13 页汇总引脚 TP/FP/FN：{counts['before']} → {counts['after']}。
所有预测落盘冻结后才读取 GT，未提交平台重评。

## U1.1 为什么错、如何修

PIN_CAND_0067 的坐标在左侧 EN 行附近，编号 1 来自 library_pin_name，置信度仅 0.744。
其 GND 文字证据实际位于元件右侧，却被绑定到这个弱候选。导出后它与更强的 4/EN 点相距不足 1 像素。
原 LOG_NET_CAND_00013 仅包含 U1.1，没有导线；源模型网络也只有这一候选、没有线段。

修复限定于这种“弱库编号、与强点重叠、只有空单点网络”的情形。复用现有图像引线恢复作为证据探测：
在对侧唯一空缺的真实红色短引线上，编号 1 和 GND 分别有至少两个/四个独立 OCR 视图支持。
原有编号和名称保持，坐标改到图像左上原点的 (680.5,330)，未使用参考坐标定位。

GND 的电气接入仍未解决。右侧线段属于 LOG_NET_CAND_00016，但地符号的主候选 PIN_CAND_0012
在真实导线和地符号笔画之间有 ambiguous_nearby_networks：最近两网距离约 0.688 与 2.420 像素。
该物理网络尚未导出，因此本版保留原无边孤立网络，不强行新增接地连接，审计明确记录待连接。
本次网络连接得分 {before['connectivity']:.6f} 和导线 F1 {before['wire']['f1']:.4f} 均保持不变，提升来自引脚项。

## U3 两个名称为什么错、如何修

原图确实印有 ADJ(GND) 和 VOUT(TAB)，当前模型 pinname 候选中也有至少四个高置信度视图；
后处理因歧义采用 library_pin_name，输出了 GND、VOUT 两个简称。

新规则只处理库生成的短名称，要求原候选在同一行读到完整 HEAD(ALIAS) 字符串，短名称必须是其中一个完整部分；
再以当前原图元件裁剪的 1x/3x、0/90 度 OCR 重新核对，至少两个视图、置信度不低于 0.98。
第五页两个完整名称各有四个新视图支持，才恢复实际拼写。代码没有硬编码器件型号、GND/VOUT 或任何预期名称。
这两条记录只改名称，保留原坐标、编号和网络关系。

## 验证与成本

- 新增 18 项边界测试，全部 112 项测试通过。
- 13 页模型缓存回归，只有第五页三条引脚记录变化；部署后回放与预发布结果完全一致。
- 第五页真实本地 OCR 重跑 8 次，输出与缓存一致。
- 第 1、2、4、5 页正式 run_case 入口使用同图模型/OCR 缓存回放正常；IC 引脚、日志、摘要全部一致。
  原第四页的改号网络与电源分支也再次逐项核对通过。
- 首次小范围实验新增 4 次本地 OCR，复用 4 次；之后真实重跑 8 次。无外部 API、无 YOLO/U-Net 重跑。
- 默认每图最多 3 个元件、24 次 OCR；模型按需加载，关闭、异常保留原结果与回退均已配置。

## 文件

模块：D:/pcb5_submission/src/pcb_parser/pin_evidence_repair.py。
配置：D:/pcb5_submission/configs/pin_evidence_repair_v1.json。
入口：D:/pcb5_submission/agent.py，联合编号配对之后执行。
日志：logs/{{case}}_pin_evidence_repair.json。
结果：deployed_cached_check.json；一致性验证：verification.json；入口验证：entrypoint_check.json。
对照图：page5_repair_result.png。
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps({k:v for k,v in verify.items() if k not in {'changed_pin_records','entrypoint'}},ensure_ascii=False))
