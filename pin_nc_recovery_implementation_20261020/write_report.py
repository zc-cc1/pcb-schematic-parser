"""Save deployment evidence and a before/after review crop."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=read(ROOT/'release_plan.json');report=read(ROOT/'deployed_cached_check.json')
assert report['gained_pins']==2 and report['lost_pins']==0 and not report['regressed_pages']
assert len(report['cases'])==13
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[];added=[]
for row in report['cases']:
    case=row['case'];a=read(WORK/f'pin_number_cycle_implementation_20261020/deployed/{case}_Pred.json');b=read(ROOT/f'deployed/{case}_Pred.json')
    assert a['components']==b['components'] and a['nets']==b['nets']
    for owner,pins in a['pins'].items():
        for key,pin in pins.items():assert b['pins'][owner][key]==pin
    for owner,pins in b['pins'].items():
        for key,pin in pins.items():
            if key not in a['pins'].get(owner,{}):added.append({'case':case,'owner':owner,'key':key,'pin':pin})
    if case!='8':assert a==b;unchanged.append(case)
assert {(r['case'],r['owner'],r['key']) for r in added}=={('8','U1','pin_2'),('8','U1','pin_16')}
audit=read(ROOT/'deployed/8_audit.json');fresh=read(ROOT/'staged_fresh_ocr_audit.json')
assert fresh['changes']==audit['changes'] and fresh['groups']==audit['groups'] and fresh['ocr_calls']==12
entry=read(ROOT/'entrypoint_deployed_check.json')
assert entry['runtime']=='deployed' and all(r['success'] for r in entry['cases'])
counts={s:{k:sum(r[s]['pin'][k] for r in report['cases']) for k in ['tp','fp','fn']} for s in ['before','after']}
verification={'release_hashes_verified':True,'unit_tests_passed':156,'new_tests':23,
              'unchanged_pages':unchanged,'added_pins':added,'all_existing_pin_records_preserved':True,
              'all_components_networks_and_wires_preserved':True,
              'fresh_local_ocr_matches_cache':True,'fresh_local_ocr_calls':12,
              'external_api_calls':0,'full_model_runs':0,'aggregate_pins':counts,
              'entrypoint':entry}
write(ROOT/'verification.json',verification)
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
source=Image.open('D:/pcb_open_gt_in/8.png').convert('RGB');crop=source.crop((945,640,1065,780))
fig,axes=plt.subplots(1,2,figsize=(10,5),dpi=160)
for ax,title,color,state in zip(axes,['修复前：缺少 2、16 号','修复后：两个 NC 均补全'],['#cc7700','#137a43'],['缺失','已补全']):
    ax.imshow(crop,extent=(945,1065,780,640),interpolation='nearest')
    for number,y in [('2',690),('16',730)]:
        ax.scatter([977.5],[y],s=100,facecolors='none',edgecolors=color,linewidths=1.8)
        ax.annotate(f'{number} 号 NC\n{state}',xy=(977.5,y),xytext=(1080,y-2),fontproperties=font,
                    color=color,fontsize=11,va='center',arrowprops={'arrowstyle':'->','color':color})
    ax.set_xlim(945,1180);ax.set_ylim(780,640);ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=13)
fig.tight_layout();fig.savefig(ROOT/'第8页NC修复对照.png',bbox_inches='tight',facecolor='white');plt.close(fig)
page=next(r for r in report['cases'] if r['case']=='8')
text=f'''# 第 8 页 NC 引脚补全：落地验证报告

修复已部署到 `D:\\pcb5_submission`。U1（PAW3395DM-T6QU）原有 14 个正确引脚；本次从原图独立识别并补入 2 号 NC、16 号 NC。未使用型号库、页号或剩余编号集合猜补。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 正确引脚 TP | 14 | 16 |
| 误检 FP | 0 | 0 |
| 漏检 FN | 2 | 0 |
| 引脚 F1 | 93.33% | 100% |
| 本地综合评分（100 分制） | {page['before']['score_100']:.5f} | {page['after']['score_100']:.5f} |

以上是本地评测，非官方提交分数。连接关系指标仍为 81.7143%，导线 F1 仍为 78%，导线 TP/FP/FN 为 39/11/11。NC 表示不连接，此次补入身份记录后保持不接网。

## 原因与图像证据

旧逻辑逐个连通块识别数字，要求单块高度至少 6 像素。原图 16 的前导 1 被绿色叉号分成高度 5 和 4 像素的两块，被旧过滤丢掉，只剩 6。新逻辑保留同一行完整数字组，局部原图检测 OCR 两次读出 16，置信度 0.99904、0.99486。

2 号数字体积很小。保留中性灰度抗锯齿边缘、去除彩色线条并对整组数字裁剪后，两次读出 2，置信度 0.98598、0.98861；原图单行检测在另一尺度也读出 2，置信度 0.99994。两行 NC 名称均由两次局部 OCR 确认，置信度大于 0.999。

补入位置来自真实红色短引脚端点：2 号为 `(977.5,502)`、16 号为 `(977.5,462)`（左下原点）。二者距参考标注均为 3.5 像素，满足此前统一使用的 20 像素容差。

## 生效条件与保护

- 必须有真实红色元件边框、短引脚及绿色不连接叉号的两条对角线；叉号中心允许小范围栅格偏移，外侧不能存在延伸导线。
- 同侧至少两个现有编号有原始多视图 OCR 证据，用来确认数字位于引脚上方的布局；邻行至少相隔 18 像素。
- 使用单行原图检测、完整中性灰度数字组和元件内侧名称裁剪，固定 3 倍及 5 倍尺度。至少两视图达到 0.98 置信度，名称必须为精确 NC；任何高置信度数字冲突均放弃。
- 已有引脚键、已有网络引用、冲突的模型身份、重复提议、边缘截断及预算不足均阻止补入。原有引脚、元件、网络成员和全部导线坐标逐项保持不变。
- 本规则只覆盖具有明确绿色叉号的横向 NC 短引脚；对模糊数字、其他方向或其他名称不做推断。

## 验证结果与成本

- 新增 23 项边界测试，部署后的完整测试共 156 项，全部通过。
- 第 8 页额外执行 12 次真实本地 OCR，与缓存实验结果完全一致；最终规则处理本页需要 12 次局部 OCR，其他 12 页未触发 OCR。
- 13 页复用既有模型缓存做回归，增加 2 个正确引脚、无丢失；其余 12 页完整 JSON 相同。全部预测先冻结，再读取参考标注评分。
- 部署后的实际导出流程复核第 1、2、4、5、6、8 页，所有引脚结果与验证版本一致，日志正常落盘；模型与 OCR 推理使用同图缓存。
- 实验阶段仅做本地 OCR 裁剪与预处理比较，缓存保存在 `ocr_cache/`；外部 API 调用、完整模型重跑均为 0。
- 13 页引脚 TP/FP/FN 从 {counts['before']['tp']}/{counts['before']['fp']}/{counts['before']['fn']} 变为 {counts['after']['tp']}/{counts['after']['fp']}/{counts['after']['fn']}，本地综合评分逐页均值从 {report['macro_before']:.5f} 变为 {report['macro_after']:.5f}。

## 部署记录

新增 `src/pcb_parser/pin_nc_recovery.py`、配置、测试和说明，在 `agent.py` 的编号交换修复后接入。部署 5 个文件均已逐项验证 SHA-256；原文件备份位于本目录 `backup/`，默认启用，设置 `PCB_PIN_NC_RECOVERY=0` 可关闭。

证据文件：`deployed_cached_check.json`、`verification.json`、`entrypoint_deployed_check.json`、`deployed/8_audit.json`、`staged_fresh_ocr_audit.json`。最终 13 页预测位于 `deployed/`，真实入口导出位于 `cached_entrypoint/`。

![第 8 页 NC 引脚修复对照]({(ROOT/'第8页NC修复对照.png').as_posix()})
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':str(ROOT/'落地验证报告.md'),'unchanged_pages':len(unchanged),'counts':counts,'release_verified':True},ensure_ascii=False))
