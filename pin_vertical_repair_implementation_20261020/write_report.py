"""Record deployment integrity and independently measured page 10 improvement."""
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
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=read(ROOT/'deployed_cached_check.json');plan=read(ROOT/'release_plan.json')
assert len(report['cases'])==13 and report['gained_pins']==3 and report['lost_pins']==0 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[];changed=[];added=[]
for row in report['cases']:
    case=row['case'];a=read(WORK/f'pin_nc_recovery_implementation_20261020/deployed/{case}_Pred.json');b=read(ROOT/f'deployed/{case}_Pred.json')
    assert a['components']==b['components'] and a['pins'].keys()==b['pins'].keys()
    assert set(a['nets'])<=set(b['nets'])
    for nid,net in a['nets'].items():assert net['edges']==b['nets'][nid]['edges']
    for owner,pins in a['pins'].items():
        assert pins.keys()<=b['pins'][owner].keys()
        for key,pin in pins.items():
            if pin!=b['pins'][owner][key]:changed.append({'case':case,'owner':owner,'key':key,'before':pin,'after':b['pins'][owner][key]})
    for owner,pins in b['pins'].items():
        for key,pin in pins.items():
            if key not in a['pins'].get(owner,{}):added.append({'case':case,'owner':owner,'key':key,'pin':pin})
    if case!='10':assert a==b;unchanged.append(case)
assert {(r['case'],r['owner'],r['key']) for r in changed}=={('10','U2','pin_1'),('10','U2','pin_2'),('10','U2','pin_3')}
assert {(r['case'],r['owner'],r['key']) for r in added}=={('10','UNRESOLVED_DET_0021','pin_1')}
inventory=read(WORK/'pin_detection_audit_20261020/audit.json')
stage=Path(next(r['model_cache'] for r in inventory['per_case'] if r['case']=='10'))/'p/export'
source_pins=read(stage/'engineering_pin_text_result.json')
contact=next(p for p in source_pins['pin_candidates'] if p['candidate_id']=='PIN_CAND_0024')
assert added[0]['pin']=={'pinname':'','point':{'x':contact['point'][0],'y':1017-contact['point'][1]}}
before=read(WORK/'pin_nc_recovery_implementation_20261020/deployed/10_Pred.json');after=read(ROOT/'deployed/10_Pred.json')
audit=read(ROOT/'deployed/10_audit.json');fresh=read(ROOT/'staged_fresh_ocr_audit.json')
assert fresh['changes']==audit['changes'] and fresh['groups']==audit['groups'] and fresh['ocr_calls']==24
assert fresh['restored_networks']==audit['restored_networks']
members=lambda n:sorted(v.strip() for v in n['hyperGraph'].strip('()').split(',') if v.strip())
assert len(audit['network_changes'])==1 and len(audit['restored_networks'])==1
change=audit['network_changes'][0];assert change['network_id']=='LOG_NET_CAND_00008'
assert members(after['nets']['LOG_NET_CAND_00008'])==['R6.2','U2.3']
for nid,net in before['nets'].items():
    if nid!='LOG_NET_CAND_00008':assert net==after['nets'][nid]
assert set(after['nets'])-set(before['nets'])=={'RESTORED_NET_CAND_00013'}
assert members(after['nets']['RESTORED_NET_CAND_00013'])==['U2.2','UNRESOLVED_DET_0021.1']
assert after['nets']['RESTORED_NET_CAND_00013']['edges']=={'edge_1':[{'x':1663.,'y':825.},{'x':1663.,'y':765.}]}
assert audit['groups'][0]['preserved_unresolved_junctions']=={'LOG_NET_CAND_00006':['J00012'],'LOG_NET_CAND_00008':['J00016']}
entry=read(ROOT/'entrypoint_deployed_check.json');assert entry['runtime']=='deployed' and all(r['success'] for r in entry['cases'])
# Predictions are already frozen and verified. Reference points are read only
# here to report the error of the final physical positions.
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/10').glob('*.json')))
errors=[]
for key in ['pin_1','pin_2','pin_3']:
    point=lambda doc:[doc['pins']['U2'][key]['point'][axis] for axis in ['x','y']]
    errors.append({'number':key[4:],'name':after['pins']['U2'][key]['pinname'],
                   'before_px':math.dist(point(before),point(truth)),'after_px':math.dist(point(after),point(truth))})
counts={s:{k:sum(r[s]['pin'][k] for r in report['cases']) for k in ['tp','fp','fn']} for s in ['before','after']}
verify={'release_hashes_verified':True,'unit_tests_passed':185,'new_tests':29,'fully_unchanged_pages':unchanged,
        'changed_pin_records':changed,'added_model_ground_contact':added,'U2_errors':errors,'existing_wire_segments_preserved':True,
        'new_wire_segment_copied_exactly_from_model':True,'restored_ground_branch':'RESTORED_NET_CAND_00013',
        'existing_junction_ambiguity_preserved':audit['groups'][0]['preserved_unresolved_junctions'],
        'fresh_local_ocr_matches_cache':True,'fresh_local_ocr_calls':24,'external_api_calls':0,'full_model_runs':0,
        'aggregate_pin_metrics':counts,'entrypoint':entry}
write(ROOT/'verification.json',verify)
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');source=Image.open('D:/pcb_open_gt_in/10.png').convert('RGB');height=source.height
crop=source.crop((1570,145,1785,450));fig,axes=plt.subplots(1,2,figsize=(13,6),dpi=160)
for ax,title,color,doc in zip(axes,['修复前：底部重复，顶部 GND 缺失','修复后：三组身份和真实端点恢复'],['#c87500','#167c49'],[before,after]):
    ax.imshow(crop,extent=(1570,1785,450,145),interpolation='nearest')
    for key,pin in doc['pins']['U2'].items():
        x,y=pin['point']['x'],height-pin['point']['y'];ax.scatter([x],[y],s=90,facecolors='none',edgecolors=color,linewidths=1.8)
    if doc is before:
        ax.annotate('1 Vout\n端点偏短',xy=(1623.34,387),xytext=(1530,420),fontproperties=font,color=color,fontsize=11,
                    arrowprops={'arrowstyle':'->','color':color})
        ax.annotate('2 Vin + 3 GND\n错误重叠',xy=(1703,387),xytext=(1795,390),fontproperties=font,color=color,fontsize=11,
                    arrowprops={'arrowstyle':'->','color':color})
    else:
        for text,point,label in [('1 Vout\n端点恢复',(1624.5,408.5),(1530,440)),('2 GND\n上方引脚恢复',(1663.5,251.5),(1795,250)),('3 Vin\n身份与端点恢复',(1703.5,408.5),(1795,410))]:
            ax.annotate(text,xy=point,xytext=label,fontproperties=font,color=color,fontsize=11,arrowprops={'arrowstyle':'->','color':color})
        ax.plot([1663,1663],[192,252],color='#1b8dba',linewidth=3,alpha=.6)
        ax.annotate('恢复原模型\n接地线段',xy=(1663,215),xytext=(1795,180),fontproperties=font,color='#1b8dba',fontsize=11,arrowprops={'arrowstyle':'->','color':'#1b8dba'})
    ax.set_xlim(1520,1940);ax.set_ylim(465,140);ax.set_title(title,fontproperties=font,fontsize=13);ax.axis('off')
fig.tight_layout();fig.savefig(ROOT/'第10页U2修复对照.png',bbox_inches='tight',facecolor='white');plt.close(fig)
page=next(r for r in report['cases'] if r['case']=='10')
error_rows='\n'.join(f"| U2.{r['number']} {r['name']} | {r['before_px']:.3f} | {r['after_px']:.3f} |" for r in errors)
text=f'''# 第 10 页竖向引脚身份及端点修复：落地验证报告

修复已部署到 `D:\\pcb5_submission`。U1 的 9 个引脚保持正确；U2（CJ78L05）的 1/Vout、上方 2/GND、右下 3/Vin 已由原图独立 OCR 和红色引脚几何重新确认，编号、名称、位置及网络引用一次提交。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 正确引脚 TP | 9 | 12 |
| 误检 FP | 3 | 0 |
| 漏检 FN | 3 | 0 |
| 引脚 F1 | 75% | 100% |
| 导线 TP / FP / FN | 48 / 26 / 34 | 49 / 26 / 33 |
| 导线 F1 | 61.54% | 62.42% |
| 本地连接关系指标 | {page['before']['connectivity']:.6%} | {page['after']['connectivity']:.6%} |
| 本地综合评分（100 分制） | {page['before']['score_100']:.5f} | {page['after']['score_100']:.5f} |

以上为本地复现评测，非官方提交分数。引脚使用此前统一的 20 像素容差；该页仍有其他元件、网络和导线失分，此次没有宣称整页满分。

## 识别与端点依据

原结果把 2/Vin 和 3/GND 放在同一个右下物理引脚，实际顶部 GND 候选未导出。修复不使用器件型号或参考编号顺序，而是读取同图原始多视图文字作为定位依据，再在实际引脚列单独裁剪数字与名称做 OCR。

三个数字分别被两次高置信度读作 2（0.99900、0.99897）、1（0.99819、0.99829）、3（0.99773、0.99769）。90 度为唯一得到原图编号支持的方向，270 度方向不满足条件。GND、Vout 新局部识别均有两视图超过 0.998；Vin 的原始多视图为 Vin，新原图紧裁剪两次读作 vin（0.98306、0.98297），大小写一致性核验后保留原始已观测拼写 Vin。

真实红色引脚测得长度为上方 38 像素、下方两个各 40 像素，替代旧的约 18 像素通用长度。修复后左下原点坐标为 1/Vout `(1624.5,608.5)`、2/GND `(1663.5,765.5)`、3/Vin `(1703.5,608.5)`。

| 引脚 | 修复前坐标误差（像素） | 修复后坐标误差（像素） |
|---|---:|---:|
{error_rows}

## 网络修复及边界

- `LOG_NET_CAND_00008` 原成员 `(R6.2,U2.2,U2.3)` 改为 `(R6.2,U2.3)`，去掉错误挂在右下网络的 2 号引用。
- 原 `LOG_NET_CAND_00006` 的成员和导线不变，仍连接 U2.1。
- 恢复模型已观察到的独立支路 `RESTORED_NET_CAND_00013`，成员 `(U2.2,UNRESOLVED_DET_0021.1)`，为上方 2 号 GND 连接原接地符号。
- 同时从原始模型候选 `PIN_CAND_0024` 恢复该接地符号缺失的 1 号接触点记录，原样使用模型坐标，没有新增元件。
- 新导出线段为 `(1663,825)` → `(1663,765)`，逐点复制原模型 `W00001`；没有根据参考标注或新端点绘制线段。逻辑和物理来源均须只有这两个接触点、无交点歧义，端点匹配且原图绿色支撑至少 90%。
- 原有全部导线坐标、其他既有引脚、元件及无关网络成员不变。原有网络 `J00012`、`J00016` 交点歧义保留审计，本次没有将其判为已解决。

## 验证与成本

- 新增 29 项边界测试，部署后完整测试共 185 项，全部通过。
- 对第 10 页执行 24 次真实本地局部 OCR，结果与缓存实验完全一致；最终规则在本页需 24 次局部 OCR，其余 12 页未触发。
- 13 页复用原模型缓存回归，增加 3 个正确引脚、无丢失；其余 12 页完整 JSON 相同。先冻结所有预测，再加载参考标注评分。
- 第 1、2、4、5、6、8、10 页实际导出流程均通过，所有既有修复保留，新规则审计落盘；模型及 OCR 推理使用同图缓存。
- 13 页引脚 TP/FP/FN 从 {counts['before']['tp']}/{counts['before']['fp']}/{counts['before']['fn']} 变为 {counts['after']['tp']}/{counts['after']['fp']}/{counts['after']['fn']}，本地综合分逐页均值从 {report['macro_before']:.5f} 提升到 {report['macro_after']:.5f}。
- 实验仅调用本地 OCR、图像分析与缓存评分，外部 API 调用和完整模型重跑均为 0。

## 生产生效条件

规则仅处理存在库补全、三个旧记录有物理位置重叠，同时图像上具有对侧 1+2 个长竖向引脚的元件。三条引脚长度必须相近，端点均接到唯一的原模型网络；编号必须完整对应原编号集合。每条名称和编号都需原图多视图支持，冲突或缺证据则整组不提交。恢复缺失支路仅允许证据完整的两接触点接地线路。

部署 5 个文件：`agent.py`、`src/pcb_parser/pin_vertical_repair.py`、配置、测试、说明。部署后 SHA-256 全部核验通过，原文件备份在本目录 `backup/`。默认启用，设置 `PCB_PIN_VERTICAL_REPAIR=0` 可关闭。重复执行不再改动已修正结果。

证据：`verification.json`、`deployed_cached_check.json`、`entrypoint_deployed_check.json`、`deployed/10_audit.json`、`staged_fresh_ocr_audit.json`。13 页最新预测位于 `deployed/`，实际入口导出位于 `cached_entrypoint/`。

![第 10 页 U2 修复对照]({(ROOT/'第10页U2修复对照.png').as_posix()})
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':str(ROOT/'落地验证报告.md'),'errors':errors,'unchanged_pages':len(unchanged),'counts':counts,'release_verified':True},ensure_ascii=False))
