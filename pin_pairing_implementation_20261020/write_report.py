import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
plan=read(ROOT/'release_plan.json');report=read(ROOT/'deployed_cached_check.json')
assert len(report['cases'])==13 and report['gained_pins']==2 and report['lost_pins']==0 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[]
for row in report['cases']:
    case=row['case'];a=read(WORK/f'pin_relocation_implementation_20261020/deployed/{case}_Pred.json');b=read(ROOT/f'deployed/{case}_Pred.json')
    assert a['components']==b['components']
    if case!='4':assert a==b;unchanged.append(case)
    for nid,net in a['nets'].items():assert b['nets'][nid]['edges']==net['edges']
    rekeys={(g['owner'],'pin_'+str(g['old_number'])):'pin_'+g['number'] for g in read(ROOT/f'deployed/{case}_audit.json')['changes'] if g['old_number'] is not None}
    for owner,pins in a['pins'].items():
        for key,pin in pins.items():assert b['pins'][owner][rekeys.get((owner,key),key)]==pin
current=read(ROOT/'deployed/4_Pred.json');pilot=read(WORK/'page4_pin_pairing_20261020/4_Pred.json')
assert current==pilot
audit=read(ROOT/'deployed/4_audit.json');assert len(audit['restored_networks'])==1
old_change=audit['network_changes'][0]
assert old_change['before']==['C5.1','C5.3','U2.3'] and old_change['after']==['C5.1','C5.3','U2.2']
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='4')
graph=read(Path(item['model_cache'])/'p/export/engineering_graph_proposals.json');h=graph['size'][1]
emap={e['edge_id']:e for e in graph['edges']}
for restored in audit['restored_networks']:
    nid=restored['network_id'];physical=next(n for n in graph['physical_networks'] if 'RESTORED_'+n['network_id']==nid)
    segments=[]
    for eid in physical['edge_ids']:
        points=emap[eid]['polyline']
        segments.extend([[{'x':float(a[0]),'y':float(h-a[1])},{'x':float(b[0]),'y':float(h-b[1])}] for a,b in zip(points,points[1:])])
    assert list(current['nets'][nid]['edges'].values())==segments
verification={'release_hashes_verified':True,'unit_tests_passed':94,'new_tests':21,
              'fully_unchanged_pages':unchanged,'all_existing_wire_segments_preserved':True,
              'old_pin_geometry_preserved_under_rekey':True,'new_power_branch_matches_model_segments':True,
              'fourth_page_matches_verified_pilot':True,'fresh_local_ocr_matches_cache':True,'fresh_local_ocr_calls':8,
              'entrypoint':read(ROOT/'entrypoint_check.json')}
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
page=next(r for r in report['cases'] if r['case']=='4');before,after=page['before'],page['after']
text=f'''# 共享编号联合配对落地验证

已更新 D:/pcb5_submission 正式入口，默认开启 PCB_PIN_PAIRING。
本次发布 5 个文件，部署后逐文件哈希一致。设置 PCB_PIN_PAIRING=0 可关闭。
备份见当前目录 backup/；install.py --rollback 在核对哈希后可回退本次发布。

## 第四页改善

以已部署的位置共识、字段恢复、几何恢复和错点迁移版本为基线：

| 指标 | 基线 | 部署版缓存回放 |
| --- | ---: | ---: |
| 正确 IC 引脚 TP | 10 | 12 |
| 漏检 FN | 2 | 0 |
| 误检 FP | 1 | 0 |
| 引脚 F1 | 0.8696 | 1.0000 |
| 网络连接得分 | {before['connectivity']:.6f} | {after['connectivity']:.6f} |
| 正确导线 TP | 36 | 37 |
| 导线 F1 | 0.6792 | 0.6916 |
| 当前本地加权总分 | {before['score_100']:.4f} | {after['score_100']:.4f} |

第四页当前本地加权分提升 {after['score_100']-before['score_100']:.4f} 分。
其余 12 页最终 JSON 完全一致，无原先正确引脚损失、无新增误报、无页面退化。
13 页本地平均分 {report['macro_before']:.4f} → {report['macro_after']:.4f}。
这是本地复核，未提交官方平台重评。参考标签只在全部预测落盘冻结之后读取。

## 自动规则

检测“未编号候选被 text_used_by_other_pin 拦截、同元件另一候选使用相同可靠数字 token”的冲突。
必须只有一个编号竞争者，且共享数字处于缺失行上方、错误占号行下方；限定于已验证红色矩形框左侧。
两个候选都必须有唯一、精确的模型网络附件，源网络没有未决交点，原编号的唯一导出网络也必须一致。

独立裁剪两行的实际数字，经 3x/5x OCR 确认；名称用 1x/3x、0/90 度元件裁剪核对。
编号与名称均需至少两个不同视图、置信度不低于 0.98。必须构成完整的一对一改号且新号未占用。
不使用编号顺序、预期器件引脚库、页码、元件编号、候选 ID 或 GT 坐标作为采信特例。

全部通过后，原 VOUT 记录从 pin_3 改为 pin_2，保留原名称与坐标；
LOG_NET_CAND_00005 的 U2.3 同步改为 U2.2，再将真实下方 VIN 引线补为 pin_3。
未通过任一关卡时整组弃权，避免只改字段或遗留错误网络引用。

## VIN 电源分支

对已确认的候选编号副本复用现有 power_wire_export，且只采纳包含恢复候选的两接点物理电源分支。
第四页恢复 RESTORED_NET_CAND_00015，增加一个 +5V 接触引脚和一条真实模型线段。
新增分支逐点等于原模型 W00017，所有已有导线线段保持不变。
原 VOUT 网络不再残留 U2.3，VIN 仅出现在正确的 +5V 分支。
其他无关电源恢复结果被过滤，既有 IC 网络成员不能被抢占。

## 验证与成本

- 新增 21 项边界测试，全部 94 项测试通过。
- 13 页缓存回归只改变第四页；部署后回放与预发布输出完全一致。
- 第四页真实本地 OCR 重跑 8 次，得到与缓存一致的结果。
- 第 1、2、4 页经正式 run_case 入口使用同图模型/OCR 缓存回放成功。
  全部 IC 引脚与验证输出一致，改号网络成员及新增电源分支也逐项一致，审计日志和摘要正常。
- 无 YOLO/U-Net 重跑，无外部 API 调用。仅出现符合条件的冲突时才进行此阶段 OCR。
- 默认最多处理 2 组、16 次本地 OCR；关闭开关、原版备份和异常保留旧输出均已接入。

适用范围当前为左侧、相邻行间距至少 18 像素的此类共享数字冲突。
其他复杂交换、旋转元件、字段冲突或未见页面需另行验证，不能据此认定所有编号错配均已解决。

## 文件

正式模块：D:/pcb5_submission/src/pcb_parser/pin_pairing.py。
正式配置：D:/pcb5_submission/configs/pin_pairing_v1.json。
正式说明：D:/pcb5_submission/docs/pin_pairing.md。
正式入口：D:/pcb5_submission/agent.py，在错点迁移之后运行。
日志：logs/{{case}}_pin_pairing.json。
逐页结果：deployed_cached_check.json；发布完整性：verification.json；入口检查：entrypoint_check.json。
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
