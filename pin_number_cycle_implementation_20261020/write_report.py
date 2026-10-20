"""Record verified deployed outputs, exact release hashes and repair limits."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=read(ROOT/'deployed_cached_check.json')
plan=read(ROOT/'release_plan.json')
assert len(report['cases'])==13 and report['gained_pins']==2 and report['lost_pins']==0
assert not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
members=lambda net:sorted(v.strip() for v in net['hyperGraph'].strip('()').split(',') if v.strip())
unchanged=[]
for row in report['cases']:
    case=row['case']
    before=read(WORK/f'pin_evidence_implementation_20261020/deployed/{case}_Pred.json')
    after=read(ROOT/f'deployed/{case}_Pred.json')
    assert before['components']==after['components']
    assert before['pins'].keys()==after['pins'].keys()
    assert before['nets'].keys()==after['nets'].keys()
    for nid,net in before['nets'].items():
        assert net['edges']==after['nets'][nid]['edges']
        assert len(members(net))==len(members(after['nets'][nid]))
    if case!='6':
        assert before==after
        unchanged.append(case)
        continue
    expected=read(ROOT/'6_pilot_Pred.json')
    assert expected==after
    a=before['pins']['U1'];b=after['pins']['U1']
    assert a.keys()==b.keys()
    assert a['pin_1']==b['pin_2'] and a['pin_2']==b['pin_1'] and a['pin_3']==b['pin_3']
    assert all(value==after['pins'][owner] for owner,value in before['pins'].items() if owner!='U1')
    audit=read(ROOT/'deployed/6_audit.json')
    changes={v['network_id']:v for v in audit['network_changes']}
    assert set(changes)=={'LOG_NET_CAND_00013','RESTORED_NET_CAND_00009'}
    for nid,net in before['nets'].items():
        if nid not in changes:assert net==after['nets'][nid]
        else:
            assert members(net)==changes[nid]['before']
            assert members(after['nets'][nid])==changes[nid]['after']
            assert {k:v for k,v in net.items() if k!='hyperGraph'}=={k:v for k,v in after['nets'][nid].items() if k!='hyperGraph'}
    assert members(after['nets']['LOG_NET_CAND_00013'])==['U1.2','UNRESOLVED_DET_0007.1']
    assert members(after['nets']['RESTORED_NET_CAND_00009'])==['U1.1','UNRESOLVED_DET_0021.1']
fresh=read(ROOT/'staged_fresh_ocr_audit.json')
assert fresh['changes']==audit['changes'] and fresh['network_changes']==audit['network_changes']
assert fresh['groups']==audit['groups'] and fresh['ocr_calls']==24
assert next(g for g in audit['groups'] if g['owner']=='U3')['status']=='verified_unchanged'
assert next(g for g in audit['groups'] if g['owner']=='U1')['preserved_unresolved_junctions']=={'LOG_NET_CAND_00013':['J00023']}
counts={stage:{k:sum(r[stage]['pin'][k] for r in report['cases']) for k in ['tp','fp','fn']} for stage in ['before','after']}
verification={'release_hashes_verified':True,'unit_tests_passed':133,'new_tests':21,
              'fully_unchanged_pages':unchanged,'changed_pin_records':2,'changed_network_references':2,
              'all_components_wire_edges_and_pin_physical_records_preserved':True,
              'all_network_ids_and_member_counts_preserved':True,
              'fresh_local_ocr_matches_cache':True,'fresh_local_ocr_calls':24,
              'external_api_calls':0,'full_model_runs':0,'total_pin_metrics':counts,
              'entrypoint':read(ROOT/'entrypoint_check.json')}
write(ROOT/'verification.json',verification)
page=next(r for r in report['cases'] if r['case']=='6')
text=f'''# 第 6 页引脚编号交换：落地验证报告

修复已部署到 `D:\\pcb5_submission`。原图中 U1 的 IN 为 1 号、GND 为 2 号；原结果的名称和物理位置正确，但两个编号颠倒。当前同时更正编号键和两条既有网络引用。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 正确引脚 TP | 14 | 16 |
| 误检 FP | 2 | 0 |
| 漏检 FN | 2 | 0 |
| 引脚 F1 | 87.5% | 100% |
| 本地综合评分（100 分制） | {page['before']['score_100']:.5f} | {page['after']['score_100']:.5f} |
| 连接关系指标 | {page['before']['connectivity']:.8f} | {page['after']['connectivity']:.8f} |

这是本地评测结果，并非官方提交分数。导线指标不变，导线 TP/FP/FN 仍为 59/10/27。

## 识别依据与实施范围

原图数字单独裁剪、屏蔽彩色线条后，放大 3 倍和 5 倍分别做本地 OCR。IN 对应的数字两次读作 1，置信度 0.98984、0.98935；GND 对应的数字两次读作 2，置信度 0.9988、0.9987。OUT 的直接观测编号 3 用于确认旋转方向，保持不变。元件内的名称也须有至少两个视图以不低于 0.98 的置信度支持。U3 作为正确对照，三个引脚均保持原样。

规则不依赖页号、元件型号或参考答案：只处理具有真实边框与短引脚证据、含库补全编号、具有直接观测方向锚点的三引脚底部行。所有数字必须构成原有编号集合的完整置换；证据缺失、冲突、网络归属不唯一或预算耗尽时整组不提交。重复执行不会再次交换。

## 网络与坐标检查

- `LOG_NET_CAND_00013`：`(U1.1,UNRESOLVED_DET_0007.1)` → `(U1.2,UNRESOLVED_DET_0007.1)`。
- `RESTORED_NET_CAND_00009`：`(U1.2,UNRESOLVED_DET_0021.1)` → `(U1.1,UNRESOLVED_DET_0021.1)`。
- 仅替换原有成员编号引用；所有元件、引脚物理记录、导线边坐标、网络 ID 和每条网络的成员数量均保持不变。
- 原来源网络的交点歧义 `J00023` 保留在审计中。本次沿用既有物理连接，只校正身份引用，未解决或新增交点连接。

## 验证与成本

- 新增 21 项边界测试；部署后的完整测试共 133 项，全部通过。
- 对 13 页复用原模型缓存做回归；第 6 页增加 2 个正确引脚、无丢失，其余 12 页完整 JSON 相同。
- 第 1、2、4、5、6 页实际 `agent.run_case` 导出流程通过；仅模型及 OCR 推理替换为同图留存结果，真实后处理与日志输出均运行。
- 对第 6 页额外执行 24 次真实本地 OCR，结果与缓存验证完全一致；其中 U1、U3 各 12 次。初始采集 32 次本地 OCR，外部 API 调用及完整模型重跑均为 0。
- 全 13 页引脚统计：TP/FP/FN 从 {counts['before']['tp']}/{counts['before']['fp']}/{counts['before']['fn']} 变为 {counts['after']['tp']}/{counts['after']['fp']}/{counts['after']['fn']}；本地综合评分逐页均值从 {report['macro_before']:.5f} 变为 {report['macro_after']:.5f}。
- 生成时先冻结全部预测，再读取参考标注评分，参考标注不进入修复逻辑。

## 部署与复查

新增 `src/pcb_parser/pin_number_cycle.py`、配置、测试及说明，并在 `agent.py` 导出前接入。5 个部署文件已逐项验证 SHA-256，原 `agent.py` 备份位于本目录 `backup/agent.py`。默认启用，环境变量 `PCB_PIN_NUMBER_CYCLE=0` 可关闭此规则。

证据文件：`deployed_cached_check.json`、`verification.json`、`entrypoint_check.json`、`deployed/6_audit.json`、`staged_fresh_ocr_audit.json`。13 页最终预测保存在 `deployed/`，正式入口输出保存在 `cached_entrypoint/`。
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':str(ROOT/'落地验证报告.md'),'counts':counts,'unchanged_pages':len(unchanged),'release_hashes_verified':True},ensure_ascii=False))
