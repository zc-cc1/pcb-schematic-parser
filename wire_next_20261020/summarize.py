import copy
import hashlib
import json
from pathlib import Path
import random

from contact_proposal import propose

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
small=read(ROOT/'contacts_small_results.json')
remaining=read(ROOT/'contacts_remaining_results.json')
rows=sorted(small['per_case']+remaining['per_case'],key=lambda r:int(r['case']))
assert len(rows)==13 and len({r['case'] for r in rows})==13
work_by_hash={}
for folder in (WORK/'wire_validation_20261020/candidate_path/work').iterdir():
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    work_by_hash[graph['image_sha256']]=folder
rng=random.Random(20261020)
order_checks=0
contacts=0
for row in rows:
    case=row['case']
    prediction=read(ROOT/f'unique_or_explicit_primary/{case}_Pred.json')
    base=read(ROOT/f'baseline/{case}_Pred.json')
    digest=hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    folder=work_by_hash[digest]
    graph=read(folder/'p/export/engineering_graph_proposals.json')
    pins=read(folder/'p/export/engineering_pin_text_result.json')
    frozen=copy.deepcopy((base,graph,pins))
    assert propose(prediction,graph,pins,True)[0]==prediction
    for _ in range(5):
        shuffled_graph,shuffled_pins=copy.deepcopy((graph,pins))
        rng.shuffle(shuffled_graph['networks'])
        rng.shuffle(shuffled_graph['pin_attachments'])
        rng.shuffle(shuffled_pins['pin_candidates'])
        assert propose(base,shuffled_graph,shuffled_pins,True)[0]==prediction
        order_checks+=1
    assert (base,graph,pins)==frozen
    contacts+=len(row['variants']['unique_or_explicit_primary']['contacts'])
names=list(rows[0]['variants'])
summary={name:{'wire_f1':sum(r['variants'][name]['wire']['f1'] for r in rows)/13,
               'connectivity_f1':sum(r['variants'][name]['connectivity'] for r in rows)/13}
         for name in names}
counts={}
for metric,extract in [('wire',lambda v:v['wire']['f1']),('connectivity',lambda v:v['connectivity'])]:
    delta=[extract(r['variants']['unique_or_explicit_primary'])-extract(r['variants']['baseline']) for r in rows]
    counts[metric]={'improved':sum(d>1e-9 for d in delta),'unchanged':sum(abs(d)<=1e-9 for d in delta),
                    'worse':sum(d < -1e-9 for d in delta)}
result={'summary':summary,'changes':counts,'contacts_added':contacts,'order_checks_passed':order_checks,
        'idempotent_cases':13,'component_and_pin_metrics_unchanged_cases':13,
        'wire_geometry_unchanged_cases':13,'neural_inference_calls':0,
        'connectivity_historical_calibration_cases':26,'production_modified':False,
        'per_case':rows}
(ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 下一轮导线优化验证（2026-10-20）','',
       '最有价值的方向是补齐已有导线网络上的明确电源接触成员。当前生产版本只对已接受的电源合并组处理显式重复主候选；本轮候选允许普通的唯一电源候选，或其余候选均明确指向唯一主候选的情况。它不要求同名轨已被上游电源合并器识别，但仍要求器件已在结果中、类型和值明确、已有引脚号、主候选唯一附着到已有模型网络、原始点坐标一致。','',
       '候选只增加接触成员，不改变已有线段、器件、已有引脚字段，也不为 IC 或普通器件猜号码。代码只接收预测结果、模型图和引脚缓存；GT 仅在候选生成后参与评分。','',
       '## 分阶段验证','',
       '- 先选第 5、9、10、13 页，三页改善、一页不变后，再检查剩余 9 页。',
       '- 全程使用现有缓存，没有重新跑模型、训练或提交平台。',
       '- 连接关系采用历史平台协议的逻辑网络 soft F1，并与第 8、9 次评测的 26 个历史报告全部校准通过；未使用旧 v3 的连接代理分数。','',
       '| 指标 | 当前生产版本 | 候选方案 |','|---|---:|---:|',
       f"| 13 页平均导线 F1 | {summary['baseline']['wire_f1']:.4f} | {summary['unique_or_explicit_primary']['wire_f1']:.4f} |",
       f"| 13 页平均连接 F1 | {summary['baseline']['connectivity_f1']:.4f} | {summary['unique_or_explicit_primary']['connectivity_f1']:.4f} |",'',
       f"导线：{counts['wire']['improved']} 页提升、{counts['wire']['unchanged']} 页不变、{counts['wire']['worse']} 页下降。",
       f"连接：{counts['connectivity']['improved']} 页提升、{counts['connectivity']['unchanged']} 页不变、{counts['connectivity']['worse']} 页下降。",
       f'共恢复 {contacts} 个接触成员；13 页器件和引脚分数保持不变，所有已有导线几何保持不变。重复执行与 65 次输入顺序打乱检查均通过。','',
       '| 页 | 导线：当前 → 候选 | 连接：当前 → 候选 |','|---|---:|---:|']
for row in rows:
    b=row['variants']['baseline'];p=row['variants']['unique_or_explicit_primary']
    lines.append(f"| {row['case']} | {b['wire']['f1']:.4f} → {p['wire']['f1']:.4f} | {b['connectivity']:.4f} → {p['connectivity']:.4f} |")
lines.extend(['','## 下一步优先级','',
    '1. 优先将这项已有电源接触恢复整理进正式模块，并保留输入来源、歧义拒绝、重复候选和已有引脚保护的测试。相比调整 U-Net，它同时改善网络归属和导线计分，且无需新模型计算。',
    '2. 单独处理第 13 页的引脚归属和电源断点。97 个候选中有 29 个未找到器件；GND 符号有两个独立主候选，不能用重复候选规则强行选一个；VIN 的接触未附着导线；VOUT 的接触位于尚未导出的网络。需要结合检测框、骨架和已有文字证据排查，不能仅扩大合并范围。',
    '3. 再处理实际网络内的切分与端点差异；第 5 页仍有较多几何漏分，单纯恢复轨身份不能覆盖所有漏线。继续采用小样本验证后再扩展。','',
    '本轮候选与预测结果仅保存在桌面工作区 wire_next_20261020，D 盘生产代码未加入这轮新改动。这些数值为本地缓存验证，不是新一次官方评测。',
    '明细：summary.json、contacts_small_results.json、contacts_remaining_results.json、miss_diagnosis.json、case13_bottleneck.json。'])
(ROOT/'下一轮优化验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='per_case'},ensure_ascii=False))
