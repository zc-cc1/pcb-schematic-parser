from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
audit=read(ROOT/'audit.json')
secondary=read(ROOT/'secondary_checks.json')
geometry=[r for r in audit['misses'] if r['category']=='candidate_offset_or_crowded_assignment']
assert len(geometry)==7 and all(not r['nearby_secondary_candidates'] for r in secondary)
review=[]
for row in geometry:
    review.append({'case':row['case'],'component':row['component'],'key':row['key'],'name':row['name'],
                   'reviewed_by':'assistant_visual_inspection','figure':'geometry_misses.png',
                   'finding':'No usable primary candidate on the expected lead; nearby candidates belong to adjacent leads.',
                   'unconnected_or_no_connect_mark':not(row['case']=='1' and row['component']=='U2')})
(ROOT/'geometry_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
labels={'candidate_offset_or_crowded_assignment':'未形成可用引脚主候选（已核对原图）',
        'candidate_fields_incomplete':'已有候选，编号或名称不完整',
        'candidate_owner_unresolved_or_wrong':'已有候选，归属未确认或挂错器件',
        'candidate_number_mismatch':'已有候选，编号不一致',
        'exported_name_or_number_mismatch':'已有导出编号，名称/编号配对不符',
        'exported_position_error':'编号名称正确，导出位置超差'}
order=['candidate_offset_or_crowded_assignment','candidate_fields_incomplete',
       'candidate_owner_unresolved_or_wrong','candidate_number_mismatch',
       'exported_name_or_number_mismatch','exported_position_error']
lines=['# 当前引脚未识别与漏分清单（2026-10-20）','',
       '基于 D:\\pcb_open_gt_v37dots_b\\final_results 的最新完整 13 页运行，并与同图保留的模型候选及原图交叉检查。本报告不运行模型、不改生产代码。',
       '按当前历史平台协议的 IC/box 引脚范围、20 px 距离门槛：GT 共 191 个引脚，149 个匹配正确，42 个未得分，另有 21 个错误/多余输出。13 页的计分引脚与此前落地 v2 的冻结预测全部一致。',
       '42 个未得分不能统称为检测不到；28 个没有导出对应编号键，另外 14 个虽然已有编号键，但名称配对或位置有问题。','',
       '## 分层结论','', '| 原因 | 数量 |','|---|---:|']
for category in order:lines.append(f"| {labels[category]} | {audit['categories'][category]} |")
lines.extend(['',
    '真正没有形成可用主候选、需要从几何检测端处理的重点有 7 个。其余 35 个已有候选或导出记录，问题在字段、归属、配对或尖端定位。类别按主要失败环节互斥统计；个别记录同时存在名称和位置错误，详细 JSON 同时保留证据。','',
    '## 7 个几何候选漏检','',
    '| 页 | 元件 | 引脚 | 名称 | 图上情况 |','|---|---|---|---|---|'])
for row in review:
    context='未接线/带 NC 标记的引脚' if row['unconnected_or_no_connect_mark'] else '与外部导线连接的 EN 引脚'
    lines.append(f"| {row['case']} | {row['component']} | {row['key'][4:]} | {row['name']} | {context} |")
lines.extend(['',
    '7 个中有 6 个是未接线或带 no-connect 标记的引脚。它们的最近候选有时只差 20–40 px，但实际属于邻脚；不能因为附近有点就认定目标脚已经检测到。',
    '已用一对一几何分配避免同一候选解释多个引脚，并逐图检查了这 7 处：目标短引线上没有可用主候选，25 px 原始邻域内也没有被抑制的次候选。这里的“漏检”指最终保留的几何候选缺失，不等同于已经验证原始概率图完全没有激活。','',
    f"![7处候选漏检核对图]({(ROOT/'geometry_misses.png').as_posix()})",'',
    '## 有候选但字段不完整的 14 个','',
    '| 页 | 元件 | 应有编号/名称 | 候选中已有字段 |','|---|---|---|---|'])
for row in audit['misses']:
    if row['category']!='candidate_fields_incomplete':continue
    p=row['candidate']
    got=f"编号={p['number'] if p['number'] is not None else '缺'}；名称={p['pinname'] if p['pinname'] is not None else '缺'}"
    lines.append(f"| {row['case']} | {row['component']} | {row['key'][4:]} / {row['name']} | {got} |")
lines.extend(['',
    '例如第 11 页 USB2 的五个引脚均有近距离几何候选，但编号未可靠绑定，导致整组没有导出。第 13 页 H1.3 也是已有候选、字段缺失；上一轮 OCR 实验已补回它，但该实验尚未合入正式代码，因此本次正式结果仍计为漏分。','',
    '## 归属问题的 4 个','',
    '| 页 | 应属元件/引脚 | 名称 | 当前候选归属 |','|---|---|---|---|'])
for row in audit['misses']:
    if row['category']=='candidate_owner_unresolved_or_wrong':
        p=row['candidate']
        lines.append(f"| {row['case']} | {row['component']}.{row['key'][4:]} | {row['name']} | {p['owner'] or p['owner_status']} |")
lines.extend(['','## 编号不一致的 3 个','',
    '| 页 | 元件/应有编号 | 名称 | 当前候选编号 |','|---|---|---|---|'])
for row in audit['misses']:
    if row['category']=='candidate_number_mismatch':
        lines.append(f"| {row['case']} | {row['component']}.{row['key'][4:]} | {row['name']} | {row['candidate']['number']} |")
lines.extend(['','## 名称或编号配对不符的 7 个','',
    '| 页 | 元件/编号 | 应有名称 | 已导出名称 | 点距 /px |','|---|---|---|---|---:|'])
for row in audit['misses']:
    if row['category']=='exported_name_or_number_mismatch':
        lines.append(f"| {row['case']} | {row['component']}.{row['key'][4:]} | {row['name']} | {row['exported']['pinname']} | {row['export_distance_px']:.2f} |")
lines.extend(['',
    '这组既有字面名称不完整（ADJ(GND)→GND、VOUT(TAB)→VOUT），也有编号与名称交叉配对；并非都能靠再读一遍相同文字解决。','',
    '## 导出位置超差的 7 个','',
    '| 页 | 元件/编号 | 名称 | 导出点距 /px |','|---|---|---|---:|'])
for row in audit['misses']:
    if row['category']=='exported_position_error':
        lines.append(f"| {row['case']} | {row['component']}.{row['key'][4:]} | {row['name']} | {row['export_distance_px']:.2f} |")
lines.extend(['',
    '其中 5 个仅偏约 21–22 px，另外 2 个偏约 96 px 和 197 px。前者可优先核对尖端后处理，后者需要检查编号与坐标的对应关系，不能只调一个统一位移。','',
    '## 按页分布','', '| 页 | 正确 TP | 未得分 FN | 错误/多余 FP |','|---|---:|---:|---:|'])
for row in audit['per_case']:
    v=row['score'];lines.append(f"| {row['case']} | {v['tp']} | {v['fn']} | {v['fp']} |")
lines.extend(['',
    '第 1 页和第 11 页合计占 26 / 42 个未得分引脚，适合优先逐元件排查。第 8 页的 3 个 NC 脚则是清晰的未连接引脚检测样本。','',
    '## 检查边界','',
    '- 统计针对平台计分的 IC/box 类型，包含此类型下的 USB、排针等元件；未将电阻、电容、电源符号的全部端子混入 PinF1 统计。',
    '- 候选来自同图保留缓存，按原始点或现有尖端投影与参考点做一对一邻域匹配；归因是诊断证据，并非所有候选均已被电气语义验证。',
    '- GT 只用于本次后验诊断，没有输入任何预测、OCR、候选恢复或生产代码。',
    '- 本次未执行 OCR、检测模型或 U-Net 推理，未修改 D 盘代码。',
    '明细：audit.json（每脚坐标、候选、状态、最近输出）、geometry_review.json（7 处图像核对）、secondary_checks.json（次候选核对）。'])
(ROOT/'当前引脚漏检清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'report':'当前引脚漏检清单.md','geometry_misses':len(review),
                  'unconnected_geometry_misses':sum(r['unconnected_or_no_connect_mark'] for r in review),
                  'unscored_total':len(audit['misses']),'model_inference_calls':0},ensure_ascii=False))
