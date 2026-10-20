import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
DEPLOY=Path('D:/pcb5_submission')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=read(ROOT/'results.json')
inventory=read(ROOT/'inventory.json')
collection=read(ROOT/'ocr_collection.json')
baseline_snapshot=read(WORK/'wire_contact_implementation_20261020/deployment_before.json')
used=['agent.py','adapter.py','src/pcb_parser/pin_text_ocr.py','src/pcb_parser/targeted_pin_ocr.py',
      'src/pcb_parser/pin_text.py','src/pcb_parser/pin_text_v2.py','src/pcb_parser/component_text.py',
      'src/pcb_parser/submission_fields.py','src/pcb_parser/pin_output_rules.py',
      'configs/targeted_pin_ocr_v1.json','configs/pin_text_v2.json','configs/pin_output_rules_v1.json']
unchanged={name:sha(DEPLOY/name)==baseline_snapshot[name] for name in used}
assert all(unchanged.values())
for model in collection['model_fingerprints'].values():
    assert sha(DEPLOY/'assets/ocr'/model['file'])==model['sha256']
outside=[name for name in ['configs/wire_graph_v1.json','scripts/run_wire_graph.py','src/pcb_parser/wire_graph.py']
         if sha(DEPLOY/name)!=baseline_snapshot[name]]
(ROOT/'input_integrity.json').write_text(json.dumps({'experiment_functions_unchanged_from_frozen_baseline':unchanged,
    'local_ocr_model_hashes_valid':True,'frozen_prediction_baseline':True,
    'other_production_changes_outside_experiment':outside,'production_writes_by_this_experiment':[]},
    ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=result['summary'];b=s['baseline'];best=s['crop_header_schema'];loose=s['crop_single99_header']
delta=best['score_100']-b['score_100']
tp=sum(r['variants']['baseline']['metrics']['pin']['tp'] for r in result['per_case'])
fn=sum(r['variants']['baseline']['metrics']['pin']['fn'] for r in result['per_case'])
fp=sum(r['variants']['baseline']['metrics']['pin']['fp'] for r in result['per_case'])
component_count=sum(len(r['components']) for r in inventory)
lines=['# 元件裁剪后二次 OCR 实验（2026-10-20）','',
       f'结论：在此次冻结版本的 13 页公开样本上，单纯增加整元件裁剪 OCR、保持原绑定与导出门槛，实际提分为 0。增加基于已有引脚的数字排针适配后，成功补回 1 个正确引脚，平均本地总分约提高 {delta:.2f} 分。收益存在，但很小。','',
       '## 实验设计','',
       '- 先测第 1、10、11、13 页，再扩展其余 9 页。',
       f'- 从预测记录中选择仍缺编号或名称的 IC 类元件，共 {component_count} 个。使用预测框外扩 32 px 裁剪原始 PNG，未使用 GT 框或 GT 引脚位置裁剪。',
       f"- 每个元件使用 1×、3×，以及 0°、90° 两种方向，共 {len(collection['views'])} 次本地 RapidOCR 调用。使用项目内三个 ONNX 权重，无外部 API；未重跑 YOLO、U-Net 或训练。",
       '- 对照组只将已有 OCR 结果送入同一绑定、采信和导出逻辑，隔离重新绑定本身的收益。',
       '- 严格方案使用现有采信门槛：置信度至少 0.98、至少两个视图支持、全局绑定余量至少 5、拒绝冲突和重复号码。不同倍率、旋转是同一模型的重复观测，不作为统计独立证据。',
       '- 裁剪、OCR、绑定及导出全部完成后，GT 才进入独立评分。器件、已导出的引脚与已有导线几何保持原值。',
       '- 新增引脚使用已有模型候选点，沿用当前 pin_lead_snap 处理；不凭 OCR 文字框凭空生成引脚尖端。','',
       '## 13 页结果','',
       '| 方法 | 平均 Pin F1 | 平均本地总分 /100 | 新导出引脚 |',
       '|---|---:|---:|---:|']
for name,label in [('baseline','冻结基线'),('old_ocr_rebind','仅旧 OCR 重新绑定'),('crop_1x','元件裁剪 1×'),
                   ('crop_3x','元件裁剪 3×'),('crop_combined','1×+3× 严格融合'),
                   ('old_ocr_header_schema','旧 OCR +数字排针适配'),('crop_header_schema','裁剪融合 +数字排针适配'),
                   ('crop_single99_header','单视图 0.99 +数字排针适配')]:
    v=s[name]
    lines.append(f"| {label} | {v['pin_f1']:.4f} | {v['score_100']:.4f} | {v['pins_exported']} |")
lines.extend(['',f"推荐保守方案的 Pin F1：{b['pin_f1']:.4f} → {best['pin_f1']:.4f}；总分：{b['score_100']:.4f} → {best['score_100']:.4f}，增加 {delta:.4f} 分。",
    f'GT 引脚计数：基线 TP={tp}、FP={fp}、FN={fn}；保守候选增加 1 个 TP、减少 1 个 FN，未新增 FP。连接和导线分数保持不变。','',
    '这些分数采用此前已核对历史平台报告的本地匹配规则，是冻结公开样本上的实验估算，不是新一次官方评测。','',
    '## 实际补回的引脚','',
    '- 第 13 页 H1 的 pin_3。OCR 在两个倍率读到数字 3；原结果已有 pin_1/name=1、pin_2/name=2，因此可将新观察到的 3 作为该元件的数字型名称，补全一个可导出的引脚。未依据 GT 推断编号或名称。',
    '- 这一适配属于绑定/导出逻辑的额外规则。只补 OCR 编号但保持旧名称门槛，不能产生这次提升。',
    '- 该页 Pin F1 从 0.9565 提升到 1.0000，本地总分约增加 1.0875 分；摊到 13 页宏平均约为 0.0837 分。其余 12 页保守方案没有分数变化。','',
    f"![第13页 H1 的原图裁剪]({(ROOT/'crops/13_H1.png').as_posix()})",'',
    '## 为什么整体收益有限','',
    '- 现有管线已经包含整图/分块/旋转 OCR、单引脚局部 OCR 和编号条带重读，简单放大通常重复读取已有文字。',
    '- 读出文本不等于形成有效引脚：重复号码会被拒绝，部分候选没有完整名称，部分本身是重复或错误的引脚候选。',
    '- 后验错误归因显示基线仍有 42 个 GT 引脚漏分：28 个缺少导出的编号键，7 个已有编号但名称错误，7 个编号名称正确但位置误差超过 20 px。这个归因仅用于解释结果，没有输入裁剪或生成候选。',
    '- 此次只补缺失字段，不改写已有错误字段，也不能靠 OCR 找回模型完全没检测到的尖端。','',
    '放宽成单视图采信会引入误报：第 1 页 SW2 多导出两个错误引脚，该页本地总分下降约 0.56 分。虽然整体平均仍略升，但低于保守方案，不建议采用。','',
    '## 建议与代码状态','',
    '优先考虑仅对“已有至少两个数字名称与编号一致、仍缺脚”的排针做条件式补读。现有证据不支持对所有元件默认增加这轮 OCR。若要获得更大提升，下一步应把 OCR 文字与局部高分辨率引脚检测/几何定位联合验证。',
    '本轮脚本、裁剪、OCR 读数与候选结果均只保存在桌面实验目录，未改动 D 盘正式代码。',
    '基线使用此前落地 v2 的冻结预测。核对时发现 D 盘另有 wire_graph.py、run_wire_graph.py 和 wire_graph_v1.json 的后续改动，未纳入这次冻结基线，也未回退或覆盖。实验实际调用的 OCR、绑定、引脚后处理代码及 OCR 权重哈希与基线一致。','',
    '复现脚本：inventory.py、collect_ocr.py、evaluate.py。结果明细：results.json；首批结果：results_small.json；输入检查：input_integrity.json；错误归因：miss_categories.json。'])
(ROOT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'components':component_count,'local_ocr_calls':len(collection['views']),
    'plain_crop_score_gain':s['crop_combined']['score_100']-b['score_100'],
    'crop_with_header_schema_score_gain':delta,'baseline_pin_tp_fp_fn':[tp,fp,fn],
    'production_modified_by_experiment':False,'baseline':'frozen_v2_predictions'},ensure_ascii=False))
