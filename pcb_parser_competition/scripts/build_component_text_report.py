"""Summarize measured import/detection/OCR results without inventing scores."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def percentage(value):
    return f"{value*100:.2f}%"


def main():
    if read("configs/baseline_v3_adopted.json").get("component_text", {}).get("version") == "semantic_v2":
        from build_text_optimization_report import main as build_optimization_report
        build_optimization_report()
        return
    imported=read("data/local_imports/v3_20261011/import_manifest.json")
    source=read("data/local_imports/v3_20261011/source_evidence/runs/eval_v3_fused_v2thr.json")
    detector=read("runs/detector_comparison_v3/adopted_fused_cpu.json")
    ocr=read("runs/s3_component_text_quick_dev/summary.json")
    evaluated=read("runs/s3_component_text_quick_dev/evaluation.json")
    cfg=read("configs/baseline_v3_adopted.json")
    weight=ROOT/cfg["weights"]
    weight_sha=hashlib.sha256(weight.read_bytes()).hexdigest()
    if weight_sha!=source["weights_sha256"] or source["manifest_sha256"]!=detector["manifest_sha256"]:
        raise ValueError("Imported weights or evaluation manifest changed")
    if cfg["detector"]["class_confidence"] != source["class_thresholds"]:
        raise ValueError("Final source thresholds changed")
    so, do=source["overall"],detector["overall"]
    metrics=evaluated["overall"]
    legacy=evaluated["legacy_overall"]
    text_rows=[]
    definitions={"designator":"位号真实覆盖率", "value":"数值严格匹配率", "Name":"型号严格匹配率"}
    for field in ["designator","value","Name"]:
        m,l=metrics[field],legacy[field]
        text_rows.append(f"| {definitions[field]} | {l['correct']}/{l['expected']}，{percentage(l['exact_match_rate'])} | {m['correct']}/{m['expected']}，{percentage(m['exact_match_rate'])} |")
    bind=metrics["designator"]
    text_rows.insert(0,f"| 位号绑定精确率 | {legacy['designator']['correct']}/{legacy['designator']['assigned']}，{percentage(legacy['designator']['binding_precision'])} | {bind['correct']}/{bind['assigned']}，{percentage(bind['binding_precision'])} |")
    case_rows=[]
    for row in ocr["cases"]:
        case_rows.append(f"| {row['case_id']} | {row['detections']} | {row['ocr_texts']} | {row['resolved_designators']} | {row['bound_values']} | {row['bound_models']} | {row['ocr_seconds']:.2f}s |")
    failures=[]
    for case in evaluated["cases"]:
        for row in case["details"]:
            if row["failure"]:
                failures.append(f"| {case['case_id']} | {row['reference']} | {row['field']} | {row['expected']} | {row['predicted'] or '空'} | {row['failure']} |")
    evidence_cases={c["case_id"]:c for c in source["cases"]}
    differences=[]
    for case in detector["cases"]:
        s=evidence_cases[case["case_id"]]
        if s["counts"]!=case["counts"]:
            a,b=s["counts"],case["counts"]
            differences.append(f"| {case['case_id']} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} |")
    report=f"""# v3迁移与OCR文字绑定实测报告

本轮已完成用户授权的v3代码/权重迁移、默认配置切换，以及下一阶段OCR与元件文字绑定原型。完整图片入口实测通过，OCR阶段的覆盖率验收尚未全部达标；引脚、导线和网络仍待实现。

## 迁移结果与固定检测配置

来源：`{imported['source_root']}`。选择性复制并在复制时校验{len(imported['files'])}个代码、配置、权重和来源证据文件。原有被替换文件备份于 `data/local_imports/v3_20261011/prior_workspace_files/`，迁移清单是 `data/local_imports/v3_20261011/import_manifest.json`。原工程、原始Dataset300和其他训练进程未改动，本轮没有新增训练。

采用的是 `detector_dataset300_v3_nano.pt`，由原v2 nano继续训练的v3配方；导入的其他架构配置不是当前实际选用的权重。默认为 `configs/baseline_v3_adopted.json`，兼容配置 `baseline_finetuned.json` 同步；v1/v2配置和旧运行结果保留。源 `baseline_v3.json` 中“待训练”的旧状态不作为采纳依据，逐类阈值取自最终评测记录。

权重SHA-256：`{weight_sha}`。

检测输入1536，基础置信度0.2；SAHI 1024切片、重叠0.25、IOS/GREEDYNMM阈值0.45；同一checkpoint的整图与切片预测经class-aware WBF（IoU 0.55）融合，再按最终28类阈值过滤。fused模式始终走两条路径，小图不跳过融合路径。它不是两个不同训练模型的集成。

## 59例复测

| 记录 | TP | FP | FN | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 原项目GPU保存结果 | {so['tp']} | {so['fp']} | {so['fn']} | {percentage(so['precision'])} | {percentage(so['recall'])} | {percentage(so['f1'])} |
| 本项目CPU重新推理 | {do['tp']} | {do['fp']} | {do['fn']} | {percentage(do['precision'])} | {percentage(do['recall'])} | {percentage(do['f1'])} |

当前显卡已有其他训练占满算力/显存，因此使用CPU完成真实复测。权重与验证manifest哈希相同、IoU匹配阈值0.5及最终置信度阈值相同。只有以下2例计数不同，根因尚未确认；GPU空闲后可只复查这2例。不能把CPU值写成与原GPU完全一致。

| 案例 | 原记录TP/FP/FN | CPU复测TP/FP/FN |
|---|---|---|
{chr(10).join(differences)}

结果：`runs/detector_comparison_v3/adopted_fused_cpu.json`；源记录：`data/local_imports/v3_20261011/source_evidence/runs/eval_v3_fused_v2thr.json`。本项是类别+框的micro F1，不是mAP，也不是包含位号、型号、引脚和网表的赛事总分。59例已参与阈值选择和开发，不能作为独立测试结果。

## OCR阶段实现

输入为PNG与冻结的 `detections.json`。使用RapidOCR/ONNX Runtime CPU：整图OCR + 768分块（重叠0.2）放大2倍 + 0/90度识别，恢复到原图坐标并合并重叠观测。相同文字在不同位置不会被全局去重；低置信度/截断的替代观测保留。

文字按位号、单位数值、型号候选、数值/引脚标记、网络标签分类。位号按类别前缀兼容、距离、对齐、置信度进行全局一对一分配，另外绑定value/Name；重复观测不作为独立歧义，粘连位号拆分需要其他OCR观测的支持。IC型号可在上侧、位号可在下侧；内部引脚文本和图纸标题不会仅凭最近距离被当作型号。

字段保留实际识别字符串，不自动把u2改成U2、0改成O或补单位。规范化只用于候选比较。不确定字段为空，候选、原始观察、绑定分数和拒绝原因留在旁路JSON。图片推理入口没有读取本页参考JSON，也没有用复核结果填答案。

实现：`src/pcb_parser/component_text.py`、`scripts/run_component_text.py`。入口：`scripts/run_ocr_binding.ps1`。产物：`runs/s3_component_text_quick_dev/<case>/texts.json`、`component_records.json`、`components_partial.json`、`binding_debug.json`和`overlay.png`。内部左上坐标、出口一次H-y变换；该出口约定仍待正式Golden校准。

| 案例 | 检测候选 | 去重OCR文本 | 已绑定位号 | 已绑定数值 | 已绑定型号 | OCR调用耗时 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(case_rows)}

上表“已绑定”不表示全部正确。OCR耗时只包括实际引擎调用，不含进程/模型加载、图像预处理和写文件，不能推算正式50例耗时。0002的8个位号、7个数值、1个型号共16个适用字段严格匹配全部正确，如IC2→BCM2835、C82→1u0、X2→19M2。

## 101字段离线评价

人工从原PNG及放大裁剪复核54个位号、41个主要数值、6个型号，共101字段；源框按图复核定位后复用。参考位于 `data/reviews/component_text_quick_dev.json`，裁剪审查图在 `runs/s3_component_text_quick_dev/review/`。电源、地、网络标签和引脚文字不计入这101字段。参考不是官方Golden，0001/0002是反复使用的开发例。

用type+框IoU≥0.5的一对一匹配确认所属元件，再严格比较字段字符、大小写和单位。所有应输出字段均进分母；漏检测、空字段和错字计为失败。对于Name/value，文字归属按匹配到的物理元件判断，位号另计，故本表不是整件元件全部字段正确率。位号绑定精确率=正确位号与所属元件的记录/全部已绑定记录；真实覆盖率=正确恢复的位号/应恢复的位号。

对照使用相同v3检测框：旧方案为原整图OCR+临时最近位号绑定，新方案为分块OCR+语义字段绑定。改进不能归因于更换检测器；本轮未做单独OCR/匹配算法消融。

| 本地指标 | 旧方案 | 新方案 |
|---|---:|---:|
{chr(10).join(text_rows)}

在49个检测框已正确匹配的位号上，新方案恢复{bind['correct']}个，条件覆盖率{percentage(bind['exact_rate_given_matched_detection'])}。型号在5个正确检测的适用器件中5/5正确，但总体5/6仍要计入Q1漏项。Name+value合计42/47，89.36%；同时保留两项独立指标，以免小样本平均掩盖型号总体未达85%的情况。

阶段目标与结果：位号精确率≥95%通过，位号真实覆盖率≥90%未通过；数值严格匹配≥85%通过，型号总体≥85%未通过。已完成原型与评价链路，阶段验收未全部完成。

评价文件：`runs/s3_component_text_quick_dev/evaluation.json`。失败字段如下，不从分母中排除：

| 案例 | 元件 | 字段 | 应为 | 预测 | 失败类型 |
|---|---|---|---|---|---|
{chr(10).join(failures)}

## 验证与下一轮工作

已冷运行PowerShell入口，从两张PNG完成v3检测→OCR→字段绑定，所有部分JSON结构/坐标检查通过。33个解析回归通过（含10个文字归属/坐标/歧义测试及8个导入的融合测试），另在文档Python环境通过3个CAD清点测试，共36个。GPU环境未装pdfplumber，CAD清点测试使用已配置该依赖的文档环境。

保持v3冻结，先提高位号覆盖率：检查重复inductor检测引起的L2歧义、密集R13/R14、TP端口对齐、u2大小写和L1的OCR误读。Q1类别错误以及微小TP框未满足IoU匹配属于上游/匹配限制，单改OCR不能恢复全部遗漏。随后实现R/C/L的电气tip定位，建立至少50个已核验tip，再处理IC引脚与导线骨架。pins/nets当前仍为空，当前文件是部分输出，不能作为完整比赛提交。

运行与修改：

```powershell
# 在本项目根目录：CPU默认，不占用其他GPU训练
.\\scripts\\run_ocr_binding.ps1
# 只改绑定参数/逻辑时复用冻结检测和OCR；OCR参数变化须重新识别
.\\scripts\\run_ocr_binding.ps1 -ReuseDetections -ReuseTexts
& .\\.venv-gpu\\Scripts\\python.exe scripts/evaluate_component_text.py
& .\\.venv-gpu\\Scripts\\python.exe scripts/build_component_text_report.py
```

下一轮验收资料与路线见 `docs/downstream_implementation_plan.md`。正式Golden/评测器/元件库尚未取得；这些本地数字不表示赛事成绩。
"""
    (ROOT/"docs/v3_import_and_ocr_binding_report.md").write_text(report,encoding="utf-8")
    state=read("configs/project.json")
    state["stage"]="S3_v3_frozen_component_text_prototype_validated_coverage_target_pending"
    state["detector"]["source_gpu_f1"]=so["f1"]
    state["detector"]["local_cpu_f1"]=do["f1"]
    state["ocr"].update(status="image_only_prototype_executed_101_fields_reviewed_coverage_target_pending",
                         manual_review_fields=evaluated["field_count"],local_metrics=metrics,
                         run="runs/s3_component_text_quick_dev/summary.json",
                         evaluation="runs/s3_component_text_quick_dev/evaluation.json",
                         official_score=None)
    (ROOT/"configs/project.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report":"docs/v3_import_and_ocr_binding_report.md","weight_hash_verified":True,"metrics":metrics},ensure_ascii=False))


if __name__=="__main__":
    main()
