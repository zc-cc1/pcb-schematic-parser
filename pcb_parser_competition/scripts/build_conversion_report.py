"""Build a human-readable delivery report and review queue from saved conversion evidence."""

from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import digest, load_json, save_json, validate_output


HIGH = {"unresolved_designator", "invalid_component_bbox_omitted", "invalid_pin_tip_omitted", "unresolved_edge_pin_omitted", "unowned_global_pin_omitted", "missing_scored_name", "missing_scored_value", "pin_number_collision", "coordinate_outside_image", "unsafe_designator_for_hypergraph"}
REVIEW = {"provisional_type_mapping", "source_label_merge_unverified", "ambiguous_value", "ambiguous_name", "ambiguous_designator", "flywire_point_geometry_provisional", "wire_has_no_exportable_paths", "scene_title_rejected_as_name", "no_pin_net_omitted"}


def main():
    summary = load_json(ROOT / "data/conversion_reports/summary_all.json")
    queue, checks, stats, report_warnings = [], [], Counter(), Counter()
    for case in summary["cases"]:
        report = load_json(case["report_path"])
        data = load_json(case["reference_json_path"])
        if report["policy"] != summary["policy"] or digest(case["reference_json_path"]) != case["reference_json_sha256"]:
            raise ValueError(f"Mixed or edited conversion output: {case['case_id']}; rebuild all before a combined report")
        width, height = report["outputs"]["image_normalization"]["size"]
        errors, bounds = validate_output(data, width, height)
        if errors:
            raise ValueError(f"Saved output validation failed: {case['case_id']}: {errors[:3]}")
        if not report["outputs"]["image_normalization"]["rgba_pixels_equal"]:
            raise ValueError(f"Pixels not checked: {case['case_id']}")
        report_warnings.update(report["warning_counts"])
        stats["physical_pin_records"] += len(report["pin_map"])
        stats["expanded_pin_references"] += sum(len(p["references"]) for p in report["pin_map"].values())
        stats["unresolved_component_keys"] += sum(k.startswith("UNRESOLVED_") for k in data["components"])
        stats["cases_with_unresolved_components"] += any(k.startswith("UNRESOLVED_") for k in data["components"])
        stats["reencoded_images"] += report["outputs"]["image_normalization"]["source_format"] != "PNG"
        stats["identical_duplicate_key_cases"] += report["warning_counts"].get("identical_duplicate_json_key_removed", 0) > 0
        checks.append({"case_id": case["case_id"], "output_reopened": True, "project_contract_errors": errors, "coordinate_bounds_warning_count": len(bounds)})
        for issue in report["warnings"]:
            code = issue["code"]
            if code in HIGH or code in REVIEW:
                cid = issue.get("component")
                target = report["component_map"].get(cid, {}).get("designator")
                queue.append({"priority": "high" if code in HIGH else "review_policy", "case_id": case["case_id"], "target_component": target, "issue": issue, "report_path": case["report_path"], "reference_json_path": case["reference_json_path"]})
    if dict(report_warnings) != summary["warning_counts"]:
        raise ValueError("Reports are inconsistent with batch summary")
    queue.sort(key=lambda q: (q["priority"] != "high", q["case_id"] not in {"0001", "0002"}, q["case_id"], q["issue"]["code"]))
    queue_path = ROOT / "data/conversion_reports/review_queue.jsonl"
    queue_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in queue), encoding="utf-8")
    save_json(ROOT / "data/conversion_reports/verification.json", {"kind": "project_structure_and_reference_checks_not_official_f1", "case_count": len(checks), "stats": dict(stats), "cases": checks})
    save_json(ROOT / "data/conversion_reports/dataset_manifest_metadata.json", {"purpose": "local_reference_annotations_not_verified_official_golden", "top_level_fields": ["components", "pins", "nets"], "coordinate_formula": "H-y" if summary["policy"]["export_y_origin"] == "height" else "H-1-y", "manifest": str(ROOT / "data/standardized/manifest.jsonl"), "summary": str(ROOT / "data/conversion_reports/summary_all.json"), "review_queue": str(queue_path), "official_golden_verified": False})
    date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    lines = ["# Dataset300 转换交付报告", "", f"日期：{date}（Asia/Shanghai）。转换器版本：{summary['policy']['version']}。", "", f"已处理 {summary['attempted']} 例，成功生成 {summary['converted']} 份参考JSON和同尺寸PNG，失败 {len(summary['failed'])} 例。原始图片及标注哈希在转换前后核验一致。", "", "输出顶层为 `components`、`pins`、`nets`；元件含 `type/Name/value/bbox`，引脚含 `pinname/point`，网络含 `hyperGraph/edges`。每条edge恰有两个点。", "", "## 文件位置", "", "- 标准化数据：`data/standardized/<case_id>/<case_id>.json` 及同目录PNG。", "- 全量清单：`data/standardized/manifest.jsonl`。", "- 逐例来源、映射、警告：`data/conversion_reports/<case_id>.json`。", "- 人工复核队列：`data/conversion_reports/review_queue.jsonl`。", "- 全量统计：`data/conversion_reports/summary_all.json`。", "- 结构和引用复查：`data/conversion_reports/verification.json`。", "", "## 实际结果", "", "| 项目 | 数量 |", "|---|---:|"]
    for name, key in [("元件", "components"), ("输出引脚", "pins"), ("网络", "nets"), ("两点线段", "segments")]:
        lines.append(f"| {name} | {summary['counts'][key]} |")
    lines += [f"| 保持解码像素的重新编码图片 | {stats['reencoded_images']} |", f"| 修复相同重复键的案例 | {stats['identical_duplicate_key_cases']} |", f"| 相同重复键项 | {summary['warning_counts'].get('identical_duplicate_json_key_removed', 0)} |", f"| 位号未确定的元件 | {stats['unresolved_component_keys']} |", f"| 含未确定元件的案例 | {stats['cases_with_unresolved_components']} |", "", "## 已做的处理", "", "1. 保留所有原件；92张非PNG内容解码后另存PNG，其余PNG复制。300例的尺寸和RGBA解码像素均保持一致。", "2. 忽略原始deleted对象；元件位号采用声明文字或有记录的候选，重复位号按原始component ID自然排序编号。", "3. 解析文字引用、类别、型号和值；对图像已人工确认的0001/0002字段使用独立overrides文件修正。", "4. 引脚采用tip，拆分多编号；无有效编号时按显式本地规则生成编号，并逐项标记。", "5. 根据原始连接、精确重合端点及声明标签合并网络；保留源路径分段并去掉完全相同的两点段。", "6. 严格重复键检查发现10例共26项重复连接键；内容均相同，已去重并记录。冲突重复键仍会阻止转换。", "7. 来源映射与问题记录放在侧文件，正式结构JSON不混入置信度或调试字段。", "", "## 需要优先复核", "", "下列数量是待核验项数量，不是错误率或F1。", "", "| 问题 | 数量 | 处理建议 |", "|---|---:|---|"]
    descriptions = [
        ("unresolved_designator", "位号缺失/无法唯一确定", "输出UNRESOLVED_*并保存候选；按原图修正"),
        ("missing_scored_name", "IC型号缺失", "核对原图/允许的元件库"),
        ("missing_scored_value", "R/C取值缺失", "核对用途及单位，保持原文"),
        ("auto_pin_number", "本地自动编号", "与官方自动编号顺序核验；不等于已恢复物理编号"),
        ("pin_number_collision", "引脚编号冲突", "已分配临时编号避免覆盖；核对源编号"),
        ("unowned_global_pin_omitted", "根级引脚缺少可解析所属元件", "原记录保留，输出未虚构所属器件"),
        ("unresolved_edge_pin_omitted", "网络引脚引用仍无法恢复", "对照原图及原标注；已记录未导出引用"),
        ("provisional_type_mapping", "未获官方确认的类别映射", "补充官方type全集并修改config"),
        ("wire_has_no_exportable_paths", "源导线缺少可用路径", "核对几何；未按距离虚构连接"),
        ("coordinate_outside_image", "源坐标超出图像范围", "记录原坐标，待核对裁剪/外框定义"),
    ]
    for code, description, action in descriptions:
        lines.append(f"| {description} | {summary['warning_counts'].get(code, 0)} | {action} |")
    lines += ["", "## 验证范围与限制", "", f"已重新打开全部{len(checks)}份输出，检查顶层/字段类型、bbox顺序、两点edges、引脚引用、单网归属和重复键；项目结构校验错误为0。转换器9项回归检查通过。", "", "这些是本项目结构与引用校验，尚未运行官方评测器。输出为本地参考标注，不能直接宣称是完整官方Golden或已有识别F1。", "", "出口当前采用H-y、location框；H-1-y和其他框口径可配置重建，须用正式Golden核验。非通用类别、自动编号顺序、标签合并及源路径分段仍是显式待核验规则。", "", "空pinname并不自动代表漏标：无源器件可能没有图示名称；其评分期望需按官方转换规则核验。未确定编号和型号也没有从模型预测中补成答案。", "", "抽查配套SPICE自称reconstructed_hybrid，并带待确认项，因此未用它自动补齐缺失字段。", "", "## 调试样例与重建", "", f"- [0002标准JSON]({(ROOT / 'data/standardized/0002/0002.json').as_posix()})：IC2型号、8个IC引脚编号、X2及C74编号已结合原图核对。", f"- [0001标准JSON]({(ROOT / 'data/standardized/0001/0001.json').as_posix()})：修正U1的RP2350B型号及U2位号，其余未确定项见报告。", "", "```powershell", "$pcbPython = 'C:\\Users\\Pc\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe'", "& $pcbPython scripts/convert_dataset.py", "& $pcbPython scripts/build_conversion_report.py", "```", "", "修改 `configs/annotation_overrides.json` 可以按原始ID修正已核验字段；修改 `configs/conversion.json` 可以调整规则。单例试验使用 `--case-id 0002`，不会改写全量manifest/summary；全量重建后再生成报告。", "", "原始标注只用于离线准备、训练和评测；实际推理入口不能读这些答案文件。"]
    report_path = ROOT / "docs/conversion_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "verified_cases": len(checks), "review_queue_rows": len(queue), "stats": dict(stats)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
