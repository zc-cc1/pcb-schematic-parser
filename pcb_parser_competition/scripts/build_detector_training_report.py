"""Summarize the completed experiment using saved evidence, not estimates."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    audit = read("data/detection_v1/audit.json")
    review = read("artifacts/detection_data_review/visual_review_record.json")
    training = read("runs/training/detector_dataset300_v1_record.json")
    progress = read("runs/training/detector_dataset300_v1_progress.json")
    before = read("runs/detector_comparison/before_whole.json")
    after = read("runs/detector_comparison/after_whole.json")
    if before["manifest_sha256"] != after["manifest_sha256"]:
        raise RuntimeError("Before/after validation manifests differ")
    if any(before[k] != after[k] for k in ("mode", "confidence", "iou_match_threshold", "split")):
        raise RuntimeError("Before/after evaluation settings differ")
    before_ids = [c["case_id"] for c in before["cases"]]
    after_ids = [c["case_id"] for c in after["cases"]]
    if before_ids != after_ids:
        raise RuntimeError("Before/after evaluation case sets differ")
    with (Path(training["run_directory"]) / "results.csv").open(encoding="utf-8-sig") as stream:
        epochs = list(csv.DictReader(stream))
    lines = ["# Dataset300 检测器首轮微调实测", "", "## 本轮完成的工作", "",
             f"已从300例源数据导出检测标签，去掉3张完全重复图后保留{audit['exported_cases']}张：训练{audit['image_counts']['train']}张、验证{audit['image_counts']['val']}张。",
             f"检测类别共{len(audit['config']['class_names'])}类；训练框{sum(audit['label_counts']['train'].values())}个，验证框{sum(audit['label_counts']['val'].values())}个。",
             f"看图复核了{review['visually_inspected_unique_objects']}个独立元件裁剪图，记录{review['overrides']}项人工修正或排除；详情在 `configs/detection_label_overrides.json`。",
             "修正例子包括：ANT中误标的运放、Fuse中误标的电阻/开关、Crystal中误标的电容，以及SEG中的标题文字与连接器。复核证据和原始裁剪图保存在 `artifacts/detection_data_review/`。", "",
             "0001和0002被固定在验证集。完全重复图、pHash相似图和可用同源文件名线索按组隔离；原始工程来源尚不完整，不能证明已隔离所有同源工程。",
             "检测训练只使用类别和location框，左上坐标归一化成YOLO格式；位号、Name、value、引脚与网络字段未作为检测标签。10个越界框已裁到图片范围，源文件保持可追溯。", "",
             "原始数据的全部标注尚未人工认证。本次指标来自经局部人工修正的原始检测标签，属于实验指标。", "",
             "## 训练配置与实际运行", "",
             f"以HR-DONK电路检测权重初始化，在RTX 5060上完成{len(epochs)}轮；总耗时{training['wall_seconds']/60:.2f}分钟。",
             "1024输入、batch=4、2个数据加载进程、AdamW、初始学习率0.001、FP32；细节在 `configs/detector_training.json` 和运行 `args.yaml`。",
             f"PyTorch分配显存峰值{progress['peak_allocated_mib']:.0f} MiB；这不是整张显卡的总占用。",
             "按类别名称复制了原权重10类在三个检测分支的30个分类输出行，并逐行验证一致；新增类别输出行单独初始化。",
             f"最佳权重：`{training['best_weights']}`。SHA256：`{training['best_weights_sha256']}`。", "",
             "## 同一验证集的前后比较", "",
             "两个模型使用完全相同的验证案例、标签版本、1024整图输入、confidence=0.25，以及类别名称匹配、IoU≥0.5的一对一框匹配。", "",
             "| 指标 | 原始权重 | 微调权重 |", "|---|---:|---:|"]
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        lines.append(f"| {label} | {before['overall'][key]:.4f} | {after['overall'][key]:.4f} |")
    lines.extend(["", f"F1变化：{after['overall']['f1']-before['overall']['f1']:+.4f}。这是类别+框的本地微平均F1，不是比赛要求位号、型号、数值和像素容差的ComponentF1，也不包含引脚/网络分数。", "",
                  "| 类别 | 验证框数 | 原始F1 | 微调F1 |", "|---|---:|---:|---:|"])
    for name in ("resistor", "capacitor", "ic", "ground", "voltage_source", "crystal", "inductor", "opamp", "mosfet"):
        b, a = before["by_class"].get(name, {}), after["by_class"].get(name, {})
        lines.append(f"| {name} | {a.get('tp',0)+a.get('fn',0)} | {b.get('f1',0):.4f} | {a.get('f1',0):.4f} |")
    absent = [n for n in audit["config"]["class_names"] if audit["label_counts"]["val"].get(n,0)==0]
    lines.extend(["", "验证集没有以下类别，因此本轮无法评价它们的泛化：" + ", ".join(absent) + "。", "",
                  "晶振与MOSFET在当前confidence=0.25下均没有有效检出，不能把类别表扩展理解为这些类别已具备可靠识别能力。后续需要复核更多稀有类标签、针对性切片增样或调整训练策略。", "",
                  "训练器保存的mAP50/mAP50-95使用Ultralytics默认AP计算口径，不与上面的固定阈值F1混用。最佳checkpoint的训练器指标：", ""])
    for key,value in training["metrics"].items():
        if key.startswith("metrics/"):
            lines.append(f"- {key}: {value:.4f}")
    fine_summary = ROOT / "runs/s2_finetuned_quick_dev/summary.json"
    if fine_summary.exists():
        fine = json.loads(fine_summary.read_text(encoding="utf-8"))
        lines.extend(["", "## 两例整图/切片复查", "", "| 案例 | 模式 | 候选数 | 暂绑定位号数 | 检测中位耗时 ms |", "|---|---|---:|---:|---:|"])
        for row in fine["runs"]:
            lines.append(f"| {row['case_id']} | {row['mode']} | {row['detection_count']} | {row['resolved_designator_count']} | {row['timing']['median_ms']:.1f} |")
        lines.extend(["", "预测框、OCR、绑定调试、overlay和局部赛事结构JSON保存在 `runs/s2_finetuned_quick_dev/`。",
                      "两例已多次用于调试，属于开发验证案例；更多候选不直接代表更高准确率。Name/value、引脚和网络尚未实现，JSON仍为局部结果。"])
        comparison = ROOT / "runs/detector_comparison/quick_dev_modes.json"
        if comparison.exists():
            mode_rows = json.loads(comparison.read_text(encoding="utf-8"))
            lines.extend(["", "已保存预测的本地IoU匹配比较：", "", "| 案例 | 模型 | 模式 | TP | FP | FN | F1 |", "|---|---|---|---:|---:|---:|---:|"])
            for row in mode_rows:
                m = row["overall"]
                lines.append(f"| {row['case_id']} | {row['model']} | {row['mode']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['f1']:.4f} |")
            lines.extend(["", "0002的新模型切片F1为0.8462，高于整图0.8000；0001切片F1为0.6632，低于整图0.6957。当前保留两种模式用于调试，后续要改善重复框/碎片框合并，再决定何时启用SAHI。",
                          "可视化确认0002的IC和竖向电容已有检测框，但X2晶振被误分类为电容；0001的Y1晶振也仍有漏检。"])
    lines.extend(["", "## 再次运行", "", "项目根目录下：", "", "```powershell",
                  r".\.venv-gpu\Scripts\python.exe scripts/prepare_detection_dataset.py",
                  r".\.venv-gpu\Scripts\python.exe scripts/train_detector.py --run-name detector_dataset300_v2",
                  r".\.venv-gpu\Scripts\python.exe scripts/run_baseline.py --config configs/baseline_finetuned.json --image data/standardized/0002/0002.png --image data/standardized/0001/0001.png --devices cuda:0 --modes whole sliced --output runs/s2_finetuned_quick_dev", "```", "",
                  "改变标签/划分后要同时重新评估原始和微调权重，比较工具会拒绝混用不同manifest。", "",
                  "## 后续优先事项", "", "优先针对晶振、MOSFET等稀有类做切片增样与类别均衡训练，并改善SAHI的重复框/碎片框合并；同时补足标签和同源工程复核。",
                  "随后推进OCR的位号/型号/数值绑定，再接入引脚、导线骨架和网络规则。", ""])
    (ROOT / "docs/detector_training_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(ROOT / "docs/detector_training_report.md"))


if __name__ == "__main__":
    main()
