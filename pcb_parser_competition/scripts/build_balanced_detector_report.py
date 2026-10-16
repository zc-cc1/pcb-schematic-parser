"""Report the second detector experiment from frozen validation evidence."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def main():
    audit = read("data/detection_balanced_v2/audit.json")
    verification = read("data/detection_balanced_v2/verification.json")
    training = read("runs/training/detector_dataset300_balanced_v2_record.json")
    progress = read("runs/training/detector_dataset300_balanced_v2_progress.json")
    evaluation_folder = ROOT/"runs/detector_comparison_v2"
    paths = [ROOT/"runs/detector_comparison/before_whole.json", ROOT/"runs/detector_comparison/after_whole.json",
             evaluation_folder/"balanced_whole.json"]
    labels = ["GitHub原始权重", "第一轮整图", "第二轮整图"]
    paths.append(evaluation_folder/"first_sliced_ios.json")
    labels.append("第一轮SAHI/IOS/GREEDYNMM")
    for filename, label in (("balanced_sliced_ios.json", "第二轮SAHI/IOS/GREEDYNMM"),
                            ("balanced_sliced_iou_nms.json", "第二轮SAHI/IOU/NMS")):
        if (evaluation_folder/filename).exists():
            paths.append(evaluation_folder/filename)
            labels.append(label)
    evaluations = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    initial = evaluations[0]
    for evaluation in evaluations:
        assert evaluation["manifest_sha256"] == initial["manifest_sha256"] == audit["source_manifest_sha256"]
        assert [r["case_id"] for r in evaluation["cases"]] == [r["case_id"] for r in initial["cases"]]
        assert all(evaluation[k] == initial[k] for k in ("confidence", "iou_match_threshold", "split"))
    assert training["status"] == "completed"
    assert all(e["weights_sha256"] == training["best_weights_sha256"] for i,e in enumerate(evaluations) if i not in (0,1,3))
    with (Path(training["run_directory"])/"results.csv").open(encoding="utf-8-sig") as stream:
        epochs = list(csv.DictReader(stream))
    lines = ["# 稀有类别增样与第二轮检测器微调", "", "## 数据和实验边界", "",
             f"保留238张原始训练页，增加{sum(audit['crop_counts'].values())}张训练局部图，合计{audit['image_counts']['train']}张训练样本；验证集继续使用第一轮的59张整图和2462个检测框。",
             "新增局部图的父图只来自训练集，父图组与源图像哈希均不跨训练/验证；0001/0002仍在验证集。",
             "切片遇到边界元件会扩展窗口，使所有与窗口相交的已标注元件完整保留，避免丢弃部分标签后把符号当作背景。",
             "增样增加了同一来源对象的训练曝光次数，不等于增加独立样本；同源工程信息和全量标注认证仍不完整。", "",
             "| 增样目标 | 局部图数 | 原训练标注数 | 增样后训练标注数（含重复曝光） |", "|---|---:|---:|---:|"]
    original = read("data/detection_v1/audit.json")
    for name, count in audit["crop_counts"].items():
        lines.append(f"| {name} | {count} | {original['label_counts']['train'][name]} | {audit['label_counts']['train'][name]} |")
    lines.extend(["", "人工查看99个原始晶振/MOSFET训练对象，确认两处训练误标：0087/C_83为4.7µF电容，0203/C_84为四脚按键。修正只写入第二轮训练派生标签，原始数据、第一轮标签和所有验证标签不变。",
                  "原始复核图册与局部图窗口示例在 `artifacts/detection_balanced_review/`，修正依据在 `configs/detection_augmentation.json`。", "",
                  f"标签坐标往返最大误差{verification['maximum_bbox_roundtrip_error_px']:.8f}px；验证集逐条一致，源组交集为0。切片边界、验证来源隔离和检测匹配6项回归测试通过。", "",
                  "## 实际训练", "",
                  f"以第一轮35类权重初始化，所有35类检测头输出按名称保留；新建优化器，完成{len(epochs)}轮训练，耗时{training['wall_seconds']/60:.2f}分钟。",
                  f"RTX 5060、FP32、1024输入、batch=4；PyTorch分配显存峰值{progress['peak_allocated_mib']:.0f}MiB，不代表整卡占用。学习率0.0005，保留原页与局部图混合训练，关闭镜像翻转。",
                  f"权重：`{training['best_weights']}`；SHA256：`{training['best_weights_sha256']}`。", "",
                  "这是一轮新配置实验，同时包含局部图增样、两处训练标签修正和训练参数变化，尚未通过消融实验分离每项变化的贡献。", "",
                  "## 固定验证集结果", "",
                  "所有结果均来自相同59张验证图，confidence=0.25，类别名称一致且IoU≥0.5的一对一匹配。SAHI采用1024窗口、0.125重叠，并加一次整图预测。", "",
                  "| 模型/推理 | Precision | Recall | 微平均F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label,e in zip(labels,evaluations):
        m=e["overall"]
        lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines.extend(["", "| 类别 | 验证标注数 | 第一轮整图F1 | 第二轮整图F1 | 第二轮切片最佳F1 |", "|---|---:|---:|---:|---:|"])
    for name in ("crystal", "mosfet", "switch", "led", "resistor", "capacitor", "ic", "ground", "voltage_source", "transistor", "opamp", "inductor"):
        before=evaluations[1]["by_class"].get(name,{})
        after=evaluations[2]["by_class"].get(name,{})
        slices=max((e["by_class"].get(name,{}).get("f1",0) for e in evaluations[4:]),default=0)
        lines.append(f"| {name} | {original['label_counts']['val'].get(name,0)} | {before.get('f1',0):.4f} | {after.get('f1',0):.4f} | {slices:.4f} |")
    lines.extend(["", "上表逐类切片最佳值可能来自不同合并配置，不能把各类最佳值拼成一个模型的总体表现。",
                  "对30个验证晶振/MOSFET对象进行了预测图复核，诊断图在 `artifacts/detection_balanced_review/validation_rare_predictions_01.png` 和 `_02.png`。IOS切片正确匹配5/14个晶振和7/16个MOSFET；许多两脚晶振仍误分类为电容/电阻，部分四脚振荡器开始有效检出。0001的Y1和0002的X2仍未正确分类，不能宣称这两张开发图的晶振问题已经解决。",
                  "本地F1只评价元件类别和框，尚不包含位号、Name/value、引脚、网络和赛事像素容差；它不是官方ComponentF1或比赛总分。验证集已用于多轮开发，未设独立最终测试集。", "",
                  "训练器默认AP口径：", ""])
    lines.extend(f"- {k}: {v:.4f}" for k,v in training["metrics"].items() if k.startswith("metrics/"))
    selection_path=evaluation_folder/"selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding="utf-8"))
        lines.extend(["", "## 开发默认配置", "", selection["explanation"],
                      "默认配置：`configs/baseline_finetuned.json`。第一轮权重及原始实验配置仍保留，可随时复查。"])
    quick_path=ROOT/"runs/s2_balanced_quick_dev/summary.json"
    if quick_path.exists():
        quick=json.loads(quick_path.read_text(encoding="utf-8"))
        assert quick["weights_sha256"] == training["best_weights_sha256"]
        lines.extend(["", "## 图片推理入口复查", "", "推理只读取图片、模型和配置，检测/OCR、暂定位号与局部JSON完整运行；四项均为CUDA检测且通过本地结构检查。Name/value、pins/nets仍未实现，局部JSON不是完整比赛答案。", "",
                      "| 图像 | 模式 | 候选数 | 暂绑定位号数 | 检测中位耗时ms |", "|---|---|---:|---:|---:|"])
        for r in quick["runs"]:
            lines.append(f"| {r['case_id']} | {r['mode']} | {r['detection_count']} | {r['resolved_designator_count']} | {r['timing']['median_ms']:.1f} |")
        lines.extend(["", "计时预热一次、重复三次；只包括检测预处理、推理、后处理和GPU同步，不含模型加载、OCR或文件读写。候选数和暂绑定位号数不等于准确率。产物在 `runs/s2_balanced_quick_dev/`。"])
    lines.extend(["", "## 复现", "", "```powershell",
                  r".\.venv-gpu\Scripts\python.exe scripts/prepare_balanced_detection_dataset.py",
                  r".\.venv-gpu\Scripts\python.exe scripts/train_detector.py --config configs/detector_training_balanced.json --run-name detector_dataset300_balanced_v3",
                  r".\.venv-gpu\Scripts\python.exe scripts/evaluate_detector.py --weights weights/detector_dataset300_balanced_v2.pt --mode whole --output runs/detector_comparison_v2/balanced_whole.json",
                  "```", "", "推理入口仍只读取图片、权重与配置；验证标签只由离线评测脚本读取。后续推进OCR位号/型号/数值绑定，并继续检查仍然低召回的类别。", ""])
    (ROOT/"docs/detector_balanced_training_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(str(ROOT/"docs/detector_balanced_training_report.md"))


if __name__ == "__main__":
    main()
