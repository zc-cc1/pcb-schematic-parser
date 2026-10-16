"""Package the accepted pin-only change while rejecting the wire trial.

Do not change the frozen search or the original acceptance report. This is a
feature acceptance decision after validation, not another tuning pass.
"""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/unet_inference_v2"
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.unet_inference import decode_pins
from evaluate_trained_unet import match_points, draw_review


def main():
    comparison = json.loads((RUN / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["acceptance_deltas"]["pin_coordinate_f1"] >= 0.005
    assert comparison["acceptance_deltas"]["wire_dice"] < 0
    config = dict(comparison["config"])
    config.update(wire_mode="native", wire_threshold=0.5,
                  acceptance_status="pin_change_accepted_wire_multiscale_trial_rejected")
    (ROOT / "configs/unet_inference_v2.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"status": config["acceptance_status"], "inference_config": config,
              "official_score": None, "independent_engineering_test": False,
              "source_trial_acceptance_passed": comparison["acceptance_passed"],
              "decision": "Reject the combined trial; adopt pin decoding only and preserve wire baseline exactly.",
              "scope": comparison["scope"], "cases": []}
    for row in comparison["cases"]:
        result["cases"].append({"case_id": row["case_id"], "subset": row["subset"], "baseline": row["baseline"],
                               "adopted": {"wire": row["baseline"]["wire"], "pin": row["optimized"]["pin"]}})
    split = json.loads((RUN / "split.json").read_text(encoding="utf-8"))
    by_case = {row["case_id"]: row for row in split["rows"]}
    for record in result["cases"]:
        row = by_case[record["case_id"]]
        probability = np.load(RUN / "cache" / row["case_id"] / "scale_1.0.npz")["probability"]
        candidates = decode_pins(probability[1], config["pin_threshold"], config["pin_peak_kernel"],
                                 config["pin_refinement"], config["pin_nms_radius"])
        with Image.open(row["wire_mask"]) as image:
            expected_wire = np.asarray(image) >= 128
        expected_pins = [p["point"] for p in json.loads((Path(row["wire_mask"]).parent / "pin_points.json").read_text(encoding="utf-8"))]
        matches, actual = match_points([p["point"] for p in candidates], expected_pins)
        assert actual == record["adopted"]["pin"]
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        folder = RUN / "adopted_examples"
        folder.mkdir(exist_ok=True)
        visual_record = {"case_id": row["case_id"] + " adopted v2", "checkpoint_epoch": split["checkpoint_epoch"],
                         "wire_pixels": record["adopted"]["wire"], "pin_localization_5px": actual}
        draw_review(rgb, expected_wire, probability[0] >= 0.5, expected_pins,
                    [p["point"] for p in candidates], matches, visual_record).save(folder / f"{row['case_id']}.png")
    for subset in ["development", "acceptance_holdout", "all_unique"]:
        original = comparison[subset]
        result[subset] = {"cases": original["cases"], "baseline": original["baseline"],
                          "adopted": {"wire": original["baseline"]["wire"], "pin": original["optimized"]["pin"]}}
    (RUN / "adopted_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# U-Net 引脚推理优化 v2：采用版本", "",
             "使用第 57 轮 best.pt，未重新训练。采用引脚概率阈值 0.8、5×5 局部峰值、对数概率抛物线亚像素定位。",
             "导线保持原尺度、0.5 阈值。开发集选出的导线多尺度候选在验收图上下降，已拒绝；候选实验报告及冻结配置完整保留。",
             "空间 NMS 功能已实现，但此次开发集选定半径为 0，采用版本没有额外空间合并。", "",
             "先按像素去重，再固定 13 张开发图/13 张验收图；只在开发图调参。原模型曾用这些页选择 checkpoint，因此不是独立工程测试。标注仍为草稿。", "",
             "| 子集 | 张数 | 导线像素 Dice 原版→采用 | 引脚坐标 F1 原版→采用 |", "|---|---:|---:|---:|"]
    for subset, name in [("development", "调参"), ("acceptance_holdout", "保留验收"), ("all_unique", "全部唯一图片")]:
        row = result[subset]
        lines.append(f"| {name} | {row['cases']} | {row['baseline']['wire']['f1_dice']:.2%} → {row['adopted']['wire']['f1_dice']:.2%} | {row['baseline']['pin']['f1_dice']:.2%} → {row['adopted']['pin']['f1_dice']:.2%} |")
    old, new = result["all_unique"]["baseline"]["pin"], result["all_unique"]["adopted"]["pin"]
    lines += ["", f"26 张去重图片：引脚 Precision {old['precision']:.2%} → {new['precision']:.2%}；Recall {old['recall']:.2%} → {new['recall']:.2%}。",
              f"引脚 TP {old['tp']} → {new['tp']}，FP {old['fp']} → {new['fp']}，FN {old['fn']} → {new['fn']}。阈值提升带来更少误检，也损失部分召回。", "",
              "坐标按原图左上角坐标、≤5px 欧氏距离一对一匹配。这里没有评价引脚归属、编号、名称与 nets，不是官方 PinF1 或比赛总分。", "",
              "## 可运行入口", "", "```powershell",
              ".\\scripts\\run_unet_optimized.ps1 -Image 'data/standardized/0002/0002.png' -Device 'cuda:0'", "```", "",
              "输出概率图、导线 mask、亚像素引脚候选和运行摘要。PNG 推理入口不读取标注。默认入口不加 inference-config 时仍保留 v1 解码。", "",
              "## 后续难点", "",
              "- 电容、地符号附近的引脚 tip 与元件接线部位仍有较大偏移；复核 tip 监督，并补充此类样例后再训练。",
              "- 0208 的整页细线仍是难例，应核查原图尺度、对比度与草稿标签。",
              "- 进一步减少误检需要关联 YOLO 元件框；引脚编号/名称与 nets 仍待实现。", "",
              "## 逐页指标", "", "| 样例 | 子集 | 引脚 F1 原版→采用 | TP/FP/FN 原版→采用 | 采用图 |", "|---|---|---:|---|---|" ]
    for row in result["cases"]:
        old, new = row["baseline"]["pin"], row["adopted"]["pin"]
        picture = (RUN / "adopted_examples" / f"{row['case_id']}.png").as_posix()
        lines.append(f"| {row['case_id']} | {row['subset']} | {old['f1_dice']:.2%} → {new['f1_dice']:.2%} | {old['tp']}/{old['fp']}/{old['fn']} → {new['tp']}/{new['fp']}/{new['fn']} | [查看](<{picture}>) |")
    (RUN / "adopted_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "inference_config"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
