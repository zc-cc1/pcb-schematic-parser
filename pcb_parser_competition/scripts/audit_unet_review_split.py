"""Audit decoded RGB duplicates in the dataset used by a completed U-Net run."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    project, review = args.project.resolve(), args.review.resolve()
    assert review != project and project not in review.parents
    training = json.loads((project / "runs/training/unet_wire_pin_v1/summary.json").read_text(encoding="utf-8"))
    validation_ids = set(training["validation_cases"])
    selected = validation_ids | set(training["train_cases"])
    manifest = [json.loads(line) for line in (project / "data/segmentation_unet_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in manifest if row["case_id"] in selected]
    groups = defaultdict(list)
    mismatches = []
    for row in rows:
        image_path = Path(row["image"])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            mismatches.append(row["case_id"])
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"))
        fingerprint = hashlib.sha256(str(array.shape).encode("ascii") + array.tobytes()).hexdigest()
        groups[fingerprint].append({"case_id": row["case_id"], "split": row["split"]})
    cross = [group for group in groups.values() if {row["split"] for row in group} == {"train", "val"}]
    validation_duplicate_groups = [group for group in groups.values() if sum(row["split"] == "val" for row in group) > 1]
    leaking_ids = {row["case_id"] for group in cross for row in group if row["split"] == "val"}
    audit = {
        "status": "decoded_RGB_exact_duplicate_audit_only_not_engineering_dedup",
        "image_count": len(rows), "image_file_hash_mismatches": mismatches,
        "cross_split_groups": cross, "validation_internal_duplicate_groups": validation_duplicate_groups,
        "validation_cases_with_identical_train_pixels": sorted(leaking_ids),
        "near_duplicate_and_engineering_group_audit": "pending",
    }
    assert not mismatches, "Source PNGs changed after manifest creation"
    evaluation_path = review / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    by_case = {row["case_id"]: row for row in rows}
    duplicate_label_conflicts = []
    for group in validation_duplicate_groups:
        signatures = []
        for item in group:
            row = by_case[item["case_id"]]
            with Image.open(row["wire_mask"]) as image:
                mask_bytes = np.asarray(image).tobytes()
            tips = json.loads((Path(row["wire_mask"]).parent / "pin_points.json").read_text(encoding="utf-8"))
            points = sorted(tuple(tip["point"]) for tip in tips)
            signatures.append(hashlib.sha256(mask_bytes + json.dumps(points).encode("ascii")).hexdigest())
        if len(set(signatures)) > 1:
            duplicate_label_conflicts.append(group)
    audit["duplicate_image_label_conflicts"] = duplicate_label_conflicts
    if not duplicate_label_conflicts:
        dropped = {row["case_id"] for group in validation_duplicate_groups
                   for row in sorted((r for r in group if r["split"] == "val"), key=lambda r: r["case_id"])[1:]}
        unique_records = [row for row in evaluation["cases"] if row["case_id"] not in dropped]
        def aggregate(key):
            tp, fp, fn = (sum(row[key][count] for row in unique_records) for count in ["tp", "fp", "fn"])
            return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp),
                    "recall": tp / (tp + fn), "f1_dice": 2 * tp / (2 * tp + fp + fn)}
        audit["unique_validation_image_count"] = len(unique_records)
        audit["duplicate_validation_cases_omitted"] = sorted(dropped)
        audit["unique_image_wire_pixels_micro"] = aggregate("wire_pixels")
        audit["unique_image_pin_localization_5px_micro"] = aggregate("pin_localization_5px")
    evaluation["decoded_pixel_split_audit"] = audit
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    (review / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["", "## 数据划分补查", "", f"已读取训练/验证的 {len(rows)} 页 PNG，并核验文件 SHA256 与清单一致。"]
    text.append(f"解码为 RGB 后，跨训练/验证的完全相同图像组：{len(cross)}；验证集内部重复组：{len(validation_duplicate_groups)}。")
    if leaking_ids:
        text.append("验证集中与训练图像像素完全相同的样例：" + "、".join(sorted(leaking_ids)) + "。当前指标包含这些样例，不能作为独立泛化成绩。")
    else:
        text.append("本次未发现训练和验证间像素完全相同的图片；近似重复与同工程不同页的分组仍未核验。")
    if not duplicate_label_conflicts:
        text.append(f"验证集两对像素重复图片的导线/引脚坐标标签一致，去除重复页后为 {len(unique_records)} 张唯一图像：导线像素 Dice {audit['unique_image_wire_pixels_micro']['f1_dice']:.2%}；引脚坐标 F1 {audit['unique_image_pin_localization_5px_micro']['f1_dice']:.2%}。这仍不是工程级独立测试成绩。")
    text += ["", "## 观察与下一步", "",
             "- 0002：导线覆盖较好，但误判了部分 IC 外框；电容和地符号附近的引脚点有偏移或漏检。",
             "- 0012：彩色导线存在较多漏检，引脚定位相对更好。",
             "- 0208：当前阈值下未输出导线，62 个标注引脚只匹配到 1 个；优先核对原图尺度、细线/低对比特征以及草稿标签。",
             "- 先复核标签与难例，随后在独立开发子集上校准引脚热图解码和阈值，再接入 YOLO 元件归属、OCR 引脚编号/名称以及 nets 构建。",
             "- 本轮仅做训练结果复核，没有重新训练或修改来源工程。", ""]
    report_path = review / "report.md"
    original_report = report_path.read_text(encoding="utf-8").partition("## 数据划分补查")[0].rstrip()
    report_path.write_text(original_report + "\n" + "\n".join(text), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
