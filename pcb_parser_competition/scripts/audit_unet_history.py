"""Reconstruct experiments and freeze the 79.46% audit before new inference."""
import json
from pathlib import Path
import subprocess
from collections import Counter

import torch

from unet_audit_common import ROOT, OUT, rows, read, write, sha, image_path, gt_path, metrics


def main():
    if (OUT/"protocol.json").exists():
        raise ValueError("Audit already prepared; use existing frozen protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = rows()
    split = read(ROOT/"runs/unet_inference_v2/split.json")
    val = split["development"]+split["acceptance_holdout"]
    checkpoint_path = ROOT/"weights/unet_wire_pin_v1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert sha(checkpoint_path) == split["source_checkpoint_sha256"]
    old = read(ROOT/"runs/unet_result_review_20261011/evaluation.json")
    original_run = Path(old["source_training_run"])
    original_history = read(original_run/"history.json")
    original_summary = read(original_run/"summary.json")
    assert len(original_history) == 60 and original_summary["checkpoint_sha256"] == sha(checkpoint_path)
    protected = read(ROOT/"data/segmentation_unet_v2_reviewed_draft/summary.json")["protected_source_sha256"]
    for row in dataset:
        for p in (image_path(row["case_id"]), gt_path(row["case_id"])):
            protected[str(p)] = sha(p)
    relevant = [checkpoint_path, original_run/"best.pt", original_run/"last.pt", original_run/"history.json",
        original_run/"summary.json", ROOT/"configs/unet_wire_pin.json", ROOT/"configs/unet_inference_v2.json",
        ROOT/"data/segmentation_unet_v1/manifest.jsonl", ROOT/"runs/unet_result_review_20261011/evaluation.json",
        ROOT/"runs/unet_inference_v2/comparison.json", ROOT/"runs/unet_inference_v2/calibration_search.json",
        ROOT/"runs/unet_contrast_verification_v1/evaluation.json", ROOT/"runs/unet_retrain_v2/evaluation/evaluation.json",
        ROOT/"runs/unet_retrain_rotation_v3/evaluation/evaluation.json"]
    for run in (ROOT/"runs/unet_retrain_v2/control", ROOT/"runs/unet_retrain_v2/contrast", ROOT/"runs/unet_retrain_rotation_v3/rotation"):
        relevant.extend(run/name for name in ("best.pt", "last.pt", "configuration.json", "history.json", "summary.json"))
    for path in relevant:
        if path.exists():
            protected[str(path)] = sha(path)
    assert all(sha(p) == h for p, h in protected.items())
    protocol = dict(status="read_only_sources_frozen_no_training", checkpoint_sha256=sha(checkpoint_path),
        checkpoint_epoch=checkpoint["epoch"], validation_unique_cases=val,
        original_validation_cases=[r["case_id"] for r in dataset if r["split"] == "val"],
        train_cases=[r["case_id"] for r in dataset if r["split"] == "train"],
        original_manifest_sha256=sha(ROOT/"data/segmentation_unet_v1/manifest.jsonl"),
        protected_sources=protected, thresholds=[round(.05*k, 2) for k in range(1, 20)],
        threshold_scan_use="descriptive only; no validation-based parameter selection or production adoption",
        inference_variants_predeclared=["native_uniform512", "single_fit512", "d4_mean", "scale075", "scale125",
            "scale_mean", "d4_multiscale_mean", "gaussian512", "gaussian1024", "closing3", "remove_area4",
            "remove_area16", "linear_components", "endpoint_bridge2", "ensemble_v1_control", "ensemble_all4"],
        error_definitions="binary draft labels; shape/semantic proxies explicitly separated from manual judgement",
        no_training=True, official_score=None, independent_test_set=False)
    write(OUT/"protocol.json", protocol)
    history = []
    def add(id, variables, count, metric, **extra):
        history.append(dict(id=id, variables=variables, validation_cases=count, wire_micro=metric,
            threshold=.5, aggregation="full-page micro, each pixel once", **extra))
    original28 = old["full_page_wire_pixels_micro"]
    add("v1_28_original", "原模型；28页含两张重复图", 28,
        metrics(*(original28[k] for k in ("tp", "fp", "fn"))), seed=20261011, epochs=60, best_epoch=57,
        checkpoint_selection="minimum original validation loss", train_cases=103, training_labels="v1 3px drafts")
    selected = [r for r in old["cases"] if r["case_id"] in val]
    dedup = metrics(*(sum(r["wire_pixels"][k] for r in selected) for k in ("tp", "fp", "fn")))
    add("v1_26_dedup", "同一模型/预测，去掉0268与0274重复图", 26, dedup,
        seed=20261011, epochs=60, best_epoch=57, checkpoint_selection="same frozen checkpoint; no refit", train_cases=103)
    contrast = read(ROOT/"runs/unet_contrast_verification_v1/evaluation.json")
    for r in contrast["groups"]["all"]:
        if r["gamma"] != 1:
            add(f"global_gamma_{r['gamma']}", "冻结模型，只对输入全图做gamma变换", 26, r["wire_micro"],
                epochs=0, best_epoch=57, seed=None, trained=False, connected_paths=r["connected_paths"])
    comparisons = [(ROOT/"runs/unet_retrain_v2/evaluation/evaluation.json", "control", ROOT/"runs/unet_retrain_v2/control"),
                   (ROOT/"runs/unet_retrain_v2/evaluation/evaluation.json", "contrast", ROOT/"runs/unet_retrain_v2/contrast"),
                   (ROOT/"runs/unet_retrain_rotation_v3/evaluation/evaluation.json", "rotation", ROOT/"runs/unet_retrain_rotation_v3/rotation")]
    for evaluation_path, arm, folder in comparisons:
        evaluation = read(evaluation_path); config = read(folder/"configuration.json"); summary = read(folder/"summary.json")
        r = next(r for r in evaluation["groups"]["all"] if r["variant"] == arm)
        variables = {"control": "v2训练标签局部修正+普通微调", "contrast": "与control配对，仅加灰度增强", "rotation": "与control配对，仅加四方向同步旋转"}[arm]
        add(arm, variables, 26, r["wire_micro"], seed=config["seed"], epochs=config["epochs"], best_epoch=summary["best_epoch"],
            checkpoint_selection=config["checkpoint_selection"], initial_weights_sha256=config["initial_sha256"],
            manifest_sha256=summary["manifest_sha256"], train_cases=103, training_labels="v2 reviewed drafts, validation masks unchanged",
            connected_paths=r["connected_paths"], config_sha256=sha(folder/"configuration.json"),
            checkpoint_sha256=summary["checkpoint_sha256"], best_record=summary["best_validation"],
            last_record=read(folder/"history.json")[-1])
    write(OUT/"experiment_history.json", dict(experiments=history, original_history=original_history,
        original_best_record=min(original_history, key=lambda r: r["val_loss"]), original_last_record=original_history[-1],
        other_existing_inference_comparison=read(ROOT/"runs/unet_inference_v2/comparison.json"),
        prior_calibration_search=read(ROOT/"runs/unet_inference_v2/calibration_search.json"),
        notebook_files=[p for p in (ROOT/"runs/unet_audit_inventory_paths.txt").read_text(encoding="utf-8-sig").splitlines() if p.endswith(".ipynb")],
        git=subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True).returncode == 0,
        history_confounds=dict(original_vs_dedup="sample weights changed, model/predictions identical",
            v1_vs_finetune="training label corrections, optimizer restart, seed/LR, epoch budget, validation dedup changed",
            control_vs_contrast_or_rotation="paired same init/config/labels/order; augmentation is sole arm variable")))
    lines = ["# 实验史重建", "", "| 实验 | 改变量 | 验证页 | 整图micro Dice@0.5 | seed | 新训练轮数 | 最佳轮 |", "|---|---|---:|---:|---|---:|---:|"]
    for h in history:
        lines.append(f"| {h['id']} | {h['variables']} | {h['validation_cases']} | {h['wire_micro']['dice']:.4%} | {h.get('seed')} | {h['epochs']} | {h['best_epoch']} |")
    lines += ["", "79.9816%→79.4613%来自去重引起的样本加权改变；逐像素预测未改变。后续26页对照均为整图micro、阈值0.5、同一验证标签，下降是真实同口径差异。",
        "原训练最佳轮指标按重叠tile统计，不能直接与整图micro混用。原模型60轮/seed20261011与warm-start16轮/seed20261014也不是单变量对照；paired control才用于区分增强效应。",
        "仓库不是Git仓库，故没有git log；notebook清单见JSON。原训练完整日志从历史记录指向的只读来源目录读取。"]
    (OUT/"01_experiment_history.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), original28=history[0]["wire_micro"], dedup26=dedup,
        protected_sources=len(protected), training_not_started=True), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
