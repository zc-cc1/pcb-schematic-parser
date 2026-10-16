"""A single predeclared YOLO11s warm-start refinement, with unchanged scoring."""
from pathlib import Path
import argparse
import copy
import csv
import json
import shutil
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "pcb_parser_competition"
RUN = "11s_finetune_seed20261012"
FOLDER = HERE / "runs" / RUN
import experiment_runtime as runtime
from matched_helpers import read, write, sha, utc, libraries, verify_inputs, collect, tune, counts, bootstrap, metric_self_check

# Reuse the already-tested Windows idle check and duplicate-process locks locally.
# The child must execute this controller, which defines the refinement recipe.
runtime.HERE, runtime.RUN, runtime.FOLDER, runtime.__file__ = HERE, RUN, FOLDER, __file__
status = runtime.status


def verify_recipe(cfg):
    import yaml
    assert runtime.HERE == HERE and runtime.RUN == RUN and runtime.FOLDER == FOLDER
    assert Path(runtime.__file__).resolve() == Path(__file__).resolve(), "Idle launcher must execute the refinement controller."
    verify_inputs(cfg)
    parent = read(cfg["parent_protocol"]["path"])
    expected = dict(parent["training"], **cfg["fine_tune_changes"])
    assert cfg["training"] == expected
    assert cfg["training"]["epochs"] == 40 and cfg["training"]["batch"] == 4 and cfg["training"]["imgsz"] == 1536
    parent_folder = Path(cfg["parent_protocol"]["path"]).parent
    assert sha(parent_folder / "dataset/manifest.jsonl") == parent["copied_manifest_sha256"]
    new = [json.loads(line) for line in (HERE / "dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    old = [json.loads(line) for line in (parent_folder / "dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [{k: v for k, v in r.items() if k not in ("image", "label")} for r in new] == [{k: v for k, v in r.items() if k not in ("image", "label")} for r in old]
    path_map = {before["image"]: after["image"] for before, after in zip(old, new)}
    for split in ("train", "calibration", "audit"):
        before = (parent_folder / "dataset" / (split + ".txt")).read_text(encoding="utf-8").splitlines()
        after = (HERE / "dataset" / (split + ".txt")).read_text(encoding="utf-8").splitlines()
        assert after == [path_map[path] for path in before]
    data = yaml.safe_load(Path(cfg["dataset"]).read_text(encoding="utf-8"))
    assert data["train"] == "train.txt" and data["val"] == "calibration.txt"
    assert Path(data["path"]).resolve() == (HERE / "dataset").resolve()
    assert yaml.safe_load((HERE / "dataset/audit_dataset.yaml").read_text(encoding="utf-8")) == {**data, "val": "audit.txt"}
    for font in (HERE / "ultralytics/Ultralytics/Arial.ttf", HERE / "Ultralytics/Arial.ttf"):
        assert sha(font) == "525979822591a3447cfc49d943d6f7683508e25543407871c0ed8fed05fd2bd9"


def summarize(cfg, saved):
    import numpy as np
    entry = {"weights": saved["weights"], "weights_sha256": saved["weights_sha256"], "modes": {}}
    for mode in ("whole", "sliced", "fused"):
        thresholds, details = tune(cfg, saved, mode)
        entry["modes"][mode] = {"thresholds_from_calibration": thresholds, "threshold_diagnostics": details,
            "audit_global_0_2": counts(saved, mode, set(cfg["audit_cases"]), {}),
            "audit_calibrated": counts(saved, mode, set(cfg["audit_cases"]), thresholds),
            "calibration_global_0_2": counts(saved, mode, set(cfg["calibration_cases"]), {}),
            "calibration_calibrated": counts(saved, mode, set(cfg["calibration_cases"]), thresholds),
            "latency_ms_median": float(np.median([p["timings_ms"][mode] for p in saved["cases"]]))}
    return entry


def parent_baseline(cfg):
    path = HERE / "parent_baseline.json"
    if path.exists():
        baseline = read(path)
        assert baseline["protocol_sha256"] == sha(HERE / "protocol.json")
        assert baseline["entry"]["weights_sha256"] == cfg["initial_weights"]["11s_finetune"]["sha256"]
        return baseline["entry"]
    status(status="evaluating_parent")
    saved = collect(cfg, "11s_parent_best", Path(cfg["initial_weights"]["11s_finetune"]["path"]))
    entry = summarize(cfg, saved)
    reference = read(cfg["references"]["comparison"]["path"])["models"]["11s_seed20261012_best"]
    for endpoint in ("audit_global_0_2", "audit_calibrated"):
        actual = entry["modes"]["fused"][endpoint]["overall"]
        expected = reference["modes"]["fused"][endpoint]["overall"]
        assert all(actual[key] == expected[key] for key in ("tp", "fp", "fn")), "Parent YOLO11s counts do not reproduce."
    write(HERE / "parent_reproduction.json", {"exact_counts_reproduced": True,
        "actual": entry["modes"]["fused"]["audit_calibrated"]["overall"],
        "expected": reference["modes"]["fused"]["audit_calibrated"]["overall"]})
    write(path, {"protocol_sha256": sha(HERE / "protocol.json"), "entry": entry})
    return entry


def train(cfg):
    torch, _, YOLO = libraries()
    from train_detector import ClassNameTransferTrainer
    record_path = HERE / "training_record.json"
    if record_path.exists():
        record = read(record_path)
        assert record["status"] == "trained", "Diagnose an interrupted stage; never restart it silently."
        for checkpoint in ("best", "last"):
            assert sha(FOLDER / f"weights/{checkpoint}.pt") == record[checkpoint + "_sha256"]
        return
    assert not FOLDER.exists()
    initial = cfg["initial_weights"]["11s_finetune"]
    model = YOLO(initial["path"])
    assert model.names == {int(k): v for k, v in cfg["names"].items()}
    training = dict(cfg["training"])
    record = {"status": "training", "started_at_utc": utc(), "initial_weights": initial,
        "stage": "new warm-start optimization stage, not resume", "expected_training": training,
        "protocol_sha256": sha(HERE / "protocol.json"), "controller_sha256": sha(__file__)}
    write(record_path, record)
    started = time.perf_counter()

    def on_start(trainer):
        actual = vars(trainer.args)
        reference_path = Path(cfg["parent_protocol"]["path"]).parent / "effective_args.json"
        expected = dict(read(reference_path), **training)
        allowed = {"model", "data", "project", "name", "save_dir"}
        differences = {k: [expected.get(k), actual.get(k)] for k in set(expected) | set(actual)
            if k not in allowed and (str(expected.get(k)) != str(actual.get(k)) if k == "device" else expected.get(k) != actual.get(k))}
        assert not differences, differences
        assert sha(actual["data"]) == cfg["dataset_yaml_sha256"]
        assert trainer.start_epoch == 0 and trainer.epochs == 40 and trainer.batch_size == 4
        assert trainer.args.mosaic == 0 and trainer.args.lr0 == .00005
        transfer = read(FOLDER / "class_transfer.json")
        copied = transfer["copied_classification_rows"]
        assert len(copied) == 3 * len(cfg["names"])
        assert {r["class_name"] for r in copied} == set(cfg["names"].values())
        record.update(effective_arguments_verified=True, file_membership_verified=True,
            parameter_count=sum(p.numel() for p in trainer.model.parameters()), copied_classification_rows=len(copied))
        write(HERE / "effective_args.json", actual); write(record_path, record)
        status(status="training", run=RUN, epoch_completed=0, epochs_planned=40)

    def on_batch(trainer):
        assert trainer.batch_size == trainer.args.batch == 4, "Do not automatically shrink the batch."

    def on_epoch(trainer):
        state = {"status": "training", "run": RUN, "epoch_completed": trainer.epoch + 1, "epochs_planned": 40,
                 "seconds": time.perf_counter() - started, "metrics": {k: float(v) for k, v in trainer.metrics.items()}}
        status(**state); print(json.dumps(state), flush=True)

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_batch_start", on_batch)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.train(trainer=ClassNameTransferTrainer, data=cfg["dataset"], project=str(HERE / "runs"), name=RUN, exist_ok=False, **training)
    with (FOLDER / "results.csv").open(encoding="utf-8") as stream:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    assert [int(float(row["epoch"])) for row in rows] == list(range(1, len(rows) + 1)) and len(rows) <= 40
    peak = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    record.update(status="trained", completed_epochs=len(rows), best_calibration_epoch=int(float(peak["epoch"])),
        best_calibration_mAP50_95=float(peak["metrics/mAP50-95(B)"]), seconds=time.perf_counter() - started,
        finished_at_utc=utc(), best_sha256=sha(FOLDER / "weights/best.pt"), last_sha256=sha(FOLDER / "weights/last.pt"))
    write(record_path, record)
    del model; torch.cuda.empty_cache()


def candidate_config(cfg, entry):
    destination = PROJECT / "weights/detector_yolo11s_finetune_20261015.pt"
    assert not destination.exists() or sha(destination) == entry["weights_sha256"]
    shutil.copy2(entry["weights"], destination)
    config = read(PROJECT / "configs/baseline_y11s.json")
    config.update(version="0.5-yolo11s-finetune-candidate", weights="weights/" + destination.name,
        weights_sha256=sha(destination), class_thresholds_file="configs/class_thresholds_y11s_finetuned_candidate.json",
        status="evaluated_finetune_candidate", recommended_mode_basis="Single planned warm-start trial; separate candidate config, no automatic deployment adoption.")
    thresholds = entry["modes"]["fused"]["thresholds_from_calibration"]
    config["detector"]["class_confidence"] = thresholds
    config["detector_provenance"] = {"experiment": str(HERE), "checkpoint": "best.pt", "selection": "calibration mAP50-95"}
    path = PROJECT / "configs/baseline_y11s_finetuned_candidate.json"
    if path.exists():
        assert read(path)["detector_provenance"]["experiment"] == str(HERE)
    write(path, config)
    write(PROJECT / "configs/class_thresholds_y11s_finetuned_candidate.json", {"thresholds": thresholds,
        "calibration_cases": cfg["calibration_cases"], "weights_sha256": sha(destination)})
    return str(path)


def evaluate(cfg, baseline):
    torch, _, YOLO = libraries()
    previous = read(cfg["references"]["comparison"]["path"])
    result = {"status": "completed", "protocol_sha256": sha(HERE / "protocol.json"), "training": read(HERE / "training_record.json"),
        "recipe_changes": cfg["fine_tune_changes"], "models": {"11s_parent_best": baseline},
        "audit_ap": {"11s_parent_best": previous["audit_ap"]["11s_seed20261012_best"]}, "limits": cfg["limitations"]}
    for checkpoint in ("best", "last"):
        name = RUN + "_" + checkpoint; weights = FOLDER / f"weights/{checkpoint}.pt"
        result["models"][name] = summarize(cfg, collect(cfg, name, weights))
        status(status="evaluating_ap", model=name)
        model = YOLO(str(weights))
        ap = model.val(data=str(HERE / "dataset/audit_dataset.yaml"), imgsz=1536, batch=4, workers=2, device=0, half=False,
            conf=.001, iou=.7, max_det=1000, plots=False, save_json=False, project=str(HERE / "audit_ap"), name=name, exist_ok=True, verbose=False)
        result["audit_ap"][name] = {"mAP50": float(ap.box.map50), "mAP50_95": float(ap.box.map)}
        del model; torch.cuda.empty_cache()
    before = baseline["modes"]["fused"]
    candidate = result["models"][RUN + "_best"]["modes"]["fused"]
    result["paired_differences_candidate_minus_parent"] = {endpoint: {
        "delta_f1": candidate[endpoint]["overall"]["f1"] - before[endpoint]["overall"]["f1"],
        "bootstrap": bootstrap(before[endpoint], candidate[endpoint], cfg["case_groups"])}
        for endpoint in ("audit_global_0_2", "audit_calibrated")}
    parent_cal = before["calibration_calibrated"]
    candidate_cal = candidate["calibration_calibrated"]
    gain = candidate_cal["overall"]["f1"] - parent_cal["overall"]["f1"]
    capacitor_delta = candidate_cal["by_class"]["capacitor"]["f1"] - parent_cal["by_class"]["capacitor"]["f1"]
    map_delta = result["training"]["best_calibration_mAP50_95"] - cfg["parent_calibration_mAP50_95"]
    gate = cfg["recommendation_gate"]
    result["calibration_recommendation_gate"] = {"passed": gain >= gate["minimum_calibrated_fused_f1_gain"] and
        capacitor_delta >= -gate["maximum_capacitor_f1_drop"] and map_delta >= -gate["maximum_calibration_map_drop"],
        "fused_f1_gain": gain, "capacitor_f1_delta": capacitor_delta, "mAP50_95_delta": map_delta,
        "policy": gate, "audit_not_used_for_gate": True}
    result["candidate_config"] = candidate_config(cfg, result["models"][RUN + "_best"])
    verify_recipe(cfg)
    result.update(frozen_inputs_unchanged_after_trial=True, completed_at_utc=utc())
    write(HERE / "comparison.json", result)
    lines = ["# YOLO11s精修对照结果", "", "从原11s第76轮best开始，lr0=0.00005、mosaic=0、最多40轮、patience12、warmup1。1536、batch4、418张训练、30页校准和29页评分及推理/阈值政策不变。", "",
        f"精修实际{result['training']['completed_epochs']}轮，校准集最佳在精修第{result['training']['best_calibration_epoch']}轮。best为主结果，last只作诊断。", "",
        "| 模型 | 固定conf0.2融合F1 | 校准后P | 校准后R | 校准后融合F1 | 整图mAP50-95 | 融合中位ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in result["models"].items():
        mode = entry["modes"]["fused"]; metric = mode["audit_calibrated"]["overall"]
        lines.append(f"| {name} | {mode['audit_global_0_2']['overall']['f1']:.4%} | {metric['precision']:.4%} | {metric['recall']:.4%} | {metric['f1']:.4%} | {result['audit_ap'][name]['mAP50_95']:.4%} | {mode['latency_ms_median']:.1f} |")
    lines += ["", "AP为整图conf0.001、NMS IoU0.7；F1为整图+切片+WBF后类别与框IoU>=0.5匹配，口径不同。延迟受当时GPU负载影响。", "", "## 精修best减父模型", ""]
    for endpoint, value in result["paired_differences_candidate_minus_parent"].items():
        lo, hi = value["bootstrap"]["ci95_delta_f1"]
        lines.append(f"- {endpoint}：{value['delta_f1']*100:+.3f}个百分点，条件分组bootstrap95%区间[{lo*100:+.3f}, {hi*100:+.3f}]。")
    recommendation = result["calibration_recommendation_gate"]
    lines += ["", "## 仅用校准集的预设推荐门槛", "",
        f"门槛是否通过：{recommendation['passed']}；融合F1变化{gain*100:+.3f}个百分点，电容F1变化{capacitor_delta*100:+.3f}个百分点，mAP50-95变化{map_delta*100:+.3f}个百分点。要求分别为至少+0.2个百分点、最多下降1个百分点、最多下降0.5个百分点。评分集不用于该门槛。", "",
        "## 逐类F1", "", "| 类别 | GT | 父11s | 精修best | 变化（百分点） |", "|---|---:|---:|---:|---:|"]
    from evaluate_detector import metric
    empty = metric({"tp": 0, "fp": 0, "fn": 0})
    for cls, row in sorted(candidate["audit_calibrated"]["by_class"].items(), key=lambda item: -(item[1]["tp"] + item[1]["fn"])):
        old = before["audit_calibrated"]["by_class"].get(cls, empty)
        lines.append(f"| {cls} | {row['tp']+row['fn']} | {old['f1']:.4%} | {row['f1']:.4%} | {(row['f1']-old['f1'])*100:+.3f} |")
    lines += ["", "## 结论边界", "", "单次精修试验、一个种子，不能归因于某一超参数；评分页是已查看的旧开发集，没有独立盲测保证。未使用评分集选择best/last或改阈值政策。这是类别和框的指标，不是官方引脚或网络得分。", "",
        f"独立候选配置：{result['candidate_config']}。父11s与原v8n配置、权重和旧下游缓存均保留，本次不自动切换部署。"]
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(status="completed", comparison=str(HERE / "comparison.md"), calibration_gate_passed=recommendation["passed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--launch-if-idle", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(); cfg = read(HERE / "protocol.json")
    if args.verify:
        verify_recipe(cfg)
        print(json.dumps({"frozen_inputs_verified": True, "planned_recipe_verified": True, "status": read(HERE / "status.json")["status"]})); return
    if args.launch_if_idle:
        runtime.launch_if_idle(cfg); return
    assert args.run
    guard = runtime.lock("run.lock")
    if guard is None:
        print("An experiment process is already active."); return
    try:
        status(status="verifying")
        verify_recipe(cfg); libraries(); metric_self_check()
        baseline = parent_baseline(cfg)
        train(cfg); verify_recipe(cfg); evaluate(cfg, baseline)
    except BaseException as exc:
        status(status="failed", error=repr(exc)); raise
    finally:
        guard.close()


if __name__ == "__main__":
    main()
