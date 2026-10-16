"""Paired COCO-initialized YOLO11n/YOLOv8n experiment; never changes deployment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "pcb_parser_competition"
os.environ["YOLO_CONFIG_DIR"] = str(HERE / "ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
CONFIG = HERE / "protocol.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def status(**values):
    write(HERE / "status.json", {"updated_at_utc": utc(), "pid": os.getpid(), **values})


def libraries():
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    assert torch.__version__ == "2.10.0+cu128"
    assert ultralytics.__version__ == "8.3.253"
    assert torch.cuda.is_available()
    return torch, ultralytics, YOLO


def prepare():
    import numpy as np
    import yaml
    torch, ultralytics, YOLO = libraries()
    assert not CONFIG.exists(), "Protocol already frozen; use existing protocol."
    manifest = ROOT / "data/detection_balanced_v2/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    val = sorted([r for r in rows if r["split"] == "val"], key=lambda r: r["case_id"])
    assert len(train) == 418 and len(val) == 59
    assert not {r["group"] for r in train} & {r["group"] for r in val}
    assert not {r["image_sha256"] for r in train} & {r["image_sha256"] for r in val}
    names = yaml.safe_load((manifest.parent / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}
    assert len(names) == 35
    # Choose the split using labels only, before either new model is trained.
    # Keep existing grouping; a near-duplicate group cannot cross this split.
    groups = sorted({r["group"] for r in val})
    matrix = np.array([[sum(o["class_name"] == names[c] for o in r["objects"])
                        for c in range(len(names))] for r in val], dtype=float)
    support = (matrix > 0).sum(axis=0)
    total = matrix.sum(axis=0)
    rng = np.random.default_rng(20261015)
    best = None
    for _ in range(5000):
        selected_groups = {groups[int(i)] for i in rng.choice(len(groups), int(rng.integers(28, 31)), replace=False)}
        idx = np.array([i for i,r in enumerate(val) if r["group"] in selected_groups], dtype=int)
        if len(idx) != 30:
            continue
        counts = matrix[idx].sum(axis=0)
        score = float(np.square((counts / np.maximum(total, 1)) - 30/59).sum())
        score += 10 * int(((support >= 2) & ((counts == 0) | (counts == total))).sum())
        if best is None or score < best[0]:
            best = (score, idx)
    calibration = {val[int(i)]["case_id"] for i in best[1]}
    audit = {r["case_id"] for r in val} - calibration
    assert not {r["group"] for r in val if r["case_id"] in calibration} & {r["group"] for r in val if r["case_id"] in audit}
    copied = []
    original_inputs = []
    for row in rows:
        r = copy.deepcopy(row)
        folder = "train" if row["split"] == "train" else "val"
        for kind, ext, dirname in [("image", ".png", "images"), ("label", ".txt", "labels")]:
            origin = Path(row[kind])
            assert sha(origin) == row[kind + "_sha256"], f"Changed source input: {origin}"
            destination = HERE / "dataset" / dirname / folder / (row["case_id"] + ext)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            assert sha(destination) == row[kind + "_sha256"]
            original_inputs.append({"path": str(origin), "sha256": row[kind + "_sha256"]})
            r[kind] = str(destination)
        copied.append(r)
    dataset_dir = HERE / "dataset"
    for label, selected in [("train", train), ("calibration", [r for r in val if r["case_id"] in calibration]),
                            ("audit", [r for r in val if r["case_id"] in audit])]:
        paths = [next(r["image"] for r in copied if r["case_id"] == x["case_id"]) for x in selected]
        (dataset_dir / f"{label}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    dataset = {"path": str(dataset_dir), "train": "train.txt", "val": "calibration.txt", "names": names}
    (dataset_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    (dataset_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in copied)+"\n", encoding="utf-8")
    initial = HERE / "initial_weights"
    shutil.copy2(HERE.parent / "y11s_match_v8n_coco_20261012/yolo11n.pt", initial / "yolo11n.pt")
    weight_info = {}
    for name, file in [("v8n", "yolov8n.pt"), ("11n", "yolo11n.pt")]:
        path = initial / file
        model = YOLO(str(path))
        assert len(model.names) == 80, "Expected COCO initialization"
        assert not set(model.names.values()) & set(names.values()), "Unexpected class-name transfer"
        weight_info[name] = {"path": str(path), "sha256": sha(path), "classes": model.names,
                             "source_url": f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{file}"}
        del model
    record_path = ROOT / "data/local_imports/v3_20261011/source_evidence/runs/training/detector_dataset300_v3_nano_record.json"
    training = read(record_path)["effective_training"]
    reference_path = ROOT / "data/local_imports/v3_20261011/source_evidence/runs/eval_v3_fused_v2thr.json"
    reference = read(reference_path)
    inference = read(ROOT / "configs/baseline_v3_adopted.json")
    assert inference["detector"]["confidence"] == .2
    baseline = ROOT / "weights/detector_dataset300_v3_nano.pt"
    assert sha(baseline) == reference["weights_sha256"]
    code_paths = [ROOT/"scripts/run_baseline.py", ROOT/"scripts/evaluate_detector.py",
                  ROOT/"src/pcb_parser/fusion.py", ROOT/"scripts/train_detector.py"]
    cfg = {"version": 1, "created_at_utc": utc(), "seeds": [20261012, 20261015],
           "training": training, "initial_weights": weight_info, "dataset": str(dataset_dir/"dataset.yaml"),
           "source_manifest_sha256": sha(manifest), "dataset_yaml_sha256": sha(dataset_dir/"dataset.yaml"),
           "copied_manifest_sha256": sha(dataset_dir/"manifest.jsonl"), "names": names,
           "calibration_cases": sorted(calibration), "audit_cases": sorted(audit),
           "case_groups": {r["case_id"]: r["group"] for r in val},
           "image_counts": {"train": 418, "calibration": 30, "audit": 29, "old_validation_total": 59},
           "split_selection": "5000 deterministic label-stratified candidate splits; no model outputs used; seed 20261015",
           "split_class_counts": {"all": total.astype(int).tolist(), "calibration": matrix[best[1]].sum(0).astype(int).tolist()},
           "inference": inference, "baseline": {"path": str(baseline), "sha256": sha(baseline),
              "reference": str(reference_path), "reference_sha256": sha(reference_path)},
           "original_inputs": original_inputs, "code_inputs": [{"path": str(p), "sha256": sha(p)} for p in code_paths],
           "runtime": {"python": sys.executable, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
           "threshold_policy": {"minimum_calibration_gt": 5, "fallback": .2,
                                "candidates": "all unique calibration prediction scores >=0.2, plus 0.2/0.25/1.000001; F1 ties choose lower"},
           "primary_endpoint": "fused micro F1 on 29 audit pages, thresholds fitted only on 30 calibration pages; also fixed global 0.2",
           "comparability": "Two paired seeds; both new models COCO-initialized and identical final-stage task recipe. Adopted v8n is separate deployment reference with historical task warm starts and historical use of all 59 validation pages.",
           "limitations": ["Old Dataset300 pages; not independent new-engineering-source test", "Same COCO dataset but architecture-specific pretrained weights; not identical pretrained tensors", "Only final-stage recipe tested; not full historical three-stage retraining", "Two seeds describe limited training variation", "Labels partly reviewed; scores are bbox/category proxies, not official ComponentF1", "Historical adopted v8n used all 59 pages for model and threshold selection"],
           "run_order": ["11n_seed20261012", "v8n_seed20261012", "v8n_seed20261015", "11n_seed20261015"]}
    write(CONFIG, cfg)
    write(HERE / "input_audit.json", {"verified_file_count": len(original_inputs), "exact_hash_train_validation_overlap": 0,
          "recorded_group_train_validation_overlap": 0, "full_engineering_grouping_verified": False})
    text = "# YOLO11n / YOLOv8n 配对实验\n\n两个新模型从各自官方COCO权重开始，复用v3末阶段配方。两种随机种子均成对训练；当前部署v8n作为另一个实际替换参照。\n\n418张训练样本及35类标签保持原内容。原59张验证页预先固定为30张校准页和29张评分页；新训练只在校准页选择checkpoint，逐类阈值也只使用校准页。评分页不用于选模型或调参。划分仅根据标注计数平衡，未用新模型结果。\n\n主指标为29页fused类别+bbox微平均F1，同时报告全局conf0.2、whole/sliced/fused、best/last、逐类与逐页结果、速度和种子差异。低于5个校准GT的类别不调阈值，保留0.2。每次训练最多120轮，早停25；1536、batch4、AMP、AdamW3e-4与v3一致。\n\n这批页属于旧Dataset300开发数据；来源分组不完整，不称全新独立工程测试或官方得分。部署v8n曾使用全部59页选模型/阈值，其结果只是历史部署参照。只验证固定v3末阶段配方下的效果，不能概括所有架构最优能力。\n\n实验文件完全独立，原始图像/标签、配置、部署权重保持。protocol.json为冻结条件，comparison.md为完成后的结果。\n"
    (HERE / "README.md").write_text(text, encoding="utf-8")
    status(status="prepared", counts=cfg["image_counts"])
    print(json.dumps({"prepared": cfg["image_counts"], "calibration": sorted(calibration), "audit": sorted(audit)}, ensure_ascii=False), flush=True)


def verify_inputs(cfg):
    assert sha(cfg["dataset"]) == cfg["dataset_yaml_sha256"]
    assert sha(HERE / "dataset/manifest.jsonl") == cfg["copied_manifest_sha256"]
    rows = [json.loads(line) for line in (HERE/"dataset/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        for k in ("image", "label"):
            assert sha(r[k]) == r[k+"_sha256"], f"Changed experiment input: {r[k]}"
    for entry in cfg["code_inputs"] + cfg["original_inputs"]:
        assert sha(entry["path"]) == entry["sha256"], f"Changed frozen source: {entry['path']}"
    assert sha(cfg["baseline"]["path"]) == cfg["baseline"]["sha256"]
    for entry in cfg["initial_weights"].values():
        assert sha(entry["path"]) == entry["sha256"]
    return rows


def train_run(cfg, name):
    torch, _, YOLO = libraries()
    verify_inputs(cfg)
    from train_detector import ClassNameTransferTrainer
    architecture, seed_text = name.split("_seed")
    seed = int(seed_text)
    folder = HERE / "runs" / name
    record_path = HERE / "records" / (name + ".json")
    if record_path.exists() and read(record_path).get("status") == "trained":
        record = read(record_path)
        for ck in ("best", "last"):
            assert sha(folder/f"weights/{ck}.pt") == record[ck+"_sha256"]
        return
    resume = record_path.exists() and (folder/"weights/last.pt").exists()
    if not resume:
        assert not folder.exists(), "Incomplete run without checkpoint requires explicit diagnosis."
    training = dict(cfg["training"], seed=seed)
    initial = cfg["initial_weights"][architecture]
    model = YOLO(str(folder/"weights/last.pt")) if resume else YOLO(initial["path"])
    started = time.perf_counter()
    record = {"status": "training", "name": name, "initial_weights": initial, "started_at_utc": utc(),
              "expected_training": training, "resumed": resume}
    write(record_path, record)
    def on_start(trainer):
        actual = vars(trainer.args)
        mismatches = {k: [v, actual.get(k)] for k, v in training.items()
                      if (str(v) != str(actual.get(k)) if k == "device" else v != actual.get(k))}
        write(HERE/"records"/(name+"_args.json"), actual)
        assert not mismatches, f"Changed effective training: {mismatches}"
        assert Path(actual["data"]).resolve() == Path(cfg["dataset"]).resolve()
        record.update(parameter_count=sum(p.numel() for p in trainer.model.parameters()), effective_arguments_verified=True)
        write(record_path, record)
        status(status="training", run=name, epoch_completed=trainer.start_epoch, epochs_planned=training["epochs"])
    def on_epoch(trainer):
        m = {k: float(v) for k, v in trainer.metrics.items()}
        state = {"status": "training", "run": name, "epoch_completed": trainer.epoch+1, "epochs_planned": trainer.epochs,
                 "seconds_this_process": time.perf_counter()-started, "metrics": m,
                 "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20}
        status(**state)
        print(json.dumps(state), flush=True)
    model.add_callback("on_train_start", on_start)
    model.add_callback("on_fit_epoch_end", on_epoch)
    if resume:
        model.train(resume=True, trainer=ClassNameTransferTrainer)
    else:
        model.train(trainer=ClassNameTransferTrainer, data=cfg["dataset"], project=str(HERE/"runs"),
                    name=name, exist_ok=False, **training)
    record.update(status="trained", finished_at_utc=utc(), seconds_this_process=time.perf_counter()-started,
                  best_sha256=sha(folder/"weights/best.pt"), last_sha256=sha(folder/"weights/last.pt"),
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write(record_path, record)
    del model
    torch.cuda.empty_cache()
    verify_inputs(cfg)


def collect(cfg, name, weights):
    cache = HERE / "evaluation" / (name + "_predictions.json")
    if cache.exists():
        saved = read(cache)
        assert saved["weights_sha256"] == sha(weights)
        assert saved["protocol_sha256"] == sha(CONFIG)
        return saved
    import numpy as np
    from PIL import Image
    from sahi import AutoDetectionModel
    from run_baseline import whole_predictions, sliced_predictions
    from pcb_parser.fusion import wbf_merge
    torch, _, YOLO = libraries()
    rows = sorted([r for r in verify_inputs(cfg) if r["split"] == "val"], key=lambda r: r["case_id"])
    inference = cfg["inference"]
    detector = inference["detector"]
    whole = YOLO(str(weights))
    sliced = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
             confidence_threshold=.2, device="cuda:0", image_size=1536)
    sliced.model.overrides.update({"iou": .5, "max_det": 1000, "half": False})
    warm = np.asarray(Image.open(rows[0]["image"]).convert("RGB"))
    whole_predictions(whole, np.ascontiguousarray(warm[:,:,::-1]), detector, "cuda:0")
    sliced_predictions(sliced, warm, inference["sahi"])
    torch.cuda.synchronize()
    saved = {"model": name, "weights": str(weights), "weights_sha256": sha(weights),
             "protocol_sha256": sha(CONFIG), "cases": [], "timing_scope": "warm inference, preprocessing, postprocessing; no file IO/model load"}
    for row in rows:
        status(status="evaluating", model=name, evaluated=len(saved["cases"]), total=len(rows))
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        timings = {}
        predictions = {}
        for mode in ("whole", "sliced", "fused"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if mode == "whole":
                output = whole_predictions(whole, np.ascontiguousarray(rgb[:,:,::-1]), detector, "cuda:0")
            elif mode == "sliced":
                output = sliced_predictions(sliced, rgb, inference["sahi"])
            else:
                output = wbf_merge(predictions["whole"]+predictions["sliced"], inference["fusion"]["iou"])
            torch.cuda.synchronize()
            timings[mode] = (time.perf_counter()-start)*1000
            predictions[mode] = output
        timings["fused"] += timings["whole"]+timings["sliced"]
        saved["cases"].append({"case_id": row["case_id"], "truth": row["objects"], "predictions": predictions, "timings_ms": timings})
    write(cache, saved)
    del whole, sliced
    torch.cuda.empty_cache()
    return saved


def counts(saved, mode, ids, thresholds):
    from evaluate_detector import counts_for_case, metric
    from pcb_parser.fusion import apply_class_confidence
    total = Counter(tp=0, fp=0, fn=0)
    by_class = {}
    pages = []
    for page in saved["cases"]:
        if page["case_id"] not in ids:
            continue
        found = counts_for_case(apply_class_confidence(page["predictions"][mode], thresholds), page["truth"], .5)
        subtotal = Counter(tp=0, fp=0, fn=0)
        for cls, values in found.items():
            subtotal.update(values)
            by_class.setdefault(cls, Counter(tp=0, fp=0, fn=0)).update(values)
        total.update(subtotal)
        pages.append({"case_id": page["case_id"], "counts": metric(subtotal)})
    return {"overall": metric(total), "by_class": {c: metric(v) for c,v in by_class.items()}, "cases": pages}


def prefix_counts(predictions, truth, name):
    """Maximum-cardinality bipartite matching for every confidence prefix."""
    import numpy as np
    from evaluate_detector import bbox_iou
    p = sorted([x for x in predictions if x["class_name"] == name], key=lambda x: -x["confidence"])
    t = [x for x in truth if x["class_name"] == name]
    adjacency = [[j for j,y in enumerate(t) if bbox_iou(x["bbox"], y["bbox"]) >= .5] for x in p]
    ownership = {}
    matches = [0]
    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in ownership or augment(ownership[j], seen):
                ownership[j] = i
                return True
        return False
    for i in range(len(p)):
        augment(i, set())
        matches.append(len(ownership))
    return np.array([-float(x["confidence"]) for x in p]), np.array(matches), len(t)


def tune(cfg, saved, mode):
    import numpy as np
    from evaluate_detector import metric
    pages = [p for p in saved["cases"] if p["case_id"] in cfg["calibration_cases"]]
    thresholds = {}
    details = {}
    for name in cfg["names"].values():
        parts = [prefix_counts(p["predictions"][mode], p["truth"], name) for p in pages]
        gt = sum(x[2] for x in parts)
        if gt < cfg["threshold_policy"]["minimum_calibration_gt"]:
            thresholds[name] = .2
            details[name] = {"calibration_gt": gt, "threshold": .2, "reason": "insufficient_calibration_support"}
            continue
        scores = {float(-s) for values, _, _ in parts for s in values if -s >= .2}
        winner = (-1., .2, None)
        for threshold in sorted(scores | {.2, .25, 1.000001}):
            kept = [int(np.searchsorted(scores_, -threshold, side="right")) for scores_, _, _ in parts]
            tp = sum(int(x[1][n]) for x,n in zip(parts, kept))
            fp = sum(kept) - tp
            m = metric({"tp": tp, "fp": fp, "fn": gt-tp})
            if m["f1"] > winner[0] + 1e-12:
                winner = (m["f1"], threshold, m)
        thresholds[name] = winner[1]
        details[name] = {"calibration_gt": gt, "threshold": winner[1], "calibration_counts": winner[2]}
    return thresholds, details


def baseline_check(cfg):
    saved = collect(cfg, "v8n_adopted", Path(cfg["baseline"]["path"]))
    reference = read(cfg["baseline"]["reference"])
    assert {p["case_id"] for p in saved["cases"]} == {p["case_id"] for p in reference["cases"]}
    m = counts(saved, "fused", {p["case_id"] for p in saved["cases"]}, reference["class_thresholds"])
    success = all(m["overall"][k] == reference["overall"][k] for k in ("tp", "fp", "fn"))
    write(HERE/"baseline_reproduction.json", {"exact_counts_reproduced": success, "actual": m["overall"], "expected": reference["overall"]})
    assert success, "Deployed baseline counts do not reproduce; diagnose before starting training."
    print(json.dumps({"baseline_reproduced": m["overall"]}), flush=True)


def bootstrap(a, b, case_groups, seed=20261015):
    import numpy as np
    x = {p["case_id"]: p["counts"] for p in a["cases"]}
    y = {p["case_id"]: p["counts"] for p in b["cases"]}
    assert set(x) == set(y)
    grouped_a = {}; grouped_b = {}
    for i in x:
        group = case_groups[i]
        grouped_a.setdefault(group, Counter()).update({k:x[i][k] for k in ("tp","fp","fn")})
        grouped_b.setdefault(group, Counter()).update({k:y[i][k] for k in ("tp","fp","fn")})
    ids = sorted(grouped_a)
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(20000,len(ids)))
    av = np.array([[grouped_a[i][k] for k in ("tp","fp","fn")] for i in ids])[indices].sum(1)
    bv = np.array([[grouped_b[i][k] for k in ("tp","fp","fn")] for i in ids])[indices].sum(1)
    delta = 2*bv[:,0]/np.maximum(2*bv[:,0]+bv[:,1]+bv[:,2],1)-2*av[:,0]/np.maximum(2*av[:,0]+av[:,1]+av[:,2],1)
    return {"ci95_delta_f1": np.quantile(delta,[.025,.975]).tolist(), "resamples": 20000,
            "unit": "paired recorded source/near-duplicate group", "groups": len(ids), "caveat": "Conditional on these checkpoints/calibration thresholds; not a new-source guarantee or training-seed CI"}


def summarize(cfg):
    import numpy as np
    models = {"v8n_adopted": Path(cfg["baseline"]["path"])}
    for name in cfg["run_order"]:
        for checkpoint in ("best", "last"):
            models[name+"_"+checkpoint] = HERE/"runs"/name/f"weights/{checkpoint}.pt"
    result = {"status": "completed", "protocol_sha256": sha(CONFIG), "primary_endpoint": cfg["primary_endpoint"],
              "limitations": cfg["limitations"], "models": {}, "paired_comparisons": {}}
    for name, weights in models.items():
        saved = collect(cfg, name, weights)
        entry = {"weights": str(weights), "weights_sha256": sha(weights), "modes": {}}
        for mode in ("whole", "sliced", "fused"):
            th, detail = tune(cfg, saved, mode)
            entry["modes"][mode] = {"thresholds_from_calibration": th, "threshold_diagnostics": detail,
                  "audit_global_0_2": counts(saved, mode, set(cfg["audit_cases"]), {}),
                  "audit_calibrated": counts(saved, mode, set(cfg["audit_cases"]), th),
                  "calibration_global_0_2": counts(saved, mode, set(cfg["calibration_cases"]), {}),
                  "calibration_calibrated": counts(saved, mode, set(cfg["calibration_cases"]), th),
                  "latency_ms_median": float(np.median([p["timings_ms"][mode] for p in saved["cases"]]))}
        if name == "v8n_adopted":
            entry["historical_deployment_thresholds_audit"] = counts(saved,"fused",set(cfg["audit_cases"]),read(cfg["baseline"]["reference"])["class_thresholds"])
        result["models"][name] = entry
        write(HERE/"comparison.json", result)
    for seed in cfg["seeds"]:
        a = result["models"][f"v8n_seed{seed}_best"]["modes"]["fused"]
        b = result["models"][f"11n_seed{seed}_best"]["modes"]["fused"]
        result["paired_comparisons"][str(seed)] = {endpoint: {
             "delta_f1_11n_minus_v8n": b[endpoint]["overall"]["f1"]-a[endpoint]["overall"]["f1"],
             "bootstrap": bootstrap(a[endpoint],b[endpoint],cfg["case_groups"])}
             for endpoint in ("audit_global_0_2","audit_calibrated")}
    result["mean_paired_delta_calibrated"] = float(np.mean([p["audit_calibrated"]["delta_f1_11n_minus_v8n"] for p in result["paired_comparisons"].values()]))
    result["completed_at_utc"] = utc()
    verify_inputs(cfg)
    result["frozen_inputs_unchanged_after_experiment"] = True
    write(HERE/"comparison.json",result)
    lines = ["# YOLO11n与YOLOv8n配对实验结果", "", "主结果：29页评分集，阈值只由30页校准集选定。两个新模型均从COCO初始化。当前部署v8n具有历史任务微调和旧验证页开发优势，单独列为实际参照。所有指标为本地类别+bbox F1。", "",
             "| 模型 | 评分集固定conf0.2 F1 | 评分集校准阈值F1 | 校准后P | 校准后R | fused中位ms |", "|---|---:|---:|---:|---:|---:|"]
    for name,entry in result["models"].items():
        m=entry["modes"]["fused"]
        g=m["audit_global_0_2"]["overall"]; c=m["audit_calibrated"]["overall"]
        lines.append(f"| {name} | {g['f1']:.4%} | {c['f1']:.4%} | {c['precision']:.4%} | {c['recall']:.4%} | {m['latency_ms_median']:.1f} |")
    lines += ["", "## 配对差异（11n减v8n，best）", ""]
    for seed,comparison in result["paired_comparisons"].items():
        c=comparison["audit_calibrated"]
        lo,hi=c["bootstrap"]["ci95_delta_f1"]
        lines.append(f"- seed {seed}：{c['delta_f1_11n_minus_v8n']*100:+.3f}个百分点；条件分组bootstrap95%区间[{lo*100:+.3f}, {hi*100:+.3f}]。")
    lines += ["", f"两个种子的平均校准后F1差：{result['mean_paired_delta_calibrated']*100:+.3f}个百分点。", "", "## 限制", ""]
    lines += ["- "+x for x in cfg["limitations"]]
    lines += ["", "训练参数/数据/评估器的冻结校验已通过。原部署权重和配置未替换。逐类/逐页结果、whole/sliced、best/last、阈值及原始预测见comparison.json和evaluation目录。"]
    (HERE/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status(status="completed", comparison=str(HERE/"comparison.md"), mean_paired_delta_calibrated=result["mean_paired_delta_calibrated"])
    print(json.dumps({"completed": True,"mean_paired_delta_calibrated":result["mean_paired_delta_calibrated"]}),flush=True)


def metric_self_check():
    import numpy as np
    from evaluate_detector import counts_for_case
    rng = np.random.default_rng(14)
    for _ in range(100):
        p=[]; t=[]
        for target,count in [(p,12),(t,8)]:
            for _ in range(count):
                x,y=rng.integers(0,8,size=2)
                target.append({"class_name":"ic","bbox":[int(x),int(y),int(x)+6,int(y)+6],"confidence":float(rng.choice([.2,.4,.6,.8]))})
        values,matches,gt=prefix_counts(p,t,"ic")
        for threshold in (.2,.4,.6,.8,1.000001):
            n=int(np.searchsorted(values,-threshold,side="right"))
            expected=counts_for_case([x for x in p if x["confidence"]>=threshold],t,.5)["ic"]
            assert expected == {"tp":int(matches[n]),"fp":n-int(matches[n]),"fn":gt-int(matches[n])}
    print("Threshold matching self-check passed: 500 comparisons against existing Hungarian evaluator.",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--baseline-only",action="store_true")
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check:
        metric_self_check()
        return
    if args.prepare:
        prepare()
        return
    cfg=read(CONFIG)
    libraries()
    metric_self_check()
    baseline_check(cfg)
    if args.baseline_only:
        status(status="baseline_verified")
        return
    if not args.evaluate_only:
        for name in cfg["run_order"]:
            train_run(cfg,name)
    summarize(cfg)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        status(status="failed",error=repr(exc))
        raise
