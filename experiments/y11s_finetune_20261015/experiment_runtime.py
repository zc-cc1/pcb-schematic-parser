"""Run the authorized YOLO11s trial only after the GPU is idle."""
from pathlib import Path
import argparse
import copy
import csv
import ctypes
import json
import msvcrt
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
from matched_helpers import read, write, sha, utc, libraries, verify_inputs, collect, tune, counts, bootstrap, baseline_check, metric_self_check

RUN = "11s_seed20261012"
FOLDER = HERE / "runs" / RUN


def status(**values):
    write(HERE / "status.json", {"updated_at_utc": utc(), "pid": os.getpid(), **values})


def lock(name):
    stream = (HERE / name).open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0"); stream.flush()
    stream.seek(0)
    try:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        stream.close()
        return None
    return stream


def process_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5  # Treat inaccessible live processes as busy.
    code = ctypes.c_uint32()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return not ok or code.value == 259


def gpu_sample():
    flags = subprocess.CREATE_NO_WINDOW
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"], text=True, creationflags=flags)
    row = next(csv.reader([gpu.splitlines()[0]]))
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], creationflags=flags).decode("utf-8", errors="replace")
    workloads = []
    for app in csv.reader(apps.splitlines()):
        if len(app) < 2:
            continue
        executable = Path(app[1].strip()).name.lower()
        if any(word in executable for word in ("python", "train", "blender", "ollama", "llama", "ffmpeg", "halcon")):
            workloads.append({"pid": app[0].strip(), "process": app[1].strip()})
    return {"utilization_percent": int(row[1]), "free_mib": int(row[2]), "compute_workloads": workloads}


def launch_if_idle(cfg):
    launch_lock = lock("launch.lock")
    if launch_lock is None:
        print("Another idle check is active."); return
    try:
        state = read(HERE / "status.json")
        if state["status"] in ("completed", "failed"):
            print(json.dumps(state)); return
        if state["status"] != "waiting_for_gpu" and state.get("pid") and process_alive(state["pid"]):
            print(json.dumps(state)); return
        if (HERE / "training_record.json").exists() or (FOLDER / "weights/last.pt").exists():
            status(status="failed", error="Previous training process ended; inspect its saved record/checkpoint before any resume. No automatic restart.")
            print(json.dumps(read(HERE / "status.json"))); return
        policy = cfg["idle_policy"]
        samples = []
        for index in range(policy["consecutive_samples"]):
            sample = gpu_sample(); samples.append(sample)
            idle = sample["utilization_percent"] <= policy["max_utilization_percent"] and sample["free_mib"] >= policy["min_free_mib"] and not sample["compute_workloads"]
            if not idle:
                status(status="waiting_for_gpu", gpu_samples=samples)
                print(json.dumps(read(HERE / "status.json"))); return
            if index + 1 < policy["consecutive_samples"]:
                time.sleep(policy["sample_seconds"])
        # Reserve the starting state before launching; the child uses a second lock.
        status(status="starting", gpu_samples=samples)
        python = cfg["runtime"]["python"]
        with (HERE / "stdout.log").open("ab") as out, (HERE / "stderr.log").open("ab") as err:
            process = subprocess.Popen([python, "-u", "-X", "utf8", str(Path(__file__).resolve()), "--run"], cwd=str(HERE), stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        write(HERE / "process.json", {"launcher_pid": process.pid, "launched_at_utc": utc(), "python": python, "gpu_samples_before_launch": samples})
        print(json.dumps({"launched": True, "launcher_pid": process.pid}))
    finally:
        launch_lock.close()


def train(cfg):
    torch, _, YOLO = libraries()
    from train_detector import ClassNameTransferTrainer
    record_path = HERE / "training_record.json"
    if record_path.exists():
        record = read(record_path)
        assert record["status"] == "trained", "Do not silently restart interrupted training."
        for checkpoint in ("best", "last"):
            assert sha(FOLDER / f"weights/{checkpoint}.pt") == record[checkpoint + "_sha256"]
        return
    assert not FOLDER.exists(), "Unexpected pre-existing run."
    initial = cfg["initial_weights"]["11s"]
    model = YOLO(initial["path"])
    assert model.names == {int(k): v for k, v in initial["classes"].items()}
    training = dict(cfg["training"], seed=20261012)
    record = {"status": "training", "started_at_utc": utc(), "initial_weights": initial,
              "expected_training": training, "protocol_sha256": sha(HERE / "protocol.json"), "controller_sha256": sha(__file__)}
    write(record_path, record)
    started = time.perf_counter()

    def on_start(trainer):
        actual = vars(trainer.args)
        reference = read(HERE.parent / "y11n_matched_control_20261015/records/11n_seed20261012_args.json")
        allowed = {"model", "data", "project", "name", "save_dir"}
        differences = {k: [reference.get(k), actual.get(k)] for k in set(reference) | set(actual) if k not in allowed and reference.get(k) != actual.get(k)}
        assert not differences, differences
        assert sha(actual["data"]) == cfg["dataset_yaml_sha256"]
        assert trainer.start_epoch == 0 and trainer.epochs == 120 and trainer.batch_size == 4
        record.update(effective_arguments_verified=True, parameter_count=sum(p.numel() for p in trainer.model.parameters()))
        write(HERE / "effective_args.json", actual); write(record_path, record)
        status(status="training", run=RUN, epoch_completed=0, epochs_planned=120)

    def on_batch(trainer):
        assert trainer.batch_size == 4 and trainer.args.batch == 4, "Batch changed; abort rather than silently change the matched recipe."

    def on_epoch(trainer):
        state = {"status": "training", "run": RUN, "epoch_completed": trainer.epoch + 1, "epochs_planned": 120,
                 "seconds": time.perf_counter() - started, "metrics": {k: float(v) for k, v in trainer.metrics.items()}}
        status(**state); print(json.dumps(state), flush=True)

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_batch_start", on_batch)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.train(trainer=ClassNameTransferTrainer, data=cfg["dataset"], project=str(HERE / "runs"), name=RUN, exist_ok=False, **training)
    with (FOLDER / "results.csv").open(encoding="utf-8") as stream:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    assert [int(float(row["epoch"])) for row in rows] == list(range(1, len(rows) + 1)) and len(rows) <= 120
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    record.update(status="trained", completed_epochs=len(rows), best_calibration_epoch=int(float(best["epoch"])),
        seconds=time.perf_counter() - started, finished_at_utc=utc(), best_sha256=sha(FOLDER / "weights/best.pt"), last_sha256=sha(FOLDER / "weights/last.pt"))
    write(record_path, record)
    del model; torch.cuda.empty_cache()


def evaluate(cfg):
    import numpy as np
    torch, _, YOLO = libraries()
    previous = read(cfg["references"]["comparison"]["path"])
    result = {"status": "completed", "protocol_sha256": sha(HERE / "protocol.json"), "training": read(HERE / "training_record.json"),
        "models": copy.deepcopy(previous["models"]), "audit_ap": copy.deepcopy(read(cfg["references"]["diagnostics"]["path"])["audit_ap"]),
        "limits": ["One matched seed; old development pages, not a blind engineering-source test.", "v8n resumed at epoch60; full RNG/data-loader state was unavailable.", "Architecture-specific COCO pretraining differs. Scores are bbox/category metrics, not official ComponentF1/PinF1/NetF1."]}
    for entry in result["models"].values():
        assert sha(entry["weights"]) == entry["weights_sha256"]
    for checkpoint in ("best", "last"):
        name = RUN + "_" + checkpoint
        weights = FOLDER / f"weights/{checkpoint}.pt"
        saved = collect(cfg, name, weights)
        entry = {"weights": str(weights), "weights_sha256": sha(weights), "modes": {}}
        for mode in ("whole", "sliced", "fused"):
            thresholds, details = tune(cfg, saved, mode)
            entry["modes"][mode] = {"thresholds_from_calibration": thresholds, "threshold_diagnostics": details,
                "audit_global_0_2": counts(saved, mode, set(cfg["audit_cases"]), {}),
                "audit_calibrated": counts(saved, mode, set(cfg["audit_cases"]), thresholds),
                "latency_ms_median": float(np.median([p["timings_ms"][mode] for p in saved["cases"]]))}
        result["models"][name] = entry
        status(status="evaluating_ap", model=name)
        model = YOLO(str(weights))
        ap = model.val(data=str(HERE / "dataset/audit_dataset.yaml"), imgsz=1536, batch=4, workers=2, device=0, half=False,
            conf=.001, iou=.7, max_det=1000, plots=False, save_json=False, project=str(HERE / "audit_ap"), name=name, exist_ok=True, verbose=False)
        result["audit_ap"][name] = {"mAP50": float(ap.box.map50), "mAP50_95": float(ap.box.map)}
        del model; torch.cuda.empty_cache()
    candidate = result["models"][RUN + "_best"]["modes"]["fused"]
    result["paired_differences_11s_minus_reference"] = {}
    for name in ("v8n_seed20261012_best", "11n_seed20261012_best"):
        reference = result["models"][name]["modes"]["fused"]
        result["paired_differences_11s_minus_reference"][name] = {endpoint: {
            "delta_f1": candidate[endpoint]["overall"]["f1"] - reference[endpoint]["overall"]["f1"],
            "bootstrap": bootstrap(reference[endpoint], candidate[endpoint], cfg["case_groups"])}
            for endpoint in ("audit_global_0_2", "audit_calibrated")}
    verify_inputs(cfg)
    result.update(frozen_inputs_unchanged_after_trial=True, completed_at_utc=utc())
    write(HERE / "comparison.json", result)
    lines = ["# YOLO11s相同训练配置对照", "", "同种子20261012，418张训练、30页校准、29页评分；1536、batch4、AMP、AdamW及完整增强配方不变。best按校准集mAP50-95选择，last仅作诊断。", "",
        f"11s实际完成{result['training']['completed_epochs']}轮（上限120，patience25），校准峰值在第{result['training']['best_calibration_epoch']}轮。", "",
        "| 模型 | 固定conf0.2融合F1 | 校准后P | 校准后R | 校准后F1 | 整图mAP50-95 | 融合中位ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("v8n_adopted", "v8n_seed20261012_best", "11n_seed20261012_best", "11n_seed20261015_best", RUN + "_best", RUN + "_last"):
        m = result["models"][name]["modes"]["fused"]; c = m["audit_calibrated"]["overall"]
        lines.append(f"| {name} | {m['audit_global_0_2']['overall']['f1']:.4%} | {c['precision']:.4%} | {c['recall']:.4%} | {c['f1']:.4%} | {result['audit_ap'][name]['mAP50_95']:.4%} | {m['latency_ms_median']:.1f} |")
    lines += ["", "AP采用整图conf0.001、NMS IoU0.7，与融合F1口径不同。", "", "## 配对差值（11s减参照）", ""]
    for name, endpoints in result["paired_differences_11s_minus_reference"].items():
        value = endpoints["audit_calibrated"]; lo, hi = value["bootstrap"]["ci95_delta_f1"]
        lines.append(f"- 相比{name}：F1差值{value['delta_f1']*100:+.3f}个百分点，条件分组bootstrap95%区间[{lo*100:+.3f}, {hi*100:+.3f}]。")
    from evaluate_detector import metric
    empty = metric({"tp": 0, "fp": 0, "fn": 0})
    lines += ["", "## 逐类F1（统一校准流程）", "", "| 类别 | GT | 新v8n | 11n同种子 | 11s |", "|---|---:|---:|---:|---:|"]
    classes = candidate["audit_calibrated"]["by_class"]
    for cls, c in sorted(classes.items(), key=lambda pair: -(pair[1]["tp"] + pair[1]["fn"])):
        gt = c["tp"] + c["fn"]
        if gt:
            old = [result["models"][n]["modes"]["fused"]["audit_calibrated"]["by_class"].get(cls, empty)["f1"] for n in ("v8n_seed20261012_best", "11n_seed20261012_best")]
            lines.append(f"| {cls} | {gt} | {old[0]:.4%} | {old[1]:.4%} | {c['f1']:.4%} |")
    lines += ["", "## 结论边界", "", *["- " + limit for limit in result["limits"]], "", "现有部署权重和配置保留。旧10月12日11s实验使用不同的验证选择划分，不作为本次匹配对照结果。"]
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status(status="completed", comparison=str(HERE / "comparison.md"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-if-idle", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = read(HERE / "protocol.json")
    if args.verify:
        verify_inputs(cfg)
        assert cfg["training"] == read(cfg["parent_protocol"]["path"])["training"]
        print(json.dumps({"frozen_inputs_verified": True, "same_training_recipe": True, "status": read(HERE / "status.json")["status"]})); return
    if args.launch_if_idle:
        launch_if_idle(cfg); return
    assert args.run
    run_lock = lock("run.lock")
    if run_lock is None:
        print("An experiment process is already active."); return
    try:
        status(status="verifying")
        verify_inputs(cfg); libraries(); metric_self_check(); baseline_check(cfg)
        train(cfg); verify_inputs(cfg); evaluate(cfg)
    except BaseException as exc:
        status(status="failed", error=repr(exc))
        raise
    finally:
        run_lock.close()


if __name__ == "__main__":
    main()
