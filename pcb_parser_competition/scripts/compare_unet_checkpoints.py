"""Evaluate every checkpoint a v2 run produced and print one decision table.

The run keeps `best.pt` (composite), `best_wire.pt`, `best_pin.pt` and `last.pt`.
Because the 59-page validation micro-F1 swings several points between epochs,
all four are scored under the identical protocol and compared against E0, so the
adoption decision never rests on a single argmax.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_unet_v2 import V2_DATA, run_evaluation  # noqa: E402

DEFAULT_RUN = Path(r"D:/pcb_unet_v2/runs/A2_full60")
DEFAULT_OUT = Path(r"D:/pcb_unet_v2/reports/A2_comparison.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline", type=Path, default=Path(r"D:/pcb_unet_v2/reports/baseline_e0.json"))
    parser.add_argument("--manifest", type=Path, default=V2_DATA / "manifest.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == "val"]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None

    checkpoints = [path for name in ("best.pt", "best_wire.pt", "best_pin.pt", "last.pt")
                   for path in [args.run_dir / name] if path.exists()]
    results = {}
    for checkpoint in checkpoints:
        report = run_evaluation(checkpoint, rows, device, 0.5, 0.8, 5.0, baseline)
        results[checkpoint.name] = {key: report[key] for key in ("epoch", "init", "metrics")}
        metrics = report["metrics"]
        print(f"{checkpoint.name:<14} epoch {report['epoch']:>3}  "
              f"wire F1 {metrics['wire']['f1']:.4f} (P {metrics['wire']['precision']:.4f} R {metrics['wire']['recall']:.4f})  "
              f"pin F1 {metrics['pin']['f1']:.4f} (P {metrics['pin']['precision']:.4f} R {metrics['pin']['recall']:.4f})",
              flush=True)

    output = {"status": "run_checkpoint_comparison", "run_dir": str(args.run_dir), "pages": len(rows),
              "baseline": {"checkpoint": baseline["checkpoint"] if baseline else None,
                           "wire_f1": baseline["wire"]["v2"]["f1"] if baseline else None,
                           "pin_f1": baseline["pin"]["v2"]["f1"] if baseline else None},
              "checkpoints": results}

    # per-source-kind and per-page deltas for the composite winner vs E0
    if baseline and results:
        winner = max(results, key=lambda name: 0.5 * results[name]["metrics"]["wire"]["f1"]
                     + 0.5 * results[name]["metrics"]["pin"]["f1"])
        winner_report = run_evaluation(args.run_dir / winner, rows, device, 0.5, 0.8, 5.0, baseline)
        kinds = {json.loads(line)["case_id"]: json.loads(line)["raw_source_kind"]
                 for line in (ROOT / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()}
        before = {r["case_id"]: {"wire": r["wire_v2"], "pin": r["pin_v2"]} for r in baseline["per_case"]}
        after = {r["case_id"]: r for r in winner_report["per_case"]}
        def bucket(field):
            groups = {}
            for case_id, row in after.items():
                kind = kinds.get(case_id, "?")
                group = groups.setdefault(kind, {"pages": 0})
                group["pages"] += 1
                for metric in ("tp", "fp", "fn"):
                    group[f"{field}_{metric}"] = group.get(f"{field}_{metric}", 0) + row[field][metric]
                    group[f"{field}_before_{metric}"] = group.get(f"{field}_before_{metric}", 0) + before[case_id][field][metric]
            def f1(tp, fp, fn):
                return round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None
            return {kind: {"pages": value["pages"],
                           "after": f1(value.get(f"{field}_tp", 0), value.get(f"{field}_fp", 0), value.get(f"{field}_fn", 0)),
                           "before": f1(value.get(f"{field}_before_tp", 0), value.get(f"{field}_before_fp", 0), value.get(f"{field}_before_fn", 0))}
                    for kind, value in groups.items()}
        wire_by_kind, pin_by_kind = bucket("wire"), bucket("pin")
        per_page = [{"case_id": case_id,
                     "wire_before": before[case_id]["wire"]["f1"], "wire_after": after[case_id]["wire"]["f1"],
                     "pin_before": before[case_id]["pin"]["f1"], "pin_after": after[case_id]["pin"]["f1"],
                     "source_kind": kinds.get(case_id, "?")} for case_id in after]
        output["winner"] = winner
        output["winner_epoch"] = results[winner]["epoch"]
        output["by_source_kind"] = {
            kind: {"pages": wire_by_kind[kind]["pages"],
                   "wire_before": wire_by_kind[kind]["before"], "wire_after": wire_by_kind[kind]["after"],
                   "pin_before": pin_by_kind.get(kind, {}).get("before"),
                   "pin_after": pin_by_kind.get(kind, {}).get("after")}
            for kind in wire_by_kind}
        output["per_page"] = per_page
        output["per_page_medians"] = {
            "wire_before": round(__import__("statistics").median([p["wire_before"] for p in per_page]), 4),
            "wire_after": round(__import__("statistics").median([p["wire_after"] for p in per_page]), 4),
            "pin_before": round(__import__("statistics").median([p["pin_before"] for p in per_page]), 4),
            "pin_after": round(__import__("statistics").median([p["pin_after"] for p in per_page]), 4),
        }
        output["pages_regressed_wire"] = sorted(
            [p for p in per_page if p["wire_after"] < p["wire_before"] - 0.02],
            key=lambda p: p["wire_after"] - p["wire_before"])[:10]
        output["pages_improved_wire"] = sorted(
            [p for p in per_page if p["wire_after"] > p["wire_before"] + 0.02],
            key=lambda p: p["wire_after"] - p["wire_before"], reverse=True)[:10]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if baseline:
        print(f"E0 baseline: wire {baseline['wire']['v2']['f1']:.4f}  pin {baseline['pin']['v2']['f1']:.4f}")
    best = max(results.items(), key=lambda kv: 0.5 * kv[1]["metrics"]["wire"]["f1"] + 0.5 * kv[1]["metrics"]["pin"]["f1"])
    print(f"best by composite: {best[0]} (epoch {best[1]['epoch']})")
    print("written:", args.output)


if __name__ == "__main__":
    main()
