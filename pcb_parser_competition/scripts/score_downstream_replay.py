"""Score a downstream replay against the frozen reference connectivity.

`evaluate_wire_graph.py` is hard-wired to `runs/s7_wire_graph_v1`.  This script
reuses its `pair_counts` (one-to-one pin matching + pair comparison) but reads any
replay directory, so the old and new U-Net can be compared on the SAME reference
snapshot without touching the delivered evaluation path.

Reports, per model:
  * localized_owned_pins        pins matched to a reference pin by owner+location
  * conditional_connectivity    pair P/R/F1 among those matched pins
  * all_reference_pair_proxy    the stricter end-to-end proxy
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_wire_graph import metrics, pair_counts  # noqa: E402
from evaluate_pin_text import component_matches  # noqa: E402

DETECTIONS = ROOT / "runs/s5_pin_ownership_v1"


def owner_map_for(case_id):
    """Detection index -> reference component id, the mapping `match` expects."""
    payload = json.loads((DETECTIONS / case_id / "detector_predictions.json").read_text(encoding="utf-8"))
    truth = [o for o in FIELDS["detector_references"][case_id]["objects"]]
    return component_matches(payload["detections"], truth)


def reference_for(case_id, references):
    mapping = references["cases"][case_id]["reference_pin_networks"]
    truth = [r for r in FIELDS["references"] if r["case_id"] == case_id]
    return mapping, truth


def aggregate(per_case, key):
    total = {"tp": 0, "fp": 0, "fn": 0}
    for row in per_case:
        block = row[key]
        for metric in total:
            total[metric] += block[metric]
    return {"tp": total["tp"], "fp": total["fp"], "fn": total["fn"],
            "precision": round(total["tp"] / (total["tp"] + total["fp"]), 4) if total["tp"] + total["fp"] else None,
            "recall": round(total["tp"] / (total["tp"] + total["fn"]), 4) if total["tp"] + total["fn"] else None,
            "f1": round(2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"]), 4)
            if 2 * total["tp"] + total["fp"] + total["fn"] else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=Path(r"D:/pcb_unet_v2/downstream_replay"))
    parser.add_argument("--references", type=Path, default=ROOT / "runs/s7_wire_graph_v1/reference_connectivity.json")
    parser.add_argument("--fields", type=Path, default=ROOT / "runs/s6_pin_text_v1/draft_field_references.json")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/downstream_eval.json"))
    args = parser.parse_args()

    global FIELDS
    FIELDS = json.loads(args.fields.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))

    report = {"status": "downstream_replay_scored", "reference": str(args.references), "runs": {}}
    for run_dir in sorted(p for p in args.replay.iterdir() if p.is_dir()):
        per_case = []
        for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            case_id = case_dir.name
            if case_id not in references["cases"]:
                continue
            graph = json.loads((case_dir / "wire_graph" / "graph_proposals.json").read_text(encoding="utf-8"))
            pin_result = json.loads((case_dir / "pin_text" / "pin_text_result.json").read_text(encoding="utf-8"))
            owner_map = owner_map_for(case_id)
            mapping, truth = reference_for(case_id, references)
            try:
                row = pair_counts(graph, pin_result, truth, mapping, owner_map)
            except Exception as error:  # noqa: BLE001 - a single broken page must not stop the report
                print(f"  {run_dir.name}/{case_id} failed: {type(error).__name__}: {error}")
                continue
            row["case_id"] = case_id
            per_case.append(row)
        report["runs"][run_dir.name] = {
            "cases": len(per_case),
            "localized_owned_pins": sum(r["localized_owned_pins"] for r in per_case),
            "conditional_connectivity": aggregate(per_case, "conditional_connectivity_pairs"),
            "all_reference_pair_proxy": aggregate(per_case, "all_reference_pair_proxy"),
            "per_case": per_case,
        }
        value = report["runs"][run_dir.name]
        print(f"{run_dir.name:<20} pins {value['localized_owned_pins']:>4}  "
              f"conditional F1 {value['conditional_connectivity']['f1']} "
              f"(P {value['conditional_connectivity']['precision']} R {value['conditional_connectivity']['recall']})  "
              f"proxy F1 {value['all_reference_pair_proxy']['f1']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== per-case conditional F1 ==")
    labels = list(report["runs"])
    if len(labels) == 2:
        old, new = (report["runs"][labels[0]], report["runs"][labels[1]])
        old_map = {r["case_id"]: r["conditional_connectivity_pairs"] for r in old["per_case"]}
        new_map = {r["case_id"]: r["conditional_connectivity_pairs"] for r in new["per_case"]}
        for case_id in sorted(set(old_map) & set(new_map)):
            print(f"  {case_id}: old F1 {old_map[case_id]['f1']} -> new F1 {new_map[case_id]['f1']}"
                  f"  (old tp/fp/fn {old_map[case_id]['tp']}/{old_map[case_id]['fp']}/{old_map[case_id]['fn']}"
                  f" -> {new_map[case_id]['tp']}/{new_map[case_id]['fp']}/{new_map[case_id]['fn']})")
    print("written:", args.output)


if __name__ == "__main__":
    main()
