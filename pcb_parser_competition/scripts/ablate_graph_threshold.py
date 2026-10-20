"""Wire-graph stage ablation: does the cleaner U-Net mask need a different threshold?

The new U-Net produces a sparser, higher-precision wire map (fewer false pixels),
which changes how much graph the skeleton stage grows from the same threshold.
This sweep replays ONLY the graph stage over cached probabilities at several wire
thresholds, for both the old and new U-Net, and scores each against the frozen
reference connectivity.

Everything downstream of the wire probability is re-derived; no model runs, no
weights change, nothing outside --root is written.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.wire_graph import build_wire_graph  # noqa: E402
from evaluate_wire_graph import pair_counts  # noqa: E402
from evaluate_pin_text import component_matches  # noqa: E402

REPLAY = Path(r"D:/pcb_unet_v2/downstream_replay")
RUNS = {"old_v1_unet": None, "new_a3_split_w2": None}
THRESHOLDS = (0.35, 0.5, 0.65, 0.8)


def owner_map_for(case_id, detector_references):
    payload = json.loads((ROOT / f"runs/s5_pin_ownership_v1/{case_id}/detector_predictions.json")
                         .read_text(encoding="utf-8"))
    return component_matches(payload["detections"], detector_references[case_id]["objects"])


def aggregate(per_case, key):
    total = {"tp": 0, "fp": 0, "fn": 0}
    for row in per_case:
        for metric in total:
            total[metric] += row[key][metric]
    return {"tp": total["tp"], "fp": total["fp"], "fn": total["fn"],
            "precision": round(total["tp"] / (total["tp"] + total["fp"]), 4) if total["tp"] + total["fp"] else None,
            "recall": round(total["tp"] / (total["tp"] + total["fn"]), 4) if total["tp"] + total["fn"] else None,
            "f1": round(2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"]), 4)
            if 2 * total["tp"] + total["fp"] + total["fn"] else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--references", type=Path, default=ROOT / "runs/s7_wire_graph_v1/reference_connectivity.json")
    parser.add_argument("--fields", type=Path, default=ROOT / "runs/s6_pin_text_v1/draft_field_references.json")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/graph_threshold_ablation.json"))
    args = parser.parse_args()

    fields = json.loads(args.fields.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))
    graph_cfg = json.loads((ROOT / "configs/wire_graph_v2.json").read_text(encoding="utf-8"))
    results = {}
    for run_name in RUNS:
        run_dir = args.replay / run_name
        if not run_dir.exists():
            continue
        cases = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
        per_threshold = {threshold: [] for threshold in THRESHOLDS}
        for case_id in cases:
            cache = run_dir / case_id / "ownership" / "probabilities.npz"
            pin_result_path = run_dir / case_id / "pin_text" / "pin_text_result.json"
            if not (cache.exists() and pin_result_path.exists()) or case_id not in references["cases"]:
                continue
            import numpy as np
            with np.load(cache, allow_pickle=False) as cached:
                probability = cached["wire"]
            pin_result = json.loads(pin_result_path.read_text(encoding="utf-8"))
            owner_map = owner_map_for(case_id, fields["detector_references"])
            truth = [r for r in fields["references"] if r["case_id"] == case_id]
            mapping = references["cases"][case_id]["reference_pin_networks"]
            for threshold in THRESHOLDS:
                cfg = dict(graph_cfg)
                cfg["wire_threshold"] = threshold
                graph, _masks = build_wire_graph(probability, pin_result, cfg)
                try:
                    scored = pair_counts(graph, pin_result, truth, mapping, owner_map)
                except Exception as error:  # noqa: BLE001
                    print(f"  {run_name}/{case_id}@{threshold}: {type(error).__name__}: {error}")
                    continue
                scored["case_id"] = case_id
                scored["graph_scale"] = graph["summary"]
                per_threshold[threshold].append(scored)
        results[run_name] = {str(threshold): {
            "cases": len(rows),
            "localized_owned_pins": sum(r["localized_owned_pins"] for r in rows),
            "conditional_connectivity": aggregate(rows, "conditional_connectivity_pairs"),
            "all_reference_pair_proxy": aggregate(rows, "all_reference_pair_proxy"),
            "graph_scale": {key: sum(r["graph_scale"][key] for r in rows) for key in
                            ("graph_nodes", "graph_edges", "network_proposals",
                             "multi_pin_network_proposals", "attached_pins", "gap_candidates_not_filled")},
        } for threshold, rows in per_threshold.items()}
        print(f"\n== {run_name} ==")
        print(f"{'thr':>5}{'nodes':>7}{'nets':>6}{'multi':>6}{'attached':>9}"
              f"{'cond P':>8}{'cond R':>8}{'cond F1':>9}{'proxy F1':>10}{'TP':>5}{'FP':>5}{'FN':>5}")
        for threshold in THRESHOLDS:
            value = results[run_name][str(threshold)]
            scale = value["graph_scale"]
            conditional = value["conditional_connectivity"]
            proxy = value["all_reference_pair_proxy"]
            print(f"{threshold:>5}{(scale['graph_nodes']):>7}{scale['network_proposals']:>6}"
                  f"{scale['multi_pin_network_proposals']:>6}{scale['attached_pins']:>9}"
                  f"{conditional['precision']:>8.4f}{conditional['recall']:>8.4f}{conditional['f1']:>9.4f}"
                  f"{proxy['f1']:>10.4f}{proxy['tp']:>5}{proxy['fp']:>5}{proxy['fn']:>5}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "graph_threshold_ablation",
                                       "graph_config": str(ROOT / "configs/wire_graph_v2.json"),
                                       "runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
