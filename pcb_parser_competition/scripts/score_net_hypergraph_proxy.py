"""Approximate the net-hypergraph score locally.  NOT the official metric.

Why this is a proxy
-------------------
The official NetHypergraphF1 needs the competition evaluator, which is not in this
workspace (`official_NetF1` is null everywhere and `export_coverage.json` records
`official_evaluator_verified: false`).  On top of that, today's strict export
produces **zero nets**, so a real submission would score 0 on the net item no matter
how good the internal graph is.

This script therefore scores the INTERNAL graph proposals against the frozen
reference net memberships, at hypergraph level:

  reference net  = set of reference pins sharing a reference net id
  predicted net  = set of predicted pins sharing a predicted network_id

A predicted net is a true positive only if its member set exactly equals a
reference net's member set (unmatched predicted nets are FP, unmatched reference
nets are FN).  Pairs inside a partially-correct net earn nothing here, which makes
this deliberately stricter than the pair-level proxy in
`score_downstream_replay.py`.

Reported alongside: what a strict export would submit today, and why the
export gate drops nets.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_pin_text import component_matches, match  # noqa: E402

REPLAY = Path(r"D:/pcb_unet_v2/downstream_replay")


def metric(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--references", type=Path, default=ROOT / "runs/s7_wire_graph_v1/reference_connectivity.json")
    parser.add_argument("--fields", type=Path, default=ROOT / "runs/s6_pin_text_v1/draft_field_references.json")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/net_hypergraph_proxy.json"))
    args = parser.parse_args()

    fields = json.loads(args.fields.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))

    report = {"status": "net_hypergraph_proxy_not_official", "runs": {}}
    for run_dir in sorted(p for p in args.replay.iterdir() if p.is_dir()):
        per_case, blocked = [], Counter()
        for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            case_id = case_dir.name
            if case_id not in references["cases"]:
                continue
            graph = json.loads((case_dir / "wire_graph" / "graph_proposals.json").read_text(encoding="utf-8"))
            pin_result = json.loads((case_dir / "pin_text" / "pin_text_result.json").read_text(encoding="utf-8"))
            coverage_path = case_dir / "wire_graph" / "export_coverage.json"
            if coverage_path.exists():
                coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
                for reason, count in (coverage.get("network_reason_counts") or {}).items():
                    blocked[reason] += count

            detections = json.loads((ROOT / f"runs/s5_pin_ownership_v1/{case_id}/detector_predictions.json")
                                    .read_text(encoding="utf-8"))["detections"]
            owner_map = component_matches(detections, fields["detector_references"][case_id]["objects"])
            truth_pins = [r for r in fields["references"] if r["case_id"] == case_id]
            assigned = [p for p in pin_result["pin_candidates"] if p["owner_status"] == "assigned"]
            pairs = match(assigned, truth_pins, owner_map)
            # predicted candidate -> reference pin uid
            candidate_to_ref = {assigned[i]["candidate_id"]: f"{truth_pins[j]['component_source_id']}/{truth_pins[j]['pin_source_id']}"
                                for i, j in pairs}

            reference_nets = {}
            for uid, net_id in references["cases"][case_id]["reference_pin_networks"].items():
                reference_nets.setdefault(net_id, set()).add(uid)
            predicted_nets = {}
            for attachment in graph["pin_attachments"]:
                if attachment.get("attachment_status") != "attached" or not attachment.get("network_id"):
                    continue
                uid = candidate_to_ref.get(attachment["candidate_id"])
                if uid:
                    predicted_nets.setdefault(attachment["network_id"], set()).add(uid)

            reference_sets = {frozenset(members) for members in reference_nets.values() if len(members) >= 2}
            predicted_sets = {frozenset(members) for members in predicted_nets.values() if len(members) >= 1}
            hits = predicted_sets & reference_sets
            per_case.append({"case_id": case_id,
                             "reference_multi_pin_nets": len(reference_sets),
                             "predicted_nets": len(predicted_sets),
                             "exact_matches": len(hits),
                             "reference_single_pin_nets": sum(1 for m in reference_nets.values() if len(m) == 1),
                             "tp": len(hits),
                             "fp": len(predicted_sets) - len(hits),
                             "fn": len(reference_sets) - len(hits)})

        tp = sum(c["tp"] for c in per_case)
        fp = sum(c["fp"] for c in per_case)
        fn = sum(c["fn"] for c in per_case)
        report["runs"][run_dir.name] = {
            "cases": len(per_case),
            "hypergraph_exact_net_match": metric(tp, fp, fn),
            "reference_multi_pin_nets": sum(c["reference_multi_pin_nets"] for c in per_case),
            "predicted_nets": sum(c["predicted_nets"] for c in per_case),
            "export_blocked_reason_counts": dict(blocked),
            "per_case": per_case,
        }
        value = report["runs"][run_dir.name]["hypergraph_exact_net_match"]
        print(f"{run_dir.name:<18} ref_net {report['runs'][run_dir.name]['reference_multi_pin_nets']:>4}  "
              f"pred_net {report['runs'][run_dir.name]['predicted_nets']:>4}  "
              f"exact {value['tp']:>3}  P {value['precision']:.4f} R {value['recall']:.4f} F1 {value['f1']:.4f}")

    if report["runs"]:
        first = next(iter(report["runs"].values()))
        print("\nexport gate blocking (sum over 12 pages):")
        for reason, count in sorted(first["export_blocked_reason_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<58} {count:>5}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)
    print("REMINDER: this is a local hypergraph proxy, never the official NetHypergraphF1.")


if __name__ == "__main__":
    main()
