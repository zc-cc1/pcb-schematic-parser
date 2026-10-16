"""Read official result artifacts without changing models or predictions."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def report_directory(root):
    dirs = [p for p in (root / "report/submission").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def count_f1(reports, section):
    return {field: sum(r[section]["f1"][field] for r in reports)
            for field in ("tp", "fp", "fn")}


def compare_pins(old, new):
    old_ids = {(c, p) for c, pins in old.items() for p in pins}
    new_ids = {(c, p) for c, pins in new.items() for p in pins}
    distances, field_changes = {}, []
    for component, pin in sorted(old_ids & new_ids):
        left, right = old[component][pin], new[component][pin]
        distances[component + "." + pin] = math.hypot(
            left["point"]["x"] - right["point"]["x"],
            left["point"]["y"] - right["point"]["y"])
        if left["pinname"] != right["pinname"]:
            field_changes.append([component, pin, left["pinname"], right["pinname"]])
    return {"max_common_pin_position_change_px": max(distances.values(), default=0),
            "common_pin_position_changes_px": distances,
            "added_pin_ids": sorted(new_ids - old_ids),
            "removed_pin_ids": sorted(old_ids - new_ids),
            "pin_name_changes": field_changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_dir, new_dir = map(report_directory, (args.old_root, args.new_root))
    summary = read(args.new_root / "summary.json")
    old_summary = read(args.old_root / "summary.json")
    ids = sorted((row["case"] for row in summary["cases"]), key=int)
    cases, reports = [], []
    fields, combinations, type_errors = Counter(), Counter(), defaultdict(Counter)
    pending_reasons, net_reasons, validations = Counter(), Counter(), Counter()
    totals = Counter()
    shared = []
    for case in ids:
        rep = read(new_dir / (case + ".json"))
        reports.append(rep)
        pred = read(args.new_root / f"final_results/{case}_Pred.json")
        image = read(args.new_root / f"logs/{case}_summary.json")
        coverage = read(args.new_root / f"logs/{case}_engineering_export_coverage.json")
        cd, pd = rep["component"]["diagnostics"], rep["pin"]["diagnostics"]
        errors = [x for x in cd["error_examples"] if "failed_checks" in x]
        assert len(errors) == cd["matched_component_count"] - rep["component"]["f1"]["tp"]
        case_fields = Counter()
        for error in errors:
            case_fields.update(error["failed_checks"])
            fields.update(error["failed_checks"])
            combinations["+".join(sorted(error["failed_checks"]))] += 1
            type_errors[pred["components"][error["component_id"]]["type"]].update(error["failed_checks"])
        for issue in rep["validation"]["pred"]["issues"]:
            validations[issue["message"]] += 1
        pending_reasons.update(coverage.get("pin_reason_counts", {}))
        net_reasons.update(coverage.get("network_reason_counts", {}))
        totals.update({"pred_components": len(pred["components"]),
                       "pred_pins_all_types": sum(map(len, pred["pins"].values())),
                       "gt_power_symbols": cd["gt_power_symbol_count"],
                       "pred_power_symbols": cd["pred_power_symbol_count"],
                       "matched_ordinary_keys": cd["matched_id_count"],
                       "gt_ordinary_components": rep["component"]["key_f1"]["tp"] + rep["component"]["key_f1"]["fn"],
                       "gt_logical_nets": rep["connectivity"]["f1"]["gt_net_count"],
                       "pred_logical_nets": rep["connectivity"]["f1"]["pred_net_count"],
                       "gt_raw_nets": rep["connectivity"]["diagnostics"]["gt_raw_net_count"],
                       "exact_net_tp": rep["connectivity"]["exact_net_f1"]["tp"],
                       "pending_pins": len(coverage.get("pending_pins", [])),
                       "withheld_networks": len(coverage.get("withheld_networks", [])),
                       "field_only_withheld_networks": sum(x.get("reasons") == ["contains_pin_with_unresolved_identity_number_or_name"]
                                                             for x in coverage.get("withheld_networks", []))})
        internal_power = None
        for line in (args.new_root / f"logs/{case}.log").read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("{'before_power':"):
                internal_power = ast.literal_eval(line)["power"]["power_components_in_review"]
        if internal_power is not None:
            totals["internal_power_review_candidates"] += internal_power
        row = {"case": case, "scores": rep["scores"], "image_sha256": image["image_sha256"],
               "image_size": image["size"], "component_counts": rep["component"]["f1"],
               "pin_counts": rep["pin"]["f1"], "wire_counts": rep["wire"]["f1"],
               "component_failed_checks": dict(case_fields),
               "component_error_examples": errors,
               "pin_error_examples": pd["error_examples"],
               "gt_power_symbols": cd["gt_power_symbol_count"],
               "pred_power_symbols": cd["pred_power_symbol_count"],
               "pred_nets": len(pred["nets"]),
               "gt_logical_nets": rep["connectivity"]["f1"]["gt_net_count"],
               "internal_power_candidates": internal_power}
        old_image_path = args.old_root / f"logs/{case}_summary.json"
        if old_image_path.exists():
            old_image = read(old_image_path)
            old_report = read(old_dir / (case + ".json"))
            old_pred = read(args.old_root / f"final_results/{case}_Pred.json")
            row["comparison"] = {
                "same_input_image_sha256": old_image["image_sha256"] == image["image_sha256"],
                "old_image_sha256": old_image["image_sha256"], "old_image_size": old_image["size"],
                "old_scores": old_report["scores"],
                "old_pin_distance_summary": old_report["pin"]["diagnostics"]["tip_distance_px"],
                "new_pin_distance_summary": pd["tip_distance_px"],
                "old_component_counts": old_report["component"]["f1"],
                "old_component_errors": old_report["component"]["diagnostics"]["error_examples"],
                **compare_pins(old_pred["pins"], pred["pins"])}
            if row["comparison"]["same_input_image_sha256"]:
                shared.append(case)
        cases.append(row)
    weights = summary["weights"]
    paired = {}
    for root, rep_dir, label in [(args.old_root, old_dir, "old"), (args.new_root, new_dir, "new")]:
        scores = [read(rep_dir / (case + ".json"))["scores"] for case in shared]
        paired[label] = {key: sum(x[key] for x in scores) / len(scores)
                         for key in (*weights.keys(), "final_score")}
    audit = {"analysis_only": True, "old_root": str(args.old_root), "new_root": str(args.new_root),
             "old_summary": old_summary, "summary": summary,
             "weighted_points": {k: weights[k] * 100 * summary["macro_avg"][k] for k in weights},
             "decision_log": read(args.new_root / "decision_log.json"),
             "orchestration": read(args.new_root / "orchestration.json"),
             "outer_metadata": (args.new_root / "metadata").read_text(encoding="utf-8-sig"),
             "totals": dict(totals), "component_counts": count_f1(reports, "component"),
             "pin_counts": count_f1(reports, "pin"), "wire_counts": count_f1(reports, "wire"),
             "component_failed_checks": dict(fields),
             "component_failure_combinations": dict(combinations),
             "component_failed_checks_by_type": {k: dict(v) for k, v in type_errors.items()},
             "pending_pin_reason_counts": dict(pending_reasons), "withheld_network_reason_counts": dict(net_reasons),
             "validation_issue_counts": dict(validations), "shared_image_cases": shared,
             "paired_image_macro_scores": paired,
             "paired_image_comparison_limit": "Same pixels do not establish unchanged ground truth or scorer. Cases 5 and 9 contradict unchanged geometric scoring.",
             "cases": cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = [args.new_root / name for name in ["summary.json", "scores.txt", "decision_log.json", "orchestration.json", "metadata"]]
    paths += sorted(new_dir.glob("*.json"))
    paths += sorted((args.new_root / "final_results").glob("*_Pred.json"))
    paths += sorted((args.new_root / "logs").glob("*_summary.json"))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (args.output / "source_sha256.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in {"cases", "old_summary", "summary", "decision_log"}}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
