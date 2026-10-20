"""Check cached historical predictions against platform wire counts; no inference."""
import argparse
import copy
import json
from pathlib import Path
import random

from platform_wire_metrics import WIRE_SCORER_VERSION, score_wire_detailed


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def permuted(document, rng):
    result = copy.deepcopy(document)
    nets = list(result["nets"].items())
    rng.shuffle(nets)
    result["nets"] = dict(nets)
    for net in result["nets"].values():
        edges = list(net["edges"].items())
        rng.shuffle(edges)
        net["edges"] = {key: list(reversed(value)) if rng.randrange(2) else value
                        for key, value in edges}
        members = str(net.get("hyperGraph") or "").strip("()").split(",")
        rng.shuffle(members)
        net["hyperGraph"] = "(" + ",".join(members) + ")"
    return result


def validate(workspace, editions, cases=None, permutations=5):
    truth_root = workspace / "赛题六公开数据集/赛题六公开数据集/open_GT/open_GT"
    rng = random.Random(20261020)
    rows = []
    for edition in editions:
        archive = workspace / "评测结果" / edition / "scoring_result"
        reports = sorted((archive / "report/submission").iterdir())
        if len(reports) != 1:
            raise ValueError(f"Expected one report run in {archive}")
        summary = read(archive / "summary.json")
        for item in summary["cases"]:
            case = item["case"]
            if cases and case not in cases:
                continue
            pred = read(archive / f"final_results/{case}_Pred.json")
            truth = read(next((truth_root / case).glob("*.json")))
            platform = read(reports[0] / f"{case}.json")
            block, detail = score_wire_detailed(pred, truth)
            expected = platform["wire"]["f1"]
            counts_match = all(block[k] == expected[k] for k in ("tp", "fp", "fn"))
            pairs = lambda values: {(tuple(v["gt_net_ids"]), tuple(v["pred_net_ids"]))
                                    for v in values}
            pairs_match = pairs(detail["matched_net_pairs"]) == pairs(
                platform["connectivity"]["diagnostics"]["matched_net_pairs"])
            invariant = all(score_wire_detailed(permuted(pred, rng), permuted(truth, rng))[0] == block
                            for _ in range(permutations))
            rows.append({"edition": edition, "case": case, "actual": block,
                         "expected": expected, "counts_match": counts_match,
                         "network_pairs_match": pairs_match,
                         "permutation_invariant": invariant})
    return {"version": WIRE_SCORER_VERSION, "cases": len(rows),
            "count_matches": sum(r["counts_match"] for r in rows),
            "pair_matches": sum(r["network_pairs_match"] for r in rows),
            "invariant_cases": sum(r["permutation_invariant"] for r in rows),
            "permutations_per_case": permutations, "per_case": rows,
            "neural_inference_calls": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--editions", nargs="+", default=["第9次测评-66.59", "第8次测评-54.92"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.workspace, args.editions, set(args.cases) if args.cases else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, ensure_ascii=False))
    if any(result[key] != result["cases"] for key in
           ("count_matches", "pair_matches", "invariant_cases")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
