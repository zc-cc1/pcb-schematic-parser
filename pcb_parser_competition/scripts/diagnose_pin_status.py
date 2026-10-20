"""Where the pin score actually goes: decompose PinF1 (and the net ceiling) per case.

Read-only: uses the cached strict export (`partial_submission.json`) and the public
Golden reference.  No model, no GPU.

Buckets on the reference-pin side (FN):
  owner_unmatched   the owning component itself did not match -> all its pins lost
  number_absent     component matched, but we exported no pin with that number
  point_far         same number, tip farther than PIN_TOLERANCE_PX
  name_diff         same number, tip in tolerance, pinname differs
And on the predicted side (FP):
  fp_owner          pin hangs on a component that did not match
  fp_number         component matched, number not in the reference (invented/wrong)

Also reports, for every reference net, whether all of its members are recoverable
(number present and tip within tolerance) - that is the net ceiling implied by pins.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.contest_metrics import PIN_TOLERANCE_PX, match_components  # noqa: E402

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
CACHE = Path(r"D:/pcb_unet_v2/golden_eval")


def load_png_height(path: Path) -> float:
    from PIL import Image
    with Image.open(path) as image:
        return float(image.height)


def net_members(network) -> list[tuple[str, str]]:
    body = str(network.get("hyperGraph") or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    out = []
    for token in body.split(","):
        token = token.strip()
        if not token or token in {"∅", "null", "None"}:
            continue
        if "." in token:
            component, pin = token.rsplit(".", 1)
            out.append((component, pin))
        else:
            out.append((token, ""))
    return out


def analyse(case_id: str, gt_folder: Path, predictions: Path | None = None) -> dict:
    truth_path = next(iter(sorted(gt_folder.glob("*_target.json")) or sorted(gt_folder.glob("*.json"))))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    height = load_png_height(next(gt_folder.glob("*.png")))
    if predictions is not None:
        candidates = [predictions / f"{case_id}_Pred.json", predictions / f"{case_id}.json"]
        pred_path = next((p for p in candidates if p.is_file()), candidates[0])
    else:
        pred_path = CACHE / case_id / "wire_graph" / "partial_submission.json"
    pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else \
        {"components": {}, "pins": {}, "nets": {}}

    mapping, *_ = match_components(pred, truth, height)
    reverse = {t: p for p, t in mapping.items()}

    buckets: Counter[str] = Counter()
    dists: list[float] = []
    fp_buckets: Counter[str] = Counter()
    truth_pins = truth.get("pins") or {}
    pred_pins = pred.get("pins") or {}
    total_truth_pins = 0
    number_ok = 0

    for truth_component, pins in truth_pins.items():
        total_truth_pins += len(pins)
        pred_component = reverse.get(truth_component)
        if pred_component is None:
            buckets["owner_unmatched"] += len(pins)
            continue
        candidates = pred_pins.get(pred_component) or {}
        for key, reference in pins.items():
            candidate = candidates.get(key)
            if candidate is None:
                # distinguish "numbering picked another number" from "no pin at all"
                buckets["number_absent" if candidates else "no_pin_exported"] += 1
                continue
            number_ok += 1
            distance = math.hypot(float(candidate["point"]["x"]) - float(reference["point"]["x"]),
                                  float(candidate["point"]["y"]) - float(reference["point"]["y"]))
            dists.append(distance)
            if distance > PIN_TOLERANCE_PX:
                buckets["point_far"] += 1
                continue
            expected, actual = reference.get("pinname"), candidate.get("pinname")
            if expected not in (None, "") and actual not in (None, "") and str(expected) != str(actual):
                buckets["name_diff"] += 1
                continue
            buckets["tp"] += 1

    for pred_component, pins in pred_pins.items():
        truth_component = mapping.get(pred_component)
        if truth_component is None:
            fp_buckets["fp_owner"] += len(pins)
            continue
        reference = truth_pins.get(truth_component) or {}
        for key in pins:
            if key not in reference:
                fp_buckets["fp_number"] += 1

    # net ceiling implied by pins: all members must have the right number and a tip
    # inside tolerance on a matched component
    recoverable = 0
    numbered = 0
    blocked_components: Counter[str] = Counter()
    truth_nets = truth.get("nets") or {}
    for net in truth_nets.values():
        ok = True
        ok_number = True
        for component, number in net_members(net):
            pred_component = reverse.get(component)
            if pred_component is None:
                ok = ok_number = False
                blocked_components["owner_unmatched"] += 1
                break
            candidate = (pred_pins.get(pred_component) or {}).get(f"pin_{number}")
            if candidate is None:
                ok = ok_number = False
                blocked_components["number_absent"] += 1
                break
            reference = (truth_pins.get(component) or {}).get(f"pin_{number}") or {}
            point = reference.get("point") or {}
            if point and math.hypot(float(candidate["point"]["x"]) - float(point["x"]),
                                    float(candidate["point"]["y"]) - float(point["y"])) > PIN_TOLERANCE_PX:
                # the net metric compares pin identity only (R08), so a far tip does
                # not cost the net - it only costs the pin (R05/R07)
                ok = False
        recoverable += 1 if ok else 0
        numbered += 1 if ok_number else 0

    dists.sort()
    p50 = dists[len(dists) // 2] if dists else 0.0
    p90 = dists[int(len(dists) * 0.9)] if dists else 0.0
    return {"case_id": case_id,
            "truth_pins": total_truth_pins,
            "exported_pins": sum(len(v) for v in pred_pins.values()),
            "buckets": dict(buckets),
            "fp_buckets": dict(fp_buckets),
            "number_present_but_bad_point": buckets["point_far"],
            "distance_p50": round(p50, 2), "distance_p90": round(p90, 2),
            "truth_nets": len(truth_nets),
            "nets_with_all_members_recoverable": recoverable,
            "nets_with_all_members_numbered": numbered,
            "net_blockers": dict(blocked_components),
            "fp_total": sum(fp_buckets.values()),
            "fn_total": total_truth_pins - buckets["tp"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(r"D:/pcb_unet_v2/reports/pin_status_diagnostic.json"))
    parser.add_argument("--gt-root", type=Path, default=GT_DIR,
                        help="reference root; cases are sorted by name, '<id>_target.json' or '<id>.json'")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="directory with '<id>_Pred.json'; omitted means the cached golden_eval tree")
    args = parser.parse_args()

    folders = [f for f in sorted(args.gt_root.iterdir()) if f.is_dir()]
    def order(folder):
        head = folder.name.split("-")[0]
        return (0, int(head)) if head.isdigit() else (1, head)
    rows = []
    for folder in sorted(folders, key=order):
        rows.append(analyse(folder.name.split("-")[0], folder, args.predictions))

    totals = Counter()
    for row in rows:
        totals.update(row["buckets"])
        totals.update({f"fp_{k}": v for k, v in row["fp_buckets"].items()})
    report = {"status": "local_reference_implementation_not_official",
              "pin_tolerance_px": PIN_TOLERANCE_PX,
              "cases": rows, "totals": dict(totals)}

    header = f"{'case':<6}{'GT':>4}{'exp':>5}{'TP':>4}{'owner':>7}{'num':>5}{'point':>6}{'name':>5}{'FPnum':>6}{'FPown':>6}{'net':>4}{'numOK':>6}{'recov':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        b, f = row["buckets"], row["fp_buckets"]
        print(f"{row['case_id']:<6}{row['truth_pins']:>4}{row['exported_pins']:>5}{b.get('tp', 0):>4}"
              f"{b.get('owner_unmatched', 0):>7}{b.get('number_absent', 0):>5}{b.get('point_far', 0):>6}"
              f"{b.get('name_diff', 0):>5}{f.get('fp_number', 0):>6}{f.get('fp_owner', 0):>6}"
              f"{row['truth_nets']:>4}{row['nets_with_all_members_numbered']:>6}"
              f"{row['nets_with_all_members_recoverable']:>6}")
    print("-" * len(header))
    print("totals:", json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    print(f"nets numbered (owner+number) : {sum(r['nets_with_all_members_numbered'] for r in rows)}"
          f" / {sum(r['truth_nets'] for r in rows)}")
    print(f"nets fully recoverable (+5px): {sum(r['nets_with_all_members_recoverable'] for r in rows)}"
          f" / {sum(r['truth_nets'] for r in rows)}")
    print("distance p50/p90 of number-matched tips:",
          [f"{r['case_id']}:{r['distance_p50']}/{r['distance_p90']}" for r in rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", args.output)


if __name__ == "__main__":
    main()
