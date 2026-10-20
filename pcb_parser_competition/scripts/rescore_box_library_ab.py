"""Rescore the stored box-library A/B predictions with the current local metrics.

Why this exists: `reports/box_library_ab.json` was produced before `contest_metrics`
switched to the PDF's key-based component matching (the report still lists "IoU >= 0.5"
in its assumptions).  Re-running the pipeline would mix two changes, so this script
reuses the *existing* `partial_submission.json` for every case and only varies the
scorer.  That isolates the metric change from the pipeline change.

Variants (see `contest_metrics.TEXT_MODES`):
    strict            the PDF rule as written (box/ic compares Name)
    designator_name   10_GTcase stores the designator in Name, so that pair is not
                      text-checked; drop this variant once the new GT ships
    none              no text comparison at all (upper bound)

Everything printed is the local reference implementation, not the platform score.

    python scripts/rescore_box_library_ab.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.contest_metrics import (  # noqa: E402
    TEXT_MODES, component_breakdown, evaluate_cases, f1,
)

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
STAGE_ROOT = Path(r"D:/pcb_unet_v2/box_library_ab")
METRICS = ("component", "pin", "net_hypergraph", "net_line")


def image_height(path: Path) -> float:
    from PIL import Image
    with Image.open(path) as handle:
        return float(handle.height)


def load_cases() -> list[dict]:
    cases = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        case_dir = STAGE_ROOT / case
        if not case_dir.is_dir():
            continue
        cases.append({
            "case": case,
            "truth": json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8")),
            "height": image_height(ROOT / f"data/standardized/{case}/{case}.png"),
        })
    return cases


def self_test(cases) -> None:
    """GT against itself must be a perfect score in every variant; catches rule bugs."""
    for mode in TEXT_MODES:
        pairs = [(c["case"], copy.deepcopy(c["truth"]), copy.deepcopy(c["truth"]), c["height"])
                 for c in cases]
        report = evaluate_cases(pairs, text_mode=mode)
        bad = {k: v["f1"] for k, v in report["metrics"].items() if v["f1"] < 0.999}
        print(f"  self-test [{mode}]: {'OK' if not bad else 'FAILED ' + json.dumps(bad)}")
        if bad:
            raise SystemExit("self-test failed; refusing to report scores")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--output", type=Path,
                        default=Path(r"D:/pcb_unet_v2/reports/box_library_ab_rescored.json"))
    args = parser.parse_args()

    cases = load_cases()
    if not cases:
        raise SystemExit(f"no staged predictions under {args.stage_root}")
    print(f"cases: {len(cases)} (staged predictions reused; pipeline not re-run)\n")
    print("self-test (truth vs itself):")
    self_test(cases)

    report = {"status": "local_reference_metrics_not_official",
              "method": "existing partial_submission.json rescored, pipeline not re-run",
              "stage_root": str(args.stage_root),
              "cases": [c["case"] for c in cases],
              "variants": {}}

    for variant, subdir in (("off", "off"), ("on", "on")):
        preds = {c["case"]: json.loads(
            (args.stage_root / c["case"] / subdir / "partial_submission.json").read_text(encoding="utf-8"))
            for c in cases}
        report["variants"][variant] = {}
        for mode in TEXT_MODES:
            pairs = [(c["case"], preds[c["case"]], c["truth"], c["height"]) for c in cases]
            result = evaluate_cases(pairs, text_mode=mode)
            breakdown = {"ok": 0, "key_missing": 0, "type_mismatch": 0,
                         "bbox_far": 0, "text_mismatch": 0, "bbox_dev_over_20px": 0}
            for c in cases:
                part = component_breakdown(preds[c["case"]], c["truth"], text_mode=mode)
                for key in ("ok", "key_missing", "type_mismatch", "bbox_far",
                            "text_mismatch", "bbox_dev_over_20px"):
                    breakdown[key] += part.get(key, 0)
            report["variants"][variant][mode] = {
                "weighted_total": result["weighted_total"],
                "metrics": result["metrics"],
                "component_breakdown": breakdown,
                "per_case": [{k: pc[k] for k in ("case_id", "weighted_total")} |
                             {name: pc[name]["f1"] for name in METRICS}
                             for pc in result["per_case"]],
            }

    header = f"{'variant':>8}{'text_mode':>18}{'Component':>11}{'Pin':>8}{'NetH':>8}{'NetL':>8}{'weighted':>10}"
    print("\n" + header)
    print("-" * len(header))
    for variant in ("off", "on"):
        for mode in TEXT_MODES:
            entry = report["variants"][variant][mode]
            m = entry["metrics"]
            print(f"{variant:>8}{mode:>18}{m['component']['f1']:>11.4f}{m['pin']['f1']:>8.4f}"
                  f"{m['net_hypergraph']['f1']:>8.4f}{m['net_line']['f1']:>8.4f}"
                  f"{entry['weighted_total']:>10.4f}")

    print("\ncomponent failure breakdown (sum over cases):")
    for variant in ("off", "on"):
        for mode in TEXT_MODES:
            b = report["variants"][variant][mode]["component_breakdown"]
            print(f"  {variant:>3} {mode:>16}: ok {b['ok']:>3}  key_missing {b['key_missing']:>3}  "
                  f"type {b['type_mismatch']:>3}  bbox>20px {b['bbox_far']:>3}  "
                  f"text {b['text_mismatch']:>3}  (bbox dev >20px on key-matched pairs: "
                  f"{b['bbox_dev_over_20px']})")

    print("\nper case (designator_name variant, on):")
    for entry in report["variants"]["on"]["designator_name"]["per_case"]:
        print(f"  {entry['case_id']:>5}  C {entry['component']:.4f}  P {entry['pin']:.4f}  "
              f"N {entry['net_hypergraph']:.4f}  L {entry['net_line']:.4f}  "
              f"total {entry['weighted_total']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
