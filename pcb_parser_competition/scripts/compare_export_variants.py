"""Compare export variants on the 10 public Golden pages under several reading modes.

Each variant is a directory of `<case>/partial_submission.json` (plus export_coverage.json),
so no pipeline stage is re-run here: this prices the *scoring* rules and the *export*
policy separately instead of mixing them into one number.

Modes, from the narrowest reading of the PDF to the one the Q&A actually describes:

    strict      key match only, box Name compared, all pins, net pin numbers exact
    gt10_text   strict + do not text-check a box whose GT Name is only its designator
    anon_pos    gt10_text + '∅' power/ground/label symbols matched by coordinate
    box_pins    anon_pos + PinF1 counts box/ic pins only
    official    box_pins + non-box net members compared without their pin number

Every number is the local reference implementation, not the platform score.

    python scripts/compare_export_variants.py \
        --variant baseline=D:/pcb_unet_v2/box_library_ab/<case>/on \
        --variant autoid=D:/pcb_unet_v2/autoid_ab
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.contest_metrics import evaluate_cases  # noqa: E402

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
METRICS = ("component", "pin", "net_hypergraph", "net_line")

MODES = {
    "strict":    dict(text_mode="strict", anonymous_components="key",
                      pin_scope="all", net_pin_identity="number"),
    "gt10_text": dict(text_mode="designator_name", anonymous_components="key",
                      pin_scope="all", net_pin_identity="number"),
    "anon_pos":  dict(text_mode="designator_name", anonymous_components="by_position",
                      pin_scope="all", net_pin_identity="number"),
    "box_pins":  dict(text_mode="designator_name", anonymous_components="by_position",
                      pin_scope="box_only", net_pin_identity="number"),
    "anon_pins": dict(text_mode="designator_name", anonymous_components="by_position",
                      pin_scope="all", net_pin_identity="number", nonbox_pin_mode="position"),
    "official":  dict(text_mode="designator_name", anonymous_components="by_position",
                      pin_scope="box_only", net_pin_identity="coarse"),
    "official_pins": dict(text_mode="designator_name", anonymous_components="by_position",
                          pin_scope="all", net_pin_identity="coarse",
                          nonbox_pin_mode="position"),
}
MODE_ORDER = list(MODES)


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
        cases.append({
            "case": case,
            "truth": json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8")),
            "height": image_height(ROOT / f"data/standardized/{case}/{case}.png"),
        })
    return cases


def read_variant(spec: str, cases) -> tuple[str, dict]:
    name, _, raw = spec.partition("=")
    if not raw:
        raise SystemExit(f"--variant wants name=path, got {spec!r}")
    path = Path(raw)
    if "<case>" in raw:
        subs = {}
        for case in cases:
            case_path = Path(raw.replace("<case>", case["case"])) / "partial_submission.json"
            if not case_path.is_file():
                raise SystemExit(f"missing prediction: {case_path}")
            subs[case["case"]] = json.loads(case_path.read_text(encoding="utf-8"))
        return name, subs
    subs = {}
    for case in cases:
        case_dir = path / case["case"]
        candidates = [path / f"{case['case']}_Pred.json", path / f"{case['case']}.json"]
        if case_dir.is_dir():
            candidates.insert(0, case_dir / "partial_submission.json")
        for candidate in candidates:
            if candidate.is_file():
                subs[case["case"]] = json.loads(candidate.read_text(encoding="utf-8"))
                break
        else:
            subs[case["case"]] = {"components": {}, "pins": {}, "nets": {}}
    return name, subs


def self_test(cases) -> None:
    print("self-test (each mode, truth against itself) must be 1.0:")
    for mode, kwargs in MODES.items():
        pairs = [(c["case"], copy.deepcopy(c["truth"]), copy.deepcopy(c["truth"]), c["height"])
                 for c in cases]
        report = evaluate_cases(pairs, **kwargs)
        bad = {k: v["f1"] for k, v in report["metrics"].items() if v["f1"] < 0.999}
        print(f"  {mode:<10} {'OK' if not bad else 'FAILED ' + json.dumps(bad)}")
        if bad:
            raise SystemExit("self-test failed; refusing to report scores")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True,
                        help="name=path, '<case>' expands to the case id")
    parser.add_argument("--output", type=Path,
                        default=Path(r"D:/pcb_unet_v2/reports/export_variant_matrix.json"))
    args = parser.parse_args()

    cases = load_cases()
    self_test(cases)
    variants = dict(read_variant(spec, cases) for spec in args.variant)

    for name, subs in variants.items():
        comps = sum(len(p.get("components") or {}) for p in subs.values())
        pins = sum(sum(len(v) for v in (p.get("pins") or {}).values()) for p in subs.values())
        nets = sum(len(p.get("nets") or {}) for p in subs.values())
        print(f"\nvariant {name}: exported {comps} components / {pins} pins / {nets} nets")

    table = {}
    print(f"\n{'variant':>10}{'mode':>11}{'Component':>11}{'Pin':>8}{'NetH':>8}{'NetL':>8}{'weighted':>10}")
    print("-" * 66)
    for name, subs in variants.items():
        table[name] = {}
        for mode in MODE_ORDER:
            pairs = [(c["case"], subs[c["case"]], c["truth"], c["height"]) for c in cases]
            report = evaluate_cases(pairs, **MODES[mode])
            m = report["metrics"]
            table[name][mode] = {
                "weighted_total": report["weighted_total"],
                "metrics": m,
                "per_case": [{k: pc[k] for k in ("case_id", "weighted_total")} |
                             {key: pc[key]["f1"] for key in METRICS}
                             for pc in report["per_case"]],
            }
            print(f"{name:>10}{mode:>11}{m['component']['f1']:>11.4f}{m['pin']['f1']:>8.4f}"
                  f"{m['net_hypergraph']['f1']:>8.4f}{m['net_line']['f1']:>8.4f}"
                  f"{report['weighted_total']:>10.4f}")

    best = max(variants, key=lambda v: table[v]["official"]["weighted_total"])
    print(f"\nbest under the 'official' reading: {best} "
          f"({table[best]['official']['weighted_total']:.4f})")

    print(f"\nper case, variant={best}, mode=official:")
    for entry in table[best]["official"]["per_case"]:
        print(f"  {entry['case_id']:>5}  C {entry['component']:.3f}  P {entry['pin']:.3f}  "
              f"N {entry['net_hypergraph']:.3f}  L {entry['net_line']:.3f}  "
              f"total {entry['weighted_total']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "local_reference_metrics_not_official",
        "modes": {name: kwargs for name, kwargs in MODES.items()},
        "variants": {name: sorted(subs) for name, subs in variants.items()},
        "table": table,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
