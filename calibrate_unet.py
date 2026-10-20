"""Re-calibrate the two decode thresholds for whatever U-Net is installed.

Both thresholds are fitted on ONE checkpoint's probability distribution, so shipping a
new checkpoint without re-scanning gives the gain away:

  13 released pages, A3_pin_l1:  wire 0.50 -> 67.22,  wire 0.35 -> 68.07
  13 released pages, A3_pin_l1:  pin  0.80 -> 68.07,  pin  0.60 -> 69.38

    python calibrate_unet.py                       # wire {0.35,0.50,0.65} then pin {0.80,0.70,0.60}
    python calibrate_unet.py --skip-pin
    python calibrate_unet.py --wire-thresholds 0.25 0.35 --pin-thresholds 0.6 0.55

Each value is one full pass over the input pages (~10 min for 13 pages).  The best pair
is written back into the live configs and the table lands in calibration_report.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
VENV = WORKSPACE / "pcb_parser_competition/.venv-gpu/Scripts/python.exe"
COMP = WORKSPACE / "pcb_parser_competition"
AGENT = Path(r"D:/pcb5_submission/agent.py")
SCORER = COMP / "scripts/evaluate_open_gt_platform.py"
WIRE_SETTER = WORKSPACE / "set_wire_threshold.py"
PIN_SETTER = WORKSPACE / "set_pin_threshold.py"

METRICS = ("component_f1", "pin_f1", "connectivity_f1", "wire_f1")


def run(argv):
    return subprocess.run([str(a) for a in argv], check=True, capture_output=True, text=True)


def measure(input_dir: Path, output: Path) -> dict:
    run([VENV, AGENT, "-input", input_dir, "-output", output])
    score = output / "score.json"
    run([VENV, SCORER, "--predictions", output / "final_results", "--output", score])
    report = json.loads(score.read_text(encoding="utf-8"))
    return {"total": round(report["final_score"] * 100, 2),
            **{k: round(report["metrics"][k]["f1"], 4) for k in METRICS}}


def sweep(label: str, values, setter: Path, extra, input_dir: Path, prefix: Path):
    rows = []
    for value in values:
        tag = f"{float(value):.2f}".replace(".", "")
        output = Path(f"{prefix}_{label}{tag}")
        print(f"[calibrate] {label}={value} -> {output}", flush=True)
        run([VENV, setter, value, *extra] if extra else [VENV, setter, value])
        row = {f"{label}_threshold": value, **measure(input_dir, output)}
        rows.append(row)
        print(f"[calibrate]   {row}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(r"D:/pcb_open_gt_in"))
    parser.add_argument("--wire-thresholds", type=float, nargs="+", default=[0.35, 0.5, 0.65])
    parser.add_argument("--pin-thresholds", type=float, nargs="+", default=[0.8, 0.7, 0.6])
    parser.add_argument("--pin-kernel", type=int, default=5)
    parser.add_argument("--skip-pin", action="store_true")
    parser.add_argument("--prefix", type=Path, default=Path("D:/pcb_open_gt_calib"))
    parser.add_argument("--report", type=Path, default=WORKSPACE / "calibration_report.json")
    args = parser.parse_args()

    wire_rows = sweep("wire", args.wire_thresholds, WIRE_SETTER, None, args.input, args.prefix)
    best_wire = max(wire_rows, key=lambda r: r["total"])
    run([VENV, WIRE_SETTER, best_wire["wire_threshold"]])
    print(f"[calibrate] wire best = {best_wire['wire_threshold']} ({best_wire['total']})", flush=True)

    pin_rows = []
    if not args.skip_pin:
        pin_rows = sweep("pin", args.pin_thresholds, PIN_SETTER, [args.pin_kernel],
                         args.input, args.prefix)
        best_pin = max(pin_rows, key=lambda r: r["total"])
        run([VENV, PIN_SETTER, best_pin["pin_threshold"], args.pin_kernel])
        print(f"[calibrate] pin best = {best_pin['pin_threshold']} ({best_pin['total']})", flush=True)

    summary = {"wire": wire_rows, "pin": pin_rows}
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {args.report}")


if __name__ == "__main__":
    main()
