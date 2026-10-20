"""A/B the component library on the 10 public Golden cases.

Runs only the wire-graph stage twice per case - frozen pin-text predictions, frozen
U-Net wire cache, no GPU, no model inference - once without the library and once with
it, then scores both with the local reference metrics.

Both numbers are local reference scores, not official ones.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.contest_metrics import evaluate_cases  # noqa: E402

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
CACHE = Path(r"D:/pcb_unet_v2/golden_eval")
GRAPH_CONFIG = ROOT / "configs/wire_graph_v2.json"
UNET_CONFIG = ROOT / "configs/unet_inference_v3.json"


def image_height(path):
    from PIL import Image
    with Image.open(path) as handle:
        return float(handle.height)


def run_stage(case, out, library_config, graph_config):
    command = [sys.executable, "scripts/run_wire_graph.py",
               "--image", str(ROOT / f"data/standardized/{case}/{case}.png"),
               "--output", str(out),
               "--pin-text", str(CACHE / case / "pin_text/pin_text_result.json"),
               "--wire-cache", str(CACHE / case / "ownership/probabilities.npz"),
               "--config", str(graph_config),
               "--unet-inference-config", str(UNET_CONFIG),
               "--device", "cpu"]
    if library_config is not None:
        command += ["--box-library-config", str(library_config)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"stage failed for {case}")
    return out


def load(case, out):
    submission = json.loads((out / "partial_submission.json").read_text(encoding="utf-8"))
    coverage = json.loads((out / "export_coverage.json").read_text(encoding="utf-8"))
    return submission, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-config", type=Path, default=ROOT / "configs/box_ic_library_v1.json")
    parser.add_argument("--root", type=Path, default=Path(r"D:/pcb_unet_v2/box_library_ab"))
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/box_library_ab.json"))
    parser.add_argument("--cases", default="", help="comma separated case ids; empty means all")
    parser.add_argument("--graph-config", type=Path, default=GRAPH_CONFIG)
    args = parser.parse_args()
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}

    pairs_off, pairs_on, inventory = [], [], []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        if wanted and case not in wanted:
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        height = image_height(ROOT / f"data/standardized/{case}/{case}.png")

        off_dir = args.root / case / "off"
        on_dir = args.root / case / "on"
        run_stage(case, off_dir, None, args.graph_config)
        run_stage(case, on_dir, args.library_config, args.graph_config)
        off_pred, off_cov = load(case, off_dir)
        on_pred, on_cov = load(case, on_dir)
        pairs_off.append((case, off_pred, truth, height))
        pairs_on.append((case, on_pred, truth, height))

        audit = (on_cov.get("pin_numbering") or {}).get("box_library") or {}
        inventory.append({
            "case": case,
            "exported_pins_off": off_cov.get("exported_pins", 0),
            "exported_pins_on": on_cov.get("exported_pins", 0),
            "exported_nets_off": off_cov.get("exported_nets", 0),
            "exported_nets_on": on_cov.get("exported_nets", 0),
            "models": audit.get("models") or [],
            "pins_numbered": audit.get("pins_numbered_from_name", 0),
            "pins_named": audit.get("pins_named_from_library", 0),
            "entries_rejected": audit.get("entries_rejected_by_consistency", 0),
            "changes": audit.get("changes") or [],
        })
        model_text = ", ".join(
            f"{m['model']}(names={m.get('name_matches', 0)} nums={m.get('number_matches', 0)} "
            f"consistency={m.get('consistency')} accepted={m.get('accepted')})"
            for m in inventory[-1]["models"]) or "-"
        print(f"  {case}: pins {inventory[-1]['exported_pins_off']}->{inventory[-1]['exported_pins_on']}, "
              f"nets {inventory[-1]['exported_nets_off']}->{inventory[-1]['exported_nets_on']}, "
              f"numbered {inventory[-1]['pins_numbered']}, named {inventory[-1]['pins_named']}, rejected {inventory[-1]['entries_rejected']}, "
              f"models: {model_text}", flush=True)

    off_report = evaluate_cases(pairs_off)
    on_report = evaluate_cases(pairs_on)
    report = {"status": "local_reference_metrics_not_official",
              "library_config": str(args.library_config),
              "off": off_report, "on": on_report, "inventory": inventory}

    print("\n== without library ==")
    for key in ("component", "pin", "net_hypergraph", "net_line"):
        value = off_report["metrics"][key]
        print(f"  {key:<15} P {value['precision']:.4f} R {value['recall']:.4f} F1 {value['f1']:.4f} "
              f"(tp {value['tp']} fp {value['fp']} fn {value['fn']})")
    print(f"  weighted {off_report['weighted_total']:.4f}")
    print("\n== with library ==")
    for key in ("component", "pin", "net_hypergraph", "net_line"):
        value = on_report["metrics"][key]
        print(f"  {key:<15} P {value['precision']:.4f} R {value['recall']:.4f} F1 {value['f1']:.4f} "
              f"(tp {value['tp']} fp {value['fp']} fn {value['fn']})")
    print(f"  weighted {on_report['weighted_total']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
