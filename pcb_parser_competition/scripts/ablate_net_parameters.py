"""Which parameter actually moves the net item?

Re-runs only the wire-graph stage on the cached ownership and pin-text outputs of the
10 public Golden cases, so each variant costs seconds instead of a full chain run.
Reports NetHypergraphF1 / NetLineF1 and the weighted total for every variant.

Run:  python scripts/ablate_net_parameters.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.contest_metrics import evaluate_cases  # noqa: E402
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
WORK = Path(r"D:/pcb_unet_v2/net_ablation")
BASE_CONFIG = ROOT / "configs" / "wire_graph_v2.json"

VARIANTS = {
    "baseline (current config)": {},
    "wire_threshold 0.35": {"wire_threshold": 0.35},
    "wire_threshold 0.65": {"wire_threshold": 0.65},
    "pin_attach_radius 4": {"pin_attach_radius_px": 4},
    "pin_attach_radius 9": {"pin_attach_radius_px": 9},
    "pin_protection_radius 4": {"pin_protection_radius_px": 4},
    "body_guard off": {"body_guard": False},
    "single-pin gate off": {"export_gates": {"single_pin": False}},
    "junction gate off": {"export_gates": {"junction": False}},
    "disconnection gate off": {"export_gates": {"disconnection": False}},
    "all confidence gates off": {"export_gates": {"single_pin": False, "junction": False,
                                                 "disconnection": False}},
}


def cases():
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        cached = BASE / case
        if not (cached / "ownership" / "probabilities.npz").exists():
            continue
        if not (cached / "pin_text" / "pin_text_result.json").exists():
            continue
        yield case, folder, cached


def run_variant(name, overrides):
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    patches = dict(overrides)
    config.update(patches)
    out_dir = WORK / name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    pairs = []
    from PIL import Image
    for case, folder, cached in cases():
        target = out_dir / case
        command = [sys.executable, "scripts/run_wire_graph.py",
                   "--image", str(ROOT / f"data/standardized/{case}/{case}.png"),
                   "--output", str(target),
                   "--pin-text", str(cached / "pin_text" / "pin_text_result.json"),
                   "--wire-cache", str(cached / "ownership" / "probabilities.npz"),
                   "--config", str(config_path),
                   "--unet-inference-config", str(ROOT / "configs" / "unet_inference_v3.json"),
                   "--device", "cuda:0"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {case}: stage failed")
            print(result.stderr[-600:])
            return None
        submission_path = target / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        with Image.open(next(folder.glob("*.png"))) as image:
            height = image.height
        pairs.append((case, json.loads(submission_path.read_text(encoding="utf-8")), truth, height))
    if not pairs:
        return None
    return evaluate_cases(pairs)


def main():
    header = f"{'variant':<30}{'net F1':>9}{'line F1':>9}{'weighted':>10}"
    print(header)
    print("-" * len(header))
    for name, overrides in VARIANTS.items():
        report = run_variant(name, overrides)
        if report is None:
            print(f"{name:<30}{'failed':>9}")
            continue
        nets = report["metrics"]["net_hypergraph"]
        lines = report["metrics"]["net_line"]
        print(f"{name:<30}{nets['f1']:>9.4f}{lines['f1']:>9.4f}{report['weighted_total']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
