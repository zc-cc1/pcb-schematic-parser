"""Score the delivered pipeline against the 10 public Golden cases, locally.

Runs the real chain (ownership -> pin text -> wire graph) on each Golden PNG, then
scores the strict export with the local reference implementation of the contest
metrics and, separately, the *internal* graph (all pins, ids ignored) to show how
much of the gap is identity resolution versus connectivity.

Two numbers, deliberately:
  strict_export   exactly what could be uploaded today
  internal_graph  what the pipeline actually knows, with pin identity bypassed for
                  diagnosis only - this number must never be presented as a score

`--unet-config` selects the weight set; the detector stays frozen.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.contest_metrics import evaluate_cases  # noqa: E402
from pipeline_inputs import write_wire_cache_metadata  # noqa: E402

GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
GRAPH_CONFIG = ROOT / "configs/wire_graph_v2.json"


def run(command, label):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-1500:])
        print(result.stderr[-2000:])
        raise SystemExit(f"stage failed: {label}")
    return result


def chain(case_id, out_root, unet_config, unet_weights, device):
    image = ROOT / f"data/standardized/{case_id}/{case_id}.png"
    out = out_root / case_id
    ownership, text_out, graph_out = out / "ownership", out / "pin_text", out / "wire_graph"
    run([sys.executable, "scripts/run_pin_ownership.py", "--image", str(image), "--output", str(ownership),
         "--device", device, "--unet-inference-config", str(unet_config), "--unet-weights", str(unet_weights)],
        f"{case_id} ownership")
    write_wire_cache_metadata(ownership, unet_config, source="golden_eval_driver")
    run([sys.executable, "scripts/run_pin_text.py", "--image", str(image), "--output", str(text_out),
         "--ownership", str(ownership / "pin_ownership.json"), "--device", device], f"{case_id} pin_text")
    run([sys.executable, "scripts/run_wire_graph.py", "--image", str(image), "--output", str(graph_out),
         "--pin-text", str(text_out / "pin_text_result.json"), "--wire-cache", str(ownership / "probabilities.npz"),
         "--config", str(GRAPH_CONFIG),
         "--unet-inference-config", str(unet_config), "--device", device], f"{case_id} wire_graph")
    return out


def internal_export(out):
    """Everything the graph knows, with unresolved identity replaced by placeholders.

    Diagnosis only: this is NOT a legal submission (pin names are invented, keys are
    detection ids) and its score must never be labelled as a contest result.  It
    exists to separate connectivity quality from identity resolution, so components
    keep their real bboxes and types and the component/pin matching can still run.
    """
    graph = json.loads((out / "wire_graph" / "graph_proposals.json").read_text(encoding="utf-8"))
    pin_result = json.loads((out / "pin_text" / "pin_text_result.json").read_text(encoding="utf-8"))
    height = pin_result["size"][1]
    components, pins, nets = {}, {}, {}
    owner_key = {}
    for component in pin_result.get("components") or []:
        # diagnosis keys must be unique: a detector may emit two boxes for one
        # designator, and collapsing them would hide detection recall
        key = f"DET_{component.get('detection_index', 0):04d}"
        owner_key[component.get("component_id")] = key
        owner_key[component.get("detection_index")] = key
        x1, y1, x2, y2 = component["bbox_top_left"]
        # Despite the field name, `bbox_top_left` holds bottom-left-origin numbers:
        # flipping it with H-y reproduces the box the wire stage exports and places
        # resistor R5 of 0025 at the reference location.  Pin points are already in
        # top-left pixel coordinates, so they are NOT flipped.
        components[key] = {"type": component.get("type") or "other",
                           "Name": component.get("designator") or component.get("Name"),
                           "value": component.get("value"),
                           "bbox": [float(x1), float(height - y2), float(x2), float(height - y1)]}
    by_candidate = {}
    auto_numbers = {}
    from pcb_parser.pin_numbering import assign_numbers
    auto_numbers, _audit = assign_numbers(pin_result["pin_candidates"], pin_result.get("components") or [])
    for pin in pin_result["pin_candidates"]:
        if pin["owner_status"] != "assigned":
            continue
        owner = owner_key.get(pin.get("owner_detection_index")) or owner_key.get(pin.get("owner"))
        if owner is None:
            continue
        number = pin.get("number") or auto_numbers.get(pin["candidate_id"]) or "?"
        key = f"pin_{number}"
        if key in pins.setdefault(owner, {}):
            continue
        x, y = pin["point"]
        pins[owner][key] = {"pinname": pin.get("pinname") or "",
                            "point": {"x": x, "y": y}}
        by_candidate[pin["candidate_id"]] = f"{owner}.{number}"
    edge_lookup = {e["edge_id"]: e for e in graph["edges"]}
    for net in graph["networks"]:
        members = sorted({by_candidate[c] for c in net["pin_candidate_ids"] if c in by_candidate})
        if len(members) < 2:
            continue
        edges, index = {}, 0
        for edge_id in net["edge_ids"]:
            points = edge_lookup.get(edge_id, {}).get("polyline") or []
            for a, b in zip(points, points[1:]):
                index += 1
                edges[f"edge_{index}"] = [{"x": a[0], "y": height - a[1]}, {"x": b[0], "y": height - b[1]}]
        nets[f"net_{len(nets) + 1}"] = {"hyperGraph": "(" + ",".join(members) + ")", "edges": edges}
    return {"components": components, "pins": pins, "nets": nets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet-config", type=Path, default=ROOT / "configs/unet_inference_v3.json")
    parser.add_argument("--unet-weights", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A3_split_w2/best.pt"))
    parser.add_argument("--root", type=Path, default=Path(r"D:/pcb_unet_v2/golden_eval"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/golden_eval.json"))
    args = parser.parse_args()

    pairs_strict, pairs_internal, inventory = [], [], []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case_id = folder.name.split("-")[0]
        target = next(folder.glob("*_target.json"))
        truth = json.loads(target.read_text(encoding="utf-8"))
        from PIL import Image
        with Image.open(next(folder.glob("*.png"))) as rendered:
            height = rendered.height
        out = args.root / case_id
        if not args.reuse or not (out / "wire_graph" / "graph_proposals.json").exists():
            out = chain(case_id, args.root, args.unet_config, args.unet_weights, args.device)
        strict_path = out / "wire_graph" / "partial_submission.json"
        coverage_path = out / "wire_graph" / "export_coverage.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8")) if strict_path.exists() else \
            {"components": {}, "pins": {}, "nets": {}}
        coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
        pairs_strict.append((case_id, strict, truth, height))
        pairs_internal.append((case_id, internal_export(out), truth, height))
        inventory.append({"case_id": case_id,
                          "truth_components": len(truth["components"]), "truth_pins": sum(len(v) for v in truth["pins"].values()),
                          "truth_nets": len(truth["nets"]),
                          "exported_components": coverage.get("exported_components", 0),
                          "exported_pins": coverage.get("exported_pins", 0),
                          "exported_nets": coverage.get("exported_nets", 0)})
        print(f"  {case_id}: truth {len(truth['components'])}c/{len(truth['nets'])}n -> "
              f"exported {coverage.get('exported_components', 0)}c/{coverage.get('exported_pins', 0)}p/"
              f"{coverage.get('exported_nets', 0)}n", flush=True)

    strict_report = evaluate_cases(pairs_strict)
    internal_report = evaluate_cases(pairs_internal)
    report = {"status": "local_golden_evaluation_not_official",
              "unet_config": str(args.unet_config), "unet_weights": str(args.unet_weights),
              "inventory": inventory,
              "strict_export": strict_report,
              "internal_graph_diagnostic": internal_report,
              "warning": "neither number is the official ComponentF1 / PinF1 / NetHypergraphF1; "
                         "internal_graph invents pin names and is diagnosis only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, block in (("strict export (what could be uploaded today)", strict_report),
                         ("internal graph (identity bypassed, diagnosis only)", internal_report)):
        print(f"\n== {label} ==")
        for key in ("component", "pin", "net_hypergraph", "net_line"):
            value = block["metrics"][key]
            print(f"  {key:<15} P {value['precision']:.4f} R {value['recall']:.4f} F1 {value['f1']:.4f}"
                  f"  (tp {value['tp']} fp {value['fp']} fn {value['fn']})")
        print(f"  weighted total {block['weighted_total']:.4f}")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
