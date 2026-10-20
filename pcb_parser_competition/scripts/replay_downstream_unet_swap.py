"""Replay the downstream chain on frozen inputs, swapping ONLY the U-Net.

Purpose: the pin head improved a lot (A2/A3), but the delivered pipeline still
loads `unet_wire_pin_v1`.  This driver re-runs ownership -> pin text -> wire graph
on the same frozen inputs (detector cache, OCR texts, pin-text cache) with two
U-Net configurations and reports the downstream pin/connectivity metrics side by
side.  Every stage keeps its own hash contract, so a mismatch fails loudly instead
of silently mixing weights.

It writes only under --root; no project data, weights or configs are modified.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ["0002", "0018", "0022", "0026", "0033", "0042", "0049", "0096", "0105", "0170", "0240", "0286"]


def run(cmd, label):
    print(f"  [{label}] " + " ".join(str(part) for part in cmd[1:]), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"stage failed: {label}")
    return result


def write_wire_metadata(ownership_dir, unet_config, image_path):
    """Emit the hash-bound metadata that run_wire_graph.py expects next to a cache."""
    import hashlib
    summary = json.loads((ownership_dir / "summary.json").read_text(encoding="utf-8"))
    probabilities = ownership_dir / "probabilities.npz"
    metadata = {
        "image_sha256": summary["image_sha256"],
        "size": summary["size"],
        "coordinate_origin": "top_left",
        "weights_sha256": summary["unet_weights_sha256"],
        "inference_config_sha256": hashlib.sha256(Path(unet_config).read_bytes()).hexdigest(),
        "probabilities_sha256": hashlib.sha256(probabilities.read_bytes()).hexdigest(),
        "prediction_only": True,
        "source": "downstream_replay_frozen_inputs",
        "unet_model_variant": summary.get("unet_model_variant"),
    }
    target = probabilities.with_suffix(".json")
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def chain(case_id, root, unet_config, unet_weights, device):
    image = ROOT / f"data/standardized/{case_id}/{case_id}.png"
    detections = ROOT / f"runs/s5_pin_ownership_v1/{case_id}/detector_predictions.json"
    records = ROOT / f"runs/s6_pin_text_v1/predictions/{case_id}/component_records.json"
    texts = ROOT / f"runs/s6_pin_text_v1/predictions/{case_id}/texts.json"
    out = root / case_id
    ownership = out / "ownership"
    text_out = out / "pin_text"
    graph_out = out / "wire_graph"
    run([sys.executable, "scripts/run_pin_ownership.py", "--image", str(image), "--output", str(ownership),
         "--device", device, "--detections", str(detections), "--component-records", str(records),
         "--unet-inference-config", str(unet_config), "--unet-weights", str(unet_weights)],
        f"{case_id} ownership")
    write_wire_metadata(ownership, unet_config, image)
    run([sys.executable, "scripts/run_pin_text.py", "--image", str(image), "--output", str(text_out),
         "--ownership", str(ownership / "pin_ownership.json"), "--texts", str(texts), "--device", device],
        f"{case_id} pin_text")
    run([sys.executable, "scripts/run_wire_graph.py", "--image", str(image), "--output", str(graph_out),
         "--pin-text", str(text_out / "pin_text_result.json"), "--wire-cache", str(ownership / "probabilities.npz"),
         "--unet-inference-config", str(unet_config),
         "--device", device],
        f"{case_id} wire_graph")
    graph_summary = json.loads((graph_out / "summary.json").read_text(encoding="utf-8"))
    proposals = json.loads((graph_out / "graph_proposals.json").read_text(encoding="utf-8"))
    pin_text = json.loads((text_out / "pin_text_result.json").read_text(encoding="utf-8"))
    candidates = pin_text.get("pin_candidates") or []
    return {"case_id": case_id,
            "ownership": json.loads((ownership / "pin_ownership.json").read_text(encoding="utf-8"))["summary"],
            "graph": graph_summary,
            "graph_detail": proposals["summary"],
            "pin_text": {"candidate_count": len(candidates),
                         "numbers_bound": sum(1 for c in candidates if c.get("number") is not None),
                         "names_bound": sum(1 for c in candidates if (c.get("pinname") or "").strip()),
                         "assigned": sum(1 for c in candidates if c.get("owner_status") == "assigned")}}


def summarise(rows):
    def total(key, source="graph_detail"):
        return sum((r[source].get(key) or 0) for r in rows)
    pins = total("candidate_count", "ownership")
    assigned = total("assigned", "ownership")
    return {
        "cases": len(rows),
        "pin_candidates": pins,
        "assigned": assigned,
        "assignment_rate": round(assigned / pins, 4) if pins else None,
        "pin_numbers_bound": total("numbers_bound", "pin_text"),
        "pin_names_bound": total("names_bound", "pin_text"),
        "graph_nodes": total("graph_nodes"),
        "graph_edges": total("graph_edges"),
        "attached_pins": total("attached_pins"),
        "network_proposals": total("network_proposals"),
        "multi_pin_networks": total("multi_pin_network_proposals"),
        "uncertain_junctions": total("uncertain_junctions"),
        "gap_candidates": total("gap_candidates_not_filled"),
        "exported_pins": sum(r["graph"].get("exported_pins", 0) for r in rows),
        "exported_nets": sum(r["graph"].get("exported_nets", 0) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:/pcb_unet_v2/downstream_replay"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()

    configs = {
        "old_v1_unet": (ROOT / "configs/unet_inference_v2.json", ROOT / "runs/training/unet_wire_pin_v1/best.pt"),
        "new_a3_split_w2": (ROOT / "configs/unet_inference_v3.json", Path(r"D:/pcb_unet_v2/runs/A3_split_w2/best.pt")),
    }
    report = {"status": "downstream_replay", "cases": args.cases, "runs": {}}
    for label, (config, weights) in configs.items():
        print(f"\n== {label} ==", flush=True)
        if not weights.exists():
            print(f"  missing weights {weights}")
            continue
        root = args.root / label
        rows = []
        for case_id in args.cases:
            try:
                rows.append(chain(case_id, root, config, weights, args.device))
            except SystemExit as error:
                print(f"  {case_id}: {error}")
        report["runs"][label] = {"config": str(config), "weights": str(weights),
                                 "aggregate": summarise(rows), "cases": rows}
        print(json.dumps(report["runs"][label]["aggregate"], ensure_ascii=False))

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "replay_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== aggregate comparison ==")
    for label, run_report in report["runs"].items():
        agg = run_report["aggregate"]
        print(f"{label:<18} pins {agg['pin_candidates']:>5}  assigned {agg['assigned']:>5} "
              f"({agg['assignment_rate']})  exported pins {agg['exported_pins']:>4}  nets {agg['exported_nets']}")
    print("written:", args.root / "replay_summary.json")


if __name__ == "__main__":
    main()
