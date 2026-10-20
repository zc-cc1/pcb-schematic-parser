"""Pick a checkpoint epoch from training history without chasing validation noise.

The 59-page validation set swings several points between epochs (wire F1 moved
0.681 <-> 0.824 inside five epochs), so `argmax(composite)` is a noise pick.
This tool smooths the curve, reports the candidate epochs under several rules,
and prints the exact re-run command that reproduces a candidate checkpoints.

It writes nothing except the report; training state is untouched.
"""
import argparse
import json
from pathlib import Path

DEFAULT_HISTORY = Path(r"D:/pcb_unet_v2/runs/A2_full60/history.json")
DEFAULT_OUT = Path(r"D:/pcb_unet_v2/reports/checkpoint_selection.json")


def ema(values, alpha):
    smoothed, previous = [], None
    for value in values:
        previous = value if previous is None else alpha * value + (1 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def rolling(values, window):
    out = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        out.append(sum(values[start:index + 1]) / (index + 1 - start))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-epoch", type=int, default=10, help="ignore the warm-up epochs")
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.2, 0.35])
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    epochs = [r["epoch"] for r in history]
    wire = [r["wire_pixel_f1"] for r in history]
    pin = [r["pin_point_f1"] for r in history]
    composite = [r["composite"] for r in history]

    candidates = {}
    for alpha in args.alphas:
        for key, series in (("composite", composite), ("wire", wire), ("pin", pin)):
            smoothed = ema(series, alpha)
            eligible = [(value, epoch) for value, epoch in zip(smoothed, epochs) if epoch >= args.min_epoch]
            if eligible:
                value, epoch = max(eligible)
                candidates[f"ema{alpha}_{key}"] = {"epoch": epoch, "smoothed": round(value, 4)}
    for window in (3, 5, 9):
        for key, series in (("composite", composite), ("wire", wire), ("pin", pin)):
            smoothed = rolling(series, window)
            eligible = [(value, epoch) for value, epoch in zip(smoothed, epochs) if epoch >= args.min_epoch]
            if eligible:
                value, epoch = max(eligible)
                candidates[f"rolling{window}_{key}"] = {"epoch": epoch, "smoothed": round(value, 4)}

    raw_best = max(range(len(composite)), key=lambda i: composite[i])
    summary = {
        "status": "offline_checkpoint_selection",
        "history": str(args.history),
        "epochs_recorded": len(history),
        "raw_best_by_composite": {"epoch": epochs[raw_best], "composite": composite[raw_best],
                                  "wire": wire[raw_best], "pin": pin[raw_best]},
        "validation_noise": {
            "wire_f1_min": round(min(wire), 4), "wire_f1_max": round(max(wire), 4),
            "wire_f1_swing": round(max(wire) - min(wire), 4),
            "pin_f1_min": round(min(pin), 4), "pin_f1_max": round(max(pin), 4),
            "pin_f1_swing": round(max(pin) - min(pin), 4),
        },
        "candidates": candidates,
        "last_epoch": {"epoch": epochs[-1], "composite": composite[-1], "wire": wire[-1], "pin": pin[-1]},
        "series": [{"epoch": e, "wire": w, "pin": p, "composite": c} for e, w, p, c in zip(epochs, wire, pin, composite)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"epochs recorded: {len(history)}")
    print(f"validation swing: wire {min(wire):.4f}..{max(wire):.4f} ({max(wire)-min(wire):.4f}), "
          f"pin {min(pin):.4f}..{max(pin):.4f} ({max(pin)-min(pin):.4f})")
    print(f"raw best by composite: epoch {epochs[raw_best]} -> composite {composite[raw_best]:.4f} "
          f"(wire {wire[raw_best]:.4f}, pin {pin[raw_best]:.4f})")
    print("\ncandidate checkpoints (smoothed):")
    for name, value in sorted(candidates.items(), key=lambda kv: -kv[1]["smoothed"]):
        print(f"  {name:<22} epoch {value['epoch']:>3}  smoothed {value['smoothed']:.4f}")
    print("\nto reproduce a candidate, re-run training up to that epoch and keep last.pt:")
    if candidates:
        pick = candidates[getattr(args, "rule", "ema0.35_composite")] if getattr(args, "rule", None) else None
        target = pick["epoch"] if pick else sorted(candidates.values(), key=lambda v: -v["smoothed"])[0]["epoch"]
        print(f"  --epochs {target} --output D:/pcb_unet_v2/runs/<name>")
    print("written:", args.output)


if __name__ == "__main__":
    main()
