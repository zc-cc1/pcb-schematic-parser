"""Net ceiling if pin naming were perfect.

Net identity in the competition is the set of `component.pinname` tokens (R08/R09).
This script recomputes net F1 with net identity given by the set of pin *positions*
instead, which is what the score would be if every pin name were correct.  It is a
ceiling, not an achievable number, and exists only to decide whether further work on
net construction or on pin naming is the better investment.

Run:  python scripts/oracle_net_ceiling.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 20.0


def resolve_positions(pins, hypergraph):
    """Map a hyperGraph member list onto pin positions."""
    out = []
    for token in str(hypergraph or "").strip("()").split(","):
        token = token.strip()
        if "." not in token:
            continue
        component, pin = token.rsplit(".", 1)
        section = pins.get(component) or {}
        entry = section.get(f"pin_{pin}") or section.get(pin)
        if entry is None:
            continue
        out.append((round(entry["point"]["x"], 1), round(entry["point"]["y"], 1)))
    return out


def snap(points, reference):
    """Snap our pin positions onto the nearest reference pin position.

    This is the coordinate half of "names were correct": two pins 20 px apart are
    genuinely distinct, so they must not be merged - they are resolved against the
    reference instead, exactly like an oracle pin-name lookup would.
    """
    snapped = []
    for point in points:
        best = None
        for other in reference:
            distance = ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2) ** 0.5
            if best is None or distance < best[0]:
                best = (distance, other)
        if best and best[0] <= TOLERANCE:
            snapped.append(best[1])
        else:
            snapped.append(("unmatched", point))
    return tuple(sorted(snapped, key=str))


def reference_positions(truth):
    out = []
    for pins in (truth.get("pins") or {}).values():
        for pin in pins.values():
            out.append((round(pin["point"]["x"], 1), round(pin["point"]["y"], 1)))
    return out


def main():
    tp = fp = fn = 0
    per_case = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission_path = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission_path.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        reference = reference_positions(truth)
        ours = set()
        for net in (submission.get("nets") or {}).values():
            points = resolve_positions(submission.get("pins") or {}, net["hyperGraph"])
            if len(points) >= 2:
                ours.add(snap(points, reference))
        theirs = set()
        for net in (truth.get("nets") or {}).values():
            points = resolve_positions(truth.get("pins") or {}, net["hyperGraph"])
            if len(points) >= 2:
                theirs.add(tuple(sorted(points, key=str)))
        matched = len(ours & theirs)
        tp += matched
        fp += len(ours) - matched
        fn += len(theirs) - matched
        per_case.append((case, len(ours), len(theirs), matched))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print("net identity by pin POSITION (upper bound: as if all pin names were correct)")
    print(f"  tp {tp}  fp {fp}  fn {fn}   P {precision:.4f}  R {recall:.4f}  F1 {f1:.4f}")
    print()
    print(f"  {'case':<7}{'ours':>6}{'ref':>6}{'exact':>7}")
    for case, ours_count, truth_count, matched_count in per_case:
        print(f"  {case:<7}{ours_count:>6}{truth_count:>6}{matched_count:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
