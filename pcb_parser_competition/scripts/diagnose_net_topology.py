"""Separate graph errors from pin-identity errors in the net item.

Every emitted net is compared against the reference nets by pin *position*, so pin
name problems cannot masquerade as topology problems.  For each emitted net the
closest reference net is classified as:
  exact     - same pin set
  subset    - our pins are a proper subset (we under-connected)
  superset  - our pins are a proper superset (we over-connected)
  partial   - overlaps but neither contains the other
  disjoint  - no shared pin position
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")
BASE = Path(r"D:/pcb_unet_v2/golden_eval")
TOLERANCE = 20.0


def points_of(pin_section, net):
    """Resolve a net's member tokens to the positions of the referenced pins."""
    out = []
    text = str(net.get("hyperGraph") or "")
    for token in text.strip("()").split(","):
        token = token.strip()
        if "." not in token:
            continue
        component, pin = token.rsplit(".", 1)
        entry = (pin_section.get(component) or {}).get(f"pin_{pin}")
        if entry is None:
            # tolerate pin keys that already carry the prefix
            entry = (pin_section.get(component) or {}).get(pin)
        if entry is None:
            continue
        out.append((round(entry["point"]["x"], 1), round(entry["point"]["y"], 1)))
    return out


def shared_count(ours, theirs):
    """How many of our pin positions coincide with theirs, within tolerance."""
    remaining = list(theirs)
    shared = 0
    for point in ours:
        best = None
        for index, other in enumerate(remaining):
            distance = ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2) ** 0.5
            if best is None or distance < best[0]:
                best = (distance, index)
        if best and best[0] <= TOLERANCE:
            shared += 1
            remaining.pop(best[1])
    return shared


def classify(ours, theirs):
    if not ours or not theirs:
        return "empty"
    shared = shared_count(ours, theirs)
    if not shared:
        return "disjoint"
    if shared == len(ours) == len(theirs):
        return "exact"
    if shared == len(ours):
        return "subset"
    if shared == len(theirs):
        return "superset"
    return "partial"


def main():
    verdicts = collections.Counter()
    examples = collections.defaultdict(list)
    for folder in sorted(GT.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        submission = BASE / case / "wire_graph" / "partial_submission.json"
        if not submission.exists():
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        pred = json.loads(submission.read_text(encoding="utf-8"))
        truth_nets = [(net, points_of(truth.get("pins") or {}, net))
                      for net in (truth.get("nets") or {}).values()]
        for net in (pred.get("nets") or {}).values():
            ours = points_of(pred.get("pins") or {}, net)
            if len(ours) < 2:
                verdicts["too_few_pins"] += 1
                continue
            best = None
            for reference, reference_points in truth_nets:
                shared = shared_count(ours, reference_points)
                if best is None or shared > best[0]:
                    best = (shared, reference_points)
            if best is None:
                verdicts["no_reference_nets"] += 1
                continue
            verdict = classify(ours, best[1])
            verdicts[verdict] += 1
            if len(examples[verdict]) < 2:
                examples[verdict].append((case, len(ours), len(best[1])))
    total = sum(verdicts.values())
    print(f"emitted nets analysed: {total}")
    for verdict, count in verdicts.most_common():
        share = 100 * count / max(1, total)
        print(f"  {verdict:<16}{count:>4}  {share:5.1f}%   examples {examples[verdict][:2]}")
    print()
    print("subset    = we under-connect (missing a pin)")
    print("superset  = we over-connect (merged a net with something else)")
    print("partial   = both wrong in different places")
    print("disjoint  = our net has no pin near any reference net (usually wrong pin names)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
