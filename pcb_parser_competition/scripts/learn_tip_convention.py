"""Learn the tip convention from Dataset300 and compare it with the contest targets.

Dataset300 gives, for every pin, a `tip` and its component's `realBBox`.  The contest
reference gives a `point` and a `bbox`.  Both are "the pin's connection point", so
measuring how far each sits outside its own component box reveals whether the two use
the same convention, and if not, what the conversion is.

This is the one pin-side question that a global/per-type shift could not answer: the
shift analysis used the detector box in some places and the reference box in others.
Here both sides are measured strictly against their own annotation's box.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_golden_cases import GT_DIR  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.jsonl"


def outside_distance(point, box):
    """Distance from the point to the box centre minus the box's half-extent along
    the point's direction, i.e. how far the point lies outside the box edge."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    dx, dy = point[0] - cx, point[1] - cy
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0
    ux, uy = dx / norm, dy / norm
    # half-extent of the axis-aligned box in the direction (ux, uy)
    hx, hy = (box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0
    denom = abs(ux) / hx + abs(uy) / hy if hx > 0 and hy > 0 else 0.0
    reach = 1.0 / denom if denom else 0.0
    return norm - reach


def main():
    manifest = {json.loads(line)["case_id"]: json.loads(line)
                for line in MANIFEST.read_text(encoding="utf-8").splitlines()}
    per_case = []
    dataset_all = []
    contest_all = []
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name.split("-")[0]
        if case not in manifest:
            continue
        truth = json.loads(next(folder.glob("*_target.json")).read_text(encoding="utf-8"))
        height = None
        from PIL import Image
        with Image.open(next(folder.glob("*.png"))) as image:
            height = image.height
        # contest side: point vs its own bbox (bbox is in the same frame as point)
        contest = []
        for key, pins in (truth.get("pins") or {}).items():
            box = (truth.get("components", {}).get(key) or {}).get("bbox")
            if not box:
                continue
            for pin in pins.values():
                contest.append(outside_distance((pin["point"]["x"], pin["point"]["y"]), box))
        # dataset side: annotation tip vs realBBox, converted into the same frame
        raw = json.loads(Path(manifest[case]["raw_annotation_path"]).read_text(encoding="utf-8"))
        dataset = []
        for component in (raw.get("components") or {}).values():
            box = component.get("realBBox") or {}
            if not all(k in box for k in ("x1", "y1", "x2", "y2")):
                continue
            flipped = [float(box["x1"]), height - float(box["y2"]),
                       float(box["x2"]), height - float(box["y1"])]
            for pin in (component.get("pins") or {}).values():
                tip = pin.get("tip")
                if not tip:
                    continue
                dataset.append(outside_distance((float(tip[0]), height - float(tip[1])), flipped))
        if contest and dataset:
            per_case.append((case, len(dataset), len(contest),
                             statistics.median(dataset), statistics.median(contest)))
            dataset_all.extend(dataset)
            contest_all.extend(contest)
    print(f"{'case':<8}{'ds pins':>9}{'gt pins':>9}{'ds median':>11}{'gt median':>11}")
    for case, n_ds, n_gt, m_ds, m_gt in per_case:
        print(f"{case:<8}{n_ds:>9}{n_gt:>9}{m_ds:>11.2f}{m_gt:>11.2f}")
    print()
    if dataset_all:
        print(f"Dataset300 pins: {len(dataset_all)}, median distance outside its own realBBox "
              f"{statistics.median(dataset_all):.2f} px")
        print(f"Contest GT pins: {len(contest_all)}, median distance outside its own bbox   "
              f"{statistics.median(contest_all):.2f} px")
        print(f"difference of medians: {statistics.median(dataset_all) - statistics.median(contest_all):+.2f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
