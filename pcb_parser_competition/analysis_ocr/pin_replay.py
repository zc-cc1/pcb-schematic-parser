"""Offline A/B harness for pin text (pin_text v2) on a kept-work directory.

The deployed chain persists every intermediate when `agent.py --keep-work` is used,
including the U-Net pin ownership and the OCR texts. This replays exactly the
`run_pin_text_v2.process` stage with an overridable config, so a knob change can be
measured in seconds against the public GT instead of a 3-6 minute pipeline run.

Scoring follows the official pin rule: same component key + same pin_<number>,
then pinname exact and tip within 5 px. Only components whose key exists in both
sides are scored (the public GT renumbers some pages, which is a key problem, not
a pin problem); the reachable GT pin count is reported so coverage stays visible.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
PIN_TOL = 5.0       # official tip tolerance
NEAR_TOL = 15.0     # diagnostic pairing window: separates text errors from tip offsets
IC_TYPES = {"box", "ic"}

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "src"))


def find_upstream(work_root: Path, image: Path):
    """Pick the ownership cache whose image hash matches this PNG."""
    import hashlib

    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    owners = sorted(work_root.glob("work/**/ownership/pin_ownership.json"))
    if not owners:
        owners = sorted(work_root.glob("**/ownership/pin_ownership.json"))
    if not owners:
        raise FileNotFoundError(f"no ownership intermediate under {work_root}")
    for ownership in owners:
        try:
            cached = json.loads(ownership.read_text(encoding="utf-8")).get("image_sha256")
        except Exception:
            continue
        if cached == digest:
            texts = ownership.parent.parent / "texts.json"
            if texts.exists():
                return ownership, texts, image
    raise FileNotFoundError(f"no ownership cache matches {image.name} ({digest[:12]}) under {work_root}")


def load_cfg(set_args):
    cfg = json.loads((BASE / "configs" / "pin_text_v2.json").read_text(encoding="utf-8"))
    for item in set_args:
        key, _, value = item.partition("=")
        try:
            cfg[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key.strip()] = value
    return cfg


def replay(case, work_root, cfg, outdir, image_override=None):
    import run_pin_text_v2 as stage
    image = Path(image_override) if image_override else next((GT_ROOT / case).glob("*.png"))
    ownership, texts, image = find_upstream(work_root, image)
    outdir.mkdir(parents=True, exist_ok=True)
    stage.process(image, ownership, texts, outdir, cfg)
    return json.loads((outdir / "pin_text_result.json").read_text(encoding="utf-8"))


def norm_type(t):
    t = str(t or "").strip().lower()
    if t in {"ground", "gnd"}:
        return "gnd"
    if t in {"v", "vcc", "vdd"}:
        return "v"
    if t in IC_TYPES:
        return "box"
    return t


def score(result, gt):
    """Key-free pin scoring: match our tips to GT tips by position.

    Component keys are unreliable on some public pages (renumbered designators),
    so matching pins by key would hide the pin-text signal. Tips are geometric and
    reliable: pair the closest GT tip within 5 px, then ask whether the number and
    the name we attached to that tip are the ones the GT puts there.
    """
    gt_ic = {}
    for comp, group in (gt.get("pins") or {}).items():
        if norm_type((gt["components"].get(comp) or {}).get("type")) != "box":
            continue
        for pin_key, want in group.items():
            gt_ic[(comp, pin_key[4:])] = ((want.get("pinname") or None),
                                          (float(want["point"]["x"]), float(want["point"]["y"])))
    ours = []
    for pin in result["pin_candidates"]:
        if pin.get("owner_status") != "assigned":
            continue
        ours.append((pin.get("number"), pin.get("pinname"), tuple(pin["point"]), pin.get("owner")))
    used = set()
    matched = within_5 = number_ok = name_ok = both = 0
    both_within_5 = 0
    for (comp, number), (want_name, want_point) in gt_ic.items():
        best = None
        for index, (_, _, point, _) in enumerate(ours):
            if index in used:
                continue
            distance = math.dist(point, want_point)
            if distance <= NEAR_TOL and (best is None or distance < best[0]):
                best = (distance, index)
        if best is None:
            continue
        used.add(best[1])
        matched += 1
        got_number, got_name, _, _ = ours[best[1]]
        same_number = (got_number or None) == number
        same_name = (got_name or None) == (want_name or None)
        number_ok += same_number
        name_ok += same_name
        both += same_number and same_name
        if best[0] <= PIN_TOL:
            within_5 += 1
            both_within_5 += same_number and same_name
    emitted = [o for o in ours if o[0] is not None]
    return {"gt_ic_pins": len(gt_ic), "gt_ic_components": len({c for (c, _) in gt_ic}),
            "our_assigned_pins": len(ours), "our_pins_with_number": len(emitted),
            "tip_matched_within_15px": matched,
            "tip_within_5px": within_5,
            "tip_recall_5px": round(within_5 / max(1, len(gt_ic)), 3),
            "number_correct_on_tips": number_ok,
            "name_correct_on_tips": name_ok,
            "number_and_name_correct": both,
            "both_correct_within_5px": both_within_5,
            "number_rate_of_matched": round(number_ok / max(1, matched), 3),
            "name_rate_of_matched": round(name_ok / max(1, matched), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.set)
    outdir = Path(args.out) if args.out else Path(r"D:\pcb_v6_check\pin_replay") / args.case
    result = replay(args.case, Path(args.work), cfg, outdir, args.image)
    gt = json.loads(next((GT_ROOT / args.case).glob("*_target_named.json")).read_text(encoding="utf-8"))
    row = score(result, gt)
    summary = result.get("pin_text_summary") or {}
    print(json.dumps({"case": args.case, "overrides": args.set, **row,
                      "number_observed": summary.get("number_observed"),
                      "name_observed": summary.get("name_observed"),
                      "number_status": summary.get("number_status_counts"),
                      "name_status": summary.get("name_status_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
