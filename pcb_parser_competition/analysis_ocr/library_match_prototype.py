"""Prototype v2: library fills the gap between "we saw a pin name" and "we could not
export that pin".

Architecture under test (matches how the pipeline is actually built):

  evidence   the FINAL exported pins of the component -> (number, name) pairs. That is
             where the information is richest (5-13 pairs per IC after targeted OCR).
  fill target the intermediate pin CANDIDATES of the same component that never made it
             into the export. Those are the tips we have but cannot legally submit.
  step 1     pick the library variant: minimal entry that covers every observed pair.
  step 2     for a candidate that carries a pin NAME but no number, look the number up
             by name; the filled pin is (number, name) and only counts if both match GT.
  guards     >=3 observed pairs, no contradiction with any observed pair, and the name
             we fill from must be unique inside the entry (multiple GND pins are not).
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LIBRARY_FILES = [
    Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\cpntLibrary"),
    Path(r"D:\pcb5_submission\data\component_library\BoxICLibrary.json"),
]
TIP_TOL = 20.0


def norm(text):
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def numeric(text):
    return bool(re.fullmatch(r"\d{1,4}", str(text or "").strip()))


def load_library():
    entries, seen = [], set()
    for path in LIBRARY_FILES:
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, value in doc.items():
            pins = value.get("pins")
            if not isinstance(pins, list) or not pins:
                continue
            pairs = defaultdict(set)
            for pin in pins:
                number = str(pin.get("number") or "").strip()
                name = norm(pin.get("name"))
                if number and name:
                    pairs[number].add(name)
            mapping = {n: sorted(s)[0] for n, s in pairs.items() if len(s) == 1}
            if len(mapping) < 3:
                continue
            signature = (norm(key), tuple(sorted(mapping.items())))
            if signature in seen:
                continue
            seen.add(signature)
            names = defaultdict(set)
            for number, name in mapping.items():
                names[name].add(number)
            entries.append({"model": key, "key": norm(key), "mapping": mapping,
                            "by_name": {n: sorted(v)[0] for n, v in names.items() if len(v) == 1},
                            "pins": len(mapping)})
    return entries


def choose_entry(entries, observed):
    if len(observed) < 3:
        return None, "too_few_observed_pairs"
    covering = []
    for entry in entries:
        mapping = entry["mapping"]
        if all(mapping.get(number) == name for number, name in observed.items()):
            covering.append(entry)
    if not covering:
        return None, "no_entry_covers_observations"
    covering.sort(key=lambda e: (e["pins"], e["key"]))
    best = covering[0]
    rivals = [c for c in covering[1:] if c["pins"] == best["pins"]]
    return best, ("unique" if not rivals else f"ambiguous_{len(rivals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=r"D:\pcb_v8_check\out\final_results")
    ap.add_argument("--work", default=r"D:\pcb_v8_check\out\work")
    ap.add_argument("--gt-root", default=r"D:\pcb_v8_check\gt")
    args = ap.parse_args()
    entries = load_library()
    print(f"library entries usable: {len(entries)}")
    pred_dir, work_root, gt_root = Path(args.pred), Path(args.work), Path(args.gt_root)

    stats = Counter()
    rows = []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob("*_target_named.json"), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        png = next(folder.glob("*.png"), None)
        if gt_file is None or not pred_file.exists() or png is None:
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        import hashlib
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        work = None
        for candidate in work_root.iterdir():
            owners = sorted(candidate.glob("**/ownership/pin_ownership.json"))
            if owners and json.loads(owners[0].read_text(encoding="utf-8")).get("image_sha256") == digest:
                work = candidate
                break
        if work is None:
            continue
        cand_files = sorted(work.glob("**/output_rules/candidate_pin_text_result.json"))
        if not cand_files:
            cand_files = sorted(work.glob("**/candidate_pin_text_result.json"))
        doc = json.loads(cand_files[0].read_text(encoding="utf-8"))
        height = float(doc.get("size", [0, 0])[1])
        candidates = doc.get("pin_candidates", [])

        for comp, component in gt["components"].items():
            if str(component.get("type")) not in ("box", "ic"):
                continue
            mine = pred["pins"].get(comp) or {}
            observed = {}
            for key, pin in mine.items():
                name = norm(pin.get("pinname"))
                if key[4:] and name and not numeric(name):
                    observed[key[4:]] = name
            chosen, how = choose_entry(entries, observed)
            row = {"case": case, "comp": comp, "gt": component.get("Name"),
                   "obs": len(observed), "entry": chosen["model"] if chosen else None, "how": how,
                   "targets": 0, "ok": 0, "bad": 0, "no_name": 0, "name_missing_in_entry": 0}
            if chosen is None:
                stats["entry_not_chosen"] += 1
                rows.append(row)
                continue
            stats["entry_chosen"] += 1
            stats["unique"] += how.startswith("unique")
            by_name = chosen["by_name"]
            counted = set(mine)
            # a candidate tip that sits on top of an already exported pin is a duplicate,
            # not a missing pin; filling it would create a false positive.
            exported_points = []
            for key, pin in mine.items():
                exported_points.append((key[4:], float(pin["point"]["x"]), float(pin["point"]["y"])))
            for pin in candidates:
                if pin.get("owner") != comp:
                    continue
                number = pin.get("number")
                if number is not None and f"pin_{number}" in counted:
                    continue
                tip = (float(pin["point"][0]), height - float(pin["point"][1]))
                if any(math.hypot(x - tip[0], y - tip[1]) <= TIP_TOL for _, x, y in exported_points):
                    stats["target_is_duplicate_of_exported"] += 1
                    continue
                row["targets"] += 1
                read_name = norm(pin.get("pinname"))
                if not read_name or numeric(read_name):
                    row["no_name"] += 1
                    stats["target_without_name"] += 1
                    continue
                got_number = by_name.get(read_name)
                if got_number is None:
                    row["name_missing_in_entry"] += 1
                    stats["name_not_unique_in_entry"] += 1
                    continue
                # judge against GT at this tip's position
                point = (float(pin["point"][0]), height - float(pin["point"][1]))
                gt_hit = None
                for pk, want in (gt["pins"].get(comp) or {}).items():
                    d = math.hypot(float(want["point"]["x"]) - point[0], float(want["point"]["y"]) - point[1])
                    if d <= TIP_TOL:
                        gt_hit = (pk, norm(want.get("pinname")))
                        break
                if gt_hit is None:
                    row["bad"] += 1
                    stats["fill_bad_no_gt_pin"] += 1
                    continue
                pk, gt_name = gt_hit
                if pk[4:] == got_number and read_name == gt_name:
                    row["ok"] += 1
                    stats["fill_ok"] += 1
                else:
                    row["bad"] += 1
                    stats["fill_bad"] += 1
            rows.append(row)

    print(f"{'case':>4} {'comp':>6} {'GT Name':18s} {'obs':>3} {'entry':26s} {'how':16s} {'tgt':>4} {'ok':>3} {'bad':>3} {'noname':>6}")
    for row in rows:
        if row["targets"] == 0:
            continue
        print(f"{row['case']:>4} {row['comp']:>6} {str(row['gt'])[:18]:18s} {row['obs']:>3} "
              f"{str(row['entry'])[:26]:26s} {row['how']:16s} {row['targets']:>4} {row['ok']:>3} "
              f"{row['bad']:>3} {row['no_name']:>6}")
    print("\n== totals ==")
    for key, value in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {key:28s} {value}")
    ok, bad = stats["fill_ok"], stats["fill_bad"] + stats["fill_bad_no_gt_pin"]
    print(f"\n  filled correctly {ok}, wrongly {bad}  -> precision {ok/max(1,ok+bad):.1%}")


if __name__ == "__main__":
    sys.exit(main())
