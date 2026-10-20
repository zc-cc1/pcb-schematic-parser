"""Mutually exclusive loss buckets for GT IC pins, excluding the already-fixed prefixes.

The USB/H/CON/CN designator prefixes are now in `component_text.REF`, so their pins
are no longer lost for that reason. This counts what is left, one cause per pin:

  COMPONENT_NO_KEY      the GT component key is absent from our output (not USB/H)
  OWNER_NO_DESIGNATOR   a tip exists, but its component was never named
  TIP_FAR               no learned tip within 15 px (bucket 15-30 / 30-60 / >60)
  NUMBER_NOT_READ       the recogniser never produced that pin number anywhere
  NUMBER_WINDOW         the number token exists but a geometric gate rejects it
  NUMBER_ASSIGNMENT     the number token passes every gate and is still not used
  NAME_NOT_READ         number fine, but the pin name string never appears
  NAME_WINDOW           number fine, name exists but a gate rejects it
  NAME_ASSIGNMENT       number fine, name passes gates and is still not used
  WRONG_NUMBER / WRONG_NAME   exported, but a field differs
  OK
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.component_text import center, iou  # noqa: E402
from pcb_parser.pin_text import contexts, local_coordinates  # noqa: E402
from pcb_parser.pin_text_v2 import cost_v2, tokens_with_views  # noqa: E402

EXCLUDED_PREFIX = re.compile(r"^(USB|CON|CN|H)\d+$", re.I)
TIP = 15.0


def gate_reason(token, ctx, field, cfg, blocked):
    component = ctx["component"]
    if token["confidence"] < cfg["min_confidence"]:
        return "low_confidence"
    if any(iou(token["bbox"], b) >= .45 for b in blocked):
        return "blocked_by_component_field"
    tangent, inward = local_coordinates(ctx, token["bbox"])
    if abs(tangent) > ctx["tangent_limit"]:
        return "tangent_outside"
    text = token["text"]
    if field == "number":
        options = cfg
        if component["type"] not in cfg["name_component_types"]:
            options = dict(cfg, number_outer_band_px=cfg["passive_number_outer_band_px"],
                           number_inner_band_px=cfg["passive_number_inner_band_px"])
        decimal = re.fullmatch(r"\d{1,4}", text)
        ball = (component["type"] in cfg["ball_number_component_types"]
                and re.fullmatch(r"[A-Z]{1,2}\d{1,3}", text))
        if not (decimal or ball):
            return "not_a_number"
        if not -options["number_outer_band_px"] <= inward <= options["number_inner_band_px"]:
            return "inward_outside_band"
        if ball and inward > cfg["ball_number_max_inward_px"]:
            return "ball_inward"
    else:
        if component["type"] not in cfg["name_component_types"]:
            return "type_not_in_name_scope"
        if not re.search(r"[A-Za-z]", text):
            return "no_letters"
        if not re.fullmatch(r"[A-Za-z0-9_+./~!#()\[\]-]+", text):
            return "name_charset"
        if not cfg["name_min_inward_px"] <= inward <= cfg["name_max_inward_px"]:
            return "inward_outside_name_band"
        cx, cy = center(token["bbox"])
        x1, y1, x2, y2 = component["bbox_top_left"]
        if not x1 <= cx <= x2 or not y1 <= cy <= y2:
            return "center_outside_body"
    axis = 1 if ctx["side"] in {"left", "right"} else 0
    if token["bbox"][axis + 2] - token["bbox"][axis] > max(14, 4 * ctx["tangent_limit"]):
        return "span_too_wide"
    return None


def classify_field(tokens, context, tip, field, want_text, cfg, blocked, candidates):
    """Return a short bucket name for one field of one pin."""
    if not want_text:
        return None
    if field == "number":
        matches = [t for t in tokens if t["text"] == want_text]
    else:
        matches = [t for t in tokens if t["text"].replace(" ", "") == want_text.replace(" ", "")]
    if not matches:
        return "READ"
    passing = []
    for token in matches:
        reason = gate_reason(token, context, field, cfg, blocked)
        if reason is None and cost_v2(token, context, field, cfg, blocked) is not None:
            reason = "cost_v2_disagrees"
        if reason is None:
            passing.append(token)
    if passing:
        return "ASSIGNMENT"
    worst = Counter(gate_reason(t, context, field, cfg, blocked) for t in matches)
    top = worst.most_common(1)[0][0] or "cost_v2_rejected_without_reason"
    return "WINDOW(" + top + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    pred_dir, work_root, gt_root = Path(args.pred), Path(args.work), Path(args.gt_root)
    cfg = json.loads((BASE / "configs" / "pin_text_v2.json").read_text(encoding="utf-8"))

    tally = Counter()
    examples = {}
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob(args.glob), None)
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
        ownership = json.loads(next(work.glob("**/ownership/pin_ownership.json")).read_text(encoding="utf-8"))
        pin_text = json.loads(next(work.glob("**/upstream/pin_text/pin_text_result.json")).read_text(encoding="utf-8"))
        owner_of = {p["candidate_id"]: p.get("owner") for p in pin_text.get("pin_candidates", [])}
        records_file = next(work.glob("**/upstream/upstream/component_records.json"), None)
        records = json.loads(records_file.read_text(encoding="utf-8")).get("records", []) if records_file else []
        texts_file = next(work.glob("**/upstream/upstream/texts.json"), None)
        texts = json.loads(texts_file.read_text(encoding="utf-8"))["texts"] if texts_file else []
        blocked = [r["fields"][f]["bbox"] for r in records for f in ("designator", "Name", "value")
                   if f in (r.get("fields") or {})]
        height = float(ownership["size"][1])
        ctx = contexts(ownership, cfg)
        tokens = [t for t in tokens_with_views(texts)
                  if not any(iou(t["bbox"], b) >= .45 for b in blocked)]
        exported = []
        for group in (pred.get("pins") or {}).values():
            for pin_key, pin in group.items():
                exported.append((float(pin["point"]["x"]), float(pin["point"]["y"]),
                                 pin_key[4:], pin.get("pinname")))

        for comp, group in (gt.get("pins") or {}).items():
            if str((gt["components"].get(comp) or {}).get("type")) not in ("box", "ic"):
                continue
            if EXCLUDED_PREFIX.match(comp):
                continue
            for pin_key, want in group.items():
                point = (float(want["point"]["x"]), float(want["point"]["y"]))
                label = f"{case}/{comp}.{pin_key[4:]}"
                if comp not in pred["components"]:
                    bucket = "COMPONENT_NO_KEY"
                else:
                    near = [(math.hypot(float(c["point"][0]) - point[0], height - float(c["point"][1]) - point[1]), c)
                            for c in ownership["pin_candidates"] if c["candidate_id"] in ctx]
                    near = [(d, c) for d, c in near if d <= TIP]
                    hit = None
                    for index, (x, y, number, name) in enumerate(exported):
                        d = math.hypot(x - point[0], y - point[1])
                        if d <= 5.0 and (hit is None or d < hit[0]):
                            hit = (d, index)
                    if hit is not None:
                        _, _, got_number, got_name = exported[hit[1]]
                        if got_number != pin_key[4:]:
                            bucket = "WRONG_NUMBER"
                        elif (got_name or None) != (want.get("pinname") or None):
                            bucket = "WRONG_NAME"
                        else:
                            bucket = "OK"
                    elif not near:
                        closest = min((math.hypot(float(c["point"][0]) - point[0],
                                                  height - float(c["point"][1]) - point[1])
                                       for c in ownership["pin_candidates"] if c["candidate_id"] in ctx),
                                      default=1e9)
                        bucket = ("TIP_FAR(" + ("15-30" if closest <= 30 else "30-60" if closest <= 60 else ">60") + ")")
                    else:
                        tip = min(near, key=lambda r: r[0])[1]
                        owner = str(owner_of.get(tip["candidate_id"]) or "")
                        if not owner or owner.startswith("UNRESOLVED"):
                            bucket = "OWNER_NO_DESIGNATOR"
                        else:
                            context = ctx[tip["candidate_id"]]
                            num = classify_field(tokens, context, tip, "number", pin_key[4:],
                                                 cfg, blocked, ownership["pin_candidates"])
                            name = classify_field(tokens, context, tip, "pinname",
                                                  want.get("pinname") or "", cfg, blocked,
                                                  ownership["pin_candidates"])
                            if num == "READ":
                                # is any digit printed near this tip at all? If not, the
                                # number cannot come from OCR - the drawing does not show it.
                                near_digits = [t for t in tokens if re.fullmatch(r"\d{1,4}", t["text"])
                                               and math.hypot(center(t["bbox"])[0] - float(tip["point"][0]),
                                                              center(t["bbox"])[1] - float(tip["point"][1])) <= 40]
                                num = "READ(digit_nearby)" if near_digits else "READ(no_digit_nearby)"
                            if num != "ASSIGNMENT" and num is not None:
                                bucket = "NUMBER_" + num
                            elif name is not None and name != "ASSIGNMENT":
                                bucket = "NAME_" + name
                            elif num is None and name is None:
                                bucket = "UNKNOWN"
                            else:
                                bucket = "GATED(" + ("NUMBER" if num == "ASSIGNMENT" else "NAME") + "_ASSIGNMENT)"
                tally[bucket] += 1
                examples.setdefault(bucket, [])
                if len(examples[bucket]) < args.examples:
                    examples[bucket].append(label)

    total = sum(tally.values())
    print(f"GT IC pins counted (USB/H/CON/CN excluded): {total}")
    for key, value in tally.most_common():
        print(f"  {key:38s} {value:4d}  {value/total:6.1%}   e.g. {', '.join(examples[key])}")
    ok = tally["OK"]
    print(f"\n  approximate pin recall {ok/total:.1%} (TP {ok} / {total} GT IC pins)")


if __name__ == "__main__":
    sys.exit(main())
