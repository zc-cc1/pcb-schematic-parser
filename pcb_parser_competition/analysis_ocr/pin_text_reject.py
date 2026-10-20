"""Why did the 23-odd "no_reliable_text" pins get no text? One cause per pin.

For every GT pin whose learned tip exists but that was never exported, this walks the
same gates `pin_text_v2.cost_v2` applies and reports the FIRST reason the observed
text was not usable:

  NO_TOKEN_NEAR          the recogniser never produced that string near this tip
  NOT_IN_TOKEN_SET       the string exists on the page but nowhere within 40 px
  REJECTED_<gate>        a token is there, and this gate threw it away
  CANDIDATE_BUT_LOST     the token passed every gate and still lost the assignment

A self-check asserts the explainer returns "rejected" exactly when `cost_v2` does;
if that ever drifts the histogram is meaningless, so it is printed.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.component_text import center, iou  # noqa: E402
from pcb_parser.pin_text import contexts, local_coordinates  # noqa: E402
from pcb_parser.pin_text_v2 import cost_v2, glyph_groups, tokens_with_views  # noqa: E402

NEAR = 40.0
TIP = 15.0


def explain(token, ctx, field, cfg, blocked):
    """Mirror of cost_v2's gates that returns the reason instead of None."""
    component = ctx["component"]
    if token["confidence"] < cfg["min_confidence"]:
        return "low_confidence"
    if any(iou(token["bbox"], b) >= .45 for b in blocked):
        return "blocked_by_component_field"
    tangent, inward = local_coordinates(ctx, token["bbox"])
    if abs(tangent) > ctx["tangent_limit"]:
        return "tangent_outside"
    text = token["text"]
    import re
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
    span = token["bbox"][axis + 2] - token["bbox"][axis]
    if span > max(14, 4 * ctx["tangent_limit"]):
        return "span_too_wide"
    return None


def norm_type(value):
    t = str(value or "").strip().lower()
    return "box" if t in {"box", "ic"} else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--glob", default="*_target_named.json")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    pred_dir, work_root, gt_root = Path(args.pred), Path(args.work), Path(args.gt_root)
    cfg = json.loads((BASE / "configs" / "pin_text_v2.json").read_text(encoding="utf-8"))

    owner_reasons = Counter()
    number_reasons = Counter()
    name_reasons = Counter()
    examples = []
    mismatches = 0
    checked = 0
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: (len(p.name), p.name)):
        case = folder.name
        gt_file = next(folder.glob(args.glob), None)
        pred_file = pred_dir / f"{case}_Pred.json"
        if gt_file is None or not pred_file.exists():
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        png = next(folder.glob("*.png"), None)
        if png is None:
            continue
        import hashlib
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        work = None
        for candidate in work_root.iterdir():
            owners = sorted(candidate.glob("**/ownership/pin_ownership.json"))
            if not owners:
                continue
            try:
                doc = json.loads(owners[0].read_text(encoding="utf-8"))
            except Exception:
                continue
            if doc.get("image_sha256") == digest:
                work = candidate
                break
        if work is None:
            continue
        ownership = json.loads(next(work.glob("**/ownership/pin_ownership.json")).read_text(encoding="utf-8"))
        # the designator only appears after identity attach, i.e. in pin_text_result;
        # the ownership file still carries the detection id (DET_xxxx).
        pin_text_doc = json.loads(
            next(work.glob("**/upstream/pin_text/pin_text_result.json")).read_text(encoding="utf-8"))
        owner_of = {p["candidate_id"]: p.get("owner") for p in pin_text_doc.get("pin_candidates", [])}
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
        by_candidate = {p["candidate_id"]: p for p in ownership["pin_candidates"]}

        # exported pins, in the pipeline's top-left space
        exported = set()
        for owner, group in (pred.get("pins") or {}).items():
            for pin in group.values():
                exported.add((round(float(pin["point"]["x"])), round(height - float(pin["point"]["y"]))))

        for comp, group in (gt.get("pins") or {}).items():
            if norm_type((gt["components"].get(comp) or {}).get("type")) != "box":
                continue
            for pin_key, want in group.items():
                point = (float(want["point"]["x"]), height - float(want["point"]["y"]))
                near = [(math.hypot(float(c["point"][0]) - point[0], float(c["point"][1]) - point[1]), c)
                        for c in ownership["pin_candidates"] if c["candidate_id"] in ctx]
                near = [(d, c) for d, c in near if d <= TIP]
                if not near:
                    continue
                tip = min(near, key=lambda r: r[0])[1]
                # "exported" must be judged at the GT point: the pipeline now snaps
                # tips onto the lead, so the exported point is not the ownership tip.
                if any(math.hypot(px - point[0], py - point[1]) <= 5.0 for px, py in exported):
                    continue                      # this pin was exported; not our question
                checked += 1
                # A pin can only be exported under a component key. If the owner was
                # never given a designator, no amount of pin text can save it - that
                # is a designator problem, not a pin-text problem.
                owner = str(owner_of.get(tip["candidate_id"]) or "")
                if not owner or owner.startswith("UNRESOLVED"):
                    owner_reasons["OWNER_HAS_NO_DESIGNATOR"] += 1
                    continue
                context = ctx[tip["candidate_id"]]
                for field, want_text in (("number", pin_key[4:]), ("pinname", want.get("pinname") or "")):
                    if field == "number":
                        matches = [t for t in tokens if t["text"] == want_text]
                    else:
                        if not want_text:
                            continue
                        matches = [t for t in tokens
                                   if t["text"].replace(" ", "") == want_text.replace(" ", "")]
                    if not matches:
                        reason = "NO_TOKEN_ON_PAGE"
                    else:
                        # judge every occurrence with the pipeline's own gates; no
                        # arbitrary search window, so "far away" cannot masquerade
                        # as "rejected".
                        evaluated = []
                        for token in matches:
                            r = explain(token, context, field, cfg, blocked)
                            if r is None and cost_v2(token, context, field, cfg, blocked) is None:
                                mismatches += 1
                            if r is not None and cost_v2(token, context, field, cfg, blocked) is not None:
                                mismatches += 1
                            distance = math.hypot(center(token["bbox"])[0] - float(tip["point"][0]),
                                                  center(token["bbox"])[1] - float(tip["point"][1]))
                            evaluated.append((r, distance, token))
                        passing = [row for row in evaluated if row[0] is None]
                        if passing:
                            token_centre = center(passing[0][2]["bbox"])
                            own = math.hypot(float(tip["point"][0]) - token_centre[0],
                                             float(tip["point"][1]) - token_centre[1])
                            nearest = min((math.hypot(float(c["point"][0]) - token_centre[0],
                                                      float(c["point"][1]) - token_centre[1]),
                                           c["candidate_id"])
                                          for c in ownership["pin_candidates"] if c["candidate_id"] in ctx)
                            reason = ("CANDIDATE_BUT_LOST"
                                      if nearest[1] == tip["candidate_id"] or nearest[0] >= own - 1.0
                                      else "TOKEN_NEARER_TO_ANOTHER_PIN")
                        else:
                            closest = min(evaluated, key=lambda row: row[1])
                            reason = closest[0]
                    (number_reasons if field == "number" else name_reasons)[reason] += 1
                    if len(examples) < args.limit and field == "number":
                        examples.append((case, comp, pin_key[4:], want.get("pinname"), reason,
                                         tip["candidate_id"], [round(v) for v in tip["point"]]))

    print(f"pins inspected (tip found, not exported): {checked}; explainer/cost disagreements: {mismatches}")
    print("\nOWNER level:")
    for key, value in owner_reasons.most_common():
        print(f"  {key:28s} {value}")
    print("\nNUMBER field reasons:")
    for key, value in number_reasons.most_common():
        print(f"  {key:28s} {value}")
    print("\nPINNAME field reasons:")
    for key, value in name_reasons.most_common():
        print(f"  {key:28s} {value}")
    print("\nexamples:")
    for row in examples:
        print(f"  case {row[0]:>2} {row[1]}.{row[2]} name={row[3]!r} -> {row[4]} "
              f"(cand {row[5]} at {row[6]})")


if __name__ == "__main__":
    sys.exit(main())


