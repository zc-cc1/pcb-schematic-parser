"""Local reference implementation of the competition accuracy metrics.

STATUS: approximate, not official.  The competition scores submissions on its own
platform; no evaluator ships with the data.  This module implements the rules the
spec states explicitly (docs/requirements.md R02-R15) so that progress can be
measured locally.  Every non-specified choice is listed in ASSUMPTIONS and echoed
into the report, because a number produced here must never be presented as the
official ComponentF1 / PinF1 / NetHypergraphF1.

Implemented, per R16 weights (.30/.25/.35/.10):
  ComponentF1         **key** match (not geometry), then type, then bbox within 20 px per side,
                      then text: Name for box/ic, value for r/c (PDF p7 + p17 note b)
  PinF1               component key + pin_<number> match, 5 px point tolerance,
                      pinname strict string equality (R05/R06/R07)
  NetHypergraphF1     net key is ignored; nets matched by pin-set identity (R08/R09)
  NetLineF1           two-point segments compared only inside matched net pairs,
                      5 px endpoint tolerance (R10/R11)

Not implemented yet:
  * VCC/GND special merging and logical flywire semantics (Q&A answers these explicitly; see
    官方资料/01_官方判分规则速查.md section 5)
  * multi-number pin splitting (PDF p18 note d)
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

COMPONENT_IOU = 0.5
COMPONENT_CENTER_TOLERANCE_PX = 20.0   # R07: "component bbox default 20 px"
PIN_TOLERANCE_PX = 5.0
LINE_TOLERANCE_PX = 5.0
WEIGHTS = {"component": 0.30, "pin": 0.25, "net_hypergraph": 0.35, "net_line": 0.10}

# GT files use short enum types ("r", "c", "gnd"); the pipeline emits class names.
# Only one-to-one correspondences belong here.  The reference splits several classes
# by polarity or pin count (bjt_npn/bjt_pnp, mosfet_npn/mosfet_pnp, crystal_2pin);
# collapsing those would let one prediction match two different reference classes,
# which inflates the score instead of measuring it, so they are left unambiguous.
TYPE_ALIASES = {
    "resistor": "r", "capacitor": "c", "inductor": "l", "diode": "d", "led": "led",
    "transistor": "bjt", "opamp": "amp", "ic": "box", "ground": "gnd",
    "voltage_source": "v", "vcc": "v", "current_source": "dc", "crystal": "crystal_2pin",
    "mosfet": "mosfet", "switch": "switch", "net_input": "net_input",
    "pin_port": "pin", "testpoint": "testpoint", "net_output": "net_output",
    "net_bidirectional": "net_bidirection", "fuse": "fuse", "buzzer": "buzzer",
    "motor": "motor", "battery": "battary", "other": "other",
    "transformer": "transfomer", "jumper": "jumper", "raw_block": "block",
    "circle_header": "circle_header", "antenna": "ant", "esd": "esd",
    "raw_m": "m3螺丝", "net_short": "net_short", "optocoupler": "opt",
    "raw_dc": "dc", "connector": "connector", "raw_m3": "m3螺丝",
}

ASSUMPTIONS = [
    "component pairing is by designator key only (PDF p7, p17 note b: bbox is not used to match);"
    " a pair is correct when the normalised type is equal, all four bbox sides deviate by <= 20 px,"
    " and the required text matches",
    "the 20 px bbox rule is read as a per-side coordinate tolerance, because the PDF does not state a"
    " norm and the released references keep the body box with pin tips outside it",
    "type comparison normalises long pipeline names to the short GT enum via TYPE_ALIASES; unknown "
    "names are compared verbatim",
    "pin match requires the owning component to be matched, the same pin_<number> key, and a point "
    "within 5 px; pinname is compared exactly when both sides carry one (R05)",
    "NetHypergraphF1 matches nets by pin-set identity through the same component/pin matching used for "
    "PinF1; net keys are ignored (R08). Net-level all-or-nothing, so it is stricter than a pair count",
    "NetLineF1 only scores segments inside matched net pairs, with 5 px endpoint tolerance (R11)",
    "scores are micro-averaged over all cases by default; PREDICTED-EMPTY cases still count in the "
    "denominator (R17)",
    "NOT implemented: VCC/GND merge rules, logical flywire semantics, multi-number pin splitting",
    "text_mode: 'strict' applies the PDF text rule verbatim; 'designator_name' skips the Name check"
    " for a box whose reference Name is only its designator (the 10_GTcase convention), because that"
    " compares a model name against a designator; 'none' is an upper-bound ablation",
    "anonymous_components='by_position' matches '∅'-keyed power/ground/label references by centre"
    " distance (20 px) instead of by key, per Q&A 评测 #4/#5; 'key' keeps the strict reading and is"
    " the default so historical numbers stay comparable",
    "pin_scope='box_only' counts only box/ic pins in PinF1, per Q&A 评测 #2/#6 (non-box pin names and"
    " numbers are not scored)",
    "net_pin_identity='coarse' drops the pin number of every non-box member when comparing nets, per"
    " Q&A 评测 #6 ('RLC 等在 hypergraph 中相当于不区分引脚'); box members keep their number",
]


def normalise_type(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return TYPE_ALIASES.get(text, text)


# --- component text comparison ------------------------------------------------
# PDF p7 / p17 note c: 'ic' compares Name, 'resistor'/'capacitor' compare value,
# everything else compares nothing.  `text_mode` picks how strictly that is applied:
#   "strict"           -- the PDF rule, verbatim
#   "designator_name"  -- same, except that a box whose reference Name is just the
#                         designator (the 10_GTcase convention) is not text-checked,
#                         because "U1" is not a model name and the comparison would be
#                         measuring the annotation convention, not the prediction
#   "none"             -- no text comparison at all (upper bound / ablation)
TEXT_MODES = ("strict", "designator_name", "none")


def name_is_designator(reference, key):
    """True when the reference stores the designator in Name (10_GTcase does this)."""
    name = str(reference.get("Name") or "").strip().lower()
    return bool(name) and name == str(key or "").strip().lower()


def is_anonymous_key(key):
    """Power/ground symbols and net labels carry no designator: the reference keys them '∅'.

    Q&A 评测 #4/#5: "Vcc 和 gnd 是 ∅，直接是坐标匹配" and "GND 的序号名字不影响评分".
    Q&A 一般 #5/#29: their pin numbers are randomly generated.
    So a key-based match cannot work for them; `anonymous_components='by_position'` models
    the rule the Q&A states instead of silently scoring them as misses.
    """
    return str(key or "").strip().startswith("∅")


def component_kind(components, key):
    """Normalised type of `key` in a components mapping (None when absent)."""
    entry = (components or {}).get(key) or {}
    return normalise_type(entry.get("type"))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_bottom_left(payload, height):
    """Convert a top-left-origin bbox to the GT bottom-left convention.

    Used only for predictions produced by the pipeline.  When scoring a payload
    against itself (self-test) the caller must pass `prediction_is_top_left=False`
    so an already-bottom-left bbox is not flipped a second time.
    """
    x1, y1, x2, y2 = (float(v) for v in payload["bbox"])
    return [x1, height - y2, x2, height - y1]


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def match_components(pred, truth, height, iou_threshold=COMPONENT_IOU, prediction_is_top_left=False,
                     center_tolerance=COMPONENT_CENTER_TOLERANCE_PX, use_center_rule=True,
                     text_mode="strict", anonymous_components="key"):
    """Match components the way the official rule does: BY KEY, not by geometry.

    Official text, p7: "Component 直接用 key 匹配，例如 GT 的 U3 只和 Pred 的 U3 比较.
    一个 component 需要同时满足：a) key 匹配；b) type 正确；c) bbox 坐标误差在阈值内，
    默认 20px；若需要比较文本，则文本严格一致."

    So the key decides the pairing; type, bbox and (for ic / resistor / capacitor) the
    text then decide whether that pair counts as correct.  Predicted keys absent from
    the reference are false positives and can never score, which is why an invented key
    is worse than no key at all.
    """
    truth_components = truth.get("components") or {}
    pred_components = pred.get("components") or {}
    if not truth_components or not pred_components:
        return {}, 0, len(pred_components), len(truth_components)
    by_position = anonymous_components == "by_position"
    mapping, tp = {}, 0
    for key, reference in truth_components.items():
        if by_position and is_anonymous_key(key):
            continue  # handled in the coordinate pass below
        candidate = pred_components.get(key)
        if candidate is None:
            continue
        if normalise_type(candidate.get("type")) != normalise_type(reference.get("type")):
            continue
        box = [float(v) for v in candidate.get("bbox") or []]
        ref_box = [float(v) for v in reference.get("bbox") or []]
        if len(box) != 4 or len(ref_box) != 4:
            continue
        if any(abs(box[i] - ref_box[i]) > center_tolerance for i in range(4)):
            continue
        kind = normalise_type(reference.get("type"))
        if text_mode != "none":
            if kind in ("ic", "box"):
                if not (text_mode == "designator_name" and name_is_designator(reference, key)):
                    if str(reference.get("Name")) != str(candidate.get("Name")):
                        continue
            elif kind in ("r", "c"):
                if str(reference.get("value")) != str(candidate.get("value")):
                    continue
        mapping[key] = key
        tp += 1
    if by_position:
        tp += _match_anonymous_by_position(truth_components, pred_components, mapping,
                                           center_tolerance)
    fp = len(pred_components) - len(mapping)
    fn = len(truth_components) - len(mapping)
    return mapping, tp, fp, fn


def _match_anonymous_by_position(truth_components, pred_components, mapping, tolerance):
    """Pair '∅'-keyed references with the remaining predictions by centre distance.

    Q&A 评测 #4/#5 state that power/ground symbols are matched by coordinate, not by the
    random '∅<n>' key.  Matching is a Hungarian assignment on centre distance, gated by
    the same 20 px tolerance and by type equality, and predictions already consumed by an
    exact key match are excluded.
    """
    anonymous = [k for k in truth_components if is_anonymous_key(k)]
    used = set(mapping)
    candidates = [k for k in pred_components if k not in used]
    if not anonymous or not candidates:
        return 0
    distance = np.full((len(anonymous), len(candidates)), 1e9)
    for i, truth_key in enumerate(anonymous):
        reference = truth_components[truth_key]
        ref_box = [float(v) for v in reference.get("bbox") or []]
        if len(ref_box) != 4:
            continue
        ref_kind = normalise_type(reference.get("type"))
        ref_cx, ref_cy = (ref_box[0] + ref_box[2]) / 2, (ref_box[1] + ref_box[3]) / 2
        for j, pred_key in enumerate(candidates):
            candidate = pred_components[pred_key]
            if normalise_type(candidate.get("type")) != ref_kind:
                continue
            box = [float(v) for v in candidate.get("bbox") or []]
            if len(box) != 4:
                continue
            gap = math.hypot((box[0] + box[2]) / 2 - ref_cx, (box[1] + box[3]) / 2 - ref_cy)
            if gap <= tolerance:
                distance[i, j] = gap
    rows, cols = linear_sum_assignment(distance)
    accepted = 0
    for i, j in zip(rows, cols):
        if distance[i, j] > tolerance:
            continue
        mapping[candidates[j]] = anonymous[i]
        accepted += 1
    return accepted


def parse_hypergraph(text):
    """'(U1.A5,R1.1)' -> ['U1.A5', 'R1.1']; tolerates an empty/∅ member list."""
    body = str(text or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    members = []
    for token in body.split(","):
        token = token.strip()
        if token and token not in {"∅", "null", "None"}:
            members.append(token)
    return members


def match_pins(pred, truth, component_mapping, height, tolerance=PIN_TOLERANCE_PX, by_position=False,
               pin_scope="all", nonbox_pin_mode="number"):
    """Match pins the way R05 specifies: by component key and pin NUMBER.

    R05: "pin 按器件 key 和 pin number 匹配；pinname 严格比较" - identity is
    (component, pin number), the point must then be within tolerance (R07, 5 px), and
    pinname is compared strictly.  R09 confirms the same reading for nets:
    `hyperGraph=(U1.A5,R1.1)` maps to `pin_A5`/`pin_1`, i.e. the token after the dot is
    the pin NUMBER, not the pin name.

    `by_position=True` is kept only as a diagnostic: it ignores numbers and pairs pins
    by nearest point, which flatters a pipeline whose numbering is wrong.  It must not
    be used for a score estimate.
    """
    # Official rule, p8: "Pin 先按 component key 匹配，再按 pin number 匹配".  The
    # component only has to KEY-match; its type, Name/value and bbox are scored by the
    # component item and do not gate the pin item.  Using `component_mapping` here
    # demanded a full component match and hid exactly that case: platform round 3 scored
    # component_f1 0.05 while pin_f1 was 0.69.
    reverse = {}
    for truth_key in (truth.get("pins") or {}):
        if truth_key in (pred.get("pins") or {}):
            reverse[truth_key] = truth_key
    for pred_key, truth_key in component_mapping.items():
        reverse.setdefault(truth_key, pred_key)
    tp, fp, fn = 0, 0, 0
    matched_pairs = []
    handled = set()
    truth_pins = truth.get("pins") or {}
    pred_pins = pred.get("pins") or {}
    truth_components = truth.get("components") or {}
    box_only = pin_scope == "box_only"
    for truth_component, pins in truth_pins.items():
        if box_only and component_kind(truth_components, truth_component) not in ("box", "ic"):
            continue  # Q&A 评测 #2/#6: non-box pin names/numbers are not scored
        pred_component = reverse.get(truth_component)
        if pred_component is None:
            fn += len(pins)
            continue
        candidates = pred_pins.get(pred_component) or {}
        if nonbox_pin_mode == "position" and component_kind(truth_components, truth_component) not in ("box", "ic"):
            # Q&A 评测 #2/#6: for non-box parts the pin number and name are not scored, and
            # #6 says they behave as if the pin were not distinguished at all.  Counting them
            # by position only is the reading that keeps them in the denominator.
            reference_keys, predicted_keys = list(pins), list(candidates)
            if not reference_keys:
                fp += len(predicted_keys)
                continue
            if not predicted_keys:
                fn += len(reference_keys)
                continue
            distance = np.full((len(predicted_keys), len(reference_keys)), 1e9)
            for i, pk in enumerate(predicted_keys):
                px = float(candidates[pk]["point"]["x"])
                py = float(candidates[pk]["point"]["y"])
                for j, tk in enumerate(reference_keys):
                    qx = float(pins[tk]["point"]["x"])
                    qy = float(pins[tk]["point"]["y"])
                    gap = math.hypot(px - qx, py - qy)
                    if gap <= tolerance:
                        distance[i, j] = gap
            rows, cols = linear_sum_assignment(distance)
            accepted = 0
            for i, j in zip(rows, cols):
                if distance[i, j] > tolerance:
                    continue
                accepted += 1
                matched_pairs.append((pred_component, predicted_keys[i], truth_component, reference_keys[j]))
            tp += accepted
            fn += len(reference_keys) - accepted
            fp += len(predicted_keys) - accepted
            handled.add(pred_component)
            continue
        if by_position and candidates:
            reference_keys, predicted_keys = list(pins), list(candidates)
            distance = np.zeros((len(predicted_keys), len(reference_keys)))
            same_key = np.zeros_like(distance, dtype=bool)
            for i, pk in enumerate(predicted_keys):
                px = float(candidates[pk]["point"]["x"])
                py = float(candidates[pk]["point"]["y"])
                for j, tk in enumerate(reference_keys):
                    qx = float(pins[tk]["point"]["x"])
                    qy = float(pins[tk]["point"]["y"])
                    distance[i, j] = math.hypot(px - qx, py - qy)
                    same_key[i, j] = pk.replace("pin_", "") == tk.replace("pin_", "")
            valid = distance <= tolerance
            reward = valid * (min(len(predicted_keys), len(reference_keys)) + 1) \
                + np.where(valid, np.clip(tolerance - distance, 0, tolerance), 0) \
                + np.where(valid & same_key, 1.0, 0.0)
            rows, cols = linear_sum_assignment(-reward)
            accepted = 0
            for i, j in zip(rows, cols):
                if not valid[i, j]:
                    continue
                expected_name = pins[reference_keys[j]].get("pinname")
                actual_name = candidates[predicted_keys[i]].get("pinname")
                if expected_name not in (None, "") and actual_name not in (None, "") \
                        and str(expected_name) != str(actual_name):
                    continue
                accepted += 1
                matched_pairs.append((pred_component, predicted_keys[i], truth_component, reference_keys[j]))
            tp += accepted
            fn += len(reference_keys) - accepted
            fp += len(predicted_keys) - accepted
            continue
        for key, reference in pins.items():
            candidate = candidates.get(key)
            if candidate is None:
                fn += 1
                continue
            dx = float(candidate["point"]["x"]) - float(reference["point"]["x"])
            dy = float(candidate["point"]["y"]) - float(reference["point"]["y"])
            if math.hypot(dx, dy) > tolerance:
                fn += 1
                continue
            expected_name = reference.get("pinname")
            actual_name = candidate.get("pinname")
            if expected_name is not None and actual_name is not None and str(expected_name) != str(actual_name):
                fn += 1
                continue
            tp += 1
            matched_pairs.append((pred_component, key, truth_component, key))
    for pred_component, pins in pred_pins.items():
        if pred_component in handled:
            continue
        truth_component = component_mapping.get(pred_component)
        if truth_component is None:
            if box_only and component_kind((pred.get("components") or {}), pred_component) not in ("box", "ic"):
                continue
            fp += len(pins)
            continue
        if by_position:
            continue  # already accounted for per component above
        if box_only and component_kind(truth_components, truth_component) not in ("box", "ic"):
            continue
        for key in pins:
            if key not in (truth_pins.get(truth_component) or {}):
                fp += 1
    return tp, fp, fn, matched_pairs


def net_members(network, pins, convention, kinds=None, identity="number"):
    """Resolve a hyperGraph member list to canonical 'component/pin' ids.

    `identity="coarse"` drops the pin number for every non-box component, which is how
    Q&A 评测 #6 describes net comparison: "非 box 原件 pinnumber 不计分，所以 RLC 等在
    hypergraph 中相当于不区分引脚".  Box pins keep their number, because that is what the
    library gives us and what the reference stores.
    """
    coarse = identity == "coarse"
    members = []
    for token in parse_hypergraph(network.get("hyperGraph")):
        if "." in token:
            component, pin = token.rsplit(".", 1)
        else:
            component, pin = token, None
        if coarse and pin is not None and (kinds or {}).get(component) not in ("box", "ic"):
            pin = None
        members.append((component, pin))
    return frozenset(members)


def match_nets(pred, truth, component_mapping, net_pin_identity="number"):
    """Nets are matched by pin-set identity; keys play no role (R08)."""
    truth_kinds = {key: normalise_type(value.get("type"))
                   for key, value in (truth.get("components") or {}).items()}
    pred_kinds = {key: normalise_type(value.get("type"))
                  for key, value in (pred.get("components") or {}).items()}
    truth_sets = {key: net_members(net, truth["pins"], "truth", truth_kinds, net_pin_identity)
                  for key, net in (truth.get("nets") or {}).items()}
    pred_sets = {key: net_members(net, pred["pins"], "pred", pred_kinds, net_pin_identity)
                 for key, net in (pred.get("nets") or {}).items()}
    # translate predicted component keys into the reference key space
    translated = {}
    for key, members in pred_sets.items():
        translated[key] = frozenset((component_mapping.get(component, component), pin) for component, pin in members)
    truth_by_set = defaultdict(list)
    for key, members in truth_sets.items():
        truth_by_set[members].append(key)
    pred_by_set = defaultdict(list)
    for key, members in translated.items():
        pred_by_set[members].append(key)
    shared = set(truth_by_set) & set(pred_by_set)
    tp = 0
    for members in shared:
        tp += min(len(truth_by_set[members]), len(pred_by_set[members]))
    fp = len(pred_sets) - tp
    fn = len(truth_sets) - tp
    return f1(tp, fp, fn), {"truth_nets": len(truth_sets), "pred_nets": len(pred_sets),
                            "shared_pin_sets": len(shared)}


def match_lines(pred, truth, matched_nets, tolerance=LINE_TOLERANCE_PX):
    """Segments are only comparable inside nets that were matched."""
    truth_lookup, pred_lookup = truth, pred
    tp = fp = fn = 0
    for pred_key, truth_key in matched_nets.items():
        reference_edges = (truth.get("nets", {}).get(truth_key) or {}).get("edges") or {}
        candidate_edges = (pred.get("nets", {}).get(pred_key) or {}).get("edges") or {}
        reference_segments = [_segment(points) for points in reference_edges.values()]
        candidate_segments = [_segment(points) for points in candidate_edges.values()]
        used = set()
        for segment in reference_segments:
            hit = None
            for index, other in enumerate(candidate_segments):
                if index in used:
                    continue
                if _segments_close(segment, other, tolerance):
                    hit = index
                    break
            if hit is None:
                fn += 1
            else:
                used.add(hit)
                tp += 1
        fp += len(candidate_segments) - len(used)
    return f1(tp, fp, fn)


def _segment(points):
    return [(float(p["x"]), float(p["y"])) for p in points]


def _segments_close(a, b, tolerance):
    if len(a) < 2 or len(b) < 2:
        return False
    def distances(first, second):
        return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(first, second))
    return (distances(a, b) <= tolerance) or (distances(a, list(reversed(b))) <= tolerance)


def flip_truth_y(truth, height):
    """Convert the parts of a reference that are bottom-left origin into image coordinates.

    Measured, not assumed.  On the public Golden cases:
      * annotation pin tips and net segment points land on rendered ink as given
        (median distance 0 px) but miss it by 5-17 px when flipped -> they are
        already top-left, so they must NOT be flipped
      * the same file's component boxes match the detector only when used as given
        (8/10 matches) and collapse to 2/10 when flipped -> boxes are top-left too

    This helper therefore exists only for references that ARE bottom-left (the
    Dataset300-derived `reference_connectivity.json` path uses it); the public
    Golden JSON needs no flip and callers should leave `flip_reference=False`.
    """
    flipped = {"components": {}, "pins": {}, "nets": json.loads(json.dumps(truth.get("nets") or {}))}
    for key, component in truth["components"].items():
        x1, y1, x2, y2 = component["bbox"]
        flipped["components"][key] = {**component,
                                      "bbox": [x1, height - y2, x2, height - y1]}
    for component, pins in (truth.get("pins") or {}).items():
        flipped["pins"][component] = {
            key: {**pin, "point": {"x": pin["point"]["x"], "y": height - pin["point"]["y"]}}
            for key, pin in pins.items()}
    for key, net in flipped["nets"].items():
        net["edges"] = {edge: [{"x": p["x"], "y": height - p["y"]} for p in points]
                        for edge, points in (net.get("edges") or {}).items()}
    return flipped


def score_case(pred, truth, height, prediction_is_top_left=False, flip_reference=False,
               text_mode="strict", anonymous_components="key", pin_scope="all",
               net_pin_identity="number", nonbox_pin_mode="number"):
    if flip_reference:
        truth = flip_truth_y(truth, height)
    component_mapping, c_tp, c_fp, c_fn = match_components(
        pred, truth, height, prediction_is_top_left=prediction_is_top_left, text_mode=text_mode,
        anonymous_components=anonymous_components)
    p_tp, p_fp, p_fn, _pairs = match_pins(pred, truth, component_mapping, height,
                                          pin_scope=pin_scope, nonbox_pin_mode=nonbox_pin_mode)
    net_metric, net_detail = match_nets(pred, truth, component_mapping,
                                        net_pin_identity=net_pin_identity)
    # nets that were matched one-to-one, for line scoring
    truth_kinds = {key: normalise_type(value.get("type"))
                   for key, value in (truth.get("components") or {}).items()}
    pred_kinds = {key: normalise_type(value.get("type"))
                  for key, value in (pred.get("components") or {}).items()}
    truth_sets = {key: net_members(net, truth["pins"], "truth", truth_kinds, net_pin_identity)
                  for key, net in (truth.get("nets") or {}).items()}
    pred_sets = {key: net_members(net, pred["pins"], "pred", pred_kinds, net_pin_identity)
                 for key, net in (pred.get("nets") or {}).items()}
    translated = {key: frozenset((component_mapping.get(c, c), p) for c, p in members)
                  for key, members in pred_sets.items()}
    matched_nets = {}
    for pred_key, members in translated.items():
        for truth_key, other in truth_sets.items():
            if members == other and truth_key not in matched_nets.values():
                matched_nets[pred_key] = truth_key
                break
    line_metric = match_lines(pred, truth, matched_nets)
    return {
        "component": f1(c_tp, c_fp, c_fn),
        "pin": f1(p_tp, p_fp, p_fn),
        "net_hypergraph": net_metric,
        "net_line": line_metric,
        "net_detail": net_detail,
        "component_matches": len(component_mapping),
    }


def weighted_total(metric):
    return round(sum(WEIGHTS[key] * metric[key]["f1"] for key in WEIGHTS), 4)


def evaluate_cases(pairs, prediction_is_top_left=False, flip_reference=False, text_mode="strict",
                   anonymous_components="key", pin_scope="all", net_pin_identity="number",
                   nonbox_pin_mode="number"):
    """pairs: iterable of (case_id, prediction, truth, height). Micro-averaged."""
    if text_mode not in TEXT_MODES:
        raise ValueError(f"text_mode must be one of {TEXT_MODES}")
    if anonymous_components not in ("key", "by_position"):
        raise ValueError("anonymous_components must be 'key' or 'by_position'")
    if pin_scope not in ("all", "box_only"):
        raise ValueError("pin_scope must be 'all' or 'box_only'")
    if net_pin_identity not in ("number", "coarse"):
        raise ValueError("net_pin_identity must be 'number' or 'coarse'")
    if nonbox_pin_mode not in ("number", "position"):
        raise ValueError("nonbox_pin_mode must be 'number' or 'position'")
    totals = {key: Counter() for key in ("component", "pin", "net_hypergraph", "net_line")}
    per_case = []
    for case_id, prediction, truth, height in pairs:
        result = score_case(prediction, truth, height, prediction_is_top_left=prediction_is_top_left,
                            flip_reference=flip_reference, text_mode=text_mode,
                            anonymous_components=anonymous_components, pin_scope=pin_scope,
                            net_pin_identity=net_pin_identity, nonbox_pin_mode=nonbox_pin_mode)
        for key in totals:
            for field in ("tp", "fp", "fn"):
                totals[key][field] += result[key][field]
        result["case_id"] = case_id
        result["weighted_total"] = weighted_total(result)
        per_case.append(result)
    aggregate = {key: f1(totals[key]["tp"], totals[key]["fp"], totals[key]["fn"]) for key in totals}
    return {"cases": len(per_case),
            "metrics": aggregate,
            "weighted_total": round(sum(WEIGHTS[key] * aggregate[key]["f1"] for key in WEIGHTS), 4),
            "per_case": per_case,
            "weights": WEIGHTS,
            "text_mode": text_mode,
            "anonymous_components": anonymous_components,
            "pin_scope": pin_scope,
            "net_pin_identity": net_pin_identity,
            "nonbox_pin_mode": nonbox_pin_mode,
            "assumptions": ASSUMPTIONS,
            "official": False,
            "note": "local reference implementation; not the official evaluator"}


def component_breakdown(pred, truth, center_tolerance=COMPONENT_CENTER_TOLERANCE_PX,
                        text_mode="strict"):
    """Why did each reference component fail? Counts, in the order the rule applies.

    Returns a dict of reason -> count plus the largest per-side bbox deviation seen on
    key-matched pairs, which is what decides the 20 px item.
    """
    reasons = Counter()
    matched_key_bbox_dev = []
    truth_components = truth.get("components") or {}
    pred_components = pred.get("components") or {}
    for key, reference in truth_components.items():
        candidate = pred_components.get(key)
        if candidate is None:
            reasons["key_missing"] += 1
            continue
        if normalise_type(candidate.get("type")) != normalise_type(reference.get("type")):
            reasons["type_mismatch"] += 1
            continue
        box = [float(v) for v in candidate.get("bbox") or []]
        ref_box = [float(v) for v in reference.get("bbox") or []]
        if len(box) != 4 or len(ref_box) != 4:
            reasons["bbox_missing"] += 1
            continue
        deviation = max(abs(box[i] - ref_box[i]) for i in range(4))
        matched_key_bbox_dev.append(deviation)
        if deviation > center_tolerance:
            reasons["bbox_far"] += 1
            continue
        kind = normalise_type(reference.get("type"))
        if text_mode != "none":
            if kind in ("ic", "box"):
                if not (text_mode == "designator_name" and name_is_designator(reference, key)):
                    if str(reference.get("Name")) != str(candidate.get("Name")):
                        reasons["text_mismatch"] += 1
                        continue
            elif kind in ("r", "c"):
                if str(reference.get("value")) != str(candidate.get("value")):
                    reasons["text_mismatch"] += 1
                    continue
        reasons["ok"] += 1
    reasons["text_mode"] = text_mode
    reasons["max_bbox_deviation_px"] = round(max(matched_key_bbox_dev), 2) if matched_key_bbox_dev else None
    reasons["bbox_dev_over_20px"] = sum(1 for d in matched_key_bbox_dev if d > center_tolerance)
    return dict(reasons)


def load_case(path, height):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, height
