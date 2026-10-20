"""A pinned copy of the contest metric rules, for reproducible open_GT numbers.

`src/pcb_parser/contest_metrics.py` is under active revision (it broke mid-edit twice while
this evaluation ran), so a score quoted from it is not reproducible.  This module restates
the rules exactly as they read in the working version of 2026-10-18 02:31 and is what the
open_GT reports use.  It is a *local reference* implementation, not the official evaluator.

Rules (docs/requirements.md):
  R02/R03/R04/R07  component: match BY KEY; type equal; every bbox coordinate within 20 px;
                   box/ic also require Name equality, r/c require value equality
  R05/R06/R07      pin: component key + pin number + point within 5 px; pinname compared
                   strictly (a missing name on either side is not a mismatch)
  R08/R09          net: matched by pin-set identity, net key ignored
  R10/R11          line: segments compared only inside matched net pairs, 5 px endpoints
  R16              weights .30/.25/.35/.10
  R17              a case with no prediction still counts in the denominator
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

COMPONENT_TOLERANCE_PX = 20.0
PIN_TOLERANCE_PX = 5.0
LINE_TOLERANCE_PX = 5.0
WEIGHTS = {"component": 0.30, "pin": 0.25, "net_hypergraph": 0.35, "net_line": 0.10}

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


def normalise_type(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return TYPE_ALIASES.get(text, text)


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def match_components(pred, truth, height=None, tolerance=COMPONENT_TOLERANCE_PX):
    truth_components = truth.get("components") or {}
    pred_components = pred.get("components") or {}
    if not truth_components or not pred_components:
        return {}, 0, len(pred_components), len(truth_components)
    mapping, tp = {}, 0
    for key, reference in truth_components.items():
        candidate = pred_components.get(key)
        if candidate is None:
            continue
        if normalise_type(candidate.get("type")) != normalise_type(reference.get("type")):
            continue
        box = [float(v) for v in candidate.get("bbox") or []]
        ref_box = [float(v) for v in reference.get("bbox") or []]
        if len(box) != 4 or len(ref_box) != 4:
            continue
        if any(abs(box[i] - ref_box[i]) > tolerance for i in range(4)):
            continue
        kind = normalise_type(reference.get("type"))
        if kind in ("ic", "box"):
            if str(reference.get("Name")) != str(candidate.get("Name")):
                continue
        elif kind in ("r", "c"):
            if str(reference.get("value")) != str(candidate.get("value")):
                continue
        mapping[key] = key
        tp += 1
    return mapping, tp, len(pred_components) - len(mapping), len(truth_components) - len(mapping)


def match_pins(pred, truth, component_mapping, tolerance=PIN_TOLERANCE_PX):
    reverse = {}
    for truth_key in (truth.get("pins") or {}):
        if truth_key in (pred.get("pins") or {}):
            reverse[truth_key] = truth_key
    for pred_key, truth_key in component_mapping.items():
        reverse.setdefault(truth_key, pred_key)

    tp, fp, fn = 0, 0, 0
    truth_pins = truth.get("pins") or {}
    pred_pins = pred.get("pins") or {}
    for truth_component, pins in truth_pins.items():
        pred_component = reverse.get(truth_component)
        if pred_component is None:
            fn += len(pins)
            continue
        candidates = pred_pins.get(pred_component) or {}
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
    for pred_component, pins in pred_pins.items():
        truth_component = component_mapping.get(pred_component)
        if truth_component is None:
            fp += len(pins)
            continue
        for key in pins:
            if key not in (truth_pins.get(truth_component) or {}):
                fp += 1
    return tp, fp, fn


def parse_hypergraph(text):
    body = str(text or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    members = []
    for token in body.split(","):
        token = token.strip()
        if token and token not in {"∅", "null", "None"}:
            members.append(token)
    return members


def net_members(network):
    members = []
    for token in parse_hypergraph(network.get("hyperGraph")):
        if "." in token:
            component, pin = token.rsplit(".", 1)
            members.append((component, pin))
        else:
            members.append((token, None))
    return frozenset(members)


def match_nets(pred, truth, component_mapping):
    truth_sets = {key: net_members(net) for key, net in (truth.get("nets") or {}).items()}
    pred_sets = {key: net_members(net) for key, net in (pred.get("nets") or {}).items()}
    translated = {key: frozenset((component_mapping.get(component, component), pin) for component, pin in members)
                  for key, members in pred_sets.items()}
    truth_by_set = defaultdict(list)
    for key, members in truth_sets.items():
        truth_by_set[members].append(key)
    pred_by_set = defaultdict(list)
    for key, members in translated.items():
        pred_by_set[members].append(key)
    shared = set(truth_by_set) & set(pred_by_set)
    tp = sum(min(len(truth_by_set[members]), len(pred_by_set[members])) for members in shared)
    return f1(tp, len(pred_sets) - tp, len(truth_sets) - tp), {
        "truth_nets": len(truth_sets), "pred_nets": len(pred_sets), "shared_pin_sets": len(shared)}


def _segment(points):
    return [(float(p["x"]), float(p["y"])) for p in points]


def _segments_close(a, b, tolerance):
    if len(a) < 2 or len(b) < 2:
        return False
    def distances(first, second):
        return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(first, second))
    return (distances(a, b) <= tolerance) or (distances(a, list(reversed(b))) <= tolerance)


def match_lines(pred, truth, matched_nets, tolerance=LINE_TOLERANCE_PX):
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
    return f1(tp, fp, fn), {}


def score_case(pred, truth, height=None):
    component_mapping, c_tp, c_fp, c_fn = match_components(pred, truth)
    p_tp, p_fp, p_fn = match_pins(pred, truth, component_mapping)
    net_metric, net_detail = match_nets(pred, truth, component_mapping)

    truth_sets = {key: net_members(net) for key, net in (truth.get("nets") or {}).items()}
    pred_sets = {key: net_members(net) for key, net in (pred.get("nets") or {}).items()}
    translated = {key: frozenset((component_mapping.get(c, c), p) for c, p in members)
                  for key, members in pred_sets.items()}
    matched_nets = {}
    for pred_key, members in translated.items():
        for truth_key, other in truth_sets.items():
            if members == other and truth_key not in matched_nets.values():
                matched_nets[pred_key] = truth_key
                break
    line_metric, _ = match_lines(pred, truth, matched_nets)
    return {"component": f1(c_tp, c_fp, c_fn), "pin": f1(p_tp, p_fp, p_fn),
            "net_hypergraph": net_metric, "net_line": line_metric, "net_detail": net_detail,
            "component_matches": len(component_mapping)}


def weighted_total(metric):
    return round(sum(WEIGHTS[key] * metric[key]["f1"] for key in WEIGHTS), 4)


def evaluate_cases(pairs, prediction_is_top_left=False, flip_reference=False):
    totals = {key: Counter() for key in ("component", "pin", "net_hypergraph", "net_line")}
    per_case = []
    for case_id, prediction, truth, height in pairs:
        result = score_case(prediction, truth, height)
        for key in totals:
            for field in ("tp", "fp", "fn"):
                totals[key][field] += result[key][field]
        result["case_id"] = case_id
        result["weighted_total"] = weighted_total(result)
        per_case.append(result)
    aggregate = {key: f1(totals[key]["tp"], totals[key]["fp"], totals[key]["fn"]) for key in totals}
    return {"cases": len(per_case), "metrics": aggregate,
            "weighted_total": round(sum(WEIGHTS[key] * aggregate[key]["f1"] for key in WEIGHTS), 4),
            "per_case": per_case, "weights": WEIGHTS, "official": False,
            "note": "pinned local reference implementation; not the official evaluator"}
