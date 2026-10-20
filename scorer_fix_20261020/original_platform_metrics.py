"""The platform's scoring protocol, transcribed from its own report.

The 2nd-eval artefacts contain the evaluator's own `protocol` block, which differs from
our local reference in three ways that change every number we quote:

  1. ordinary components match by key/type/Name/value, but POWER symbols (ground/supply)
     match by type+value and one-to-one geometry, IGNORING key and Name.  So an
     `UNRESOLVED_DET_0017` ground symbol is not a guaranteed false positive.
  2. PinF1 is scoped to ICs only ("pin_scope": "IC only").  Passive and symbol pins are
     not scored at all, in either direction.
  3. Connectivity merges nets that share a global power rail, then matches nets one-to-one
     with a soft vertex-set F1, instead of demanding an identical pin set.

Rules reproduced here, verbatim from the protocol block:

    component_match  ordinary: key/type/Name/value exact; power: type/value and
                     one-to-one geometry, ignore key/Name; ordered bbox, every coordinate
                     within threshold
    pin_match        component key and pin_<number> exact; owner exists in both
                     components; pinname exact; point Euclidean threshold
    pin_scope        IC only: GT type for known component keys; predicted-only IC pins
                     count as FP
    connectivity     merge nets sharing global power rails, then Hungarian one-to-one
                     mixed-vertex-set soft F1

Assumptions kept explicit: the pin tolerance is R07's 5 px, the component tolerance is the
protocol's 20 px, and the soft F1 of a matched net pair is 2|A∩B|/(|A|+|B|) whose
intersection contributes tp, |A|-|∩| fp and |B|-|∩| fn.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

COMPONENT_TOLERANCE_PX = 20.0
PIN_TOLERANCE_PX = 5.0
WEIGHTS = {"component_f1": 0.30, "pin_f1": 0.25, "connectivity_f1": 0.35, "wire_f1": 0.10}

ALIASES = {
    "resistor": "r", "capacitor": "c", "inductor": "l", "diode": "d", "led": "led",
    "transistor": "bjt", "opamp": "amp", "ic": "box", "ground": "gnd",
    "voltage_source": "v", "vcc": "v", "current_source": "dc", "crystal": "crystal_2pin",
    "pin_port": "pin", "net_bidirectional": "net_bidirection", "battery": "battary",
    "transformer": "transfomer", "raw_block": "block", "optocoupler": "opt",
    "raw_dc": "dc", "raw_m3": "m3螺丝", "raw_m": "m3螺丝",
}
POWER_KIND = {"gnd": "ground", "v": "supply", "vcc": "supply", "vdd": "supply"}
IC_TYPES = {"box", "ic"}


def ntype(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return ALIASES.get(text, text)


def is_ic(value):
    return str(value or "").strip().lower() in IC_TYPES


def f1_block(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def _box_error(a, b):
    if len(a) != 4 or len(b) != 4:
        return float("inf")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def score_components(pred, truth):
    """Ordinary by key, power by one-to-one geometry; every bbox coordinate within 20 px."""
    pred_components = pred.get("components") or {}
    truth_components = truth.get("components") or {}

    def split(components):
        ordinary, power = {}, defaultdict(list)
        for key, item in components.items():
            kind = ntype(item.get("type"))
            if kind in POWER_KIND:
                power[(POWER_KIND[kind], str(item.get("value")))].append((key, item))
            else:
                ordinary[key] = item
        return ordinary, power

    pred_ordinary, pred_power = split(pred_components)
    truth_ordinary, truth_power = split(truth_components)

    tp = 0
    for key, reference in truth_ordinary.items():
        candidate = pred_ordinary.get(key)
        if candidate is None:
            continue
        if ntype(candidate.get("type")) != ntype(reference.get("type")):
            continue
        if str(candidate.get("Name")) != str(reference.get("Name")):
            continue
        if str(candidate.get("value")) != str(reference.get("value")):
            continue
        if _box_error(candidate.get("bbox") or [], reference.get("bbox") or []) > COMPONENT_TOLERANCE_PX:
            continue
        tp += 1

    matched_power = 0
    for group, truth_items in truth_power.items():
        pred_items = pred_power.get(group) or []
        if not pred_items:
            continue
        cost = np.full((len(pred_items), len(truth_items)), 1e6)
        for i, (_pk, pitem) in enumerate(pred_items):
            for j, (_tk, titem) in enumerate(truth_items):
                error = _box_error(pitem.get("bbox") or [], titem.get("bbox") or [])
                if error <= COMPONENT_TOLERANCE_PX:
                    cost[i, j] = error
        rows, cols = linear_sum_assignment(cost)
        matched_power += sum(1 for i, j in zip(rows, cols) if cost[i, j] < 1e6)

    total = tp + matched_power
    fp = len(pred_components) - total
    fn = len(truth_components) - total
    return f1_block(total, fp, fn), {"ordinary_tp": tp, "power_tp": matched_power}


def score_pins(pred, truth, tolerance=None):
    """IC-only scope; key + pin_<number>; owner present on both sides; name exact; 5 px."""
    tolerance = PIN_TOLERANCE_PX if tolerance is None else float(tolerance)
    truth_components = truth.get("components") or {}
    pred_components = pred.get("components") or {}
    truth_pins = truth.get("pins") or {}
    pred_pins = pred.get("pins") or {}

    def ic_keys(components):
        return {key for key, item in components.items() if is_ic(item.get("type"))}

    truth_ic = ic_keys(truth_components)
    pred_ic = ic_keys(pred_components)

    tp = 0
    scored_truth = 0
    for component_key in truth_ic:
        for pin_key, reference in (truth_pins.get(component_key) or {}).items():
            scored_truth += 1
            candidate = (pred_pins.get(component_key) or {}).get(pin_key)
            if candidate is None:
                continue
            if str(candidate.get("pinname")) != str(reference.get("pinname")):
                continue
            dx = float(candidate["point"]["x"]) - float(reference["point"]["x"])
            dy = float(candidate["point"]["y"]) - float(reference["point"]["y"])
            if math.hypot(dx, dy) > tolerance:
                continue
            tp += 1

    scored_pred = sum(len(pred_pins.get(key) or {}) for key in pred_ic)
    fp = scored_pred - tp
    fn = scored_truth - tp
    if scored_truth == 0 and scored_pred == 0:
        return f1_block(0, 0, 0), {"empty_scope": True}
    return f1_block(tp, fp, fn), {"gt_scored": scored_truth, "pred_scored": scored_pred}


def _members(network):
    body = str(network.get("hyperGraph") or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    return [token.strip() for token in body.split(",") if token.strip() and token.strip() not in {"∅", "null", "None"}]


def _vertex_key(token, components, rail_fn):
    if "." in token:
        component, port = token.rsplit(".", 1)
    else:
        component, port = token, None
    kind = ntype((components.get(component) or {}).get("type"))
    rail = rail_fn(component, components)
    if rail is not None:
        return (rail, port)
    if is_ic(kind):
        return (component, port)
    return (component, None)


def _rails(components):
    def rail(component, table):
        item = table.get(component) or {}
        kind = ntype(item.get("type"))
        if kind == "gnd":
            return "GND"
        if kind in ("v", "vcc", "vdd"):
            value = str(item.get("value") or "").strip()
            return value or "VCC"
        return None
    return rail


def score_connectivity(pred, truth):
    truth_components = truth.get("components") or {}
    pred_components = pred.get("components") or {}
    truth_rail, pred_rail = _rails(truth_components), _rails(pred_components)

    truth_nets = [set(_vertex_key(t, truth_components, truth_rail) for t in _members(net))
                  for net in (truth.get("nets") or {}).values()]
    pred_nets = [set(_vertex_key(t, pred_components, pred_rail) for t in _members(net))
                 for net in (pred.get("nets") or {}).values()]
    truth_nets = [n for n in truth_nets if n]
    pred_nets = [n for n in pred_nets if n]
    if not truth_nets or not pred_nets:
        return f1_block(0, len(pred_nets), len(truth_nets))

    score = np.zeros((len(pred_nets), len(truth_nets)))
    for i, p in enumerate(pred_nets):
        for j, t in enumerate(truth_nets):
            inter = len(p & t)
            if inter:
                score[i, j] = 2 * inter / (len(p) + len(t))
    rows, cols = linear_sum_assignment(-score)
    tp = fp = fn = 0
    for i, j in zip(rows, cols):
        if score[i, j] <= 0:
            continue
        inter = len(pred_nets[i] & truth_nets[j])
        tp += inter
        fp += len(pred_nets[i]) - inter
        fn += len(truth_nets[j]) - inter
    matched_pred = {i for i, j in zip(rows, cols) if score[i, j] > 0}
    matched_truth = {j for i, j in zip(rows, cols) if score[i, j] > 0}
    fp += sum(len(pred_nets[i]) for i in range(len(pred_nets)) if i not in matched_pred)
    fn += sum(len(truth_nets[j]) for j in range(len(truth_nets)) if j not in matched_truth)
    return f1_block(tp, fp, fn)


def _segments(net):
    return [[(float(p["x"]), float(p["y"])) for p in points]
            for points in ((net.get("edges") or {}).values())]


def _close(a, b, tolerance=5.0):
    if len(a) < 2 or len(b) < 2:
        return False
    def dist(first, second):
        return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(first, second))
    return dist(a, b) <= tolerance or dist(a, list(reversed(b))) <= tolerance


def score_wire(pred, truth, tolerance=None):
    tolerance = 5.0 if tolerance is None else float(tolerance)
    truth_nets = list((truth.get("nets") or {}).values())
    pred_nets = list((pred.get("nets") or {}).values())
    if not truth_nets or not pred_nets:
        return f1_block(0, sum(len(_segments(n)) for n in pred_nets), sum(len(_segments(n)) for n in truth_nets))
    tp = fp = fn = 0
    used_truth = set()
    for pred_net in pred_nets:
        segments = _segments(pred_net)
        best = None
        for index, truth_net in enumerate(truth_nets):
            if index in used_truth:
                continue
            pool = _segments(truth_net)
            hits = sum(1 for segment in segments
                       if any(_close(segment, other, tolerance) for other in pool))
            if best is None or hits > best[0]:
                best = (hits, index, pool)
        if best is None:
            fp += len(segments)
            continue
        hits, index, pool = best
        used_truth.add(index)
        tp += hits
        fp += len(segments) - hits
        fn += len(pool) - hits
    for index, truth_net in enumerate(truth_nets):
        if index not in used_truth:
            fn += len(_segments(truth_net))
    return f1_block(tp, fp, fn)


def score_case(pred, truth, height=None):
    component, comp_detail = score_components(pred, truth)
    pin, pin_detail = score_pins(pred, truth)
    connectivity = score_connectivity(pred, truth)
    wire = score_wire(pred, truth)
    total = round(WEIGHTS["component_f1"] * component["f1"] + WEIGHTS["pin_f1"] * pin["f1"]
                  + WEIGHTS["connectivity_f1"] * connectivity["f1"] + WEIGHTS["wire_f1"] * wire["f1"], 4)
    return {"component_f1": component, "pin_f1": pin, "connectivity_f1": connectivity,
            "wire_f1": wire, "final_score": total, "detail": {**comp_detail, **pin_detail}}


def evaluate_cases(pairs):
    totals = {key: Counter() for key in ("component_f1", "pin_f1", "connectivity_f1", "wire_f1")}
    per_case = []
    for case_id, prediction, truth, _height in pairs:
        result = score_case(prediction, truth)
        for key in totals:
            for field in ("tp", "fp", "fn"):
                totals[key][field] += result[key][field]
        result["case_id"] = case_id
        per_case.append(result)
    aggregate = {key: f1_block(totals[key]["tp"], totals[key]["fp"], totals[key]["fn"]) for key in totals}
    final = round(sum(WEIGHTS[key] * aggregate[key]["f1"] for key in WEIGHTS), 4)
    return {"cases": len(per_case), "metrics": aggregate, "final_score": final, "per_case": per_case,
            "weights": WEIGHTS, "official": False,
            "note": "protocol transcribed from the platform report; local re-implementation, not the platform"}
