"""New scoring rules (announced 2026-10-18), transcribed into a local evaluator.

Changes against the previous protocol:

  1. the spatial threshold is 20 px for components, IC pins and wire endpoints alike
     (it used to be 20 / 5 / 5);
  2. ordinary components' pins carry no position score and no per-pin connectivity:
     only IC pins are matched as pins;
  3. connectivity is two-level: an ordinary component is aligned as a whole component,
     an IC is still aligned pin by pin;
  4. ground / supply symbols are no longer aligned by their auto-generated key: they are
     matched by category, value and geometry.

Everything else (weights, soft vertex F1, power-rail merge, hyperGraph parsing) is inherited
from `platform_metrics.py`, which reproduces the previous protocol.
"""
import math
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import platform_metrics as old  # noqa: E402

COMPONENT_TOLERANCE_PX = 20.0
PIN_TOLERANCE_PX = 20.0
WIRE_TOLERANCE_PX = 20.0
WEIGHTS = dict(old.WEIGHTS)
f1_block = old.f1_block
ntype = old.ntype
is_ic = old.is_ic
members = old._members
rails = old._rails


def vertex_key(token, components, rail_fn):
    """IC -> (key, port); ordinary part -> (key,); power symbol -> (rail aside)."""
    if "." in token:
        component, port = token.rsplit(".", 1)
    else:
        component, port = token, None
    kind = ntype((components.get(component) or {}).get("type"))
    rail = rail_fn(component, components)
    if rail is not None:
        # Rule 4: the auto-generated pin number of a ground/supply symbol is not part of its
        # identity any more - category (GND vs supply) and value are.
        return (rail,)
    if is_ic(kind):
        return (component, port)
    return (component,)


def nets_of(document):
    components = document.get("components") or {}
    rail = rails(components)
    out = []
    for net in (document.get("nets") or {}).values():
        vertices = {vertex_key(token, components, rail) for token in members(net)}
        if vertices:
            out.append(vertices)
    return out


def score_connectivity(pred, truth):
    truth_nets = nets_of(truth)
    pred_nets = nets_of(pred)
    if not truth_nets or not pred_nets:
        return f1_block(0, len(pred_nets), len(truth_nets))
    score = np.zeros((len(pred_nets), len(truth_nets)))
    for i, p in enumerate(pred_nets):
        for j, t in enumerate(truth_nets):
            inter = len(p & t)
            if inter:
                score[i, j] = 2 * inter / (len(p) + len(t))
    rows, cols = old.linear_sum_assignment(-score)
    tp = fp = fn = 0
    matched_pred, matched_truth = set(), set()
    for i, j in zip(rows, cols):
        if score[i, j] <= 0:
            continue
        inter = len(pred_nets[i] & truth_nets[j])
        tp += inter
        fp += len(pred_nets[i]) - inter
        fn += len(truth_nets[j]) - inter
        matched_pred.add(i)
        matched_truth.add(j)
    for i in range(len(pred_nets)):
        if i not in matched_pred:
            fp += len(pred_nets[i])
    for j in range(len(truth_nets)):
        if j not in matched_truth:
            fn += len(truth_nets[j])
    return f1_block(tp, fp, fn)


def score_pins(pred, truth):
    return old.score_pins(pred, truth, PIN_TOLERANCE_PX)[0]


def _segments(net):
    return old._segments(net)


def score_wire(pred, truth):
    return old.score_wire(pred, truth, WIRE_TOLERANCE_PX)


def score_case(pred, truth, height=None):
    component, _detail = old.score_components(pred, truth)
    pin = score_pins(pred, truth)
    connectivity = score_connectivity(pred, truth)
    wire = score_wire(pred, truth)
    total = round(WEIGHTS["component_f1"] * component["f1"] + WEIGHTS["pin_f1"] * pin["f1"]
                  + WEIGHTS["connectivity_f1"] * connectivity["f1"]
                  + WEIGHTS["wire_f1"] * wire["f1"], 4)
    return {"component_f1": component, "pin_f1": pin, "connectivity_f1": connectivity,
            "wire_f1": wire, "final_score": total}


def evaluate_cases(pairs):
    totals = {key: 0 for key in ("component_f1", "pin_f1", "connectivity_f1", "wire_f1")}
    per_case, scores = [], []
    for case_id, prediction, truth, _height in pairs:
        result = score_case(prediction, truth)
        per_case.append({"case_id": str(case_id), "final_score": result["final_score"],
                         **{key: result[key] for key in totals}})
        scores.append(result["final_score"])
        for key in totals:
            totals[key] += result[key]["f1"]
    count = max(len(pairs), 1)
    metrics = {key: round(value / count, 6) for key, value in totals.items()}
    return {"per_case": per_case, "metrics": metrics,
            "final_score": round(sum(scores) / count, 6)}
