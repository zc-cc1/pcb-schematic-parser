"""Order-independent local reproduction of the platform's wire protocol.

Source: archived platform reports, protocol 2026-10-08-global-power-rails.
Logical nets sharing a typed global rail are pooled first, nets are assigned by
mixed-vertex soft F1, and valid two-point edges are matched one-to-one inside
those pairs. Network IDs break assignment ties deterministically; they are not
used as electrical identities. This module never reads platform result files.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

WIRE_SCORER_VERSION = "logical-rails-hungarian-v1"


def _natural_key(value):
    """Canonical source order: net_2 precedes net_10, independent of JSON order."""
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", str(value)) if part)


@dataclass
class LogicalNet:
    source_ids: tuple
    vertices: frozenset
    edges: tuple


def _vertex(token, components):
    component, port = token.rsplit(".", 1) if "." in token else (token, "")
    item = components.get(component) or {}
    kind = str(item.get("type") or "").strip().lower()
    if kind == "gnd":
        return ("rail", "ground", "GND")
    if kind in {"v", "vcc", "vdd"}:
        value = item.get("value")
        value = "" if value is None else str(value).strip()
        return ("rail", "supply", value or "VCC")
    if kind in {"box", "ic"}:
        return ("ic", component, port)
    return ("ordinary", component)


def _members(net):
    body = str(net.get("hyperGraph") or "").strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    return [token.strip() for token in body.split(",")
            if token.strip() and token.strip() not in {"∅", "null", "None"}]


def _point(value):
    if not isinstance(value, Mapping):
        return None
    coordinates = [value.get("x"), value.get("y")]
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in coordinates):
        return None
    return tuple(map(float, coordinates))


def _edge(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = map(_point, value)
    if a is None or b is None:
        return None
    # Canonical direction also stabilizes assignment diagnostics on reversed edges.
    return min((a, b), (b, a))


def logical_nets(document):
    """Pool source branches through rail references, retaining edge multiplicity."""
    components = document.get("components") or {}
    raw = document.get("nets") or {}
    if not isinstance(raw, Mapping):
        return []
    source = []
    for nid, net in sorted(raw.items(), key=lambda item: (_natural_key(item[0]), str(item[0]))):
        if not isinstance(net, Mapping):
            continue
        vertices = frozenset(_vertex(token, components) for token in _members(net))
        records = net.get("edges") or {}
        if not isinstance(records, Mapping):
            records = {}
        edges = tuple((str(eid), _edge(value)) for eid, value in
                      sorted(records.items(), key=lambda item: str(item[0])))
        source.append(LogicalNet((str(nid),), vertices, edges))
    parent = list(range(len(source)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    rails = {}
    for i, net in enumerate(source):
        for vertex in sorted(net.vertices):
            if vertex[0] != "rail":
                continue
            if vertex in rails:
                a, b = find(i), find(rails[vertex])
                parent[max(a, b)] = min(a, b)
            else:
                rails[vertex] = i
    groups = {}
    for i, net in enumerate(source):
        groups.setdefault(find(i), []).append(net)
    result = []
    for group in groups.values():
        ids = tuple(sorted(nid for net in group for nid in net.source_ids))
        vertices = frozenset(vertex for net in group for vertex in net.vertices)
        # Keep the source ID with each edge for deterministic ordering after pooling.
        edges = tuple((f"{net.source_ids[0]}/{eid}", edge)
                      for net in group for eid, edge in net.edges)
        result.append(LogicalNet(ids, vertices, edges))
    return sorted(result, key=lambda net: (min(_natural_key(nid) for nid in net.source_ids), net.source_ids))


def _stable_assignment(cost):
    """Hungarian shortest augmenting paths with ascending-index tie traversal.

    Different optimal assignments can yield different downstream wire counts.
    Keep this tie policy explicit rather than depending on a solver's incidental
    tie handling. Canonical net order makes it independent of JSON insertion.
    """
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.T if transposed else cost
    n, m = matrix.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, previous = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        owner[0], column = row, 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[column] = True
            current_row, delta, next_column = owner[column], np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = matrix[current_row - 1, j - 1] - u[current_row] - v[j]
                if reduced < minimum[j]:
                    minimum[j], previous[j] = reduced, column
                if minimum[j] < delta:
                    delta, next_column = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while True:
            next_column = previous[column]
            owner[column], column = owner[next_column], next_column
            if column == 0:
                break
    pairs = [(int(owner[j] - 1), j - 1) for j in range(1, m + 1) if owner[j]]
    return [(j, i) for i, j in pairs] if transposed else pairs


def _assign_nets(pred_nets, truth_nets):
    similarity = np.zeros((len(truth_nets), len(pred_nets)))
    for i, truth in enumerate(truth_nets):
        for j, pred in enumerate(pred_nets):
            denominator = len(truth.vertices) + len(pred.vertices)
            if denominator:
                similarity[i, j] = 2 * len(truth.vertices & pred.vertices) / denominator
    if not similarity.size:
        return []
    return [(i, j, float(similarity[i, j]))
            for i, j in sorted(_stable_assignment(1.0 - similarity))
            if similarity[i, j] > 0]


def _match_edges(pred_edges, truth_edges, tolerance):
    pred = [edge for _, edge in pred_edges if edge is not None]
    truth = [edge for _, edge in truth_edges if edge is not None]
    if not pred or not truth:
        return 0
    p, t = np.asarray(pred), np.asarray(truth)
    forward = np.linalg.norm(p[:, None] - t[None, :], axis=3).max(axis=2)
    reverse = np.linalg.norm(p[:, None] - t[None, :, ::-1], axis=3).max(axis=2)
    eligible = np.minimum(forward, reverse) <= tolerance
    rows, cols = linear_sum_assignment(-eligible.astype(np.int8))
    return int(eligible[rows, cols].sum())


def score_wire_detailed(prediction, truth, tolerance=20.0):
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Wire tolerance must be finite and nonnegative")
    pn, tn = logical_nets(prediction), logical_nets(truth)
    pairs = _assign_nets(pn, tn)
    tp = sum(_match_edges(pn[j].edges, tn[i].edges, tolerance) for i, j, _ in pairs)
    pcount = sum(len(net.edges) for net in pn)
    tcount = sum(len(net.edges) for net in tn)
    fp, fn = pcount - tp, tcount - tp
    precision, recall = tp / pcount if pcount else 0.0, tp / tcount if tcount else 0.0
    block = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
             "recall": round(recall, 4), "f1": round(2 * tp / (pcount + tcount), 4)
             if pcount + tcount else 0.0}
    diagnostics = {"version": WIRE_SCORER_VERSION, "tolerance_px": tolerance,
        "pred_logical_net_count": len(pn), "gt_logical_net_count": len(tn),
        "invalid_pred_segment_count": sum(edge is None for n in pn for _, edge in n.edges),
        "matched_net_pairs": [{"gt_net_ids": list(tn[i].source_ids),
                               "pred_net_ids": list(pn[j].source_ids),
                               "pin_set_f1": score} for i, j, score in pairs]}
    return block, diagnostics


def score_wire(prediction, truth, tolerance=20.0):
    return score_wire_detailed(prediction, truth, tolerance)[0]
