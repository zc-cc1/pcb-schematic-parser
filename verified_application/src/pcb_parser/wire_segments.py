"""Model mask to two-point geometry, independent of annotation and net identity."""
import math

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .wire_graph import compress_skeleton, pixel_adjacency, polyline


DEFAULTS = dict(wire_threshold=.5, polyline_tolerance_px=2., prune_spur_length_px=2.,
                fit_lines=True, axis_angle_degrees=3., maximum_gap_px=4.,
                merge_angle_degrees=8., merge_offset_px=1.5, merge_min_fragment_px=6.,
                merge_min_support=.8, support_radius_px=1.5, pin_guard_radius_px=2.)


def _prune(skeleton, maximum):
    cleaned = skeleton.copy()
    if maximum <= 0:
        return cleaned, 0
    removed = 0
    for _ in range(2):
        adjacent = pixel_adjacency(cleaned)
        remove = set()
        for start in sorted(p for p, neighbours in adjacent.items() if len(neighbours) == 1):
            path = [start]; previous = None; current = start; length = 0.
            while True:
                following = [q for q in adjacent[current] if q != previous]
                if not following or (current != start and len(adjacent[current]) != 2):
                    break
                nxt = following[0]; length += math.dist(current, nxt)
                path.append(nxt); previous, current = current, nxt
                if length > maximum:
                    break
            if length <= maximum and len(adjacent[current]) >= 3:
                remove.update(path[:-1])
        if not remove:
            break
        for y, x in remove:
            cleaned[y, x] = False
        removed += len(remove)
    return cleaned, removed


def _fit(points, axis_angle):
    points = np.asarray(points, float)
    a, b = points[0].copy(), points[-1].copy()
    if len(points) <= 2 or np.linalg.norm(b-a) < 3:
        return np.stack((a, b))
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh((points-center).T@(points-center))
    direction = vectors[:, -1]
    if np.dot(direction, b-a) < 0:
        direction = -direction
    sine = math.sin(math.radians(axis_angle))
    if abs(direction[1]) <= sine:
        return np.array([[a[0], np.median(points[:, 1])], [b[0], np.median(points[:, 1])]])
    if abs(direction[0]) <= sine:
        return np.array([[np.median(points[:, 0]), a[1]], [np.median(points[:, 0]), b[1]]])
    return np.stack((center+np.dot(a-center, direction)*direction,
                     center+np.dot(b-center, direction)*direction))


def _records(graph, cfg):
    result = []
    for u, v, data in graph.edges(data=True):
        points = np.asarray(data["path"], float)
        simplified = polyline(points, cfg["polyline_tolerance_px"])
        indices = [0]; cursor = 0
        for point in simplified[1:-1]:
            candidates = np.flatnonzero(np.all(points[cursor+1:] == point, axis=1))
            cursor += int(candidates[0])+1
            indices.append(cursor)
        indices.append(len(points)-1)
        for part, (begin, end) in enumerate(zip(indices, indices[1:])):
            fitted = _fit(points[begin:end+1], cfg["axis_angle_degrees"]) if cfg["fit_lines"] else points[[begin, end]].copy()
            protected = [False, False]
            for side, index in ((0, begin), (1, end)):
                if index not in (0, len(points)-1):
                    # Real bend, not a dangling fragment to merge across.
                    protected[side] = True
                    fitted[side] = points[index]
                else:
                    node = min((u, v), key=lambda n: math.dist(points[index], graph.nodes[n]["point"]))
                    protected[side] = graph.nodes[node]["kind"] != "endpoint"
                    if protected[side]:
                        fitted[side] = points[index]
            if math.dist(*fitted) > 0:
                result.append(dict(points=fitted.tolist(), protected=protected,
                                   source_fragments=[f"{data['edge_id']}:{part}"], bridges=[]))
    return result


def _pin_pair_at_gap(a, b, pins, radius):
    if not len(pins) or radius <= 0:
        return False
    near_a = np.flatnonzero(np.linalg.norm(pins-a, axis=1) <= radius)
    near_b = np.flatnonzero(np.linalg.norm(pins-b, axis=1) <= radius)
    return any(i != j and np.linalg.norm(pins[i]-pins[j]) > 1 for i in near_a for j in near_b)


def _candidate(left, right, cfg, tree, blocked, pins):
    if not left["source_fragments"] or not right["source_fragments"]:
        return None
    p, q = np.asarray(left["points"]), np.asarray(right["points"])
    choices = [(np.linalg.norm(p[i]-q[j]), i, j) for i in (0, 1) for j in (0, 1)]
    distance, i, j = min(choices)
    if distance > cfg["maximum_gap_px"] or left["protected"][i] or right["protected"][j]:
        return None
    outer_a, a, b, outer_b = p[1-i], p[i], q[j], q[1-j]
    va, vb = a-outer_a, outer_b-b
    la, lb = np.linalg.norm(va), np.linalg.norm(vb)
    if min(la, lb) < cfg["merge_min_fragment_px"]:
        return None
    da, db = va/la, vb/lb
    if np.dot(da, db) < math.cos(math.radians(cfg["merge_angle_degrees"])):
        return None
    delta = b-a
    if np.dot(delta, da) < -.5 or np.dot(delta, db) < -.5:
        return None
    cross = lambda v, d: abs(float(v[0]*d[1]-v[1]*d[0]))
    if max(cross(delta, da), cross(delta, db), cross(outer_b-outer_a, da), cross(outer_b-outer_a, db)) > cfg["merge_offset_px"]:
        return None
    if _pin_pair_at_gap(a, b, pins, cfg["pin_guard_radius_px"]):
        return None
    bridge = np.linspace(a, b, max(2, math.ceil(distance)+1))
    xy = np.rint(bridge).astype(int)
    if (xy < 0).any() or (xy[:, 0] >= blocked.shape[1]).any() or (xy[:, 1] >= blocked.shape[0]).any():
        return None
    if blocked[xy[:, 1], xy[:, 0]].any():
        return None
    merged = np.stack((outer_a, outer_b))
    if cfg["fit_lines"]:
        # Fit orientation using existing fragments only, never invented bridge pixels.
        samples = np.vstack((np.linspace(*p, max(2, math.ceil(la)+1)), np.linspace(*q, max(2, math.ceil(lb)+1))))
        direction = outer_b-outer_a
        if abs(direction[1]) <= math.sin(math.radians(cfg["axis_angle_degrees"]))*np.linalg.norm(direction):
            merged[:, 1] = np.median(samples[:, 1])
        elif abs(direction[0]) <= math.sin(math.radians(cfg["axis_angle_degrees"]))*np.linalg.norm(direction):
            merged[:, 0] = np.median(samples[:, 0])
    scan = np.linspace(*merged, max(2, math.ceil(math.dist(*merged))+1))
    support = float(np.mean(tree.query(scan)[0] <= cfg["support_radius_px"]))
    if support < cfg["merge_min_support"]:
        return None
    return dict(points=merged.tolist(), protected=[left["protected"][1-i], right["protected"][1-j]],
                source_fragments=left["source_fragments"]+right["source_fragments"],
                bridges=left["bridges"]+right["bridges"]+[dict(points=[a.tolist(), b.tolist()], gap_px=float(distance), support=support)])


def _merge(records, cfg, mask, blocked, pins):
    if cfg["maximum_gap_px"] <= 0 or not records:
        return records
    y, x = np.where(mask)
    tree = cKDTree(np.column_stack((x, y)))
    live = {i: r for i, r in enumerate(records)}; next_id = len(live)
    # Rebuild after each pass so a chain of several genuine fragments can combine.
    while True:
        ids = sorted(live)
        points = np.array([p for i in ids for p in live[i]["points"]])
        neighbours = cKDTree(points).query_pairs(cfg["maximum_gap_px"])
        pairs = {(ids[a//2], ids[b//2]) for a, b in neighbours if a//2 != b//2}
        proposals = []
        for a, b in sorted(pairs):
            merged = _candidate(live[a], live[b], cfg, tree, blocked, pins)
            if merged is not None:
                proposals.append((merged["bridges"][-1]["gap_px"], a, b, merged))
        changed = False
        for _, a, b, merged in sorted(proposals, key=lambda item: item[:3]):
            if a not in live or b not in live:
                continue
            del live[a]; del live[b]; live[next_id] = merged; next_id += 1; changed = True
        if not changed:
            break
    return list(live.values())


def component_bridge_guard(shape, components):
    """Protect model-detected symbol bodies from vector bridges; leave pixels intact."""
    height, width = shape
    guard = np.zeros(shape, bool)
    for component in components:
        if component.get("confidence", 0) < .5:
            continue
        box = component.get("bbox_top_left")
        if box is None or len(box) != 4 or not np.isfinite(box).all():
            continue
        x0, y0, x1, y1 = map(float, box)
        if x1 <= x0 or y1 <= y0:
            continue
        inset = max(1., min(3., .1*min(x1-x0, y1-y0)))
        a, b = max(0, math.ceil(x0+inset)), max(0, math.ceil(y0+inset))
        c, d = min(width, math.floor(x1-inset)+1), min(height, math.floor(y1-inset)+1)
        if a < c and b < d:
            guard[b:d, a:c] = True
    return guard


def decode_wire_segments(probability, config=None, blocked_mask=None, pin_points=None, bridge_blocked_mask=None):
    """Return geometry candidates, never assign a net or consult an annotation."""
    cfg = DEFAULTS | (config or {})
    p = np.asarray(probability)
    if p.ndim != 2 or not all(p.shape) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite HxW model probabilities in [0,1]")
    for key in ("polyline_tolerance_px", "prune_spur_length_px", "axis_angle_degrees", "maximum_gap_px", "merge_angle_degrees", "merge_offset_px", "merge_min_fragment_px", "support_radius_px", "pin_guard_radius_px"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["wire_threshold"] <= 1 or not 0 <= cfg["merge_min_support"] <= 1:
        raise ValueError("Invalid threshold or support fraction")
    blocked = np.zeros_like(p, bool) if blocked_mask is None else np.asarray(blocked_mask, bool)
    if blocked.shape != p.shape:
        raise ValueError("Blocked region shape differs from probability")
    bridge_guard = blocked if bridge_blocked_mask is None else np.asarray(bridge_blocked_mask, bool) | blocked
    if bridge_guard.shape != p.shape:
        raise ValueError("Bridge guard shape differs from probability")
    pins = np.asarray(pin_points if pin_points is not None else [], float).reshape(-1, 2)
    if not np.isfinite(pins).all():
        raise ValueError("Invalid pin points")
    mask = (p >= cfg["wire_threshold"]) & ~blocked
    skeleton, removed = _prune(skeletonize(mask, method="zhang"), cfg["prune_spur_length_px"])
    graph = compress_skeleton(skeleton, p, {"polyline_tolerance_px": cfg["polyline_tolerance_px"]})
    records = _records(graph, cfg)
    before = len(records)
    records = _merge(records, cfg, mask, bridge_guard, pins)
    seen = set(); segments = []
    height, width = p.shape
    for record in records:
        points = np.asarray(record["points"], float)
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width-1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height-1))
        key = min(tuple(map(tuple, np.round(points, 6))), tuple(map(tuple, np.round(points[::-1], 6))))
        if key in seen or math.dist(*points) <= 0:
            continue
        seen.add(key)
        segments.append(dict(segment_id=f"edge_{len(segments)+1}", points=points.tolist(),
                             source_fragments=record["source_fragments"], bridges=record["bridges"], net_id=None))
    return dict(status="geometry_candidates_net_membership_pending", coordinate_origin="top_left", size=[width, height],
                segments=segments, summary=dict(segments=len(segments), before_gap_merging=before,
                    merged_gaps=sum(len(r["bridges"]) for r in segments), pruned_spur_pixels=removed,
                    bridge_guard_pixels=int(bridge_guard.sum()),
                    original_mask_modified=False, net_membership_inferred=False), config=cfg,
                reference_NetLineF1=None, reference_annotations_used=False)


def export_candidate_edges(result):
    """Geometry only: do not misrepresent unassigned segments as a complete net."""
    height = result["size"][1]
    edges = {r["segment_id"]: [dict(x=float(x), y=float(height-y)) for x, y in r["points"]]
             for r in result["segments"]}
    return dict(status="unassigned_geometry_not_complete_target_JSON", coordinate_origin="bottom_left",
                coordinate_transform="H-y (existing local convention)", edges=edges, nets_assigned=False)
