"""Refine two-point geometry with predicted tips and conservative T nodes only."""
from copy import deepcopy
import math
import numpy as np


DEFAULTS = dict(enabled=True, pin_confidence=.9, snap_radius_px=3., maximum_perpendicular_px=2.,
                pin_choice_margin_px=.75, split_at_pins=False, split_at_T_nodes=False,
                interior_margin_px=6., minimum_piece_length_px=6., node_radius_px=3.,
                minimum_branch_length_px=10., minimum_branch_angle_degrees=45.)


def _projection(point, segment):
    a, b = np.asarray(segment, float)
    delta = b-a; squared = float(np.dot(delta, delta))
    t = float(np.dot(np.asarray(point)-a, delta)/squared) if squared else 0.
    projected = a+np.clip(t, 0, 1)*delta
    return projected, t, float(np.linalg.norm(projected-point))


def _blocked(a, b, guard):
    if guard is None:
        return False
    scan = np.rint(np.linspace(a, b, max(2, math.ceil(math.dist(a, b))+1))).astype(int)
    if (scan < 0).any() or (scan[:, 0] >= guard.shape[1]).any() or (scan[:, 1] >= guard.shape[0]).any():
        return True
    return bool(guard[scan[:, 1], scan[:, 0]].any())


def refine_segment_nodes(source, pin_candidates, config=None, bridge_guard=None):
    """No GT/annotation arguments; a node is geometric, not an electrical join."""
    cfg = DEFAULTS | (config or {})
    for key in ("snap_radius_px", "maximum_perpendicular_px", "pin_choice_margin_px", "interior_margin_px",
                "minimum_piece_length_px", "node_radius_px", "minimum_branch_length_px", "minimum_branch_angle_degrees"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= cfg["pin_confidence"] <= 1:
        raise ValueError("Invalid pin confidence")
    if source.get("coordinate_origin") != "top_left":
        raise ValueError("Expected top-left source geometry")
    width, height = source["size"]
    guard = None if bridge_guard is None else np.asarray(bridge_guard, bool)
    if guard is not None and guard.shape != (height, width):
        raise ValueError("Invalid body guard shape")
    result = deepcopy(source)
    if not cfg["enabled"]:
        result["node_refinement"] = dict(config=cfg, status="disabled", snapped_endpoints=0, interior_splits=0)
        return result
    pins = []
    for index, pin in enumerate(pin_candidates):
        point = np.asarray(pin.get("point", []), float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Invalid predicted pin")
        if pin.get("confidence", 0) >= cfg["pin_confidence"] and 0 <= point[0] < width and 0 <= point[1] < height:
            if not any(np.linalg.norm(point-p["point"]) < 1 for p in pins):
                pins.append(dict(point=point, index=index, confidence=float(pin["confidence"])))
    records = result["segments"]
    changes = []
    for index, segment in enumerate(records):
        points = np.asarray(segment["points"], float)
        if points.shape != (2, 2) or not np.isfinite(points).all() or math.dist(*points) <= 0:
            raise ValueError("Invalid nonzero two-point segment")
        original = points.copy()
        choices = []
        for side in (0, 1):
            ranked = []
            for pin in pins:
                _, t, _ = _projection(pin["point"], original)
                line = original[0]+t*(original[1]-original[0])
                normal = float(np.linalg.norm(line-pin["point"]))
                distance = float(np.linalg.norm(original[side]-pin["point"]))
                if distance > cfg["snap_radius_px"] or normal > cfg["maximum_perpendicular_px"]:
                    continue
                if math.dist(pin["point"], points[1-side]) < cfg["minimum_piece_length_px"]:
                    continue
                if _blocked(original[side], pin["point"], guard):
                    continue
                ranked.append((distance, pin["index"], pin))
            ranked.sort(key=lambda r: (r[0], r[1]))
            if ranked and (len(ranked)==1 or ranked[1][0]-ranked[0][0] >= cfg["pin_choice_margin_px"]):
                choices.append((ranked[0][0], side, ranked[0][2]))
        used = set()
        for distance, side, pin in sorted(choices, key=lambda c: (c[0], c[1])):
            if pin["index"] in used or distance < 1e-6:
                continue
            used.add(pin["index"]); points[side] = pin["point"]
            changes.append(dict(kind="endpoint_snap", segment_index=index, side=side, old=original[side].tolist(),
                                point=pin["point"].tolist(), pin_index=pin["index"], distance=distance))
        segment["points"] = points.tolist()
    cuts = {i: [] for i in range(len(records))}
    if cfg["split_at_pins"]:
        for index, segment in enumerate(records):
            points = np.asarray(segment["points"]); length = math.dist(*points)
            for pin in pins:
                projected, t, distance = _projection(pin["point"], points)
                if min(t, 1-t)*length < cfg["interior_margin_px"] or distance > cfg["maximum_perpendicular_px"]:
                    continue
                if _blocked(projected, pin["point"], guard):
                    continue
                cuts[index].append(dict(point=projected.tolist(), t=t, kind="predicted_pin_projection", pin_index=pin["index"]))
    ambiguities = []
    if cfg["split_at_T_nodes"]:
        candidates = []
        max_dot = math.cos(math.radians(cfg["minimum_branch_angle_degrees"]))
        for branch_index, branch in enumerate(records):
            branch_points = np.asarray(branch["points"]); branch_length = math.dist(*branch_points)
            if branch_length < cfg["minimum_branch_length_px"]:
                continue
            for side in (0, 1):
                endpoint = branch_points[side]
                direction = (endpoint-branch_points[1-side])/branch_length
                targets = []
                for target_index, target in enumerate(records):
                    if branch_index == target_index:
                        continue
                    points = np.asarray(target["points"]); length = math.dist(*points)
                    projected, t, distance = _projection(endpoint, points)
                    if distance > cfg["node_radius_px"] or min(t, 1-t)*length < cfg["interior_margin_px"]:
                        continue
                    unit = (points[1]-points[0])/length
                    if abs(float(np.dot(unit, direction))) > max_dot:
                        continue
                    delta = projected-endpoint
                    if np.dot(delta, direction) < -.5 or _blocked(endpoint, projected, guard):
                        continue
                    targets.append((distance, target_index, projected, t))
                targets.sort(key=lambda item: (item[0], item[1]))
                if not targets or (len(targets)>1 and targets[1][0]-targets[0][0] < cfg["pin_choice_margin_px"]):
                    continue
                distance, target, projected, t = targets[0]
                candidates.append(dict(branch=branch_index, side=side, target=target, point=projected, t=t, direction=direction))
        accepted_ends = set()
        for candidate in candidates:
            crossing = any(other is not candidate and other["target"]==candidate["target"]
                           and np.linalg.norm(other["point"]-candidate["point"]) <= 2*cfg["node_radius_px"]
                           and np.dot(other["direction"], candidate["direction"]) < -.85 for other in candidates)
            if crossing:
                ambiguities.append(dict(branch=candidate["branch"], point=candidate["point"].tolist(), reason="opposing_arms_possible_crossing"))
                continue
            key = (candidate["branch"], candidate["side"])
            if key in accepted_ends:
                continue
            accepted_ends.add(key)
            branch = records[candidate["branch"]]
            before = branch["points"][candidate["side"]]
            branch["points"][candidate["side"]] = candidate["point"].tolist()
            cuts[candidate["target"]].append(dict(point=candidate["point"].tolist(), t=candidate["t"], kind="predicted_branch_endpoint"))
            changes.append(dict(kind="T_endpoint_projection", segment_index=candidate["branch"], side=candidate["side"], old=before,
                                point=candidate["point"].tolist(), target_segment=candidate["target"]))
    output = []; cut_events = []
    for index, segment in enumerate(records):
        a, b = np.asarray(segment["points"], float)
        length = float(np.linalg.norm(b-a)); ordered = []
        for cut in sorted(cuts[index], key=lambda c: c["t"]):
            point, t, distance = _projection(np.asarray(cut["point"]), (a, b))
            if min(t, 1-t)*length < cfg["interior_margin_px"] or distance > cfg["maximum_perpendicular_px"]:
                continue
            if ordered and np.linalg.norm(point-np.asarray(ordered[-1]["point"])) < cfg["minimum_piece_length_px"]:
                continue
            ordered.append(cut | dict(point=point.tolist(), t=t))
        bounds = [a.tolist()]+[c["point"] for c in ordered]+[b.tolist()]
        if any(math.dist(x, y) < cfg["minimum_piece_length_px"] for x, y in zip(bounds, bounds[1:])) and ordered:
            ordered = []; bounds = [a.tolist(), b.tolist()]
        for left, right in zip(bounds, bounds[1:]):
            child = deepcopy(segment)
            child.update(segment_id=f"edge_{len(output)+1}", points=[left, right], source_segment_id=segment["segment_id"],
                         node_cut_count=len(ordered), net_id=None)
            output.append(child)
        cut_events.extend(dict(segment_index=index, **cut) for cut in ordered)
    result["segments"] = output
    result["summary"] = dict(result["summary"], segments=len(output), node_snapped_endpoints=sum(c["kind"]=="endpoint_snap" for c in changes),
                             node_T_projections=sum(c["kind"]=="T_endpoint_projection" for c in changes), node_interior_splits=len(cut_events))
    result["node_refinement"] = dict(config=cfg, changes=changes, cuts=cut_events, uncertain_geometric_nodes=ambiguities,
                                     snapped_endpoints=result["summary"]["node_snapped_endpoints"], interior_splits=len(cut_events),
                                     reference_annotations_used=False, electrical_connections_inferred=False)
    return result
