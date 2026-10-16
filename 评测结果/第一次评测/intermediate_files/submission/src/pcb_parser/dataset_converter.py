"""Convert CVJsonStd to auditable local references in the competition JSON shape."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from PIL import Image


def natural_key(value):
    return tuple((1, int(p)) if p.isdigit() else (0, p.casefold()) for p in re.split(r"(\d+)", str(value)))


def active(value):
    return isinstance(value, dict) and value.get("state") != "deleted"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def raw_point(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(number(v) for v in value):
        return tuple(float(v) for v in value)
    return None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)


def load_source_annotation(path):
    """Only identical repeated source keys may be collapsed, with pointer records."""
    class Pairs(list):
        pass
    duplicates = []
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    def visit(value, pointer=""):
        if isinstance(value, Pairs):
            result = {}
            for key, item in value:
                child = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                item = visit(item, child)
                if key in result:
                    if result[key] != item:
                        raise ValueError(f"Conflicting duplicate JSON key at {child}")
                    duplicates.append({"source_pointer": child, "key": key, "policy": "identical_value_collapsed"})
                result[key] = item
            return result
        if isinstance(value, list):
            return [visit(item, pointer + f"/{i}") for i, item in enumerate(value)]
        return value
    parsed = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=Pairs, parse_constant=reject_constant)
    return visit(parsed), duplicates


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            previous = self.parent[key]
            self.parent[key] = root
            key = previous
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_pin_numbers(value):
    """Accept short pad identifiers and comma lists; return [] for OCR noise."""
    if not isinstance(value, str) or not value.strip() or value.strip() in {"∅", "null", "None", "?"}:
        return []
    parts = [part.strip() for part in re.split(r"[,，]", value)]
    pattern = r"(?:[A-Za-z]{1,3}|[A-Za-z]{0,3}\d{1,4}(?:[A-Za-z]{1,3}\d{1,4})?[A-Za-z]{0,3})"
    if not all(re.fullmatch(pattern, part) for part in parts):
        return []
    return list(dict.fromkeys(parts))


class Converter:
    def __init__(self, data, case_id, policy, overrides=None):
        self.data, self.case_id, self.policy = data, case_id, policy
        self.overrides = overrides or {}
        self.width, self.height = data["canvas"]["width"], data["canvas"]["height"]
        if not number(self.width) or not number(self.height) or min(self.width, self.height) <= 0:
            raise ValueError("Invalid canvas dimensions")
        self.top = float(self.height) - (1 if policy["export_y_origin"] == "height_minus_one" else 0)
        self.raw_components = {k: v for k, v in data["components"].items() if active(v)}
        self.raw_edges = {k: v for k, v in data["edges"].items() if active(v)}
        self.texts = {k: v for k, v in data["texts"].items() if active(v)}
        self.out = {"components": {}, "pins": {}, "nets": {}}
        self.warnings, self.components, self.pins, self.net_trace = [], {}, {}, {}
        self.component_map, self.pin_map = {}, {}

    def warn(self, code, **details):
        self.warnings.append({"code": code, **details})

    def point(self, value):
        p = raw_point(value)
        if p is None:
            raise ValueError("Invalid point")
        return {"x": p[0], "y": self.top - p[1]}

    def text_ref(self, reference, owner=None):
        if reference is None:
            return None
        if not isinstance(reference, str):
            self.warn("non_string_text_reference", value=reference)
            return None
        if re.fullmatch(r"T_\d+", reference):
            text = self.texts.get(reference)
            if text is None and owner:
                candidate = owner.get("associatedTexts", {}).get(reference)
                text = candidate if active(candidate) else None
            if text is None:
                self.warn("unresolved_text_reference", reference=reference)
                return None
            return text.get("content") if isinstance(text.get("content"), str) else None
        return reference

    def associated(self, component, role=None):
        result = []
        for tid, text in sorted(component.get("associatedTexts", {}).items(), key=lambda x: natural_key(x[0])):
            if active(text) and isinstance(text.get("content"), str):
                if role is None or str(text.get("details", "")).casefold() == role.casefold():
                    result.append({"id": tid, "text": text["content"], "role": text.get("details", "")})
        return result

    def unique_text(self, entries, code, cid):
        values = list(dict.fromkeys(e["text"] for e in entries if e["text"].strip()))
        if len(values) > 1:
            self.warn(code, component=cid, candidates=values)
        return values[0] if values else None

    def identity(self, cid, c):
        override = self.overrides.get("components", {}).get(cid, {})
        if "designator" in override:
            self.warn("manual_designator_override", component=cid, value=override["designator"])
            return override["designator"], "manual_review"
        declared = self.unique_text(self.associated(c, "designator"), "ambiguous_designator", cid)
        if declared:
            return declared, "declared_designator"
        cat = c.get("category")
        if cat == "Gnd":
            self.warn("ground_identity_from_category", component=cid)
            return "GND", "category_rule"
        labels = self.associated(c, "netLabel")
        if cat in self.policy["label_categories"] and labels:
            return self.unique_text(labels, "ambiguous_label_name", cid), "declared_net_label"
        if cat == "V":
            candidates = labels or [e for e in self.associated(c) if re.fullmatch(r"[+\-]?(?:\d+(?:\.\d+)?(?:[Vv]|[Vv]\d+)|(?:VCC|VDD|VBUS|VSS|AVDD|DVDD)(?:[_+\-].*)?)", e["text"])]
            name = self.unique_text(candidates, "ambiguous_power_name", cid)
            if name:
                self.warn("power_identity_from_text", component=cid, value=name)
                return name, "power_text_rule"
        prefixes = {"Box": {"U", "IC", "J", "CN"}, "C": {"C"}, "R": {"R"}, "L": {"L"}, "Crystal": {"X", "Y"}, "MOSFET": {"Q"}, "BJT": {"Q"}, "D": {"D"}, "LED": {"D", "LED"}, "PIN": {"TP", "J", "P"}, "Switch": {"SW", "K"}, "Other": {"SW", "U", "J"}, "Fuse": {"F"}, "Opt": {"U"}, "AMP": {"U", "IC"}}.get(cat, set())
        candidates = []
        for entry in self.associated(c):
            match = re.fullmatch(r"([A-Za-z]+)\d+[A-Za-z]?", entry["text"])
            if match and match.group(1).upper() in prefixes:
                candidates.append(entry)
        names = list(dict.fromkeys(e["text"] for e in candidates))
        if len(names) == 1:
            self.warn("designator_inferred_from_associated_text", component=cid, value=names[0])
            return names[0], "associated_text_candidate"
        self.warn("unresolved_designator", component=cid, candidates=names)
        return f"UNRESOLVED_{cid}", "unresolved_placeholder"

    def component_value(self, cid, c, target_type):
        entries = self.associated(c, "value")
        values = list(dict.fromkeys(e["text"] for e in entries if e["text"].strip()))
        if target_type in {"capacitor", "resistor"}:
            eligible = [v for v in values if not re.fullmatch(r"\s*[\d.]+\s*(?:[Vv]|%|[Ww])\s*", v)]
            if target_type == "capacitor":
                preferred = [v for v in eligible if re.search(r"[\d.]\s*[pnuµμm]?[Ff]\b", v)]
                eligible = preferred or eligible
            values = eligible
        if len(values) > 1:
            self.warn("ambiguous_value", component=cid, candidates=values, selected=values[0])
        if not values and target_type in {"capacitor", "resistor"}:
            self.warn("missing_scored_value", component=cid)
        return values[0] if values else None

    def convert_components(self):
        groups = defaultdict(list)
        reserved = set()
        for cid, c in sorted(self.raw_components.items(), key=lambda x: natural_key(x[0])):
            base, source = self.identity(cid, c)
            if not isinstance(base, str) or not base.strip() or any(ch in base for ch in ",()"):
                self.warn("unsafe_designator_for_hypergraph", component=cid, original=base)
                base, source = f"UNRESOLVED_{cid}", "unresolved_placeholder"
            groups[base].append((cid, c, source))
            reserved.add(base)
        for base, items in sorted(groups.items(), key=lambda x: natural_key(x[0])):
            for i, (cid, c, source) in enumerate(items, 1):
                designator = base if len(items) == 1 else f"{base}_{i}"
                suffix = i
                while designator in self.out["components"] or (len(items) > 1 and designator in reserved):
                    suffix += 1
                    designator = f"{base}_{suffix}"
                    self.warn("designator_suffix_collision", component=cid, value=designator)
                box = c.get(self.policy["bbox_source"])
                if not isinstance(box, dict) or not all(number(box.get(k)) for k in ("x1", "y1", "x2", "y2")) or box["x1"] >= box["x2"] or box["y1"] >= box["y2"]:
                    self.warn("invalid_component_bbox_omitted", component=cid, value=box)
                    continue
                cat = c.get("category", "Other")
                typ = self.policy["type_mapping"].get(cat, str(cat).lower())
                if cat not in self.policy["pdf_supported_source_types"]:
                    self.warn("provisional_type_mapping", component=cid, source=cat, target=typ)
                name = None
                for field in ("searchNameByLib", "searchNameByVLM", "searchNameByOCR"):
                    name = self.text_ref(c.get(field), c)
                    if name:
                        break
                name = name or self.unique_text(self.associated(c, "searchName"), "ambiguous_name", cid)
                is_power = cat in self.policy["power_categories"]
                explicit_labels = self.associated(c, "netLabel")
                logical_label = self.unique_text(explicit_labels, "ambiguous_label_name", cid) if explicit_labels else (base if is_power else None)
                if logical_label:
                    name = logical_label
                override = self.overrides.get("components", {}).get(cid, {})
                if "Name" in override:
                    name = override["Name"]
                    self.warn("manual_name_override", component=cid, value=name)
                elif name and name.startswith("scenario_"):
                    self.warn("scene_title_rejected_as_name", component=cid, value=name)
                    name = None
                if typ == "ic" and not name:
                    self.warn("missing_scored_name", component=cid)
                value = override.get("value", self.component_value(cid, c, typ))
                bbox = [float(box["x1"]), self.top - float(box["y2"]), float(box["x2"]), self.top - float(box["y1"])]
                self.out["components"][designator] = {"type": typ, "Name": name, "value": value, "bbox": bbox}
                self.out["pins"][designator] = {}
                self.components[cid] = {"designator": designator, "identity_source": source, "base_identity": base, "source_category": cat, "type": typ, "logical_label": logical_label, "is_power": is_power, "is_label": cat in self.policy["label_categories"] and bool(explicit_labels)}
                self.component_map[cid] = dict(self.components[cid])

    def convert_pins(self):
        owned = defaultdict(dict)
        for cid in self.components:
            for pid, local in self.raw_components[cid].get("pins", {}).items():
                if active(local):
                    root = self.data["pins"].get(pid, {})
                    owned[cid][pid] = {**(root if active(root) else {}), **local}
        for eid, edge in self.raw_edges.items():
            for pid, connection in edge.get("connect", {}).items():
                if not isinstance(connection, dict):
                    continue
                cid, actual_pid = connection.get("cpntId"), connection.get("pinId", pid)
                if cid in self.components and actual_pid not in owned[cid] and active(self.data["pins"].get(actual_pid)):
                    owned[cid][actual_pid] = self.data["pins"][actual_pid]
                    self.warn("global_pin_recovered_for_component", component=cid, pin=actual_pid, edge=eid)
        for cid, records in owned.items():
            drafts = []
            component = self.raw_components[cid]
            overrides = self.overrides.get("pins", {}).get(cid, {})
            for pid, pin in sorted(records.items(), key=lambda x: natural_key(x[0])):
                tip = raw_point(pin.get("tip"))
                if tip is None:
                    self.warn("invalid_pin_tip_omitted", component=cid, pin=pid)
                    continue
                override = overrides.get(pid, {})
                raw_num = override.get("number")
                if raw_num is None:
                    raw_num = self.text_ref(pin.get("pinNumberByVLM"), component) or self.text_ref(pin.get("pinNumberByOCR"), component)
                nums = parse_pin_numbers(raw_num)
                name = override.get("pinname")
                if name is None:
                    name = self.text_ref(pin.get("pinNameByVLM"), component) or self.text_ref(pin.get("pinNameByOCR"), component) or ""
                if not name:
                    self.warn("missing_pin_name", component=cid, pin=pid)
                if "number" in override:
                    self.warn("manual_pin_number_override", component=cid, pin=pid, value=raw_num)
                drafts.append((pid, pin, tip, raw_num, nums, name))
            reserved = {n for _, _, _, _, nums, _ in drafts for n in nums}
            assigned, cursor = {}, 1
            for pid, pin, tip, raw_num, nums, name in drafts:
                refs, status = [], "source_number"
                if not nums:
                    while str(cursor) in reserved or str(cursor) in assigned:
                        cursor += 1
                    nums, status = [str(cursor)], "auto_numbered"
                    self.warn("auto_pin_number", component=cid, pin=pid, raw_value=raw_num, assigned=nums[0])
                    cursor += 1
                for n in nums:
                    if n in assigned and assigned[n] != (tip, name):
                        original = n
                        while str(cursor) in reserved or str(cursor) in assigned:
                            cursor += 1
                        n, status = str(cursor), "collision_auto_numbered"
                        cursor += 1
                        self.warn("pin_number_collision", component=cid, pin=pid, original=original, assigned=n)
                    assigned[n] = (tip, name)
                    designator = self.components[cid]["designator"]
                    self.out["pins"][designator][f"pin_{n}"] = {"pinname": name, "point": self.point(tip)}
                    refs.append(f"{designator}.{n}")
                self.pins[(cid, pid)] = {"raw_tip": tip, "refs": refs, "source_edge_id": pin.get("edgeId"), "number_status": status, "raw_number": raw_num}
                self.pin_map[f"{cid}/{pid}"] = {"references": refs, "number_status": status, "source_number_text": raw_num, "source_tip": list(tip), "source_base": pin.get("base")}
        owned_ids = {pid for _, pid in self.pins}
        for pid, pin in self.data["pins"].items():
            if active(pin) and pid not in owned_ids:
                self.warn("unowned_global_pin_omitted", pin=pid, source_edge=pin.get("edgeId"), source_tip=pin.get("tip"))

    def resolve_connection(self, eid, pid, connection):
        cid = connection.get("cpntId")
        pid = connection.get("pinId", pid)
        if (cid, pid) in self.pins:
            return (cid, pid)
        point = raw_point(connection.get("location"))
        matches = [uid for uid, pin in self.pins.items() if uid[0] == cid and point is not None and pin["raw_tip"] == point]
        if len(matches) == 1:
            self.warn("connection_alias_resolved_by_exact_tip", edge=eid, component=cid, raw_pin=pid, resolved_pin=matches[0][1])
            return matches[0]
        self.warn("unresolved_edge_pin_omitted", edge=eid, component=cid, pin=pid)
        return None

    def edge_segments(self, eid, edge):
        result = []
        paths = edge.get("keyPointPaths", {}).get("wire", [])
        for index, path in enumerate(paths):
            if not isinstance(path, list) or len(path) < 2:
                self.warn("invalid_wire_path_omitted", edge=eid, path_index=index)
                continue
            for offset in range(len(path) - 1):
                p, q = raw_point(path[offset]), raw_point(path[offset + 1])
                if p is None or q is None:
                    self.warn("invalid_segment_point_omitted", edge=eid, path_index=index, offset=offset)
                    continue
                result.append((p, q, {"source_edge": eid, "path_index": index, "pair_index": offset}))
        if not result:
            self.warn("wire_has_no_exportable_paths", edge=eid)
        return result

    def convert_nets(self):
        dsu = DisjointSet()
        for uid in self.pins:
            dsu.find(("pin", *uid))
        edge_pins = defaultdict(set)
        for eid, edge in self.raw_edges.items():
            enode = ("edge", eid)
            dsu.find(enode)
            for pid, connection in edge.get("connect", {}).items():
                if isinstance(connection, dict):
                    uid = self.resolve_connection(eid, pid, connection)
                    if uid:
                        edge_pins[eid].add(uid)
                        dsu.union(enode, ("pin", *uid))
        for uid, pin in self.pins.items():
            eid = pin["source_edge_id"]
            if eid in self.raw_edges:
                if uid not in edge_pins[eid]:
                    self.warn("pin_edge_backreference_used", component=uid[0], pin=uid[1], edge=eid)
                edge_pins[eid].add(uid)
                dsu.union(("edge", eid), ("pin", *uid))
        coincident = defaultdict(list)
        for uid, pin in self.pins.items():
            coincident[pin["raw_tip"]].append(uid)
        for uids in coincident.values():
            for uid in uids[1:]:
                dsu.union(("pin", *uids[0]), ("pin", *uid))
        labels, label_components = defaultdict(list), defaultdict(set)
        if self.policy["merge_declared_edge_labels"]:
            for eid, edge in self.raw_edges.items():
                tid = edge.get("netlabel")
                text = self.texts.get(tid, {}) if isinstance(tid, str) else {}
                content = text.get("content")
                if str(text.get("details", "")).casefold() == "netlabel" and isinstance(content, str) and content.strip():
                    labels[content].append(("edge", eid))
        if self.policy["merge_same_name_label_components"]:
            for cid, component in self.components.items():
                label = component["logical_label"]
                if label and (component["is_power"] or component["is_label"]):
                    nodes = [("pin", *uid) for uid in self.pins if uid[0] == cid]
                    labels[label].extend(nodes)
                    if component["is_label"] and nodes:
                        label_components[label].add(cid)
        for label, nodes in labels.items():
            for node in nodes[1:]:
                dsu.union(nodes[0], node)
            if len(nodes) > 1:
                self.warn("source_label_merge_unverified", label=label, nodes=len(nodes))
        groups = defaultdict(lambda: {"pins": [], "edges": []})
        for node in list(dsu.parent):
            group = groups[dsu.find(node)]
            group["pins" if node[0] == "pin" else "edges"].append(node[1:] if node[0] == "pin" else node[1])
        ordered = sorted(groups.values(), key=lambda g: (0 if g["edges"] else 1, natural_key(min(g["edges"], key=natural_key)) if g["edges"] else natural_key(str(min(g["pins"], key=lambda x: (natural_key(x[0]), natural_key(x[1])))))))
        for group in ordered:
            uids = sorted(group["pins"], key=lambda x: (natural_key(x[0]), natural_key(x[1])))
            eids = sorted(group["edges"], key=natural_key)
            if not uids:
                self.warn("no_pin_net_omitted", source_edges=eids)
                continue
            if not eids and len(uids) == 1:
                continue
            refs = sorted({ref for uid in uids for ref in self.pins[uid]["refs"]}, key=natural_key)
            fly_labels = [label for label, cids in label_components.items() if len(cids) > 1 and len(cids.intersection(uid[0] for uid in uids)) > 1]
            points = set()
            raw_segments = []
            if fly_labels and self.policy["flywire_geometry"] == "label_points":
                for uid in uids:
                    comp = self.components[uid[0]]
                    if comp["is_label"] and comp["logical_label"] in fly_labels:
                        points.add(self.pins[uid]["raw_tip"])
                raw_segments = [(p, p, {"logical_label_point": True}) for p in sorted(points)]
                self.warn("flywire_point_geometry_provisional", labels=fly_labels, source_edges=eids, points=len(points))
            else:
                for eid in eids:
                    raw_segments.extend(self.edge_segments(eid, self.raw_edges[eid]))
                if not raw_segments and len({self.pins[uid]["raw_tip"] for uid in uids}) == 1:
                    p = self.pins[uids[0]]["raw_tip"]
                    raw_segments = [(p, p, {"direct_pin_contact": True})]
            net_id = f"net_{len(self.out['nets']) + 1}"
            exported, sources = {}, {}
            seen = {}
            for p, q, evidence in raw_segments:
                key = tuple(sorted((p, q)))
                if key in seen:
                    sources[seen[key]].append(evidence)
                    continue
                edge_id = f"edge_{len(exported) + 1}"
                seen[key] = edge_id
                exported[edge_id] = [self.point(p), self.point(q)]
                sources[edge_id] = [evidence]
            self.out["nets"][net_id] = {"hyperGraph": "(" + ",".join(refs) + ")", "edges": exported}
            self.net_trace[net_id] = {"source_edge_ids": eids, "source_pin_ids": [f"{cid}/{pid}" for cid, pid in uids], "fly_labels": fly_labels, "segment_sources": sources}

    def run(self):
        self.convert_components()
        self.convert_pins()
        self.convert_nets()
        errors, bounds = validate_output(self.out, self.width, self.height)
        for issue in bounds:
            self.warn("coordinate_outside_image", **issue)
        report = {"case_id": self.case_id, "conversion_version": self.policy["version"], "status": "format_valid_semantics_unverified" if not errors else "validation_failed", "coordinate_policy": {"source": "top_left", "destination": "bottom_left", "y_formula": "H-1-y" if self.policy["export_y_origin"] == "height_minus_one" else "H-y", "golden_verified": False}, "component_map": self.component_map, "pin_map": self.pin_map, "net_map": self.net_trace, "warning_counts": dict(Counter(w["code"] for w in self.warnings)), "warnings": self.warnings, "validation_errors": errors, "counts": {"components": len(self.out["components"]), "pins": sum(len(p) for p in self.out["pins"].values()), "nets": len(self.out["nets"]), "segments": sum(len(n["edges"]) for n in self.out["nets"].values())}, "caveats": ["Official Golden/evaluator equivalence is unverified.", "Category/auto-number/label/segmentation policies are explicit local choices.", "Warnings and source evidence must be reviewed before treating fields as training/evaluation truth."]}
        return self.out, report


def validate_output(data, width, height):
    """Project contract checks, including references; not official F1 or a schema engine."""
    errors, bounds = [], []
    if not isinstance(data, dict) or set(data) != {"components", "pins", "nets"} or not all(isinstance(data[k], dict) for k in data):
        return ["Invalid top-level objects"], []
    def check_point(p, path):
        if not isinstance(p, dict) or set(p) != {"x", "y"} or not all(number(p[k]) for k in p):
            errors.append(f"Invalid point: {path}")
        elif not (0 <= p["x"] <= width and 0 <= p["y"] <= height):
            bounds.append({"path": path, "point": p})
    for cid, c in data["components"].items():
        if not isinstance(cid, str) or not cid or not isinstance(c, dict) or set(c) != {"type", "Name", "value", "bbox"}:
            errors.append(f"Invalid component fields: {cid}")
            continue
        if not isinstance(c["type"], str) or not c["type"] or any(c[k] is not None and not isinstance(c[k], str) for k in ("Name", "value")):
            errors.append(f"Invalid component types: {cid}")
        b = c["bbox"]
        if not isinstance(b, list) or len(b) != 4 or not all(number(v) for v in b) or not (b[0] < b[2] and b[1] < b[3]):
            errors.append(f"Invalid bbox: {cid}")
        elif not (0 <= b[0] <= width and 0 <= b[2] <= width and 0 <= b[1] <= height and 0 <= b[3] <= height):
            bounds.append({"path": f"components.{cid}.bbox", "bbox": b})
    for cid, pins in data["pins"].items():
        if cid not in data["components"] or not isinstance(pins, dict):
            errors.append(f"Invalid pin owner: {cid}")
            continue
        for pid, pin in pins.items():
            if not isinstance(pid, str) or not re.fullmatch(r"pin_.+", pid) or not isinstance(pin, dict) or set(pin) != {"pinname", "point"} or not isinstance(pin["pinname"], str):
                errors.append(f"Invalid pin fields: {cid}/{pid}")
            else:
                check_point(pin["point"], f"pins.{cid}.{pid}")
    ownership = {}
    for nid, net in data["nets"].items():
        if not isinstance(nid, str) or not nid or not isinstance(net, dict) or set(net) != {"hyperGraph", "edges"} or not isinstance(net["hyperGraph"], str) or not re.fullmatch(r"\([^()]*\)", net["hyperGraph"]) or not isinstance(net["edges"], dict):
            errors.append(f"Invalid net fields: {nid}")
            continue
        tokens = net["hyperGraph"][1:-1].split(",")
        if len(tokens) != len(set(tokens)):
            errors.append(f"Duplicate hypergraph refs: {nid}")
        for token in tokens:
            parts = token.rsplit(".", 1)
            if len(parts) != 2 or "pin_" + parts[1] not in data["pins"].get(parts[0], {}):
                errors.append(f"Unresolved reference: {nid}/{token}")
            if token in ownership:
                errors.append(f"Pin assigned to multiple nets: {token}")
            ownership[token] = nid
        for eid, points in net["edges"].items():
            if not isinstance(eid, str) or not eid or not isinstance(points, list) or len(points) != 2:
                errors.append(f"Invalid edge: {nid}/{eid}")
            else:
                for i, p in enumerate(points):
                    check_point(p, f"nets.{nid}.{eid}[{i}]")
    return errors, bounds


def normalize_image(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        fmt, size = im.format, im.size
        canonical = im.convert("RGBA").tobytes()
        if fmt == "PNG":
            shutil.copy2(source, destination)
        else:
            converted = im if im.mode in {"RGB", "RGBA", "L", "LA"} else im.convert("RGBA")
            converted.save(destination, format="PNG")
    with Image.open(destination) as normalized:
        if normalized.format != "PNG" or normalized.size != size or normalized.convert("RGBA").tobytes() != canonical:
            raise ValueError("Image normalization changed decoded pixels or dimensions")
    return {"source_format": fmt, "target_format": "PNG", "size": list(size), "rgba_pixels_equal": True, "source_sha256": digest(source), "output_sha256": digest(destination)}
