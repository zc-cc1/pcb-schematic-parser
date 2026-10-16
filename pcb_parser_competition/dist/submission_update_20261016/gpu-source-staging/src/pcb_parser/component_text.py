"""Image-only OCR and geometry-based component fields; no annotation access.

All boxes stay in original top-left image coordinates until export. Uncertain
fields remain empty; alternatives and rejected matches are retained for review.
"""
import math
import re
import time
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


REF = re.compile(r"^(IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+$", re.I)
REF_SCAN = re.compile(r"(?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+",re.I)
PREFIXES = {
    "resistor": {"R"}, "capacitor": {"C"}, "inductor": {"L", "FB"},
    "diode": {"D"}, "led": {"D", "LED"}, "bjt": {"Q"}, "mosfet": {"Q", "FET"},
    "ic": {"U", "IC"}, "opamp": {"U", "IC"}, "crystal": {"X", "Y"},
    "switch": {"SW", "S"}, "fuse": {"F"}, "connector": {"J", "P"},
    "vbus_in": {"TP", "J", "P"}, "vbus_out": {"TP", "J", "P"},
    "vbus_bidirectional": {"TP", "J", "P"}, "jumper": {"J", "JP"},
}


def bbox(points):
    points = np.asarray(points, dtype=float)
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def iou(a, b):
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection/(area_a+area_b-intersection) if area_a+area_b-intersection else 0


def inside(point, box, inset=0):
    return box[0]+inset < point[0] < box[2]-inset and box[1]+inset < point[1] < box[3]-inset


def restore_polygon(points, crop, scale, rotation):
    """Inverse of cv2.rotate(..., ROTATE_90_CLOCKWISE) then uniform resize."""
    points = np.asarray(points, dtype=float).copy() / scale
    x1, y1, x2, y2 = crop
    if rotation == 90:
        points = np.column_stack((points[:, 1], (y2-y1-1)-points[:, 0]))
    points += [x1, y1]
    return points.tolist()


def positions(size, tile, overlap):
    if size <= tile:
        return [0]
    values = list(range(0, size-tile+1, max(1, int(tile*(1-overlap)))))
    return sorted(set(values+[size-tile]))


def merge_texts(texts, threshold=.45):
    """Merge overlapping observations, retaining alternatives and raw spelling.

    Equal strings at different positions must never collapse into one token.
    A short crop splitting a longer line is also retained as separate evidence.
    """
    output = []
    for row in sorted(texts, key=lambda t: -t["confidence"]):
        box = bbox(row["polygon"])
        duplicate = None
        for existing in output:
            if iou(box, existing["bbox"]) < threshold:
                continue
            # Conflicting words with very different lengths can be line splits.
            if min(len(row["text"]), len(existing["text"]))/max(1, len(row["text"]), len(existing["text"])) < .6:
                continue
            duplicate = existing
            break
        if duplicate is None:
            output.append({**row, "bbox": box, "observations": [{"text": row["text"],
                           "confidence": row["confidence"], "view": row["view"], "bbox": box}]})
        else:
            duplicate["observations"].append({"text": row["text"], "confidence": row["confidence"], "view": row["view"], "bbox": box})
    output.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for idx, row in enumerate(output):
        row["text_id"] = f"text_{idx+1:04d}"
    return output


def run_ocr(rgb, engine, cfg):
    h, w = rgb.shape[:2]
    views = [{"crop": [0, 0, w, h], "scale": 1., "rotation": 0, "id": "global"}]
    tile = int(cfg.get("tile_size", 768))
    for y in positions(h, tile, cfg.get("overlap", .2)):
        for x in positions(w, tile, cfg.get("overlap", .2)):
            crop = [x, y, min(w, x+tile), min(h, y+tile)]
            for rotation in cfg.get("rotations", [0, 90]):
                views.append({"crop": crop, "scale": float(cfg.get("local_scale", 2)),
                              "rotation": rotation, "id": f"tile_{x}_{y}_r{rotation}"})
    texts, timings = [], []
    for view in views:
        x1, y1, x2, y2 = view["crop"]
        patch = rgb[y1:y2, x1:x2]
        if view["rotation"] == 90:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        if view["scale"] != 1:
            patch = cv2.resize(patch, None, fx=view["scale"], fy=view["scale"], interpolation=cv2.INTER_CUBIC)
        start = time.perf_counter()
        result = engine(np.ascontiguousarray(patch[:, :, ::-1]), use_det=True, use_cls=True, use_rec=True)
        timings.append({**view, "wall_seconds": time.perf_counter()-start,
                        "text_count": len(result.txts) if result.txts is not None else 0})
        if result.boxes is not None:
            for points, text, score in zip(result.boxes, result.txts, result.scores):
                if float(score) < cfg.get("min_confidence", .55):
                    continue
                polygon = restore_polygon(points, view["crop"], view["scale"], view["rotation"])
                # OCR occasionally extends beyond a crop edge.
                polygon = np.clip(np.asarray(polygon), [0, 0], [w, h]).tolist()
                texts.append({"polygon": polygon, "text": str(text), "confidence": float(score), "view": view["id"]})
    return merge_texts(texts), {"views": timings, "raw_observations": len(texts),
                                "wall_seconds": sum(v["wall_seconds"] for v in timings)}


def value_kind(text):
    text = text.strip().replace(" ", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[munpµμ][Ff]?|[munpµμ]\d+(?:[Ff])?)", text):
        return "capacitance"
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[munpµμ]H|H)", text, re.I):
        return "inductance"
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[kMG]?Hz)|\d+[kM]\d+", text, re.I):
        return "frequency"
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[kKmMR]|[kKmM]?\d*(?:[oO]hm|Ω))|\d+[RkKmM]\d+", text):
        return "resistance"
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?V|\d+(?:\.\d+)?A|\d+(?:\.\d+)?%", text, re.I):
        return "rating"
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "numeric"
    return None


def tokens_from_texts(texts):
    # Preserve a full word alternative when a higher-scoring OCR view truncates
    # its suffix. A scalar "0.1" must not erase an observed capacitance "0.1uF".
    expanded = []
    for row in texts:
        seen = set()
        observations = row.get("observations", []) + [{"text": row["text"], "confidence": row["confidence"], "bbox": row.get("bbox", bbox(row["polygon"]))}]
        for observation in sorted(observations, key=lambda o: -o["confidence"]):
            if observation["text"] in seen:
                continue
            seen.add(observation["text"])
            expanded.append({**row, "text": observation["text"], "confidence": observation["confidence"],
                             "view": observation.get("view",row.get("view")),
                             "bbox": observation.get("bbox", row.get("bbox", bbox(row["polygon"])))})
    independent_refs = []
    for row in expanded:
        for word in row["text"].split():
            for match in reference_spans(word):
                independent_refs.append((match[0],row["bbox"]))
    tokens = []
    for row in expanded:
        source = row["text"]
        matches = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", source)]
        if len(matches)>1 and value_kind(source) is not None:
            matches=[(source,0,len(source))]
        separated=[]
        for word,start,end in matches:
            spans=reference_spans(word)
            if not spans:
                units=list(re.finditer(r"\d+(?:\.\d+)?(?:[kKmMR]|[munpµμ][FfHh]?|[kMG]?Hz|[kKmM]?(?:Ω|[oO]hm))",word))
                if len(units)>1 and "".join(m[0] for m in units)==word and len({value_kind(m[0]) for m in units})==1:
                    spans=[(m[0],m.start(),m.end()) for m in units]
            separated.extend([(text,start+a,start+b) for text,a,b in spans] if spans else [(word,start,end)])
        matches=separated
        # OCR may fuse a reference and a unit-bearing value (e.g. R91k).
        if len(matches) == 1:
            supported = [(ref, box) for ref, box in independent_refs if source.upper().startswith(ref.upper())
                         and len(source) > len(ref) and gap(box, row["bbox"]) < 5
                         and (not REF.fullmatch(source) or len(source)-len(ref) >= 3)
                         and value_kind(source[len(ref):]) is not None]
            if supported:
                ref = max(supported, key=lambda item: len(item[0]))[0]
                matches = [(source[:len(ref)], 0, len(ref)), (source[len(ref):], len(ref), len(source))]
            elif not REF.fullmatch(source):
                fused = re.fullmatch(r"((?:R|C|L)\d+?)(\d+(?:\.\d+)?(?:[kKmMRmunpµμ](?:F|H)?))", source)
                if fused:
                    matches = [(fused[1], 0, len(fused[1])), (fused[2], len(fused[1]), len(source))]
        for text, start, end in matches:
            box = row.get("bbox", bbox(row["polygon"]))[:]
            approximate = text != source
            if approximate:
                length = box[2]-box[0]
                box[0], box[2] = box[0]+length*start/len(source), box[0]+length*end/len(source)
            ref = REF.fullmatch(text)
            kind = value_kind(text)
            if ref:
                role = "designator"
            elif kind:
                role = kind
            elif "_" in text or text.upper() in {"GND", "VCC", "VDD", "OSC"}:
                role = "net_or_pin_label"
            elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+/\-]*", text) and re.search(r"[A-Za-z]", text) and re.search(r"\d", text):
                role = "model_candidate"
            else:
                role = "other"
            tokens.append({"token_id": f"token_{len(tokens)+1:04d}", "text_id": row["text_id"], "text": text,
                           "bbox": box, "confidence": row["confidence"], "role": role,
                           "source_view": row.get("view"),
                           "prefix": ref[1].upper() if ref else None,
                           "source_text": source, "approximate_split_geometry": approximate})
    # Splitting a long OCR line and an independent local word can produce the
    # same physical token twice. They must not create artificial ambiguity.
    distinct = []
    for token in sorted(tokens, key=lambda t: -t["confidence"]):
        if any(iou(token["bbox"], other["bbox"]) > .45
               and (token["text"] == other["text"]
                    or token["role"] == other["role"] == "numeric") for other in distinct):
            continue
        distinct.append(token)
    return distinct


def reference_spans(source):
    matches=list(REF_SCAN.finditer(source))
    if not matches:
        return []
    remainder=REF_SCAN.sub("",source)
    if re.search(r"[A-Za-z0-9]",remainder):
        return []
    prefixes={REF.fullmatch(m[0])[1].upper() for m in matches}
    if len(matches)>1 and len(prefixes)>1:
        return []  # do not split a mixed model code into unrelated references
    return [(m[0],m.start(),m.end()) for m in matches]


def choose_global(costs, dummy_cost, margin, details, role):
    if not costs.size:
        return {}
    n, m = costs.shape
    # A private dummy column per component allows every field to remain unknown.
    padded = np.concatenate([costs, np.full((n, n), dummy_cost)], axis=1)
    rows, columns = linear_sum_assignment(padded)
    best_total = float(padded[rows, columns].sum())
    output = {}
    for row, column in zip(rows, columns):
        if column >= m or costs[row, column] >= dummy_cost:
            continue
        score = float(costs[row, column])
        alternatives = sorted((float(costs[row, j]), j) for j in range(m) if j != column and costs[row, j] < dummy_cost)
        competitors = sorted((float(costs[i, column]), i) for i in range(n) if i != row and costs[i, column] < dummy_cost)
        # Re-solve with this edge forbidden. Local second-best scores can be
        # close while the complete one-to-one assignment is unambiguous.
        alternative_costs = padded.copy()
        alternative_costs[row, column] = 1e6
        ar, ac = linear_sum_assignment(alternative_costs)
        assignment_margin = float(alternative_costs[ar, ac].sum()) - best_total
        if assignment_margin < margin:
            details.append({"field": role, "detection_index": int(row), "candidate_index": int(column),
                            "reason": "ambiguous_geometry", "score": score, "alternatives": alternatives[:3],
                            "competing_components": competitors[:3], "global_assignment_margin": assignment_margin})
            continue
        output[int(row)] = (int(column), score, assignment_margin)
    return output


def duplicate_detections(detections):
    """Choose an owner for near-coincident boxes; retain every raw prediction.

    Suppression is only within the text assignment matrix, with strict center
    agreement plus overlap/containment. Neighboring parts remain distinct.
    """
    active, duplicates = [], {}
    for i in sorted(range(len(detections)), key=lambda i: -detections[i]["confidence"]):
        a = detections[i]["bbox"]
        aw, ah = a[2]-a[0], a[3]-a[1]
        for j in active:
            if detections[i]["class_name"] != detections[j]["class_name"]:
                continue
            b = detections[j]["bbox"]
            bw, bh = b[2]-b[0], b[3]-b[1]
            dx, dy = abs(center(a)[0]-center(b)[0]), abs(center(a)[1]-center(b)[1])
            intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            containment = intersection/max(1.,min(aw*ah,bw*bh))
            if dx <= .25*min(aw,bw) and dy <= .25*min(ah,bh) and (iou(a,b)>=.65 or containment>=.9):
                duplicates[i] = j
                break
        else:
            active.append(i)
    return duplicates


def reference_case_style(refs):
    confident = [t for t in refs if t["confidence"] >= .95]
    if len(confident) < 8:
        return None
    uppercase = sum(re.match(r"[A-Za-z]+", t["text"])[0].isupper() for t in confident)
    return "upper" if uppercase/len(confident) >= .9 else "lower" if uppercase/len(confident) <= .1 else None


def bind_components(detections, texts, width, height, cfg):
    options = cfg.get("component_text", {}).get("binding", {})
    max_gap = float(options.get("max_gap_px", 100))
    margin = float(options.get("ambiguity_margin", 3))
    types = [cfg["class_mapping"].get(d["class_name"], d["class_name"]) for d in detections]
    tokens = tokens_from_texts(texts)
    refs = [t for t in tokens if t["role"] == "designator"]
    duplicates = duplicate_detections(detections)
    case_style = reference_case_style(refs)
    # Same designator cannot identify two components, even if OCR repeats it.
    grouped = {}
    for token in refs:
        grouped.setdefault(token["text"].upper(), []).append(token)
    groups = list(grouped.values())
    costs = np.full((len(detections), len(groups)), 1e6)
    selected = {}
    details = []
    for i, (d, kind) in enumerate(zip(detections, types)):
        if i in duplicates:
            continue
        db = d["bbox"]
        for j, group in enumerate(groups):
            for token in group:
                if token["prefix"] not in PREFIXES.get(kind, set()):
                    continue
                tb, distance = token["bbox"], gap(token["bbox"], db)
                if distance > max_gap or inside(center(tb), db, inset=3):
                    continue
                score = distance + .18*math.dist(center(tb), center(db)) + 8*(1-token["confidence"])
                dx, dy = abs(center(tb)[0]-center(db)[0]), abs(center(tb)[1]-center(db)[1])
                score += options.get("alignment_weight", .8)*min(dx,dy)
                # Style can rank observed OCR alternatives; it never changes a
                # character or creates an uppercase reference without evidence.
                if case_style and token["confidence"] >= .75:
                    prefix_is_upper = re.match(r"[A-Za-z]+", token["text"])[0].isupper()
                    if prefix_is_upper != (case_style == "upper"):
                        score += options.get("case_style_penalty", 2.)
                if kind in {"ic", "opamp"} and db[1] < center(tb)[1] < db[3]:
                    score += 35  # long-IC side labels are often ball/pin IDs
                if score < costs[i, j]:
                    costs[i, j], selected[i, j] = score, token
    assignments = choose_global(costs, options.get("unassigned_reference_cost", 90), margin, details, "designator")
    bindings = {i: {"designator": selected[i, j], "designator_score": score, "designator_assignment_margin": stability}
                for i, (j, score, stability) in assignments.items()}

    led_reference_fragments = {}
    for field in ("value", "Name"):
        candidates = [t for t in tokens if t["role"] in ({"capacitance", "inductance", "resistance", "frequency", "numeric"}
                                                        if field == "value" else {"model_candidate"})]
        if field == "Name":
            model_candidates = []
            for token in candidates:
                owners = [i for i, binding in bindings.items()
                          if types[i] == 'led' and 'designator' in binding
                          and binding['designator']['confidence'] >= .95
                          and re.fullmatch(r'LED\d+', binding['designator']['text'], re.I)
                          and token['text'].upper() == binding['designator']['text'][1:].upper()
                          and iou(token['bbox'], binding['designator']['bbox']) >= .45]
                if owners:
                    for i in owners:
                        led_reference_fragments.setdefault(i, []).append(token)
                        details.append({'field':'Name','detection_index':i,
                                        'reason':'truncated_LED_reference_is_not_separate_model',
                                        'fragment':token['text'],'observed_reference':bindings[i]['designator']['text'],
                                        'fragment_bbox':token['bbox']})
                else:
                    model_candidates.append(token)
            candidates = model_candidates
        field_groups = []
        for token in candidates:
            for group in field_groups:
                if any(iou(token["bbox"],other["bbox"])>.5 for other in group):
                    group.append(token)
                    break
            else:
                field_groups.append([token])
        field_selected = {}
        candidates = [t for group in field_groups for t in group]
        costs = np.full((len(detections), len(candidates)), 1e6)
        for i, (d, kind) in enumerate(zip(detections, types)):
            if i in duplicates:
                continue
            has_reference = i in bindings and "designator" in bindings[i]
            if not has_reference and not options.get("allow_unreferenced_fields",False):
                continue
            if field == "Name" and kind not in {"ic", "opamp", "mosfet", "bjt", "diode", "led"}:
                continue
            expected = {"capacitor": {"capacitance"}, "resistor": {"resistance", "numeric"},
                        "inductor": {"inductance", "resistance"}, "crystal": {"frequency"}}
            db = d["bbox"]
            anchor = bindings[i]["designator"]["bbox"] if has_reference else db
            for j, token in enumerate(candidates):
                tb = token["bbox"]
                if not has_reference and (token["confidence"]<.97 or gap(tb,db)>options.get("unreferenced_field_gap_px",45)):
                    continue
                if field == "value" and token["role"] not in expected.get(kind, set()):
                    continue
                if field == "value" and has_reference:
                    intersection = max(0,min(tb[2],anchor[2])-max(tb[0],anchor[0]))*max(0,min(tb[3],anchor[3])-max(tb[1],anchor[1]))
                    anchor_area = max(1.,(anchor[2]-anchor[0])*(anchor[3]-anchor[1]))
                    if intersection/anchor_area > .6 or inside(center(anchor),tb):
                        continue  # a fused misread covering the reference itself
                    if token["role"]=="numeric" and any(
                        other["role"] in {"designator","model_candidate","net_or_pin_label"}
                        and re.search(r"[A-Za-z]",other["text"]) and token["text"] in other["text"]
                        and other["confidence"]>=token["confidence"]+.1
                        and iou(tb,other["bbox"])>.5 for other in tokens):
                        continue  # partial OCR of a stronger pin/ref label
                cx, cy = center(tb)
                interior_model = (field=="Name" and kind in {"ic","opamp"}
                                  and db[0]+.3*(db[2]-db[0])<cx<db[2]-.3*(db[2]-db[0])
                                  and db[1]+.2*(db[3]-db[1])<cy<db[3]-.2*(db[3]-db[1])
                                  and tb[2]-tb[0] < .75*(db[2]-db[0])
                                  and tb[3]-tb[1] < .35*(db[3]-db[1])
                                  and len(re.findall(r"\d",token["text"]))>=3
                                  and len(re.findall(r"[A-Za-z]",token["text"]))>=3
                                  and not re.match(r"(?:GPIO|GPI\d|ADC|PIN|IO\d)",token["text"],re.I))
                if field=="Name" and not has_reference and not interior_model:
                    continue
                if (inside(center(tb), db, 2) and not interior_model) or gap(tb, db) > max_gap:
                    continue
                # A field must also be close to its reference, not merely to a huge IC.
                anchor_distance = math.dist(center(tb), center(anchor))
                # Some symbols place the model above and reference below the IC.
                # Accept centered edge labels independently of a distant reference.
                centered_edge_model = (field == "Name" and kind in {"ic", "opamp"}
                                      and db[0]+.2*(db[2]-db[0]) < cx < db[2]-.2*(db[2]-db[0])
                                      and (cy < db[1] or cy > db[3]) and gap(tb, db) <= 40)
                if anchor_distance > options.get("field_anchor_radius_px", 90) and not (centered_edge_model or interior_model):
                    continue
                score = gap(tb, db)*.45 + anchor_distance*.65 + abs(center(tb)[0]-center(anchor)[0])*.18
                score += options.get("field_alignment_weight", .6)*min(abs(center(tb)[0]-center(anchor)[0]),abs(center(tb)[1]-center(anchor)[1]))
                if centered_edge_model:
                    score = min(score, 10 + gap(tb, db)*.7 + abs(cx-center(db)[0])*.2)
                if interior_model:
                    score = min(score,15+abs(cx-center(db)[0])*.2)
                score += 8*(1-token["confidence"])
                if token["role"] == "numeric":
                    score += 15
                costs[i, j] = score
        grouped_costs = np.full((len(detections), len(field_groups)),1e6)
        candidate_index = {t["token_id"]:j for j,t in enumerate(candidates)}
        for i in range(len(detections)):
            for j,group in enumerate(field_groups):
                eligible = [t for t in group if costs[i,candidate_index[t["token_id"]]] < 90]
                if eligible:
                    # Recognition alternatives share a physical word. Small
                    # crop-box jitter must not outweigh the better reading.
                    chosen = max(eligible,key=lambda t:(t["confidence"],not t["approximate_split_geometry"],
                                                       -costs[i,candidate_index[t["token_id"]]]))
                    grouped_costs[i,j] = costs[i,candidate_index[chosen["token_id"]]]
                    field_selected[i,j] = chosen
        fields = choose_global(grouped_costs, 90, margin, details, field)
        for i, (j, score, stability) in fields.items():
            bindings.setdefault(i,{})[field], bindings[i][field+"_score"] = field_selected[i,j], score
            bindings[i][field+"_assignment_margin"] = stability

    for i, fragments in led_reference_fragments.items():
        if 'Name' not in bindings[i]:
            # The full spelling was actually observed and bound as this part's
            # designator; do not add letters to an isolated ED1 observation.
            bindings[i]['Name'] = {**bindings[i]['designator'],
                                   'name_source':'observed_full_LED_reference_same_glyph'}
            bindings[i]['Name_recovery'] = {'reason':'same_glyph_fragment_replaced_by_observed_full_reference',
                                            'rejected_fragments':[t['text'] for t in fragments]}

    output = {"components": {}, "pins": {}, "nets": {}}
    records = []
    for i, (d, kind) in enumerate(zip(detections, types)):
        binding = bindings.get(i, {})
        designator = binding.get("designator", {}).get("text")
        cid = designator or f"UNRESOLVED_DET_{i+1:04d}"
        x1, y1, x2, y2 = d["bbox"]
        export = {"type": kind, "Name": binding.get("Name", {}).get("text"),
                  "value": binding.get("value", {}).get("text"),
                  "bbox": [max(0., x1), max(0., height-y2), min(float(width), x2), min(float(height), height-y1)]}
        if kind == "gnd":
            export["Name"] = "GND"  # symbol class, not a guessed OCR label
        output["components"][cid] = export
        records.append({"component_id": cid, "detection_index": i, "type": kind, "designator": designator,
                        "Name": export["Name"], "value": export["value"], "bbox_top_left": d["bbox"],
                        "detection_confidence": d["confidence"], "fields": binding,
                        "duplicate_of_detection_index": duplicates.get(i),
                        "status": "provisional_image_prediction_not_manually_verified"})
    used = {b[field]["token_id"] for b in bindings.values() for field in ("designator", "Name", "value") if field in b}
    resolved_count = sum("designator" in b for b in bindings.values())
    counts = {"detections": len(detections), "resolved_designators": resolved_count,
              "unresolved_designators": len(detections)-resolved_count,
              "bound_values": sum("value" in b for b in bindings.values()),
              "bound_models": sum("Name" in b for b in bindings.values())}
    return output, records, {"counts": counts, "tokens": tokens, "rejected_matches": details,
                             "duplicate_detections": duplicates, "reference_case_style": case_style,
                             "ambiguous_units": [{"token_id":t["token_id"],"raw_text":t["text"],"reason":"unit_0_O_ambiguity_no_automatic_rewrite"}
                                                 for t in tokens if re.search(r"0hm$",t["text"])],
                             "unassigned_tokens": [t["token_id"] for t in tokens if t["token_id"] not in used],
                             "role_counts": dict(Counter(t["role"] for t in tokens)),
                             "input_contract": "image OCR + frozen detector predictions/config only",
                             "status": "components_stage_only_pins_and_nets_pending"}
