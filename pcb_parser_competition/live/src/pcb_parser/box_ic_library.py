"""Fill IC pin numbers and names from the released component library.

The contest ships a component library — `cpntLibrary` with its field guide, and the
fuller `BoxICLibrary.json`, which are two different slices of the same EasyEDA symbol
pool (148 entries shared, 852 only in the box library, 189 only in the official one).
Each entry maps a part number to that part's pins: a number and a name per pin.

Why this matters for the pipeline: on datasheet-style pages the schematic prints pin
NUMBERS only for some pins, and prints pin NAMES for the rest (0059 prints `S-8252
Series` plus `VDD/VSS/VC/VM/CO/DO` and no numbers at all).  OCR therefore reads the
name but has no number to bind, so the pin stays unexportable and any net through it
loses its identity — the nets are keyed by pin NUMBER (R05/R09), not by name.

The rules here are deliberately conservative, because a wrong number is worse than no
number: it creates a false-positive pin AND a wrong net member.

  * only box-ish components (`types`) are touched;
  * a model is accepted only when the printed model text lies within
    `model_search_margin_px` of the component box, the entry has at least `min_pins`
    pins, and at least `min_name_matches` of this component's OCR-read pin names occur
    in the entry — the model text is matched against the part, not against the page;
  * a pin's printed NAME wins over its OCR-read number.  Measured on the 300 labelled
    pages: with perfect OCR the name rule reaches 78.0% while the number rule reaches
    50.2%, so the name is the better evidence for which pin this is;
  * names are only filled where nothing was read, never overwritten;
  * leftover library numbers are distributed by reading order only when
    `fill_remaining_numbers` is on (default off).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TYPES = ("ic", "box", "connector", "raw_block", "block")
DEFAULT_EVIDENCE = ("observed_OCR_text",)


def _norm(text):
    """Compare part numbers and pin names without punctuation or case."""
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


DEFAULT_EVIDENCE_PREFIXES = ("observed_OCR", "targeted_OCR")


def _is_evidence(status, evidence_statuses, prefixes=None):
    """A name/number counts as evidence when it is an OCR observation.

    Exact-string matching missed the near-tip fills (`observed_OCR_v2_missing_name_fill`):
    those names were read off the page by OCR and only attributed later, so they are just
    as good as a direct reading.  Synthesised fallbacks (`pin_number_fallback`) never match.
    """
    text = str(status or "").strip()
    if text in set(evidence_statuses):
        return True
    if "fallback" in text:
        return False
    return any(text.startswith(prefix) for prefix in (prefixes or DEFAULT_EVIDENCE_PREFIXES))


def _name_rank(name, count):
    """Pick one spelling when the library aggregates several (VOUT / Out / Output).

    Most frequent wins; a fully upper-case spelling then wins over a mixed one, because
    the reference pin names are upper case; a non-numeric name wins last.
    """
    return (count, 1 if str(name).isupper() else 0, 1 if name and not _norm(name).isdigit() else 0)


def canonical_number(number):
    """`6_1`/`6_2` are one package pin split into two symbol leads (ISENA/ISENB).

    The reference keys pins by the package number, so the suffix has to go; the split
    leads then share a number and are separated by their names instead.  Anything that
    is not "<digits>_<digits>" is kept verbatim - BGA ball names such as `A1B12` are a
    genuinely multi-ball pin and splitting them needs a rule the spec does not state.
    """
    text = str(number or "").strip()
    base = text.split("_")[0]
    if base.isdigit() and text != base and text[len(base):].strip("_").isdigit():
        return base
    return text


def load_library(paths):
    """{normalised model: {"model": original, "source": path, "pins": {number: name}}}"""
    index = {}
    for path in paths:
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        for key, record in document.items():
            model = _norm(key)
            if not model:
                continue
            slot = index.setdefault(model, {"model": key, "source": str(path), "names": {}})
            for pin in record.get("pins") or []:
                number = str(pin.get("number") or "").strip()
                if not number:
                    continue
                name = str(pin.get("name") or "").strip()
                counts = slot["names"].setdefault(number, {})
                counts[name] = counts.get(name, 0) + 1
    library = {}
    for model, slot in index.items():
        chosen = {}
        for number, counts in slot["names"].items():
            chosen.setdefault(canonical_number(number), {})[number] = counts
        pins = {}
        for canonical, by_spelling in chosen.items():
            merged = {}
            for counts in by_spelling.values():
                for name, count in counts.items():
                    merged[name] = merged.get(name, 0) + count
            pins[canonical] = max(merged.items(), key=lambda item: _name_rank(*item))[0]
        aliases = {}
        for number, counts in slot["names"].items():
            canonical = canonical_number(number)
            for name, count in counts.items():
                key = _norm(name)
                if not key:
                    continue
                best = aliases.get(key)
                if best is None or count > best[1]:
                    aliases[key] = (canonical, count)
        library[model] = {"model": slot["model"], "source": slot["source"], "pins": pins,
                          "aliases": {name: number for name, (number, _count) in aliases.items()}}
    return library


def _inside(token_box, component_box, margin):
    tx1, ty1, tx2, ty2 = (float(v) for v in token_box)
    cx1, cy1, cx2, cy2 = (float(v) for v in component_box)
    return not (tx2 < cx1 - margin or tx1 > cx2 + margin or ty2 < cy1 - margin or ty1 > cy2 + margin)


def identify_models(components, pins_by_owner, tokens, library, config):
    """Return {detection_index: model evidence} for box-ish components.

    The model text must sit next to the component box, and the entry must agree with the
    pin names OCR actually read on that component — a page-level text match alone picks
    the wrong part when two similar ICs share a page (measured on 0030: LM334 vs LM385).
    """
    types = set(config.get("types") or DEFAULT_TYPES)
    margin = float(config.get("model_search_margin_px", 40))
    min_pins = int(config.get("min_pins", 3))
    min_matches = int(config.get("min_name_matches", 1))
    min_chars = int(config.get("min_model_chars", 4))
    require_digit = bool(config.get("require_digit_in_model", True))
    evidence = config.get("evidence_statuses") or DEFAULT_EVIDENCE

    found = {}
    for component in components:
        if str(component.get("type") or "") not in types:
            continue
        owned = pins_by_owner.get(component.get("detection_index")) or []
        if not owned:
            continue
        names = {_norm(pin.get("pinname")) for pin in owned
                 if _is_evidence(pin.get("pinname_status"), evidence) and _norm(pin.get("pinname"))}
        observed = {str(pin.get("number") or "").strip() for pin in owned
                    if _is_evidence(pin.get("number_status"), evidence) and str(pin.get("number") or "").strip()}
        box = component.get("bbox_top_left")
        if not box:
            continue
        best = None
        for token in tokens:
            text = str(token.get("text") or "").strip()
            model = _norm(text)
            if len(model) < min_chars or (require_digit and not any(ch.isdigit() for ch in text)):
                continue
            entry = library.get(model)
            if not entry or len(entry["pins"]) < min_pins:
                continue
            if not _inside(token.get("bbox") or [0, 0, 0, 0], box, margin):
                continue
            entry_names = {_norm(name) for name in entry["pins"].values() if _norm(name)}
            name_matches = len(names & entry_names)
            number_matches = len({canonical_number(n) for n in observed} & set(entry["pins"]))
            rank = (name_matches + number_matches, name_matches, len(entry["pins"]), len(model))
            if best is None or rank > best[0]:
                best = (rank, model, text, name_matches, number_matches)
        if best is None or best[0][0] < min_matches:
            continue
        found[component["detection_index"]] = {
            "model": library[best[1]]["model"], "key": best[1], "printed_text": best[2],
            "name_matches": best[3], "number_matches": best[4],
            "library_pins": len(library[best[1]]["pins"])}
    return found


def _reading_order(pin):
    point = pin.get("point") or [0, 0]
    return (-round(float(point[1]), 3), round(float(point[0]), 3))


def _number_order(number):
    text = str(number)
    return (0, int(text)) if text.isdigit() else (1, 0)


def _has_observed_number(pin, evidence, prefixes=None):
    """A number the OCR actually read; anything else is a derived reading-order guess."""
    return bool(_is_evidence(pin.get("number_status"), evidence, prefixes)
                and str(pin.get("number") or "").strip())


def _entry_pairs(pins, numbers, evidence, prefixes=None):
    """Pins that carry BOTH an observed number and an observed name - the only way to
    check whether the entry's own numbering agrees with this page."""
    pairs = []
    for pin in pins:
        number = canonical_number(pin.get("number"))
        name = _norm(pin.get("pinname"))
        if not number or not name:
            continue
        if not (_is_evidence(pin.get("number_status"), evidence, prefixes)
                and _is_evidence(pin.get("pinname_status"), evidence, prefixes)):
            continue
        pairs.append((number, name))
    return pairs


def apply_library(pin_result, tokens, library, config):
    """Rewrite pin numbers/names in place from page text; return an audit dict.

    What is deliberately NOT done: an observed number is never overwritten by a
    name-derived one, and a number is never invented from the entry's pin order.  Both
    were measured to lose points - the library's own numbering does not always follow the
    reference (EasyEDA splits one package pin into `6_1`/`6_2`, and 0099's IR4428 entry
    puts a pin at 9 where the reference says 6), so a wrong number costs a false-positive
    pin AND a wrong net member.  The library therefore only fills what is missing, and
    only when the entry has been checked against the page.
    """
    components = pin_result.get("components") or []
    pins = pin_result.get("pin_candidates") or []
    by_owner = {}
    for pin in pins:
        if pin.get("owner_status") != "assigned":
            continue
        by_owner.setdefault(pin.get("owner_detection_index"), []).append(pin)
    models = identify_models(components, by_owner, tokens, library, config)
    return apply_models(pin_result, models, library, config)


def apply_library_from_records(pin_result, records, library, config):
    """Same filling, but the part is read from the component record's chosen Name.

    The deployed chain already resolves an IC's model name into `records[*]['Name']` before
    export, so there is no need to re-find the printed text here.  Only the model lookup is
    different; every guard (no overwriting, consistency gate) is shared with `apply_library`.
    """
    models = {}
    min_pins = int(config.get("min_pins", 3))
    fuzzy = bool(config.get("model_fuzzy_match", True))
    pattern = bool(config.get("name_set_pattern_match", True))
    by_owner = {}
    for pin in pin_result.get("pin_candidates") or []:
        if pin.get("owner_status") != "assigned":
            continue
        by_owner.setdefault(pin.get("owner_detection_index"), []).append(pin)
    for record in records or []:
        # Name is the chip model only on the new annotation; on the old one it is the
        # designator, and the library happens to contain a real part called U1 - adopting
        # that entry would bind 30 unrelated pins.  Require Name != key/designator.
        printed = _norm(record.get("Name"))
        if printed and printed in {_norm(record.get("component_id")), _norm(record.get("designator"))}:
            continue
        entry = _resolve_entry(record.get("Name"), library, by_owner.get(record.get("detection_index")) or [],
                               config, fuzzy=fuzzy, pattern=pattern)
        if entry is None:
            continue
        models[record.get("detection_index")] = {
            "model": entry["model"], "key": entry["key"], "printed_text": str(record.get("Name")),
            "identified_by": entry["how"], "library_pins": len(library[entry["key"]]["pins"]),
            "name_matches": entry.get("name_matches"), "number_matches": entry.get("number_matches")}
    return apply_models(pin_result, models, library, config)


def _name_set(entry):
    return {_norm(name) for name in entry["pins"].values() if _norm(name)}


def _number_for_name(entry, name):
    """Which number carries this name, using EVERY alias the library observed.

    The libraries aggregate several symbol instances, so one number can carry `GND`,
    `ADJ`, `ADJ(GND)`, `GND/ADJ`...  Collapsing to one spelling loses the others; the 13
    released pages need three such aliases (`Vin`, `ADJ(GND)`, `TAB`).
    """
    key = _norm(name)
    alias = (entry.get("aliases") or {}).get(key)
    if alias is not None:
        return alias
    for number, value in number_order_items(entry["pins"]):
        if _norm(value) == key:
            return number
    return None


def _resolve_entry(name, library, pins, config, fuzzy=True, pattern=True):
    """Exact key, then a near-miss spelling, then the pin-name set alone.

    Three levels, each stricter than the last about what it must agree with:

    * exact: the printed model is a library key;
    * fuzzy: one character dropped/added or one prefix of the other (OCR truncation, e.g.
      `HS96L03W2C0` vs the real `HS96L03W2C03`), and at least one read pin name is in the
      entry, and the ambiguous case is declined;
    * pattern: no model, but the entry's pin names cover every name we read AND reproduce
      every pin number we also read (at least two), which pins down the numbering without
      knowing the part.
    """
    key = _norm(name)
    min_pins = int(config.get("min_pins", 3))
    if key:
        entry = library.get(key)
        if entry and len(entry["pins"]) >= min_pins:
            return {"model": entry["model"], "key": key, "how": "component_record_Name"}
        if fuzzy and len(key) >= 8:
            near = [k for k in library
                    if len(library[k]["pins"]) >= min_pins and len(k) >= 8
                    and (k.startswith(key) or key.startswith(k))
                    and abs(len(k) - len(key)) <= 2]
            if len(near) == 1:
                entry = library[near[0]]
                read = {_norm(p.get("pinname")) for p in pins
                        if _is_evidence(p.get("pinname_status"), config.get("evidence_statuses") or DEFAULT_EVIDENCE,
                                        config.get("evidence_prefixes")) and _norm(p.get("pinname"))}
                if read & _name_set(entry):
                    return {"model": entry["model"], "key": near[0], "how": "fuzzy_model_spelling"}
    if not pattern:
        return None
    evidence = config.get("evidence_statuses") or DEFAULT_EVIDENCE
    prefixes = config.get("evidence_prefixes")
    read_names = {_norm(p.get("pinname")) for p in pins
                  if _is_evidence(p.get("pinname_status"), evidence, prefixes) and _norm(p.get("pinname"))}
    if len(read_names) < 2:
        return None
    pairs = [(canonical_number(p.get("number")), _norm(p.get("pinname"))) for p in pins
             if _is_evidence(p.get("number_status"), evidence)
             and _is_evidence(p.get("pinname_status"), evidence)
             and str(p.get("number") or "").strip() and _norm(p.get("pinname"))]
    if len(pairs) < 2:
        return None
    candidates = None
    for name_key in read_names:
        holders = {k for k, entry in library.items()
                   if len(entry["pins"]) >= min_pins and name_key in _name_set(entry)}
        candidates = holders if candidates is None else (candidates & holders)
        if not candidates:
            return None
    best = None
    for candidate in candidates or ():
        entry = library[candidate]
        names = _name_set(entry)
        if not read_names <= names:
            continue
        if any(_number_for_name(entry, name) != number for number, name in pairs):
            continue
        score = (len(pairs), -abs(len(entry["pins"]) - len(read_names)))
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        return None
    entry = library[best[1]]
    return {"model": entry["model"], "key": best[1], "how": "pin_name_pattern",
            "name_matches": len(read_names), "number_matches": len(pairs)}


def apply_models(pin_result, models, library, config):
    """Fill missing pin numbers/names for the components in `models` (in place)."""
    components = pin_result.get("components") or []
    pins = pin_result.get("pin_candidates") or []
    evidence = config.get("evidence_statuses") or DEFAULT_EVIDENCE
    prefixes = config.get("evidence_prefixes")
    fill_names = bool(config.get("fill_names_from_number", True))
    fill_numbers = bool(config.get("fill_missing_numbers_from_names", True))
    fill_leftover = bool(config.get("fill_remaining_numbers", False))
    min_pairs = int(config.get("min_consistency_pairs", 3))
    min_consistency = float(config.get("min_consistency", 0.6))

    by_owner = {}
    for pin in pins:
        if pin.get("owner_status") != "assigned":
            continue
        by_owner.setdefault(pin.get("owner_detection_index"), []).append(pin)

    changes = []
    audit = {"components_identified": len(models), "models": [], "pins_numbered_from_name": 0,
             "pins_named_from_library": 0, "pins_leftover_numbered": 0,
             "entries_rejected_by_consistency": 0,
             "rule": "fill only what is missing, after checking the entry against the page"}

    for component in components:
        info = models.get(component.get("detection_index"))
        if not info:
            continue
        owned = by_owner.get(component["detection_index"]) or []
        numbers = library[info["key"]]["pins"]
        name_to_number = {}
        for number, name in number_order_items(numbers):
            key = _norm(name)
            if key and key not in name_to_number:
                name_to_number[key] = number
        for alias, number in (library[info["key"]].get("aliases") or {}).items():
            name_to_number.setdefault(alias, number)

        pairs = _entry_pairs(owned, numbers, evidence, prefixes)
        agreed = sum(1 for number, name in pairs if name_to_number.get(name) == number)
        consistency = (agreed / len(pairs)) if pairs else None
        trusted = consistency is None or len(pairs) < min_pairs or consistency >= min_consistency
        if not trusted:
            audit["entries_rejected_by_consistency"] += 1
            audit["models"].append({**info, "consistency": round(consistency, 3),
                                    "consistency_pairs": len(pairs), "accepted": False})
            continue

        filled = set()
        for pin in owned:                                     # 1. name -> number
            if not fill_numbers or _has_observed_number(pin, evidence, prefixes):
                continue
            # A name that was only *filled* is derived; using it to derive a number would
            # compound two guesses, so numbering needs a directly observed name.
            if not _is_evidence(pin.get("pinname_status"), evidence):
                continue
            number = name_to_number.get(_norm(pin.get("pinname")))
            if number is None or number in filled:
                continue
            filled.add(number)
            previous = pin.get("number")
            pin["number"] = number
            pin["number_source"] = "library_pin_name"
            audit["pins_numbered_from_name"] += 1
            changes.append({"candidate_id": pin["candidate_id"], "model": info["model"],
                            "field": "number", "old": previous, "new": number,
                            "source": "library_pin_name"})
        # 2. number by elimination.  When the model is known and every pin but one already
        # carries a library number, the last number is not a guess - it is the only one left.
        # 39 of the 56 missing pin numbers on the 13 released pages fall in this case.
        if bool(config.get("fill_by_elimination", True)):
            used = {canonical_number(p.get("number")) for p in owned
                    if str(p.get("number") or "").strip()}
            library_numbers = {canonical_number(n) for n in numbers}
            unnumbered = [p for p in owned if not str(p.get("number") or "").strip()]
            leftover = library_numbers - used
            # a foreign number means our pin set does not match the entry (duplicate tips,
            # a neighbouring part): elimination would then invent an assignment
            if used <= library_numbers and len(leftover) == 1 and len(unnumbered) == 1:
                pin = unnumbered[0]
                number = leftover.pop()
                pin["number"] = number
                pin["number_source"] = "library_elimination"
                if not str(pin.get("pinname") or "").strip():
                    name = numbers.get(number)
                    if name:
                        pin["pinname"] = name
                        pin["pinname_source"] = "library_elimination"
                audit["pins_numbered_by_elimination"] = audit.get("pins_numbered_by_elimination", 0) + 1
                changes.append({"candidate_id": pin["candidate_id"], "model": info["model"],
                                "field": "number", "old": None, "new": number,
                                "source": "library_elimination"})

        if fill_leftover:                                     # 2. entry order (opt-in)
            used = {canonical_number(p.get("number")) for p in owned if str(p.get("number") or "").strip()}
            leftover = [n for n in sorted(numbers, key=_number_order) if n not in used]
            rest = sorted((p for p in owned if not str(p.get("number") or "").strip()), key=_reading_order)
            for pin, number in zip(rest, leftover):
                pin["number"] = number
                pin["number_source"] = "library_leftover_reading_order"
                audit["pins_leftover_numbered"] += 1
                changes.append({"candidate_id": pin["candidate_id"], "model": info["model"],
                                "field": "number", "old": None, "new": number,
                                "source": "library_leftover_reading_order"})
        for pin in owned:                                     # 3. names from the entry
            if not fill_names or str(pin.get("pinname") or "").strip():
                continue
            number = canonical_number(pin.get("number"))
            library_name = numbers.get(number) or ""
            if not library_name:
                continue
            pin["pinname"] = library_name
            pin["pinname_source"] = "library_pin_name"
            audit["pins_named_from_library"] += 1
            changes.append({"candidate_id": pin["candidate_id"], "model": info["model"],
                            "field": "pinname", "old": None, "new": library_name,
                            "source": "library_pin_name"})
        audit["models"].append({**info, "consistency": None if consistency is None else round(consistency, 3),
                                "consistency_pairs": len(pairs), "accepted": True})

    audit["changes"] = changes
    return audit


def number_order_items(numbers):
    return sorted(numbers.items(), key=lambda item: _number_order(item[0]))


def library_from_config(config_path):
    """Load the libraries named by a config file; returns (library, config).

    Relative paths are resolved against the working directory first and then against the
    project root (the parent of `configs/`), so the config works from either place.
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.resolve().parents[1]
    paths = []
    for item in config.get("libraries") or []:
        path = Path(item)
        if not path.is_absolute():
            candidates = [Path.cwd() / path, project_root / path]
            path = next((c for c in candidates if c.exists()), candidates[0])
        paths.append(path)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"component library not found: {missing}")
    return load_library(paths), config
