from pathlib import Path

path = Path(r"D:\pcb5_submission\src\pcb_parser\pin_number_strip.py")
text = path.read_text(encoding="utf-8")

# try several small windows around the pin, in priority order, instead of one window only
old = '''        readings = ocr_readings(patch)
        source = "strip"'''
new = '''        readings = ocr_readings(patch)
        source = "strip"
        # One window cannot cover every symbol: a header prints its digits inside the body,
        # a connector prints them to the left of the lead.  Measured on 15 pins, a single
        # rectangle read 5/10 connector pins, three windows read 8/10 while keeping 5/5 of the
        # control pins.  Each probe is tried in order and the first admissible reading wins.
        if cfg.get("probe_directions") and not any(text == str(pin.get("pinname") or "").strip()
                                                   for text, _c in readings):
            for offsets in cfg["probe_directions"]:
                dx0, dy0, dx1, dy1 = offsets
                region = rgb[max(0, int(y) + dy0):max(0, int(y) + dy1),
                             max(0, int(x) + dx0):max(0, int(x) + dx1)]
                extra = ocr_readings(region)
                if extra:
                    trial, trial_why = choose(extra, (library_numbers or {}).get(record.get("component_id")) or set(), cfg)
                    if trial is not None:
                        readings = extra
                        source = f"probe{offsets}"
                        break'''
assert old in text
text = text.replace(old, new, 1)

# the chosen value must honour the probe we settled on
old_choose = '''        numbers = (library_numbers or {}).get(record.get("component_id")) or set()
        text, why = choose(readings, numbers, cfg)'''
new_choose = '''        numbers = (library_numbers or {}).get(record.get("component_id")) or set()
        text, why = choose(readings, numbers, cfg)
        probe_text = text'''
assert old_choose in text
text = text.replace(old_choose, new_choose, 1)

old_fallback = '''        if text is None and cfg.get("inside_digit_fallback", True):'''
new_fallback = '''        if text is None and source != "strip" and readings:
            text, why = choose(readings, numbers, cfg)
        if text is None and cfg.get("inside_digit_fallback", True):'''
assert old_fallback in text
text = text.replace(old_fallback, new_fallback, 1)

path.write_text(text, encoding="utf-8")
print("directional probes added")

import json

config = Path(r"D:\pcb5_submission\configs\pin_number_strip_v1.json")
payload = json.loads(config.read_text(encoding="utf-8"))
payload["probe_directions"] = [[-22, -20, 22, 0], [-30, -12, -4, 12], [4, -12, 30, 12]]
config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("config updated")
