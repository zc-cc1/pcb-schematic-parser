from pathlib import Path

path = Path(r"D:\pcb5_submission\src\pcb_parser\pin_number_strip.py")
text = path.read_text(encoding="utf-8")

# a second window: the strip INSIDE the body on the pin's own row
old = '''def strip_crop(point, cfg):
    """Axis-aligned window centred on the pin tip; the number is printed just above it."""
    x, y = point
    return (int(x - cfg["along_px"]), int(y - cfg["up_px"]),
            int(x + cfg["along_px"]), int(y + cfg["down_px"]))'''
new = '''def strip_crop(point, cfg):
    """Axis-aligned window centred on the pin tip; the number is printed just above it."""
    x, y = point
    return (int(x - cfg["along_px"]), int(y - cfg["up_px"]),
            int(x + cfg["along_px"]), int(y + cfg["down_px"]))


def inside_crop(point, box, side, cfg):
    """Window INSIDE the body on the pin's own row.

    A header prints its pins as bare digits inside the body ("1 2 3"), and the reference uses
    those digits as the pin number too.  Only used when the outside strip found nothing.
    """
    depth = float(cfg.get("inside_depth_px", 34))
    half = float(cfg.get("inside_half_px", 8))
    x, y = point
    x1, y1, x2, y2 = box
    if side == "left":
        return (int(x1 + 3), int(y - half), int(x1 + depth), int(y + half))
    if side == "right":
        return (int(x2 - depth), int(y - half), int(x2 - 3), int(y + half))
    if side == "top":
        return (int(x - half), int(y1 + 3), int(x + half), int(y1 + depth))
    return (int(x - half), int(y2 - depth), int(x + half), int(y2 - 3))'''
assert old in text
text = text.replace(old, new, 1)

old_read = '''        readings = []
        for rotation in cfg.get("rotations", [0, 90]):
            upright = patch if rotation == 0 else cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
            for scale in cfg["scales"]:
                image = cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                result = engine(np.ascontiguousarray(image[:, :, ::-1]), use_det=True, use_cls=True, use_rec=True)
                for text, confidence in zip(result.txts or [], result.scores or []):
                    text = str(text).strip()
                    if not (NUMBER.match(text) or BALL.match(text)):
                        continue
                    if float(confidence) < cfg["min_confidence"]:
                        continue
                    readings.append((text, float(confidence)))'''
new_read = '''        def ocr_readings(region):
            collected = []
            if region.size == 0:
                return collected
            for rotation in cfg.get("rotations", [0, 90]):
                upright = region if rotation == 0 else cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
                for scale in cfg["scales"]:
                    image = cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    result = engine(np.ascontiguousarray(image[:, :, ::-1]), use_det=True, use_cls=True, use_rec=True)
                    for text, confidence in zip(result.txts or [], result.scores or []):
                        text = str(text).strip()
                        if not (NUMBER.match(text) or BALL.match(text)):
                            continue
                        if float(confidence) < cfg["min_confidence"]:
                            continue
                        collected.append((text, float(confidence)))
            return collected

        readings = ocr_readings(patch)
        source = "strip"'''
assert old_read in text
text = text.replace(old_read, new_read, 1)

old_choose = '''        numbers = (library_numbers or {}).get(record.get("component_id")) or set()
        text, why = choose(readings, numbers, cfg)'''
new_choose = '''        numbers = (library_numbers or {}).get(record.get("component_id")) or set()
        text, why = choose(readings, numbers, cfg)
        if text is None and cfg.get("inside_digit_fallback", True):
            inner = inside_crop(image_point, [float(v) for v in box], side, cfg)
            region = rgb[max(0, inner[1]):max(0, inner[3]), max(0, inner[0]):max(0, inner[2])]
            inside_readings = ocr_readings(region)
            text, why = choose(inside_readings, numbers, cfg)
            if text is not None:
                source = "inside_body"
                readings = inside_readings'''
assert old_choose in text
text = text.replace(old_choose, new_choose, 1)

old_row = '''        adopted.append({"candidate_id": pin["candidate_id"], "component_id": record.get("component_id"),
                        "number": text, "how": why, "readings": readings[:4]})'''
new_row = '''        adopted.append({"candidate_id": pin["candidate_id"], "component_id": record.get("component_id"),
                        "number": text, "how": why, "source": source, "readings": readings[:4]})'''
assert old_row in text
text = text.replace(old_row, new_row, 1)

path.write_text(text, encoding="utf-8")
print("inside-body digit fallback added")

import json

config = Path(r"D:\pcb5_submission\configs\pin_number_strip_v1.json")
payload = json.loads(config.read_text(encoding="utf-8"))
payload["inside_digit_fallback"] = True
payload["inside_depth_px"] = 34
payload["inside_half_px"] = 8
config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("config updated")
