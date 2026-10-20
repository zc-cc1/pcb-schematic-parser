"""离线修复：裁切 OCR + 元件库，补引脚编号/名称/位置。

四步（每一步都能单独开关，默认只开 fill —— 见下面的实测）：

  fill  —— 库表里有、但没检到的编号：按"同名行"找到对应引线，补一个脚（默认开）
  name  —— 裁切 OCR 读出框内引脚名，按库表 名称→编号 重编号（默认关，实测中性）
  tip   —— 把引脚点吸附到引线外端（默认关，实测 -1 TP：会把已对的脚推坏）
  read  —— 直接按印刷编号建脚（默认关，实测 +1 TP/ -7 FP：编号常被 OCR 粘成 "23"）
  model —— 型号写成了位号时，用框内丝印替换（始终尝试；同时解锁库）

13 例实测（v33 产物为基线）：
    基线                 pin tp/fp/fn = 140/22/51  PinF1 0.7932
    fill（缺失 ≤6）       pin tp/fp/fn = 147/26/44  PinF1 0.8077   <= 默认
    name 叠加            不变
    tip                  tp 139  -> 关闭
    read                 tp 141 / fp 29 -> 关闭

坐标系：预测 JSON 是 bottom-left，图像与引线检测是 top-left，所有写入点都做 height-y 转换。
只改 box/ic；所有改动写入 report，任何一步不满足闸门就跳过。

    & $py pin_repair_library.py --predictions D:/pcb_open_gt_v33angle120/final_results --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹")
sys.path.insert(0, str(ROOT / "pcb_parser_competition/scripts"))
from platform_metrics import is_ic, score_case  # noqa: E402

GT = ROOT / "赛题六公开数据集/赛题六公开数据集/open_GT/open_GT"
LIB_PATH = ROOT / "官方资料/BoxICLibrary.json"

MARGIN = 25
SCALE = 3.0
NAME_RADIUS = 55.0

_engine = None
_CACHE = {}
_CACHE_PATH = ROOT / "runs" / "crop_ocr_cache.json"


def load_cache():
    global _CACHE
    if _CACHE_PATH.is_file():
        try:
            _CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}


def save_cache():
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")


def engine():
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def norm(text):
    return re.sub(r"[\s\-_/()（）°]+", "", str(text).upper())


class Library:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.models = {}
        for model, body in raw.items():
            table = {}
            order = []
            for pin in body.get("pins") or []:
                number = str(pin.get("number") or "").strip()
                name = str(pin.get("name") or "").strip()
                if not number or number in table:
                    continue
                table[number] = name
                order.append(number)
            self.models[model] = table
        self.by_norm = defaultdict(list)
        for model in self.models:
            self.by_norm[norm(model)].append(model)

    def resolve(self, text, designator=""):
        """型号 -> 标准键。返回 (model, 方式)。位号/数字/太短的候选直接拒绝。"""
        raw = str(text or "").strip()
        if len(raw) < 3 or not re.search(r"[A-Za-z]", raw):
            return None, "候选太短"
        if designator and raw.upper() == str(designator).upper():
            return None, "等于位号"
        if raw in self.models:
            return raw, "精确"
        key = norm(raw)
        if key in self.by_norm:
            return self.by_norm[key][0], "去符号"
        best = max(self.by_norm, key=lambda k: SequenceMatcher(None, key, k).ratio())
        ratio = SequenceMatcher(None, key, best).ratio()
        if ratio >= 0.9 and len(key) >= 4:
            return self.by_norm[best][0], f"近似{ratio:.2f}"
        return None, "库中无"

    def name_to_number(self, model):
        out = {}
        for number, name in self.models[model].items():
            if name:
                out.setdefault(norm(name), number)
        return out


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def crop_ocr(image, box, margin=MARGIN, scale=SCALE, cache_key=None):
    """box 是左上原点的像素框；返回 [(text, conf, x, y)]（坐标已还原到整图）。"""
    if cache_key and cache_key in _CACHE:
        return [tuple(item) for item in _CACHE[cache_key]]
    height, width = image.shape[:2]
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(width, int(box[2]) + margin)
    y2 = min(height, int(box[3]) + margin)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = engine()(big)
    if not getattr(result, "txts", None):
        return []
    out = []
    for box_pts, text, score in zip(result.boxes, result.txts, result.scores):
        pts = np.asarray(box_pts, dtype=float).reshape(-1, 2)
        out.append((str(text).strip(), float(score),
                    x1 + float(pts[:, 0].mean()) / scale, y1 + float(pts[:, 1].mean()) / scale))
    if cache_key:
        _CACHE[cache_key] = out
    return out


def dark_red(image):
    b, g, r = image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)
    return (r > 70) & (g < 75) & (b < 75) & (r - g > 35)


def find_leads(image, box, max_len=70, min_run=5):
    """在四条边外侧找引线，返回 [(side, cross, tip_x, tip_y, length)]。

    引线是细的深红直线；碰到 GND/VCC 符号时线会延续下去，表现为长度远超同元件其它引线。
    这里按"同元件中位长度"把这种异常剔除，再用中位长度回填（GT 的点就在引线外端）。
    """
    mask = dark_red(image)
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    raw = []

    def scan(cross_range, positions, side):
        for cross in cross_range:
            run, outer = 0, None
            for pos in positions:
                row, col = (cross, pos) if side in ("L", "R") else (pos, cross)
                if mask[row, col]:
                    run += 1
                    outer = pos
                elif run:
                    break
            if run >= min_run and outer is not None:
                raw.append((side, cross, outer, run))

    scan(range(y1, y2 + 1), range(x2 + 2, min(width, x2 + max_len)), "R")
    scan(range(y1, y2 + 1), range(x1 - 2, max(0, x1 - max_len), -1), "L")
    scan(range(x1, x2 + 1), range(y1 - 2, max(0, y1 - max_len), -1), "T")
    scan(range(x1, x2 + 1), range(y2 + 2, min(height, y2 + max_len)), "B")

    # 合并 cross 相邻的行，得到一条引线
    merged = []
    for side, cross, outer, run in sorted(raw, key=lambda r: (r[0], r[1])):
        if merged and merged[-1]["side"] == side and cross - merged[-1]["crosses"][-1] <= 2:
            merged[-1]["crosses"].append(cross)
            merged[-1]["outers"].append(outer)
            merged[-1]["runs"].append(run)
        else:
            merged.append({"side": side, "crosses": [cross], "outers": [outer], "runs": [run]})

    leads = []
    for item in merged:
        if len(item["crosses"]) > 12:     # 太厚，是符号不是引线
            continue
        length = float(np.median(item["runs"]))
        cross = float(np.mean(item["crosses"]))
        outer = float(np.median(item["outers"]))
        if item["side"] in ("L", "R"):
            leads.append((item["side"], cross, outer, cross, length))
        else:
            leads.append((item["side"], cross, cross, outer, length))

    if not leads:
        return []
    return leads


def snap_pins(existing, leads, px_box, height):
    """把引脚吸附到引线外端；符号粘连的异常长引线用同元件中位长度回填。"""
    x1, y1, x2, y2 = px_box
    matched = []
    for pin_key, pin in existing.items():
        point = (float(pin["point"]["x"]), float(pin["point"]["y"]))
        side = pin_side(point, px_box, height)
        same = [lead for lead in leads if lead[0] == side]
        if not same:
            continue
        nearest = min(same, key=lambda lead: distance_to_lead(point, lead))
        cross_gap, along = distance_to_lead(point, nearest)
        if cross_gap > 4.0:
            continue
        matched.append((pin_key, point, side, nearest, cross_gap, along))
    if not matched:
        return 0
    median = float(np.median([item[3][4] for item in matched]))
    snapped = 0
    for pin_key, point, side, lead, cross_gap, along in matched:
        length = lead[4]
        if length < median * 0.5:          # 太短，是文字/杂点，不是引线
            continue
        if length > median * 1.6:
            tip = {"R": (x2 + median, point[1]), "L": (x1 - median, point[1]),
                   "T": (point[0], y1 - median), "B": (point[0], y2 + median)}[side]
        else:
            tip = (lead[2], lead[3])
        # 尖端到本体边缘的距离必须像一条引线（5–55px），否则是杂点或远处的别的器件
        edge_gap = {"R": tip[0] - x2, "L": x1 - tip[0], "T": y1 - tip[1], "B": tip[1] - y2}[side]
        if not (5.0 <= edge_gap <= 55.0):
            continue
        # 只处理"尖端还在里面"的：向外位移 0.5–40px（point 是 bottom-left，tip 是图像坐标）
        point_image = (point[0], height - point[1])
        outward = {"R": tip[0] - point_image[0], "L": point_image[0] - tip[0],
                   "T": point_image[1] - tip[1], "B": tip[1] - point_image[1]}.get(side, 0)
        if not (6.0 <= outward <= 40):
            continue
        existing[pin_key]["point"] = {"x": float(tip[0]), "y": float(height - tip[1])}
        snapped += 1
    return snapped


def pin_side(point, box, height):
    """引脚在元件哪一侧：按它相对 bbox 的偏移方向判断。"""
    cx = (box[0] + box[2]) / 2
    cy = height - (box[1] + box[3]) / 2
    dx = point[0] - cx
    dy = point[1] - cy
    if abs(dx) >= abs(dy):
        return "R" if dx > 0 else "L"
    return "T" if dy > 0 else "B"


def side_of_lead(lead):
    return lead[0]


def read_component_pins(texts, leads, px_box):
    """从裁切图直接读引脚：印刷编号 + 同名行的引脚名 + 对应引线的外端点。

    不依赖元件库，适用于"编号印在图上"的器件（USB1、H1、5V2A 框等）。
    """
    names = [(t, x, y) for t, _c, x, y in texts
             if not re.fullmatch(r"\d{1,2}", t) and 1 <= len(t) <= 18]
    found = {}
    for text, _conf, x, y in texts:
        label = text.strip()
        if not re.fullmatch(r"\d{1,2}", label):
            continue
        best = None
        for lead in leads:
            side, tip_x, tip_y = lead[0], lead[2], lead[3]
            if side in ("L", "R"):
                if abs(y - tip_y) > 10 or abs(x - tip_x) > 50:
                    continue
                gap = abs(y - tip_y)
            else:
                if abs(x - tip_x) > 10 or abs(y - tip_y) > 50:
                    continue
                gap = abs(x - tip_x)
            if best is None or gap < best[0]:
                best = (gap, lead)
        if best is None:
            continue
        lead = best[1]
        name, name_gap = None, None
        for text_name, nx, ny in names:
            if lead[0] in ("L", "R"):
                if abs(ny - lead[3]) > 8 or not (px_box[0] - 5 <= nx <= px_box[2] + 5):
                    continue
                gap = abs(nx - lead[2])
            else:
                if abs(nx - lead[2]) > 8 or not (px_box[1] - 5 <= ny <= px_box[3] + 5):
                    continue
                gap = abs(ny - lead[3])
            if name is None or gap < name_gap:
                name, name_gap = text_name, gap
        found.setdefault(label, (lead, name))
    return found


def distance_to_lead(point, lead):
    side, cross, tip_x, tip_y = lead[0], lead[1], lead[2], lead[3]
    if side in ("L", "R"):
        return abs(point[1] - cross), abs(point[0] - tip_x)
    return abs(point[0] - cross), abs(point[1] - tip_y)


def build(args):
    library = Library(LIB_PATH)
    source = Path(args.predictions)
    out_dir = Path(args.out)
    (out_dir / "final_results").mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "components": [], "cases": []}
    totals = Counter()

    for folder in sorted((p for p in args.gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        pred = json.loads((source / f"{case}_Pred.json").read_text(encoding="utf-8"))
        image = read_image(next(folder.glob("*.png")))
        height = image.shape[0]

        pins = pred.setdefault("pins", {})
        components = pred.setdefault("components", {})
        renamed = {}
        for key, component in list(components.items()):
            if not is_ic(component.get("type")):
                continue
            bbox = component.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            px_box = [bbox[0], height - bbox[3], bbox[2], height - bbox[1]]
            texts = crop_ocr(image, px_box, cache_key=f"{case}:{key}")
            designator = key
            existing = pins.get(key) or {}
            leads = find_leads(image, px_box)
            read = read_component_pins(texts, leads, px_box) if args.read else {}

            # --- model：型号若等于位号，尝试用框内丝印替换 ---
            model = component.get("Name")
            model_how = "未改"
            if str(model or "").strip().upper() == designator.upper() or not model:
                best = None
                for text, conf, x, y in texts:
                    candidate, how = library.resolve(text, designator)
                    if not candidate:
                        continue
                    inside = px_box[0] - 10 <= x <= px_box[2] + 10 and px_box[1] - 10 <= y <= px_box[3] + 10
                    if len(library.models[candidate]) < max(len(existing), 1):
                        continue
                    score = (inside, conf, len(candidate))
                    if best is None or score > best[0]:
                        best = (score, candidate, how)
                if best:
                    model, model_how = best[1], best[2]
                    renamed[key] = f"{component.get('Name')!r}->{model!r}"
                    component["Name"] = model
            table = library.models.get(model) if model and model in library.models else None
            by_name = library.name_to_number(model) if table else {}

            # --- read：直接按裁切图上的印刷编号建脚（不需要元件库）---
            built = 0
            if args.read and read and len(existing) <= 1:
                names = [name for _lead, name in read.values() if name]
                header_style = not names          # 全都没有名称 -> 排针式（名称=编号）
                new_pins, remap = {}, {}
                ok = True
                for number, (lead, name) in read.items():
                    label = name or (number if header_style else None)
                    if label is None:
                        ok = False
                        break
                    new_pins[f"pin_{number}"] = {"pinname": label,
                                                 "point": {"x": lead[2], "y": height - lead[3]}}
                if ok and len(new_pins) >= max(2, len(existing)):
                    for number, (lead, _name) in read.items():
                        near = min(existing.items(),
                                   key=lambda kv: math.hypot(kv[1]["point"]["x"] - lead[2],
                                                             kv[1]["point"]["y"] - lead[3]),
                                   default=(None, None))
                        if near[0] and math.hypot(near[1]["point"]["x"] - lead[2],
                                                  near[1]["point"]["y"] - lead[3]) <= 3:
                            remap[near[0]] = f"pin_{number}"
                    built = len(new_pins)
                    pins[key] = new_pins
                    existing = new_pins
                    if remap:
                        for net in (pred.get("nets") or {}).values():
                            body = net.get("hyperGraph")
                            if not body or f"{key}." not in str(body):
                                continue

                            def sub(match, remap=remap, key=key):
                                token = match.group(0)
                                owner, _, port = token.partition(".")
                                if owner == key and f"pin_{port}" in remap:
                                    return f"{owner}.{remap[f'pin_{port}'].replace('pin_', '')}"
                                return token

                            net["hyperGraph"] = re.sub(rf"{re.escape(key)}\.[A-Za-z0-9_]+", sub, str(body))

            # --- tip：把已有引脚吸附到引线外端 ---
            snapped = snap_pins(existing, leads, px_box, height) if args.tip else 0

            # --- name：裁切名称 -> 库编号，重编号 ---
            # 闸门：只在"现在的编号和库表明显不符"时才动，避免把已经对的脚改坏。
            agree = total = 0
            for pin_key, pin in existing.items():
                number = pin_key.replace("pin_", "")
                if table and number in table and pin.get("pinname"):
                    total += 1
                    agree += int(norm(pin["pinname"]) == norm(table[number]))
            numbered_wrong = total > 0 and agree / total < 0.7
            stray = [k for k in existing if table and k.replace("pin_", "") not in table]

            mapping = {}
            if args.name and table and (numbered_wrong or stray):
                for pin_key, pin in existing.items():
                    point = (float(pin["point"]["x"]), float(pin["point"]["y"]))
                    side = pin_side(point, px_box, height)
                    best = None
                    for text, conf, x, y in texts:
                        number = by_name.get(norm(text))
                        if not number:
                            continue
                        if not (px_box[0] - 5 <= x <= px_box[2] + 5 and px_box[1] - 5 <= y <= px_box[3] + 5):
                            continue
                        # 名称必须和引脚同一行（L/R 侧）或同一列（T/B 侧）
                        if side in ("L", "R"):
                            gap = abs(y - point[1])
                        else:
                            gap = abs(x - point[0])
                        if gap > 9:
                            continue
                        if best is None or gap < best[0]:
                            best = (gap, number)
                    if not best:
                        continue
                    number = best[1]
                    if number in mapping.values():
                        continue
                    mapping[pin_key] = number

            if mapping and len(set(mapping.values())) == len(mapping):
                rebuilt = {}
                for pin_key, pin in existing.items():
                    number = mapping.get(pin_key)
                    new_key = f"pin_{number}" if number else pin_key
                    if new_key in rebuilt:
                        rebuilt[pin_key] = pin
                        continue
                    rebuilt[new_key] = pin
                    if number:
                        pin["pinname"] = table.get(number) or pin.get("pinname")
                if len(rebuilt) == len(existing):
                    pins[key] = rebuilt
                    existing = rebuilt
                    # 同步 net 里的 元件.编号 引用
                    for net in (pred.get("nets") or {}).values():
                        body = net.get("hyperGraph")
                        if not body or f"{key}." not in str(body):
                            continue
                        def sub(match, mapping=mapping, key=key):
                            token = match.group(0)
                            owner, _, port = token.partition(".")
                            number = mapping.get(f"pin_{port}")
                            return f"{owner}.{number}" if owner == key and number else token
                        net["hyperGraph"] = re.sub(rf"{re.escape(key)}\.[A-Za-z0-9_]+", sub, str(body))

            # --- fill：库里有、但没检到的编号，用同侧引线补 ---
            added = 0
            if args.fill and table:
                used = {k.replace("pin_", "") for k in existing}
                missing = [n for n in table if n not in used]
                if 0 < len(missing) <= 6:
                    by_side = defaultdict(list)
                    for lead in leads:
                        side, cross, tip_x, tip_y = lead[0], lead[1], lead[2], lead[3]
                        near_pin = [p for p in existing.values()
                                    if math.hypot(p["point"]["x"] - tip_x, p["point"]["y"] - tip_y) <= 25]
                        if not near_pin:
                            by_side[side].append(lead)
                    for number in missing:
                        name = table[number]
                        want = norm(name)
                        best = None
                        for side, candidates in by_side.items():
                            for lead in candidates:
                                tip_x, tip_y = lead[2], lead[3]
                                for text, conf, x, y in texts:
                                    if norm(text) != want:
                                        continue
                                    gap = abs(y - tip_y) if side in ("L", "R") else abs(x - tip_x)
                                    if gap > 12:
                                        continue
                                    if best is None or gap < best[0]:
                                        best = (gap, lead)
                        if not best:
                            continue
                        _, lead = best
                        bucket = pins.setdefault(key, {})
                        bucket[f"pin_{number}"] = {"pinname": name,
                                                   "point": {"x": lead[2], "y": height - lead[3]}}
                        by_side[lead[0]].remove(lead)
                        added += 1

            report["components"].append({"case": case, "key": key,
                                         "status": "ok" if table else "无库表(仅read)",
                                         "model": model, "how": model_how,
                                         "pins": len(pins.get(key) or {}), "read": built,
                                         "snapped": snapped, "renumbered": len(mapping), "added": added})
            totals["built"] += built
            totals["snapped"] += snapped
            totals["renumbered"] += len(mapping)
            totals["added"] += added
            totals["components"] += 1

        (out_dir / "final_results" / f"{case}_Pred.json").write_text(
            json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
        before = score_case(pred, truth)
        report["cases"].append({"case": case, "pin_f1": before["pin_f1"]["f1"],
                                "pin": {k: before["pin_f1"][k] for k in ("tp", "fp", "fn")},
                                "component_f1": before["component_f1"]["f1"],
                                "final": before["final_score"]})
        if renamed:
            report.setdefault("renamed", {})[case] = renamed

    (out_dir / "pin_repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    pins_tp = sum(c["pin"]["tp"] for c in report["cases"])
    pins_fp = sum(c["pin"]["fp"] for c in report["cases"])
    pins_fn = sum(c["pin"]["fn"] for c in report["cases"])
    print(f"改动统计: {dict(totals)}")
    print(f"修复后 pin tp/fp/fn = {pins_tp}/{pins_fp}/{pins_fn} "
          f"(F1 {2 * pins_tp / max(2 * pins_tp + pins_fp + pins_fn, 1):.4f})")
    print("输出:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gt-root", type=Path, default=GT)
    parser.add_argument("--tip", action="store_true", default=False)
    parser.add_argument("--no-tip", dest="tip", action="store_false")
    parser.add_argument("--name", action="store_true", default=False)
    parser.add_argument("--no-name", dest="name", action="store_false")
    parser.add_argument("--fill", action="store_true", default=True)
    parser.add_argument("--no-fill", dest="fill", action="store_false")
    parser.add_argument("--read", action="store_true", default=False)
    parser.add_argument("--no-read", dest="read", action="store_false")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_cache()
    try:
        build(args)
    finally:
        save_cache()


if __name__ == "__main__":
    main()
