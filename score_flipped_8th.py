"""Score the platform's 8th-eval predictions locally after undoing the H-y export flip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from open_gt_metrics import evaluate_cases  # noqa: E402
from PIL import Image  # noqa: E402

PRED = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第8次测评-54.92/intermediate_files/final_results")
GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
IMG = Path(r"D:/pcb_open_gt_in")


def flip(doc, height):
    for comp in (doc.get("components") or {}).values():
        b = comp.get("bbox")
        if b and len(b) == 4:
            comp["bbox"] = [b[0], height - b[3], b[2], height - b[1]]
    for pins in (doc.get("pins") or {}).values():
        for pin in pins if isinstance(pins, list) else pins.values():
            pt = pin.get("point") if isinstance(pin, dict) else None
            if pt:
                pt["y"] = height - pt["y"]
    return doc


pairs = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
    with Image.open(next(folder.glob("*.png"))) as h:
        height = float(h.height)
    path = PRED / f"{case}_Pred.json"
    pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
    pairs.append((case, flip(pred, height), truth, height))

report = evaluate_cases(pairs)
print(f"component {report['metrics']['component']['f1']:.4f}  pin {report['metrics']['pin']['f1']:.4f}  "
      f"net {report['metrics']['net_hypergraph']['f1']:.4f}  wire {report['metrics']['net_line']['f1']:.4f}")
print(f"weighted {report['weighted_total']:.4f}")
for e in report["per_case"]:
    print(f"  case {e['case_id']:>2}: C {e['component']['f1']:.3f}  P {e['pin']['f1']:.3f}  "
          f"N {e['net_hypergraph']['f1']:.3f}  W {e['net_line']['f1']:.3f}")
