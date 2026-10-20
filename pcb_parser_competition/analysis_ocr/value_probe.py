"""Evidence for every passive component whose value decision differs from GT."""
import json
import math
import sys
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapter  # noqa: E402
from pcb_parser import component_text as CT  # noqa: E402
from replay_binder import build_case, load_configs, inverse_vocabulary  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]
adopted = load_configs()
inv = inverse_vocabulary(adopted)


def main():
    for case in CASES:
        gt = json.loads(next((GT_ROOT / case).glob("*_target_named.json")).read_text(encoding="utf-8"))
        detections, texts, w, h, cfg = build_case(case, adopted, inv)
        output, records, diag = CT.bind_components(detections, texts, w, h, cfg)
        types = [cfg["class_mapping"].get(d["class_name"], d["class_name"]) for d in detections]
        shown = 0
        for rec in records:
            cid = rec["component_id"]
            if types[rec["detection_index"]] not in {"resistor", "capacitor", "inductor"}:
                continue
            g = gt["components"].get(cid)
            if g is None:
                continue
            gv = g.get("value")
            pv = rec["value"]
            if (gv or None) == (pv or None):
                continue
            fields = rec.get("fields", {})
            dbox = (fields.get("designator") or {}).get("bbox")
            vtok = fields.get("value") or {}
            # nearest other designator to that value token
            near = []
            if vtok.get("bbox"):
                for other in records:
                    od = (other.get("fields", {}).get("designator") or {})
                    if other["component_id"] == cid or not od.get("bbox"):
                        continue
                    near.append((round(math.dist(CT.center(vtok["bbox"]), CT.center(od["bbox"])), 1),
                                 other["component_id"]))
            near.sort()
            print(f"\n{case}/{cid} type={types[rec['detection_index']]:10s} "
                  f"GT={gv!r} PRED={pv!r}")
            print(f"   designator bbox={[round(v) for v in dbox] if dbox else None}")
            print(f"   value token text={vtok.get('text')!r} conf={vtok.get('confidence')} "
                  f"role={vtok.get('role')} bbox={[round(v) for v in vtok.get('bbox', [])] or None}")
            print(f"   nearest other designators: {near[:3]}")
            shown += 1
            if shown >= 8:
                break


if __name__ == "__main__":
    sys.exit(main())
