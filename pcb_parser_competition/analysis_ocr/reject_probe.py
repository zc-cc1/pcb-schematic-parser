"""Why did a designator token that sits on the symbol lose the assignment?"""
import json
import sys
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcb_parser import component_text as CT  # noqa: E402
from replay_binder import build_case, load_configs, inverse_vocabulary, BATCH  # noqa: E402

adopted = load_configs()
inv = inverse_vocabulary(adopted)
for case in ("0001", "0018", "0080", "0102"):
    detections, texts, w, h, cfg = build_case(case, adopted, inv)
    output, records, diag = CT.bind_components(detections, texts, w, h, cfg)
    print(f"\n================ {case} ================")
    print("counts:", diag["counts"])
    used = {b[f]["token_id"] for b in (r["fields"] for r in records) for f in ("designator", "Name", "value") if f in b}
    designators = [t for t in diag["tokens"] if t["role"] == "designator"]
    unassigned = [t for t in designators if t["token_id"] not in used]
    print(f"designator tokens: {len(designators)}, unassigned: {len(unassigned)}")
    types = [cfg["class_mapping"].get(d["class_name"], d["class_name"]) for d in detections]
    for t in unassigned[:14]:
        # nearest 3 detections with the prefix gate evaluated
        scored = []
        for i, (d, kind) in enumerate(zip(detections, types)):
            g = CT.gap(t["bbox"], d["bbox"])
            allowed = t["prefix"] in CT.PREFIXES.get(kind, set())
            scored.append((g, i, kind, allowed))
        scored.sort()
        top = ", ".join(f"d{i}:{k}{'' if a else '(prefix×)'}@{g:.0f}px" for g, i, k, a in scored[:3])
        print(f"  {t['text']:10s} conf={t['confidence']:.3f} prefix={t['prefix']:4s} -> {top}")
    print("--- rejected_matches by reason ---")
    from collections import Counter
    print(Counter(r["reason"] for r in diag["rejected_matches"]))
    for r in diag["rejected_matches"][:6]:
        print("   ", {k: r[k] for k in r if k in ("field", "reason", "detection_index", "candidate_index",
                                                  "score", "global_assignment_margin", "fragment")})
