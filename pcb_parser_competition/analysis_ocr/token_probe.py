"""Dump the tokenizer's view of a page: roles, fused leftovers, suspects."""
import json
import sys
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcb_parser.component_text import tokens_from_texts, value_kind, REF, REF_SCAN  # noqa: E402
from replay_binder import build_case, load_configs, inverse_vocabulary, BATCH  # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "0001"
adopted = load_configs()
inv = inverse_vocabulary(adopted)
detections, texts, w, h, cfg = build_case(case, adopted, inv)
tokens = tokens_from_texts(texts)
print(f"{case}: {len(texts)} text rows -> {len(tokens)} tokens")
from collections import Counter
print("roles:", Counter(t["role"] for t in tokens))
print("\n-- every token whose source_text was split or that looks ref/value fused --")
for t in tokens:
    if t["approximate_split_geometry"]:
        print(f"  {t['role']:16s} {t['text']!r:14s} from {t['source_text']!r:16s} conf={t['confidence']:.3f} bbox={[round(v) for v in t['bbox']]}")
print("\n-- ref-shaped source strings that still contain a value unit --")
for t in tokens:
    s = t["source_text"].strip()
    if REF.fullmatch(s):
        tail = [s[:i] for i in range(1, len(s))]
        for ref in tail:
            if REF.fullmatch(ref) and value_kind(s[len(ref):]) is not None and len(s) - len(ref) >= 2:
                print(f"  unresolved fusion: {s!r} could be {ref!r} + {s[len(ref):]!r} (role={t['role']})")
                break
print("\n-- model candidates (Name) --")
for t in tokens:
    if t["role"] == "model_candidate":
        print(f"  {t['text']!r:28s} conf={t['confidence']:.3f} bbox={[round(v) for v in t['bbox']]}")
