"""Shared read-only audit helpers. Derived artifacts never enter production."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"runs/unet_root_cause_audit_v1"
sys.path.insert(0, str(ROOT/"src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def rows():
    source = [json.loads(s) for s in (ROOT/"data/segmentation_unet_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    return [r for r in source if r["structurally_usable_draft"] and r["split"] in ("train", "val")]


def image_path(case):
    return ROOT/f"data/standardized/{case}/{case}.png"


def gt_path(case):
    return ROOT/f"data/segmentation_unet_v1/{case}/wire_mask.png"


def image(case):
    return np.asarray(Image.open(image_path(case)).convert("RGB"))


def truth(case):
    return np.asarray(Image.open(gt_path(case))) >= 128


def probability(case):
    with np.load(OUT/f"baseline/{case}/probabilities.npz", allow_pickle=False) as source:
        return source["wire"].copy()


def counts(mask, gt, roi=None):
    if roi is not None:
        mask, gt = mask[roi], gt[roi]
    return int((mask & gt).sum()), int((mask & ~gt).sum()), int((~mask & gt).sum())


def metrics(tp, fp, fn, empty=0.):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
        precision=tp/(tp+fp) if tp+fp else empty,
        recall=tp/(tp+fn) if tp+fn else empty,
        dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else empty)


def aggregate(records, key="wire"):
    return metrics(*(sum(r[key][name] for r in records) for name in ("tp", "fp", "fn")))


def assert_protected():
    protected = read(OUT/"protocol.json")["protected_sources"]
    changed = [p for p, h in protected.items() if sha(p) != h]
    if changed:
        raise ValueError(f"Frozen sources changed: {changed[:5]}")
    return len(protected)


def val_cases():
    return read(OUT/"protocol.json")["validation_unique_cases"]
