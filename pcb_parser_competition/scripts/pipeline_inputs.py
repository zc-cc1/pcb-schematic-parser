"""Shared helpers for replaying the pipeline on frozen or freshly produced files.

`run_wire_graph.py` hash-binds its wire cache to (image, model, inference config,
probability bytes).  When a driver produces the cache through
`run_pin_ownership.py`, nothing writes that metadata file, so every driver would
have to re-implement it.  This module is that one implementation.
"""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_wire_cache_metadata(ownership_dir, inference_config_path, source="pipeline_driver"):
    """Create the metadata JSON that run_wire_graph.py expects beside probabilities.npz."""
    ownership_dir = Path(ownership_dir)
    probabilities = ownership_dir / "probabilities.npz"
    summary = json.loads((ownership_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = {
        "image_sha256": summary["image_sha256"],
        "size": summary["size"],
        "coordinate_origin": "top_left",
        "weights_sha256": summary["unet_weights_sha256"],
        "inference_config_sha256": sha256_file(inference_config_path),
        "probabilities_sha256": sha256_file(probabilities),
        "prediction_only": True,
        "source": source,
        "unet_model_variant": summary.get("unet_model_variant"),
    }
    target = probabilities.with_suffix(".json")
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
