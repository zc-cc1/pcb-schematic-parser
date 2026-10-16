"""Explicit local supervision decisions and masks for uncertain annotations."""
import numpy as np
from PIL import Image


def wire_supervision_decision(edge):
    review = edge.get("wire_supervision")
    if review is None:
        return "include", None
    if not isinstance(review, dict):
        raise ValueError("wire_supervision must be a review object")
    decision = review.get("decision")
    classification = review.get("classification")
    if decision not in {"exclude", "ignore"}:
        raise ValueError("Unsupported explicit wire supervision decision")
    if classification not in {"symbol_outline", "embedded_image", "unresolved_association"}:
        raise ValueError("Unsupported wire supervision classification")
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        raise ValueError("An explicit supervision exclusion requires a reason")
    if (decision == "ignore") != (classification == "unresolved_association"):
        raise ValueError("Unresolved associations require an ignore decision")
    return decision, classification


def load_valid_mask(row):
    path = row.get("valid_mask")
    if not path:
        return None
    with Image.open(path) as image:
        if image.size != (row["width"], row["height"]):
            raise ValueError("Supervision mask and image dimensions differ")
        values = np.asarray(image.convert("L")).copy()
    if not np.isin(values, [0, 255]).all():
        raise ValueError("Supervision mask must contain only 0 and 255")
    return values.astype(np.float32) / 255
