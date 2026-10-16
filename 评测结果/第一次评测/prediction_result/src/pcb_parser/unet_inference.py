"""Image-only U-Net inference and decoding of learned probability maps."""
import numpy as np
from PIL import Image
import torch

from .unet_data import predict_tiled


def predict_at_scale(model, rgb, device, tile_size, stride, scale=1.0):
    """Resize input for learned inference, then restore original coordinates."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be positive and finite")
    height, width = rgb.shape[:2]
    if scale == 1:
        return predict_tiled(model, rgb, device, tile_size, stride)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    enlarged = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC))
    probability = predict_tiled(model, enlarged, device, tile_size, stride)
    return np.stack([np.asarray(Image.fromarray(channel).resize((width, height), Image.Resampling.BILINEAR))
                     for channel in probability]).astype(np.float32)


def combine_probabilities(predictions, mode):
    """A fixed mean of scales, never a choice based on the current label."""
    modes = {
        "native": (1.0,), "zoom15": (1.5,), "zoom20": (2.0,),
        "mean_native_zoom15": (1.0, 1.5),
        "mean_native_zoom20": (1.0, 2.0),
        "mean_all": (1.0, 1.5, 2.0),
    }
    if mode not in modes:
        raise ValueError(f"Unknown probability fusion mode: {mode}")
    return np.mean([predictions[scale] for scale in modes[mode]], axis=0)


def scales_for_mode(mode):
    names = {"native": (1.0,), "zoom15": (1.5,), "zoom20": (2.0,),
             "mean_native_zoom15": (1.0, 1.5), "mean_native_zoom20": (1.0, 2.0),
             "mean_all": (1.0, 1.5, 2.0)}
    if mode not in names:
        raise ValueError(f"Unknown probability fusion mode: {mode}")
    return names[mode]


def peak_candidates(probability, threshold=0.3, kernel=9, refinement="none"):
    """Find peaks on the learned heatmap, with optional subpixel coordinates."""
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Expected a finite two-dimensional probability map")
    if kernel < 1 or kernel % 2 != 1 or not 0 <= threshold <= 1:
        raise ValueError("Peak kernel must be positive odd and threshold in [0,1]")
    if refinement not in {"none", "quadratic"}:
        raise ValueError("Unknown peak refinement")
    heat = torch.from_numpy(np.ascontiguousarray(probability))[None, None]
    peaks = (heat == torch.nn.functional.max_pool2d(heat, kernel, stride=1, padding=kernel // 2)) & (heat >= threshold)
    ys, xs = torch.where(peaks[0, 0])
    result = []
    height, width = probability.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        dx = dy = 0.0
        if refinement == "quadratic" and 0 < x < width - 1 and 0 < y < height - 1:
            # A local parabola in log probability approximates a Gaussian peak.
            patch = np.log(np.maximum(probability[y - 1:y + 2, x - 1:x + 2], 1e-8))
            for axis in [0, 1]:
                a, b, c = patch[:, 1] if axis == 0 else patch[1, :]
                denominator = float(a - 2 * b + c)
                offset = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5)) if denominator < -1e-8 else 0.0
                if axis == 0:
                    dy = offset
                else:
                    dx = offset
        result.append({"point": [float(x + dx), float(y + dy)], "confidence": float(probability[y, x])})
    return sorted(result, key=lambda item: -item["confidence"])


def select_candidates(candidates, threshold=0.5, nms_radius=0.0, limit=500):
    """Deduplicate nearby probability peaks without consulting source pixels."""
    if not 0 <= threshold <= 1 or nms_radius < 0 or limit <= 0:
        raise ValueError("Invalid candidate selection settings")
    kept = []
    # Spatial bins avoid comparing each peak to every accepted peak.
    bins = {}
    cell_size = max(nms_radius, 1.0)
    for candidate in candidates:
        if candidate["confidence"] < threshold:
            continue
        x, y = candidate["point"]
        cell = (int(x // cell_size), int(y // cell_size))
        if nms_radius and any((x - px) ** 2 + (y - py) ** 2 < nms_radius ** 2
                              for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                              for px, py in bins.get((cell[0] + dx, cell[1] + dy), [])):
            continue
        kept.append(candidate)
        bins.setdefault(cell, []).append((x, y))
        if len(kept) == limit:
            break
    return kept


def decode_pins(probability, threshold=0.5, kernel=9, refinement="none", nms_radius=0.0, limit=500):
    return select_candidates(peak_candidates(probability, threshold, kernel, refinement), threshold, nms_radius, limit)


def predict_configured(model, rgb, device, cfg, inference):
    """All routing is fixed in a config selected before evaluation."""
    wire_mode, pin_mode = inference.get("wire_mode", "native"), inference.get("pin_mode", "native")
    scales = sorted(set(scales_for_mode(wire_mode)) | set(scales_for_mode(pin_mode)))
    predictions = {scale: predict_at_scale(model, rgb, device, cfg["tile_size"], cfg["stride"], scale)
                   for scale in scales}
    return np.stack([combine_probabilities(predictions, wire_mode)[0],
                     combine_probabilities(predictions, pin_mode)[1]])
