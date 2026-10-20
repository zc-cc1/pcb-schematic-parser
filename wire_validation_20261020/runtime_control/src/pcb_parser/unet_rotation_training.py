"""Training-only lossless quarter-turn augmentation of images and supervision."""
import numpy as np
import torch

from .unet_retraining import CachedTrainingTiles


def rotate_supervised_tile(image, targets, valid, quarter_turns):
    if not isinstance(quarter_turns, int) or isinstance(quarter_turns, bool) or quarter_turns not in range(4):
        raise ValueError("Quarter turns must be 0, 1, 2 or 3")
    if image.ndim != 3 or image.shape[0] != 3 or image.shape[-1] != image.shape[-2]:
        raise ValueError("Expected a square CHW RGB tile")
    if targets.shape != (2, *image.shape[-2:]) or valid.shape != image.shape[-2:]:
        raise ValueError("Image, two targets and valid mask must share geometry")
    if not quarter_turns:
        return image, targets, valid
    return tuple(torch.rot90(value, quarter_turns, (-2, -1)).contiguous()
                 for value in (image, targets, valid))


class RotationTrainingTiles(CachedTrainingTiles):
    def __init__(self, rows, tile_size=512, stride=384, rotation=None, seed=20261014):
        super().__init__(rows, tile_size, stride, augmentation=None, seed=seed)
        rotation = rotation or {"quarter_turns": [0, 1, 2, 3], "probabilities": [.25]*4}
        turns = rotation["quarter_turns"]
        probabilities = np.asarray(rotation["probabilities"], dtype=float)
        if (not turns or any(not isinstance(k, int) or isinstance(k, bool) or k not in range(4) for k in turns)
                or len(set(turns)) != len(turns) or probabilities.shape != (len(turns),)
                or not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("Invalid rotation distribution")
        self.turns = turns
        self.probabilities = probabilities
        self.rotation_rng = np.random.default_rng(seed)
        self.rotation_counts = np.zeros(4, dtype=np.int64)

    def __getitem__(self, index):
        values = super().__getitem__(index)
        k = int(self.rotation_rng.choice(self.turns, p=self.probabilities))
        self.rotation_counts[k] += 1
        return rotate_supervised_tile(*values, k)
