"""Training-only photometric augmentation; production inference is unchanged."""
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .unet_data import tile_starts
from .unet_supervision import load_valid_mask


def augment_rgb(rgb, rng, config):
    """Vary contrast both ways without moving pixels or manufacturing labels."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB uint8 input")
    identity = config["identity_probability"]
    low, high = config["gamma_range"]
    amin, amax = config["ink_strength_range"]
    if not (0 <= identity <= 1 and 0 < low <= high and 0 < amin <= amax):
        raise ValueError("Invalid augmentation ranges")
    if rng.random() < identity:
        return rgb.copy()
    gamma = np.exp(rng.uniform(np.log(low), np.log(high)))
    strength = rng.uniform(amin, amax)
    normalized = rgb.astype(np.float32)/255
    changed = 1-strength*(1-normalized**gamma)
    return np.round(np.clip(changed, 0, 1)*255).astype(np.uint8)


class CachedTrainingTiles(Dataset):
    def __init__(self, rows, tile_size=512, stride=384, augmentation=None, seed=20261014):
        self.rows = rows
        self.size = tile_size
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)
        self.tiles = [(i, x, y) for i, row in enumerate(rows)
                      for y in tile_starts(row["height"], tile_size, stride)
                      for x in tile_starts(row["width"], tile_size, stride)]

    def __len__(self):
        return len(self.tiles)

    @lru_cache(maxsize=None)
    def source(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as im:
            image = np.asarray(im.convert("RGB")).copy()
        with Image.open(row["wire_mask"]) as im:
            wire = np.asarray(im.convert("L")).copy()
        with Image.open(row["pin_heatmap"]) as im:
            pin = np.asarray(im.convert("L")).copy()
        expected = (row["height"], row["width"])
        if image.shape[:2] != expected or wire.shape != expected or pin.shape != expected:
            raise ValueError("Image and labels differ in size")
        return image, wire, pin

    @lru_cache(maxsize=None)
    def supervision(self, index):
        return load_valid_mask(self.rows[index])

    def __getitem__(self, index):
        i, x, y = self.tiles[index]
        row = self.rows[i]; size = self.size
        width, height = min(size, row["width"]-x), min(size, row["height"]-y)
        rgb, wire, pin = self.source(i)
        image = np.full((size, size, 3), 255, np.uint8)
        image[:height, :width] = rgb[y:y+height, x:x+width]
        if self.augmentation:
            image = augment_rgb(image, self.rng, self.augmentation)
        target = np.zeros((2, size, size), np.float32)
        target[0, :height, :width] = wire[y:y+height, x:x+width].astype(np.float32)/255
        target[1, :height, :width] = pin[y:y+height, x:x+width].astype(np.float32)/255
        valid = np.zeros((size, size), np.float32)
        supervision = self.supervision(i)
        valid[:height, :width] = 1 if supervision is None else supervision[y:y+height,x:x+width]
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()).float()/255,
                torch.from_numpy(target), torch.from_numpy(valid))

    def clear_cache(self):
        self.source.cache_clear()
        self.supervision.cache_clear()
