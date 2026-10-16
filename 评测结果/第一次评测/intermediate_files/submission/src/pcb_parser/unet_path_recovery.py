"""Recover only contiguous, image-supported source traces; never bridge gaps."""
from .unet_data import point


def recover_contiguous_path(coordinates, width, height, rgb):
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError("At least two source coordinates are required")
    parsed = [point(value) for value in coordinates]
    if any(value is None for value in parsed):
        raise ValueError("Invalid source coordinate")
    if any(x != round(x) or y != round(y) or not (0 <= x < width and 0 <= y < height) for x, y in parsed):
        raise ValueError("Source coordinates must be integer image pixels")
    unique = set(parsed)
    if len(unique) < 2:
        raise ValueError("A single distinct pixel is not a path")
    if any(int(min(rgb[int(y), int(x)])) >= 224 for x, y in unique):
        raise ValueError("Every recovered source pixel must have original-image ink")
    xs = [value[0] for value in unique]
    ys = [value[1] for value in unique]
    if len(set(ys)) == 1 and len(unique) == int(max(xs) - min(xs)) + 1:
        return [[int(min(xs)), int(ys[0])], [int(max(xs)), int(ys[0])]]
    if len(set(xs)) == 1 and len(unique) == int(max(ys) - min(ys)) + 1:
        return [[int(xs[0]), int(min(ys))], [int(xs[0]), int(max(ys))]]
    result = []
    for x, y in parsed:
        value = [int(x), int(y)]
        if result and max(abs(value[0] - result[-1][0]), abs(value[1] - result[-1][1])) > 1:
            raise ValueError("Sparse endpoints cannot be joined automatically")
        if not result or value != result[-1]:
            result.append(value)
    # Preserve every turn and return segment, even when the trace loops back.
    return result
