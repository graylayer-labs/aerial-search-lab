"""Overlapping tiles of a source frame, and the boxes that go with them (#96).

People are a few to a few tens of pixels wide in the source frames, so shrinking
a frame to a detector's input size erases them. Instead the frame is cut into
tiles of the detector's input size at native resolution. This module is pure
geometry: where the tiles are, which boxes a tile holds, how a tile's boxes go
back to source pixels, and how duplicates from the overlaps are merged.

A tile is `(x0, y0, x1, y1)` in source pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Tile = tuple[int, int, int, int]


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # the last tile sits flush with the edge
    return starts


def tile_grid(
    width: int, height: int, size: tuple[int, int], overlap: int
) -> list[Tile]:
    """Tiles of `size` covering the whole frame, row by row.

    Neighbouring tiles overlap by at least `overlap` pixels; the last column and
    row are moved back to sit flush with the right and bottom edge (so they
    overlap more) rather than hanging over it. A frame smaller than a tile in
    one direction gets one tile as wide as the frame in that direction.
    """
    tile_w, tile_h = size
    if not 0 <= overlap < min(tile_w, tile_h):
        raise ValueError(f"overlap {overlap} must be in [0, {min(tile_w, tile_h)})")
    xs = _starts(width, tile_w, tile_w - overlap)
    ys = _starts(height, tile_h, tile_h - overlap)
    return [
        (x, y, min(x + tile_w, width), min(y + tile_h, height)) for y in ys for x in xs
    ]


def boxes_in_tile(
    boxes: np.ndarray, tile: Tile, min_visible: float
) -> tuple[np.ndarray, np.ndarray]:
    """The boxes a tile shows, clipped to it, in tile pixels.

    A box is kept when at least `min_visible` of its area lies inside the tile
    (1.0: only boxes wholly inside). Returns the clipped, tile-relative boxes
    and the indices of the source boxes they came from.
    """
    x0, y0, x1, y1 = tile
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(x0, x1)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(y0, y1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    seen = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(area > 0, seen / area, 0.0)
    keep = (fraction >= min_visible) & (seen > 0)
    return clipped[keep] - np.array([x0, y0, x0, y0]), np.flatnonzero(keep)


def to_source(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    """Tile-relative boxes back to source pixels."""
    x0, y0 = tile[0], tile[1]
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) + np.array(
        [x0, y0, x0, y0]
    )


def merge_detections(
    boxes: np.ndarray, scores: np.ndarray, iou: float
) -> tuple[np.ndarray, np.ndarray]:
    """Class-agnostic non-maximum suppression over a stitched frame.

    The same person seen by two overlapping tiles gives two boxes; the lower
    scoring one is dropped when its IoU with a kept box exceeds `iou`. Returns
    the kept boxes and scores, highest score first.
    """
    import torch
    from torchvision.ops import nms

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    return boxes[keep], scores[keep]


def touches_box(boxes: np.ndarray, tile: Tile) -> bool:
    """True if any box overlaps the tile at all (sharing only an edge is not)."""
    x0, y0, x1, y1 = tile
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return bool(
        ((b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0)).any()
    )


def pick_tiles(
    has_person: Sequence[bool],
    empty_ratio: float,
    seed: int,
    touches_person: Sequence[bool] | None = None,
) -> list[int]:
    """Indices of the tiles to train on: every tile with a person, plus empty
    tiles chosen at random, `empty_ratio` of the person tiles (at least one
    tile's worth if none have a person), in index order.

    A tile that touches a person (`touches_person`) without labelling them,
    because too little of the box shows, is not a true negative and is never
    picked as an empty one."""
    if touches_person is None:
        touches_person = has_person
    positive = [i for i, p in enumerate(has_person) if p]
    empty = [i for i, p in enumerate(has_person) if not p and not touches_person[i]]
    wanted = math.ceil(empty_ratio * max(len(positive), 1))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(empty, size=min(wanted, len(empty)), replace=False)
    return sorted(positive + [int(i) for i in chosen])
