"""The baseline's data: whole frames, tiles, and the batches of each step (#67, #96)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchvision.transforms.v2.functional as vision
from aerial_search import runner
from aerial_search.data import tiling
from aerial_search.data.records import record_boxes
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

if TYPE_CHECKING:
    from finetuned_detector.experiment import TileSpec


class Frames(Dataset):
    """Frames of one split: the image stretched to the input size, its boxes
    scaled with it (one label, "person"), and the scale back to source pixels."""

    def __init__(
        self, records: Sequence[dict], data_root: Path, size: tuple[int, int] | None
    ):
        self.records = list(records)
        self.data_root = data_root
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = decode_image(
            str(self.data_root / record["image"]), mode=ImageReadMode.RGB
        )
        height, width = image.shape[-2:]
        if [width, height] != list(record["size"]):
            raise ValueError(
                f"{record['image']} is {width}x{height} on disk, "
                f"{record['size']} in the manifest"
            )
        sx = sy = 1.0
        if self.size is not None:
            sx, sy = self.size[0] / width, self.size[1] / height
            image = vision.resize(image, [self.size[1], self.size[0]], antialias=True)
        boxes = torch.as_tensor(
            record_boxes(record) * [sx, sy, sx, sy], dtype=torch.float32
        )
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target = {
            "boxes": boxes[keep],
            "labels": torch.ones(int(keep.sum()), dtype=torch.int64),
        }
        return image.float() / 255, target, torch.tensor([sx, sy])


def _decode(path: Path, record: dict) -> torch.Tensor:
    image = decode_image(str(path), mode=ImageReadMode.RGB)
    height, width = image.shape[-2:]
    if [width, height] != list(record["size"]):
        raise ValueError(
            f"{record['image']} is {width}x{height} on disk, "
            f"{record['size']} in the manifest"
        )
    return image


class TiledTrainSet(Dataset):
    """Training tiles of native-resolution frames: every tile showing a person
    plus a random share of the empty ones, each with the boxes it shows
    (clipped to the tile, kept if `min_visible` of the box is inside)."""

    def __init__(
        self, records: Sequence[dict], data_root: Path, spec: TileSpec, seed: int
    ):
        self.records, self.data_root, self.spec = list(records), data_root, spec
        every: list[tuple[int, tiling.Tile, np.ndarray]] = []
        touches: list[bool] = []
        self.n_boxes = self.n_unlabelled_boxes = 0
        for r, record in enumerate(self.records):
            width, height = record["size"]
            truth = record_boxes(record)
            labelled: set[int] = set()
            for tile in tiling.tile_grid(width, height, spec.size, spec.overlap):
                local, index = tiling.boxes_in_tile(truth, tile, spec.min_visible)
                every.append((r, tile, local))
                touches.append(tiling.touches_box(truth, tile))
                labelled.update(int(i) for i in index)
            self.n_boxes += len(truth)
            self.n_unlabelled_boxes += len(truth) - len(labelled)
        picked = tiling.pick_tiles(
            [len(local) > 0 for _, _, local in every],
            spec.empty_ratio,
            seed,
            touches,
        )
        self.tiles = [every[i] for i in picked]
        self.n_candidates = len(every)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int):
        r, (x0, y0, x1, y1), local = self.tiles[index]
        record = self.records[r]
        image = _decode(self.data_root / record["image"], record)[:, y0:y1, x0:x1]
        boxes = torch.as_tensor(local, dtype=torch.float32)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target = {
            "boxes": boxes[keep],
            "labels": torch.ones(int(keep.sum()), dtype=torch.int64),
        }
        return image.float() / 255, target, torch.ones(2)


class TiledFrames(Dataset):
    """Whole frames for scoring: each item is every tile of one frame and the
    tiles' places in it. `predict` stitches the detections back together."""

    def __init__(self, records: Sequence[dict], data_root: Path, spec: TileSpec):
        self.records, self.data_root, self.spec = list(records), data_root, spec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = _decode(self.data_root / record["image"], record)
        width, height = record["size"]
        grid = tiling.tile_grid(width, height, self.spec.size, self.spec.overlap)
        tiles = [image[:, y0:y1, x0:x1].float() / 255 for x0, y0, x1, y1 in grid]
        return tiles, grid


def _collate(batch):
    return tuple(zip(*batch, strict=True))


class BatchSteps:
    """The head's batching, for `steps` steps: consecutive slices of a running
    order of frames, a fresh shuffle appended whenever fewer than a batch
    remain, so a split smaller than a batch gives batches of the whole split."""

    def __init__(self, n: int, batch_size: int, steps: int, seed: int, start: int = 0):
        self.n, self.batch_size, self.steps, self.seed = n, batch_size, steps, seed
        self.start = start  # steps already taken: a resumed run skips them

    def __len__(self) -> int:
        return self.steps - self.start

    def __iter__(self):
        return runner.batch_order(
            self.n, self.batch_size, self.steps, self.seed, self.start
        )
