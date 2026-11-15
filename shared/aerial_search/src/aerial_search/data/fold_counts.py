"""Counts for the fold table: site-days, clips, frames, boxes, size buckets."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from aerial_search.evaluation.detection import SIZE_BUCKETS

if TYPE_CHECKING:
    from aerial_search.data.folds import Record


def _describe(
    records: list[Record], view: str, *, buckets: bool = False
) -> dict[str, object]:
    """Counts for the fold table: site-days, clips, frames, boxes."""
    described: dict[str, object] = {
        "site_days": sorted({str(r["site_day"]) for r in records}),
        "clips": len({r["collection_id"] for r in records}),
        "frames": len(records),
    }
    cameras = ["rgb", "thermal"] if view == "paired" else [view]
    if view == "unlabelled":
        return described
    for camera in cameras:
        prefix = f"{camera}_" if view == "paired" else ""
        boxes = [
            (box, _size(record, f"{prefix}size"))
            for record in records
            for box in _boxes(record, f"{prefix}boxes")
        ]
        described[f"{camera}_boxes"] = len(boxes)
        if buckets:
            described[f"{camera}_size_buckets"] = _bucket_counts(boxes)
    return described


def _bucket_counts(boxes: Iterable[tuple[dict, list[int]]]) -> dict[str, int]:
    counts = dict.fromkeys(SIZE_BUCKETS, 0)
    for box, size in boxes:
        width, height = size
        side = math.sqrt(box["width"] * width * box["height"] * height)
        for name, (low, high) in SIZE_BUCKETS.items():
            if low <= side < high:
                counts[name] += 1
    return counts


def _boxes(record: Record, key: str) -> list[dict]:
    boxes = record[key]
    assert isinstance(boxes, list)
    return boxes


def _size(record: Record, key: str) -> list[int]:
    size = record[key]
    assert isinstance(size, list)
    return size
