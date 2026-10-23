"""The labelled frames a run sees, and their boxes (#66, #67).

The one place that decides which frames of a fold's manifests a run trains,
selects and is scored on, and refuses a manifest that would let the test
site-day leak into training or selection. Every approach reads its frames
through `select_records`, so all of them are scored on the same frames.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aerial_search.data.folds import TEST, VALIDATION, site_day, train_file, view_dir


class LeakError(Exception):
    """A manifest would let the test site-day reach training or selection."""


def record_boxes(record: Mapping[str, Any]) -> np.ndarray:
    """A fold record's boxes (normalised centre and size) in source pixels."""
    width, height = record["size"]
    out = [
        [
            (b["x_center"] - b["width"] / 2) * width,
            (b["y_center"] - b["height"] / 2) * height,
            (b["x_center"] + b["width"] / 2) * width,
            (b["y_center"] + b["height"] / 2) * height,
        ]
        for b in record["boxes"]
    ]
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def load_split(path: Path, *, fold: str, role: str) -> list[dict]:
    """The records of one camera-view manifest, checked against the fold.

    `role` "test" must hold only the fold's site-day; "train" and
    "validation" must hold none of it, judged by both the record's
    `site_day` and its image path. Every record must carry `boxes`: a record
    in a labelled manifest with an empty list is a frame labelled empty, and
    is a negative. A record without the field is refused, never read as empty.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    for record in records:
        if "boxes" not in record:
            raise ValueError(f"{path}: {record['image']} has no boxes field")
        days = {record["site_day"], site_day(record["image"])}
        if role == "test" and days != {fold}:
            raise LeakError(f"{path}: {record['image']} is not from {fold}")
        if role != "test" and fold in days:
            raise LeakError(
                f"{path}: {record['image']} is from the test site-day {fold}, in "
                f"the {role} split"
            )
    return records


def select_records(
    manifests: Path, fold: str, camera: str, percent: int
) -> tuple[dict[str, Path], dict[str, list[dict]]]:
    """The manifest files and records of one run's train, validation and test.

    The one place that decides which labelled frames a run sees at a given
    fold, camera and fraction. The supervised baseline (#67) calls it too, so
    it and the head are trained, selected and scored on the same frames.
    """
    view = view_dir(manifests / "folds", fold, camera)
    paths = {
        "train": view / train_file(percent),
        "validation": view / VALIDATION,
        "test": view / TEST,
    }
    records = {role: load_split(p, fold=fold, role=role) for role, p in paths.items()}
    return paths, records


def subsample(records: Sequence[dict], limit: int) -> list[dict]:
    """`limit` records spread evenly through the list (all if limit is 0)."""
    if limit <= 0 or len(records) <= limit:
        return list(records)
    picks = np.linspace(0, len(records) - 1, limit).round().astype(int)
    return [records[i] for i in picks]
