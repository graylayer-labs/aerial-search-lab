"""The test predictions file of a detector run, and re-scoring it (#112).

One JSON line per test frame: the frame id, its boxes in source pixels
(x1, y1, x2, y2) and their scores, after any tile merge and the per-frame cap.
Each line is flushed and synced as it is written, so a run stopped mid-test
loses at most the frame it was on. On resume the frames already present are
skipped; a last line cut short by the stop is dropped, and an unreadable line
anywhere else is refused. The run's score is computed from this file, so
`rescore` gives the same numbers by construction, without the model.

A file written beside a checkpoint starts with a header line,
`{"model_id": ...}`, naming the trained weights that made it (SHA-256 of the
weights). A resume uses the file only if that id equals its trained
checkpoint's: S3 sync is asynchronous, so after a stop the file can sit beside
another model's checkpoint, and two models' frames must never be mixed.

Floats are written with Python's shortest round-trip repr, so the arrays read
back are bit-identical to those written.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aerial_search.data.records import record_boxes, subsample
from aerial_search.evaluation.detection import (
    DetectionReport,
    GroundTruth,
    Prediction,
    evaluate_detections,
)

PREDICTIONS_FILENAME = "test_predictions.jsonl"

__all__ = [
    "PREDICTIONS_FILENAME",
    "Prediction",
    "PredictionLog",
    "read_model_id",
    "read_predictions",
    "rescore",
    "score_file",
    "truths",
]


def _encode(prediction: Prediction) -> str:
    boxes = np.asarray(prediction.boxes, dtype=np.float64)
    scores = np.asarray(prediction.scores, dtype=np.float64)
    line = {
        "image": prediction.image_id,
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
    }
    return json.dumps(line, allow_nan=False) + "\n"


def _decode(text: str) -> Prediction:
    line = json.loads(text)
    return Prediction(
        str(line["image"]),
        np.asarray(line["boxes"], dtype=np.float64).reshape(-1, 4),
        np.asarray(line["scores"], dtype=np.float64),
    )


def _header(raw: bytes) -> str | None:
    """The model id if `raw` is the header line, else None."""
    try:
        line = json.loads(raw.decode())
    except ValueError:
        return None
    if isinstance(line, dict) and set(line) == {"model_id"}:
        return str(line["model_id"])
    return None


def _read(path: Path) -> tuple[dict[str, Prediction], int, str | None]:
    """The complete predictions in `path`, the byte length they occupy (with
    the header), and the header's model id (None if there is none)."""
    done: dict[str, Prediction] = {}
    if not path.exists():
        return done, 0, None
    data = path.read_bytes()
    lines = data.split(b"\n")
    # A file ending in "\n" splits into a final empty piece; anything else in
    # that last piece is a line the writer did not finish, and is dropped (a
    # line whose newline was lost too: that frame is predicted again).
    complete = lines[:-1]
    end = 0
    model_id = _header(complete[0]) if complete else None
    if model_id is not None:
        end = len(complete[0]) + 1
    for number, raw in enumerate(complete, start=1):
        if number == 1 and model_id is not None:
            continue
        try:
            prediction = _decode(raw.decode())
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"{path} line {number} is unreadable ({error}); only the last "
                "line can be cut short by a stop, so the file is corrupt"
            ) from None
        if prediction.image_id in done:
            raise ValueError(f"{path} line {number}: {prediction.image_id} repeated")
        done[prediction.image_id] = prediction
        end += len(raw) + 1
    return done, end, model_id


class PredictionLog:
    """Append-only predictions file that resumes where a stopped run left off."""

    def __init__(self, path: Path):
        self.path = path
        self.done, end, self.model_id = _read(path)
        if path.exists() and path.stat().st_size != end:
            with path.open("r+b") as handle:  # drop the unfinished last line
                handle.truncate(end)

    def bind(self, model_id: str) -> None:
        """Tie the file to the trained weights `model_id`: write the header if
        the file is empty, refuse if it holds another model's predictions."""
        if self.model_id == model_id:
            return
        if self.model_id is not None or self.done:
            raise ValueError(
                f"{self.path} holds predictions of model {self.model_id}, not "
                f"of {model_id}; set it aside before predicting"
            )
        with self.path.open("w") as handle:
            handle.write(json.dumps({"model_id": model_id}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.model_id = model_id

    def append(self, prediction: Prediction) -> None:
        if prediction.image_id in self.done:
            raise ValueError(f"{prediction.image_id} is already in {self.path}")
        with self.path.open("a") as handle:
            handle.write(_encode(prediction))
            handle.flush()
            os.fsync(handle.fileno())
        self.done[prediction.image_id] = prediction


def read_predictions(path: Path) -> list[Prediction]:
    """Every complete prediction in the file, in file order."""
    return list(_read(path)[0].values())


def read_model_id(path: Path) -> str | None:
    """The model id in the file's header, or None (no file, no header)."""
    if not path.exists():
        return None
    with path.open("rb") as handle:
        first = handle.readline()
    return _header(first[:-1]) if first.endswith(b"\n") else None


def truths(records: Sequence[dict]) -> list[GroundTruth]:
    """The scorer's ground truth for fold records (camera and flight grouping)."""
    return [
        GroundTruth(
            str(r["image"]), record_boxes(r), str(r["camera"]), str(r["collection_id"])
        )
        for r in records
    ]


def score_file(path: Path, records: Sequence[dict]) -> DetectionReport:
    """Score the predictions file against `records`; every frame must be in it."""
    found = {p.image_id: p for p in read_predictions(path)}
    expected = [str(r["image"]) for r in records]
    missing = [i for i in expected if i not in found]
    extra = sorted(set(found) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{path} does not hold exactly the test frames: {len(missing)} missing "
            f"(first {missing[:3]}), {len(extra)} not in the test split "
            f"(first {extra[:3]})"
        )
    return evaluate_detections(truths(records), [found[i] for i in expected])


def _test_records(run_dir: Path, record: dict) -> list[dict]:
    """The run's test frames, from the manifest it recorded, checked by hash."""
    import hashlib

    tests = [i for i in record["inputs"] if Path(i["path"]).name == "test.jsonl"]
    if len(tests) != 1:
        raise ValueError(f"{run_dir}/run.json records no single test.jsonl input")
    path = Path(tests[0]["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != tests[0]["sha256"]:
        raise ValueError(f"{path} changed since the run; it cannot be re-scored")
    records = [json.loads(line) for line in data.decode().splitlines() if line]
    return subsample(records, int(record["config"].get("limit_frames") or 0))


def rescore(run_dir: Path) -> DetectionReport:
    """Score a run's predictions file with the current scorer, without the model."""
    record = json.loads((run_dir / "run.json").read_text())
    return score_file(run_dir / PREDICTIONS_FILENAME, _test_records(run_dir, record))
