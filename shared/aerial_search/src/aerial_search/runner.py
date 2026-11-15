"""What every experiment repeats, in one place (#216).

- `experiment_run`: `start_run`, then `finish_run` as completed or failed;
- `run_name`, `write_training` and `save_weights`: where and how a run's
  outputs are named and written;
- `score`: a split scored through `evaluation.detection`, with the scorer's
  camera and flight grouping;
- `batch_order` and `train_best_step`: the running order of frames and the
  loop that keeps the weights of the step that scores best on validation.

The projects bring their own model, loss and optimiser; this module never
imports a project.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerial_search import run as run_module
from aerial_search.evaluation.detection import (
    DetectionReport,
    Prediction,
    evaluate_detections,
)
from aerial_search.evaluation.predictions import truths


def run_name(prefix: str, camera: str, fold: str, percent: int, seed: int) -> str:
    """`<prefix>/<camera>-<fold>-<percent>pct-seed<seed>`."""
    return f"{prefix}/{camera}-{fold}-{percent}pct-seed{seed}"


@contextmanager
def experiment_run(
    name: str,
    config: dict[str, Any],
    seed: int,
    *,
    device: str,
    scratch: bool,
    argv: list[str] | None,
    repo: Path | None,
    inputs: Sequence[Path],
    fold: str,
    view: str,
    manifests: Path,
    data_root: Path,
) -> Generator[Path]:
    """`start_run` for one fold and view; yields the run directory.

    The run is marked `completed` when the block ends, or `failed`, with the
    error, when anything (an interrupt too) leaves it, and the error goes on.
    """
    directory = run_module.start_run(
        name,
        config,
        seed,
        device=device,
        scratch=scratch,
        argv=argv,
        repo=repo,
        inputs=inputs,
        fold=fold,
        view=view,
        manifests=manifests,
        data_root=data_root,
    )
    try:
        yield directory
    except BaseException as error:
        run_module.finish_run(directory, error)
        raise
    run_module.finish_run(directory)


def write_training(directory: Path, history: dict[str, Any]) -> None:
    """`training.json`: loss, validation scores, chosen step and the rest."""
    (directory / "training.json").write_text(
        json.dumps(history, indent=2, allow_nan=True) + "\n"
    )


def save_weights(directory: Path, name: str, module: torch.nn.Module) -> None:
    """`<name>.pt` in the run directory: the module's state dict."""
    torch.save(module.state_dict(), directory / f"{name}.pt")


def score(
    records: Sequence[dict],
    predictions: Sequence[Prediction],
    *,
    boxes: Sequence[np.ndarray] | None = None,
    camera: str | None = None,
) -> DetectionReport:
    """Score `predictions` on `records` through `evaluate_detections`.

    `boxes` (default: the records' own) and `camera` (default: each record's)
    are for a caller that already holds them."""
    return evaluate_detections(truths(records, boxes, camera), predictions)


def batch_order(
    n: int, batch_size: int, steps: int, seed: int, start: int = 0
) -> Iterator[list[int]]:
    """The frames of each step: consecutive slices of a running order.

    Whenever fewer than a batch remain, a fresh shuffle of all `n` frames is
    appended, so every frame is seen once per pass; a batch can hold the end
    of one pass and the start of the next (and then one frame twice), and
    `n` below `batch_size` gives batches of all `n`. Each batch is sorted.
    The first `start` steps are drawn but not yielded (a resumed run)."""
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for step in range(steps):
        if len(order) < batch_size:
            order += rng.permutation(n).tolist()
        if step >= start:
            yield sorted(order[:batch_size])
        del order[:batch_size]


def new_history(**extra: Any) -> dict[str, Any]:
    """The history `train_best_step` fills, with a caller's own fields after."""
    return {"loss": [], "validation": [], "best_step": None, **extra}


@dataclass
class Selection:
    """The history so far, the best validation score and that step's weights."""

    history: dict[str, Any] = field(default_factory=new_history)
    best: float = -math.inf
    state: Any = None


def train_best_step[B](
    batches: Iterable[B],
    step: Callable[[int, B], float],
    *,
    steps: int,
    eval_every: int,
    validate: Callable[[], float] | None,
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    selection: Selection | None = None,
    start: int = 0,
    after_step: Callable[[int, Selection], None] | None = None,
) -> Selection:
    """Run `step(number, batch)` for each batch, numbered from `start + 1`.

    `step` trains on one batch and returns its loss, which is appended to the
    history. Every `eval_every` steps, and at step `steps`, `validate` scores
    the model; a better score than any before keeps `snapshot()` (the earlier
    step wins a tie, and a NaN score never wins). `after_step` is called after
    each step and its validation. At the end the kept weights are put back
    with `restore`; without `validate`, or if no score ever won, the final
    weights stay. `selection` continues an earlier one (a resumed run).
    """
    selection = selection or Selection()
    history = selection.history
    for number, batch in enumerate(batches, start=start + 1):
        loss = step(number, batch)
        history["loss"].append(loss)
        if validate is not None and (number % eval_every == 0 or number == steps):
            value = validate()
            history["validation"].append({"step": number, "ap_iou25": value})
            print(
                f"step {number}: loss {loss:.4f}, val ap_iou25 {value:.4f}", flush=True
            )
            if value > selection.best:  # NaN compares false
                selection.best, selection.state = value, snapshot()
                history["best_step"] = number
        if after_step is not None:
            after_step(number, selection)
    if selection.state is not None:
        restore(selection.state)
    return selection
