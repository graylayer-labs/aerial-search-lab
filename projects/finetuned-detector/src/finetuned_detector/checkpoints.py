"""The baseline's checkpoints: identity, RNG state, weight ids and saving (#112)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aerial_search.evaluation.predictions import PREDICTIONS_FILENAME, read_model_id
from numpy.random import mtrand

# Checkpoints (#112). A training step takes about 3.4 s on the CPU and saving
# a checkpoint (weights, best weights, momentum: 228 MB MobileNet, 497 MB
# ResNet-50) 0.1 to 0.6 s, so every 50 steps a spot stop loses at most about
# 3 minutes of training, at under 0.5% overhead.
CHECKPOINT_EVERY = 50
CHECKPOINT_FILENAME = "checkpoint.pt"
ATTEMPTS_FILENAME = "attempts.json"
LOCK_FILENAME = "lock"


class CheckpointError(ValueError):
    """A checkpoint directory holds another run's state, or is inconsistent."""


def checkpoint_identity(
    config: dict[str, Any],
    seed: int,
    commit: str | None,
    *,
    scratch: bool,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """What must match for a checkpoint to continue a run: the run's config
    (recipe, tiling, camera, fold, fraction, frame counts), the seed, the
    code's commit, scratch or not, and the SHA-256 of its manifests."""
    identity = {
        "config": config,
        "seed": seed,
        "commit": commit,
        "scratch": scratch,
        "inputs": inputs or {},
    }
    # Through JSON so tuples and lists compare equal after a save and load.
    return json.loads(json.dumps(identity, sort_keys=True, default=str))


def _rng_state() -> dict[str, Any]:
    numpy = mtrand.get_bit_generator().state
    # No has_gauss/gauss here (the legacy normal-draw cache): resume is exact
    # because no code draws normals from the global numpy generator.
    assert isinstance(numpy, dict)
    # Plain lists and tensors only, so the checkpoint loads with weights_only.
    key = np.asarray(numpy["state"]["key"]).tolist()
    return {
        "python": random.getstate(),
        "numpy": {**numpy, "state": {**numpy["state"], "key": key}},
        "torch": torch.get_rng_state(),
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    key = np.asarray(numpy["state"]["key"], dtype=np.uint32)
    mtrand.get_bit_generator().state = {
        **numpy,
        "state": {**numpy["state"], "key": key},
    }
    torch.set_rng_state(state["torch"])


def weights_id(weights: dict[str, torch.Tensor]) -> str:
    """SHA-256 over the weights' names, dtypes, shapes and bytes: the id of a
    trained model, written in the header of the predictions it makes."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class Checkpoints:
    """One run's checkpoint in `directory`: `CHECKPOINT_FILENAME`, replaced
    atomically every `every` steps and once training ends, beside the test
    predictions file. `load` refuses a checkpoint of another identity.

    The trained checkpoint carries `model_id` (`weights_id` of its weights).
    The predictions file's header names the model that made it; `load` sets
    aside a file whose model is not the trained checkpoint's."""

    def __init__(self, directory: Path, identity: dict[str, Any], every: int):
        if every < 1:
            raise ValueError(f"checkpoint_every must be at least 1, not {every}")
        self.directory, self.identity, self.every = directory, identity, every
        self.state: dict[str, Any] | None = None
        self.model_id: str | None = None  # set once a trained state exists
        self._lock: int | None = None

    def lock(self) -> None:
        """Hold the directory for this process. A second live run on it would
        write test frames twice and interleave checkpoints, so it is refused;
        the lock is released when the process ends, however it ends."""
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = os.open(self.directory / LOCK_FILENAME, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(handle)
            raise CheckpointError(
                f"{self.directory} is in use by another live run"
            ) from None
        self._lock = handle

    def unlock(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)
            self._lock = None

    @property
    def path(self) -> Path:
        return self.directory / CHECKPOINT_FILENAME

    @property
    def predictions(self) -> Path:
        return self.directory / PREDICTIONS_FILENAME

    def load(self) -> dict[str, Any] | None:
        """The saved state, or None for a fresh start."""
        if not self.path.exists():
            if self.predictions.exists():
                self._set_predictions_aside(f"{self.path.name} does not")
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        if state.get("identity") != self.identity:
            theirs, mine = state.get("identity") or {}, self.identity
            differ = sorted(
                k for k in set(theirs) | set(mine) if theirs.get(k) != mine.get(k)
            )
            raise CheckpointError(
                f"the checkpoint in {self.directory} does not match this run "
                f"({', '.join(differ) or 'no identity'} differ); a run resumes "
                "only with the config, seed, commit and manifests it started "
                "with. Use another checkpoint directory."
            )
        if state.get("phase") != "trained":
            if self.predictions.exists():
                self._set_predictions_aside("the checkpoint is mid-training")
        else:
            # A checkpoint saved before model ids: the id of its weights.
            self.model_id = state.get("model_id") or weights_id(state["model"])
            if self.predictions.exists():
                found = read_model_id(self.predictions)
                if found != self.model_id:
                    self._set_predictions_aside(
                        f"it was made by another model ({found}), not the "
                        f"trained checkpoint's ({self.model_id})"
                    )
        self.state = state
        return state

    def _set_predictions_aside(self, why: str) -> None:
        """Move a predictions file that has no trained model behind it out of
        the way, so the run can resume instead of failing on every restart.

        S3 sync is asynchronous: after a stop it can hold the predictions
        file beside an older checkpoint (or none). The predictions are the
        trained model's, which resume rebuilds from the older state, so the
        file is dropped from use, kept as `*.stale-<time>` for inspection,
        and a warning says so.
        """
        stale = self.predictions.with_name(
            f"{self.predictions.stem}.stale-{time.time_ns()}.jsonl"
        )
        os.replace(self.predictions, stale)
        warnings.warn(
            f"{self.predictions} exists but {why}; set aside as {stale.name} and "
            "the test pass will start again",
            stacklevel=3,
        )

    def start_attempt(self, run_name: str) -> list[str]:
        """Record that `run_name` now works from this directory; return the
        runs that did before it (a small JSON file, so the large checkpoint
        is not rewritten to count resumes)."""
        path = self.directory / ATTEMPTS_FILENAME
        earlier = json.loads(path.read_text()) if path.exists() else []
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps([*earlier, run_name]) + "\n")
        os.replace(partial, path)
        return earlier

    def rewrite(self) -> None:
        """Write `checkpoint.pt` again, unchanged, as a new file.

        Called once the first test prediction is appended: S3 sync is
        asynchronous, and a small predictions line can reach S3 before the
        large trained checkpoint does. A fresh write queues the checkpoint
        for upload again, behind that line, so a stop later finds both.
        """
        if not self.path.exists():
            return
        partial = self.path.with_suffix(".partial")
        shutil.copyfile(self.path, partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, self.path)

    def save(self, **state: Any) -> None:
        if state.get("phase") == "trained":
            self.model_id = weights_id(state.get("model") or {})
            state["model_id"] = self.model_id
        state = {**state, "identity": self.identity, "rng": _rng_state()}
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".partial")
        torch.save(state, partial)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, self.path)  # a stop mid-save keeps the last one
