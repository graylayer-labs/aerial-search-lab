"""The baseline's checkpoint identity, RNG state and weight ids (#112)."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
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
