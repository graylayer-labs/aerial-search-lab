"""Warm start: a run's head begins as a finished arm-A run's trained head (#68)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from frozen_head.feature_cache import CacheError
from frozen_head.thermal_input import DEFAULT_ARM

if TYPE_CHECKING:
    from frozen_head.head_experiment import Recipe


def load_warm_start(
    init_head: Path,
    *,
    fold: str,
    percent: int,
    camera: str,
    seed: int,
    recipe: Recipe,
    cache_settings: Mapping[str, Any],
    manifests: Sequence[Path],
    scratch: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The trained head of a finished arm-A run, and the record of where it is from.

    The run must be completed, cold (not itself warm-started), made with the
    replicate input, and share this run's fold, percent, camera, seed, head
    size and cache settings; anything else is refused, so a warm start can
    never carry another fold's labels or another seed's head into this run.
    A scratch run's head is accepted only by a scratch run. Its manifests
    (by the SHA-256 in its run.json inputs) must be this run's `manifests`.
    """
    run_json = init_head / "run.json"
    try:
        record = json.loads(run_json.read_text())
        state = torch.load(init_head / "head.pt", map_location="cpu")
    except FileNotFoundError as error:
        raise CacheError(f"{init_head} is not a finished run: {error}") from None
    if record.get("status") != "completed":
        raise CacheError(
            f"{init_head} has status {record.get('status')!r}, not completed"
        )
    if record.get("scratch") is not False and not scratch:
        raise CacheError(f"{init_head} is a scratch run; a normal run cannot use it")
    config = record.get("config", {})
    if config.get("warm_start") or config.get("input_handling", DEFAULT_ARM) != (
        DEFAULT_ARM
    ):
        raise CacheError(f"{init_head} is not a cold {DEFAULT_ARM!r} arm run")
    mine = {"camera": camera, "fold": fold, "percent": percent}
    for field, value in mine.items():
        if config.get(field) != value:
            raise CacheError(
                f"{init_head}: {field} is {config.get(field)!r}, this run has {value!r}"
            )
    if record.get("seed") != seed:
        raise CacheError(
            f"{init_head}: seed is {record.get('seed')!r}, this run has {seed!r}"
        )
    shape = config.get("recipe", {})
    for field in ("hidden", "upsample"):
        if shape.get(field) != getattr(recipe, field):
            raise CacheError(
                f"{init_head}: recipe.{field} is {shape.get(field)!r}, this run "
                f"has {getattr(recipe, field)!r}"
            )
    stored = config.get("cache", {}).get("settings", {})
    keep = {k: v for k, v in stored.items() if k != "input_handling"}
    here = {k: v for k, v in cache_settings.items() if k != "input_handling"}
    if keep != here:
        raise CacheError(f"{init_head} was trained on a different cache")
    made_with = {Path(i["path"]).name: i["sha256"] for i in record.get("inputs", [])}
    for path in manifests:
        theirs = made_with.get(path.name)
        mine = hashlib.sha256(path.read_bytes()).hexdigest()
        if theirs != mine:
            raise CacheError(
                f"{init_head}: its {path.name} (sha256 {str(theirs)[:12]}) is not "
                f"this run's ({mine[:12]}); the manifests differ"
            )
    sha = hashlib.sha256(run_json.read_bytes()).hexdigest()
    return state, {
        "init_head": str(init_head.resolve()),
        "init_run": str(record.get("run_name")),
        "run_json_sha256": sha,
    }
