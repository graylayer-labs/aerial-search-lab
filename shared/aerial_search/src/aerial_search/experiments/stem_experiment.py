"""Arm C of the thermal input ablation: a learned stem before the frozen backbone (#68).

The stem sits in front of the frozen SigLIP 2 tower, so no feature cache can
serve it: every training step runs the tower on the stem's output and sends
gradients back through it. Everything else follows `head_experiment.run_head`:
the same manifests, head, loss, targets, decoding, step selection on the
validation split, and test scoring through `aerial_search.evaluation.detection`.

Differences that cannot be avoided, all recorded in `run.json`:

- the batch is built from `micro_batch` frames at a time and the gradients are
  summed, because a batch of 32 frames with gradients does not fit in memory;
- the frozen tower runs in half precision, so the loss is scaled by
  `loss_scale` before the backward pass and the gradients are unscaled before
  the step; a step whose gradients are not finite is skipped and counted;
- the geometry of each frame (its feature grid) is read from the replicate
  cache's index, and each batch is checked against it. No feature file is read.

Cost, measured: see docs/thermal-input-ablation.md.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from aerial_search.data.folds import TEST, VALIDATION, train_file, view_dir
from aerial_search.evaluation.detection import (
    GroundTruth,
    Prediction,
    evaluate_detections,
)
from aerial_search.experiments.feature_cache import CacheError
from aerial_search.experiments.head_experiment import (
    Geometry,
    Recipe,
    Split,
    build_head,
    cache_arm,
    check_cache,
    decode,
    load_split,
    load_warm_start,
    record_boxes,
)
from aerial_search.models.backbone import NaflexExtractor, Prepared
from aerial_search.models.head import CentreHead, centre_loss
from aerial_search.models.thermal_input import DEFAULT_ARM, StemBackbone, ThermalStem

ARM = "stem"
WORKERS = 3


@dataclass(frozen=True)
class StemRecipe(Recipe):
    """The head's recipe plus what the stem arm needs."""

    micro_batch: int = 4  # frames per forward/backward; gradients are summed
    eval_batch: int = 16
    stem_hidden: int = 16
    loss_scale: float = 256.0  # for the half-precision backward pass
    # score every Nth validation frame when choosing the step; 1 is all of them.
    # The test split is always scored whole.
    validation_stride: int = 1


def collate(prepared: Sequence[Prepared]) -> dict[str, Tensor]:
    """One batch from single-frame processor outputs."""
    return {
        key: torch.cat([p.inputs[key] for p in prepared])
        for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    }


class FrameSource:
    """Decodes and patchifies frames in worker threads, one batch ahead."""

    def __init__(
        self, extractor: NaflexExtractor, data_root: Path, records: Sequence[dict]
    ) -> None:
        self.extractor, self.data_root, self.records = extractor, data_root, records

    def _prepare(self, i: int) -> Prepared:
        with Image.open(self.data_root / str(self.records[i]["image"])) as image:
            return self.extractor.prepare(image.copy())

    def batches(
        self, indices: Sequence[Sequence[int]]
    ) -> Iterator[tuple[Sequence[int], dict[str, Tensor]]]:
        with ThreadPoolExecutor(WORKERS) as pool:

            def submit(group: Sequence[int]):
                return group, [pool.submit(self._prepare, i) for i in group]

            pending: deque = deque()
            groups = iter(indices)
            for _ in range(2):
                group = next(groups, None)
                if group is not None:
                    pending.append(submit(group))
            while pending:
                group, jobs = pending.popleft()
                nxt = next(groups, None)
                if nxt is not None:
                    pending.append(submit(nxt))
                yield group, collate([job.result() for job in jobs])


@dataclass(eq=False)
class Frames:
    """One split with the cache index entry of each frame and a pixel source."""

    split: Split
    entries: list[dict]
    source: FrameSource


def _check_grid(inputs: dict[str, Tensor], entries: list[dict]) -> None:
    for shape, entry in zip(inputs["spatial_shapes"], entries, strict=True):
        if [int(v) for v in shape] != list(entry["patch_grid"]):
            raise CacheError(
                f"{entry['path']}: the processor gives patch grid "
                f"{[int(v) for v in shape]}, the cache index says "
                f"{entry['patch_grid']}; the geometry would not match"
            )


@dataclass(eq=False)
class StemModel:
    """Stem + frozen tower + head, and what is needed to run it on a split."""

    backbone: StemBackbone
    head: CentreHead
    device: torch.device

    def parameters(self):
        return [*self.backbone.stem.parameters(), *self.head.parameters()]

    def train(self, mode: bool) -> None:
        self.backbone.stem.train(mode)
        self.head.train(mode)

    def forward(self, inputs: dict[str, Tensor]):
        return self.head(self.backbone(inputs))

    def state(self) -> dict[str, dict]:
        return {
            "stem": copy.deepcopy(self.backbone.stem.state_dict()),
            "head": copy.deepcopy(self.head.state_dict()),
        }

    def load(self, state: dict[str, dict]) -> None:
        self.backbone.stem.load_state_dict(state["stem"])
        self.head.load_state_dict(state["head"])


@torch.no_grad()
def predict(
    model: StemModel,
    data: Frames,
    indices: Sequence[int],
    batch: int,
    top_k: int,
) -> list[Prediction]:
    model.train(False)
    split = data.split
    groups = [indices[s : s + batch] for s in range(0, len(indices), batch)]
    out = []
    for group, inputs in data.source.batches(groups):
        _check_grid(inputs, [data.entries[i] for i in group])
        heat, offset, size = (t.cpu().numpy() for t in model.forward(inputs))
        for k, n in enumerate(group):
            boxes, scores = decode(
                heat[k], offset[k], size[k], split.geometry[n], top_k
            )
            out.append(Prediction(str(split.records[n]["image"]), boxes, scores))
    return out


def evaluate(model: StemModel, data: Frames, recipe: StemRecipe, *, stride: int = 1):
    split = data.split
    indices = list(range(0, len(split), stride))
    truths = [
        GroundTruth(
            str(split.records[i]["image"]),
            split.boxes[i],
            split.camera,
            str(split.records[i]["collection_id"]),
        )
        for i in indices
    ]
    predictions = predict(model, data, indices, recipe.eval_batch, recipe.top_k)
    return evaluate_detections(truths, predictions)


def accumulate(
    forward: Callable[[dict[str, Tensor]], tuple[Tensor, Tensor, Tensor]],
    batches: Iterator[tuple[Sequence[int], dict[str, Tensor]]],
    targets: dict[str, np.ndarray],
    index: np.ndarray,
    *,
    loss_scale: float,
    device: torch.device,
    check: Callable[[Sequence[int], dict[str, Tensor]], None] | None = None,
) -> float:
    """Backward through each micro-batch, summing to one full-batch gradient.

    `centre_loss` divides by the people in the batch it is given, so a
    micro-batch's loss is weighted by `max(c_k, 1) / max(c_total, 1)`, c being
    the count of people. Then the sum of the weighted losses, and of their
    gradients, equals the loss of one batch of all `index` frames exactly,
    micro-batches without people included. (Weighting by frames would not.)
    Gradients are scaled by `loss_scale` for the half-precision tower; the
    caller unscales them. Returns the full-batch loss, unscaled.
    """
    heat = targets["heat"]
    total_people = max(int((heat[index] == 1).sum()), 1)
    total = 0.0
    for group, inputs in batches:
        if check is not None:
            check(group, inputs)
        batch = {k: torch.from_numpy(v[group]).to(device) for k, v in targets.items()}
        people = max(int((heat[group] == 1).sum()), 1)
        weight = people / total_people
        loss = centre_loss(forward(inputs), batch)
        (loss * weight * loss_scale).backward()
        total += float(loss.item()) * weight
    return total


def train(
    model: StemModel,
    data: Frames,
    recipe: StemRecipe,
    *,
    validate: Callable[[], float] | None,
    seed: int,
) -> dict[str, Any]:
    """AdamW with cosine decay over stem and head together, as `train_head`.

    Each step takes `batch_size` frames (the same running order of reshuffled
    passes as `train_head`), runs them in groups of `micro_batch`, and sums
    the gradients weighted by group size.
    """
    split = data.split
    params = model.parameters()
    optimiser = torch.optim.AdamW(
        params, lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, recipe.steps)
    rng = np.random.default_rng(seed)
    order: list[int] = []
    history: dict[str, Any] = {
        "loss": [],
        "validation": [],
        "best_step": None,
        "skipped_steps": 0,
        "step_seconds": [],
        "validation_seconds": 0.0,
    }
    best, best_state = -math.inf, None
    for step in range(1, recipe.steps + 1):
        started = time.perf_counter()
        model.train(True)
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(split)).tolist()
        index = np.sort(np.array(order[: recipe.batch_size]))
        del order[: recipe.batch_size]
        groups = [
            index[s : s + recipe.micro_batch].tolist()
            for s in range(0, len(index), recipe.micro_batch)
        ]
        optimiser.zero_grad()
        total = accumulate(
            model.forward,
            data.source.batches(groups),
            split.targets,
            index,
            loss_scale=recipe.loss_scale,
            device=model.device,
            check=lambda group, inputs: _check_grid(
                inputs, [data.entries[i] for i in group]
            ),
        )
        finite = True
        for p in params:
            if p.grad is not None:
                p.grad /= recipe.loss_scale
                finite = finite and bool(torch.isfinite(p.grad).all())
        if finite:
            optimiser.step()
        else:
            history["skipped_steps"] += 1
        schedule.step()
        history["loss"].append(total)
        history["step_seconds"].append(time.perf_counter() - started)
        if validate is not None and (
            step % recipe.eval_every == 0 or step == recipe.steps
        ):
            began = time.perf_counter()
            score = validate()
            history["validation_seconds"] += time.perf_counter() - began
            history["validation"].append({"step": step, "ap_iou25": score})
            print(f"step {step}: loss {total:.4f}, val ap_iou25 {score:.4f}")
            if score > best:  # NaN compares false
                best, best_state = score, model.state()
                history["best_step"] = step
    if best_state is not None:
        model.load(best_state)
    return history


def default_run_name(
    cache_dir: Path,
    camera: str,
    fold: str,
    percent: int,
    seed: int,
    warm: bool = False,
):
    suffix = "-warm" if warm else ""
    return (
        f"detection-head/{cache_dir.name}-{ARM}{suffix}/"
        f"{camera}-{fold}-{percent}pct-seed{seed}"
    )


def _load_backbone(model_name: str, device: torch.device):
    from transformers import AutoImageProcessor

    from aerial_search.models import backbone

    spec = backbone.SPECS[model_name]
    processor = AutoImageProcessor.from_pretrained(spec.repo, revision=spec.revision)
    return backbone.load(spec, str(device)), processor


def run_stem_head(
    *,
    cache_dir: Path,
    manifests: Path,
    data_root: Path,
    fold: str,
    camera: str,
    percent: int,
    seed: int,
    recipe: StemRecipe,
    scratch: bool,
    run_name: str | None,
    device: torch.device,
    repo: Path | None = None,
    argv: list[str] | None = None,
    load_backbone: Callable[[str, torch.device], tuple[Any, Any]] | None = None,
    weights: Callable[[str, str], dict[str, str]] | None = None,
    init_head: Path | None = None,
) -> dict[str, Any]:
    """One run of arm C: train stem and head, select on validation, score on test.

    `init_head` (a finished cold arm-A run directory with the same fold,
    percent, camera and seed) starts the head from its trained weights; the
    stem still starts as the identity, so step 0 is exactly that arm-A head.
    This is the warm start: it is recorded in `config.warm_start`, and its
    matched control is `run_head(init_head=...)` for the same steps.

    `cache_dir` is the replicate cache; only its index (the geometry of each
    frame) and settings (model, weights, token budget, pooling) are read, and
    the tower is loaded and checked against those weights. Thermal only.
    """
    from aerial_search import run as run_module
    from aerial_search.data import checksums
    from aerial_search.models import backbone

    if camera != "thermal":
        raise ValueError("the learned input stem is for the thermal camera only")
    view = view_dir(manifests / "folds", fold, camera)
    paths = {
        "train": view / train_file(percent),
        "validation": view / VALIDATION,
        "test": view / TEST,
    }
    records = {role: load_split(p, fold=fold, role=role) for role, p in paths.items()}
    pinned = None if scratch else {e.path: e.sha256 for e in checksums.committed_list()}
    cache = check_cache(
        cache_dir,
        [r for rs in records.values() for r in rs],
        camera,
        scratch=scratch,
        pinned=pinned,
    )
    if cache_arm(cache.settings) != DEFAULT_ARM:
        raise CacheError(
            f"{cache_dir} holds {cache_arm(cache.settings)!r} features; the stem "
            f"arm takes its geometry from the {DEFAULT_ARM!r} cache"
        )
    if cache.missing:  # a scratch run only
        print(f"scratch: {len(cache.missing)} images not in the cache, left out")
        absent = set(cache.missing)
        records = {
            role: [r for r in rs if r["image"] not in absent]
            for role, rs in records.items()
        }
    stored = cache.settings["settings"]
    spec = backbone.SPECS[stored["model"]]
    if (spec.repo, spec.revision) != (stored["repo"], stored["revision"]):
        raise CacheError(
            f"the cache was made with {stored['repo']}@{stored['revision']}, this "
            f"code loads {spec.repo}@{spec.revision}"
        )
    found = (weights or backbone.weight_checksums)(spec.repo, spec.revision)
    if found != stored["weights_sha256"]:
        raise CacheError("the weights on disk are not those the cache was made with")
    warm, warm_record = None, {}
    if init_head is not None:
        warm, warm_record = load_warm_start(
            init_head,
            fold=fold,
            percent=percent,
            camera=camera,
            seed=seed,
            recipe=recipe,
            cache_settings=stored,
            manifests=list(paths.values()),
            scratch=scratch,
        )
    dims = {int(e["dim"]) for e in cache.entries.values()}
    if len(dims) > 1:
        raise CacheError(f"{cache_dir} holds several feature sizes: {sorted(dims)}")
    config = {
        "camera": camera,
        "fold": fold,
        "percent": percent,
        "input_handling": ARM,
        **({"warm_start": warm_record} if warm_record else {}),
        "recipe": asdict(recipe),
        "cache_dir": str(cache_dir),
        "cache": cache.settings,
        "cache_sessions": cache.sessions,
        "images": {role: len(rs) for role, rs in records.items()},
        "images_missing_from_cache": cache.missing,
        "head": "CentreHead as in run_head, after a residual two-layer 3x3 conv "
        f"stem ({recipe.stem_hidden} hidden, zero-initialised output) on the "
        "normalised image, then the frozen tower and 2x2 pooling; stem and head "
        "train together; no feature file is read",
    }
    directory = run_module.start_run(
        run_name
        or default_run_name(cache_dir, camera, fold, percent, seed, warm is not None),
        config,
        seed,
        device=str(device),
        scratch=scratch,
        argv=argv,
        repo=repo,
        inputs=[
            *paths.values(),
            cache_dir / "cache.json",
            cache_dir / "index.jsonl",
            *([init_head / "head.pt"] if init_head else []),
        ],
        fold=fold,
        view=camera,
        manifests=manifests,
        data_root=data_root,
    )
    try:
        # the head first, straight after the seed, exactly as run_head does, so
        # at one seed it starts from the same weights as arm A; the stem and
        # the tower draw from their own state afterwards
        head = build_head(seed, dims.pop() if dims else 0, recipe)
        if warm is not None:
            head.load_state_dict(warm)
        head.to(device)
        tower, processor = (load_backbone or _load_backbone)(stored["model"], device)
        if int(tower.config.hidden_size) != head.norm.normalized_shape[0]:
            raise CacheError("the tower's feature size is not the cache's")
        for p in tower.parameters():
            p.requires_grad_(False)
        tower.eval()
        extractor = NaflexExtractor(
            tower, processor, stored["token_budget"], stored["pooling"]
        )
        with torch.random.fork_rng(devices=[]):  # the stem's own generator
            torch.manual_seed(seed + 1)
            stem = ThermalStem(recipe.stem_hidden).to(device)
        stack = StemBackbone(tower, stem, int(processor.patch_size), stored["pooling"])
        model = StemModel(stack, head, device)
        data: dict[str, Frames] = {}
        for role, rs in records.items():
            entries = [cache.entries[str(r["image"])] for r in rs]
            geometry = [Geometry.from_entry(e, recipe.upsample) for e in entries]
            split = Split.build(
                rs,
                np.zeros((len(rs), 0), dtype="f2"),  # no features: pixels are read
                geometry,
                [record_boxes(r) for r in rs],
                camera=camera,
            )
            data[role] = Frames(split, entries, FrameSource(extractor, data_root, rs))

        def validate() -> float:
            return evaluate(
                model, data["validation"], recipe, stride=recipe.validation_stride
            ).overall.ap_iou25

        history = train(model, data["train"], recipe, validate=validate, seed=seed)
        began = time.perf_counter()
        report = evaluate(model, data["test"], recipe)
        history["test_seconds"] = time.perf_counter() - began
        report.write_json(directory)
        torch.save(head.state_dict(), directory / "head.pt")
        torch.save(stem.state_dict(), directory / "stem.pt")
        history["targets_lost_to_shared_subcells"] = {
            role: d.split.lost for role, d in data.items()
        }
        history["crowding"] = {role: d.split.crowding() for role, d in data.items()}
        history["test_predictions_tied"] = report.overall.n_predictions_tied
        (directory / "training.json").write_text(
            json.dumps(history, indent=2, allow_nan=True) + "\n"
        )
    except BaseException as error:
        run_module.finish_run(directory, error)
        raise
    run_module.finish_run(directory)
    return {
        "run": str(directory),
        "best_step": history["best_step"],
        "skipped_steps": history["skipped_steps"],
        "test": report.overall.to_dict(),
        "mean_step_seconds": float(np.mean(history["step_seconds"])),
        "validation_seconds": history["validation_seconds"],
        "test_seconds": history["test_seconds"],
    }
