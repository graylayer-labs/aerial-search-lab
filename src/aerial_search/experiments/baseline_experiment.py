"""Fine-tune an off-the-shelf detector on the same labels as the head (#67).

The supervised baseline: a torchvision detector with COCO weights, trained on
the same fold, camera, label fraction and seed as `train-head`, selected on the
fold's validation split and scored on its test site-day through
`aerial_search.evaluation.detection`. Which frames it sees comes from
`head_experiment.select_records`, the function the head uses, and the score is
the same `evaluate_detections`, so the two sit in one table
(`head_table.markdown_compare`).
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2.functional as vision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image

from aerial_search.data import tiling
from aerial_search.evaluation.detection import (
    DetectionReport,
    GroundTruth,
    Prediction,
    evaluate_detections,
)
from aerial_search.experiments.head_experiment import record_boxes, select_records

ARCHITECTURES = (
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "fcos_resnet50_fpn",
)

# The model input of the frozen backbone's cache (docs/detection-head-design.md):
# every image is stretched to this, whatever its source size. "cache" gives the
# baseline the same pixels the head's features were computed from.
CACHE_INPUT = {"rgb": (672, 384), "thermal": (560, 448)}
INPUTS = ("cache", "native", "tiled")
# Tiled input (#96): overlapping tiles of the source frame at native resolution,
# each tile the size of the camera's cache input so the model is unchanged.
# The overlap (pixels between neighbouring tiles) is chosen so that most people
# are wholly inside at least one tile: RGB people are 90th percentile 104x146 px
# and thermal 50x76 px in the fold manifests, so 160 and 96 cover them; the
# largest boxes (RGB up to 972 px) cannot fit any overlap and are seen cut.
TILE_OVERLAP = {"rgb": 160, "thermal": 96}
MERGE_IOU = 0.5  # class-agnostic NMS over a stitched frame
# Anchor sizes per feature-pyramid level (default 32 to 512): the smallest
# people are 4 px wide at the head's input size. The number of anchors per
# location must equal the pretrained RPN head's (a mismatch indexes the wrong
# logits without any error): ResNet-50 has five levels with 1 size each (3
# anchors with 3 ratios); MobileNet has three levels (finest stride 32 at
# 672x384 input) with 5 sizes each (15 anchors).
ANCHORS: dict[str, tuple[tuple[int, ...], ...]] = {
    "fasterrcnn_resnet50_fpn": tuple((s,) for s in (8, 16, 32, 64, 128)),
    "fasterrcnn_mobilenet_v3_large_fpn": ((8, 16, 32, 64, 128),) * 3,
}


@dataclass(frozen=True)
class BaselineRecipe:
    """Everything that defines how the baseline is built, trained and decoded."""

    arch: str = "fasterrcnn_mobilenet_v3_large_fpn"
    input: str = "cache"  # "cache": the head's model input; "native": no resize
    steps: int = 300
    batch_size: int = 8
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    eval_every: int = 100
    val_frames: int = 100  # validation frames used to pick the step (0: all)
    top_k: int = 100
    score_floor: float = 0.001  # keep low-scoring boxes: AP needs the whole curve
    workers: int = 4
    device: str = "cpu"  # MPS gave a diverged loss (docs/decisions.md)
    # Tiled input only (#96). 0 overlap: the camera's default (TILE_OVERLAP).
    tile_overlap: int = 0
    min_visible: float = 0.5  # train: keep a cut box if this much of it shows
    empty_ratio: float = 1.0  # train: empty tiles per tile with a person
    frame_top_k: int = 300  # eval: boxes kept per frame after merging tiles


def input_size(camera: str, mode: str) -> tuple[int, int] | None:
    """(width, height) the images are stretched to, or None for native."""
    if mode in ("native", "tiled"):
        return None
    if mode == "cache":
        return CACHE_INPUT[camera]
    raise ValueError(f"input must be one of {INPUTS}, not {mode!r}")


@dataclass(frozen=True)
class TileSpec:
    """How a frame is cut into tiles, and how cut boxes are treated."""

    size: tuple[int, int]  # (width, height) of a tile, source pixels
    overlap: int
    min_visible: float
    empty_ratio: float
    merge_iou: float = MERGE_IOU


def tile_spec(camera: str, recipe: BaselineRecipe) -> TileSpec | None:
    """The recipe's tiling, or None when the input is not tiled."""
    if recipe.input != "tiled":
        return None
    return TileSpec(
        CACHE_INPUT[camera],
        recipe.tile_overlap or TILE_OVERLAP[camera],
        recipe.min_visible,
        recipe.empty_ratio,
    )


def subsample(records: Sequence[dict], limit: int) -> list[dict]:
    """`limit` records spread evenly through the list (all if limit is 0)."""
    if limit <= 0 or len(records) <= limit:
        return list(records)
    picks = np.linspace(0, len(records) - 1, limit).round().astype(int)
    return [records[i] for i in picks]


# --- model ----------------------------------------------------------------------


def build_model(
    arch: str,
    *,
    pretrained: bool,
    anchor_sizes: Sequence[Sequence[int]] = (),
    score_floor: float = 0.001,
    top_k: int = 100,
):
    """A torchvision detector with one person class (plus background).

    `pretrained` loads COCO weights and replaces the final classifier; tests
    pass False and load nothing. The internal resize is switched off: images
    are resized by the dataset, so the model never rescales them and its
    boxes come back in the pixels it was given. `anchor_sizes` replaces the
    default anchors (32 to 512 px), which cannot match a person 4 pixels
    wide, with one size per feature level; it is only for Faster R-CNN.
    """
    from torchvision.models import detection

    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}, not {arch!r}")
    ctor = getattr(detection, arch)
    kwargs: dict[str, Any] = {}
    if pretrained:
        kwargs["weights"] = "COCO_V1"
    else:
        kwargs["weights"] = None
        kwargs["weights_backbone"] = None
    if arch.startswith("fasterrcnn"):
        model = ctor(
            **kwargs, box_detections_per_img=top_k, box_score_thresh=score_floor
        )
        if anchor_sizes:
            from torchvision.models.detection.anchor_utils import AnchorGenerator

            ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            generator = AnchorGenerator(tuple(map(tuple, anchor_sizes)), ratios)
            head = model.rpn.head
            if generator.num_anchors_per_location() != [
                head.cls_logits.out_channels
            ] * len(anchor_sizes):
                raise ValueError(
                    f"{arch}: {generator.num_anchors_per_location()} anchors per "
                    "location, but the RPN head predicts "
                    f"{head.cls_logits.out_channels}"
                )
            model.rpn.anchor_generator = generator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, 2
        )
    elif arch == "retinanet_resnet50_fpn":
        model = ctor(**kwargs, detections_per_img=top_k, score_thresh=score_floor)
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        old = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            old.cls_logits.in_channels, old.num_anchors, 2
        )
    else:
        model = ctor(**kwargs, detections_per_img=top_k, score_thresh=score_floor)
        from torchvision.models.detection.fcos import FCOSClassificationHead

        old = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(
            old.cls_logits.in_channels, old.num_anchors, 2
        )
    model.transform.resize = lambda image, target=None: (image, target)
    return model


# --- data -----------------------------------------------------------------------


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
        for r, record in enumerate(self.records):
            width, height = record["size"]
            truth = record_boxes(record)
            for tile in tiling.tile_grid(width, height, spec.size, spec.overlap):
                local, _ = tiling.boxes_in_tile(truth, tile, spec.min_visible)
                every.append((r, tile, local))
        picked = tiling.pick_tiles(
            [len(local) > 0 for _, _, local in every], spec.empty_ratio, seed
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

    def __init__(self, n: int, batch_size: int, steps: int, seed: int):
        self.n, self.batch_size, self.steps, self.seed = n, batch_size, steps, seed

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order: list[int] = []
        for _ in range(self.steps):
            if len(order) < self.batch_size:
                order += rng.permutation(self.n).tolist()
            yield sorted(order[: self.batch_size])
            del order[: self.batch_size]


# --- training and scoring ---------------------------------------------------------


def _lr_factor(step: int, recipe: BaselineRecipe) -> float:
    if step <= recipe.warmup_steps:
        return step / recipe.warmup_steps
    progress = (step - recipe.warmup_steps) / max(recipe.steps - recipe.warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_baseline(
    model,
    frames: Frames | TiledTrainSet,
    recipe: BaselineRecipe,
    *,
    device: torch.device,
    validate: Callable[[], float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """SGD with momentum, linear warm-up and cosine decay for `recipe.steps`.

    Batches follow `batch_order`. Every `eval_every` steps (and at the end)
    `validate` scores the model; the best-scoring weights are restored at the
    end (the earlier step wins a tie; NaN never wins).
    """
    model.to(device)
    optimiser = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe.learning_rate,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: _lr_factor(s + 1, recipe)
    )
    loader = DataLoader(
        frames,
        batch_sampler=BatchSteps(len(frames), recipe.batch_size, recipe.steps, seed),
        num_workers=recipe.workers,
        collate_fn=_collate,
        persistent_workers=recipe.workers > 0,
    )
    history: dict[str, Any] = {"loss": [], "validation": [], "best_step": None}
    best, best_state = -math.inf, None
    for step, (images, targets, _) in enumerate(loader, start=1):
        model.train()
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss {loss.item()} at step {step}")
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        history["loss"].append(float(loss.item()))
        if validate is not None and (
            step % recipe.eval_every == 0 or step == recipe.steps
        ):
            score = validate()
            history["validation"].append({"step": step, "ap_iou25": score})
            print(
                f"step {step}: loss {loss.item():.4f}, val ap_iou25 {score:.4f}",
                flush=True,
            )
            if score > best:
                best, best_state = score, copy.deepcopy(model.state_dict())
                history["best_step"] = step
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_tiled(
    model,
    frames: TiledFrames,
    device: torch.device,
    *,
    batch_size: int = 8,
    workers: int = 0,
    frame_top_k: int = 300,
) -> list[Prediction]:
    """Detections per whole frame: each tile's boxes moved to source pixels,
    then duplicates from the overlaps merged by class-agnostic NMS."""
    model.to(device).eval()
    loader = DataLoader(frames, batch_size=1, num_workers=workers, collate_fn=_collate)
    out: list[Prediction] = []
    for n, ((tiles,), (grid,)) in enumerate(loader):
        boxes, scores = [], []
        for i in range(0, len(tiles), batch_size):
            outputs = model([t.to(device) for t in tiles[i : i + batch_size]])
            for o, tile in zip(outputs, grid[i : i + batch_size], strict=True):
                local = o["boxes"].cpu().numpy().astype(np.float64)
                boxes.append(tiling.to_source(local, tile))
                scores.append(o["scores"].cpu().numpy().astype(np.float64))
        merged, merged_scores = tiling.merge_detections(
            np.concatenate(boxes), np.concatenate(scores), frames.spec.merge_iou
        )
        merged, merged_scores = merged[:frame_top_k], merged_scores[:frame_top_k]
        out.append(Prediction(str(frames.records[n]["image"]), merged, merged_scores))
    return out


@torch.no_grad()
def predict(
    model,
    frames: Frames | TiledFrames,
    device: torch.device,
    *,
    batch_size: int = 8,
    workers: int = 0,
    frame_top_k: int = 300,
) -> list[Prediction]:
    """Detections in source pixels (the dataset's scale undone) per frame."""
    if isinstance(frames, TiledFrames):
        return predict_tiled(
            model,
            frames,
            device,
            batch_size=batch_size,
            workers=workers,
            frame_top_k=frame_top_k,
        )
    model.to(device).eval()
    loader = DataLoader(
        frames, batch_size=batch_size, num_workers=workers, collate_fn=_collate
    )
    out: list[Prediction] = []
    n = 0
    for images, _, scales in loader:
        outputs = model([im.to(device) for im in images])
        for o, scale in zip(outputs, scales, strict=True):
            boxes = o["boxes"].cpu().numpy().astype(np.float64)
            boxes = boxes / np.tile(scale.numpy().astype(np.float64), 2)
            width, height = frames.records[n]["size"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            out.append(
                Prediction(
                    str(frames.records[n]["image"]),
                    boxes,
                    o["scores"].cpu().numpy().astype(np.float64),
                )
            )
            n += 1
    return out


def evaluate(
    model,
    frames: Frames | TiledFrames,
    device: torch.device,
    *,
    workers: int = 0,
    frame_top_k: int = 300,
) -> DetectionReport:
    """Score on a split with the same module and camera/flight grouping as the head."""
    truths = [
        GroundTruth(
            str(r["image"]), record_boxes(r), str(r["camera"]), str(r["collection_id"])
        )
        for r in frames.records
    ]
    predictions = predict(
        model, frames, device, workers=workers, frame_top_k=frame_top_k
    )
    return evaluate_detections(truths, predictions)


# --- one run ----------------------------------------------------------------------


def default_run_name(
    recipe: BaselineRecipe, camera: str, fold: str, percent: int, seed: int
):
    return (
        f"detection-baseline/{recipe.arch}-{recipe.input}/"
        f"{camera}-{fold}-{percent}pct-seed{seed}"
    )


def run_baseline(
    *,
    manifests: Path,
    data_root: Path,
    fold: str,
    camera: str,
    percent: int,
    seed: int,
    recipe: BaselineRecipe,
    scratch: bool,
    run_name: str | None,
    device: torch.device,
    repo: Path | None = None,
    argv: list[str] | None = None,
    limit_frames: int = 0,
) -> dict[str, Any]:
    """One run: fine-tune at `percent`, select on validation, score on test.

    `limit_frames` (scratch only) keeps that many frames of each split, for a
    timing or pipeline check. Writes `run.json`, `detection_metrics.json`
    (test) and `training.json` into the run directory.
    """
    from aerial_search import run as run_module

    if limit_frames and not scratch:
        raise ValueError("limit_frames needs a scratch run")
    if device.type != "cpu":
        if not scratch:
            raise ValueError(
                f"a normal run must use the cpu, not {device.type}: Faster R-CNN "
                "on MPS diverged (loss 7e7) and hung"
            )
        warnings.warn(f"{device.type} is unreliable for this baseline", stacklevel=2)
    recipe = replace(recipe, device=device.type)
    paths, records = select_records(manifests, fold, camera, percent)
    if limit_frames:
        records = {r: subsample(rs, limit_frames) for r, rs in records.items()}
    size = input_size(camera, recipe.input)
    spec = tile_spec(camera, recipe)
    config = {
        "method": f"baseline:{recipe.arch}@{recipe.input}",
        "camera": camera,
        "fold": fold,
        "percent": percent,
        "recipe": asdict(recipe),
        "input_size": size,
        "tiling": asdict(spec) if spec else None,
        "weights": "torchvision COCO_V1",
        "images": {role: len(rs) for role, rs in records.items()},
        "limit_frames": limit_frames,
    }
    directory = run_module.start_run(
        run_name or default_run_name(recipe, camera, fold, percent, seed),
        config,
        seed,
        device=str(device),
        scratch=scratch,
        argv=argv,
        repo=repo,
        inputs=list(paths.values()),
        fold=fold,
        view=camera,
        manifests=manifests,
        data_root=data_root,
    )
    try:
        torch.manual_seed(seed)
        model = build_model(
            recipe.arch,
            pretrained=True,
            anchor_sizes=ANCHORS.get(recipe.arch, ()),
            score_floor=recipe.score_floor,
            top_k=recipe.top_k,
        )
        val_records = subsample(records["validation"], recipe.val_frames)
        train: Frames | TiledTrainSet
        if spec is None:
            train = Frames(records["train"], data_root, size)
            val = Frames(val_records, data_root, size)
            test = Frames(records["test"], data_root, size)
        else:
            train = TiledTrainSet(records["train"], data_root, spec, seed)
            val = TiledFrames(val_records, data_root, spec)
            test = TiledFrames(records["test"], data_root, spec)

        def validate() -> float:
            report = evaluate(
                model,
                val,
                device,
                workers=recipe.workers,
                frame_top_k=recipe.frame_top_k,
            )
            return report.overall.ap_iou25

        history = train_baseline(
            model, train, recipe, device=device, validate=validate, seed=seed
        )
        report = evaluate(
            model,
            test,
            device,
            workers=recipe.workers,
            frame_top_k=recipe.frame_top_k,
        )
        report.write_json(directory)
        history["validation_frames"] = len(val)
        if isinstance(train, TiledTrainSet):
            history["train_tiles"] = len(train)
            history["train_tile_candidates"] = train.n_candidates
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
        "test": report.overall.to_dict(),
        "by_size": {k: m.to_dict() for k, m in report.by_size.items()},
    }
