"""Train and score the person-centre head on cached features (#66).

One run is one fold, one camera, one label fraction and one seed. It trains on
the fold's training subset at that fraction, keeps the step that scores best
on the fold's validation split, and scores that on the fold's test site-day
through `aerial_search.evaluation.detection`. Features are read only through
the cache's own functions. The design is in docs/detection-head-design.md.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import ArrayLike

from aerial_search.data.folds import TEST, VALIDATION, site_day, train_file, view_dir
from aerial_search.evaluation.detection import (
    DetectionReport,
    GroundTruth,
    Prediction,
    evaluate_detections,
)
from aerial_search.experiments.feature_cache import (
    CacheError,
    _feature_path,
    cell_boxes,
    read_features,
    read_index,
)
from aerial_search.models.head import CentreHead, centre_loss
from aerial_search.models.thermal_input import DEFAULT_ARM

LOG_SIZE_RANGE = (-8.0, 4.0)  # clamp before exp: 0.0003 to 55 cells


class LeakError(Exception):
    """A manifest would let the test site-day reach training or selection."""


@dataclass(frozen=True)
class Recipe:
    """Everything that defines how a head is trained and decoded."""

    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 250
    upsample: int = 4
    hidden: int = 256
    top_k: int = 100


# --- geometry, targets, decoding ---------------------------------------------


def _subdivide(edges: np.ndarray, parts: int) -> np.ndarray:
    steps = np.arange(parts) / parts
    inner = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * steps
    return np.append(inner.reshape(-1), edges[-1])


@dataclass(frozen=True, eq=False)
class Geometry:
    """Source-pixel edges of the head's output grid for one image.

    Each feature cell (from `cell_boxes`, so a narrow edge cell stays narrow)
    is split into `upsample` equal parts each way. Sizes are measured in
    units of one full cell, `cell`, which does not depend on the image's
    resolution relative to the model input.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    cell: tuple[float, float]
    image_size: tuple[int, int]
    upsample: int

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], upsample: int) -> Geometry:
        boxes = cell_boxes(entry)
        x = np.append(boxes[0, :, 0], boxes[0, -1, 2])
        y = np.append(boxes[:, 0, 1], boxes[-1, 0, 3])
        iw, ih = entry["image_size"]
        ph, pw = entry["patch_grid"]
        k = entry["pooling"]
        return cls(
            _subdivide(x, upsample),
            _subdivide(y, upsample),
            (k * iw / pw, k * ih / ph),
            (int(iw), int(ih)),
            upsample,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y_edges) - 1, len(self.x_edges) - 1


@dataclass(frozen=True)
class Targets:
    heat: np.ndarray  # (H, W), 1 at a person's sub-cell
    offset: np.ndarray  # (2, H, W), centre position inside the sub-cell, [0, 1]
    size: np.ndarray  # (2, H, W), log(width / cell width), log(height / cell height)
    lost: int  # people dropped because another already held their sub-cell


def _locate(edges: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(edges) - 1
    index = np.clip(np.searchsorted(edges, centres, side="right") - 1, 0, n - 1)
    offset = (centres - edges[index]) / (edges[index + 1] - edges[index])
    return index, np.clip(offset, 0.0, 1.0)


def assign_targets(geometry: Geometry, boxes: ArrayLike) -> Targets:
    """One peak per person at the sub-cell holding its centre.

    Where two people share a sub-cell, the first in the list keeps it and the
    other is counted in `lost`.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = geometry.shape
    heat = np.zeros((h, w), dtype=np.float32)
    offset = np.zeros((2, h, w), dtype=np.float64)
    size = np.zeros((2, h, w), dtype=np.float64)
    columns, dx = _locate(geometry.x_edges, (b[:, 0] + b[:, 2]) / 2)
    rows, dy = _locate(geometry.y_edges, (b[:, 1] + b[:, 3]) / 2)
    cw, ch = geometry.cell
    lost = 0
    for k in range(len(b)):
        r, c = rows[k], columns[k]
        if heat[r, c]:
            lost += 1
            continue
        heat[r, c] = 1
        offset[:, r, c] = dx[k], dy[k]
        size[:, r, c] = (
            math.log((b[k, 2] - b[k, 0]) / cw),
            math.log((b[k, 3] - b[k, 1]) / ch),
        )
    return Targets(heat, offset, size, lost)


@dataclass(frozen=True)
class Crowding:
    """People in one image that one-peak-per-sub-cell decoding cannot all find.

    `shared`: people dropped because another holds their sub-cell.
    `adjacent`: people whose sub-cell touches (8-neighbourhood) another
    occupied one. `unreachable`: the fewest of those a 3 x 3 peak test must
    lose: in each group of touching sub-cells, the group size minus the most
    sub-cells that can be kept with no two touching (exact up to 16 sub-cells
    in a group, a greedy upper bound beyond).
    """

    shared: int
    adjacent: int
    unreachable: int


EXACT_GROUP = 16


def _most_apart(cells: list[tuple[int, int]]) -> int:
    """The largest set of cells with no two touching (king's-move graph)."""

    def touch(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1

    n = len(cells)
    if n > EXACT_GROUP:  # greedy: a valid set, so possibly too small
        chosen: list[tuple[int, int]] = []
        for cell in sorted(cells):
            if not any(touch(cell, c) for c in chosen):
                chosen.append(cell)
        return len(chosen)
    clash = [
        sum(1 << j for j in range(n) if touch(cells[i], cells[j])) for i in range(n)
    ]
    best = 0
    for mask in range(1 << n):
        if all(not (mask >> i & 1) or not (mask & clash[i]) for i in range(n)):
            best = max(best, mask.bit_count())
    return best


def crowding(geometry: Geometry, boxes: ArrayLike) -> Crowding:
    targets = assign_targets(geometry, boxes)
    occupied = {(int(r), int(c)) for r, c in np.argwhere(targets.heat == 1)}
    adjacent, unreachable, seen = 0, 0, set()
    for start in sorted(occupied):
        if start in seen:
            continue
        group, todo = [], [start]
        seen.add(start)
        while todo:
            r, c = todo.pop()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near = (r + dr, c + dc)
                    if near in occupied and near not in seen:
                        seen.add(near)
                        todo.append(near)
        if len(group) > 1:
            adjacent += len(group)
            unreachable += len(group) - _most_apart(group)
    return Crowding(targets.lost, adjacent, unreachable)


def as_arrays(targets: Targets) -> dict[str, np.ndarray]:
    return {
        "heat": targets.heat,
        "offset": targets.offset.astype(np.float32),
        "size": targets.size.astype(np.float32),
    }


def decode(
    heat: ArrayLike,
    offset: ArrayLike,
    size: ArrayLike,
    geometry: Geometry,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (n, 4) in source pixels and scores (n,) for one image.

    CenterNet decoding: a sub-cell is a detection if its centre probability
    is the maximum of its 3 x 3 neighbourhood; the `top_k` highest are kept.
    Boxes are clipped to the image.
    """
    logits = torch.as_tensor(np.asarray(heat, dtype=np.float64))
    scores = logits.sigmoid()
    pooled = functional.max_pool2d(scores[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = torch.nonzero(scores == pooled).numpy()
    values = scores.numpy()[peaks[:, 0], peaks[:, 1]]
    order = np.lexsort((peaks[:, 1], peaks[:, 0], -values))[:top_k]
    rows, columns = peaks[order, 0], peaks[order, 1]
    off = np.asarray(offset, dtype=np.float64)[:, rows, columns]
    log_size = np.clip(
        np.asarray(size, dtype=np.float64)[:, rows, columns], *LOG_SIZE_RANGE
    )
    xe, ye = geometry.x_edges, geometry.y_edges
    cx = xe[columns] + off[0] * (xe[columns + 1] - xe[columns])
    cy = ye[rows] + off[1] * (ye[rows + 1] - ye[rows])
    w = np.exp(log_size[0]) * geometry.cell[0]
    h = np.exp(log_size[1]) * geometry.cell[1]
    iw, ih = geometry.image_size
    boxes = np.stack(
        [
            np.clip(cx - w / 2, 0, iw),
            np.clip(cy - h / 2, 0, ih),
            np.clip(cx + w / 2, 0, iw),
            np.clip(cy + h / 2, 0, ih),
        ],
        axis=1,
    )
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], values[order][keep]


# --- manifests -----------------------------------------------------------------


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


# --- the cache -----------------------------------------------------------------


@dataclass(frozen=True)
class CacheView:
    entries: dict[str, dict]  # index entry of every image the run needs
    sessions: dict[str, str | None]  # cache session -> its run status
    settings: dict
    missing: list[str]  # images not in the cache; only a scratch run gets here


def check_cache(
    cache_dir: Path,
    records: Sequence[Mapping[str, Any]],
    camera: str,
    *,
    scratch: bool,
    pinned: Mapping[str, str] | None = None,
) -> CacheView:
    """Refuse a cache a run cannot trust; return the entries it will read.

    Every image in the cache must be of this camera, with the image size the
    manifest records, and one grid shape. A normal run refuses an image not
    in the index or without its feature file (the cache is incomplete); a
    scratch run leaves such images out and lists them in `missing`. A normal
    run also refuses a cache under `scratch-features/` or holding a
    `session.lock` (being written), any image written by a cache session
    that was scratch, not completed, or has no run record, and (given
    `pinned`) any image whose source hash is not the pinned one.
    """
    try:
        settings = json.loads((cache_dir / "cache.json").read_text())
    except FileNotFoundError:
        raise CacheError(
            f"{cache_dir} has no cache.json; not a feature cache"
        ) from None
    if not scratch and cache_dir.parent.name.startswith("scratch"):
        raise CacheError(f"{cache_dir} is a scratch cache; a normal run cannot use it")
    lock = cache_dir / "session.lock"
    if not scratch and lock.exists():
        raise CacheError(
            f"{lock} exists ({lock.read_text().strip()}): a session is writing the "
            "cache, so its index may still change; wait for it to finish"
        )
    index = read_index(cache_dir)
    images = [str(r["image"]) for r in records]
    missing = [
        i for i in images if i not in index or not _feature_path(cache_dir, i).exists()
    ]
    if missing and not scratch:
        raise CacheError(
            f"{len(missing)} of {len(images)} images this run needs are not in "
            f"{cache_dir} (first: {missing[0]}); the cache is incomplete"
        )
    absent = set(missing)
    entries = {i: index[i] for i in images if i not in absent}
    for record in records:
        entry = entries.get(str(record["image"]))
        if entry is None:
            continue
        if entry["camera"] != camera:
            raise CacheError(f"{record['image']} is {entry['camera']} in the cache")
        if list(entry["image_size"]) != list(record["size"]):
            raise CacheError(
                f"{record['image']}: image size {entry['image_size']} in the cache, "
                f"{record['size']} in the manifest"
            )
        if pinned is not None and entry.get("sha256") != pinned.get(entry["path"]):
            raise CacheError(
                f"{record['image']}: the cache's source hash is not pinned"
            )
    grids = {tuple(e["grid"]) for e in entries.values()}
    if len(grids) != 1:
        raise CacheError(f"{camera} images have several grid shapes: {sorted(grids)}")
    sessions: dict[str, str | None] = {}
    for name in sorted({str(e["run"]) for e in entries.values()}):
        try:
            run = json.loads((cache_dir / "runs" / name / "run.json").read_text())
        except FileNotFoundError:
            run = None
        if not scratch and (run is None or run.get("scratch") is not False):
            raise CacheError(
                f"cache session {name} is scratch or has no run record; a normal "
                "run cannot use its features"
            )
        if not scratch and run is not None and run.get("status") != "completed":
            raise CacheError(
                f"cache session {name} has status {run.get('status')!r}, not "
                "completed; a normal run cannot use its features"
            )
        sessions[name] = None if run is None else run.get("status")
    return CacheView(entries, sessions, settings, missing)


def cache_arm(cache_json: Mapping[str, Any]) -> str:
    """How the cache's frames reached the backbone; absent means replicate."""
    return str(cache_json.get("settings", {}).get("input_handling", DEFAULT_ARM))


# --- data, training, scoring ---------------------------------------------------


@dataclass(eq=False)
class Split:
    """One split held in memory: features, geometry, labels, and targets."""

    records: list[dict]
    features: np.ndarray  # (n, h, w, d) float16
    geometry: list[Geometry]
    boxes: list[np.ndarray]
    targets: dict[str, np.ndarray]  # heat (n, H, W), offset and size (n, 2, H, W)
    lost: int
    camera: str

    @classmethod
    def build(
        cls,
        records: list[dict],
        features: np.ndarray,
        geometry: list[Geometry],
        boxes: list[np.ndarray],
        *,
        camera: str,
    ) -> Split:
        per_image = [assign_targets(g, b) for g, b in zip(geometry, boxes, strict=True)]
        stacked = {
            key: np.stack([as_arrays(t)[key] for t in per_image])
            for key in ("heat", "offset", "size")
        }
        lost = sum(t.lost for t in per_image)
        return cls(records, features, geometry, boxes, stacked, lost, camera)

    def __len__(self) -> int:
        return len(self.records)

    def crowding(self) -> dict[str, int]:
        """People, and those a one-peak decoder cannot all find, summed."""
        per_image = [
            crowding(g, b) for g, b in zip(self.geometry, self.boxes, strict=True)
        ]
        return {
            "people": int(sum(len(b) for b in self.boxes)),
            "shared": sum(c.shared for c in per_image),
            "adjacent": sum(c.adjacent for c in per_image),
            "unreachable": sum(c.unreachable for c in per_image),
        }


def load_split_features(
    cache_dir: Path,
    records: list[dict],
    entries: Mapping[str, dict],
    camera: str,
    upsample: int,
) -> Split:
    features = np.stack(
        [np.asarray(read_features(cache_dir, str(r["image"]))) for r in records]
    )
    geometry = [
        Geometry.from_entry(entries[str(r["image"])], upsample) for r in records
    ]
    boxes = [record_boxes(r) for r in records]
    return Split.build(records, features, geometry, boxes, camera=camera)


def _batch(split: Split, index: np.ndarray, device: torch.device):
    features = torch.from_numpy(split.features[index]).to(device)
    targets = {
        k: torch.from_numpy(v[index]).to(device) for k, v in split.targets.items()
    }
    return features, targets


def train_head(
    head: CentreHead,
    split: Split,
    recipe: Recipe,
    *,
    device: torch.device,
    validate: Callable[[], float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """AdamW with cosine decay for `recipe.steps` steps.

    Batches are consecutive slices of a running order of frames: whenever
    fewer than a batch remain, a fresh shuffle of the whole split is appended.
    So every frame is seen once per pass; a batch can hold the end of one
    pass and the start of the next, and then may hold one frame twice; and a
    split smaller than `batch_size` gives batches of the whole split (at 1%,
    30 frames make batches of 30, not 32). Every `eval_every` steps (and at
    the end)
    `validate` scores the head; the best-scoring weights are restored at the
    end (the earlier step wins a tie, and a NaN score never wins). Without
    `validate`, the final weights are kept.
    """
    head.to(device)
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, recipe.steps)
    rng = np.random.default_rng(seed)
    order: list[int] = []
    history: dict[str, Any] = {"loss": [], "validation": [], "best_step": None}
    best, best_state = -math.inf, None
    for step in range(1, recipe.steps + 1):
        head.train()
        if len(order) < recipe.batch_size:
            order += rng.permutation(len(split)).tolist()
        index = np.sort(np.array(order[: recipe.batch_size]))
        del order[: recipe.batch_size]
        features, targets = _batch(split, index, device)
        loss = centre_loss(head(features), targets)
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
            print(f"step {step}: loss {loss.item():.4f}, val ap_iou25 {score:.4f}")
            if score > best:  # NaN compares false
                best, best_state = score, copy.deepcopy(head.state_dict())
                history["best_step"] = step
    if best_state is not None:
        head.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(
    head: CentreHead, split: Split, device: torch.device, top_k: int, batch: int = 64
) -> list[Prediction]:
    head.to(device).eval()
    out = []
    for start in range(0, len(split), batch):
        index = np.arange(start, min(start + batch, len(split)))
        features, _ = _batch(split, index, device)
        heat, offset, size = (t.cpu().numpy() for t in head(features))
        for k, n in enumerate(index):
            boxes, scores = decode(
                heat[k], offset[k], size[k], split.geometry[n], top_k
            )
            out.append(Prediction(str(split.records[n]["image"]), boxes, scores))
    return out


def evaluate(
    head: CentreHead, split: Split, device: torch.device, top_k: int
) -> DetectionReport:
    """Score the head on a split through the evaluation module."""
    truths = [
        GroundTruth(str(r["image"]), b, split.camera, str(r["collection_id"]))
        for r, b in zip(split.records, split.boxes, strict=True)
    ]
    return evaluate_detections(truths, predict(head, split, device, top_k))


# --- one run -------------------------------------------------------------------


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
        f"detection-head/{cache_dir.name}{suffix}/"
        f"{camera}-{fold}-{percent}pct-seed{seed}"
    )


def build_head(seed: int, dim: int, recipe: Recipe) -> CentreHead:
    """Seed, then build the head: the one place a run's initial head is made.

    Arm C (`stem_experiment`) calls this too, before it draws anything else
    from the random stream, so at one seed every arm starts from the same head.
    """
    torch.manual_seed(seed)
    return CentreHead(dim=dim, hidden=recipe.hidden, upsample=recipe.upsample)


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


def run_head(
    *,
    cache_dir: Path,
    manifests: Path,
    data_root: Path,
    fold: str,
    camera: str,
    percent: int,
    seed: int,
    recipe: Recipe,
    scratch: bool,
    run_name: str | None,
    device: torch.device,
    repo: Path | None = None,
    argv: list[str] | None = None,
    input_handling: str | None = None,
    init_head: Path | None = None,
) -> dict[str, Any]:
    """One run: train at `percent`, select on validation, score on test.

    The input arm (#68) is the one the cache was made with
    (`cache.json` settings `input_handling`; a cache without the field is
    "replicate") and is recorded in `run.json`. `input_handling`, if given,
    must equal it; a mismatch is refused.

    Writes `run.json`, `detection_metrics.json` (test), `training.json`
    (loss, validation scores, chosen step) and `head.pt` into the run
    directory. Returns a short summary.

    `init_head`, a finished cold arm-A run directory with the same fold,
    percent, camera and seed, starts the head from that run's trained head
    (a warm start, for the matched control of the stem arm; see
    `load_warm_start`). It is recorded in `config.warm_start`.
    """
    from aerial_search import run as run_module
    from aerial_search.data import checksums

    paths, records = select_records(manifests, fold, camera, percent)
    pinned = None if scratch else {e.path: e.sha256 for e in checksums.committed_list()}
    cache = check_cache(
        cache_dir,
        [r for rs in records.values() for r in rs],
        camera,
        scratch=scratch,
        pinned=pinned,
    )
    arm = cache_arm(cache.settings)
    if input_handling is not None and input_handling != arm:
        raise CacheError(
            f"{cache_dir} holds {arm!r} features, not {input_handling!r}; use the "
            "cache made with that input handling"
        )
    if cache.missing:  # a scratch run only; check_cache refuses a normal one
        print(f"scratch: {len(cache.missing)} images not in the cache, left out")
        absent = set(cache.missing)
        records = {
            role: [r for r in rs if r["image"] not in absent]
            for role, rs in records.items()
        }
    warm, warm_record = None, {}
    if init_head is not None:
        if arm != DEFAULT_ARM:
            raise CacheError(
                f"--init-head is for the {DEFAULT_ARM!r} control and the stem arm, "
                f"not {arm!r}"
            )
        warm, warm_record = load_warm_start(
            init_head,
            fold=fold,
            percent=percent,
            camera=camera,
            seed=seed,
            recipe=recipe,
            cache_settings=cache.settings["settings"],
            manifests=list(paths.values()),
            scratch=scratch,
        )
    config = {
        "camera": camera,
        "fold": fold,
        "percent": percent,
        "input_handling": arm,
        **({"warm_start": warm_record} if warm_record else {}),
        "recipe": asdict(recipe),
        "cache_dir": str(cache_dir),
        "cache": cache.settings,
        "cache_sessions": cache.sessions,
        "images": {role: len(rs) for role, rs in records.items()},
        "images_missing_from_cache": cache.missing,
        "head": "CentreHead: LayerNorm, conv1x1, GELU, conv3x3, GELU, conv1x1, "
        "pixel shuffle; CenterNet focal + L1 offset + L1 log size",
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
        splits = {
            role: load_split_features(
                cache_dir, rs, cache.entries, camera, recipe.upsample
            )
            for role, rs in records.items()
        }
        head = build_head(seed, splits["train"].features.shape[-1], recipe)
        if warm is not None:
            head.load_state_dict(warm)

        def validate() -> float:
            return evaluate(
                head, splits["validation"], device, recipe.top_k
            ).overall.ap_iou25

        history = train_head(
            head, splits["train"], recipe, device=device, validate=validate, seed=seed
        )
        report = evaluate(head, splits["test"], device, recipe.top_k)
        report.write_json(directory)
        torch.save(head.state_dict(), directory / "head.pt")
        history["targets_lost_to_shared_subcells"] = {
            role: s.lost for role, s in splits.items()
        }
        # people: labelled; shared: lost to a shared sub-cell; adjacent: next
        # to another person's sub-cell; unreachable: of those, the fewest a
        # 3x3 peak test must lose (see Crowding)
        history["crowding"] = {role: s.crowding() for role, s in splits.items()}
        history["test_predictions_tied"] = report.overall.n_predictions_tied
        (directory / "training.json").write_text(
            json.dumps(history, indent=2, allow_nan=True) + "\n"
        )
    except BaseException as error:
        run_module.finish_run(directory, error)
        raise
    run_module.finish_run(directory)
    test = report.overall.to_dict()
    return {
        "run": str(directory),
        "best_step": history["best_step"],
        "test": test,
        "test_crowding": history["crowding"]["test"],
        "test_predictions_tied": history["test_predictions_tied"],
        "by_size": {k: m.to_dict() for k, m in report.by_size.items()},
    }
