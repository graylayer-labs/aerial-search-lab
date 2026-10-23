"""A cache of frozen-backbone features, one file per image (#65).

Layout of a cache directory `outputs/features/<model>-<budget>tok/`:

    cache.json         the settings that define the cache; never changed
    index.jsonl        one line per finished image, appended as work proceeds
    features/<clip dir>/<stem>.npy
                       h x w x d float16 array, readable with mmap

One file per image, mirroring the dataset's clip directories, because a write
is then one atomic rename, a crash loses at most the image in flight, and a
fold or a single image is read by opening only the files it names. The cost is
about 30,000 files of a few hundred kilobytes each, which the filesystem
handles without trouble. `index.jsonl` says which images are done and, for
each, the camera, the grid, the SHA-256 of the source image (from the pinned
dataset list), and the run and commit that wrote it.

The settings are the identity of the cache: model, weights, token budget,
pooling and dtype. A second run with any of them different refuses to write
into it. A resumed run skips finished images and recomputes a sample of them
to prove the weights and code still give the same features.
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from aerial_search.models.backbone import Extraction
from aerial_search.models.thermal_input import DEFAULT_ARM, input_transform

# A resumed run recomputes a sample on the same device with the same weights,
# which gives the same numbers up to the last bit of a float16. The root-mean-
# square difference, relative to the stored features' RMS, must stay below this:
# about one float16 step on a single element. A swapped or stale file is far
# above it.
VERIFY_TOLERANCE = 1e-3


class CacheError(Exception):
    """The cache cannot be written or trusted."""


class Extractor(Protocol):
    """Two steps, so image decoding runs ahead of the device in worker threads."""

    def prepare(self, image: Image.Image, /) -> Any:
        """The CPU work for one image (thread-safe); input to `extract`."""
        ...

    def extract(self, prepared: Any, /) -> Extraction:
        """Pooled float16 features and their geometry, from the device."""
        ...


WORKERS = 3  # images decoded and preprocessed ahead of the device
LOOKAHEAD = 6


@dataclass(frozen=True)
class CacheSettings:
    """What makes two caches interchangeable. All of it is part of the key."""

    model: str  # alias, for example "siglip2-base-naflex"
    repo: str  # model identifier on the hub
    revision: str  # hub commit of the weights
    weights_sha256: dict[str, str]  # per weight file
    token_budget: int
    pooling: int  # k of a k x k average pool; 1 is none
    dtype: str
    # bumped in backbone.PREPROCESSING_VERSION when what a feature file holds
    # changes; a cache made under another version is not interchangeable
    preprocessing_version: int
    # how a thermal frame reaches the backbone (#68): "replicate" (the frame
    # as stored) or "equalise" (histogram equalisation first). A cache made
    # before this field existed holds "replicate" features.
    input_handling: str = DEFAULT_ARM

    def name(self) -> str:
        suffix = "" if self.input_handling == DEFAULT_ARM else f"-{self.input_handling}"
        return f"{self.model}-{self.token_budget}tok{suffix}"


@dataclass(frozen=True)
class Frame:
    path: str  # relative to the data root
    camera: str  # "rgb" or "thermal"
    collection_id: str
    sha256: str | None  # from the pinned list, None if not listed


@dataclass
class Summary:
    requested: int = 0
    computed: int = 0
    skipped: int = 0
    verified: int = 0
    stale: int = 0  # finished images recomputed because their source hash changed
    max_drift: float = 0.0  # worst relative RMS difference of the verified sample
    seconds: float = 0.0


def fold_manifest_files(folds: Path) -> list[Path]:
    """Every manifest under the folds directory, in a fixed order."""
    return sorted(folds.rglob("*.jsonl")) if folds.is_dir() else []


def fold_frames(folds: Path, camera: str) -> list[tuple[str, str]]:
    """(path, collection) of each distinct image of one camera that any fold
    manifest references, sorted by path.

    A per-camera record names its image in `image` and carries `camera`; a
    paired or unlabelled record names `rgb_image` and `thermal_image`.
    """
    pair_key = "rgb_image" if camera == "rgb" else "thermal_image"
    found: dict[str, str] = {}
    for file in fold_manifest_files(folds):
        for line in file.read_text().splitlines():
            if not line:
                continue
            record = json.loads(line)
            if record.get("camera") == camera and "image" in record:
                path = record["image"]
            elif pair_key in record:
                path = record[pair_key]
            else:
                continue
            found.setdefault(path, record["collection_id"])
    return sorted(found.items())


def select_frames(
    manifest: Path,
    camera: str,
    pinned: Mapping[str, str],
    collections: Sequence[str] | None = None,
    limit: int | None = None,
    part: tuple[int, int] | None = None,
    folds: Path | None = None,
) -> list[Frame]:
    """The distinct images of one camera in a pair manifest, in manifest order,
    then (with `folds`) the images only the fold manifests reference, by path.
    """
    key = "rgb_image" if camera == "rgb" else "thermal_image"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if collections:
        known = {row["collection_id"] for row in rows}
        missing = sorted(set(collections) - known)
        if missing:
            raise CacheError(f"not in {manifest}: {', '.join(missing)}")
        rows = [row for row in rows if row["collection_id"] in collections]
    frames: dict[str, Frame] = {}
    for row in rows:
        path = row[key]
        frames.setdefault(
            path, Frame(path, camera, row["collection_id"], pinned.get(path))
        )
    if folds is not None:
        for path, collection in fold_frames(folds, camera):
            if collections and collection not in collections:
                continue
            frames.setdefault(path, Frame(path, camera, collection, pinned.get(path)))
    selected = list(frames.values())
    files: dict[Path, str] = {}
    for frame in selected:
        target = Path(frame.path).with_suffix(".npy")
        if files.setdefault(target, frame.path) != frame.path:
            raise CacheError(
                f"{files[target]} and {frame.path} map to the same feature file "
                f"{target}"
            )
    if part:
        k, n = part
        if not 1 <= k <= n:
            raise CacheError(f"part {k}/{n}: K must be from 1 to N")
        size = -(-len(selected) // n)  # ceiling, so the parts cover everything
        selected = selected[(k - 1) * size : k * size]
    return selected[:limit] if limit else selected


def _read_record(cache_dir: Path) -> dict | None:
    try:
        return json.loads((cache_dir / "cache.json").read_text())
    except FileNotFoundError:
        return None


def _transformers_version() -> str:
    return version("transformers")


def check_settings(cache_dir: Path, settings: CacheSettings) -> None:
    """Raise `CacheError` if the cache exists with other settings. Only warn
    if it was made with another `transformers` version."""
    record = _read_record(cache_dir)
    if record is None:
        return
    made_with = record.get("environment", {}).get("transformers")
    if made_with != _transformers_version():
        warnings.warn(
            f"{cache_dir} was made with transformers {made_with}, this is "
            f"{_transformers_version()}; the resume check will show if the "
            "features still agree",
            stacklevel=2,
        )
    stored, wanted = record["settings"], asdict(settings)
    # a cache written before `input_handling` existed is the replicate cache
    stored = {"input_handling": DEFAULT_ARM, **stored}
    differ = sorted(k for k in wanted if stored.get(k) != wanted[k])
    if differ:
        raise CacheError(
            f"{cache_dir} was made with different settings ({', '.join(differ)}); "
            "refusing to mix them. Use another cache, or delete this one."
        )


@contextmanager
def session_lock(cache_dir: Path, session: str) -> Iterator[None]:
    """One session at a time: the index has no other protection.

    Creates `session.lock` exclusively and removes it on exit. A session that
    was killed leaves it behind; the error says how to clear it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "session.lock"
    try:
        with path.open("x") as handle:
            handle.write(f"pid {os.getpid()} session {session}\n")
    except FileExistsError:
        raise CacheError(
            f"{path} exists ({path.read_text().strip()}): another session is "
            "writing this cache. If no session is running (one was killed), "
            f"delete it with `rm {path}` and run again."
        ) from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def cell_boxes(entry: Mapping[str, Any]) -> np.ndarray:
    """Source-pixel boxes of every cell of an index entry's feature grid.

    Returns an array of shape (grid h, grid w, 4) holding (x0, y0, x1, y1).
    The processor stretches the source image to `resized_size`, a whole
    number of patches, so one patch is `rw / pw` by `rh / ph` resized pixels,
    which is `iw / pw` by `ih / ph` source pixels. With pooling k, cell
    (i, j) holds patch rows k*i .. min(k*(i+1), ph) - 1 and patch columns
    k*j .. min(k*(j+1), pw) - 1, so

        x0 = iw * (k*j) / pw        x1 = iw * min(k*(j+1), pw) / pw
        y0 = ih * (k*i) / ph        y1 = ih * min(k*(i+1), ph) / ph

    An odd last row or column holds fewer than k patches and its box is
    correspondingly narrower. The boxes tile the source image exactly.
    """
    iw, ih = entry["image_size"]
    ph, pw = entry["patch_grid"]
    gh, gw = entry["grid"]
    k = entry["pooling"]
    x = np.minimum(np.arange(gw + 1) * k, pw) * iw / pw
    y = np.minimum(np.arange(gh + 1) * k, ph) * ih / ph
    boxes = np.empty((gh, gw, 4))
    boxes[..., 0] = x[None, :-1]
    boxes[..., 2] = x[None, 1:]
    boxes[..., 1] = y[:-1, None]
    boxes[..., 3] = y[1:, None]
    return boxes


def _feature_path(cache_dir: Path, image_path: str) -> Path:
    return cache_dir / "features" / Path(image_path).with_suffix(".npy")


def read_index(cache_dir: Path) -> dict[str, dict]:
    """Finished images by path. An unreadable line (a torn write) is skipped."""
    entries: dict[str, dict] = {}
    try:
        text = (cache_dir / "index.jsonl").read_text()
    except FileNotFoundError:
        return entries
    for line in text.splitlines():
        try:
            entry = json.loads(line)
            entries[entry["path"]] = entry
        except (ValueError, KeyError):
            continue
    return entries


def read_features(cache_dir: Path, image_path: str) -> np.ndarray:
    """The h x w x d float16 features of one image, memory-mapped."""
    return np.load(_feature_path(cache_dir, image_path), mmap_mode="r")


def _save_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _open(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def _prepare(extractor: Extractor, path: Path) -> Any:
    return extractor.prepare(_open(path))


def _prepared(extractor: Extractor, frames: Sequence[Frame], data_root: Path):
    """Yield (frame, prepared) in order, with the CPU work done ahead."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending: deque = deque()
        remaining = iter(frames)

        def submit() -> None:
            frame = next(remaining, None)
            if frame is not None:
                job = pool.submit(_prepare, extractor, data_root / frame.path)
                pending.append((frame, job))

        for _ in range(LOOKAHEAD):
            submit()
        try:
            while pending:
                frame, job = pending.popleft()
                submit()
                yield frame, job.result()
        finally:
            for _, job in pending:
                job.cancel()


def _verify(
    cache_dir: Path,
    done: list[Frame],
    extractor: Extractor,
    data_root: Path,
    count: int,
    seed: int,
) -> tuple[int, float]:
    """Recompute a sample; returns how many and the worst relative RMS drift."""
    sample = random.Random(seed).sample(done, min(count, len(done)))
    index = read_index(cache_dir)
    worst = 0.0
    for frame in sample:
        stored = np.asarray(read_features(cache_dir, frame.path), dtype=np.float32)
        prepared = _prepare(extractor, data_root / frame.path)
        extraction = extractor.extract(prepared)
        fresh = np.asarray(extraction.array, dtype=np.float32)
        entry = index[frame.path]
        if (
            list(extraction.image_size) != entry["image_size"]
            or list(extraction.resized_size) != entry["resized_size"]
            or list(extraction.patch_grid) != entry["patch_grid"]
        ):
            raise CacheError(
                f"{frame.path}: recomputed geometry differs from the index "
                f"({extraction.resized_size} from {extraction.image_size}); the "
                "cache does not match this code"
            )
        if stored.shape != fresh.shape:
            raise CacheError(
                f"{frame.path}: stored grid {stored.shape} but recomputed "
                f"{fresh.shape}; the cache does not match this code or weights"
            )
        rms = float(np.sqrt(np.mean((stored - fresh) ** 2)))
        drift = rms / max(float(np.sqrt(np.mean(stored**2))), 1e-6)
        worst = max(worst, drift)
        if drift > VERIFY_TOLERANCE:
            raise CacheError(
                f"{frame.path}: recomputed features differ from the stored ones "
                f"(relative RMS {drift:.2e}, limit {VERIFY_TOLERANCE:.0e}); the "
                "cache does not match this code or weights"
            )
    return len(sample), worst


def build_cache(
    cache_dir: Path,
    settings: CacheSettings,
    frames: Sequence[Frame],
    extractor: Extractor,
    data_root: Path,
    *,
    run: str,
    commit: str | None,
    verify: int = 8,
    seed: int = 0,
    progress: Callable[[int, int, float], None] | None = None,
) -> Summary:
    """Write features for the frames not yet in the cache, after verifying a
    sample of those that are. Returns what was done.

    Raises `CacheError` if the cache has other settings or a verified image no
    longer matches. Safe to repeat after a crash.
    """
    check_settings(cache_dir, settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "settings": asdict(settings),
        "environment": {"transformers": _transformers_version()},
        "format": 1,
    }
    try:  # exclusive, so two first sessions cannot both define the cache
        with (cache_dir / "cache.json").open("x") as handle:
            handle.write(json.dumps(record, indent=2) + "\n")
    except FileExistsError:
        check_settings(cache_dir, settings)

    started = time.perf_counter()
    index = read_index(cache_dir)
    done, stale = [], 0
    for f in frames:
        if f.path in index and _feature_path(cache_dir, f.path).exists():
            # a source image whose pinned hash changed is no longer the one
            # this feature file was made from
            if f.sha256 is not None and index[f.path].get("sha256") != f.sha256:
                stale += 1
            else:
                done.append(f)
    finished = {f.path for f in done}
    todo = [f for f in frames if f.path not in finished]
    summary = Summary(requested=len(frames), skipped=len(done), stale=stale)
    summary.verified, summary.max_drift = _verify(
        cache_dir, done, extractor, data_root, verify, seed
    )

    index_path = cache_dir / "index.jsonl"
    tail = index_path.read_bytes() if index_path.exists() else b""
    if tail and not tail.endswith(b"\n"):
        with index_path.open("ab") as handle:  # close off a torn last line
            handle.write(b"\n")
    with index_path.open("a") as handle:
        for n, (frame, prepared) in enumerate(_prepared(extractor, todo, data_root), 1):
            extraction = extractor.extract(prepared)
            array = extraction.array
            if array.dtype != np.dtype(settings.dtype):
                raise CacheError(
                    f"{frame.path}: extractor gave {array.dtype}, the cache "
                    f"settings say {settings.dtype}"
                )
            _save_atomic(_feature_path(cache_dir, frame.path), array)
            entry = {
                "path": frame.path,
                "camera": frame.camera,
                "collection_id": frame.collection_id,
                "image_size": list(extraction.image_size),
                "resized_size": list(extraction.resized_size),
                "patch_grid": list(extraction.patch_grid),
                "pooling": settings.pooling,
                "grid": list(array.shape[:2]),
                "dim": int(array.shape[2]),
                "sha256": frame.sha256,
                "run": run,
                "commit": commit,
            }
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            summary.computed += 1
            if progress:
                progress(n, len(todo), time.perf_counter() - started)
    summary.seconds = time.perf_counter() - started
    return summary


def missing_frames(
    cache_dir: Path, manifests: Path, cameras: Sequence[str] = ("rgb", "thermal")
) -> dict[str, list[str]]:
    """Per camera, the images the manifests reference that the index lacks.

    `cameras` narrows the check, for a cache that holds one camera only (the
    equalised thermal cache, #68)."""
    pinned: Mapping[str, str] = {}
    done = read_index(cache_dir)
    missing: dict[str, list[str]] = {}
    for camera in cameras:
        frames = select_frames(
            manifests / "all_pairs.jsonl", camera, pinned, folds=manifests / "folds"
        )
        missing[camera] = [f.path for f in frames if f.path not in done]
    return missing


SEED = 7  # picks which finished images a resumed run recomputes


def cache_features(
    *,
    model: str,
    camera: str,
    data_root: Path,
    manifests: Path,
    collections: Sequence[str] | None,
    limit: int | None,
    part: tuple[int, int] | None = None,
    token_budget: int,
    pooling: int,
    verify: int,
    scratch: bool,
    session: str | None = None,
    repo: Path | None = None,
    load_extractor: Callable[..., Extractor] | None = None,
    weights: Callable[[str, str], dict[str, str]] | None = None,
    argv: list[str] | None = None,
    input_handling: str = DEFAULT_ARM,
) -> Summary:
    """One session of the feature cache: one camera, optionally some collections.

    The cache is `outputs/features/<model>-<budget>tok/` (`outputs/scratch-features/`
    for a scratch run) and each session is a run under its `runs/<session>/`
    with the usual `run.json`, which records the settings, the weight checksums
    and the data verification. A cache with other settings is refused before
    any run is started. `load_extractor` and `weights` are for tests.
    """
    from aerial_search import run as run_module
    from aerial_search.data import checksums
    from aerial_search.models import backbone
    from aerial_search.models.components import get_device

    if limit and not scratch:
        raise CacheError("--limit builds a partial cache; use it with --scratch only")
    transform = input_transform(input_handling, camera)  # raises for rgb + equalise
    spec = backbone.SPECS[model]
    settings = CacheSettings(
        model=model,
        repo=spec.repo,
        revision=spec.revision,
        weights_sha256=(weights or backbone.weight_checksums)(spec.repo, spec.revision),
        token_budget=token_budget,
        pooling=pooling,
        dtype="float16",
        preprocessing_version=backbone.PREPROCESSING_VERSION,
        input_handling=input_handling,
    )
    top = "scratch-features" if scratch else "features"
    outputs = run_module.outputs_dir(scratch=scratch, repo=repo)
    cache_dir = outputs / top / settings.name()
    check_settings(cache_dir, settings)

    if session is None:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        pieces = [
            camera,
            *(collections or []),
            *([f"part{part[0]}of{part[1]}"] if part else []),
        ]
        session = "-".join([*pieces, stamp])
    with session_lock(cache_dir, session):
        manifest = manifests / "all_pairs.jsonl"
        pinned = {e.path: e.sha256 for e in checksums.committed_list()}
        folds = manifests / "folds"
        frames = select_frames(
            manifest, camera, pinned, collections, limit, part, folds=folds
        )
        config = {
            "settings": asdict(settings),
            "camera": camera,
            "collections": list(collections or []),
            "limit": limit,
            "part": list(part) if part else None,
            "verify": verify,
            "images": len(frames),
            "cache_dir": str(cache_dir),
            "preprocessing": (
                "histogram equalisation of the grey values (PIL ImageOps."
                "equalize), then "
                if input_handling == "equalise"
                else ""
            )
            + "image.convert('RGB'), then the NaFlex processor at "
            "max_num_patches=token_budget; a one-channel thermal frame becomes "
            "three identical channels; features are the last hidden state",
        }
        directory = run_module.start_run(
            f"features/{settings.name()}/runs/{session}",
            config,
            SEED,
            device=str(get_device()),
            scratch=scratch,
            argv=argv,
            repo=repo,
            inputs=[manifest, *fold_manifest_files(folds)],
            manifests=manifests,
            data_root=data_root,
        )
        try:
            if directory.parent.parent.resolve() != cache_dir.resolve():
                raise CacheError(f"run {directory} is not inside the cache {cache_dir}")
            record = json.loads((directory / "run.json").read_text())
            extractor = (load_extractor or _load_naflex(model))(
                token_budget, pooling, transform
            )
            summary = build_cache(
                cache_dir,
                settings,
                frames,
                extractor,
                data_root,
                run=directory.name,
                commit=record.get("commit"),
                verify=verify,
                seed=SEED,
                progress=_print_progress,
            )
            (directory / "summary.json").write_text(
                json.dumps(asdict(summary), indent=2) + "\n"
            )
        except BaseException as error:
            run_module.finish_run(directory, error)
            raise
        run_module.finish_run(directory)
        return summary


def _load_naflex(model: str) -> Callable[..., Extractor]:
    def load(token_budget: int, pooling: int, transform=None) -> Extractor:
        from transformers import AutoImageProcessor

        from aerial_search.models import backbone
        from aerial_search.models.components import get_device

        spec = backbone.SPECS[model]
        return backbone.NaflexExtractor(
            backbone.load(spec, str(get_device())),
            AutoImageProcessor.from_pretrained(spec.repo, revision=spec.revision),
            token_budget,
            pooling,
            transform,
        )

    return load


def _print_progress(done: int, total: int, seconds: float) -> None:
    if done % 200 == 0 or done == total:
        rate = done / seconds if seconds else 0.0
        left = (total - done) / rate / 60 if rate else 0.0
        print(f"{done}/{total} images, {rate:.1f}/s, {left:.1f} min left", flush=True)
