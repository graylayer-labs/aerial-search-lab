"""Leave-one-site-day-out folds over the prepared WiSARD manifests.

The unit that never crosses a split is the site-day: every clip recorded at one
site on one date. Each labelled site-day is the test set of one fold; the
other site-days supply training, validation, and the unlabelled pool. A site-day
with no labels is never a test set and is unlabelled data in every fold. The
design and its evidence are in docs/site-folds-review.md.

Every record carries ``frame``, its time in the clip counted in RGB frame
numbers (a thermal frame number minus the clip's thermal offset), so frames of
the two cameras are compared on one clock.

Layout written under the destination::

    folds.json                      settings, validation sources, counts
    <site-day>/unlabelled.jsonl     pairs for self-supervised learning
    <site-day>/<view>/train_<p>pct.jsonl, validation.jsonl, test.jsonl

Views: ``paired`` (pairs labelled in both cameras), ``rgb``, ``thermal``.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from aerial_search.data.wisard import (
    WISARD_COLLECTIONS,
    Collections,
    _extract_frame_index,
)
from aerial_search.evaluation.detection import SIZE_BUCKETS

VIEWS = ("paired", "rgb", "thermal")
PERCENTS = (1, 5, 10, 100)
# Where a training site-day has one labelled clip in a view, its validation is
# the last 15% of the clip, and the GAP_FRAMES frames before it are used
# nowhere. At 5 frames a second, 250 frames is 50 seconds; by then two frames
# of Baker or Carnation are no more alike than two frames drawn at random from
# the same clip (tools/frame_similarity_lags.py).
VALIDATION_FRACTION = 0.15
GAP_FRAMES = 250
# The smallest gap check_folds accepts, whatever the builder used. Written out,
# not derived from GAP_FRAMES, so a builder whose gap shrinks fails its check.
MIN_GAP_FRAMES = 250
# Label-fraction subsets are made of blocks of this many consecutive labelled
# records of one clip, as an annotator would label a stretch of footage. A
# block is 2 seconds where the labels are unbroken, longer where it spans a
# hole in them.
BLOCK_FRAMES = 10
SEED = 7
SUMMARY = "folds.json"
UNLABELLED = "unlabelled.jsonl"
VALIDATION = "validation.jsonl"
TEST = "test.jsonl"
SOURCES = {
    "paired": "full.jsonl",
    "rgb": "rgb_labelled.jsonl",
    "thermal": "thermal_labelled.jsonl",
}
ALL_PAIRS = "all_pairs.jsonl"
QUALITY = "data_quality.json"


@dataclass(frozen=True)
class LabelOverride:
    """Frames of one clip whose label is not trusted, and why.

    ``rule`` says which frames: "empty_label" is those whose label file exists
    and holds no box. They leave every labelled manifest of ``camera`` (train,
    validation, and test, in every fold) and stay in the unlabelled pool, which
    is built from pairs, so a fold that trains on the site-day pools them.
    """

    collection_id: str
    camera: str
    rule: str
    reason: str


RULES = ("empty_label",)

# Where an empty label file is not evidence of an empty frame. One reviewer found
# unboxed people in about 86% of this clip's empty-label thermal frames (95%
# interval 70.6 to 93.7); labelling appears to stop about 215 frames in.
# docs/label-completeness-review.md, "Thermal clip FHL_0403"; issue #53.
LABEL_OVERRIDES: tuple[LabelOverride, ...] = (
    LabelOverride(
        "210924_FHL_Enterprise_0403",
        "thermal",
        "empty_label",
        "labelling stops about 215 frames in: about 86% of the empty-label "
        "frames show people with no box (#49, 95% interval 70.6 to 93.7)",
    ),
)

ImageSize = Callable[[str], tuple[int, int]]
Record = dict[str, object]

_SITE_DAY = re.compile(r"^(\d{6})_([A-Za-z]+)_")


def site_day(name: str) -> str:
    """Return the date and site at the start of a WiSARD directory name.

    Accepts a directory name, a collection id, or an image path relative to
    the dataset root: ``210924_FHL_Enterprise_VIS_0126/...`` gives
    ``210924_FHL``. Raises ValueError for a name without a date and site.
    """
    first = name.split("/", 1)[0]
    match = _SITE_DAY.match(first)
    if match is None:
        raise ValueError(f"No date and site at the start of {name!r}")
    return f"{match.group(1)}_{match.group(2)}"


def train_file(percent: int) -> str:
    """File name of the training subset holding this percent of the labels."""
    return f"train_{percent}pct.jsonl"


def view_dir(folds: Path, fold: str, view: str) -> Path:
    """Directory of one view of one fold."""
    return folds / fold / view


def image_paths(record: Record) -> list[str]:
    """Every image a manifest record points at."""
    return [
        str(record[key])
        for key in ("image", "rgb_image", "thermal_image")
        if key in record
    ]


def source_state(manifests: Path) -> str:
    """Whether ``manifests`` is "full", a "subset", or "unmarked".

    Unmarked means data_quality.json is missing or has no ``subset`` key, as
    manifests made before the key existed.
    """
    path = manifests / QUALITY
    subset = json.loads(path.read_text()).get("subset") if path.is_file() else None
    if subset is None:
        return "unmarked: no subset key in " + QUALITY
    return "subset" if subset else "full"


def write_folds(
    manifests: Path,
    destination: Path,
    image_size: ImageSize,
    *,
    collections: Collections = WISARD_COLLECTIONS,
    seed: int = SEED,
    overrides: Iterable[LabelOverride] = LABEL_OVERRIDES,
) -> dict[str, object]:
    """Build the folds from the manifests in ``manifests`` and write them.

    Replaces a destination that holds folds written before (it has a
    folds.json); refuses any other non-empty destination. Returns the summary
    also written to folds.json.
    """
    state = source_state(manifests)
    if state != "full":
        raise ValueError(
            f"{manifests} is not a full set of manifests ({state}): {QUALITY} must "
            f"say subset: false. Folds from a subset would look like the full "
            f"folds with smaller counts. Run prepare without --collection."
        )
    files, summary = build_folds(
        manifests, image_size, collections=collections, seed=seed, overrides=overrides
    )
    if destination.exists() and any(destination.iterdir()):
        if not (destination / SUMMARY).is_file():
            raise FileExistsError(
                f"{destination} is not empty and holds no {SUMMARY}; not replacing it"
            )
        shutil.rmtree(destination)
    for relative, records in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_jsonl(records))
    (destination / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@dataclass
class _Split:
    train: list[Record] = field(default_factory=list)
    validation: list[Record] = field(default_factory=list)
    test: list[Record] = field(default_factory=list)


def build_folds(
    manifests: Path,
    image_size: ImageSize,
    *,
    collections: Collections = WISARD_COLLECTIONS,
    seed: int = SEED,
    overrides: Iterable[LabelOverride] = LABEL_OVERRIDES,
) -> tuple[dict[str, list[Record]], dict[str, object]]:
    """Return every fold manifest, keyed by path relative to the folds root,
    and a summary of what each holds.

    Frames named by ``overrides`` are taken out of the labelled manifests of
    their camera after the split, so which clip is held out for validation does
    not change. They stay in the unlabelled pool.
    """
    overrides = list(overrides)
    pairs = [
        _annotate(r, "unlabelled", collections, image_size)
        for r in _read(manifests / ALL_PAIRS)
    ]
    views = {
        view: [
            _annotate(r, view, collections, image_size)
            for r in _read(manifests / source)
        ]
        for view, source in SOURCES.items()
    }
    for override in overrides:
        if override.rule not in RULES or override.camera not in ("rgb", "thermal"):
            raise ValueError(f"Unknown label override: {override}")
    overridden = {
        view: {
            image
            for override in overrides
            if override.camera == view
            for image in _override_images(
                views[view], override, lambda r: str(r["collection_id"])
            )
        }
        for view in ("rgb", "thermal")
    }
    folds = sorted({str(r["site_day"]) for records in views.values() for r in records})
    unlabelled_only = sorted({str(r["site_day"]) for r in pairs} - set(folds))
    cuts = _tail_cuts(pairs)

    files: dict[str, list[Record]] = {}
    fold_summaries: dict[str, object] = {}
    for fold in folds:
        excluded: dict[str, int] = {}  # clip -> first frame used nowhere
        view_summaries: dict[str, object] = {}
        for view in VIEWS:
            split, sources = _split(views[view], fold, cuts, excluded)
            for part in (split.train, split.validation, split.test):
                part[:] = [
                    r for r in part if r.get("image") not in overridden.get(view, ())
                ]
            subsets = _fraction_subsets(split.train, f"{seed}/{fold}/{view}")
            base = f"{fold}/{view}"
            for percent, subset in subsets.items():
                files[f"{base}/{train_file(percent)}"] = subset
            files[f"{base}/{VALIDATION}"] = split.validation
            files[f"{base}/{TEST}"] = split.test
            view_summaries[view] = {
                "validation_sources": sources,
                **{f"train_{p}pct": _describe(s, view) for p, s in subsets.items()},
                "validation": _describe(split.validation, view),
                "test": _describe(split.test, view, buckets=True),
            }
        pool = [
            r
            for r in pairs
            if r["site_day"] != fold
            and int(str(r["frame"])) < excluded.get(str(r["collection_id"]), math.inf)
        ]
        files[f"{fold}/{UNLABELLED}"] = pool
        fold_summaries[fold] = {
            "unlabelled": _describe(pool, "unlabelled"),
            "views": view_summaries,
        }

    summary: dict[str, object] = {
        "seed": seed,
        "views": list(VIEWS),
        "percents": list(PERCENTS),
        "validation_fraction": VALIDATION_FRACTION,
        "gap_frames": GAP_FRAMES,
        "block_frames": BLOCK_FRAMES,
        "folds": folds,
        "never_test": unlabelled_only,
        "label_overrides": [
            {
                "collection_id": o.collection_id,
                "camera": o.camera,
                "rule": o.rule,
                "reason": o.reason,
                "frames": len(
                    _override_images(
                        views[o.camera], o, lambda r: str(r["collection_id"])
                    )
                ),
            }
            for o in overrides
        ],
        "per_fold": fold_summaries,
    }
    return files, summary


def _override_images(
    records: Iterable[dict], override: LabelOverride, clip_of: Callable[[dict], str]
) -> set[str]:
    """Images of ``records`` that ``override`` names, by the rule it states."""
    return {
        str(r["image"])
        for r in records
        if clip_of(r) == override.collection_id and not r["boxes"]
    }


def _split(
    records: list[Record],
    fold: str,
    cuts: dict[str, int],
    excluded: dict[str, int],
) -> tuple[_Split, dict[str, object]]:
    """Split one view's labelled records for one fold.

    Records ``excluded[clip]``, the first frame of each clip that validation
    holds or that the gap before it removes, so the unlabelled pool can leave
    them out too.
    """
    split = _Split()
    by_site: dict[str, dict[str, list[Record]]] = {}
    for record in records:
        if record["site_day"] == fold:
            split.test.append(record)
            continue
        site = by_site.setdefault(str(record["site_day"]), {})
        site.setdefault(str(record["collection_id"]), []).append(record)

    sources: dict[str, object] = {}
    for site in sorted(by_site):
        clips = by_site[site]
        if len(clips) > 1:
            held = min(clips, key=lambda c: (len(clips[c]), c))
            for clip, clip_records in clips.items():
                target = split.validation if clip == held else split.train
                target.extend(clip_records)
            excluded[held] = -1
            sources[site] = {"kind": "whole clip", "clip": held}
            continue
        ((clip, clip_records),) = clips.items()
        cut = cuts[clip]
        split.validation += [r for r in clip_records if _frame(r) >= cut]
        split.train += [r for r in clip_records if _frame(r) < cut - GAP_FRAMES]
        excluded[clip] = min(excluded.get(clip, cut), cut - GAP_FRAMES)
        sources[site] = {
            "kind": "end of clip",
            "clip": clip,
            "validation_from_frame": cut,
            "gap_frames": GAP_FRAMES,
        }
    for part in (split.train, split.validation, split.test):
        part.sort(key=_order)
    return split, sources


def _tail_cuts(pairs: list[Record]) -> dict[str, int]:
    """First frame of the end-of-clip validation block of every clip.

    Set from all pairs of the clip, labelled or not, so it is the same in
    every view.
    """
    frames: dict[str, list[int]] = {}
    for record in pairs:
        frames.setdefault(str(record["collection_id"]), []).append(_frame(record))
    cuts = {}
    for clip, numbers in frames.items():
        numbers.sort()
        cuts[clip] = numbers[int(len(numbers) * (1 - VALIDATION_FRACTION))]
    return cuts


def _fraction_subsets(train: list[Record], seed: str) -> dict[int, list[Record]]:
    """Nested subsets of the training records, made of contiguous blocks.

    Each clip's records, in frame order, are cut into blocks of BLOCK_FRAMES.
    Each site-day's blocks are shuffled with the seed, then the site-days are
    interleaved in proportion to their number of blocks. A subset is the
    shortest prefix of that order holding at least its percent of the
    records, and never less than one block, so smaller subsets are prefixes
    of larger ones.
    """
    blocks: dict[str, list[list[Record]]] = {}
    for clip_records in _group(train, "collection_id").values():
        site = blocks.setdefault(str(clip_records[0]["site_day"]), [])
        for start in range(0, len(clip_records), BLOCK_FRAMES):
            site.append(clip_records[start : start + BLOCK_FRAMES])
    for site, site_blocks in blocks.items():
        random.Random(f"{seed}/{site}").shuffle(site_blocks)

    total = sum(len(b) for b in blocks.values())
    taken = dict.fromkeys(blocks, 0)
    order: list[list[Record]] = []
    for step in range(total):
        open_sites = [s for s in sorted(blocks) if taken[s] < len(blocks[s])]
        site = max(
            open_sites, key=lambda s: len(blocks[s]) / total * (step + 1) - taken[s]
        )
        order.append(blocks[site][taken[site]])
        taken[site] += 1

    subsets: dict[int, list[Record]] = {}
    for percent in PERCENTS:
        target = percent / 100 * len(train)
        chosen: list[Record] = []
        for block in order:
            if chosen and len(chosen) >= target:
                break
            chosen += block
        subsets[percent] = sorted(chosen, key=_order)
    return subsets


def _annotate(
    record: dict, view: str, collections: Collections, image_size: ImageSize
) -> Record:
    """Return a fold record: site-day, clip, clip time, images, sizes, boxes."""
    clip = str(record["collection_id"])
    day = site_day(clip)
    for image in image_paths(record):
        if site_day(image) != day:
            raise ValueError(f"{image} is not from site-day {day} of {clip}")
    if view == "thermal":
        offset = collections[clip].thermal_offset
        frame = _extract_frame_index(Path(record["image"])) - offset
    else:
        frame = _extract_frame_index(Path(record.get("rgb_image", record.get("image"))))
    out: Record = {"site_day": day, "collection_id": clip, "frame": frame}
    if view in {"paired", "unlabelled"}:
        out["rgb_image"] = record["rgb_image"]
        out["thermal_image"] = record["thermal_image"]
    if view == "paired":
        out["rgb_size"] = list(image_size(record["rgb_image"]))
        out["thermal_size"] = list(image_size(record["thermal_image"]))
        out["rgb_boxes"] = record["rgb_boxes"]
        out["thermal_boxes"] = record["thermal_boxes"]
    if view in {"rgb", "thermal"}:
        out["camera"] = view
        out["image"] = record["image"]
        out["size"] = list(image_size(record["image"]))
        out["boxes"] = record["boxes"]
    return out


def _describe(
    records: list[Record], view: str, *, buckets: bool = False
) -> dict[str, object]:
    """Counts for the fold table: site-days, clips, frames, boxes."""
    described: dict[str, object] = {
        "site_days": sorted({str(r["site_day"]) for r in records}),
        "clips": len({r["collection_id"] for r in records}),
        "frames": len(records),
    }
    cameras = ["rgb", "thermal"] if view == "paired" else [view]
    if view == "unlabelled":
        return described
    for camera in cameras:
        prefix = f"{camera}_" if view == "paired" else ""
        boxes = [
            (box, _size(record, f"{prefix}size"))
            for record in records
            for box in _boxes(record, f"{prefix}boxes")
        ]
        described[f"{camera}_boxes"] = len(boxes)
        if buckets:
            described[f"{camera}_size_buckets"] = _bucket_counts(boxes)
    return described


def _bucket_counts(boxes: Iterable[tuple[dict, list[int]]]) -> dict[str, int]:
    counts = dict.fromkeys(SIZE_BUCKETS, 0)
    for box, size in boxes:
        width, height = size
        side = math.sqrt(box["width"] * width * box["height"] * height)
        for name, (low, high) in SIZE_BUCKETS.items():
            if low <= side < high:
                counts[name] += 1
    return counts


def _boxes(record: Record, key: str) -> list[dict]:
    boxes = record[key]
    assert isinstance(boxes, list)
    return boxes


def _size(record: Record, key: str) -> list[int]:
    size = record[key]
    assert isinstance(size, list)
    return size


def _frame(record: Record) -> int:
    return int(str(record["frame"]))


def _order(record: Record) -> tuple[str, int, str]:
    return (str(record["collection_id"]), _frame(record), image_paths(record)[0])


def _group(records: list[Record], key: str) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in sorted(records, key=_order):
        grouped.setdefault(str(record[key]), []).append(record)
    return grouped


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _jsonl(records: Iterable[Record]) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


@dataclass
class CheckReport:
    """What check_folds verified, and every problem it found."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def check_folds(
    manifests: Path,
    folds_dir: Path,
    image_size: ImageSize,
    *,
    collections: Collections = WISARD_COLLECTIONS,
    overrides: Iterable[LabelOverride] = LABEL_OVERRIDES,
) -> CheckReport:
    """Verify the fold manifests on disk, from the files themselves.

    Site-day, clip, and clip time are read again from each image path, using
    only the directory names in ``collections``; no field the builder wrote
    is trusted. Checks that no image of a fold's test site-day is in its
    training, validation, or unlabelled manifests; that the test set holds
    every labelled frame of its site-day; that splits are disjoint; that the
    two images of a pair are one moment of one clip; that label-fraction
    subsets are nested and drawn from training; that training and unlabelled
    frames stay more than MIN_GAP_FRAMES from validation frames of the same
    clip, whatever gap folds.json records; and that the files are rebuilt
    byte for byte from the seed in folds.json. Each label override is applied:
    its frames, found again in the source manifest by clip directory and empty
    label, are in no manifest of its camera and are in the unlabelled pool of
    every fold whose test site-day is another one; folds.json records it.
    """
    overrides = list(overrides)
    report = CheckReport()
    problems = report.problems
    state = source_state(manifests)
    report.lines.append(f"source manifests: {state}")
    if state != "full":
        problems.append(f"source manifests are not full ({state})")
    summary = json.loads((folds_dir / SUMMARY).read_text())
    gap = MIN_GAP_FRAMES
    if int(summary["gap_frames"]) < gap:
        problems.append(
            f"{SUMMARY} records gap_frames {summary['gap_frames']}, "
            f"below the minimum of {gap}"
        )
    where = _image_clips(collections)
    percents = [int(p) for p in summary["percents"]]
    sources = {view: _read(manifests / name) for view, name in SOURCES.items()}
    labelled_days = sorted(
        {site_day(i) for rs in sources.values() for r in rs for i in image_paths(r)}
    )
    all_days = sorted(
        {site_day(i) for r in _read(manifests / ALL_PAIRS) for i in image_paths(r)}
    )
    on_disk = sorted(p.name for p in folds_dir.iterdir() if p.is_dir())
    if not (summary["folds"] == on_disk == labelled_days):
        problems.append(
            f"folds {summary['folds']} on disk {on_disk} "
            f"but labelled site-days are {labelled_days}"
        )
    never = sorted(set(all_days) - set(labelled_days))
    report.lines.append(
        f"folds: {', '.join(on_disk)}; never a test set: {', '.join(never) or 'none'}"
    )

    overridden = {
        o: {
            image_paths(r)[0]
            for r in sources[o.camera]
            if where(image_paths(r)[0])[0] == o.collection_id and not r["boxes"]
        }
        for o in overrides
    }
    recorded = {
        (r["collection_id"], r["camera"], r["rule"]): r["frames"]
        for r in summary.get("label_overrides", [])
    }
    for o, images in overridden.items():
        if recorded.get((o.collection_id, o.camera, o.rule)) != len(images):
            problems.append(
                f"{SUMMARY} does not record overridden {o.camera} frames of "
                f"{o.collection_id}: {len(images)} found in the source manifest"
            )

    checked_images = 0
    closest = math.inf
    for fold in on_disk:
        pool = _read(folds_dir / fold / UNLABELLED)
        pool_images = {i for r in pool for i in image_paths(r)}
        views: dict[str, dict[str, list[dict]]] = {}
        for view in VIEWS:
            directory = view_dir(folds_dir, fold, view)
            names = [train_file(p) for p in percents] + [VALIDATION, TEST]
            views[view] = {name: _read(directory / name) for name in names}
        non_test = {UNLABELLED: pool_images} | {
            f"{view}/{name}": {i for r in records for i in image_paths(r)}
            for view, files in views.items()
            for name, records in files.items()
            if name != TEST
        }
        test_images = {
            i for files in views.values() for r in files[TEST] for i in image_paths(r)
        }
        for name, images in non_test.items():
            checked_images += len(images)
            days = {site_day(i) for i in images}
            if fold in days:
                problems.append(f"{fold}/{name} holds images of test site-day {fold}")
            if images & test_images:
                problems.append(f"{fold}/{name} shares images with the test set")
        for o, images in overridden.items():
            held_in = {
                name: {i for r in records for i in image_paths(r)} & images
                for name, records in views[o.camera].items()
            }
            for name, found in held_in.items():
                if found:
                    problems.append(
                        f"{fold}/{o.camera}/{name} holds {len(found)} overridden "
                        f"frames of {o.collection_id}"
                    )
            if site_day(o.collection_id) != fold:
                in_pool = {str(r[f"{o.camera}_image"]) for r in pool} & images
                if in_pool != images:
                    problems.append(
                        f"{fold}/{UNLABELLED} lacks {len(images - in_pool)} "
                        f"overridden {o.camera} frames of {o.collection_id} "
                        f"from the pool"
                    )
        for view, files in views.items():
            test = files[TEST]
            other_days = {site_day(i) for r in test for i in image_paths(r)} - {fold}
            if other_days:
                problems.append(f"{fold}/{view}/{TEST} holds {sorted(other_days)}")
            skipped = {
                i
                for o, images in overridden.items()
                if o.camera == view
                for i in images
            }
            expected = sorted(
                image_paths(r)[0]
                for r in sources[view]
                if site_day(image_paths(r)[0]) == fold
                and image_paths(r)[0] not in skipped
            )
            if sorted(image_paths(r)[0] for r in test) != expected:
                problems.append(
                    f"{fold}/{view}/{TEST} does not hold every labelled frame of {fold}"
                )
            if _keys(files[train_file(percents[-1])]) & _keys(files[VALIDATION]):
                problems.append(f"{fold}/{view}: training and validation overlap")
            for smaller, larger in zip(percents, percents[1:], strict=False):
                a = {json.dumps(r) for r in files[train_file(smaller)]}
                b = {json.dumps(r) for r in files[train_file(larger)]}
                if not a <= b:
                    problems.append(
                        f"{fold}/{view}: {train_file(smaller)} is not nested "
                        f"in {train_file(larger)}"
                    )
            for name, records in files.items():
                for record in records:
                    if len({where(i) for i in image_paths(record)}) > 1:
                        problems.append(
                            f"{fold}/{view}/{name}: the images of one record are "
                            f"not one moment of one clip: {image_paths(record)}"
                        )
            held = _clip_times(files[VALIDATION], where)
            near = _clip_times(files[train_file(percents[-1])], where)
            for record_set, label in ((near, "training"), (None, UNLABELLED)):
                others = (
                    record_set if record_set is not None else _clip_times(pool, where)
                )
                for clip, frames in held.items():
                    distance = _min_distance(frames, others.get(clip, []))
                    closest = min(closest, distance)
                    if distance <= gap:
                        problems.append(
                            f"{fold}/{view}: {label} frame {distance} frames from "
                            f"validation in {clip}, inside the gap of {gap}"
                        )

    rebuilt, rebuilt_summary = build_folds(
        manifests,
        image_size,
        collections=collections,
        seed=int(summary["seed"]),
        overrides=overrides,
    )
    on_disk_files = {str(p.relative_to(folds_dir)) for p in folds_dir.rglob("*.jsonl")}
    if on_disk_files != set(rebuilt):
        mismatch = sorted(on_disk_files ^ set(rebuilt))
        problems.append(f"files on disk differ from a rebuild: {mismatch}")
    differ = sorted(
        name
        for name, records in rebuilt.items()
        if name in on_disk_files and (folds_dir / name).read_text() != _jsonl(records)
    )
    if differ:
        problems.append(f"not reproduced from seed {summary['seed']}: {differ}")
    if rebuilt_summary != summary:
        problems.append(f"{SUMMARY} differs from a rebuild")

    report.lines += [
        f"no image of a test site-day in training, validation, or unlabelled "
        f"manifests ({checked_images:,} image references checked)",
        "every test set holds every labelled frame of its site-day",
        "training, validation, and test are disjoint in every view",
        "label fractions nested: "
        + " in ".join(f"{p}%" for p in percents)
        + ", all drawn from training",
        f"training and unlabelled frames are more than {gap} frames from "
        f"validation in the same clip (closest: {closest} frames)",
        *[
            f"override {o.collection_id} {o.camera} {o.rule}: {len(images):,} frames "
            f"in no {o.camera} manifest, in the pool of every other fold"
            for o, images in overridden.items()
        ],
        f"{len(rebuilt)} manifests rebuilt byte for byte from seed {summary['seed']}",
    ]
    return report


def _keys(records: list[dict]) -> set[str]:
    return {i for r in records for i in image_paths(r)}


def _image_clips(collections: Collections) -> Callable[[str], tuple[str, int]]:
    """Return a reader of (clip, clip time) from an image path alone.

    The clip comes from the image's directory, looked up among the listed
    directories; the time is the frame number in the file name, less the
    thermal offset for a thermal directory. Raises ValueError for an image
    in a directory that is not listed.
    """
    directories: dict[str, tuple[str, int]] = {}
    for clip, collection in collections.items():
        directories[collection.rgb_dir] = (clip, 0)
        directories[collection.thermal_dir] = (clip, collection.thermal_offset)

    def where(image: str) -> tuple[str, int]:
        directory = image.split("/", 1)[0]
        if directory not in directories:
            raise ValueError(f"{image} is not in a listed clip directory")
        clip, offset = directories[directory]
        return clip, _extract_frame_index(Path(image)) - offset

    return where


def _clip_times(
    records: list[dict], where: Callable[[str], tuple[str, int]]
) -> dict[str, list[int]]:
    """Clip time of every image of the records, grouped by clip, both read
    from the image paths."""
    times: dict[str, list[int]] = {}
    for record in records:
        for image in image_paths(record):
            clip, frame = where(image)
            times.setdefault(clip, []).append(frame)
    for frames in times.values():
        frames.sort()
    return times


def _min_distance(held: list[int], others: list[int]) -> float:
    """Smallest distance between a frame in held and one in others (sorted)."""
    best = math.inf
    for frame in held:
        index = bisect.bisect_left(others, frame)
        for neighbour in others[max(0, index - 1) : index + 1]:
            best = min(best, abs(frame - neighbour))
    return best


def fold_table(summary: dict) -> str:
    """Markdown tables of what each fold holds, from a folds.json summary."""
    per_fold = summary["per_fold"]
    lines = []
    for view in summary["views"]:
        cameras = ["rgb", "thermal"] if view == "paired" else [view]
        boxes = " | ".join(
            f"{c.upper() if c == 'rgb' else c.title()} boxes" for c in cameras
        )
        lines += [
            f"### View: {view}",
            "",
            f"| Fold (test site-day) | Split | Site-days | Clips | Frames | {boxes} |",
            "|---|---|---|---|---|" + "---|" * len(cameras),
        ]
        for fold in summary["folds"]:
            parts = per_fold[fold]["views"][view]
            for split in ("train_100pct", "validation", "test"):
                part = parts[split]
                counts = " | ".join(f"{part[f'{c}_boxes']:,}" for c in cameras)
                lines.append(
                    f"| {_short(fold)} | {split.removesuffix('_100pct')} | "
                    f"{', '.join(_short(s) for s in part['site_days'])} | "
                    f"{part['clips']} | {part['frames']:,} | {counts} |"
                )
        lines += ["", "Validation source per training site-day:", ""]
        for fold in summary["folds"]:
            sources = per_fold[fold]["views"][view]["validation_sources"]
            described = "; ".join(
                f"{_short(site)}: {_source(source)}" for site, source in sources.items()
            )
            lines.append(f"- fold {_short(fold)}: {described}")
        buckets = list(SIZE_BUCKETS)
        lines += [
            "",
            "Test boxes by size (side of the box in pixels, "
            + ", ".join(f"{b} [{lo:g}, {hi:g})" for b, (lo, hi) in SIZE_BUCKETS.items())
            + "):",
            "",
            "| Fold | Camera | " + " | ".join(buckets) + " |",
            "|---|---|" + "---|" * len(buckets),
        ]
        for fold in summary["folds"]:
            test = per_fold[fold]["views"][view]["test"]
            for camera in cameras:
                counts = test[f"{camera}_size_buckets"]
                lines.append(
                    f"| {_short(fold)} | {camera} | "
                    + " | ".join(f"{counts[b]:,}" for b in buckets)
                    + " |"
                )
        percents = summary["percents"]
        lines += [
            "",
            "Label fractions, frames / "
            + " + ".join(f"{c} boxes" for c in cameras)
            + " / site-days:",
            "",
            "| Fold | " + " | ".join(f"{p}%" for p in percents) + " |",
            "|---|" + "---|" * len(percents),
        ]
        for fold in summary["folds"]:
            cells = []
            for percent in percents:
                part = per_fold[fold]["views"][view][f"train_{percent}pct"]
                box_counts = " + ".join(f"{part[f'{c}_boxes']:,}" for c in cameras)
                cells.append(
                    f"{part['frames']:,} / {box_counts} / {len(part['site_days'])}"
                )
            lines.append(f"| {_short(fold)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "### Unlabelled pool",
        "",
        "| Fold | Site-days | Clips | Pairs |",
        "|---|---|---|---|",
    ]
    for fold in summary["folds"]:
        pool = per_fold[fold]["unlabelled"]
        lines.append(
            f"| {_short(fold)} | {', '.join(_short(s) for s in pool['site_days'])} | "
            f"{pool['clips']} | {pool['frames']:,} |"
        )
    return "\n".join(lines) + "\n"


def _short(site: str) -> str:
    return site.split("_", 1)[1]


def _source(source: dict) -> str:
    clip = str(source["clip"]).rsplit("_", 1)[1]
    if source["kind"] == "whole clip":
        return f"whole clip {clip}"
    return (
        f"clip {clip} from frame {source['validation_from_frame']}, "
        f"{source['gap_frames']} frames dropped before it"
    )
