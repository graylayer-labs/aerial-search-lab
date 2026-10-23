"""Load synchronized RGB and thermal samples from WiSARD."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from aerial_search.data import checksums


@dataclass(frozen=True)
class ImagePair:
    """Paths for one synchronized RGB–thermal observation."""

    collection_id: str
    rgb_image: Path
    thermal_image: Path
    rgb_labels: Path
    thermal_labels: Path


@dataclass(frozen=True)
class BoundingBox:
    """A normalized YOLO person bounding box."""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Collection:
    """One VIS clip and the IR clip recorded with it.

    rgb_clip and thermal_clip are the DJI clip numbers that every file name in
    the directory carries. thermal_offset is added to an RGB frame number to
    get the thermal frame number of the same moment; it is set from where each
    directory's numbering starts, never fitted.
    """

    rgb_dir: str
    thermal_dir: str
    rgb_clip: str
    thermal_clip: str
    thermal_offset: int = 0


class CollectionCounts(TypedDict):
    """How the frames of one collection were paired."""

    rgb_dir: str
    thermal_dir: str
    thermal_offset: int
    rgb_frames: int
    thermal_frames: int
    pairs: int
    labelled_pairs: int
    rgb_only_labelled_pairs: int
    thermal_only_labelled_pairs: int
    unlabelled_pairs: int
    rgb_only_frames: int
    thermal_only_frames: int


class PairingReport(TypedDict):
    """How the image directories under a dataset root were paired."""

    collections: dict[str, CollectionCounts]
    # Directories of listed collections that were not selected for this run.
    unselected_directories: list[str]
    # Directories that no listed collection uses.
    unpaired_directories: list[str]


Collections = Mapping[str, Collection]


def _enterprise(flight: str, vis: int, *, thermal_offset: int = 0) -> Collection:
    """A Mavic 2 Enterprise clip pair: VIS is DJI clip n, IR is clip n+1."""
    return Collection(
        f"{flight}_VIS_{vis:04d}",
        f"{flight}_IR_{vis + 1:04d}",
        f"{vis:04d}",
        f"{vis + 1:04d}",
        thermal_offset,
    )


# VIS and IR directories that were recorded together, keyed by collection id.
#
# Only these directories are paired. Every entry was checked against the images:
# camera motion between frames agrees across the two directories to within two
# frames, and contact sheets of start, middle, and end pairs show the same
# scene. See docs/wisard-pairing-review.md.
#
# The Mavic 2 Enterprise Advanced writes the VIS and IR video of one recording
# as consecutive DJI clip numbers, so VIS n goes with IR n+1. Directories not
# listed here are not paired, and are reported as unpaired. In particular the
# 210327 Airfield FLIR directories are left out: their VIS and IR frame numbers
# run at different rates, so equal numbers are not the same moment.
WISARD_COLLECTIONS: Collections = {
    # VIS frames are numbered from 0 and IR frames from 1, 264 of each.
    "210417_MtErie_Enterprise_0003": _enterprise(
        "210417_MtErie_Enterprise", 3, thermal_offset=1
    ),
    "210417_MtErie_Enterprise_0005": _enterprise("210417_MtErie_Enterprise", 5),
    "210417_MtErie_Enterprise_0007": _enterprise("210417_MtErie_Enterprise", 7),
    "210529_Carnation_Enterprise_0023": _enterprise("210529_Carnation_Enterprise", 23),
    "210529_Carnation_Enterprise_0025": _enterprise("210529_Carnation_Enterprise", 25),
    "210812_Hannegan_Enterprise_0053": _enterprise("210812_Hannegan_Enterprise", 53),
    "210812_Hannegan_Enterprise_0055": _enterprise("210812_Hannegan_Enterprise", 55),
    **{
        f"210924_FHL_Enterprise_{vis:04d}": _enterprise("210924_FHL_Enterprise", vis)
        for vis in (126, 134, 401, 403, 405, 407, 409, 564, 566)
    },
    # Directories named _1, but the files are DJI_0582 (VIS) and DJI_0583 (IR).
    # IR_2 (DJI_0585) has no VIS.
    "220109_Baker_Enterprise_1": Collection(
        "220109_Baker_Enterprise_VIS_1", "220109_Baker_Enterprise_IR_1", "0582", "0583"
    ),
}

# A listed pair must share at least this fraction of the frame numbers of each
# directory, after the offset. This catches two directories whose numbering
# does not line up: a wrong or missing offset, or clips at different frame
# rates, such as Airfield VIS_4 (572 frames) with IR_4 (1,061). It does not
# catch two different clips of similar length numbered from 0, which is most
# wrong pairings from one flight day. Those are caught by the rule that the IR
# clip number is the VIS clip number plus one, which
# test_wisard_collections_follow_the_clip_numbering enforces. The check that
# file names carry their directory's clip number only proves that the files
# belong to their directory.
MIN_SHARED_FRAMES = 0.95

# WiSARD labels are written to six decimals, so a box drawn to the image edge
# can overshoot it by up to 1e-6. That is clipped but not counted as clipping.
_EDGE_TOLERANCE = 1e-6

_FRAME_NUMBER = re.compile(r"[_ ](\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def select_collections(ids: Iterable[str]) -> dict[str, Collection]:
    """Return the named WiSARD collections, to prepare a deliberate subset."""
    ids = list(ids)
    if not ids:
        raise ValueError("No collections selected")
    unknown = [cid for cid in ids if cid not in WISARD_COLLECTIONS]
    if unknown:
        raise KeyError(f"Unknown WiSARD collections: {', '.join(unknown)}")
    return {cid: WISARD_COLLECTIONS[cid] for cid in ids}


def _extract_frame_index(path: Path) -> int:
    """Return the frame number at the end of a WiSARD image filename.

    The number follows the last underscore or space, and may have any number of
    digits:
    - 210327_Airfield_FLIR_VIS_1_00000075.jpg → 75
    - DJI_0402.mp4_00000.jpg → 0
    - 20200929_134258_IR 127.jpg → 127

    Raises ValueError if the name has no frame number, so that a file is never
    paired on a guess.
    """
    return _split_frame_number(path)[1]


def _split_frame_number(path: Path) -> tuple[str, int]:
    """Return the part of the name before the frame number, and the number."""
    match = _FRAME_NUMBER.search(path.name)
    if match is None:
        raise ValueError(f"No frame number in image filename: {path}")
    return path.name[: match.start()], int(match.group(1))


def _frames_by_index(directory: Path, clip: str | None = None) -> dict[int, Path]:
    """Map frame number to image for every JPEG in a directory.

    Raises ValueError if two images share a frame number, or, when clip is
    given, if an image name does not carry that clip number.
    """
    clip_token = re.compile(rf"(?<!\d){re.escape(clip)}(?!\d)") if clip else None
    frames: dict[int, Path] = {}
    for image in sorted(directory.iterdir()):
        if image.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        prefix, index = _split_frame_number(image)
        if clip_token is not None and not clip_token.search(prefix):
            raise ValueError(
                f"{image.name} in {directory} does not carry clip number {clip}"
            )
        if index in frames:
            raise ValueError(
                f"Duplicate frame {index} in {directory}: "
                f"{frames[index].name} and {image.name}"
            )
        frames[index] = image
    return frames


def _group_collections(
    root: Path, collections: Collections = WISARD_COLLECTIONS
) -> dict[str, tuple[Path, Path]]:
    """Return the directories of every listed collection.

    Raises FileNotFoundError if any listed directory is missing. To prepare a
    subset, pass only those collections (see select_collections).
    """
    grouped = {}
    missing = []
    for cid, collection in collections.items():
        rgb_dir = root / collection.rgb_dir
        thermal_dir = root / collection.thermal_dir
        missing += [str(d) for d in (rgb_dir, thermal_dir) if not d.is_dir()]
        grouped[cid] = (rgb_dir, thermal_dir)
    if missing:
        raise FileNotFoundError(
            "Listed collection directories are missing: " + ", ".join(missing)
        )
    return grouped


def _pair_collection(
    root: Path, collection_id: str, collection: Collection
) -> tuple[list[ImagePair], CollectionCounts]:
    """Pair the images of one collection by frame number.

    RGB frame i pairs with thermal frame i + thermal_offset. Returns every
    pair, labelled or not, and counts of frames, pairs, labels, and frames
    that have no partner.
    """
    rgb_dir = root / collection.rgb_dir
    thermal_dir = root / collection.thermal_dir
    rgb_by_frame = _frames_by_index(rgb_dir, collection.rgb_clip)
    thermal_by_frame = _frames_by_index(thermal_dir, collection.thermal_clip)
    if not rgb_by_frame or not thermal_by_frame:
        raise ValueError(
            f"Expected non-empty RGB and thermal collections for "
            f"{collection_id}, got {len(rgb_by_frame)} RGB and "
            f"{len(thermal_by_frame)} thermal"
        )

    offset = collection.thermal_offset
    shared = sorted(i for i in rgb_by_frame if i + offset in thermal_by_frame)
    larger = max(len(rgb_by_frame), len(thermal_by_frame))
    if len(shared) < MIN_SHARED_FRAMES * larger:
        raise ValueError(
            f"{collection_id}: only {len(shared)} of {larger} frames of "
            f"{rgb_dir.name} and {thermal_dir.name} line up at offset {offset}; "
            f"the directories do not share a frame numbering"
        )

    pairs = []
    for frame in shared:
        rgb_image = rgb_by_frame[frame]
        thermal_image = thermal_by_frame[frame + offset]
        pairs.append(
            ImagePair(
                collection_id,
                rgb_image,
                thermal_image,
                rgb_image.with_suffix(".txt"),
                thermal_image.with_suffix(".txt"),
            )
        )
    rgb_labelled = [p.rgb_labels.exists() for p in pairs]
    thermal_labelled = [p.thermal_labels.exists() for p in pairs]
    labels = list(zip(rgb_labelled, thermal_labelled, strict=True))
    counts: CollectionCounts = {
        "rgb_dir": rgb_dir.name,
        "thermal_dir": thermal_dir.name,
        "thermal_offset": offset,
        "rgb_frames": len(rgb_by_frame),
        "thermal_frames": len(thermal_by_frame),
        "pairs": len(pairs),
        "labelled_pairs": labels.count((True, True)),
        "rgb_only_labelled_pairs": labels.count((True, False)),
        "thermal_only_labelled_pairs": labels.count((False, True)),
        "unlabelled_pairs": labels.count((False, False)),
        "rgb_only_frames": len(rgb_by_frame) - len(shared),
        "thermal_only_frames": len(thermal_by_frame) - len(shared),
    }
    return pairs, counts


def _is_labelled(pair: ImagePair) -> bool:
    return pair.rgb_labels.exists() and pair.thermal_labels.exists()


def load_pairs(
    root: Path,
    stats: dict[str, int] | None = None,
    *,
    labelled_only: bool = True,
    collections: Collections = WISARD_COLLECTIONS,
) -> list[ImagePair]:
    """Return synchronized pairs from the listed collections under root.

    With labelled_only, pairs without a label file in both modalities are left
    out, and counted in stats["frames_skipped"] if stats is given.
    """
    _group_collections(root, collections)
    pairs: list[ImagePair] = []
    for cid, collection in collections.items():
        for pair in _pair_collection(root, cid, collection)[0]:
            if labelled_only and not _is_labelled(pair):
                if stats is not None:
                    stats["frames_skipped"] = stats.get("frames_skipped", 0) + 1
                continue
            pairs.append(pair)
    return pairs


def pairing_report(
    root: Path, collections: Collections = WISARD_COLLECTIONS
) -> PairingReport:
    """Describe how the image directories under root were paired.

    Gives counts per collection, the directories of listed collections that
    were not selected, and every image directory no listed collection uses.
    Raises FileNotFoundError if a listed directory is missing.
    """
    grouped = _group_collections(root, collections)
    per_collection = {
        cid: _pair_collection(root, cid, collection)[1]
        for cid, collection in collections.items()
    }
    used = {path.name for dirs in grouped.values() for path in dirs}
    unselected = {
        name
        for cid, c in WISARD_COLLECTIONS.items()
        if cid not in collections
        for name in (c.rgb_dir, c.thermal_dir)
    } - used
    unselected_present, unpaired = [], []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in used or not any(path.glob("*.jp*g")):
            continue
        (unselected_present if path.name in unselected else unpaired).append(path.name)
    return {
        "collections": per_collection,
        "unselected_directories": unselected_present,
        "unpaired_directories": unpaired,
    }


def load_boxes(path: Path, stats: dict[str, int] | None = None) -> list[BoundingBox]:
    """Load normalized person boxes from a WiSARD annotation file.

    Each box is clipped to the image. A box with no area left after clipping
    is dropped. If stats is given, 'boxes_clipped' counts boxes that ran more
    than _EDGE_TOLERANCE past the image and 'boxes_dropped' counts the dropped ones.
    """
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Invalid person annotation at {path}:{line_number}")
        x, y, w, h = (float(v) for v in fields[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Invalid person annotation at {path}:{line_number}")
        x0, x1 = max(0.0, x - w / 2), min(1.0, x + w / 2)
        y0, y1 = max(0.0, y - h / 2), min(1.0, y + h / 2)
        overshoot = max(-(x - w / 2), x + w / 2 - 1, -(y - h / 2), y + h / 2 - 1)
        clipped = overshoot > _EDGE_TOLERANCE
        empty = x1 <= x0 or y1 <= y0
        if stats is not None:
            stats["boxes_clipped"] = stats.get("boxes_clipped", 0) + clipped
            stats["boxes_dropped"] = stats.get("boxes_dropped", 0) + empty
        if not empty:
            boxes.append(BoundingBox((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return boxes


def load_pairs_by_collection(
    root: Path,
    stats: dict[str, int] | None = None,
    *,
    collections: Collections = WISARD_COLLECTIONS,
) -> dict[str, list[ImagePair]]:
    """Return labelled pairs grouped by collection_id."""
    grouped: dict[str, list[ImagePair]] = {}
    for pair in load_pairs(root, stats=stats, collections=collections):
        grouped.setdefault(pair.collection_id, []).append(pair)
    return grouped


class _Labels:
    """Loads each label file once and counts clipped boxes per collection."""

    def __init__(self, collection_ids: Iterable[str]) -> None:
        self.boxes: dict[Path, list[dict[str, float]]] = {}
        self.dropped: dict[Path, int] = {}
        # A clip with no labels still gets zeros, so it is not missing.
        self.stats: dict[str, dict[str, int]] = {
            cid: {
                "rgb_clipped": 0,
                "rgb_dropped": 0,
                "thermal_clipped": 0,
                "thermal_dropped": 0,
            }
            for cid in collection_ids
        }

    def load(self, path: Path, collection_id: str, camera: str) -> list[dict]:
        if path not in self.boxes:
            counts: dict[str, int] = {}
            self.boxes[path] = [asdict(box) for box in load_boxes(path, counts)]
            self.dropped[path] = counts.get("boxes_dropped", 0)
            clip_stats = self.stats[collection_id]
            clip_stats[f"{camera}_clipped"] += counts.get("boxes_clipped", 0)
            clip_stats[f"{camera}_dropped"] += self.dropped[path]
        return self.boxes[path]


def prepare_manifests(
    source: Path,
    destination: Path,
    *,
    collections: Collections = WISARD_COLLECTIONS,
    provenance: Mapping[str, object] | None = None,
    verify: bool = True,
    pinned: Sequence[checksums.Entry] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Write the all-pairs manifest and the labelled manifests.

    Writes:
    - all_pairs.jsonl: every pair, labelled or not (for SSL)
    - full.jsonl: pairs labelled in both cameras
    - rgb_labelled.jsonl, thermal_labelled.jsonl: every frame of a listed
      directory that has a label file for that camera, whether or not its
      partner has one (for detection on one camera); each carries
      boxes_dropped, the boxes of its label file that were dropped
    - data_quality.json: whether this is a subset and which collections it
      holds, counts per collection, clipped boxes per collection, and the
      directories that were left out or not paired, `provenance` (the commit
      and scratch flag the caller passes), and `data`: whether the directories
      of the selected collections were verified against the pinned checksum
      list, and its root hash if so

    Splits are not made here: they are by site-day, in folds/ (see
    aerial_search.data.folds). Split files left by an older version are
    removed, so nothing reads them by mistake. Returns the number of records
    in each manifest.

    With `verify` (the default), the directories of the selected collections
    are hashed and checked against `pinned` (default: the committed list)
    before anything is written; a difference raises DataMismatchError naming
    the files. A source none of whose directories are in the list is not a
    pinned dataset: it is recorded as such, with no root hash. With
    `verify=False` (scratch runs) nothing is checked and `verified` is false.
    A scratch `provenance` refuses to overwrite manifests that are not
    scratch, unless `force`.

    Raises FileNotFoundError if a listed directory is missing, and ValueError
    if no collections are given or a subset would replace full manifests.
    """
    if not collections:
        raise ValueError("No collections to prepare")
    subset = set(collections) != set(WISARD_COLLECTIONS)
    if subset:
        _refuse_to_replace_full_manifests(destination)
    if (provenance or {}).get("scratch") is True and not force:
        _refuse_to_replace_quotable_manifests(destination)
    report = pairing_report(source, collections)
    data = _verify_source(source, collections, verify, pinned)
    destination.mkdir(parents=True, exist_ok=True)
    for old_split in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (destination / old_split).unlink(missing_ok=True)
    labels = _Labels(collections)

    all_pairs_unlabeled = load_pairs(
        source, labelled_only=False, collections=collections
    )
    _write_jsonl(
        destination / "all_pairs.jsonl",
        (
            {
                "collection_id": pair.collection_id,
                "rgb_image": str(pair.rgb_image.relative_to(source)),
                "thermal_image": str(pair.thermal_image.relative_to(source)),
            }
            for pair in all_pairs_unlabeled
        ),
    )
    print(f"Wrote {len(all_pairs_unlabeled):,} pairs to all_pairs.jsonl")

    # Pairs labelled in both cameras.
    records = {
        pair: _pair_record(pair, source, labels)
        for pair in all_pairs_unlabeled
        if _is_labelled(pair)
    }
    _write_jsonl(destination / "full.jsonl", records.values())
    print(f"Wrote {len(records):,} labelled pairs to full.jsonl")
    counts = {"all_pairs": len(all_pairs_unlabeled), "full": len(records)}

    for camera in ("rgb", "thermal"):
        frames = _camera_records(source, collections, camera, labels)
        _write_jsonl(destination / f"{camera}_labelled.jsonl", frames)
        print(f"Wrote {len(frames):,} labelled frames to {camera}_labelled.jsonl")
        counts[f"{camera}_labelled"] = len(frames)

    box_stats = labels.stats
    with (destination / "data_quality.json").open("w") as f:
        json.dump(
            {
                "subset": subset,
                "selected_collections": list(collections),
                "total_pairs": len(all_pairs_unlabeled),
                "labeled_pairs": len(records),
                "frames_skipped": len(all_pairs_unlabeled) - len(records),
                "boxes_clipped": sum(
                    s["rgb_clipped"] + s["thermal_clipped"] for s in box_stats.values()
                ),
                "boxes_dropped": sum(
                    s["rgb_dropped"] + s["thermal_dropped"] for s in box_stats.values()
                ),
                "boxes": box_stats,
                "provenance": dict(provenance) if provenance else None,
                "data": data,
                **report,
            },
            f,
            indent=2,
        )

    return counts


def _verify_source(
    source: Path,
    collections: Collections,
    verify: bool,
    pinned: Sequence[checksums.Entry] | None,
) -> dict[str, object]:
    """Check the selected collections' directories against the pinned list."""
    unverified: dict[str, object] = {
        "dataset": None,
        "root_hash": None,
        "verified": False,
    }
    if not verify:
        return unverified
    entries = list(pinned) if pinned is not None else checksums.committed_list()
    directories = sorted(
        {d for c in collections.values() for d in (c.rgb_dir, c.thermal_dir)}
    )
    listed = {e.path.split("/")[0] for e in entries}
    if not listed & set(directories):
        return unverified
    started = time.monotonic()
    difference = checksums.verify_directories(
        source, directories, entries, checksums.print_progress
    )
    if difference:
        raise checksums.DataMismatchError(
            f"{source} differs from the pinned list; not writing manifests:\n"
            + "\n".join(difference.lines())
        )
    return {
        "dataset": checksums.DATASET,
        "root_hash": checksums.root_hash(entries),
        "verified": True,
        "directories": directories,
        "seconds": round(time.monotonic() - started, 1),
    }


def _refuse_to_replace_quotable_manifests(destination: Path) -> None:
    """Raise if destination holds manifests that are not marked scratch."""
    quality = destination / "data_quality.json"
    if quality.exists():
        try:
            scratch = json.loads(quality.read_text())["provenance"]["scratch"] is True
        except (ValueError, KeyError, TypeError):
            scratch = False
    else:
        scratch = not any(
            (destination / name).exists() for name in ("all_pairs.jsonl", "full.jsonl")
        )
    if not scratch:
        raise ValueError(
            f"{destination} holds manifests that are not marked scratch; a scratch "
            "prepare would overwrite what a run may quote. Use another --output, "
            "or pass --force."
        )


def _refuse_to_replace_full_manifests(destination: Path) -> None:
    """Raise if destination holds manifests that are not marked as a subset."""
    quality = destination / "data_quality.json"
    if quality.exists():
        marked = json.loads(quality.read_text()).get("subset") is True
    else:
        marked = not any(
            (destination / name).exists() for name in ("all_pairs.jsonl", "full.jsonl")
        )
    if not marked:
        raise ValueError(
            f"{destination} holds full manifests, or manifests that do not say "
            f"what they are; not writing a subset over them. Use another --output."
        )


def _camera_records(
    source: Path, collections: Collections, camera: str, labels: _Labels
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for cid, collection in collections.items():
        if camera == "rgb":
            directory, clip = collection.rgb_dir, collection.rgb_clip
        else:
            directory, clip = collection.thermal_dir, collection.thermal_clip
        for image in _frames_by_index(source / directory, clip).values():
            label_file = image.with_suffix(".txt")
            if label_file.exists():
                frames.append(
                    {
                        "collection_id": cid,
                        "image": str(image.relative_to(source)),
                        "boxes": labels.load(label_file, cid, camera),
                        # Boxes with no area left after clipping, so that a
                        # label file that lost all its boxes is not read as an
                        # empty one.
                        "boxes_dropped": labels.dropped[label_file],
                    }
                )
    return frames


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def _pair_record(pair: ImagePair, source: Path, labels: _Labels) -> dict[str, object]:
    cid = pair.collection_id
    return {
        "collection_id": cid,
        "rgb_image": str(pair.rgb_image.relative_to(source)),
        "thermal_image": str(pair.thermal_image.relative_to(source)),
        "rgb_boxes": labels.load(pair.rgb_labels, cid, "rgb"),
        "thermal_boxes": labels.load(pair.thermal_labels, cid, "thermal"),
    }
