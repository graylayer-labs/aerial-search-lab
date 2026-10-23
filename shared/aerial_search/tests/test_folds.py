"""Leave-one-site-day-out folds: nothing of a test site-day reaches training."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from aerial_search.data import folds as folds_module
from aerial_search.data.folds import (
    BLOCK_FRAMES,
    PERCENTS,
    SUMMARY,
    UNLABELLED,
    VIEWS,
    LabelOverride,
    check_folds,
    site_day,
    train_file,
    write_folds,
)
from aerial_search.data.wisard import WISARD_COLLECTIONS, Collection

# A dataset in the shape of WiSARD, small enough to build in a test. Clip
# lengths are in frames; "labels" says which cameras are labelled.
CLIPS = [
    # site-day, VIS clip number, frames, labels, thermal offset
    ("240101_Alpha", 1, 300, "both", 1),
    ("240101_Alpha", 3, 200, "both", 0),
    ("240101_Alpha", 5, 120, "both", 0),
    ("240202_Bravo", 11, 1000, "both", 0),
    ("240303_Charlie", 21, 400, "both", 0),
    ("240303_Charlie", 23, 600, "rgb", 0),
    ("240404_Delta", 31, 300, "none", 0),
]
# The gap required between training and validation in one clip, written out
# here rather than imported, so a builder whose gap shrinks fails these tests.
MIN_GAP = 250


def _directories() -> dict[str, tuple[str, int]]:
    """Image directory -> (clip, thermal offset), from CLIPS alone."""
    found = {}
    for site, vis, _, _, offset in CLIPS:
        flight = f"{site}_Enterprise"
        clip = f"{flight}_{vis:04d}"
        found[f"{flight}_VIS_{vis:04d}"] = (clip, 0)
        found[f"{flight}_IR_{vis + 1:04d}"] = (clip, offset)
    return found


DIRECTORIES = _directories()


def _where(image: str) -> tuple[str, int]:
    """Clip and clip time of an image, read from its path, not the record."""
    directory, name = image.split("/")
    clip, offset = DIRECTORIES[directory]
    match = re.search(r"_(\d+)\.jpeg$", name)
    assert match, image
    return clip, int(match.group(1)) - offset


def _size(path: str) -> tuple[int, int]:
    return (640, 512) if "_IR_" in path else (1920, 1080)


def _dataset(root: Path) -> tuple[Path, dict[str, Collection]]:
    manifests = root / "manifests"
    manifests.mkdir()
    collections = {}
    rows: dict[str, list[dict]] = {
        "all_pairs": [],
        "full": [],
        "rgb_labelled": [],
        "thermal_labelled": [],
    }
    for site, vis, frames, labels, offset in CLIPS:
        flight = f"{site}_Enterprise"
        cid = f"{flight}_{vis:04d}"
        rgb_dir, thermal_dir = f"{flight}_VIS_{vis:04d}", f"{flight}_IR_{vis + 1:04d}"
        collections[cid] = Collection(
            rgb_dir, thermal_dir, f"{vis:04d}", f"{vis + 1:04d}", offset
        )
        for i in range(frames):
            rgb = f"{rgb_dir}/{rgb_dir}_{i:05d}.jpeg"
            thermal = f"{thermal_dir}/{thermal_dir}_{i + offset:05d}.jpeg"
            box = [{"x_center": 0.5, "y_center": 0.5, "width": 0.01, "height": 0.02}]
            pair = {"collection_id": cid, "rgb_image": rgb, "thermal_image": thermal}
            rows["all_pairs"].append(pair)
            if labels == "both":
                rows["full"].append(pair | {"rgb_boxes": box, "thermal_boxes": box})
            if labels in {"both", "rgb"}:
                rows["rgb_labelled"].append(
                    {"collection_id": cid, "image": rgb, "boxes": box}
                )
            if labels in {"both", "thermal"}:
                rows["thermal_labelled"].append(
                    {"collection_id": cid, "image": thermal, "boxes": box}
                )
    for name, records in rows.items():
        (manifests / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
    (manifests / "data_quality.json").write_text(json.dumps({"subset": False}))
    return manifests, collections


@pytest.fixture
def folds(tmp_path: Path) -> tuple[Path, Path, dict[str, Collection]]:
    manifests, collections = _dataset(tmp_path)
    destination = manifests / "folds"
    write_folds(manifests, destination, _size, collections=collections)
    return manifests, destination, collections


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _images(record: dict) -> list[str]:
    return [record[k] for k in ("image", "rgb_image", "thermal_image") if k in record]


def test_site_day_is_date_and_site_of_every_wisard_collection() -> None:
    clips = Counter()
    for cid, collection in WISARD_COLLECTIONS.items():
        day = site_day(cid)
        assert site_day(collection.rgb_dir) == day, cid
        assert site_day(collection.thermal_dir) == day, cid
        assert site_day(f"{collection.rgb_dir}/frame_00001.jpeg") == day, cid
        clips[day] += 1
    assert clips == {
        "210417_MtErie": 3,
        "210529_Carnation": 2,
        "210812_Hannegan": 2,
        "210924_FHL": 9,
        "220109_Baker": 1,
    }


@pytest.mark.parametrize("name", ["DJI_0583", "Baker_220109_VIS", "21092_FHL_VIS"])
def test_site_day_refuses_names_without_date_and_site(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        site_day(name)


def test_one_fold_per_labelled_site_day_and_unlabelled_one_is_never_tested(
    folds: tuple[Path, Path, dict],
) -> None:
    _, destination, _ = folds
    summary = json.loads((destination / SUMMARY).read_text())
    expected = ["240101_Alpha", "240202_Bravo", "240303_Charlie"]
    assert summary["folds"] == expected
    assert sorted(p.name for p in destination.iterdir() if p.is_dir()) == expected
    for fold in expected:
        pool = _records(destination / fold / UNLABELLED)
        assert "240404_Delta" in {site_day(r["rgb_image"]) for r in pool}


def test_no_site_day_or_image_appears_on_both_sides_of_any_fold(
    folds: tuple[Path, Path, dict],
) -> None:
    _, destination, _ = folds
    for fold in ("240101_Alpha", "240202_Bravo", "240303_Charlie"):
        seen = set()
        for record in _records(destination / fold / UNLABELLED):
            seen.update(_images(record))
        for view in VIEWS:
            for percent in PERCENTS:
                for r in _records(destination / fold / view / train_file(percent)):
                    seen.update(_images(r))
            for r in _records(destination / fold / view / "validation.jsonl"):
                seen.update(_images(r))
        assert fold not in {site_day(image) for image in seen}
        for view in VIEWS:
            test = _records(destination / fold / view / "test.jsonl")
            assert test, (fold, view)
            test_images = {image for r in test for image in _images(r)}
            assert {site_day(image) for image in test_images} <= {fold}
            assert not test_images & seen


def test_test_set_holds_every_labelled_frame_of_its_site_day(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, _ = folds
    rgb = _records(manifests / "rgb_labelled.jsonl")
    test = _records(destination / "240303_Charlie" / "rgb" / "test.jsonl")
    expected = [r["image"] for r in rgb if site_day(r["image"]) == "240303_Charlie"]
    assert [r["image"] for r in test] == expected


def test_validation_is_a_whole_clip_where_the_site_day_has_several(
    folds: tuple[Path, Path, dict],
) -> None:
    _, destination, _ = folds
    view = destination / "240202_Bravo" / "paired"
    validation = _records(view / "validation.jsonl")
    train = _records(view / train_file(100))
    held = [_where(i) for r in validation for i in _images(r)]
    trained = {_where(i)[0] for r in train for i in _images(r)}
    alpha = [(c, f) for c, f in held if "Alpha" in c]
    # The smallest Alpha clip is held out whole, and none of it is trained on.
    assert {c for c, _ in alpha} == {"240101_Alpha_Enterprise_0005"}
    assert len({f for _, f in alpha}) == 120
    assert "240101_Alpha_Enterprise_0005" not in trained
    # Charlie has two clips, but only one labelled in both cameras: its
    # validation is the end of that clip.
    charlie = [(c, f) for c, f in held if "Charlie" in c]
    assert {c for c, _ in charlie} == {"240303_Charlie_Enterprise_0021"}
    assert min(f for _, f in charlie) == 340


def test_gap_between_training_and_validation_within_a_clip(
    folds: tuple[Path, Path, dict],
) -> None:
    _, destination, _ = folds
    for fold in ("240101_Alpha", "240202_Bravo", "240303_Charlie"):
        pool = _records(destination / fold / UNLABELLED)
        for view in VIEWS:
            validation = _records(destination / fold / view / "validation.jsonl")
            train = _records(destination / fold / view / train_file(100))
            held = [_where(i) for r in validation for i in _images(r)]
            others = [_where(i) for r in train + pool for i in _images(r)]
            for clip, frame in held:
                near = [abs(frame - f) for c, f in others if c == clip]
                assert min(near, default=MIN_GAP + 1) > MIN_GAP, (fold, view, clip)
    # Bravo is one clip of 1,000 frames: validation is its last 150, and the
    # 250 frames before that are used nowhere.
    view = destination / "240101_Alpha" / "paired"
    bravo = [
        _where(r["rgb_image"])[1]
        for r in _records(view / train_file(100))
        if site_day(r["rgb_image"]) == "240202_Bravo"
    ]
    assert max(bravo) == 599


def test_thermal_frames_use_the_clip_time_of_their_rgb_partner(
    folds: tuple[Path, Path, dict],
) -> None:
    # Alpha clip 1 numbers thermal frames from 1: thermal frame n is RGB frame n-1.
    _, destination, _ = folds
    test = _records(destination / "240101_Alpha" / "thermal" / "test.jsonl")
    first = next(
        r for r in test if r["collection_id"] == "240101_Alpha_Enterprise_0001"
    )
    assert first["image"].endswith("_00001.jpeg")
    assert first["frame"] == 0


def test_label_fractions_are_nested_contiguous_and_reproducible(
    tmp_path: Path, folds: tuple[Path, Path, dict]
) -> None:
    manifests, destination, collections = folds
    for fold in ("240101_Alpha", "240202_Bravo", "240303_Charlie"):
        for view in VIEWS:
            subsets = [
                [
                    json.dumps(r)
                    for r in _records(destination / fold / view / train_file(p))
                ]
                for p in PERCENTS
            ]
            for smaller, larger in zip(subsets, subsets[1:], strict=False):
                assert set(smaller) <= set(larger), (fold, view)
            full = len(subsets[-1])
            for percent, subset in zip(PERCENTS[:-1], subsets, strict=False):
                assert len(subset) >= percent / 100 * full
                assert len(subset) < percent / 100 * full + BLOCK_FRAMES
    # Contiguous blocks: the 1% subset is runs of consecutive frames.
    one = _records(destination / "240303_Charlie" / "paired" / train_file(1))
    assert len(one) == BLOCK_FRAMES * 2
    frames = sorted(_where(r["rgb_image"]) for r in one)
    runs = sum(
        1
        for a, b in zip(frames, frames[1:], strict=False)
        if a[0] != b[0] or b[1] != a[1] + 1
    )
    assert runs <= 1

    again = tmp_path / "again"
    write_folds(manifests, again, _size, collections=collections)
    other = tmp_path / "other"
    write_folds(manifests, other, _size, collections=collections, seed=8)
    for path in destination.rglob("*.jsonl"):
        assert (again / path.relative_to(destination)).read_text() == path.read_text()
    one_pct = destination / "240303_Charlie" / "paired" / train_file(1)
    assert (other / one_pct.relative_to(destination)).read_text() != one_pct.read_text()


def test_small_fractions_keep_site_days_in_proportion(
    folds: tuple[Path, Path, dict],
) -> None:
    _, destination, _ = folds
    # In the Charlie fold, Alpha trains on 500 paired frames and Bravo on 600:
    # 10% is 11 blocks, and both site-days must be in it.
    ten = _records(destination / "240303_Charlie" / "paired" / train_file(10))
    counts = Counter(site_day(r["rgb_image"]) for r in ten)
    assert set(counts) == {"240101_Alpha", "240202_Bravo"}


def test_records_carry_what_evaluation_needs(folds: tuple[Path, Path, dict]) -> None:
    _, destination, _ = folds
    paired = _records(destination / "240202_Bravo" / "paired" / "test.jsonl")[0]
    assert paired["rgb_size"] == [1920, 1080]
    assert paired["thermal_size"] == [640, 512]
    assert paired["site_day"] == "240202_Bravo"
    assert paired["collection_id"] == "240202_Bravo_Enterprise_0011"
    single = _records(destination / "240202_Bravo" / "thermal" / "test.jsonl")[0]
    assert single["camera"] == "thermal"
    assert single["size"] == [640, 512]


def test_check_passes_on_built_folds(folds: tuple[Path, Path, dict]) -> None:
    manifests, destination, collections = folds
    report = check_folds(manifests, destination, _size, collections=collections)
    assert report.problems == []
    assert report.lines


def test_check_says_in_its_first_line_that_the_sources_were_full(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, collections = folds
    report = check_folds(manifests, destination, _size, collections=collections)
    assert report.lines[0] == "source manifests: full"


@pytest.mark.parametrize("quality", [{"subset": True}, {}, None])
def test_folds_refuse_manifests_that_are_not_marked_full(
    tmp_path: Path, quality: dict | None
) -> None:
    manifests, collections = _dataset(tmp_path)
    marker = manifests / "data_quality.json"
    if quality is None:
        marker.unlink()
    else:
        marker.write_text(json.dumps(quality))
    destination = manifests / "folds"
    with pytest.raises(ValueError, match="subset"):
        write_folds(manifests, destination, _size, collections=collections)
    assert not destination.exists()


def test_check_fails_on_manifests_that_are_not_full(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, collections = folds
    (manifests / "data_quality.json").write_text(json.dumps({"subset": True}))
    report = check_folds(manifests, destination, _size, collections=collections)
    assert report.lines[0] == "source manifests: subset"
    assert any("subset" in p for p in report.problems)


def _corrupt(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def test_check_finds_a_test_image_in_training(folds: tuple[Path, Path, dict]) -> None:
    manifests, destination, collections = folds
    fold = destination / "240202_Bravo"
    leaked = _records(fold / "paired" / "test.jsonl")[0]
    _corrupt(fold / "rgb" / train_file(100), leaked)
    report = check_folds(manifests, destination, _size, collections=collections)
    assert (
        "240202_Bravo/rgb/train_100pct.jsonl holds images of test site-day "
        "240202_Bravo" in report.problems
    )
    assert (
        "240202_Bravo/rgb/train_100pct.jsonl shares images with the test set"
        in report.problems
    )


def test_check_finds_a_test_image_in_the_unlabelled_pool(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, collections = folds
    fold = destination / "240101_Alpha"
    leaked = _records(fold / "paired" / "test.jsonl")[5]
    pair = {k: leaked[k] for k in ("site_day", "collection_id", "frame")}
    pair |= {"rgb_image": leaked["rgb_image"], "thermal_image": leaked["thermal_image"]}
    _corrupt(fold / UNLABELLED, pair)
    report = check_folds(manifests, destination, _size, collections=collections)
    assert (
        f"240101_Alpha/{UNLABELLED} holds images of test site-day 240101_Alpha"
        in report.problems
    )


def test_check_finds_validation_too_close_to_training(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, collections = folds
    view = destination / "240101_Alpha" / "paired"
    near = next(
        r
        for r in _records(destination / "240202_Bravo" / "paired" / "test.jsonl")
        if _where(r["rgb_image"])[1] == 800
    )
    _corrupt(view / train_file(100), near)
    report = check_folds(manifests, destination, _size, collections=collections)
    assert any("inside the gap" in p for p in report.problems)


def test_check_reads_the_clip_from_the_path_not_the_record(
    folds: tuple[Path, Path, dict],
) -> None:
    # A gap frame of Bravo moved into training, its collection_id and frame
    # changed to another clip's: the check must still see it is Bravo frame
    # 700, 150 frames from Bravo's validation, without the rebuild.
    manifests, destination, collections = folds
    view = destination / "240101_Alpha" / "paired"
    gap_frame = next(
        r
        for r in _records(destination / "240202_Bravo" / "paired" / "test.jsonl")
        if _where(r["rgb_image"])[1] == 700
    )
    gap_frame |= {
        "site_day": "240303_Charlie",
        "collection_id": "240303_Charlie_Enterprise_0021",
        "frame": 10,
    }
    _corrupt(view / train_file(100), gap_frame)
    report = check_folds(manifests, destination, _size, collections=collections)
    assert any(
        "240101_Alpha/paired: training frame 150 frames from validation in "
        "240202_Bravo_Enterprise_0011" in p
        for p in report.problems
    ), report.problems


def test_check_holds_a_minimum_gap_whatever_the_builder_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, collections = _dataset(tmp_path)
    destination = manifests / "folds"
    monkeypatch.setattr(folds_module, "GAP_FRAMES", 0)
    write_folds(manifests, destination, _size, collections=collections)

    report = check_folds(manifests, destination, _size, collections=collections)

    assert any(f"gap of {MIN_GAP}" in p for p in report.problems), report.problems
    assert any("gap_frames 0" in p for p in report.problems), report.problems


def test_check_finds_a_subset_that_is_not_nested(
    folds: tuple[Path, Path, dict],
) -> None:
    manifests, destination, collections = folds
    view = destination / "240101_Alpha" / "rgb"
    extra = _records(view / "validation.jsonl")[0]
    _corrupt(view / train_file(5), extra)
    report = check_folds(manifests, destination, _size, collections=collections)
    assert any("nested" in p for p in report.problems)


def test_refuses_to_overwrite_a_directory_it_did_not_write(tmp_path: Path) -> None:
    manifests, collections = _dataset(tmp_path)
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    (destination / "keep.txt").write_text("not ours")
    with pytest.raises(FileExistsError):
        write_folds(manifests, destination, _size, collections=collections)
    assert (destination / "keep.txt").exists()


CLIP = "240101_Alpha_Enterprise_0003"
OVERRIDE = LabelOverride(CLIP, "thermal", "empty_label", "labelling stops early")


def _override_dataset(root: Path) -> tuple[Path, dict[str, Collection], set[str]]:
    """The dataset with the first 120 thermal frames of CLIP made empty-label."""
    manifests, collections = _dataset(root)
    path = manifests / "thermal_labelled.jsonl"
    rows = _records(path)
    emptied = set()
    for row in rows:
        if row["collection_id"] == CLIP and int(row["image"][-10:-5]) < 120:
            row["boxes"] = []
            emptied.add(row["image"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifests, collections, emptied


def _build_with_override(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Collection], set[str]]:
    manifests, collections, emptied = _override_dataset(tmp_path)
    destination = manifests / "folds"
    write_folds(
        manifests, destination, _size, collections=collections, overrides=[OVERRIDE]
    )
    return manifests, destination, collections, emptied


def test_override_frames_land_only_in_the_pool(tmp_path: Path) -> None:
    _, destination, _, emptied = _build_with_override(tmp_path)
    assert emptied
    for fold in ("240202_Bravo", "240303_Charlie"):
        pool = {r["thermal_image"] for r in _records(destination / fold / UNLABELLED)}
        assert emptied <= pool
        thermal = destination / fold / "thermal"
        for path in thermal.glob("*.jsonl"):
            assert not {r["image"] for r in _records(path)} & emptied, path
    test_fold = destination / "240101_Alpha"
    for path in (test_fold / "thermal").glob("*.jsonl"):
        assert not {r["image"] for r in _records(path)} & emptied, path
    assert emptied.isdisjoint(
        {r["thermal_image"] for r in _records(test_fold / UNLABELLED)}
    )


def test_override_leaves_other_frames_and_the_rgb_view_alone(tmp_path: Path) -> None:
    manifests, destination, _, emptied = _build_with_override(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()
    base, collections = _dataset(plain)
    write_folds(base, base / "folds", _size, collections=collections)
    for view in ("rgb", "paired"):
        for name in (train_file(100), "validation.jsonl"):
            a = (destination / "240202_Bravo" / view / name).read_text()
            b = (base / "folds" / "240202_Bravo" / view / name).read_text()
            assert a == b
    test = _records(destination / "240101_Alpha" / "thermal" / "test.jsonl")
    all_test = _records(base / "folds" / "240101_Alpha" / "thermal" / "test.jsonl")
    assert len(test) == len(all_test) - len(emptied)


def test_override_is_recorded_in_the_summary(tmp_path: Path) -> None:
    _, destination, _, emptied = _build_with_override(tmp_path)
    summary = json.loads((destination / SUMMARY).read_text())
    assert summary["label_overrides"] == [
        {
            "collection_id": CLIP,
            "camera": "thermal",
            "rule": "empty_label",
            "reason": "labelling stops early",
            "frames": len(emptied),
        }
    ]


def test_check_passes_with_an_override_applied(tmp_path: Path) -> None:
    manifests, destination, collections, _ = _build_with_override(tmp_path)
    report = check_folds(
        manifests, destination, _size, collections=collections, overrides=[OVERRIDE]
    )
    assert report.problems == []
    assert any("override" in line for line in report.lines)


def test_check_finds_an_overridden_frame_in_a_thermal_manifest(
    tmp_path: Path,
) -> None:
    manifests, destination, collections, emptied = _build_with_override(tmp_path)
    fold = destination / "240202_Bravo"
    leaked = _records(fold / "thermal" / train_file(100))[0] | {
        "image": sorted(emptied)[0],
        "collection_id": CLIP,
        "boxes": [],
    }
    _corrupt(fold / "thermal" / train_file(100), leaked)
    report = check_folds(
        manifests, destination, _size, collections=collections, overrides=[OVERRIDE]
    )
    assert any("overridden" in p and "thermal" in p for p in report.problems)


def test_check_finds_an_overridden_frame_missing_from_the_pool(
    tmp_path: Path,
) -> None:
    manifests, destination, collections, emptied = _build_with_override(tmp_path)
    pool_file = destination / "240202_Bravo" / UNLABELLED
    kept = [r for r in _records(pool_file) if r["thermal_image"] != sorted(emptied)[0]]
    pool_file.write_text("".join(json.dumps(r) + "\n" for r in kept))
    report = check_folds(
        manifests, destination, _size, collections=collections, overrides=[OVERRIDE]
    )
    assert any("pool" in p and "overridden" in p for p in report.problems)


def test_check_finds_an_override_that_was_not_applied(tmp_path: Path) -> None:
    manifests, collections, _ = _override_dataset(tmp_path)
    destination = manifests / "folds"
    write_folds(manifests, destination, _size, collections=collections, overrides=[])
    report = check_folds(
        manifests, destination, _size, collections=collections, overrides=[OVERRIDE]
    )
    assert any("overridden" in p for p in report.problems)
