import json
import re
from pathlib import Path

import pytest

from aerial_search.data import checksums, wisard
from aerial_search.data.wisard import (
    WISARD_COLLECTIONS,
    Collection,
    _extract_frame_index,
    _frames_by_index,
    load_boxes,
    load_pairs,
    pairing_report,
    prepare_manifests,
    select_collections,
)

# One flight in the shape of WiSARD: the drone records VIS as DJI file n and IR
# as DJI file n+1, and every file name carries its clip number.
FLIGHT = {
    "flight_0001": Collection("flight_VIS_0001", "flight_IR_0002", "0001", "0002")
}
WISARD_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "wisard-full"


def test_pairs_frames_by_index_not_position(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    # Each camera dropped a different frame. Position-based pairing would slip
    # by one frame between frames 5 and 20; index-based pairing does not.
    for number in range(40):
        if number != 5:
            _sample(rgb, number)
        if number != 20:
            _sample(thermal, number)

    pairs = load_pairs(tmp_path, collections=FLIGHT)

    assert len(pairs) == 38
    assert all(_frame(p.rgb_image) == _frame(p.thermal_image) for p in pairs)


def test_applies_thermal_frame_offset(tmp_path: Path) -> None:
    # The layout of MtErie_0003: VIS numbered from 0, IR numbered from 1.
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        _sample(rgb, number)
        _sample(thermal, number + 1)
    offset = {
        "flight_0001": Collection(
            "flight_VIS_0001", "flight_IR_0002", "0001", "0002", thermal_offset=1
        )
    }

    pairs = load_pairs(tmp_path, collections=offset)

    assert len(pairs) == 40
    assert all(_frame(p.thermal_image) == _frame(p.rgb_image) + 1 for p in pairs)


def test_pairs_flight_whose_frame_counts_differ(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    # Real WiSARD naming: 8-digit RGB numbers, 5-digit thermal numbers from a
    # video export. RGB has 39 frames, thermal has 40.
    for number in range(39):
        _touch(rgb / f"flight_VIS_0001_{number:08d}.jpg")
    for number in range(40):
        _touch(thermal / f"DJI_0002.mp4_{number:05d}.jpg")

    pairs = load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)
    report = pairing_report(tmp_path, collections=FLIGHT)

    assert [_frame(p.rgb_image) for p in pairs] == list(range(39))
    assert [_frame(p.thermal_image) for p in pairs] == list(range(39))
    counts = report["collections"]["flight_0001"]
    assert counts["rgb_frames"] == 39
    assert counts["thermal_frames"] == 40
    assert counts["pairs"] == 39
    assert counts["thermal_only_frames"] == 1


def test_counts_pairs_by_which_camera_is_labelled(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        (rgb / f"flight_VIS_0001_{number:05d}.jpg").touch()
        (thermal / f"flight_IR_0002_{number:05d}.jpg").touch()
    for number in range(10):  # both labelled
        (rgb / f"flight_VIS_0001_{number:05d}.txt").touch()
        (thermal / f"flight_IR_0002_{number:05d}.txt").touch()
    for number in range(10, 13):  # RGB only
        (rgb / f"flight_VIS_0001_{number:05d}.txt").touch()
    for number in range(13, 20):  # thermal only
        (thermal / f"flight_IR_0002_{number:05d}.txt").touch()

    counts = pairing_report(tmp_path, collections=FLIGHT)["collections"]["flight_0001"]

    assert counts["labelled_pairs"] == 10
    assert counts["rgb_only_labelled_pairs"] == 3
    assert counts["thermal_only_labelled_pairs"] == 7
    assert counts["unlabelled_pairs"] == 20


def test_skips_missing_annotation(tmp_path: Path) -> None:
    """Pairs without labels in both modalities are left out of load_pairs."""
    rgb, thermal = _flight_dirs(tmp_path)
    _sample(rgb, 0)  # Has annotation
    (thermal / "flight_IR_0002_00000.jpeg").touch()  # Same frame, no annotation

    assert load_pairs(tmp_path, collections=FLIGHT) == []
    assert len(load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)) == 1


def test_pairs_only_listed_directories_and_reports_the_rest(tmp_path: Path) -> None:
    # Two VIS and three IR directories at one site. Grouping by site name alone
    # would have to guess which goes with which; only listed pairs are used.
    for name in [
        "site_VIS_0001",
        "site_IR_0002",
        "site_VIS_0003",
        "site_IR_0004",
        "site_IR_0005",
    ]:
        (tmp_path / name).mkdir()
        _sample(tmp_path / name, 0)
    collections = {
        "site_0003": Collection("site_VIS_0003", "site_IR_0004", "0003", "0004")
    }

    pairs = load_pairs(tmp_path, collections=collections)
    report = pairing_report(tmp_path, collections=collections)

    assert [(p.rgb_image.parent.name, p.thermal_image.parent.name) for p in pairs] == [
        ("site_VIS_0003", "site_IR_0004")
    ]
    assert report["unpaired_directories"] == [
        "site_IR_0002",
        "site_IR_0005",
        "site_VIS_0001",
    ]


def test_missing_listed_directory_stops_the_run(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    _sample(rgb, 0)
    _sample(thermal, 0)
    thermal.rename(tmp_path / "renamed")

    with pytest.raises(FileNotFoundError, match="flight_IR_0002"):
        load_pairs(tmp_path, collections=FLIGHT)
    with pytest.raises(FileNotFoundError, match="flight_IR_0002"):
        prepare_manifests(tmp_path, tmp_path / "out", collections=FLIGHT)


def test_select_collections_names_a_subset_explicitly() -> None:
    subset = select_collections(["220109_Baker_Enterprise_1"])

    assert list(subset) == ["220109_Baker_Enterprise_1"]
    with pytest.raises(KeyError, match="no_such_clip"):
        select_collections(["no_such_clip"])


def test_an_empty_selection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No collections"):
        select_collections([])
    with pytest.raises(ValueError, match="No collections"):
        prepare_manifests(tmp_path, tmp_path / "out", collections={})
    assert not (tmp_path / "out").exists()


def _two_clip_root(root: Path) -> dict[str, Collection]:
    """Two listed clips and one unlisted directory, one frame each."""
    clips = {
        "a_0001": Collection("a_VIS_0001", "a_IR_0002", "0001", "0002"),
        "b_0003": Collection("b_VIS_0003", "b_IR_0004", "0003", "0004"),
    }
    for c in clips.values():
        for directory, clip in [
            (c.rgb_dir, c.rgb_clip),
            (c.thermal_dir, c.thermal_clip),
        ]:
            (root / directory).mkdir()
            _touch(root / directory / f"{directory}_{clip}_00000.jpg")
    (root / "stray_VIS_9").mkdir()
    _touch(root / "stray_VIS_9" / "stray_9_00000.jpg")
    return clips


def test_a_subset_says_so_and_names_what_it_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = _two_clip_root(tmp_path)
    monkeypatch.setattr(wisard, "WISARD_COLLECTIONS", clips)
    full, subset = tmp_path / "full", tmp_path / "subset"

    prepare_manifests(tmp_path, full, collections=clips)
    prepare_manifests(tmp_path, subset, collections={"a_0001": clips["a_0001"]})

    whole = json.loads((full / "data_quality.json").read_text())
    part = json.loads((subset / "data_quality.json").read_text())
    assert whole["subset"] is False
    assert whole["selected_collections"] == ["a_0001", "b_0003"]
    assert whole["unselected_directories"] == []
    assert part["subset"] is True
    assert part["selected_collections"] == ["a_0001"]
    assert part["unselected_directories"] == ["b_IR_0004", "b_VIS_0003"]
    assert part["unpaired_directories"] == ["stray_VIS_9"]


def test_a_subset_is_not_written_over_full_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = _two_clip_root(tmp_path)
    monkeypatch.setattr(wisard, "WISARD_COLLECTIONS", clips)
    out = tmp_path / "out"
    prepare_manifests(tmp_path, out, collections=clips)
    before = (out / "full.jsonl").read_text()

    with pytest.raises(ValueError, match="full manifests"):
        prepare_manifests(tmp_path, out, collections={"a_0001": clips["a_0001"]})

    assert (out / "full.jsonl").read_text() == before
    # Manifests with no record of what they are count as full.
    (out / "data_quality.json").unlink()
    with pytest.raises(ValueError, match="full manifests"):
        prepare_manifests(tmp_path, out, collections={"a_0001": clips["a_0001"]})
    # A subset may be rewritten, and a full set may replace it.
    other = tmp_path / "other"
    subset = {"a_0001": clips["a_0001"]}
    prepare_manifests(tmp_path, other, collections=subset)
    prepare_manifests(tmp_path, other, collections=subset)
    prepare_manifests(tmp_path, other, collections=clips)


def test_rejects_file_from_another_clip(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        _sample(rgb, number)
        _sample(thermal, number)
    _touch(thermal / "flight_IR_0004_00000041.jpg")  # a frame of clip 0004

    with pytest.raises(ValueError, match="flight_IR_0004_00000041.jpg"):
        load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)


def test_rejects_duplicate_frame_numbers_in_one_directory(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    _touch(rgb / "flight_VIS_0001_00000007.jpg")
    _touch(rgb / "flight_VIS_0001_7.jpeg")  # also frame 7
    _touch(thermal / "flight_IR_0002_00007.jpg")

    with pytest.raises(ValueError, match="frame 7"):
        load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)


def test_rejects_image_without_frame_number(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    _sample(rgb, 0)
    _sample(thermal, 0)
    _touch(thermal / "thumbnail.jpg")

    with pytest.raises(ValueError, match="thumbnail.jpg"):
        load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)


@pytest.mark.parametrize(
    ("rgb_numbers", "thermal_numbers"),
    [
        (range(10), range(8, 18)),  # numbering does not overlap
        (range(10), range(20)),  # one camera has twice the frames
    ],
)
def test_rejects_listed_pair_whose_frame_numbers_do_not_match_up(
    tmp_path: Path, rgb_numbers: range, thermal_numbers: range
) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in rgb_numbers:
        _sample(rgb, number)
    for number in thermal_numbers:
        _sample(thermal, number)

    with pytest.raises(ValueError, match="flight_0001"):
        load_pairs(tmp_path, collections=FLIGHT, labelled_only=False)


def test_wisard_collections_follow_the_clip_numbering() -> None:
    rgb_dirs = [c.rgb_dir for c in WISARD_COLLECTIONS.values()]
    thermal_dirs = [c.thermal_dir for c in WISARD_COLLECTIONS.values()]
    assert len(set(rgb_dirs + thermal_dirs)) == len(rgb_dirs) + len(thermal_dirs)

    for cid, c in WISARD_COLLECTIONS.items():
        rgb_flight, rgb_suffix = c.rgb_dir.split("_VIS_")
        thermal_flight, thermal_suffix = c.thermal_dir.split("_IR_")
        assert rgb_flight == thermal_flight, cid
        # The drone writes VIS as clip n and IR as clip n+1.
        assert int(c.thermal_clip) == int(c.rgb_clip) + 1, cid
        if len(rgb_suffix) == 4:  # directories named after their DJI clip
            assert (rgb_suffix, thermal_suffix) == (c.rgb_clip, c.thermal_clip), cid
            assert cid == f"{rgb_flight}_{rgb_suffix}", cid
    offsets = {cid: c.thermal_offset for cid, c in WISARD_COLLECTIONS.items()}
    assert {cid for cid, o in offsets.items() if o} == {"210417_MtErie_Enterprise_0003"}


@pytest.mark.skipif(not WISARD_ROOT.is_dir(), reason="WiSARD data not present")
def test_wisard_collections_match_the_data_on_disk() -> None:
    for cid, c in WISARD_COLLECTIONS.items():
        for directory, clip in [
            (c.rgb_dir, c.rgb_clip),
            (c.thermal_dir, c.thermal_clip),
        ]:
            path = WISARD_ROOT / directory
            assert path.is_dir(), f"{cid}: {path} missing"
            token = re.compile(rf"(?<!\d){clip}(?!\d)")
            names = [p.name for p in path.iterdir() if p.suffix in {".jpg", ".jpeg"}]
            assert names, cid
            assert all(token.search(name) for name in names), (cid, directory)
    report = pairing_report(WISARD_ROOT)
    assert len(report["collections"]) == len(WISARD_COLLECTIONS)


def test_clips_boxes_to_the_image_and_drops_empty_ones(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text(
        "0 0.5 0.4 0.1 0.2\n"  # inside
        "0 0.98 0.5 0.1 0.2\n"  # runs past the right edge
        "0 1.2 0.5 0.1 0.2\n"  # wholly outside: nothing left
    )
    stats: dict[str, int] = {}

    boxes = load_boxes(labels, stats=stats)

    assert len(boxes) == 2
    assert boxes[0].width == pytest.approx(0.1)
    assert boxes[1].x_center == pytest.approx(0.965)
    assert boxes[1].width == pytest.approx(0.07)
    assert stats == {"boxes_clipped": 2, "boxes_dropped": 1}


def test_does_not_count_label_rounding_as_clipping(tmp_path: Path) -> None:
    # A real Baker IR label: the left edge is -0.0000005, from rounding the
    # coordinates to six decimals, not from a box drawn past the image.
    labels = tmp_path / "labels.txt"
    labels.write_text("0 0.060937 0.818359 0.121875 0.193359\n")
    stats: dict[str, int] = {}

    boxes = load_boxes(labels, stats=stats)

    assert boxes[0].x_center - boxes[0].width / 2 >= 0
    assert stats == {"boxes_clipped": 0, "boxes_dropped": 0}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_rejects_a_label_value_that_is_not_finite(tmp_path: Path, value: str) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text(f"0 0.5 0.5 {value} 0.2\n")

    with pytest.raises(ValueError, match="labels.txt:1"):
        load_boxes(labels)


def test_counts_an_overshoot_just_above_the_rounding_tolerance(tmp_path: Path) -> None:
    # 1e-4 past the edge is a drawn box, not six-decimal rounding (at most 1e-6).
    labels = tmp_path / "labels.txt"
    labels.write_text("0 0.99995 0.5 0.0002 0.2\n")
    stats: dict[str, int] = {}

    load_boxes(labels, stats=stats)

    assert stats == {"boxes_clipped": 1, "boxes_dropped": 0}


def test_prepare_writes_no_split_and_removes_an_old_one(tmp_path: Path) -> None:
    # Splits are by site-day and live in folds/ (aerial_search.data.folds).
    # An old random split left beside the manifests could be used by mistake.
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(4):
        _sample(rgb, number)
        _sample(thermal, number)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (out / name).write_text("{}\n")

    counts = prepare_manifests(tmp_path, out, collections=FLIGHT)

    assert counts == {
        "all_pairs": 4,
        "full": 4,
        "rgb_labelled": 4,
        "thermal_labelled": 4,
    }
    assert sorted(p.name for p in out.iterdir()) == [
        "all_pairs.jsonl",
        "data_quality.json",
        "full.jsonl",
        "rgb_labelled.jsonl",
        "thermal_labelled.jsonl",
    ]


def test_writes_one_manifest_per_camera_and_counts_boxes_once(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        (rgb / f"flight_VIS_0001_{number:05d}.jpg").touch()
        (thermal / f"flight_IR_0002_{number:05d}.jpg").touch()
    # Frame 0 labelled in both, with one RGB box past the edge.
    (rgb / "flight_VIS_0001_00000.txt").write_text("0 0.98 0.5 0.1 0.2\n")
    (thermal / "flight_IR_0002_00000.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    # Frame 1 labelled in RGB only, frame 2 in thermal only.
    (rgb / "flight_VIS_0001_00001.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    (thermal / "flight_IR_0002_00002.txt").write_text("")
    out = tmp_path / "out"

    prepare_manifests(tmp_path, out, collections=FLIGHT)

    rgb_records = _records(out / "rgb_labelled.jsonl")
    thermal_records = _records(out / "thermal_labelled.jsonl")
    assert [Path(r["image"]).name for r in rgb_records] == [
        "flight_VIS_0001_00000.jpg",
        "flight_VIS_0001_00001.jpg",
    ]
    assert [Path(r["image"]).name for r in thermal_records] == [
        "flight_IR_0002_00000.jpg",
        "flight_IR_0002_00002.jpg",
    ]
    assert len(_records(out / "full.jsonl")) == 1
    quality = json.loads((out / "data_quality.json").read_text())
    assert quality["boxes"]["flight_0001"] == {
        "rgb_clipped": 1,
        "rgb_dropped": 0,
        "thermal_clipped": 0,
        "thermal_dropped": 0,
    }


def test_clips_without_labels_have_zero_box_counts(tmp_path: Path) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        (rgb / f"flight_VIS_0001_{number:05d}.jpg").touch()
        (thermal / f"flight_IR_0002_{number:05d}.jpg").touch()
    out = tmp_path / "out"

    prepare_manifests(tmp_path, out, collections=FLIGHT)

    quality = json.loads((out / "data_quality.json").read_text())
    assert quality["boxes"] == {
        "flight_0001": {
            "rgb_clipped": 0,
            "rgb_dropped": 0,
            "thermal_clipped": 0,
            "thermal_dropped": 0,
        }
    }


def test_a_label_file_that_lost_all_its_boxes_differs_from_an_empty_one(
    tmp_path: Path,
) -> None:
    rgb, thermal = _flight_dirs(tmp_path)
    for number in range(40):
        (rgb / f"flight_VIS_0001_{number:05d}.jpg").touch()
        (thermal / f"flight_IR_0002_{number:05d}.jpg").touch()
    (rgb / "flight_VIS_0001_00000.txt").write_text("")  # no people in frame
    (rgb / "flight_VIS_0001_00001.txt").write_text("0 1.2 0.5 0.1 0.2\n")  # outside
    (rgb / "flight_VIS_0001_00002.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    out = tmp_path / "out"

    prepare_manifests(tmp_path, out, collections=FLIGHT)

    by_name = {Path(r["image"]).name: r for r in _records(out / "rgb_labelled.jsonl")}
    empty = by_name["flight_VIS_0001_00000.jpg"]
    dropped = by_name["flight_VIS_0001_00001.jpg"]
    kept = by_name["flight_VIS_0001_00002.jpg"]
    assert (empty["boxes"], empty["boxes_dropped"]) == ([], 0)
    assert (dropped["boxes"], dropped["boxes_dropped"]) == ([], 1)
    assert (len(kept["boxes"]), kept["boxes_dropped"]) == (1, 0)


@pytest.mark.parametrize(
    ("name", "frame"),
    [
        # Every naming scheme found in the WiSARD dataset.
        ("210327_Airfield_FLIR_VIS_4_00000494.jpg", 494),
        ("200402_Karen_Inspire_VIS_695.jpeg", 695),
        ("20200929_134258_IR 127.jpg", 127),
        ("200704_baker__FLIR_1_ 0001.jpg", 1),
        ("DJI_0021 (1-6-2022 8-21-53 AM).mp4_00499.jpg", 499),
        ("DJI_0402.mp4_00000.jpg", 0),
        ("210417_MtErie_Enterprise_IR_0008_out_frame_00179.jpeg", 179),
        ("210417_MtErie_Enterprise_IR_0006_00290.jpeg", 290),
    ],
)
def test_parses_frame_numbers_of_any_length(name: str, frame: int) -> None:
    assert _extract_frame_index(Path(name)) == frame


@pytest.mark.parametrize("name", ["thumbnail.jpg", "DJI0001.jpg", "frame_12.png"])
def test_rejects_names_without_a_frame_number(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        _extract_frame_index(Path(name))


def test_frames_by_index_checks_clip_number(tmp_path: Path) -> None:
    _touch(tmp_path / "DJI_0583 (1-10-2022 2-51-48 PM).mp4_00000.jpg")

    assert list(_frames_by_index(tmp_path, clip="0583")) == [0]
    with pytest.raises(ValueError, match="0582"):
        _frames_by_index(tmp_path, clip="0582")


def test_a_clip_number_must_stand_alone_in_a_file_name(tmp_path: Path) -> None:
    # 0583 sits inside the longer numbers 10583 and 05831.
    _touch(tmp_path / "DJI_10583_00000.jpg")
    with pytest.raises(ValueError, match="DJI_10583_00000.jpg"):
        _frames_by_index(tmp_path, clip="0583")
    (tmp_path / "DJI_10583_00000.jpg").unlink()
    _touch(tmp_path / "DJI_05831_00000.jpg")
    with pytest.raises(ValueError, match="DJI_05831_00000.jpg"):
        _frames_by_index(tmp_path, clip="0583")


@pytest.mark.skipif(not WISARD_ROOT.is_dir(), reason="WiSARD data not present")
def test_each_listed_pair_starts_at_the_same_moment_on_disk() -> None:
    # Both recordings start together, so once the offset is applied the first
    # frame of the VIS clip and of the IR clip must have the same number.
    for cid, c in WISARD_COLLECTIONS.items():
        rgb = _frames_by_index(WISARD_ROOT / c.rgb_dir, c.rgb_clip)
        thermal = _frames_by_index(WISARD_ROOT / c.thermal_dir, c.thermal_clip)
        assert min(rgb) + c.thermal_offset == min(thermal), cid


def _flight_dirs(root: Path) -> tuple[Path, Path]:
    rgb = root / "flight_VIS_0001"
    thermal = root / "flight_IR_0002"
    rgb.mkdir()
    thermal.mkdir()
    return rgb, thermal


def _sample(directory: Path, number: int) -> None:
    stem = f"{directory.name}_{number:05d}"
    (directory / f"{stem}.jpeg").touch()
    (directory / f"{stem}.txt").touch()


def _touch(image: Path) -> None:
    image.touch()
    image.with_suffix(".txt").touch()


def _frame(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


PROVENANCE = {"commit": "abc123", "scratch": False}


def test_prepare_on_an_unpinned_tree_records_no_root_hash(tmp_path: Path) -> None:
    clips = _two_clip_root(tmp_path)
    out = tmp_path / "out"
    prepare_manifests(tmp_path, out, collections=clips, provenance=PROVENANCE)

    quality = json.loads((out / "data_quality.json").read_text())
    assert quality["provenance"] == PROVENANCE
    assert quality["data"]["dataset"] is None
    assert quality["data"]["verified"] is False
    assert quality["data"]["root_hash"] is None


def test_prepare_verifies_the_directories_it_reads(tmp_path: Path) -> None:
    from aerial_search.data.checksums import hash_tree, root_hash

    clips = _two_clip_root(tmp_path)
    pinned = hash_tree(tmp_path)
    out = tmp_path / "out"
    prepare_manifests(
        tmp_path, out, collections=clips, provenance=PROVENANCE, pinned=pinned
    )

    data = json.loads((out / "data_quality.json").read_text())["data"]
    assert data["verified"] is True
    assert data["dataset"] == "wisard-full"
    assert data["root_hash"] == root_hash(pinned)
    assert data["directories"] == ["a_IR_0002", "a_VIS_0001", "b_IR_0004", "b_VIS_0003"]
    assert data["seconds"] >= 0


def test_prepare_refuses_a_changed_file_and_writes_nothing(tmp_path: Path) -> None:
    from aerial_search.data.checksums import hash_tree

    clips = _two_clip_root(tmp_path)
    pinned = hash_tree(tmp_path)
    (tmp_path / "a_VIS_0001" / "a_VIS_0001_0001_00000.jpg").write_bytes(b"edited")
    out = tmp_path / "out"
    with pytest.raises(checksums.DataMismatchError, match="a_VIS_0001_0001_00000"):
        prepare_manifests(
            tmp_path, out, collections=clips, provenance=PROVENANCE, pinned=pinned
        )
    assert not out.exists()


def test_scratch_prepare_may_skip_verification(tmp_path: Path) -> None:
    from aerial_search.data.checksums import hash_tree

    clips = _two_clip_root(tmp_path)
    pinned = hash_tree(tmp_path)
    (tmp_path / "a_VIS_0001" / "a_VIS_0001_0001_00000.jpg").write_bytes(b"edited")
    out = tmp_path / "out"
    prepare_manifests(
        tmp_path,
        out,
        collections=clips,
        provenance={"commit": "c", "scratch": True},
        pinned=pinned,
        verify=False,
    )
    data = json.loads((out / "data_quality.json").read_text())["data"]
    assert data["verified"] is False and data["root_hash"] is None


def test_scratch_prepare_does_not_overwrite_quotable_manifests(tmp_path: Path) -> None:
    clips = _two_clip_root(tmp_path)
    out = tmp_path / "out"
    prepare_manifests(tmp_path, out, collections=clips, provenance=PROVENANCE)
    scratch = {"commit": "c", "scratch": True}
    with pytest.raises(ValueError, match="--force"):
        prepare_manifests(
            tmp_path, out, collections=clips, provenance=scratch, verify=False
        )
    assert (
        json.loads((out / "data_quality.json").read_text())["provenance"]["scratch"]
        is False
    )
    prepare_manifests(
        tmp_path, out, collections=clips, provenance=scratch, verify=False, force=True
    )
    # a scratch directory can be rewritten by another scratch run without force
    prepare_manifests(
        tmp_path, out, collections=clips, provenance=scratch, verify=False
    )
