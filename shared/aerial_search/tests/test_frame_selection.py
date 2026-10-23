"""The head and the supervised baseline label exactly the same frames (#67).

Also how a fold record's boxes are read, and the manifest leak checks (#66).
"""

import json
from pathlib import Path

import pytest

from aerial_search.data import records as rc


def _rec(site: str, name: str) -> dict:
    return {
        "site_day": site,
        "collection_id": f"{site}_x",
        "camera": "rgb",
        "image": f"{site}_VIS_1/{name}.jpg",
        "size": [1920, 1080],
        "boxes": [],
    }


def _fold(tmp_path: Path) -> Path:
    view = tmp_path / "folds" / "220109_Baker" / "rgb"
    view.mkdir(parents=True)
    files = {
        "train_1pct.jsonl": ["a"],
        "train_5pct.jsonl": ["a", "b"],
        "validation.jsonl": ["v1", "v2"],
        "test.jsonl": ["t"],
    }
    for file, names in files.items():
        site = "220109_Baker" if file == "test.jsonl" else "210417_MtErie"
        (view / file).write_text(
            "".join(json.dumps(_rec(site, n)) + "\n" for n in names)
        )
    return tmp_path


def test_select_records_reads_the_fractions_train_file_and_both_other_splits(tmp_path):
    manifests = _fold(tmp_path)
    paths, records = rc.select_records(manifests, "220109_Baker", "rgb", 5)
    assert paths["train"].name == "train_5pct.jsonl"
    assert [r["image"].split("/")[1] for r in records["train"]] == ["a.jpg", "b.jpg"]
    assert len(records["validation"]) == 2 and len(records["test"]) == 1


def test_select_records_still_refuses_a_leaking_manifest(tmp_path):
    manifests = _fold(tmp_path)
    leaked = manifests / "folds" / "220109_Baker" / "rgb" / "train_1pct.jsonl"
    leaked.write_text(json.dumps(_rec("220109_Baker", "a")) + "\n")
    with pytest.raises(rc.LeakError):
        rc.select_records(manifests, "220109_Baker", "rgb", 1)


def write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def rec(site_day: str, image: str, boxes: list | None = None) -> dict:
    record = {
        "site_day": site_day,
        "collection_id": image.split("/")[0],
        "camera": "thermal",
        "image": image,
        "size": [640, 512],
    }
    if boxes is not None:
        record["boxes"] = boxes
    return record


def test_record_boxes_convert_normalised_centres_to_pixels():
    record = {
        "size": [640, 512],
        "boxes": [{"x_center": 0.5, "y_center": 0.25, "width": 0.1, "height": 0.2}],
    }
    assert rc.record_boxes(record)[0] == pytest.approx([288.0, 76.8, 352.0, 179.2])


def test_a_training_manifest_holding_a_test_site_day_frame_is_refused(tmp_path):
    path = write(
        tmp_path / "train.jsonl",
        [
            rec("210417_MtErie", "210417_MtErie_IR_1/a.jpg", []),
            rec("210417_MtErie", "220109_Baker_IR_1/b.jpg", []),  # leaked path
        ],
    )
    with pytest.raises(rc.LeakError, match="220109_Baker"):
        rc.load_split(path, fold="220109_Baker", role="train")
    path = write(
        tmp_path / "v.jsonl", [rec("220109_Baker", "220109_Baker_I/b.jpg", [])]
    )
    with pytest.raises(rc.LeakError):
        rc.load_split(path, fold="220109_Baker", role="validation")


def test_a_test_manifest_must_hold_only_the_test_site_day(tmp_path):
    path = write(
        tmp_path / "t.jsonl", [rec("210417_MtErie", "210417_MtErie_I/a.jpg", [])]
    )
    with pytest.raises(rc.LeakError):
        rc.load_split(path, fold="220109_Baker", role="test")


def test_only_frames_the_manifest_lists_as_labelled_become_negatives(tmp_path):
    empty = rec("210417_MtErie", "210417_MtErie_IR_1/a.jpg", [])
    unlabelled = rec("210417_MtErie", "210417_MtErie_IR_1/b.jpg")  # no "boxes"
    path = write(tmp_path / "train.jsonl", [empty])
    records = rc.load_split(path, fold="220109_Baker", role="train")
    assert [r["image"] for r in records] == [empty["image"]]
    assert rc.record_boxes(records[0]).shape == (0, 4)

    path = write(tmp_path / "bad.jsonl", [empty, unlabelled])
    with pytest.raises(ValueError, match="no boxes field"):
        rc.load_split(path, fold="220109_Baker", role="train")


# --- the cache a run reads ---------------------------------------------------


def test_subsample_spreads_its_picks_and_keeps_all_when_unlimited():
    records = [{"i": i} for i in range(10)]
    assert [r["i"] for r in rc.subsample(records, 3)] == [0, 4, 9]
    assert rc.subsample(records, 0) == records
    assert rc.subsample(records, 20) == records
