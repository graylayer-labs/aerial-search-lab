"""The head and the supervised baseline label exactly the same frames (#67)."""

import json
from pathlib import Path

import pytest

from aerial_search.experiments import head_experiment as he


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
    paths, records = he.select_records(manifests, "220109_Baker", "rgb", 5)
    assert paths["train"].name == "train_5pct.jsonl"
    assert [r["image"].split("/")[1] for r in records["train"]] == ["a.jpg", "b.jpg"]
    assert len(records["validation"]) == 2 and len(records["test"]) == 1


def test_select_records_still_refuses_a_leaking_manifest(tmp_path):
    manifests = _fold(tmp_path)
    leaked = manifests / "folds" / "220109_Baker" / "rgb" / "train_1pct.jsonl"
    leaked.write_text(json.dumps(_rec("220109_Baker", "a")) + "\n")
    with pytest.raises(he.LeakError):
        he.select_records(manifests, "220109_Baker", "rgb", 1)
