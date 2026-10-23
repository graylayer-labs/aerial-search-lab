"""The light detection head on cached features (#66), tested on fake features."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from aerial_search.evaluation.detection import (
    GroundTruth,
    Prediction,
    evaluate_detections,
)
from aerial_search.experiments import head_experiment as he
from aerial_search.experiments.feature_cache import CacheError
from aerial_search.models.head import CentreHead, centre_loss

# Geometry as the real cache records it (index.jsonl), with a small feature dim.
RGB = {
    "camera": "rgb",
    "image_size": [3840, 2160],
    "resized_size": [672, 384],
    "patch_grid": [24, 42],
    "pooling": 2,
    "grid": [12, 21],
    "dim": 8,
}
THERMAL = {
    "camera": "thermal",
    "image_size": [640, 512],
    "resized_size": [560, 448],
    "patch_grid": [28, 35],
    "pooling": 2,
    "grid": [14, 18],
    "dim": 8,
}


def box(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# --- target assignment -----------------------------------------------------


def test_rgb_person_lands_in_its_cell_and_subcell_with_its_offset():
    geometry = he.Geometry.from_entry(RGB, upsample=4)
    targets = he.assign_targets(geometry, np.array([box(1000.0, 500.0, 12, 20)]))

    # cell column 5 spans 914.29..1097.14 (3840/21 wide); 1000 is 0.46875 of
    # the way across, so sub-column 1 of 4 at offset 0.875: column 5*4+1.
    # cell row 2 spans 360..540; 500 is 0.7778 across, sub-row 3 at 0.1111.
    assert targets.heat.shape == (12 * 4, 21 * 4)
    assert np.argwhere(targets.heat == 1).tolist() == [[11, 21]]
    assert targets.heat.sum() == 1
    assert targets.offset[:, 11, 21] == pytest.approx([0.875, 1 / 9], abs=1e-9)
    assert targets.size[:, 11, 21] == pytest.approx(
        [math.log(12 / (3840 / 21)), math.log(20 / 180)]
    )
    assert targets.lost == 0


def test_thermal_person_in_the_narrow_edge_column_is_placed_inside_it():
    geometry = he.Geometry.from_entry(THERMAL, upsample=4)
    # 35 patch columns pooled by 2: the last cell holds one patch, so it spans
    # 640*34/35 = 621.714 to 640, 18.29 px, not 36.57.
    x0, width = 640 * 34 / 35, 640 / 35
    targets = he.assign_targets(geometry, np.array([box(630.0, 100.0, 6, 10)]))

    position = (630.0 - x0) / width * 4  # 1.8125 sub-columns into the edge cell
    row = 100.0 / (512 / 14) * 4  # rows are all full: 28 patches pooled by 2
    assert np.argwhere(targets.heat == 1).tolist() == [[int(row), 17 * 4 + 1]]
    assert targets.offset[:, int(row), 69] == pytest.approx(
        [position - 1, row - int(row)], abs=1e-9
    )
    # the size is normalised by a full cell, not the narrow one
    assert targets.size[0, int(row), 69] == pytest.approx(math.log(6 / (640 / 17.5)))


def test_a_person_on_the_far_edge_stays_in_the_last_subcell():
    geometry = he.Geometry.from_entry(THERMAL, upsample=4)
    targets = he.assign_targets(geometry, np.array([[636.0, 508.0, 640.0, 512.0]]))
    assert np.argwhere(targets.heat == 1).tolist() == [[14 * 4 - 1, 18 * 4 - 1]]


def test_two_people_in_one_subcell_are_counted_as_lost():
    geometry = he.Geometry.from_entry(RGB, upsample=4)
    boxes = np.array([box(1000.0, 500.0, 12, 20), box(1001.0, 501.0, 12, 20)])
    assert he.assign_targets(geometry, boxes).lost == 1


def at_subcells(geometry: he.Geometry, cells: list[tuple[int, int]]) -> np.ndarray:
    """10-px people centred in the given (row, column) sub-cells."""
    xe, ye = geometry.x_edges, geometry.y_edges
    return np.array(
        [
            box((xe[c] + xe[c + 1]) / 2, (ye[r] + ye[r + 1]) / 2, 10, 10)
            for r, c in cells
        ]
    ).reshape(-1, 4)


@pytest.mark.parametrize(
    ("cells", "shared", "adjacent", "unreachable"),
    [
        ([(11, 21), (30, 60)], 0, 0, 0),  # far apart
        ([(11, 21), (12, 22)], 0, 2, 1),  # diagonal neighbours: one peak survives
        ([(11, 21), (11, 22), (11, 23)], 0, 3, 1),  # a row of three keeps the ends
        ([(11, 21), (11, 23)], 0, 0, 0),  # one empty sub-cell between them
        ([(11, 21), (11, 21), (11, 22)], 1, 2, 1),  # shared, then adjacent
        ([(0, 0), (0, 1), (1, 0), (1, 1)], 0, 4, 3),  # a 2x2 block keeps one
    ],
)
def test_crowding_counts_people_a_3x3_peak_test_cannot_all_keep(
    cells, shared, adjacent, unreachable
):
    geometry = he.Geometry.from_entry(RGB, upsample=4)
    crowding = he.crowding(geometry, at_subcells(geometry, cells))
    assert (crowding.shared, crowding.adjacent, crowding.unreachable) == (
        shared,
        adjacent,
        unreachable,
    )


# --- decoding and scoring --------------------------------------------------


def perfect_outputs(
    targets: he.Targets,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = np.where(targets.heat == 1, 10.0, -10.0)
    return heat, targets.offset, targets.size


@pytest.mark.parametrize("entry", [RGB, THERMAL], ids=["rgb", "thermal"])
def test_perfect_predictions_decode_to_a_perfect_score(entry):
    geometry = he.Geometry.from_entry(entry, upsample=4)
    w, h = entry["image_size"]
    people = {
        "a": np.array([box(0.2 * w, 0.3 * h, 9, 15), box(0.97 * w, 0.9 * h, 5, 7)]),
        "b": np.array([box(0.5 * w, 0.5 * h, 30, 40)]),
        "c": np.zeros((0, 4)),  # a labelled-empty frame
    }
    truths, predictions = [], []
    for name, boxes in people.items():
        targets = he.assign_targets(geometry, boxes)
        found, scores = he.decode(*perfect_outputs(targets), geometry, top_k=50)
        truths.append(GroundTruth(name, boxes, "rgb", "clip"))
        predictions.append(Prediction(name, found, scores))
        assert found.shape[0] >= len(boxes)

    report = evaluate_detections(truths, predictions)
    assert report.overall.ap_iou25 == pytest.approx(1.0)
    assert report.overall.ap_iou50 == pytest.approx(1.0)


def test_record_boxes_convert_normalised_centres_to_pixels():
    record = {
        "size": [640, 512],
        "boxes": [{"x_center": 0.5, "y_center": 0.25, "width": 0.1, "height": 0.2}],
    }
    assert he.record_boxes(record)[0] == pytest.approx([288.0, 76.8, 352.0, 179.2])


# --- the head ----------------------------------------------------------------


def test_the_head_upsamples_the_grid_by_the_factor():
    head = CentreHead(dim=8, hidden=16, upsample=4)
    heat, offset, size = head(torch.zeros(2, 14, 18, 8))
    assert heat.shape == (2, 56, 72)
    assert offset.shape == size.shape == (2, 2, 56, 72)
    assert ((offset >= 0) & (offset <= 1)).all()


def fake_split(entry, n_images: int, seed: int) -> he.Split:
    rng = np.random.default_rng(seed)
    gh, gw = entry["grid"]
    w, h = entry["image_size"]
    features = rng.standard_normal((n_images, gh, gw, entry["dim"])).astype("f2")
    records, boxes = [], []
    for k in range(n_images):
        count = k % 3  # some frames are labelled-empty
        cx = rng.uniform(0.05, 0.95, count) * w
        cy = rng.uniform(0.05, 0.95, count) * h
        b = np.array([box(x, y, 10, 16) for x, y in zip(cx, cy, strict=True)])
        boxes.append(b.reshape(-1, 4))
        records.append({"image": f"clip/{k}.jpg", "collection_id": "clip"})
    geometry = [he.Geometry.from_entry(entry, upsample=4)] * n_images
    return he.Split.build(records, features, geometry, boxes, camera=entry["camera"])


def test_the_head_overfits_a_tiny_fake_set_and_scores_on_it():
    split = fake_split(RGB, n_images=6, seed=0)
    torch.manual_seed(0)
    head = CentreHead(dim=8, hidden=64, upsample=4)
    history = he.train_head(
        head,
        split,
        he.Recipe(steps=400, batch_size=6, learning_rate=3e-3, eval_every=400),
        device=torch.device("cpu"),
    )
    assert history["loss"][-1] < 0.1 * history["loss"][0]
    report = he.evaluate(head, split, torch.device("cpu"), top_k=20)
    assert report.overall.ap_iou25 > 0.9


def test_centre_loss_is_lower_for_the_right_answer():
    targets = he.assign_targets(
        he.Geometry.from_entry(RGB, upsample=4), np.array([box(1000, 500, 12, 20)])
    )
    batch = {k: torch.as_tensor(v)[None] for k, v in he.as_arrays(targets).items()}
    heat, offset, size = (torch.as_tensor(a)[None] for a in perfect_outputs(targets))
    right = centre_loss((heat.float(), offset.float(), size.float()), batch)
    wrong = centre_loss((-heat.float(), 1 - offset.float(), size.float() + 1), batch)
    assert right < 0.01 < wrong


# --- splits: leakage and labelled-empty frames -------------------------------


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


def test_a_training_manifest_holding_a_test_site_day_frame_is_refused(tmp_path):
    path = write(
        tmp_path / "train.jsonl",
        [
            rec("210417_MtErie", "210417_MtErie_IR_1/a.jpg", []),
            rec("210417_MtErie", "220109_Baker_IR_1/b.jpg", []),  # leaked path
        ],
    )
    with pytest.raises(he.LeakError, match="220109_Baker"):
        he.load_split(path, fold="220109_Baker", role="train")
    path = write(
        tmp_path / "v.jsonl", [rec("220109_Baker", "220109_Baker_I/b.jpg", [])]
    )
    with pytest.raises(he.LeakError):
        he.load_split(path, fold="220109_Baker", role="validation")


def test_a_test_manifest_must_hold_only_the_test_site_day(tmp_path):
    path = write(
        tmp_path / "t.jsonl", [rec("210417_MtErie", "210417_MtErie_I/a.jpg", [])]
    )
    with pytest.raises(he.LeakError):
        he.load_split(path, fold="220109_Baker", role="test")


def test_only_frames_the_manifest_lists_as_labelled_become_negatives(tmp_path):
    empty = rec("210417_MtErie", "210417_MtErie_IR_1/a.jpg", [])
    unlabelled = rec("210417_MtErie", "210417_MtErie_IR_1/b.jpg")  # no "boxes"
    path = write(tmp_path / "train.jsonl", [empty])
    records = he.load_split(path, fold="220109_Baker", role="train")
    assert [r["image"] for r in records] == [empty["image"]]
    assert he.record_boxes(records[0]).shape == (0, 4)

    path = write(tmp_path / "bad.jsonl", [empty, unlabelled])
    with pytest.raises(ValueError, match="no boxes field"):
        he.load_split(path, fold="220109_Baker", role="train")


# --- the cache a run reads ---------------------------------------------------


def fake_cache(
    tmp_path: Path,
    records: list[dict],
    scratch: bool = False,
    settings: dict | None = None,
) -> Path:
    top = "scratch-features" if scratch else "features"
    cache = tmp_path / "outputs" / top / "m-1024tok"
    (cache / "runs" / "s1").mkdir(parents=True)
    (cache / "cache.json").write_text(
        json.dumps({"settings": {"model": "m", **(settings or {})}})
    )
    (cache / "runs" / "s1" / "run.json").write_text(
        json.dumps({"scratch": scratch, "status": "completed"})
    )
    lines = []
    for r in records:
        path = cache / "features" / Path(r["image"]).with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros((14, 18, 8), dtype="f2"))
        lines.append(json.dumps({**THERMAL, "path": r["image"], "run": "s1"}))
    (cache / "index.jsonl").write_text("\n".join(lines) + "\n")
    return cache


def test_a_normal_run_refuses_a_scratch_cache(tmp_path):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records, scratch=True)
    with pytest.raises(CacheError, match="scratch"):
        he.check_cache(cache, records, "thermal", scratch=False)
    he.check_cache(cache, records, "thermal", scratch=True)  # a scratch run may


def test_a_normal_run_refuses_a_cache_missing_any_frame_it_needs(tmp_path):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records)
    missing = [*records, rec("x", "210417_MtErie_I/b.jpg", [])]
    with pytest.raises(CacheError, match="1 of 2"):
        he.check_cache(cache, missing, "thermal", scratch=False)

    # a scratch run, for debugging only, leaves them out and lists them
    view = he.check_cache(cache, missing, "thermal", scratch=True)
    assert view.missing == ["210417_MtErie_I/b.jpg"]
    assert list(view.entries) == ["210417_MtErie_I/a.jpg"]


def test_a_normal_run_refuses_frames_written_by_a_scratch_session(tmp_path):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records)
    (cache / "runs" / "s1" / "run.json").write_text(json.dumps({"scratch": True}))
    with pytest.raises(CacheError, match="s1"):
        he.check_cache(cache, records, "thermal", scratch=False)


@pytest.mark.parametrize("status", ["started", "failed", None])
def test_a_normal_run_refuses_features_from_an_unfinished_session(tmp_path, status):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records)
    record = {"scratch": False, **({"status": status} if status else {})}
    (cache / "runs" / "s1" / "run.json").write_text(json.dumps(record))
    with pytest.raises(CacheError, match="s1.*not completed"):
        he.check_cache(cache, records, "thermal", scratch=False)
    he.check_cache(cache, records, "thermal", scratch=True)


def test_a_normal_run_refuses_a_cache_being_written(tmp_path):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records)
    (cache / "session.lock").write_text("pid 1 session s2\n")
    with pytest.raises(CacheError, match="session.lock"):
        he.check_cache(cache, records, "thermal", scratch=False)
    he.check_cache(cache, records, "thermal", scratch=True)


def test_a_cache_entry_whose_image_size_disagrees_is_refused(tmp_path):
    records = [rec("x", "210417_MtErie_I/a.jpg", [])]
    cache = fake_cache(tmp_path, records)
    other = [{**records[0], "size": [320, 256]}]
    with pytest.raises(CacheError, match="size"):
        he.check_cache(cache, other, "thermal", scratch=False)


# --- one run, end to end, on a fake cache -----------------------------------


def scratch_setup(tmp_path: Path, settings: dict | None = None) -> tuple[Path, str]:
    """Fold manifests and a fake scratch cache for a tiny end-to-end run."""
    fold, other = "220109_Baker", "210417_MtErie"
    rng = np.random.default_rng(1)

    def frames(site: str, n: int) -> list[dict]:
        out = []
        for k in range(n):
            x, y = rng.uniform(0.1, 0.9, 2)
            b = [{"x_center": x, "y_center": y, "width": 0.02, "height": 0.03}]
            out.append(rec(site, f"{site}_Enterprise_IR_1/{k}.jpg", b if k % 2 else []))
        return out

    train, validation, test = frames(other, 6), frames(other, 3), frames(fold, 3)
    view = tmp_path / "manifests" / "folds" / fold / "thermal"
    write(view / "train_100pct.jsonl", train)
    write(view / "validation.jsonl", validation)
    write(view / "test.jsonl", test)
    cache = fake_cache(
        tmp_path, train + validation + test, scratch=True, settings=settings
    )
    return cache, fold


def tiny_run(tmp_path: Path, cache: Path, fold: str, run_name="head-test", **kw):
    return he.run_head(
        cache_dir=cache,
        manifests=tmp_path / "manifests",
        data_root=tmp_path / "raw",
        fold=fold,
        camera="thermal",
        percent=100,
        seed=7,
        recipe=he.Recipe(steps=4, batch_size=2, eval_every=2, hidden=8),
        scratch=True,
        run_name=run_name,
        device=torch.device("cpu"),
        repo=tmp_path / "repo",
        argv=["aerial-search", "train-head"],
        **kw,
    )


def test_a_scratch_run_trains_selects_and_writes_its_report(tmp_path):
    cache, fold = scratch_setup(tmp_path)

    summary = tiny_run(tmp_path, cache, fold)

    directory = tmp_path / "repo" / "outputs" / "scratch-head-test"
    record = json.loads((directory / "run.json").read_text())
    assert record["status"] == "completed" and record["scratch"] is True
    assert record["fold"] == fold and record["view"] == "thermal"
    assert record["config"]["percent"] == 100
    hashed = {Path(i["path"]).name for i in record["inputs"]}
    assert {"cache.json", "index.jsonl", "test.jsonl"} <= hashed
    metrics = json.loads((directory / "detection_metrics.json").read_text())
    assert metrics["overall"]["n_images"] == 3  # scored on the test site-day only
    assert summary["test"]["n_images"] == 3
    assert (directory / "head.pt").exists()
    training = json.loads((directory / "training.json").read_text())
    crowd = training["crowding"]["test"]
    assert set(crowd) == {"people", "shared", "adjacent", "unreachable"}
    assert crowd["people"] == 1  # frames 1 of 3 carry one person each
    tied = metrics["overall"]["n_predictions_tied"]
    assert training["test_predictions_tied"] == tied == summary["test_predictions_tied"]


def run_record(tmp_path: Path) -> dict:
    path = tmp_path / "repo" / "outputs" / "scratch-head-test" / "run.json"
    return json.loads(path.read_text())


def test_a_run_records_which_input_arm_its_features_came_from(tmp_path):
    cache, fold = scratch_setup(tmp_path, {"input_handling": "equalise"})

    tiny_run(tmp_path, cache, fold, input_handling="equalise")

    assert run_record(tmp_path)["config"]["input_handling"] == "equalise"


def test_a_cache_without_the_field_is_arm_replicate(tmp_path):
    cache, fold = scratch_setup(tmp_path)  # as the #66 cache is

    tiny_run(tmp_path, cache, fold)

    assert run_record(tmp_path)["config"]["input_handling"] == "replicate"


def test_asking_for_an_arm_the_cache_was_not_made_with_is_refused(tmp_path):
    cache, fold = scratch_setup(tmp_path, {"input_handling": "equalise"})

    with pytest.raises(CacheError, match="equalise"):
        tiny_run(tmp_path, cache, fold, input_handling="replicate")
    assert not (tmp_path / "repo" / "outputs" / "scratch-head-test").exists()


def test_run_head_starts_from_the_head_a_seed_builds(tmp_path, monkeypatch):
    seen = []

    def capture(head, *args, **kwargs):
        seen.append({k: v.clone() for k, v in head.state_dict().items()})
        raise StopIteration

    monkeypatch.setattr(he, "train_head", capture)
    cache, fold = scratch_setup(tmp_path)

    with pytest.raises(StopIteration):
        tiny_run(tmp_path, cache, fold)

    dim = seen[0]["norm.weight"].shape[0]
    torch.manual_seed(7)
    reference = CentreHead(dim=dim, hidden=8, upsample=4).state_dict()
    assert all(torch.equal(seen[0][k], reference[k]) for k in reference)


def test_a_warm_run_starts_from_the_saved_head_and_records_it(tmp_path, monkeypatch):
    cache, fold = scratch_setup(tmp_path)
    tiny_run(tmp_path, cache, fold)  # arm A, cold
    cold = tmp_path / "repo" / "outputs" / "scratch-head-test"
    saved = torch.load(cold / "head.pt")
    seen = []

    def capture(head, *args, **kwargs):
        seen.append({k: v.clone() for k, v in head.state_dict().items()})
        raise StopIteration

    monkeypatch.setattr(he, "train_head", capture)
    with pytest.raises(StopIteration):
        tiny_run(tmp_path, cache, fold, init_head=cold, run_name="warm")

    assert all(torch.equal(seen[0][k], saved[k]) for k in saved)
    monkeypatch.undo()
    tiny_run(tmp_path, cache, fold, init_head=cold, run_name="warm2")
    record = json.loads(
        (tmp_path / "repo" / "outputs" / "scratch-warm2" / "run.json").read_text()
    )
    assert record["config"]["warm_start"]["init_head"] == str(cold)
    assert len(record["config"]["warm_start"]["run_json_sha256"]) == 64


def test_a_warm_start_is_refused_from_a_warm_or_other_seed_run(tmp_path):
    cache, fold = scratch_setup(tmp_path)
    tiny_run(tmp_path, cache, fold)
    cold = tmp_path / "repo" / "outputs" / "scratch-head-test"
    path = cold / "run.json"
    record = json.loads(path.read_text())

    path.write_text(json.dumps({**record, "seed": 1}))
    with pytest.raises(CacheError, match="seed"):
        tiny_run(tmp_path, cache, fold, init_head=cold, run_name="w")

    warm = {**record["config"], "warm_start": {"init_head": "x"}}
    path.write_text(json.dumps({**record, "config": warm}))
    with pytest.raises(CacheError, match="cold"):
        tiny_run(tmp_path, cache, fold, init_head=cold, run_name="w")


# --- the results table ---------------------------------------------------------


def fake_run(
    root: Path,
    camera,
    fold,
    percent,
    ap25,
    ap50,
    recipe=None,
    cache_hash="c0",
    tiny=True,
    arm=None,
    **record,
) -> None:
    directory = root / f"{camera}-{fold}-{percent}"
    directory.mkdir(parents=True)
    config = {
        "camera": camera,
        "fold": fold,
        "percent": percent,
        "recipe": {**asdict(he.Recipe()), **(recipe or {})},
        **({} if arm is None else {"input_handling": arm}),
    }
    base = {
        "status": "completed",
        "scratch": False,
        "fold": fold,
        "commit": "abc",
        "seed": 7,
        "config": config,
        "inputs": [
            {"path": "x/test.jsonl", "sha256": "t"},
            {"path": "cache/cache.json", "sha256": cache_hash},
        ],
    }
    (directory / "run.json").write_text(json.dumps({**base, **record}))
    fppi = {"0.01": 0.0, "0.1": 0.1, "1.0": 0.5}
    bucket = {"ap_iou25": ap25, "ap_iou50": ap50, "recall_at_fppi": fppi}
    empty = {"ap_iou25": None, "ap_iou50": None, "recall_at_fppi": {}}
    metrics = {
        "overall": {"ap_iou25": ap25, "ap_iou50": ap50, "recall_at_fppi": fppi},
        "by_size": {"tiny": bucket if tiny else empty, "small": bucket},
    }
    (directory / "detection_metrics.json").write_text(json.dumps(metrics))


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"recipe": {"steps": 500}}, "recipe.steps"),
        ({"recipe": {"hidden": 64}}, "recipe.hidden"),
        ({"seed": 8}, "seed"),
        ({"cache_hash": "c1"}, "cache.json"),
    ],
)
def test_the_table_refuses_runs_made_differently(tmp_path, change, field):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "rgb", "A", 100, 0.2, 0.1)
    fake_run(tmp_path, "thermal", "B", 1, 0.4, 0.3, **change)
    with pytest.raises(head_table.TableError, match=field.replace(".", r"\.")):
        head_table.collect(tmp_path)


def test_the_table_records_commits_and_warns_when_they_differ(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "rgb", "A", 100, 0.2, 0.1)
    fake_run(tmp_path, "rgb", "B", 100, 0.4, 0.3, commit="def")
    runs = head_table.collect(tmp_path)
    with pytest.warns(UserWarning, match="abc.*def"):
        summary = head_table.summarise(runs)
    assert summary["rgb"]["100"]["fold_commits"] == {"A": "abc", "B": "def"}


def test_size_cells_say_how_many_folds_had_people_of_that_size(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "rgb", "A", 100, 0.2, 0.1, tiny=False)  # no tiny people
    fake_run(tmp_path, "rgb", "B", 100, 0.4, 0.3)
    summary = head_table.summarise(head_table.collect(tmp_path))
    sizes = summary["rgb"]["100"]["by_size"]
    assert sizes["tiny"]["n"] == 1 and sizes["small"]["n"] == 2
    assert sizes["tiny"]["ap_iou25"][0] == pytest.approx(0.4)
    text = head_table.markdown(summary)
    assert "0.400 / 0.300 (n=1)" in text
    assert "validation" in text  # the selection caveat is printed


def test_the_table_gives_each_fold_and_the_mean_and_spread(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "rgb", "A", 100, 0.2, 0.1)
    fake_run(tmp_path, "rgb", "B", 100, 0.4, 0.3)
    fake_run(tmp_path, "thermal", "A", 1, 0.5, 0.25)

    summary = head_table.summarise(head_table.collect(tmp_path))
    rgb = summary["rgb"]["100"]
    assert rgb["folds"]["A"]["ap_iou25"] == 0.2
    assert rgb["mean"]["ap_iou25"] == pytest.approx(0.3)
    assert rgb["mean"]["ap_iou50"] == pytest.approx(0.2)
    assert rgb["std"]["ap_iou25"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert rgb["n_folds"] == 2
    assert math.isnan(summary["thermal"]["1"]["std"]["ap_iou25"])  # one fold

    text = head_table.markdown(summary)
    assert "ap_iou25" in text and "ap_iou50" in text
    assert "0.300 ± 0.141" in text


@pytest.mark.parametrize(
    "record", [{"scratch": True}, {"status": "started"}, {"status": "failed"}]
)
def test_the_table_refuses_scratch_or_unfinished_runs(tmp_path, record):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "rgb", "A", 100, 0.2, 0.1)
    fake_run(tmp_path, "rgb", "B", 100, 0.4, 0.3, **record)
    with pytest.raises(head_table.TableError, match="rgb-B-100"):
        head_table.collect(tmp_path)


def test_the_table_refuses_two_runs_of_the_same_cell(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path / "one", "rgb", "A", 100, 0.2, 0.1)
    fake_run(tmp_path / "two", "rgb", "A", 100, 0.4, 0.3)
    with pytest.raises(head_table.TableError, match="twice"):
        head_table.summarise(head_table.collect(tmp_path))


# --- the input-arm ablation table (#68) ---------------------------------------


def arm_dirs(tmp_path: Path, scores: dict[str, dict[str, tuple[float, float]]]):
    """One directory per arm; scores[arm][fold] = (ap25, ap50), thermal at 10%."""
    roots = []
    for arm, folds in scores.items():
        root = tmp_path / arm
        for fold, (a25, a50) in folds.items():
            fake_run(root, "thermal", fold, 10, a25, a50, arm=arm, cache_hash=arm)
        roots.append(root)
    return roots


def test_a_run_without_an_arm_is_the_replicate_arm(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path, "thermal", "A", 10, 0.2, 0.1)  # a #66 run

    assert head_table.collect(tmp_path)[0]["arm"] == "replicate"


def test_the_ablation_table_has_each_arm_per_fold_with_spread_and_the_paired_change(
    tmp_path,
):
    from aerial_search.experiments import head_table

    roots = arm_dirs(
        tmp_path,
        {
            "replicate": {"A": (0.2, 0.1), "B": (0.4, 0.2)},
            "equalise": {"A": (0.3, 0.1), "B": (0.5, 0.3)},
        },
    )

    by_arm = head_table.collect_arms(roots, camera="thermal", percents=["10"])
    summary = head_table.summarise_arms(by_arm)
    entry = summary["thermal"]["10"]

    assert entry["replicate"]["mean"]["ap_iou25"] == pytest.approx(0.3)
    assert entry["equalise"]["mean"]["ap_iou25"] == pytest.approx(0.4)
    assert entry["equalise"]["delta"]["ap_iou25"][0] == pytest.approx(0.1)
    assert entry["equalise"]["delta"]["ap_iou25"][1] == pytest.approx(0.0, abs=1e-12)
    assert "delta" not in entry["replicate"]
    text = head_table.markdown_arms(summary, head_table.arm_notes(by_arm))
    assert "| replicate |" in text and "| equalise |" in text
    assert "0.200 / 0.100" in text and "0.500 / 0.300" in text  # per fold
    assert "0.400 ± 0.141" in text  # mean and spread of ap_iou25


def test_the_ablation_needs_the_control_arm(tmp_path):
    from aerial_search.experiments import head_table

    roots = arm_dirs(tmp_path, {"equalise": {"A": (0.3, 0.1)}})
    by_arm = head_table.collect_arms(roots, camera="thermal", percents=["10"])

    with pytest.raises(head_table.TableError, match="replicate"):
        head_table.summarise_arms(by_arm)


def test_one_directory_may_hold_one_arm_and_an_arm_may_not_appear_twice(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path / "x", "thermal", "A", 10, 0.2, 0.1, arm="replicate")
    fake_run(tmp_path / "x", "thermal", "B", 10, 0.2, 0.1, arm="equalise")
    with pytest.raises(head_table.TableError, match="one arm"):
        head_table.collect_arms([tmp_path / "x"], camera="thermal", percents=["10"])

    one = arm_dirs(tmp_path / "one", {"replicate": {"A": (0.2, 0.1)}})
    two = arm_dirs(tmp_path / "two", {"replicate": {"A": (0.4, 0.3)}})
    with pytest.raises(head_table.TableError, match="twice"):
        head_table.collect_arms([*one, *two], camera="thermal", percents=["10"])


def test_arms_must_share_a_seed_but_a_recipe_difference_is_stated_not_hidden(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path / "a", "thermal", "A", 10, 0.2, 0.1, arm="replicate")
    fake_run(
        tmp_path / "c",
        "thermal",
        "A",
        10,
        0.3,
        0.1,
        arm="stem",
        recipe={"steps": 300},
        cache_hash="geometry",
    )
    by_arm = head_table.collect_arms(
        [tmp_path / "a", tmp_path / "c"], camera="thermal", percents=["10"]
    )

    notes = head_table.arm_notes(by_arm)

    assert any("stem" in n and "recipe.steps" in n and "300" in n for n in notes)
    assert "recipes differ" in head_table.markdown_arms(
        head_table.summarise_arms(by_arm), notes
    )
    fake_run(tmp_path / "d", "thermal", "A", 10, 0.3, 0.1, arm="equalise", seed=9)
    with pytest.raises(head_table.TableError, match="seed"):
        head_table.collect_arms(
            [tmp_path / "a", tmp_path / "d"], camera="thermal", percents=["10"]
        )


def test_warm_arms_are_compared_with_the_warm_control_not_the_cold_arm(tmp_path):
    from aerial_search.experiments import head_table

    for arm, a25 in (("replicate", 0.5), ("replicate-warm", 0.2), ("stem-warm", 0.3)):
        root = tmp_path / arm
        fake_run(root, "thermal", "A", 10, a25, 0.1, arm=arm.split("-")[0])
        if arm.endswith("warm"):
            path = root / "thermal-A-10" / "run.json"
            record = json.loads(path.read_text())
            record["config"]["warm_start"] = {"init_head": "h"}
            path.write_text(json.dumps(record))

    by_arm = head_table.collect_arms(
        [tmp_path / a for a in ("replicate", "replicate-warm", "stem-warm")],
        camera="thermal",
        percents=["10"],
    )
    entry = head_table.summarise_arms(by_arm)["thermal"]["10"]

    assert set(entry) == {"replicate", "replicate-warm", "stem-warm"}
    assert entry["stem-warm"]["delta"]["ap_iou25"][0] == pytest.approx(0.1)
    assert "delta" not in entry["replicate-warm"]
    assert "delta" not in entry["replicate"]


def test_an_arm_whose_cache_settings_differ_is_noted(tmp_path):
    from aerial_search.experiments import head_table

    base = {"model": "m", "pooling": 2, "token_budget": 1024}
    for arm, settings in (
        ("replicate", {**base}),
        ("equalise", {**base, "input_handling": "equalise", "pooling": 1}),
    ):
        fake_run(tmp_path / arm, "thermal", "A", 10, 0.2, 0.1, arm=arm)
        path = tmp_path / arm / "thermal-A-10" / "run.json"
        record = json.loads(path.read_text())
        record["config"]["cache"] = {"settings": settings}
        path.write_text(json.dumps(record))
    by_arm = head_table.collect_arms(
        [tmp_path / "replicate", tmp_path / "equalise"],
        camera="thermal",
        percents=["10"],
    )

    notes = head_table.arm_notes(by_arm)

    assert any("pooling" in n for n in notes)
    assert not any("input_handling" in n for n in notes)


def test_the_ablation_warns_when_arms_come_from_different_commits(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path / "a", "thermal", "A", 10, 0.2, 0.1, arm="replicate")
    fake_run(tmp_path / "b", "thermal", "A", 10, 0.3, 0.1, arm="equalise", commit="def")
    by_arm = head_table.collect_arms(
        [tmp_path / "a", tmp_path / "b"], camera="thermal", percents=["10"]
    )

    with pytest.warns(UserWarning, match="2 commits"):
        head_table.summarise_arms(by_arm)


def test_the_plain_table_refuses_warm_and_non_replicate_runs(tmp_path):
    from aerial_search.experiments import head_table

    fake_run(tmp_path / "w", "thermal", "A", 10, 0.2, 0.1)
    path = tmp_path / "w" / "thermal-A-10" / "run.json"
    record = json.loads(path.read_text())
    record["config"]["warm_start"] = {"init_head": "h"}
    path.write_text(json.dumps(record))
    with pytest.raises(head_table.TableError, match="--ablation"):
        head_table.collect(tmp_path / "w")

    fake_run(tmp_path / "e", "thermal", "A", 10, 0.2, 0.1, arm="equalise")
    with pytest.raises(head_table.TableError, match="--ablation"):
        head_table.collect(tmp_path / "e")
    assert head_table.collect(tmp_path / "e", ablation=True)[0]["arm"] == "equalise"


def test_a_warm_start_is_refused_for_the_equalise_arm(tmp_path):
    cache, fold = scratch_setup(tmp_path, {"input_handling": "equalise"})

    with pytest.raises(CacheError, match="control and the stem arm"):
        tiny_run(tmp_path, cache, fold, init_head=tmp_path)
