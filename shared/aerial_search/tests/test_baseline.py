"""The supervised baseline (#67): same frames, same scoring, boxes in source pixels."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aerial_search.experiments import baseline_experiment as be
from aerial_search.experiments import head_experiment as he

FOLD, OTHER = "220109_Baker", "210417_MtErie"
MANIFESTS = Path("data/manifests/wisard-full")


def rec(site: str, name: str, boxes: list[dict]) -> dict:
    return {
        "site_day": site,
        "collection_id": f"{site}_Enterprise_1",
        "camera": "thermal",
        "image": f"{site}_Enterprise_IR_1/{name}.jpg",
        "size": [64, 48],
        "boxes": boxes,
    }


def person(x: float, y: float) -> dict:
    return {"x_center": x, "y_center": y, "width": 0.25, "height": 0.25}


def fake_fold(tmp_path: Path) -> tuple[Path, Path]:
    manifests, raw = tmp_path / "manifests", tmp_path / "raw"
    view = manifests / "folds" / FOLD / "thermal"
    view.mkdir(parents=True)
    splits = {
        "train_100pct.jsonl": [
            rec(OTHER, f"t{k}", [person(0.5, 0.5)] * (k % 2)) for k in range(4)
        ],
        "validation.jsonl": [rec(OTHER, f"v{k}", [person(0.3, 0.6)]) for k in range(2)],
        "test.jsonl": [rec(FOLD, f"x{k}", [person(0.4, 0.4)]) for k in range(3)],
    }
    for file, records in splits.items():
        (view / file).write_text("".join(json.dumps(r) + "\n" for r in records))
        for r in records:
            path = raw / r["image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((48, 64, 3), 90, dtype="u1")).save(path)
    return manifests, raw


REAL_BUILD = be.build_model


def tiny_model(arch, **kwargs):
    kwargs["pretrained"] = False  # no download in tests
    return REAL_BUILD(arch, **kwargs)


def test_baseline_and_head_select_the_same_frames_in_every_cell():
    if not MANIFESTS.exists():
        pytest.skip("fold manifests not available")
    for fold in sorted(p.name for p in (MANIFESTS / "folds").iterdir() if p.is_dir()):
        for camera in ("rgb", "thermal"):
            for percent in (1, 5, 10, 100):
                head_paths, head = he.select_records(MANIFESTS, fold, camera, percent)
                path = MANIFESTS / "folds" / fold / camera / f"train_{percent}pct.jsonl"
                assert head_paths["train"] == path
                on_disk = [
                    json.loads(x)["image"] for x in path.read_text().splitlines()
                ]
                assert [r["image"] for r in head["train"]] == on_disk


def test_a_scratch_run_uses_the_selected_frames_and_scores_the_test_site_day(
    tmp_path, monkeypatch
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(be, "build_model", tiny_model)
    seen = {}
    real = be.Frames

    class Spy(real):
        def __init__(self, records, data_root, size):
            seen[len(seen)] = [r["image"] for r in records]
            super().__init__(records, data_root, size)

    monkeypatch.setattr(be, "Frames", Spy)
    recipe = be.BaselineRecipe(
        input="native",
        steps=2,
        batch_size=2,
        eval_every=1,
        val_frames=0,
        workers=0,
        warmup_steps=1,
    )
    summary = be.run_baseline(
        manifests=manifests,
        data_root=raw,
        fold=FOLD,
        camera="thermal",
        percent=100,
        seed=7,
        recipe=recipe,
        scratch=True,
        run_name="bl-test",
        device=torch.device("cpu"),
        repo=tmp_path / "repo",
        argv=["x"],
    )
    _, expected = he.select_records(manifests, FOLD, "thermal", 100)
    assert list(seen.values()) == [
        [r["image"] for r in expected[k]] for k in ("train", "validation", "test")
    ]
    directory = tmp_path / "repo" / "outputs" / "scratch-bl-test"
    run = json.loads((directory / "run.json").read_text())
    assert run["status"] == "completed" and run["scratch"] is True
    assert run["config"]["camera"] == "thermal" and run["config"]["percent"] == 100
    assert {Path(i["path"]).name for i in run["inputs"]} == {
        "train_100pct.jsonl",
        "validation.jsonl",
        "test.jsonl",
    }
    metrics = json.loads((directory / "detection_metrics.json").read_text())
    assert metrics["overall"]["n_images"] == summary["test"]["n_images"] == 3


def test_frames_scale_boxes_with_the_image_and_predictions_scale_back(tmp_path):
    _, raw = fake_fold(tmp_path)
    record = rec(FOLD, "x0", [person(0.5, 0.5)])  # 16 x 12 px box at 64 x 48
    frames = be.Frames([record], raw, (32, 24))
    image, target, scale = frames[0]
    assert image.shape == (3, 24, 32)
    assert target["boxes"].tolist() == [[12.0, 9.0, 20.0, 15.0]]
    assert scale.tolist() == [0.5, 0.5]

    class Echo(torch.nn.Module):  # predicts the ground truth it is given
        def forward(self, images):
            return [
                {
                    "boxes": torch.tensor([[12.0, 9.0, 20.0, 15.0]]),
                    "scores": torch.tensor([0.9]),
                }
            ]

    (prediction,) = be.predict(Echo(), frames, torch.device("cpu"))
    assert np.asarray(prediction.boxes).tolist() == [[24.0, 18.0, 40.0, 30.0]]


def test_a_manifest_size_that_disagrees_with_the_image_is_refused(tmp_path):
    _, raw = fake_fold(tmp_path)
    record = {**rec(FOLD, "x0", []), "size": [128, 96]}
    with pytest.raises(ValueError, match="on disk"):
        be.Frames([record], raw, None)[0]


def test_batches_follow_the_heads_running_order():
    batches = list(be.BatchSteps(3, 2, 4, seed=1))
    assert [len(b) for b in batches] == [2, 2, 2, 2]
    assert sorted(sum(batches[:3], [])) == [0, 0, 1, 1, 2, 2]  # two full passes


def test_input_size_is_the_caches_model_input_or_native():
    assert be.input_size("rgb", "cache") == (672, 384)
    assert be.input_size("thermal", "cache") == (560, 448)
    assert be.input_size("rgb", "native") is None


# --- one table for the head and the baseline --------------------------------


def fake_table_run(root, camera, fold, percent, ap25, ap50, method, train_hash="h"):
    directory = root / f"{camera}-{fold}-{percent}"
    directory.mkdir(parents=True)
    config = {
        "method": method,
        "camera": camera,
        "fold": fold,
        "percent": percent,
        "recipe": {"steps": 1, "val_frames": 100},
    }
    inputs = [
        {
            "path": f"m/folds/{fold}/{camera}/train_{percent}pct.jsonl",
            "sha256": train_hash,
        },
        {"path": f"m/folds/{fold}/{camera}/validation.jsonl", "sha256": "v"},
        {"path": f"m/folds/{fold}/{camera}/test.jsonl", "sha256": "t"},
    ]
    run = {
        "status": "completed",
        "scratch": False,
        "commit": "c",
        "seed": 7,
        "config": config,
        "inputs": inputs,
    }
    (directory / "run.json").write_text(json.dumps(run))
    overall = {"ap_iou25": ap25, "ap_iou50": ap50, "recall_at_fppi": {}}
    (directory / "detection_metrics.json").write_text(
        json.dumps({"overall": overall, "by_size": {}})
    )


def test_one_table_holds_head_and_baseline_rows_per_fold_with_mean_and_spread(tmp_path):
    from aerial_search.experiments import head_table as ht

    for fold, a, b in (("A", 0.2, 0.4), ("B", 0.4, 0.6)):
        fake_table_run(tmp_path / "head", "rgb", fold, 100, a, a / 2, "head")
        fake_table_run(
            tmp_path / "base", "rgb", fold, 100, b, b / 2, "baseline:x@cache"
        )
    text = ht.markdown_compare(
        {
            "head": ht.summarise(ht.collect(tmp_path / "head")),
            "baseline:x@cache": ht.summarise(ht.collect(tmp_path / "base")),
        }
    )
    assert "| 100% | head | 0.200 / 0.100 | 0.400 / 0.200 | 0.300 ± 0.141" in text
    assert (
        "| 100% | baseline:x@cache | 0.400 / 0.200 | 0.600 / 0.300 | 0.500 ± 0.141"
        in text
    )


def test_the_comparison_refuses_runs_that_saw_different_frames(tmp_path):
    from aerial_search.experiments import head_table as ht

    fake_table_run(tmp_path / "head", "rgb", "A", 1, 0.2, 0.1, "head", train_hash="h1")
    fake_table_run(
        tmp_path / "base", "rgb", "A", 1, 0.3, 0.1, "baseline", train_hash="h2"
    )
    with pytest.raises(ht.TableError, match="train_1pct.jsonl"):
        ht.check_same_frames(
            {
                "head": ht.collect(tmp_path / "head"),
                "baseline": ht.collect(tmp_path / "base"),
            }
        )


def test_the_default_architecture_returns_boxes_in_the_pixels_it_is_given():
    model = be.build_model(
        be.BaselineRecipe().arch,
        pretrained=False,
        anchor_sizes=be.ANCHORS[be.BaselineRecipe().arch],
    ).eval()
    (out,) = model([torch.zeros(3, 96, 160)])  # no internal resize
    assert out["boxes"].shape[1] == 4
    assert out["boxes"].numel() == 0 or float(out["boxes"].max()) <= 160


def test_every_architectures_anchors_fit_its_rpn_head():
    for arch, sizes in be.ANCHORS.items():
        model = be.build_model(arch, pretrained=False, anchor_sizes=sizes)
        per_location = model.rpn.anchor_generator.num_anchors_per_location()
        assert len(per_location) == len(sizes)
        assert model.rpn.head.cls_logits.out_channels == per_location[0]
        assert model.rpn.head.bbox_pred.out_channels == 4 * per_location[0]


def test_anchors_that_do_not_fit_the_rpn_head_are_refused():
    with pytest.raises(ValueError, match="anchors per location"):
        be.build_model(
            "fasterrcnn_mobilenet_v3_large_fpn",
            pretrained=False,
            anchor_sizes=((8,), (16,), (32,)),
        )


def test_a_normal_run_refuses_a_device_other_than_cpu(tmp_path):
    with pytest.raises(ValueError, match="cpu"):
        be.run_baseline(
            manifests=tmp_path,
            data_root=tmp_path,
            fold=FOLD,
            camera="thermal",
            percent=1,
            seed=0,
            recipe=be.BaselineRecipe(),
            scratch=False,
            run_name=None,
            device=torch.device("mps"),
        )


def test_the_comparison_refuses_a_missing_cell_or_a_missing_split_file(tmp_path):
    from aerial_search.experiments import head_table as ht

    fake_table_run(tmp_path / "head", "rgb", "A", 1, 0.2, 0.1, "head")
    fake_table_run(tmp_path / "head2", "rgb", "A", 5, 0.2, 0.1, "head")
    fake_table_run(tmp_path / "base", "rgb", "A", 1, 0.3, 0.1, "baseline")
    head = ht.collect(tmp_path / "head") + ht.collect(tmp_path / "head2")
    base = ht.collect(tmp_path / "base")
    with pytest.raises(ht.TableError, match="rgb/A/5"):
        ht.check_same_frames({"head": head, "baseline": base})
    base[0]["manifests"].pop("A/rgb/validation.jsonl")
    with pytest.raises(ht.TableError, match="validation.jsonl is missing"):
        ht.check_same_frames({"head": head[:1], "baseline": base})


def test_the_caveat_names_the_validation_frames_the_baseline_used(tmp_path):
    from aerial_search.experiments import head_table as ht

    fake_table_run(tmp_path / "base", "rgb", "A", 1, 0.3, 0.1, "baseline")
    text = ht.compare_caveat({"baseline": ht.collect(tmp_path / "base")})
    assert "baseline picked its step on 100 evenly spaced" in text
