"""Checkpoint, resume and the test predictions file of the baseline (#112).

A spot instance can be stopped at any point. These tests check what would
silently corrupt a result: a resumed run that differs from an uninterrupted
one, test frames predicted twice or not at all, a half-written line, a
re-score that differs from the run's own, and a resume under another config.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aerial_search import cli
from aerial_search.evaluation import predictions as pr
from finetuned_detector import experiment as be

FOLD, OTHER = "220109_Baker", "210417_MtErie"
SPEC = be.TileSpec(size=(32, 24), overlap=16, min_visible=0.5, empty_ratio=0.5)
REAL_BUILD = be.build_model


def rec(site: str, name: str, boxes: list[dict]) -> dict:
    return {
        "site_day": site,
        "collection_id": f"{site}_Enterprise_1",
        "camera": "thermal",
        "image": f"{site}_Enterprise_IR_1/{name}.jpg",
        "size": [64, 48],
        "boxes": boxes,
    }


def box(cx: float, cy: float) -> dict:
    return {"x_center": cx, "y_center": cy, "width": 0.25, "height": 0.25}


def fake_fold(tmp_path: Path) -> tuple[Path, Path]:
    manifests, raw = tmp_path / "manifests", tmp_path / "raw"
    view = manifests / "folds" / FOLD / "thermal"
    view.mkdir(parents=True)
    splits = {
        "train_100pct.jsonl": [rec(OTHER, f"t{k}", [box(0.5, 0.5)]) for k in range(4)],
        "validation.jsonl": [rec(OTHER, f"v{k}", [box(0.3, 0.6)]) for k in range(2)],
        "test.jsonl": [rec(FOLD, f"x{k}", [box(0.4, 0.4)]) for k in range(4)],
    }
    noise = np.random.default_rng(0)
    for file, records in splits.items():
        (view / file).write_text("".join(json.dumps(r) + "\n" for r in records))
        for r in records:
            path = raw / r["image"]
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = noise.integers(0, 255, (48, 64, 3), dtype="u1")
            Image.fromarray(pixels).save(path)
    return manifests, raw


class Crash(RuntimeError):
    pass


def model_factory(
    crash_at_train_call: int | None = None, eval_sizes: list[int] | None = None
):
    """The tiny untrained detector; optionally raise on the n-th training call
    and record the number of images in each inference call."""

    def build(arch, **kwargs):
        kwargs["pretrained"] = False  # no download in tests
        model = REAL_BUILD(arch, **kwargs)
        # Sample a few of the anchors and proposals, so training draws on the
        # torch RNG (on 64x48 images the default sample takes them all) and a
        # resume that lost the RNG state would differ.
        model.rpn.fg_bg_sampler.batch_size_per_image = 4
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 4
        if crash_at_train_call is not None or eval_sizes is not None:
            real_forward = model.forward
            calls = {"n": 0}

            def forward(images, targets=None):
                if model.training:
                    calls["n"] += 1
                    if calls["n"] == crash_at_train_call:
                        raise Crash("spot instance stopped")
                elif eval_sizes is not None:
                    eval_sizes.append(len(images))
                return real_forward(images, targets)

            model.forward = forward
        return model

    return build


def recipe(mode: str, steps: int = 4) -> be.BaselineRecipe:
    return be.BaselineRecipe(
        input=mode,
        steps=steps,
        batch_size=2,
        eval_every=2,
        val_frames=0,
        workers=0,
        warmup_steps=1,
    )


def run(tmp_path, manifests, raw, name, mode, *, checkpoint_dir=None, seed=7, steps=4):
    return be.run_baseline(
        manifests=manifests,
        data_root=raw,
        fold=FOLD,
        camera="thermal",
        percent=100,
        seed=seed,
        recipe=recipe(mode, steps),
        scratch=True,
        run_name=name,
        device=torch.device("cpu"),
        repo=tmp_path / "repo",
        argv=["x"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=2,
    )


def run_dir(tmp_path: Path, name: str) -> Path:
    return tmp_path / "repo" / "outputs" / f"scratch-{name}"


def final_weights(directory: Path) -> dict[str, torch.Tensor]:
    state = torch.load(directory / be.CHECKPOINT_FILENAME, weights_only=True)
    return state["model"]


@pytest.mark.parametrize("mode", ["native", "tiled"])
def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path, monkeypatch, mode):
    manifests, raw = fake_fold(tmp_path)
    if mode == "tiled":
        monkeypatch.setattr(be, "tile_spec", lambda camera, recipe: SPEC)

    monkeypatch.setattr(be, "build_model", model_factory())
    own = tmp_path / "ckpt-whole"
    whole = run(tmp_path, manifests, raw, "whole", mode, checkpoint_dir=own)

    # Stopped during step 3, after the checkpoint at step 2: step 3 is redone.
    shared = tmp_path / "checkpoints"
    monkeypatch.setattr(be, "build_model", model_factory(crash_at_train_call=3))
    with pytest.raises(Crash):
        run(tmp_path, manifests, raw, "first", mode, checkpoint_dir=shared)
    first = json.loads((run_dir(tmp_path, "first") / "run.json").read_text())
    assert first["status"] == "failed"

    monkeypatch.setattr(be, "build_model", model_factory())
    resumed = run(tmp_path, manifests, raw, "second", mode, checkpoint_dir=shared)

    record = json.loads((run_dir(tmp_path, "second") / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["resumed_from_step"] == 2 and record["resumes"] == 1
    assert record["checkpoint"]["directory"] == str(shared)

    whole_dir, second_dir = run_dir(tmp_path, "whole"), run_dir(tmp_path, "second")
    a = json.loads((whole_dir / "training.json").read_text())
    b = json.loads((second_dir / "training.json").read_text())
    np.testing.assert_allclose(a["loss"], b["loss"], rtol=1e-6)
    assert a["validation"] == b["validation"] and a["best_step"] == b["best_step"]
    wa = final_weights(own)
    wb = final_weights(shared)
    assert wa.keys() == wb.keys()
    for key in wa:
        torch.testing.assert_close(wa[key], wb[key], rtol=1e-6, atol=1e-7)
    assert (whole_dir / pr.PREDICTIONS_FILENAME).read_text() == (
        second_dir / pr.PREDICTIONS_FILENAME
    ).read_text()
    assert whole["test"] == resumed["test"]


@pytest.mark.parametrize("mode", ["native", "tiled"])
def test_predictions_survive_a_crash_mid_test_without_duplicates_or_gaps(
    tmp_path, monkeypatch, mode
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(be, "build_model", model_factory())
    if mode == "tiled":
        monkeypatch.setattr(be, "tile_spec", lambda camera, recipe: SPEC)
    shared = tmp_path / "checkpoints"
    real_append = pr.PredictionLog.append
    appended: list[str] = []

    def crash_after_two(self, prediction):
        if len(appended) == 2:
            raise Crash("spot instance stopped")
        appended.append(prediction.image_id)
        real_append(self, prediction)

    monkeypatch.setattr(pr.PredictionLog, "append", crash_after_two)
    with pytest.raises(Crash):
        run(tmp_path, manifests, raw, "first", mode, checkpoint_dir=shared)
    assert len(appended) == 2

    def count(self, prediction):
        appended.append(prediction.image_id)
        real_append(self, prediction)

    monkeypatch.setattr(pr.PredictionLog, "append", count)
    sizes: list[int] = []
    monkeypatch.setattr(be, "build_model", model_factory(eval_sizes=sizes))
    run(tmp_path, manifests, raw, "second", mode, checkpoint_dir=shared)
    if mode == "native":
        # The partly written batch is run again whole (all 4 test frames in
        # one batch of 8), so its numbers equal the uninterrupted run's.
        assert sizes == [4]
    tests = [
        json.loads(line)["image"]
        for line in (manifests / "folds" / FOLD / "thermal" / "test.jsonl")
        .read_text()
        .splitlines()
    ]
    assert appended == tests  # the first two once, then only the rest
    lines = (run_dir(tmp_path, "second") / pr.PREDICTIONS_FILENAME).read_text()
    header, *frames = lines.splitlines()
    assert list(json.loads(header)) == ["model_id"]  # the trained weights' id
    assert [json.loads(x)["image"] for x in frames] == tests
    record = json.loads((run_dir(tmp_path, "second") / "run.json").read_text())
    assert record["resumed_from_step"] == 4 and record["resumed_predictions"] == 2

    monkeypatch.setattr(be, "build_model", model_factory())
    run(tmp_path, manifests, raw, "whole", mode)
    whole = run_dir(tmp_path, "whole") / pr.PREDICTIONS_FILENAME
    assert whole.read_text() == lines


def test_a_truncated_last_line_is_dropped_and_overwritten(tmp_path):
    path = tmp_path / pr.PREDICTIONS_FILENAME
    full = [
        pr.Prediction("a", [[0, 0, 4, 4]], [0.9]),
        pr.Prediction("b", np.zeros((0, 4)), np.zeros(0)),
    ]
    log = pr.PredictionLog(path)
    for p in full:
        log.append(p)
    with path.open("a") as handle:
        handle.write('{"image": "c", "boxes": [[1.0, 2')  # killed mid-write
    resumed = pr.PredictionLog(path)
    assert list(resumed.done) == ["a", "b"]
    resumed.append(pr.Prediction("c", [[1, 2, 3, 4]], [0.5]))
    read = pr.read_predictions(path)
    assert [p.image_id for p in read] == ["a", "b", "c"]
    np.testing.assert_array_equal(read[0].boxes, [[0, 0, 4, 4]])
    assert np.asarray(read[1].boxes).shape == (0, 4)


def test_a_corrupt_line_before_the_last_is_refused(tmp_path):
    path = tmp_path / pr.PREDICTIONS_FILENAME
    path.write_text('{"image": "a", "bo\n{"image": "b", "boxes": [], "scores": []}\n')
    with pytest.raises(ValueError, match="line 1"):
        pr.PredictionLog(path)


def test_rescoring_the_predictions_file_equals_the_runs_own_score(
    tmp_path, monkeypatch, capsys
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(be, "build_model", model_factory())
    run(tmp_path, manifests, raw, "scored", "native")
    directory = run_dir(tmp_path, "scored")
    report = pr.rescore(directory)
    written = json.loads((directory / "detection_metrics.json").read_text())
    assert json.loads(json.dumps(report.to_dict())) == written
    assert set(report.by_size) == {"very_tiny", "tiny", "small", "medium_plus"}

    capsys.readouterr()
    cli.main(["rescore", str(directory), "--by-size"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["overall"] == written["overall"]
    assert printed["by_size"] == written["by_size"]


def test_rescoring_refuses_a_predictions_file_missing_a_test_frame(
    tmp_path, monkeypatch
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(be, "build_model", model_factory())
    run(tmp_path, manifests, raw, "scored", "native")
    path = run_dir(tmp_path, "scored") / pr.PREDICTIONS_FILENAME
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="missing"):
        pr.rescore(path.parent)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "native", "steps": 6},
        {"mode": "tiled"},
        {"mode": "native", "seed": 8},
    ],
)
def test_resuming_under_another_config_or_seed_is_refused(
    tmp_path, monkeypatch, change
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(
        be, "tile_spec", lambda camera, r: SPEC if r.input == "tiled" else None
    )
    shared = tmp_path / "checkpoints"
    monkeypatch.setattr(be, "build_model", model_factory(crash_at_train_call=3))
    with pytest.raises(Crash):
        run(tmp_path, manifests, raw, "first", "native", checkpoint_dir=shared)
    monkeypatch.setattr(be, "build_model", model_factory())
    with pytest.raises(be.CheckpointError, match="does not match"):
        run(tmp_path, manifests, raw, "second", checkpoint_dir=shared, **change)
    assert not run_dir(tmp_path, "second").exists()


def test_a_normal_run_does_not_resume_from_a_scratch_checkpoint(tmp_path):
    state = {"identity": be.checkpoint_identity({"a": 1}, 7, "abc", scratch=True)}
    shared = tmp_path / "checkpoints"
    shared.mkdir()
    torch.save(state, shared / be.CHECKPOINT_FILENAME)
    identity = be.checkpoint_identity({"a": 1}, 7, "abc", scratch=False)
    with pytest.raises(be.CheckpointError, match="does not match"):
        be.Checkpoints(shared, identity, every=2).load()


def test_two_live_runs_cannot_share_a_checkpoint_directory(tmp_path):
    # A second process on the same directory would write each test frame
    # twice and interleave checkpoints; the lock dies with its process.
    identity = be.checkpoint_identity({"a": 1}, 7, "abc", scratch=True)
    first = be.Checkpoints(tmp_path / "checkpoints", identity, every=2)
    first.lock()
    second = be.Checkpoints(tmp_path / "checkpoints", identity, every=2)
    with pytest.raises(be.CheckpointError, match="in use"):
        second.lock()
    first.unlock()
    second.lock()
    second.unlock()


@pytest.mark.parametrize("mode", ["native", "tiled"])
def test_scoring_the_file_equals_scoring_the_model_in_process(tmp_path, mode):
    manifests, raw = fake_fold(tmp_path)
    records = [
        json.loads(line)
        for line in (manifests / "folds" / FOLD / "thermal" / "test.jsonl")
        .read_text()
        .splitlines()
    ]
    torch.manual_seed(0)
    model = model_factory()("fasterrcnn_mobilenet_v3_large_fpn")
    frames: be.Frames | be.TiledFrames
    if mode == "tiled":
        frames = be.TiledFrames(records, raw, SPEC)
    else:
        frames = be.Frames(records, raw, None)
    device = torch.device("cpu")
    path = tmp_path / pr.PREDICTIONS_FILENAME
    be.predict_to_file(model, frames, device, pr.PredictionLog(path))
    from_file = pr.score_file(path, records)
    in_process = be.evaluate(model, frames, device)
    assert from_file.overall.n_predictions > 0
    assert from_file.to_dict() == in_process.to_dict()


def test_resuming_after_the_test_manifest_changed_is_refused(tmp_path, monkeypatch):
    manifests, raw = fake_fold(tmp_path)
    shared = tmp_path / "checkpoints"
    monkeypatch.setattr(be, "build_model", model_factory(crash_at_train_call=3))
    with pytest.raises(Crash):
        run(tmp_path, manifests, raw, "first", "native", checkpoint_dir=shared)
    test = manifests / "folds" / FOLD / "thermal" / "test.jsonl"
    test.write_text(test.read_text().replace('"x_center": 0.4', '"x_center": 0.45'))
    monkeypatch.setattr(be, "build_model", model_factory())
    with pytest.raises(be.CheckpointError, match="inputs differ"):
        run(tmp_path, manifests, raw, "second", "native", checkpoint_dir=shared)
    assert not run_dir(tmp_path, "second").exists()


def test_a_default_checkpoint_is_deleted_on_completion_an_explicit_one_kept(
    tmp_path, monkeypatch
):
    manifests, raw = fake_fold(tmp_path)
    monkeypatch.setattr(be, "build_model", model_factory())
    run(tmp_path, manifests, raw, "default", "native")
    directory = run_dir(tmp_path, "default")
    assert not (directory / "checkpoints" / be.CHECKPOINT_FILENAME).exists()
    assert (directory / pr.PREDICTIONS_FILENAME).exists()
    explicit = tmp_path / "explicit"
    run(tmp_path, manifests, raw, "explicit", "native", checkpoint_dir=explicit)
    assert (explicit / be.CHECKPOINT_FILENAME).exists()
