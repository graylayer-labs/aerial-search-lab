"""The shared runner: run wiring, scoring and the best-step loop (#216)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from aerial_search import runner
from aerial_search.evaluation.detection import Prediction


def loop(scores, *, steps=None, eval_every=1, selection=None, start=0, after=None):
    """The loop on a fake model whose weights are the step number."""
    steps = steps or len(scores)
    model = {"w": 0}
    validate = (lambda: scores[model["w"] - 1]) if scores else None

    def step(number, batch):
        model["w"] = number
        return float(batch)

    result = runner.train_best_step(
        [10 * n for n in range(start + 1, steps + 1)],
        step,
        steps=steps,
        eval_every=eval_every,
        validate=validate,
        snapshot=lambda: dict(model),
        restore=model.update,
        selection=selection,
        start=start,
        after_step=after,
    )
    return result, model["w"]


def test_the_best_step_is_kept_the_earlier_wins_a_tie_and_nan_never_wins():
    result, weights = loop([0.2, 0.5, math.nan, 0.5, 0.1])
    assert result.history["best_step"] == 2
    assert weights == 2
    assert result.history["loss"] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert [v["step"] for v in result.history["validation"]] == [1, 2, 3, 4, 5]


def test_validation_runs_every_eval_every_steps_and_at_the_last():
    result, _ = loop([0.1] * 5, eval_every=2)
    assert [v["step"] for v in result.history["validation"]] == [2, 4, 5]


def test_without_validation_or_a_winning_score_the_final_weights_stay():
    result, weights = loop([], steps=3)
    assert (result.history["best_step"], weights) == (None, 3)
    result, weights = loop([math.nan] * 3)
    assert (result.history["best_step"], weights) == (None, 3)


def test_a_resumed_loop_continues_numbering_history_and_the_best_so_far():
    first = runner.Selection(best=0.9, state={"w": 1})
    first.history.update(loss=[10.0], best_step=1)
    seen = []
    result, weights = loop(
        [0.0, 0.5, 0.95], selection=first, start=1, after=lambda n, s: seen.append(n)
    )
    assert result.history["loss"] == [10.0, 20.0, 30.0]
    assert (result.history["best_step"], weights) == (3, 3)
    assert seen == [2, 3]


def test_the_batch_order_sees_every_frame_once_per_pass_and_resumes_in_step():
    batches = list(runner.batch_order(5, 2, 5, seed=3))
    assert all(b == sorted(b) for b in batches)
    first_pass = [i for b in batches[:2] for i in b] + batches[2][:1]
    assert sorted(first_pass) == [0, 1, 2, 3, 4]
    assert list(runner.batch_order(5, 2, 5, seed=3, start=2)) == batches[2:]
    assert next(runner.batch_order(3, 8, 2, seed=0)) == [0, 1, 2]


def test_a_run_is_completed_or_marked_failed_with_its_error(tmp_path):
    def run(name: str) -> Path:
        with runner.experiment_run(
            name,
            {"a": 1},
            7,
            device="cpu",
            scratch=True,
            argv=["x"],
            repo=tmp_path / "repo",
            inputs=[],
            fold="220109_Baker",
            view="thermal",
            manifests=tmp_path / "manifests",
            data_root=tmp_path / "raw",
        ) as directory:
            runner.write_training(directory, {"loss": [math.nan]})
            if name == "bad":
                raise KeyboardInterrupt
        return directory

    good = json.loads((run("good") / "run.json").read_text())
    assert (good["status"], good["fold"], good["view"]) == (
        "completed",
        "220109_Baker",
        "thermal",
    )
    with pytest.raises(KeyboardInterrupt):
        run("bad")
    bad = tmp_path / "repo" / "outputs" / "scratch-bad"
    record = json.loads((bad / "run.json").read_text())
    assert (record["status"], record["error"]["type"]) == (
        "failed",
        "KeyboardInterrupt",
    )
    assert (bad / "training.json").read_text() == '{\n  "loss": [\n    NaN\n  ]\n}\n'


def test_score_uses_the_records_or_the_boxes_and_camera_given():
    record = {
        "image": "a.jpg",
        "camera": "rgb",
        "collection_id": "c",
        "size": [100, 100],
        "boxes": [{"x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}],
    }
    hit = Prediction("a.jpg", np.array([[40.0, 40, 60, 60]]), np.array([0.9]))
    assert runner.score([record], [hit]).overall.ap_iou25 == 1.0
    empty = [np.zeros((0, 4))]
    report = runner.score([record], [hit], boxes=empty, camera="thermal")
    assert report.overall.n_ground_truth == 0
    assert list(report.by_modality) == ["thermal"]


def test_run_names_follow_one_pattern():
    assert runner.run_name("p/q", "rgb", "220109_Baker", 5, 2) == (
        "p/q/rgb-220109_Baker-5pct-seed2"
    )
