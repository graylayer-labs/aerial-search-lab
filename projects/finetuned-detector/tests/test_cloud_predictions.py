"""Predictions beside a trained checkpoint: the checkpoint race (#112 review, #113)."""

from pathlib import Path
from typing import Any

import pytest

# --- the checkpoint race (#112 review) --------------------------------------------


def test_predictions_next_to_a_mid_training_checkpoint_are_set_aside_not_fatal(
    tmp_path: Path,
) -> None:
    from finetuned_detector.experiment import Checkpoints

    identity = {"seed": 7}
    Checkpoints(tmp_path, identity, 10).save(phase="training", step=5)
    (tmp_path / "test_predictions.jsonl").write_text('{"image": "a"}\n')
    state = Checkpoints(tmp_path, identity, 10).load()
    assert state is not None and state["step"] == 5
    assert not (tmp_path / "test_predictions.jsonl").exists()
    assert list(tmp_path.glob("test_predictions.stale-*"))


def test_the_trained_checkpoint_is_rewritten_after_the_first_prediction(
    tmp_path: Path,
) -> None:
    from finetuned_detector.experiment import Checkpoints

    saver = Checkpoints(tmp_path, {"seed": 7}, 10)
    saver.save(phase="trained", step=5)
    before = saver.path.stat().st_ino
    saver.rewrite()
    assert saver.path.stat().st_ino != before
    assert saver.load() is not None


def trained(directory: Path, weight: float) -> Any:
    import torch

    from finetuned_detector.experiment import Checkpoints

    saver = Checkpoints(directory, {"seed": 7}, 10)
    saver.save(phase="trained", step=5, model={"w": torch.full((2,), weight)})
    return saver


def predicted(saver: Any, image: str = "a") -> None:
    from aerial_search.evaluation import predictions as pr

    log = pr.PredictionLog(saver.predictions)
    log.bind(saver.model_id)
    log.append(pr.Prediction(image, [[0, 0, 1, 1]], [0.5]))


def test_another_models_predictions_beside_a_trained_checkpoint_are_set_aside(
    tmp_path: Path,
) -> None:
    from aerial_search.evaluation import predictions as pr
    from finetuned_detector.experiment import Checkpoints

    other = trained(tmp_path / "other", 1.0)
    predicted(other)
    mine = trained(tmp_path / "mine", 2.0)
    assert mine.model_id != other.model_id
    # S3 sync kept the first model's predictions beside a retrained checkpoint
    (tmp_path / "other" / "test_predictions.jsonl").rename(mine.predictions)
    loader = Checkpoints(tmp_path / "mine", {"seed": 7}, 10)
    with pytest.warns(UserWarning, match="another model"):
        assert loader.load() is not None
    assert not loader.predictions.exists()
    assert list((tmp_path / "mine").glob("test_predictions.stale-*"))
    log = pr.PredictionLog(loader.predictions)
    assert not log.done  # every test frame is predicted again
    assert loader.model_id is not None
    log.bind(loader.model_id)
    assert pr.read_model_id(loader.predictions) == mine.model_id


def test_predictions_with_no_model_id_beside_a_trained_checkpoint_are_set_aside(
    tmp_path: Path,
) -> None:
    from finetuned_detector.experiment import Checkpoints

    mine = trained(tmp_path, 2.0)
    mine.predictions.write_text(
        '{"image": "a", "boxes": [[0, 0, 1, 1]], "scores": [0.5]}\n'
    )
    loader = Checkpoints(tmp_path, {"seed": 7}, 10)
    with pytest.warns(UserWarning):
        loader.load()
    assert not loader.predictions.exists()


def test_the_same_models_predictions_resume(tmp_path: Path) -> None:
    from aerial_search.evaluation import predictions as pr
    from finetuned_detector.experiment import Checkpoints

    mine = trained(tmp_path, 2.0)
    predicted(mine)
    loader = Checkpoints(tmp_path, {"seed": 7}, 10)
    loader.load()
    assert list(pr.PredictionLog(loader.predictions).done) == ["a"]
    assert [p.image_id for p in pr.read_predictions(loader.predictions)] == ["a"]


def test_a_predictions_file_refuses_a_second_model(tmp_path: Path) -> None:
    from aerial_search.evaluation import predictions as pr

    predicted(trained(tmp_path, 1.0))
    with pytest.raises(ValueError, match="model"):
        pr.PredictionLog(tmp_path / pr.PREDICTIONS_FILENAME).bind("f" * 64)
