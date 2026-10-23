"""`aerial-search train-baseline` runs this project's experiment (#160)."""

from typing import Any

from aerial_search import cli
from finetuned_detector import experiment


def test_train_baseline_runs_the_project_experiment(monkeypatch, capsys) -> None:
    seen: dict[str, Any] = {}

    def run_baseline(**kw: Any) -> dict[str, int]:
        seen.update(kw)
        return {"ok": 1}

    monkeypatch.setattr(experiment, "run_baseline", run_baseline)
    cli.main(["train-baseline", "rgb", "--percent", "10", "--fold", "220109_Baker"])
    assert seen["fold"] == "220109_Baker" and seen["percent"] == 10
    assert seen["camera"] == "rgb" and seen["scratch"] is False
    assert '"ok": 1' in capsys.readouterr().out
