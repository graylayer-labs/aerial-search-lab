"""The fine-tuned detector's commands, run through `aerial-search` (#160)."""

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from aerial_search import REPO_ROOT, cli
from aerial_search.cloud import entry, launcher
from finetuned_detector import cli as project
from finetuned_detector import experiment

LAUNCH = ["sagemaker-train", "rgb", "--percent", "10", "--fold", "f"]


def test_train_baseline_runs_in_the_job_from_the_members_src_alone() -> None:
    for name in ("train-baseline", "sagemaker-train"):
        assert cli.PROJECT_COMMANDS[name].module == "finetuned_detector.cli"
    # The job installs no member: only their src/ dirs and the third-party
    # packages are on sys.path. -S keeps the editable installs' .pth files out,
    # and -I ignores PYTHONPATH, so the paths are set in the child itself.
    paths = [str(d) for d in entry.source_dirs(REPO_ROOT)]
    paths += [p for p in sys.path if p.endswith("site-packages")]
    code = (
        f"import sys; sys.path[:0] = {paths!r}; "
        "import finetuned_detector; print(finetuned_detector.__file__); "
        "from aerial_search import cli; cli.main(['train-baseline', '--help'])"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(str(REPO_ROOT / "projects")), done.stdout


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


@pytest.mark.parametrize("hours", ["0", "-1", "nan"])
def test_the_command_line_refuses_expected_hours_that_are_not_positive(
    hours: str,
) -> None:
    # parse only: a parse that let the value through would go on to AWS calls
    with pytest.raises(SystemExit) as refused:
        project.build_parser().parse_args([*LAUNCH, "--expected-hours", hours])
    assert refused.value.code == 2


def test_sagemaker_train_launches_with_the_forwarded_flags(monkeypatch, capsys) -> None:
    seen: dict[str, Any] = {}

    def fake_launch(**kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(estimate=None, request={"Name": "j"})

    monkeypatch.setattr(launcher, "launch", fake_launch)
    monkeypatch.setattr(project, "asdict", lambda estimate: {})
    cli.main([*LAUNCH, "--dry-run"])
    assert '"dry_run": true' in capsys.readouterr().out
    assert seen["baseline_args"][0] == "rgb" and "--fold" in seen["baseline_args"]
    assert seen["spot"] is True and seen["session_factory"] is cli._aws_session


def test_the_command_line_has_no_on_demand_flag() -> None:
    with pytest.raises(SystemExit):
        cli.main([*LAUNCH, "--on-demand"])
