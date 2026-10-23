"""The cloud commands on the command line (#113)."""

from types import SimpleNamespace
from typing import Any

import pytest

from aerial_search.cloud import launcher

# --- the command line -------------------------------------------------------------


@pytest.mark.parametrize("hours", ["0", "-1", "nan"])
def test_the_command_line_refuses_expected_hours_that_are_not_positive(
    hours: str,
) -> None:
    from aerial_search import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["sagemaker-train", "rgb", "--percent", "10", "--fold", "f"]
            + ["--expected-hours", hours]
        )


def test_sagemaker_train_launches_with_the_forwarded_flags(monkeypatch, capsys) -> None:
    from aerial_search import cli

    seen: dict[str, Any] = {}

    def fake_launch(**kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(estimate=None, request={"Name": "j"})

    monkeypatch.setattr(launcher, "launch", fake_launch)
    monkeypatch.setattr(cli, "asdict", lambda estimate: {})
    cli.main(["sagemaker-train", "rgb", "--percent", "10", "--fold", "f", "--dry-run"])
    assert '"dry_run": true' in capsys.readouterr().out
    assert seen["baseline_args"][0] == "rgb" and "--fold" in seen["baseline_args"]
    assert seen["spot"] is True and seen["session_factory"] is cli._aws_session


def test_sagemaker_fetch_lists_what_it_fetched(monkeypatch, capsys, tmp_path) -> None:
    from aerial_search import cli

    monkeypatch.setattr(cli, "_aws_session", lambda: "session")
    monkeypatch.setattr(launcher, "fetch", lambda job, session, out: [tmp_path / "r"])
    cli.main(["sagemaker-fetch", "job-1"])
    assert capsys.readouterr().out.strip() == str(tmp_path / "r")


def test_the_command_line_has_no_on_demand_flag() -> None:
    from aerial_search import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["sagemaker-train", "rgb", "--percent", "10", "--fold", "f"]
            + ["--on-demand"]
        )
