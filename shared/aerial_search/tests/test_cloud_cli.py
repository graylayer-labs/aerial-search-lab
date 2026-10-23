"""The cloud commands on the command line (#113)."""

from types import SimpleNamespace
from typing import Any

import pytest

from aerial_search.cloud import launcher

# --- the request ------------------------------------------------------------------

BASE: dict[str, Any] = dict(
    logical_run="detection-baseline/fasterrcnn_resnet50_fpn-tiled/rgb-220109_Baker-10pct-seed7",
    attempt=1,
    commit="a" * 40,
    source_sha256="b" * 64,
    manifests_sha256="c" * 64,
    constraints_sha256="d" * 64,
    baseline_args=["rgb", "--percent", "10", "--fold", "220109_Baker"],
    role_arn="arn:aws:iam::123456789012:role/r",
    spot=True,
    max_runtime_hours=12.0,
    max_wait_hours=18.0,
    scratch=False,
)


def request(**over: Any) -> dict[str, Any]:
    return launcher.build_request(**{**BASE, **over})


# --- the command line -------------------------------------------------------------


def test_the_job_gets_the_same_flags_train_baseline_would_parse() -> None:
    from aerial_search import cli

    parser = cli.build_parser()
    cloud = parser.parse_args(
        ["sagemaker-train", "rgb", "--percent", "10", "--fold", "220109_Baker"]
        + ["--input", "tiled", "--scratch"]
    )
    forwarded = cli._baseline_args(cloud)
    local = parser.parse_args(["train-baseline", *forwarded])
    for name in (*cli.BASELINE_FLAG_NAMES, "camera", "scratch"):
        if name == "device":
            continue  # the job defaults to cuda
        assert getattr(local, name) == getattr(cloud, name), name
    assert local.device == "cuda"
    args = request(baseline_args=forwarded, logical_run="x" * 100)
    assert len(args["Environment"]["AERIAL_SEARCH_BASELINE_ARGS"]) <= 512


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
