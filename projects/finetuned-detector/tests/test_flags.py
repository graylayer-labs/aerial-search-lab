"""One definition of train-baseline's flags serves the laptop and the job (#160).

`sagemaker-train` forwards train-baseline's flags to the job, which runs
`aerial-search train-baseline` with them. If the two disagreed, the job would
refuse its argv or, worse, run another recipe than the one launched.
"""

import argparse
from typing import Any

from aerial_search.cloud import launcher
from finetuned_detector import cli

JOB_SETS = {"data_root", "manifests", "checkpoint_dir", "run_name"}


def parser(*, cloud: bool) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    cli.add_baseline_flags(out, cloud=cloud)
    return out


def changed(action: argparse.Action) -> Any:
    """A valid value for the flag that is not its default."""
    if action.choices:
        return next(c for c in action.choices if c != action.default)
    if isinstance(action.default, bool):
        return not action.default
    if action.type is int:
        return (action.default or 0) + 1
    if action.type is float:
        return (action.default or 1.0) / 2
    return "cpu" if action.dest == "device" else "x"


def test_every_train_baseline_flag_the_job_does_not_set_is_forwarded() -> None:
    flags = {a.dest for a in parser(cloud=False)._actions} - {"help"}
    assert flags - JOB_SETS == {*cli.BASELINE_FLAG_NAMES, "camera", "scratch"}


def test_train_baseline_accepts_every_forwarded_flag_at_any_value() -> None:
    cloud = parser(cloud=True)
    sent = cloud.parse_args(["rgb", "--percent", "10", "--fold", "220109_Baker"])
    for action in cloud._actions:
        if action.dest in {*cli.BASELINE_FLAG_NAMES, "camera", "scratch"}:
            setattr(sent, action.dest, changed(action))
    forwarded = cli.baseline_args(sent)
    received = parser(cloud=False).parse_args(forwarded)
    for name in (*cli.BASELINE_FLAG_NAMES, "camera", "scratch"):
        assert getattr(received, name) == getattr(sent, name), name


def test_unset_flags_keep_train_baseline_defaults_except_the_device() -> None:
    sent = parser(cloud=True).parse_args(
        ["rgb", "--percent", "10", "--fold", "220109_Baker", "--input", "tiled"]
    )
    received = parser(cloud=False).parse_args(cli.baseline_args(sent))
    for name in cli.BASELINE_FLAG_NAMES:
        if name != "device":  # the job defaults to cuda
            assert getattr(received, name) == getattr(sent, name), name
    assert received.device == "cuda"
    request = launcher.build_request(
        logical_run="x" * 100,
        attempt=1,
        commit="a" * 40,
        source_sha256="b" * 64,
        manifests_sha256="c" * 64,
        constraints_sha256="d" * 64,
        baseline_args=cli.baseline_args(sent),
        role_arn="arn:aws:iam::123456789012:role/r",
        spot=True,
        max_runtime_hours=12.0,
        max_wait_hours=18.0,
        scratch=False,
    )
    assert len(request["Environment"]["AERIAL_SEARCH_BASELINE_ARGS"]) <= 512
