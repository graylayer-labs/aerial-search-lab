"""`sagemaker-train-head`: train-head as a CPU SageMaker spot job (#188).

It forwards train-head's flags to the job, which would run `aerial-search
train-head` with them. If the two disagreed, the job would refuse its argv
or, worse, run another recipe than the one launched.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aerial_search import cli
from aerial_search.cloud import launcher
from frozen_head import cli as project

LAUNCH = ["sagemaker-train-head", "rgb", "--percent", "10", "--fold", "220109_Baker"]
JOB_SETS = {"data_root", "manifests", "cache", "run_name"}
NOT_IN_A_JOB = {"init_head"}  # a local run directory the job cannot read
SENT = {*project.HEAD_FLAG_NAMES, "camera", "scratch"}


def parser(*, cloud: bool) -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    project.add_head_flags(out, cloud=cloud)
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


def test_the_command_is_registered_by_the_frozen_head() -> None:
    assert cli.PROJECT_COMMANDS["sagemaker-train-head"].module == "frozen_head.cli"


def test_every_train_head_flag_the_job_does_not_set_is_forwarded() -> None:
    flags = {a.dest for a in parser(cloud=False)._actions} - {"help"}
    assert flags - JOB_SETS - NOT_IN_A_JOB == SENT


def test_train_head_accepts_every_forwarded_flag_at_any_value() -> None:
    cloud = parser(cloud=True)
    sent = cloud.parse_args(LAUNCH[1:])
    for action in cloud._actions:
        if action.dest in SENT:
            setattr(sent, action.dest, changed(action))
    received = parser(cloud=False).parse_args(project.head_args(sent))
    for name in SENT:
        assert getattr(received, name) == getattr(sent, name), name


@pytest.fixture
def launched(monkeypatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_launch(**kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(estimate=None, request={"Name": "j"})

    monkeypatch.setattr(launcher, "launch", fake_launch)
    monkeypatch.setattr(project, "asdict", lambda estimate: {})
    return seen


def test_a_dry_run_is_a_cpu_spot_train_head_job_and_names_the_missing_input(
    launched, capsys
) -> None:
    cli.main([*LAUNCH, "--dry-run", "--month-to-date-eur", "0"])
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert "siglip2-base-naflex-1024tok" in out["missing_inputs"][0]
    assert launched["instance_type"] == "ml.m5.xlarge"
    assert launched["command"] == "train-head"
    assert launched["spot"] is True
    assert launched["max_runtime_hours"] == 2.0
    assert launched["baseline_args"][:3] == ["rgb", "--percent", "10"]
    assert launched["logical_run"] == (
        "detection-head/siglip2-base-naflex-1024tok/rgb-220109_Baker-10pct-seed7"
    )


def test_sending_is_refused_until_the_job_can_read_the_cache(launched) -> None:
    with pytest.raises(SystemExit, match="feature cache"):
        cli.main(LAUNCH)
    assert launched == {}


def test_a_refused_launch_exits_with_the_reason(monkeypatch) -> None:
    from aerial_search.cloud import cost

    def refuse(**_: Any) -> Any:
        raise cost.CostRefused("over the 5 EUR month limit")

    monkeypatch.setattr(launcher, "launch", refuse)
    with pytest.raises(SystemExit, match="refusing to launch: over the 5 EUR"):
        cli.main([*LAUNCH, "--dry-run"])


@pytest.mark.parametrize(
    "flags",
    [
        ["--eur-per-usd", "0.94"],
        ["--eur-per-usd", "nan"],
        ["--month-to-date-eur", "-0.01"],
        ["--month-to-date-eur", "nan"],
    ],
)
def test_a_rate_below_the_guards_or_a_negative_spend_is_refused(
    launched, flags: list[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.main([*LAUNCH, "--dry-run", *flags])
    assert launched == {}


def test_the_guards_rate_and_zero_spend_are_accepted(launched) -> None:
    cli.main(
        [*LAUNCH, "--dry-run", "--eur-per-usd", "0.95", "--month-to-date-eur", "0"]
    )
    assert launched["eur_per_usd"] == 0.95
    assert launched["month_to_date_eur"] == 0.0


# --- the cache files a job gets (#191) ---------------------------------------------

FOLD = "220109_Baker"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def small_cache(tmp_path: Path, top: str = "scratch-features") -> tuple[Path, Path]:
    """Fold manifests for thermal, and a cache holding them plus an rgb image
    from another session, which a thermal job does not need."""
    manifests = tmp_path / "manifests"
    images: list[str] = []
    for role in ("train_100pct", "validation", "test"):
        site = FOLD if role == "test" else "210417_MtErie"
        records = []
        for k in range(2):
            image = f"{site}_Enterprise_IR_1/{role}{k}.jpg"
            images.append(image)
            records.append(
                {"site_day": site, "collection_id": image.split("/")[0]}
                | {"camera": "thermal", "image": image, "size": [640, 512]}
                | {"boxes": []}
            )
        write_jsonl(manifests / "folds" / FOLD / "thermal" / f"{role}.jsonl", records)
    cache = tmp_path / "outputs" / top / "m-1024tok"
    entries = [
        {"path": image, "camera": "thermal", "image_size": [640, 512]}
        | {"grid": [14, 18], "run": "s1"}
        for image in images
    ] + [{"path": "VIS_1/x.jpg", "camera": "rgb", "grid": [12, 21], "run": "s2"}]
    write_jsonl(cache / "index.jsonl", entries)
    (cache / "cache.json").write_text("{}")
    for session in ("s1", "s2"):
        (cache / "runs" / session).mkdir(parents=True)
        (cache / "runs" / session / "run.json").write_text(
            json.dumps({"scratch": True, "status": "completed"})
        )
    for entry in entries:
        path = cache / "features" / Path(str(entry["path"])).with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"f")
    return cache, manifests


def test_a_job_gets_one_cameras_features_and_their_records(tmp_path: Path) -> None:
    from frozen_head import head_experiment

    cache, manifests = small_cache(tmp_path)
    files = head_experiment.job_files(
        cache, manifests, fold=FOLD, camera="thermal", percent=100, scratch=True
    )
    assert files[:3] == ("cache.json", "index.jsonl", "runs/s1/run.json")
    assert len(files[3:]) == 6
    assert all(f.startswith("features/") and "_IR_1/" in f for f in files[3:])


def test_the_cache_is_checked_as_the_run_would_before_choosing(tmp_path: Path) -> None:
    from frozen_head import head_experiment
    from frozen_head.feature_cache import CacheError

    cache, manifests = small_cache(tmp_path)
    with pytest.raises(CacheError, match="scratch"):
        head_experiment.job_files(
            cache, manifests, fold=FOLD, camera="thermal", percent=100, scratch=False
        )


def test_the_command_line_has_no_on_demand_or_instance_flag() -> None:
    for flag in ("--on-demand", "--instance-type"):
        with pytest.raises(SystemExit):
            cli.main([*LAUNCH, "--dry-run", flag, "ml.m5.large"])
