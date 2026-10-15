"""The in-job entry point (#113)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from aerial_search.cloud import entry, launcher, source

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


def write_manifests(path: Path, *, verified: bool = True, scratch: bool = False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "data_quality.json").write_text(
        json.dumps(
            {
                "provenance": {"commit": "abc", "scratch": scratch},
                "data": {"verified": verified, "directories": ["d1"]},
            }
        )
    )
    (path / "other.jsonl").write_text("{}\n")


PINS = b"pycocotools==2.0.11\ntorch==2.13.0\n"


# --- inside the job ---------------------------------------------------------------


def job_env(code: Path, manifests: Path, **over: str) -> dict[str, str]:

    r = request(
        source_sha256=source.digest_dir(code),
        manifests_sha256=source.digest_dir(manifests),
        constraints_sha256=hashlib.sha256(PINS).hexdigest(),
    )
    env = dict(r["Environment"])
    env["TRAINING_JOB_NAME"] = "ssl-aerial-exp-1"
    env.update(over)
    return env


@pytest.fixture
def job(tmp_path: Path) -> dict[str, Path]:
    code = tmp_path / "code"
    (code / "src" / "aerial_search").mkdir(parents=True)
    (code / "src" / "aerial_search" / "a.py").write_text("x = 1\n")
    manifests = tmp_path / "in" / "manifests"
    write_manifests(manifests)
    pins = tmp_path / "in" / "constraints" / "constraints.txt"
    pins.parent.mkdir(parents=True)
    pins.write_bytes(PINS)
    return {
        "code": code,
        "manifests": manifests,
        "constraints": pins,
        "data": tmp_path / "in" / "dataset",
        "checkpoints": tmp_path / "ckpt",
    }


def prepare(job: dict[str, Path], env: dict[str, str]) -> list[str]:
    return entry.prepare(
        env,
        code_root=job["code"],
        data_root=job["data"],
        manifests=job["manifests"],
        constraints=job["constraints"],
        checkpoint_dir=job["checkpoints"],
    )


def test_the_job_passes_the_baseline_flags_plus_its_own_paths(job) -> None:
    argv = prepare(job, job_env(job["code"], job["manifests"]))
    assert argv[0] == "train-baseline"
    assert argv[1:6] == BASE["baseline_args"][:5]
    flags = dict(zip(argv[1:], argv[2:], strict=False))
    assert flags["--checkpoint-dir"] == str(job["checkpoints"])
    assert flags["--data-root"] == str(job["data"])
    assert flags["--manifests"] == str(job["manifests"])
    assert "-a1-" in flags["--run-name"]


def test_modified_code_in_the_job_is_refused(job) -> None:
    env = job_env(job["code"], job["manifests"])
    (job["code"] / "src" / "aerial_search" / "a.py").write_text("x = 2\n")
    with pytest.raises(entry.JobRefused, match="source"):
        prepare(job, env)


def test_modified_manifests_in_the_job_are_refused(job) -> None:
    env = job_env(job["code"], job["manifests"])
    (job["manifests"] / "other.jsonl").write_text("{}\n{}\n")
    with pytest.raises(entry.JobRefused, match="manifests"):
        prepare(job, env)


def test_modified_constraints_in_the_job_are_refused(job) -> None:
    env = job_env(job["code"], job["manifests"])
    job["constraints"].write_bytes(PINS + b"numpy==2.0.0\n")
    with pytest.raises(entry.JobRefused, match="constraints"):
        prepare(job, env)


def test_an_incomplete_environment_is_refused(job) -> None:
    env = job_env(job["code"], job["manifests"])
    del env["AERIAL_SEARCH_ATTESTATION"]
    with pytest.raises(entry.JobRefused, match="incomplete"):
        prepare(job, env)
    env = job_env(job["code"], job["manifests"], AERIAL_SEARCH_BASELINE_ARGS="{")
    with pytest.raises(entry.JobRefused, match="incomplete"):
        prepare(job, env)


def test_outside_a_training_job_is_refused(job) -> None:
    env = job_env(job["code"], job["manifests"], TRAINING_JOB_NAME="")
    with pytest.raises(entry.JobRefused, match="TRAINING_JOB_NAME"):
        prepare(job, env)


def test_unreadable_constraints_are_refused(job) -> None:
    env = job_env(job["code"], job["manifests"])
    job["constraints"].unlink()
    with pytest.raises(entry.JobRefused, match="cannot read the constraints"):
        prepare(job, env)


def test_a_restart_never_reuses_a_run_name(tmp_path: Path) -> None:
    logical = "exp/run"
    first = entry.attempt_run_name(logical, tmp_path)
    assert "-a1-" in first
    # attempts.json may lag behind a restart: a stale list that already holds
    # an `-a2` name must still push the next attempt to `-a3`.
    (tmp_path / "attempts.json").write_text(json.dumps([first, "exp-run-a2-x"]))
    third = entry.attempt_run_name(logical, tmp_path)
    assert "-a3-" in third
    assert third not in {first, "exp-run-a2-x"}


def test_pins_for_the_containers_own_packages_are_dropped() -> None:
    assert entry.requirement_name("Torch_Vision==1") == "torch-vision"
    assert entry.requirement_name("# comment") is None
    pins = "numpy==2.0\ntorch==2.13.0\ntorchvision==0.1\n# c\n"
    assert entry.without_provided(pins) == "numpy==2.0\n# c\n"
