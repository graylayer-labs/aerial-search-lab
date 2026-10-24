"""The in-job entry point (#113)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from aerial_search import PACKAGE_DIR
from aerial_search.cloud import entry, launcher, source

WORKSPACE = (
    "[tool.uv.workspace]\n"
    'members = ["shared/*", "projects/*"]\n'
    'exclude = ["projects/_template"]\n'
)

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
    (code / PACKAGE_DIR).mkdir(parents=True)
    (code / PACKAGE_DIR / "a.py").write_text("x = 1\n")
    (code / "pyproject.toml").write_text(WORKSPACE)
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
    (job["code"] / PACKAGE_DIR / "a.py").write_text("x = 2\n")
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


# --- installing and running -------------------------------------------------------


def test_the_container_keeps_its_own_torch(job, monkeypatch) -> None:
    (job["code"] / "shared" / "aerial_search" / "pyproject.toml").write_text(
        '[project]\nname = "x"\n'
        'dependencies = ["torch>=2", "TorchVision", "numpy>=2"]\n'
    )
    deps = entry._dependencies(job["code"])
    assert deps == ["numpy>=2"]
    seen: dict[str, Any] = {}

    def fake_pip(cmd: list[str], check: bool) -> None:
        seen["cmd"] = cmd
        seen["pins"] = Path(cmd[cmd.index("-c") + 1]).read_text()

    monkeypatch.setattr(entry.subprocess, "run", fake_pip)
    entry._install(deps, job["constraints"])
    assert seen["pins"] == "pycocotools==2.0.11\n"
    assert seen["cmd"][-1] == "numpy>=2"


def test_installed_versions_name_what_is_missing() -> None:
    got = entry._installed(["pytest>=8", "no-such-package-xyz"])
    assert got["pytest"]
    assert got["no-such-package-xyz"] is None
    assert "torch" in got


@pytest.fixture
def in_job(job, tmp_path: Path, monkeypatch) -> dict[str, Any]:
    (job["code"] / "shared" / "aerial_search" / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["pytest>=8"]\n'
    )
    for name, key in [
        ("CODE_ROOT", "code"),
        ("DATASET", "data"),
        ("MANIFESTS", "manifests"),
        ("CONSTRAINTS", "constraints"),
        ("CHECKPOINTS", "checkpoints"),
    ]:
        monkeypatch.setattr(entry, name, job[key])
    monkeypatch.setattr(entry, "OUTPUT", tmp_path / "uploaded")
    monkeypatch.setattr(entry, "_install", lambda deps, constraints: None)
    return {**job, "env": job_env(job["code"], job["manifests"])}


def test_a_refused_job_exits_before_installing_anything(in_job, monkeypatch) -> None:
    (in_job["code"] / PACKAGE_DIR / "a.py").write_text("x = 2\n")

    def boom(*_: Any) -> None:
        raise AssertionError("installed")

    monkeypatch.setattr(entry, "_install", boom)
    with pytest.raises(SystemExit, match="refusing to run: .*source"):
        entry.main(in_job["env"])


def test_outputs_and_packages_are_handed_back_when_the_run_fails(
    in_job, monkeypatch
) -> None:
    from aerial_search import cli

    outputs = in_job["code"] / "outputs"

    def failing(argv: list[str]) -> None:
        directory = outputs / argv[argv.index("--run-name") + 1]
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"status": "failed"}))
        raise RuntimeError("out of memory")

    monkeypatch.setattr(cli, "main", failing)
    with pytest.raises(RuntimeError, match="out of memory"):
        entry.main(in_job["env"])
    [copied] = list(entry.OUTPUT.iterdir())
    record = json.loads((copied / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["installed_packages"]["pytest"]


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ("torch>=2.13.0", "torch"),
        ("torch~=2.13", "torch"),
        ("torch<3", "torch"),
        ("Torch==2.13.0", "torch"),
        ("torchvision[extra] >= 0.28", "torchvision"),
        ("torch ; sys_platform == 'linux'", "torch"),
        ("torch@ https://example.com/torch.whl", "torch"),
        ("torch-audio>=1", "torch-audio"),
        ("Py_Coco.Tools>=2", "py-coco-tools"),
        ("    # via torch", None),
        ("", None),
    ],
)
def test_requirement_names_are_parsed_per_pep_508(spec: str, name: str | None) -> None:
    assert entry.requirement_name(spec) == name


def test_torch_and_torchvision_from_the_container_are_never_reinstalled(
    tmp_path: Path, monkeypatch
) -> None:
    for member, text in {
        "shared/aerial_search": '"torch~=2.13", "TorchVision[x]>=0.28", '
        '"torchmetrics>=1", "pycocotools>=2.0.11"',
        "projects/b": '"aerial-search", "torchmetrics>=1"',
        "projects/_template": '"never-installed"',
    }.items():
        (tmp_path / member).mkdir(parents=True)
        (tmp_path / member / "pyproject.toml").write_text(
            f'[project]\nname = "{Path(member).name}"\ndependencies = [{text}]\n'
        )
    (tmp_path / "pyproject.toml").write_text(WORKSPACE)
    dependencies = entry._dependencies(tmp_path)
    assert dependencies == ["torchmetrics>=1", "pycocotools>=2.0.11"]
    pins = tmp_path / "constraints.txt"
    pins.write_text(
        "torch==2.13.0\n    # via torchvision\ntorchvision==0.28.0 ; "
        "sys_platform == 'linux'\npycocotools==2.0.11\n"
    )
    seen: list[list[str]] = []

    def run(args: list[str], **_: Any) -> None:
        seen.append(args)
        used = Path(args[args.index("-c") + 1]).read_text()
        assert "pycocotools==2.0.11" in used
        assert "torch==" not in used and "torchvision==" not in used

    monkeypatch.setattr(entry.subprocess, "run", run)
    entry._install(dependencies, pins)
    assert seen and seen[0][-2:] == dependencies
    assert not any(a.startswith(("torch>", "torch=", "torchvision")) for a in seen[0])


def test_every_attempt_gets_a_run_name_no_earlier_attempt_used(job) -> None:
    seen: list[str] = []
    for _ in range(3):
        argv = prepare(job, job_env(job["code"], job["manifests"]))
        name = argv[argv.index("--run-name") + 1]
        assert name not in seen
        seen.append(name)
        # what Checkpoints.start_attempt does after start_run
        job["checkpoints"].mkdir(exist_ok=True)
        (job["checkpoints"] / "attempts.json").write_text(json.dumps(seen) + "\n")
    assert len(set(seen)) == 3


def test_a_restart_before_attempts_json_synced_still_gets_a_new_name(job) -> None:
    # A spot restart keeps the job name, and attempts.json may not have
    # reached S3 before the stop: both attempts see the same state.
    from datetime import UTC, datetime

    first = entry.attempt_run_name(
        "exp", job["checkpoints"], now=datetime(2026, 10, 15, 9, 0, 0, tzinfo=UTC)
    )
    second = entry.attempt_run_name(
        "exp", job["checkpoints"], now=datetime(2026, 10, 15, 9, 40, 0, tzinfo=UTC)
    )
    assert first != second
    assert first.startswith("exp-a1-") and second.startswith("exp-a1-")


def test_a_name_in_attempts_json_is_never_reused_even_if_counts_disagree(job) -> None:
    job["checkpoints"].mkdir()
    logical = entry.safe_name(BASE["logical_run"])
    (job["checkpoints"] / "attempts.json").write_text(
        json.dumps([f"{logical}-a2"]) + "\n"
    )
    argv = prepare(job, job_env(job["code"], job["manifests"]))
    assert argv[argv.index("--run-name") + 1] != f"{logical}-a2"


@pytest.mark.parametrize("command", ["train-head", "evaluate"])
def test_a_job_for_a_command_it_has_no_inputs_for_is_refused(job, command) -> None:
    # train-head needs the features channel (#191); any other command has no
    # inputs at all. Neither may run train-baseline in its place.
    env = job_env(job["code"], job["manifests"], AERIAL_SEARCH_COMMAND=command)
    with pytest.raises(entry.JobRefused, match=command):
        prepare(job, env)


# --- train-head reads the features channel (#191) ---------------------------------

CACHE = "features/siglip2-base-naflex-1024tok"


@pytest.fixture
def head_job(job, tmp_path: Path) -> dict[str, Any]:
    features = tmp_path / "in" / "features"
    (features / "features" / "v").mkdir(parents=True)
    (features / "cache.json").write_text("{}")
    (features / "features" / "v" / "a.npy").write_bytes(b"\1\2")
    (features / source.MARKER).write_text("")  # S3 holds it; it does not count
    head = request(
        source_sha256=source.digest_dir(job["code"]),
        manifests_sha256=source.digest_dir(job["manifests"]),
        constraints_sha256=hashlib.sha256(PINS).hexdigest(),
        instance_type="ml.m5.xlarge",
        command="train-head",
        max_runtime_hours=2.0,
        features_cache=CACHE,
        features_sha256=source.digest_dir(features),
    )
    env = {**head["Environment"], "TRAINING_JOB_NAME": "ssl-aerial-exp-1"}
    return {**job, "features": features, "links": tmp_path / "links", "env": env}


def prepare_head(job: dict[str, Any], env: dict[str, str]) -> list[str]:
    return entry.prepare(
        env,
        code_root=job["code"],
        data_root=job["data"],
        manifests=job["manifests"],
        constraints=job["constraints"],
        checkpoint_dir=job["checkpoints"],
        features=job["features"],
        feature_links=job["links"],
    )


def test_a_head_job_runs_train_head_on_the_cache_under_its_own_names(
    head_job,
) -> None:
    argv = prepare_head(head_job, head_job["env"])
    assert argv[0] == "train-head"
    assert argv[1:6] == BASE["baseline_args"][:5]
    flags = dict(zip(argv[1:], argv[2:], strict=False))
    assert "--checkpoint-dir" not in flags
    assert flags["--data-root"] == str(head_job["data"])
    assert flags["--manifests"] == str(head_job["manifests"])
    cache = Path(flags["--cache"])
    assert cache == head_job["links"] / CACHE
    assert cache.resolve() == head_job["features"].resolve()
    assert (cache / "features" / "v" / "a.npy").read_bytes() == b"\1\2"
    assert "-a1-" in flags["--run-name"]


def test_modified_features_in_the_job_are_refused(head_job) -> None:
    (head_job["features"] / "features" / "v" / "a.npy").write_bytes(b"\0")
    with pytest.raises(entry.JobRefused, match="features"):
        prepare_head(head_job, head_job["env"])


@pytest.mark.parametrize("cache", ["../x", "features/../../x", "x", "/abs/x", "a/b/c"])
def test_a_cache_name_that_is_not_two_plain_names_is_refused(head_job, cache) -> None:
    sent = json.loads(head_job["env"][launcher.FEATURES_ENV])
    env = {
        **head_job["env"],
        launcher.FEATURES_ENV: json.dumps({**sent, "cache": cache}),
    }
    with pytest.raises(entry.JobRefused, match="cache"):
        prepare_head(head_job, env)


def test_the_run_records_the_verified_features_in_its_attestation(
    in_job, head_job, monkeypatch
) -> None:
    # in_job first: it changes the code that head_job's request then hashes
    from aerial_search import cli

    monkeypatch.setattr(entry, "FEATURES", head_job["features"])
    monkeypatch.setattr(entry, "FEATURE_LINKS", head_job["links"])
    monkeypatch.setenv(entry.ATTESTATION_ENV, "")  # restored after the test
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "main",
        lambda argv: seen.update(json.loads(entry.os.environ[entry.ATTESTATION_ENV])),
    )
    entry.main(head_job["env"])
    sent = json.loads(head_job["env"][launcher.FEATURES_ENV])
    assert seen["features_sha256"] == sent["features_sha256"]
    assert seen["features_cache"] == CACHE
    assert seen["commit"] == "a" * 40
