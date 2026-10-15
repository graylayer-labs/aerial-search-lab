"""The SageMaker launcher and its in-job entry point, without AWS (#113)."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from aerial_search import run as run_module
from aerial_search.cloud import cost, entry, launcher, source

# --- source: the archive and its digest ------------------------------------------


def run_git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo shaped like the project, HEAD equal to a fake origin/main."""
    path = tmp_path / "repo"
    (path / "src" / "aerial_search").mkdir(parents=True)
    run_git(path, "init", "-b", "main")
    (path / ".gitignore").write_text("outputs/\n")
    (path / "src" / "aerial_search" / "a.py").write_text("x = 1\n")
    (path / "run.sh").write_text("#!/bin/sh\n")
    (path / "run.sh").chmod(0o755)
    run_git(path, "add", ".")
    run_git(path, "commit", "-m", "init")
    run_git(path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return path


def extract(archive: bytes, where: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(where, filter="data")


def test_the_archive_digest_equals_the_digest_of_its_extracted_files(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    assert source.digest_dir(tmp_path / "code") == digest


def test_a_modified_file_changes_the_digest(repo: Path, tmp_path: Path) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    (tmp_path / "code" / "src" / "aerial_search" / "a.py").write_text("x = 2\n")
    assert source.digest_dir(tmp_path / "code") != digest


def test_an_added_file_or_changed_mode_changes_the_digest(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    (tmp_path / "code" / "run.sh").chmod(0o644)
    assert source.digest_dir(tmp_path / "code") != digest
    (tmp_path / "code" / "run.sh").chmod(0o755)
    assert source.digest_dir(tmp_path / "code") == digest
    (tmp_path / "code" / "extra.py").write_text("")
    assert source.digest_dir(tmp_path / "code") != digest


def test_bytecode_and_the_outputs_directory_do_not_count(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    cache = tmp_path / "code" / "src" / "aerial_search" / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "code" / "outputs" / "r").mkdir(parents=True)
    (tmp_path / "code" / "outputs" / "r" / "run.json").write_text("{}")
    assert source.digest_dir(tmp_path / "code") == digest


# --- cost guard -------------------------------------------------------------------


def estimate(**over: Any) -> cost.Estimate:
    args: dict[str, Any] = dict(
        expected_hours=3.0,
        max_runtime_hours=12.0,
        spot=True,
        month_to_date_eur=0.0,
    )
    return cost.check(**{**args, **over})


def test_the_estimate_is_price_times_expected_hours() -> None:
    e = estimate(eur_per_usd=1.0, spot_price_usd=0.40)
    assert e.estimate_usd == pytest.approx(1.2)
    assert e.worst_case_usd == pytest.approx(0.822 * 12)
    on_demand = estimate(eur_per_usd=1.0, spot=False)
    assert on_demand.estimate_usd == pytest.approx(0.822 * 3)


def test_the_rate_converts_to_euros() -> None:
    e = estimate(eur_per_usd=0.5)
    assert e.worst_case_eur == pytest.approx(0.822 * 12 * 0.5)


def test_a_worst_case_over_the_run_limit_is_refused() -> None:
    # 15 EUR at rate 1.0 buys 18.248 on-demand hours
    estimate(eur_per_usd=1.0, max_runtime_hours=18.0)
    with pytest.raises(cost.CostRefused, match="worst case"):
        estimate(eur_per_usd=1.0, max_runtime_hours=18.3, expected_hours=1.0)


def test_month_to_date_plus_estimate_over_the_month_limit_is_refused() -> None:
    e = estimate(eur_per_usd=1.0, expected_hours=2.5, month_to_date_eur=58.0)
    assert e.month_total_eur == pytest.approx(59.0)
    with pytest.raises(cost.CostRefused, match="month"):
        estimate(eur_per_usd=1.0, expected_hours=2.5, month_to_date_eur=59.5)


def test_an_unknown_month_to_date_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="month-to-date"):
        estimate(month_to_date_eur=None)


def test_a_dry_run_may_skip_the_month_check_but_says_so() -> None:
    e = estimate(month_to_date_eur=None, month_required=False)
    assert e.month_total_eur is None
    assert "not checked" in e.month_check


def test_expected_hours_beyond_the_max_runtime_is_refused() -> None:
    with pytest.raises(cost.CostRefused, match="expected"):
        estimate(expected_hours=13.0)


def test_cost_explorer_failure_is_not_a_zero() -> None:
    class Broken:
        def get_cost_and_usage(self, **_: Any) -> Any:
            raise RuntimeError("AccessDenied")

    with pytest.raises(cost.CostRefused, match="--month-to-date-eur"):
        cost.month_to_date_eur(Broken(), eur_per_usd=1.0, today=cost.date(2026, 10, 14))


def test_cost_explorer_amounts_are_summed_and_converted() -> None:
    seen: dict[str, Any] = {}

    class Ce:
        def get_cost_and_usage(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return {"ResultsByTime": [{"Total": {"UnblendedCost": {"Amount": "10.0"}}}]}

    assert cost.month_to_date_eur(
        Ce(), eur_per_usd=0.9, today=cost.date(2026, 10, 14)
    ) == pytest.approx(9.0)
    assert seen["TimePeriod"] == {"Start": "2026-10-01", "End": "2026-10-15"}


# --- the request ------------------------------------------------------------------

BASE: dict[str, Any] = dict(
    logical_run="detection-baseline/fasterrcnn_resnet50_fpn-tiled/rgb-220109_Baker-10pct-seed7",
    attempt=1,
    commit="a" * 40,
    source_sha256="b" * 64,
    manifests_sha256="c" * 64,
    baseline_args=["rgb", "--percent", "10", "--fold", "220109_Baker"],
    role_arn="arn:aws:iam::123456789012:role/r",
    spot=True,
    max_runtime_hours=12.0,
    max_wait_hours=18.0,
    scratch=False,
)


def request(**over: Any) -> dict[str, Any]:
    return launcher.build_request(**{**BASE, **over})


def test_the_request_is_a_spot_job_on_the_one_allowed_instance() -> None:
    r = request()
    assert r["ResourceConfig"]["InstanceType"] == "ml.g4dn.xlarge"
    assert r["ResourceConfig"]["InstanceCount"] == 1
    assert r["EnableManagedSpotTraining"] is True
    stop = r["StoppingCondition"]
    assert stop["MaxRuntimeInSeconds"] == 12 * 3600
    assert stop["MaxWaitTimeInSeconds"] == 18 * 3600
    assert stop["MaxWaitTimeInSeconds"] >= stop["MaxRuntimeInSeconds"]


def test_on_demand_has_no_spot_fields_and_no_max_wait() -> None:
    r = request(spot=False)
    assert "EnableManagedSpotTraining" not in r
    assert "MaxWaitTimeInSeconds" not in r["StoppingCondition"]
    assert r["StoppingCondition"]["MaxRuntimeInSeconds"] == 12 * 3600


def test_a_max_wait_below_the_max_runtime_is_refused() -> None:
    with pytest.raises(ValueError, match="wait"):
        request(max_wait_hours=11.0)


def test_the_checkpoint_uri_is_fixed_per_logical_run_not_per_attempt() -> None:
    a, b = request(attempt=1), request(attempt=2)
    assert a["CheckpointConfig"] == b["CheckpointConfig"]
    assert a["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"
    uri = a["CheckpointConfig"]["S3Uri"]
    assert uri.startswith("s3://ssl-aerial-person-detection-data-eu-west1/sagemaker/")
    assert request(logical_run="other")["CheckpointConfig"]["S3Uri"] != uri
    assert "//" not in uri.removeprefix("s3://")


def test_everything_the_job_writes_is_under_sagemaker() -> None:
    r = request()
    bucket = "s3://ssl-aerial-person-detection-data-eu-west1/"
    assert r["OutputDataConfig"]["S3OutputPath"] == bucket + "sagemaker/outputs/"
    assert r["CheckpointConfig"]["S3Uri"].startswith(bucket + "sagemaker/")
    for channel in r["InputDataConfig"]:
        uri = channel["DataSource"]["S3DataSource"]["S3Uri"]
        assert uri.startswith(bucket + "wisard/raw/") or uri.startswith(
            bucket + "sagemaker/"
        )
    hp = json.loads(r["HyperParameters"]["sagemaker_submit_directory"])
    assert hp.startswith(bucket + "sagemaker/")


def test_input_channels_are_data_in_fastfile_and_manifests() -> None:
    channels = {c["ChannelName"]: c for c in request()["InputDataConfig"]}
    assert set(channels) == {"dataset", "manifests"}
    data = channels["dataset"]
    assert data["InputMode"] == "FastFile"
    assert (
        data["DataSource"]["S3DataSource"]["S3Uri"]
        == "s3://ssl-aerial-person-detection-data-eu-west1/wisard/raw/wisard-full/"
    )
    manifests = channels["manifests"]["DataSource"]["S3DataSource"]["S3Uri"]
    assert manifests.endswith("/" + "c" * 16 + "/")


def test_the_job_is_tagged_with_project_and_stop_by() -> None:
    tags = {t["Key"]: t["Value"] for t in request()["Tags"]}
    assert tags["project"] == "ssl-aerial-person-detection"
    assert "stop-by" in tags


def test_job_names_fit_and_are_distinct_per_attempt_and_run() -> None:
    names = {
        launcher.job_name(BASE["logical_run"], 1),
        launcher.job_name(BASE["logical_run"], 2),
        launcher.job_name(BASE["logical_run"] + "x", 1),
        launcher.job_name("short", 1),
    }
    assert len(names) == 4
    for n in names:
        assert len(n) <= 63
        assert n.startswith("ssl-aerial-")
        assert n.replace("-", "").isalnum()
        assert n.endswith(("-1", "-2"))
    assert launcher.job_name("short", 3) == "ssl-aerial-short-3"


def test_the_environment_attests_the_code_and_fits_the_limits() -> None:
    r = request()
    env = r["Environment"]
    assert all(len(v) <= 512 for v in env.values())
    attestation = json.loads(env["AERIAL_SEARCH_ATTESTATION"])
    assert attestation["commit"] == "a" * 40
    assert attestation["source_sha256"] == "b" * 64
    assert attestation["manifests_sha256"] == "c" * 64
    assert attestation["instance_type"] == "ml.g4dn.xlarge"
    assert attestation["spot"] is True
    assert json.loads(env["AERIAL_SEARCH_BASELINE_ARGS"]) == BASE["baseline_args"]
    assert env["AERIAL_SEARCH_LOGICAL_RUN"] == BASE["logical_run"]


def test_the_job_runs_the_entry_point_from_the_pinned_source() -> None:
    hp = request()["HyperParameters"]
    assert json.loads(hp["sagemaker_program"]) == "tools/sagemaker_entry.py"
    assert json.loads(hp["sagemaker_submit_directory"]).endswith(
        "/" + "a" * 12 + "-" + "b" * 12 + "/sourcedir.tar.gz"
    )


# --- the launch: refusals and the dry run -----------------------------------------


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


def launch(repo: Path | None, tmp_path: Path, **over: Any) -> Any:
    manifests = tmp_path / "manifests"
    if not manifests.exists():
        write_manifests(manifests)

    def no_aws() -> Any:
        raise AssertionError("a dry run must not touch AWS")

    args: dict[str, Any] = dict(
        repo=repo,
        logical_run="exp",
        baseline_args=["rgb", "--percent", "10", "--fold", "f"],
        manifests=manifests,
        scratch=False,
        attempt=1,
        spot=True,
        max_runtime_hours=12.0,
        max_wait_hours=18.0,
        expected_hours=3.0,
        eur_per_usd=1.0,
        month_to_date_eur=0.0,
        role_arn="arn:aws:iam::1:role/r",
        image_uri=None,
        dry_run=True,
        session_factory=no_aws,
    )
    return launcher.launch(**{**args, **over})


def test_a_dry_run_builds_the_whole_request_and_sends_nothing(
    repo: Path, tmp_path: Path
) -> None:
    plan = launch(repo, tmp_path)
    assert plan.request["TrainingJobName"] == "ssl-aerial-exp-1"
    assert plan.estimate.estimate_usd == pytest.approx(1.2)
    attestation = json.loads(plan.request["Environment"]["AERIAL_SEARCH_ATTESTATION"])
    assert attestation["commit"] == run_git(repo, "rev-parse", "HEAD")


def test_the_cli_default_archives_the_whole_repository(
    repo: Path, tmp_path: Path, monkeypatch
) -> None:
    (repo / "tools").mkdir()
    (repo / "tools" / "sagemaker_entry.py").write_text("")
    (repo / "pyproject.toml").write_text("[project]\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "entry")
    run_git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(
        run_module, "_code_location", lambda: repo / "src" / "aerial_search"
    )
    plan = launch(None, tmp_path)
    with tarfile.open(fileobj=io.BytesIO(plan.archive), mode="r:gz") as tar:
        names = tar.getnames()
    assert "tools/sagemaker_entry.py" in names
    assert "pyproject.toml" in names
    assert "src/aerial_search/a.py" in names


def test_a_dirty_tree_is_refused(repo: Path, tmp_path: Path) -> None:
    (repo / "src" / "aerial_search" / "a.py").write_text("x = 3\n")
    with pytest.raises(run_module.ProvenanceError, match="uncommitted"):
        launch(repo, tmp_path)


def test_a_commit_not_on_origin_main_is_refused(repo: Path, tmp_path: Path) -> None:
    (repo / "b.py").write_text("")
    run_git(repo, "add", "b.py")
    run_git(repo, "commit", "-m", "b")
    with pytest.raises(run_module.ProvenanceError, match="origin/main"):
        launch(repo, tmp_path)


def test_a_scratch_cloud_run_may_be_off_main_but_never_dirty(
    repo: Path, tmp_path: Path
) -> None:
    (repo / "b.py").write_text("")
    run_git(repo, "add", "b.py")
    run_git(repo, "commit", "-m", "b")
    assert launch(repo, tmp_path, scratch=True).request
    (repo / "b.py").write_text("changed")
    with pytest.raises(run_module.ProvenanceError, match="commit"):
        launch(repo, tmp_path, scratch=True)


def test_unverified_or_scratch_manifests_are_refused_before_submission(
    repo: Path, tmp_path: Path
) -> None:
    write_manifests(tmp_path / "m2", verified=False)
    with pytest.raises(run_module.ProvenanceError, match="verified"):
        launch(repo, tmp_path, manifests=tmp_path / "m2")
    write_manifests(tmp_path / "m3", scratch=True)
    with pytest.raises(run_module.ProvenanceError, match="scratch"):
        launch(repo, tmp_path, manifests=tmp_path / "m3")


def test_the_cost_guard_runs_before_anything_is_sent(
    repo: Path, tmp_path: Path
) -> None:
    with pytest.raises(cost.CostRefused):
        launch(repo, tmp_path, max_runtime_hours=24.0, max_wait_hours=30.0)


def test_submission_uploads_the_source_then_creates_the_job(
    repo: Path, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []

    class S3:
        def head_object(self, **kw: Any) -> Any:
            calls.append(("head", kw["Key"]))
            raise launcher.NotFound()

        def put_object(self, **kw: Any) -> None:
            calls.append(("put", kw["Key"]))

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            calls.append(("upload", key))

    class Sm:
        def create_training_job(self, **kw: Any) -> Any:
            calls.append(("create", kw["TrainingJobName"]))
            return {"TrainingJobArn": "arn"}

    class Session:
        def client(self, name: str, **_: Any) -> Any:
            return {"s3": S3(), "sagemaker": Sm()}[name]

    launch(repo, tmp_path, dry_run=False, session_factory=Session)
    kinds = [k for k, _ in calls]
    assert kinds[-1] == "create"
    assert "put" in kinds[:-1]
    assert ("create", "ssl-aerial-exp-1") in calls
    assert all(key.startswith("sagemaker/") for kind, key in calls if kind != "create")


# --- inside the job ---------------------------------------------------------------


def job_env(code: Path, manifests: Path, **over: str) -> dict[str, str]:
    r = request(
        source_sha256=source.digest_dir(code),
        manifests_sha256=source.digest_dir(manifests),
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
    return {
        "code": code,
        "manifests": manifests,
        "data": tmp_path / "in" / "dataset",
        "checkpoints": tmp_path / "ckpt",
    }


def prepare(job: dict[str, Path], env: dict[str, str]) -> list[str]:
    return entry.prepare(
        env,
        code_root=job["code"],
        data_root=job["data"],
        manifests=job["manifests"],
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
    assert flags["--run-name"].endswith("-a1")


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


def test_a_name_in_attempts_json_is_never_reused_even_if_counts_disagree(job) -> None:
    job["checkpoints"].mkdir()
    logical = entry.safe_name(BASE["logical_run"])
    (job["checkpoints"] / "attempts.json").write_text(
        json.dumps([f"{logical}-a2"]) + "\n"
    )
    argv = prepare(job, job_env(job["code"], job["manifests"]))
    assert argv[argv.index("--run-name") + 1] != f"{logical}-a2"


# --- provenance without git: start_run in the job ---------------------------------


@pytest.fixture
def jobcode(tmp_path: Path) -> Path:
    code = tmp_path / "jobcode"
    (code / "src" / "aerial_search").mkdir(parents=True)
    (code / "src" / "aerial_search" / "a.py").write_text("x = 1\n")
    return code / "src" / "aerial_search"


@pytest.fixture
def verified_data(tmp_path: Path) -> dict[str, Any]:
    """A data root, its pinned list, and verified manifests from a normal prepare."""
    from aerial_search.data.checksums import hash_tree, root_hash

    root = tmp_path / "raw"
    (root / "d1").mkdir(parents=True)
    (root / "d1" / "f.jpg").write_bytes(b"frame")
    pinned = hash_tree(root)
    manifests = tmp_path / "manifests-verified"
    manifests.mkdir()
    (manifests / "data_quality.json").write_text(
        json.dumps(
            {
                "provenance": {"commit": "abc", "scratch": False},
                "data": {
                    "dataset": "wisard-full",
                    "root_hash": root_hash(pinned),
                    "verified": True,
                    "directories": ["d1"],
                },
            }
        )
    )
    return {"manifests": manifests, "data_root": root, "pinned": pinned}


def attest(monkeypatch, jobcode: Path, **over: Any) -> None:
    attestation = {
        "commit": "a" * 40,
        "source_sha256": source.digest_dir(jobcode.parents[1]),
        "manifests_sha256": "c" * 64,
        "instance_type": "ml.g4dn.xlarge",
        "spot": True,
        "scratch": False,
        **over,
    }
    monkeypatch.setenv("AERIAL_SEARCH_ATTESTATION", json.dumps(attestation))
    monkeypatch.setenv("TRAINING_JOB_NAME", "ssl-aerial-exp-1")


def test_an_attested_job_records_commit_instance_spot_and_job_name(
    jobcode: Path, monkeypatch, verified_data
) -> None:
    attest(monkeypatch, jobcode)
    directory = run_module.start_run(
        "r", {}, 7, device="cpu", repo=jobcode, **verified_data
    )
    record = json.loads((directory / "run.json").read_text())
    assert directory == jobcode.parents[1] / "outputs" / "r"
    assert record["commit"] == "a" * 40
    assert record["scratch"] is False
    assert record["attestation"]["instance_type"] == "ml.g4dn.xlarge"
    assert record["attestation"]["spot"] is True
    assert record["attestation"]["job_name"] == "ssl-aerial-exp-1"
    assert record["data_verified_at"]


def test_attested_code_that_was_modified_is_refused(
    jobcode: Path, monkeypatch, verified_data
) -> None:
    attest(monkeypatch, jobcode)
    (jobcode / "a.py").write_text("x = 2\n")
    with pytest.raises(run_module.ProvenanceError, match="source"):
        run_module.start_run("r", {}, 7, device="cpu", repo=jobcode, **verified_data)
    assert not (jobcode.parents[1] / "outputs").exists()


def test_an_attestation_outside_a_sagemaker_job_is_not_honoured(
    jobcode: Path, monkeypatch, verified_data
) -> None:
    attest(monkeypatch, jobcode)
    monkeypatch.delenv("TRAINING_JOB_NAME")
    with pytest.raises(run_module.ProvenanceError):
        run_module.start_run("r", {}, 7, device="cpu", repo=jobcode, **verified_data)


# --- the checkpoint race (#112 review) --------------------------------------------


def test_predictions_next_to_a_mid_training_checkpoint_are_set_aside_not_fatal(
    tmp_path: Path,
) -> None:
    from aerial_search.experiments.baseline_experiment import Checkpoints

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
    from aerial_search.experiments.baseline_experiment import Checkpoints

    saver = Checkpoints(tmp_path, {"seed": 7}, 10)
    saver.save(phase="trained", step=5)
    before = saver.path.stat().st_ino
    saver.rewrite()
    assert saver.path.stat().st_ino != before
    assert saver.load() is not None


def trained(directory: Path, weight: float) -> Any:
    import torch

    from aerial_search.experiments.baseline_experiment import Checkpoints

    saver = Checkpoints(directory, {"seed": 7}, 10)
    saver.save(phase="trained", step=5, model={"w": torch.full((2,), weight)})
    return saver


def predicted(saver: Any, image: str = "a") -> None:
    from aerial_search.experiments import predictions as pr

    log = pr.PredictionLog(saver.predictions)
    log.bind(saver.model_id)
    log.append(pr.Prediction(image, [[0, 0, 1, 1]], [0.5]))


def test_another_models_predictions_beside_a_trained_checkpoint_are_set_aside(
    tmp_path: Path,
) -> None:
    from aerial_search.experiments import predictions as pr
    from aerial_search.experiments.baseline_experiment import Checkpoints

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
    from aerial_search.experiments.baseline_experiment import Checkpoints

    mine = trained(tmp_path, 2.0)
    mine.predictions.write_text(
        '{"image": "a", "boxes": [[0, 0, 1, 1]], "scores": [0.5]}\n'
    )
    loader = Checkpoints(tmp_path, {"seed": 7}, 10)
    with pytest.warns(UserWarning):
        loader.load()
    assert not loader.predictions.exists()


def test_the_same_models_predictions_resume(tmp_path: Path) -> None:
    from aerial_search.experiments import predictions as pr
    from aerial_search.experiments.baseline_experiment import Checkpoints

    mine = trained(tmp_path, 2.0)
    predicted(mine)
    loader = Checkpoints(tmp_path, {"seed": 7}, 10)
    loader.load()
    assert list(pr.PredictionLog(loader.predictions).done) == ["a"]
    assert [p.image_id for p in pr.read_predictions(loader.predictions)] == ["a"]


def test_a_predictions_file_refuses_a_second_model(tmp_path: Path) -> None:
    from aerial_search.experiments import predictions as pr

    predicted(trained(tmp_path, 1.0))
    with pytest.raises(ValueError, match="model"):
        pr.PredictionLog(tmp_path / pr.PREDICTIONS_FILENAME).bind("f" * 64)


# --- fetching a finished job ------------------------------------------------------


def test_fetch_puts_the_run_directory_under_outputs_and_records_billing(
    tmp_path: Path,
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        data = b'{"status": "completed"}'
        info = tarfile.TarInfo("r-a1/run.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    class S3:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            assert key == "sagemaker/outputs/ssl-aerial-exp-1/output/output.tar.gz"
            Path(filename).write_bytes(payload.getvalue())

    class Sm:
        def describe_training_job(self, **kw: Any) -> Any:
            return {
                "TrainingJobName": kw["TrainingJobName"],
                "TrainingJobStatus": "Completed",
                "BillableTimeInSeconds": 5400,
                "TrainingTimeInSeconds": 7200,
                "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
            }

    class Session:
        def client(self, name: str, **_: Any) -> Any:
            return {"s3": S3(), "sagemaker": Sm()}[name]

    out = tmp_path / "outputs"
    out.mkdir()
    directories = launcher.fetch("ssl-aerial-exp-1", Session(), out)
    assert directories == [out / "r-a1"]
    assert (out / "r-a1" / "run.json").exists()
    job_record = json.loads((out / "r-a1" / "cloud-job.json").read_text())
    assert job_record["BillableTimeInSeconds"] == 5400
    with pytest.raises(FileExistsError):
        launcher.fetch("ssl-aerial-exp-1", Session(), out)


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
