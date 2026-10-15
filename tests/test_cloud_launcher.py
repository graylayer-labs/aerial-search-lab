"""The SageMaker launcher, without AWS (#113)."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from aerial_search import run as run_module
from aerial_search.cloud import cost, launcher, source

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
    assert set(channels) == {"dataset", "manifests", "constraints"}
    data = channels["dataset"]
    assert data["InputMode"] == "FastFile"
    assert (
        data["DataSource"]["S3DataSource"]["S3Uri"]
        == "s3://ssl-aerial-person-detection-data-eu-west1/wisard/raw/wisard-full/"
    )
    manifests = channels["manifests"]["DataSource"]["S3DataSource"]["S3Uri"]
    assert manifests.endswith("/" + "c" * 16 + "/")
    pins = channels["constraints"]["DataSource"]["S3DataSource"]["S3Uri"]
    assert pins.endswith("sagemaker/inputs/constraints/" + "d" * 16 + "/")


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
    assert attestation["constraints_sha256"] == "d" * 64
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


def run_git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


PINS = b"pycocotools==2.0.11\ntorch==2.13.0\n"


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
        export_constraints=lambda root: PINS,
        list_running=lambda session: [],
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


def test_the_launch_ships_the_locked_versions_and_attests_them(
    repo: Path, tmp_path: Path
) -> None:

    plan = launch(repo, tmp_path)
    digest = hashlib.sha256(PINS).hexdigest()
    attestation = json.loads(plan.request["Environment"]["AERIAL_SEARCH_ATTESTATION"])
    assert attestation["constraints_sha256"] == digest
    assert plan.constraints == PINS
    assert plan.constraints_key == (
        f"sagemaker/inputs/constraints/{digest[:16]}/constraints.txt"
    )


def test_the_constraints_come_from_uv_export_of_the_lock() -> None:
    root = Path(__file__).resolve().parents[1]
    text = source.export_constraints(root).decode()
    assert "\npycocotools==" in text
    assert "--frozen" in text and "--no-hashes" in text


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


def test_unreadable_manifests_are_refused_unless_scratch(tmp_path: Path) -> None:
    with pytest.raises(run_module.ProvenanceError, match="cannot read"):
        launcher.check_manifests(tmp_path, scratch=False)
    launcher.check_manifests(tmp_path, scratch=True)
