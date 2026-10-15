"""The SageMaker launcher, without AWS (#113)."""

import json
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
