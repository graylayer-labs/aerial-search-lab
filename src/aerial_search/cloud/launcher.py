"""Build, check and send one SageMaker training job; fetch its run directory.

The launch refuses, before anything is sent, if the tree is dirty, HEAD is not
on `origin/main` (a scratch run may be off main, never dirty), the manifests
are not from a normal verified `prepare`, or the cost guard says no. It then
archives the commit, sends its digest, and creates the job. `--dry-run` stops
after building the request and touches no AWS API.

Everything the job writes goes under `sagemaker/` in the project bucket,
which is all its role may write. The job's design is in `entry.py` and the
provenance argument in `source.py`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from aerial_search.cloud import safe_name

REGION = "eu-west-1"
BUCKET = "ssl-aerial-person-detection-data-eu-west1"
DATA_PREFIX = "wisard/raw/wisard-full/"
INSTANCE_TYPE = "ml.g4dn.xlarge"
PROJECT_TAG = "ssl-aerial-person-detection"
ROLE_NAME = "ssl-aerial-person-detection-sagemaker-exec"
VOLUME_GB = 30
# AWS Deep Learning Container, PyTorch training, GPU. uv.lock pins torch
# 2.13.0 and the project needs Python 3.12 or later, so the tag is the torch
# 2.13.0 py312 CUDA build. UNVERIFIED: the tag follows the DLC naming scheme
# but was not looked up in the registry; check it (ecr describe-images on
# registry 763104351884, repository pytorch-training) before the first job.
IMAGE_URI = (
    "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:"
    "2.13.0-gpu-py312-cu128-ubuntu22.04-sagemaker"
)
ENTRY_POINT = "tools/sagemaker_entry.py"
CHECKPOINT_LOCAL = "/opt/ml/checkpoints"


class NotFound(Exception):
    """An S3 object does not exist."""


def _s3(path: str) -> str:
    return f"s3://{BUCKET}/{path}"


def job_name(logical_run: str, attempt: int) -> str:
    """`ssl-aerial-<run-name>-<attempt>`, at most 63 characters.

    Over the limit, the run name is cut and a hash of the whole name keeps two
    long names apart. The role only allows jobs named `ssl-aerial-*`.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", logical_run).strip("-")
    prefix, suffix = "ssl-aerial-", f"-{attempt}"
    if len(prefix) + len(slug) + len(suffix) <= 63:
        return prefix + slug + suffix
    digest = hashlib.sha256(logical_run.encode()).hexdigest()[:6]
    suffix = f"-{digest}{suffix}"
    return prefix + slug[: 63 - len(prefix) - len(suffix)].rstrip("-") + suffix


def constraints_prefix(constraints_sha256: str) -> str:
    return f"sagemaker/inputs/constraints/{constraints_sha256[:16]}/"


def build_request(
    *,
    logical_run: str,
    attempt: int,
    commit: str,
    source_sha256: str,
    manifests_sha256: str,
    constraints_sha256: str,
    baseline_args: list[str],
    role_arn: str,
    spot: bool,
    max_runtime_hours: float,
    max_wait_hours: float,
    scratch: bool,
    image_uri: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The `CreateTrainingJob` request. Pure: nothing is read or sent."""
    if spot and max_wait_hours < max_runtime_hours:
        raise ValueError(
            f"max wait {max_wait_hours} h must be at least the max runtime "
            f"{max_runtime_hours} h"
        )
    now = now or datetime.now(UTC)
    runtime = int(max_runtime_hours * 3600)
    stopping: dict[str, int] = {"MaxRuntimeInSeconds": runtime}
    if spot:
        stopping["MaxWaitTimeInSeconds"] = int(max_wait_hours * 3600)
    # Latest the job can still be running, plus a day, as a date.
    stop_by = now + timedelta(seconds=stopping.get("MaxWaitTimeInSeconds", runtime))
    name = job_name(logical_run, attempt)
    run_key = safe_name(logical_run)
    code_uri = _s3(
        f"sagemaker/source/{commit[:12]}-{source_sha256[:12]}/sourcedir.tar.gz"
    )
    attestation = {
        "commit": commit,
        "source_sha256": source_sha256,
        "manifests_sha256": manifests_sha256,
        "constraints_sha256": constraints_sha256,
        "instance_type": INSTANCE_TYPE,
        "spot": spot,
        "scratch": scratch,
        "launched_at": now.isoformat(timespec="seconds"),
    }
    request: dict[str, Any] = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri or IMAGE_URI,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "dataset",
                "InputMode": "FastFile",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": _s3(DATA_PREFIX),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "manifests",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": _s3(
                            f"sagemaker/inputs/manifests/{manifests_sha256[:16]}/"
                        ),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "constraints",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": _s3(constraints_prefix(constraints_sha256)),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "OutputDataConfig": {"S3OutputPath": _s3("sagemaker/outputs/")},
        "ResourceConfig": {
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": VOLUME_GB,
        },
        "StoppingCondition": stopping,
        "CheckpointConfig": {
            "S3Uri": _s3(f"sagemaker/checkpoints/{run_key}/"),
            "LocalPath": CHECKPOINT_LOCAL,
        },
        # The PyTorch container's toolkit downloads this archive, unpacks it
        # in /opt/ml/code and runs the program; values are JSON-encoded.
        "HyperParameters": {
            "sagemaker_program": json.dumps(ENTRY_POINT),
            "sagemaker_submit_directory": json.dumps(code_uri),
            "sagemaker_region": json.dumps(REGION),
            "sagemaker_job_name": json.dumps(name),
        },
        "Environment": {
            "AERIAL_SEARCH_ATTESTATION": json.dumps(attestation),
            "AERIAL_SEARCH_BASELINE_ARGS": json.dumps(baseline_args),
            "AERIAL_SEARCH_LOGICAL_RUN": logical_run,
        },
        "Tags": [
            {"Key": "project", "Value": PROJECT_TAG},
            {
                "Key": "stop-by",
                "Value": (stop_by + timedelta(days=1)).date().isoformat(),
            },
        ],
    }
    if spot:
        request["EnableManagedSpotTraining"] = True
    for key, value in request["Environment"].items():
        if len(value) > 512:
            raise ValueError(f"environment variable {key} is over 512 characters")
    return request
