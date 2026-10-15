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
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from aerial_search import run as run_module
from aerial_search.cloud import cost, safe_name, source

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


def build_request(
    *,
    logical_run: str,
    attempt: int,
    commit: str,
    source_sha256: str,
    manifests_sha256: str,
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


@dataclass
class Plan:
    request: dict[str, Any]
    estimate: cost.Estimate
    archive: bytes
    source_key: str
    manifests_dir: Path
    manifests_prefix: str


def check_manifests(manifests: Path, scratch: bool) -> None:
    """Refuse early what the job would refuse: manifests that are not from a
    normal, verified `prepare`. (The job still verifies the data itself.)"""
    path = manifests / "data_quality.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        if scratch:
            return
        raise run_module.ProvenanceError(f"cannot read {path}: {error}") from None
    if scratch:
        return
    provenance, data = record.get("provenance"), record.get("data")
    if not isinstance(provenance, dict) or provenance.get("scratch") is not False:
        raise run_module.ProvenanceError(
            f"{path} comes from a scratch `prepare`; regenerate it from a clean "
            "tree on main, or pass --scratch"
        )
    if not isinstance(data, dict) or data.get("verified") is not True:
        raise run_module.ProvenanceError(
            f"{path} says its data was not verified (data.verified is not true)"
        )


def _exists(s3: Any, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except NotFound:
        return False
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise
    return True


def _upload(plan: Plan, s3: Any) -> None:
    """Content-addressed uploads: the source and the manifests are skipped if
    already there. Only `sagemaker/` is written."""
    if not _exists(s3, plan.source_key):
        s3.put_object(Bucket=BUCKET, Key=plan.source_key, Body=plan.archive)
    marker = plan.manifests_prefix + source.MARKER
    if not _exists(s3, marker):
        for path in sorted(plan.manifests_dir.rglob("*")):
            if path.is_file():
                key = (
                    plan.manifests_prefix
                    + path.relative_to(plan.manifests_dir).as_posix()
                )
                s3.upload_file(str(path), BUCKET, key)
        s3.put_object(Bucket=BUCKET, Key=marker, Body=b"")


def launch(
    *,
    repo: Path | None,
    logical_run: str,
    baseline_args: list[str],
    manifests: Path,
    scratch: bool,
    attempt: int,
    spot: bool,
    max_runtime_hours: float,
    max_wait_hours: float,
    expected_hours: float,
    eur_per_usd: float,
    month_to_date_eur: float | None,
    role_arn: str | None,
    image_uri: str | None,
    dry_run: bool,
    session_factory: Callable[[], Any],
    spot_price_usd: float = cost.SPOT_USD_PER_HOUR,
    today: date | None = None,
) -> Plan:
    """Check everything, build the job, and send it unless `dry_run`."""
    state = run_module.code_state(scratch=scratch, repo=repo)
    commit = state["commit"]
    if not commit or state["dirty"] or state["hidden_changes"]:
        raise run_module.ProvenanceError(
            "the job runs a commit, not the working tree: commit your changes "
            "first (even a scratch cloud run)"
        )
    check_manifests(manifests, scratch)
    session = None if dry_run else session_factory()
    if month_to_date_eur is None and session is not None:
        month_to_date_eur = cost.month_to_date_eur(
            session.client("ce", region_name="us-east-1"),
            eur_per_usd=eur_per_usd,
            today=today or datetime.now(UTC).date(),
        )
    estimate = cost.check(
        expected_hours=expected_hours,
        max_runtime_hours=max_runtime_hours,
        spot=spot,
        month_to_date_eur=month_to_date_eur,
        month_required=not dry_run,
        eur_per_usd=eur_per_usd,
        spot_price_usd=spot_price_usd,
    )
    # The whole repository: the entry point and pyproject.toml sit outside
    # the package, which is where the running code (the CLI default) lives.
    code_root = run_module._repo_root((repo or run_module._code_location()).resolve())
    archive, digest = source.archive_commit(code_root, commit)
    manifests_digest = source.digest_dir(manifests)
    if role_arn is None and session is not None:
        account = session.client("sts", region_name=REGION).get_caller_identity()[
            "Account"
        ]
        role_arn = f"arn:aws:iam::{account}:role/{ROLE_NAME}"
    request = build_request(
        logical_run=logical_run,
        attempt=attempt,
        commit=commit,
        source_sha256=digest,
        manifests_sha256=manifests_digest,
        baseline_args=baseline_args,
        role_arn=role_arn or f"arn:aws:iam::<account-id>:role/{ROLE_NAME}",
        spot=spot,
        max_runtime_hours=max_runtime_hours,
        max_wait_hours=max_wait_hours,
        scratch=scratch,
        image_uri=image_uri,
    )
    plan = Plan(
        request,
        estimate,
        archive,
        f"sagemaker/source/{commit[:12]}-{digest[:12]}/sourcedir.tar.gz",
        manifests,
        f"sagemaker/inputs/manifests/{manifests_digest[:16]}/",
    )
    if session is not None:
        _upload(plan, session.client("s3", region_name=REGION))
        session.client("sagemaker", region_name=REGION).create_training_job(**request)
    return plan


def fetch(job: str, session: Any, outputs: Path) -> list[Path]:
    """Copy a finished job's run directory from S3 into `outputs/`.

    Refuses to overwrite a run directory. Writes `cloud-job.json` beside each
    run's `run.json`: the job's status and billable seconds, which only
    SageMaker knows once the job has ended (`run.json` is left as the job
    wrote it).
    """
    description = session.client("sagemaker", region_name=REGION).describe_training_job(
        TrainingJobName=job
    )
    key = f"sagemaker/outputs/{job}/output/output.tar.gz"
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "output.tar.gz"
        session.client("s3", region_name=REGION).download_file(
            BUCKET, key, str(archive)
        )
        unpacked = Path(scratch) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        entries = sorted(unpacked.iterdir())
        clashes = [e.name for e in entries if (outputs / e.name).exists()]
        if clashes:
            raise FileExistsError(
                f"{', '.join(clashes)} already under {outputs}; runs are never "
                "overwritten"
            )
        moved = []
        for entry in entries:
            shutil.move(str(entry), outputs / entry.name)
            moved.append(outputs / entry.name)
    for directory in moved:
        if (directory / "run.json").exists():
            (directory / "cloud-job.json").write_text(
                json.dumps(description, indent=2, default=str) + "\n"
            )
    return moved
