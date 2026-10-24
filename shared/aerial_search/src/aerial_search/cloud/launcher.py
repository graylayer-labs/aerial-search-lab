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
# The same container's CPU build, for the CPU instance (#188). UNVERIFIED in
# the same way: look the tag up in the registry before the first CPU job.
CPU_IMAGE_URI = (
    "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:"
    "2.13.0-cpu-py312-ubuntu22.04-sagemaker"
)
IMAGE_URIS = {INSTANCE_TYPE: IMAGE_URI, "ml.m5.xlarge": CPU_IMAGE_URI}
DEFAULT_COMMAND = "train-baseline"  # what the job runs if the request names none
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


FEATURES_ENV = "AERIAL_SEARCH_FEATURES"


@dataclass(frozen=True)
class Features:
    """Files of a feature cache that a job reads as its `features` channel.

    `root` is the cache directory; `files` are paths relative to it. The job
    gets the cache's parent and own directory names (`cache`), so a run reads
    it under the same names as on the laptop (#191)."""

    root: Path
    files: tuple[str, ...]

    @property
    def cache(self) -> str:
        return f"{self.root.parent.name}/{self.root.name}"


def features_prefix(cache: str, features_sha256: str) -> str:
    """Content-addressed like the manifests: one prefix per cache and digest."""
    name = safe_name(Path(cache).name)
    return f"sagemaker/inputs/features/{name}/{features_sha256[:16]}/"


def _channel(name: str, uri: str, mode: str = "File") -> dict[str, Any]:
    return {
        "ChannelName": name,
        "InputMode": mode,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


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
    instance_type: str = INSTANCE_TYPE,
    command: str = DEFAULT_COMMAND,
    features_cache: str | None = None,
    features_sha256: str | None = None,
) -> dict[str, Any]:
    """The `CreateTrainingJob` request. Pure: nothing is read or sent.

    With `features_cache` and `features_sha256`, a `features` channel and its
    digest in `AERIAL_SEARCH_FEATURES`, which the job verifies and adds to the
    attestation it records. Not in the attestation variable itself: with the
    digest that would pass the 512-character limit."""
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
        "instance_type": instance_type,
        "spot": spot,
        "scratch": scratch,
        "launched_at": now.isoformat(timespec="seconds"),
    }
    request: dict[str, Any] = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri or IMAGE_URIS[instance_type],
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
            "InstanceType": instance_type,
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
    if command != DEFAULT_COMMAND:  # the GPU request stays as it was
        request["Environment"]["AERIAL_SEARCH_COMMAND"] = command
    if features_cache is not None and features_sha256 is not None:
        request["InputDataConfig"].append(
            _channel("features", _s3(features_prefix(features_cache, features_sha256)))
        )
        request["Environment"][FEATURES_ENV] = json.dumps(
            {"features_sha256": features_sha256, "cache": features_cache}
        )
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
    constraints: bytes
    constraints_key: str
    features: Features | None = None
    features_prefix: str | None = None


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


def running_jobs(session: Any, now: datetime | None = None) -> list[dict[str, Any]]:
    """Every SageMaker training job in the region that is in progress, stopping
    or changed in the last `cost.LAG_HOURS`, described (account-wide, like the
    month-to-date spend). Each job is described once."""
    sm = session.client("sagemaker", region_name=REGION)
    since = (now or datetime.now(UTC)) - timedelta(hours=cost.LAG_HOURS)
    names: dict[str, None] = {}
    for filters in (
        {"StatusEquals": "InProgress"},
        {"StatusEquals": "Stopping"},
        {"LastModifiedTimeAfter": since},
    ):
        for page in sm.get_paginator("list_training_jobs").paginate(**filters):
            for summary in page["TrainingJobSummaries"]:
                names[summary["TrainingJobName"]] = None
    return [sm.describe_training_job(TrainingJobName=name) for name in names]


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


def _upload_files(s3: Any, root: Path, files: list[str], prefix: str) -> None:
    """Each file under `prefix`, then the marker; skipped if the marker is
    there (the prefix names the content's digest)."""
    marker = prefix + source.MARKER
    if not _exists(s3, marker):
        for name in files:
            s3.upload_file(str(root / name), BUCKET, prefix + name)
        s3.put_object(Bucket=BUCKET, Key=marker, Body=b"")


def _upload(plan: Plan, s3: Any) -> None:
    """Content-addressed uploads: the source, the manifests and the features
    are skipped if already there. Only `sagemaker/` is written."""
    if not _exists(s3, plan.source_key):
        s3.put_object(Bucket=BUCKET, Key=plan.source_key, Body=plan.archive)
    if not _exists(s3, plan.constraints_key):
        s3.put_object(Bucket=BUCKET, Key=plan.constraints_key, Body=plan.constraints)
    manifests = [
        path.relative_to(plan.manifests_dir).as_posix()
        for path in sorted(plan.manifests_dir.rglob("*"))
        if path.is_file()
    ]
    _upload_files(s3, plan.manifests_dir, manifests, plan.manifests_prefix)
    if plan.features is not None and plan.features_prefix is not None:
        _upload_files(
            s3, plan.features.root, list(plan.features.files), plan.features_prefix
        )


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
    spot_price_usd: float | None = None,  # default: the instance's
    today: date | None = None,
    export_constraints: Callable[[Path], bytes] = source.export_constraints,
    list_running: Callable[[Any], list[dict[str, Any]]] | None = None,
    instance_type: str = INSTANCE_TYPE,
    command: str = DEFAULT_COMMAND,
    features: Features | None = None,
) -> Plan:
    """Check everything, build the job, and send it unless `dry_run`.

    `instance_type` must be on the cost guard's list, which sets its prices
    and limits; `command` is the `aerial-search` command the job runs;
    `features`, checked by the caller, become the job's `features` channel."""
    instance = cost.limits(
        instance_type, spot=spot, max_runtime_hours=max_runtime_hours
    )
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
            usage_family=instance.usage_family,  # None, all SageMaker, for the GPU
        )
    running_eur = None
    if session is not None:
        running_eur = cost.running_jobs_eur(
            (list_running or running_jobs)(session),
            eur_per_usd=eur_per_usd,
            usage_family=instance.usage_family,
        )
    estimate = cost.check(
        expected_hours=expected_hours,
        max_runtime_hours=max_runtime_hours,
        spot=spot,
        month_to_date_eur=month_to_date_eur,
        running_jobs_eur=running_eur,
        month_required=not dry_run,
        eur_per_usd=eur_per_usd,
        spot_price_usd=instance.spot_usd if spot_price_usd is None else spot_price_usd,
        on_demand_price_usd=instance.on_demand_usd,
        run_limit_eur=instance.run_limit_eur,
        month_limit_eur=instance.month_limit_eur,
    )
    # The whole repository: the entry point and pyproject.toml sit outside
    # the package, which is where the running code (the CLI default) lives.
    code_root = run_module._repo_root((repo or run_module._code_location()).resolve())
    archive, digest = source.archive_commit(code_root, commit)
    manifests_digest = source.digest_dir(manifests)
    constraints = export_constraints(code_root)
    constraints_digest = hashlib.sha256(constraints).hexdigest()
    features_digest = (
        None if features is None else source.digest_paths(features.root, features.files)
    )
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
        constraints_sha256=constraints_digest,
        baseline_args=baseline_args,
        role_arn=role_arn or f"arn:aws:iam::<account-id>:role/{ROLE_NAME}",
        spot=spot,
        max_runtime_hours=max_runtime_hours,
        max_wait_hours=max_wait_hours,
        scratch=scratch,
        image_uri=image_uri,
        instance_type=instance_type,
        command=command,
        features_cache=None if features is None else features.cache,
        features_sha256=features_digest,
    )
    plan = Plan(
        request,
        estimate,
        archive,
        f"sagemaker/source/{commit[:12]}-{digest[:12]}/sourcedir.tar.gz",
        manifests,
        f"sagemaker/inputs/manifests/{manifests_digest[:16]}/",
        constraints,
        constraints_prefix(constraints_digest) + "constraints.txt",
        features,
        None
        if features is None or features_digest is None
        else features_prefix(features.cache, features_digest),
    )
    if session is not None:
        _upload(plan, session.client("s3", region_name=REGION))
        session.client("sagemaker", region_name=REGION).create_training_job(**request)
    return plan


def check_attestation(record: dict[str, Any], repo_root: Path) -> None:
    """Refuse a run whose attestation is not of a commit this repository has.

    The job's attestation came through its environment, which anyone able to
    create a job could have written. Here, on the laptop, the attested commit
    is archived again: its digest must equal the attested one and the one the
    job verified, and a normal run's commit must be on `origin/main`.
    """
    attestation = record.get("attestation")
    if not isinstance(attestation, dict) or not attestation.get("commit"):
        raise run_module.ProvenanceError("run.json has no attestation of its code")
    commit = str(attestation["commit"])
    resolved = run_module._git(
        repo_root, "rev-parse", "--verify", f"{commit}^{{commit}}"
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != commit:
        raise run_module.ProvenanceError(
            f"the attested commit {commit!r} is not a full commit SHA this "
            "repository has; a branch name or abbreviation could move"
        )
    if record.get("commit") != commit:
        raise run_module.ProvenanceError(
            f"run.json says commit {record.get('commit')} but the attestation "
            f"says {commit}"
        )
    try:
        digest = source.archive_commit(repo_root, commit)[1]
    except RuntimeError as error:
        raise run_module.ProvenanceError(
            f"cannot archive the attested commit {commit} here ({error}); "
            "`git fetch` it first"
        ) from None
    for field in ("source_sha256", "verified_source_sha256"):
        if attestation.get(field) != digest:
            raise run_module.ProvenanceError(
                f"the attested {field} {attestation.get(field)} is not the digest "
                f"of commit {commit} ({digest}); the job did not run that commit"
            )
    if record.get("scratch") is True and attestation.get("scratch") is True:
        return
    on_main = run_module._git(
        repo_root, "merge-base", "--is-ancestor", commit, run_module.MAIN_REF
    )
    if on_main.returncode != 0:
        raise run_module.ProvenanceError(
            f"the attested commit {commit} is not on origin/main "
            f"({on_main.stderr.strip() or 'not an ancestor'}); a normal run "
            "must be. `git fetch` if main moved"
        )


def fetch(
    job: str, session: Any, outputs: Path, repo: Path | None = None
) -> list[Path]:
    """Copy a finished job's run directory from S3 into `outputs/`.

    Refuses to overwrite a run directory, and refuses a run whose attestation
    does not match its commit archived again here (`check_attestation`).
    `repo` is for tests; by default the repository holding the running code.

    Writes `cloud-job.json` beside each run's `run.json`: the job's status
    and billable seconds, which only SageMaker knows once the job has ended
    (`run.json` is left as the job wrote it).
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
        root = run_module._repo_root((repo or run_module._code_location()).resolve())
        for entry in entries:
            if (entry / "run.json").exists():
                check_attestation(json.loads((entry / "run.json").read_text()), root)
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
