"""Fetching a finished job (#113)."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from aerial_search import run as run_module
from aerial_search.cloud import launcher, source


def run_git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# --- fetching a finished job ------------------------------------------------------


def job_session(record: dict[str, Any]) -> Any:
    """A fake AWS session whose finished job wrote one run with `record`."""
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        data = json.dumps(record).encode()
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

    return Session()


def attested_record(repo: Path, **over: Any) -> dict[str, Any]:
    commit = run_git(repo, "rev-parse", "HEAD")
    digest = source.archive_commit(repo, commit)[1]
    attestation = {"commit": commit, "source_sha256": digest, "scratch": False}
    return {
        "status": "completed",
        "scratch": False,
        "commit": commit,
        "attestation": {**attestation, "verified_source_sha256": digest, **over},
    }


def test_fetch_puts_the_run_directory_under_outputs_and_records_billing(
    repo: Path, tmp_path: Path
) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    session = job_session(attested_record(repo))
    directories = launcher.fetch("ssl-aerial-exp-1", session, out, repo=repo)
    assert directories == [out / "r-a1"]
    assert (out / "r-a1" / "run.json").exists()
    job_record = json.loads((out / "r-a1" / "cloud-job.json").read_text())
    assert job_record["BillableTimeInSeconds"] == 5400
    with pytest.raises(FileExistsError):
        launcher.fetch("ssl-aerial-exp-1", session, out, repo=repo)


def test_fetch_refuses_an_attested_digest_that_is_not_the_commits(
    repo: Path, tmp_path: Path
) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    for record in (
        attested_record(repo, source_sha256="e" * 64),
        attested_record(repo, verified_source_sha256="e" * 64),
        {"status": "completed", "scratch": False},  # no attestation at all
    ):
        with pytest.raises(run_module.ProvenanceError):
            launcher.fetch("ssl-aerial-exp-1", job_session(record), out, repo=repo)
    assert not list(out.iterdir())


def test_fetch_refuses_a_normal_run_whose_commit_is_not_on_origin_main(
    repo: Path, tmp_path: Path
) -> None:
    (repo / "b.py").write_text("")
    run_git(repo, "add", "b.py")
    run_git(repo, "commit", "-m", "off main")
    out = tmp_path / "outputs"
    out.mkdir()
    with pytest.raises(run_module.ProvenanceError, match="origin/main"):
        launcher.fetch(
            "ssl-aerial-exp-1", job_session(attested_record(repo)), out, repo=repo
        )
    assert not list(out.iterdir())
    scratch = {**attested_record(repo, scratch=True), "scratch": True}
    assert launcher.fetch("ssl-aerial-exp-1", job_session(scratch), out, repo=repo)
