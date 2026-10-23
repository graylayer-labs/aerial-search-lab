"""Provenance without git: start_run inside a cloud job (#113)."""

import json
from pathlib import Path
from typing import Any

import pytest

from aerial_search import PACKAGE_DIR
from aerial_search import run as run_module
from aerial_search.cloud import source

# --- provenance without git: start_run in the job ---------------------------------


@pytest.fixture
def jobcode(tmp_path: Path) -> Path:
    code = tmp_path / "jobcode"
    (code / PACKAGE_DIR).mkdir(parents=True)
    (code / PACKAGE_DIR / "a.py").write_text("x = 1\n")
    return code / PACKAGE_DIR


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
        "source_sha256": source.digest_dir(jobcode.parents[3]),
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
    assert directory == jobcode.parents[3] / "outputs" / "r"
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
    assert not (jobcode.parents[3] / "outputs").exists()


def test_an_attestation_outside_a_sagemaker_job_is_not_honoured(
    jobcode: Path, monkeypatch, verified_data
) -> None:
    attest(monkeypatch, jobcode)
    monkeypatch.delenv("TRAINING_JOB_NAME")
    with pytest.raises(run_module.ProvenanceError):
        run_module.start_run("r", {}, 7, device="cpu", repo=jobcode, **verified_data)
