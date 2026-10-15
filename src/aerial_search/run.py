"""Run provenance: tie every run to a commit on main, or mark it as scratch.

Call `start_run` first in every experiment entry point, and `finish_run` when
the experiment returns or raises. A normal run starts only from a clean
working tree whose HEAD is on the local `origin/main`. A scratch run starts
from any state and is labelled so its numbers cannot be quoted. The check
fails closed: if git cannot answer, a normal run is refused.

The repository inspected is the one that holds this module (the running
code), never the current directory. `outputs/` is created at its root.
A non-editable install, whose code is not inside a git work tree, cannot make
a normal run.

A normal run is quotable only if its `run.json` says `"status": "completed"`.
`start_run` writes `"started"`; `finish_run` sets `"completed"` or `"failed"`.
A run left at `"started"` crashed or is still running: do not quote it.

Inputs (`inputs`, `checkpoint`) are recorded with their SHA-256. A normal run
that starts from a checkpoint needs a `run.json` beside it with `scratch`
false; its name and commit are recorded as `parent`.

A run on a fold records `fold` (the test site-day) and `view` (the label view)
at the top level of `run.json`. A normal run refuses a checkpoint whose run
recorded another fold or view, or none: pretraining in another fold has seen
this fold's test site-day. A scratch run may, and records the mismatch.

Runs are never overwritten. If `outputs/<run-name>/` already exists,
`start_run` raises `ProvenanceError`; choose another run name.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from aerial_search.data import checksums

MAIN_REF = "refs/remotes/origin/main"
SCRATCH_PREFIX = "scratch-"
PACKAGES = ("torch", "torchvision")


class ProvenanceError(RuntimeError):
    """A normal run was refused, or the run directory cannot be created."""


class GitError(ProvenanceError):
    """Git is missing or cannot answer (for example, not a work tree)."""


def _code_location() -> Path:
    """The directory of the running code. Tests replace this."""
    return Path(__file__).resolve().parent


def _git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # A hook's GIT_DIR must not redirect us to another repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        return subprocess.run(
            ["git", "-C", str(where), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except OSError as error:
        raise GitError(f"cannot run git: {error}") from error


def _git_ok(where: Path, *args: str) -> str:
    result = _git(where, *args)
    if result.returncode != 0:
        raise GitError(
            f"`git {' '.join(args)}` failed: {result.stderr.strip() or 'no output'}"
        )
    return result.stdout


def _repo_root(start: Path) -> Path:
    result = _git(start, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise GitError(
            f"{start} is not inside a git work tree (a non-editable install?), "
            f"so the running code cannot be tied to a commit: {result.stderr.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def _inspect(root: Path, start: Path) -> dict[str, Any]:
    """Commit, dirty output, hidden changes and on-main status of `root`."""
    if _git(root, "check-ignore", "-q", str(start)).returncode == 0:
        raise GitError(
            f"{start} is ignored by git, so it is not code the commit describes "
            "(a virtualenv install?)."
        )
    commit = _git_ok(root, "rev-parse", "HEAD").strip()
    # Untracked files are listed; ignored files are not.
    dirty = _git_ok(root, "status", "--porcelain")
    # Tracked files hidden from status: assume-unchanged (lowercase tag) or
    # skip-worktree ("S").
    hidden = [
        line[2:]
        for line in _git_ok(root, "ls-files", "-v").splitlines()
        if line[:1].islower() or line[:1] == "S"
    ]
    main_error = None
    on_main = None
    if _git(root, "rev-parse", "--verify", "--quiet", MAIN_REF).returncode != 0:
        main_error = (
            "origin/main is not known in this repository, so it cannot be "
            "checked whether HEAD is on main. A `git fetch` may be needed."
        )
    else:
        ancestor = _git(root, "merge-base", "--is-ancestor", "HEAD", MAIN_REF)
        if ancestor.returncode in (0, 1):
            on_main = ancestor.returncode == 0
        else:
            main_error = f"cannot compare HEAD with origin/main: {ancestor.stderr}"
    return {
        "commit": commit,
        "dirty": dirty,
        "hidden_changes": hidden,
        "commit_on_main": on_main,
        "main_error": main_error,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _accelerator(device: str) -> dict[str, Any]:
    """GPU name, CUDA and cuDNN versions and the determinism flags of a CUDA
    run; None for each on any other device. A GPU run's numbers may differ
    from a CPU run's, so the row must say which hardware made it."""
    if not device.startswith("cuda"):
        return {"gpu": None, "cuda": None, "cudnn": None, "determinism": None}
    import torch

    return {
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise ProvenanceError(f"cannot read input {path}: {error}") from error
    return digest.hexdigest()


def _parent(
    checkpoint: Path,
    scratch: bool,
    fold: str | None = None,
    view: str | None = None,
) -> dict[str, Any] | None:
    """The run that produced `checkpoint`. A normal run requires a normal one,
    from the same fold and label view."""
    record_path = checkpoint.parent / "run.json"
    try:
        record = json.loads(record_path.read_text())
        parent = {
            "run_name": record.get("run_name"),
            "commit": record.get("commit"),
            "path": str(record_path),
            "fold": record.get("fold"),
            "view": record.get("view"),
        }
        is_scratch = record.get("scratch") is not False
        status = record.get("status")
    except (OSError, ValueError):
        if scratch:
            return None
        raise ProvenanceError(
            f"{record_path} is missing or unreadable, so checkpoint {checkpoint} "
            "cannot be traced to a run. Use a checkpoint from a normal run, or "
            "pass --scratch."
        ) from None
    if is_scratch and not scratch:
        raise ProvenanceError(
            f"checkpoint {checkpoint} comes from a scratch run ({record_path} "
            'has "scratch" not false). Use a checkpoint from a normal run, or '
            "pass --scratch."
        )
    if status != "completed" and not scratch:
        # A run that crashed may have saved a checkpoint partway through.
        raise ProvenanceError(
            f"checkpoint {checkpoint} comes from a run that did not complete "
            f'({record_path} has "status": {json.dumps(status)}). Use a '
            "checkpoint from a completed run, or pass --scratch."
        )
    mismatch = [
        f"checkpoint {checkpoint} comes from a run on {name} "
        f"{json.dumps(parent[name])}, but this run is on {name} {json.dumps(mine)}"
        for name, mine in (("fold", fold), ("view", view))
        if parent[name] != mine
    ]
    if mismatch and not scratch:
        raise ProvenanceError(
            "; ".join(mismatch)
            + ". Pretraining in another fold has seen this fold's test site-day. "
            "Use a checkpoint from a run on the same fold and view, or pass "
            "--scratch."
        )
    parent["mismatch"] = mismatch
    return parent


ATTESTATION_ENV = "AERIAL_SEARCH_ATTESTATION"
JOB_ENV = "TRAINING_JOB_NAME"  # set by SageMaker in a training job


def _attested_state(
    start: Path, error: GitError, scratch: bool
) -> dict[str, Any] | None:
    """The code's state inside a SageMaker job, which has no `.git`.

    The launcher checked the tree and `origin/main` on the laptop and sent the
    commit and a digest of the archive it made from that commit
    (`aerial_search.cloud`). Here the digest of the code actually running must
    match it. Honoured only inside a training job, and a normal run needs an
    attestation from a normal launch. Returns None when there is none.
    """
    from aerial_search.cloud import source

    raw = os.environ.get(ATTESTATION_ENV)
    if not raw:
        return None
    job = os.environ.get(JOB_ENV)
    if not job:
        raise ProvenanceError(
            f"{ATTESTATION_ENV} is set but this is not a SageMaker training job "
            f"({JOB_ENV} is not set), so it is not honoured"
        ) from error
    try:
        attestation = json.loads(raw)
        commit, expected = attestation["commit"], attestation["source_sha256"]
    except (ValueError, KeyError, TypeError) as bad:
        raise ProvenanceError(f"unreadable {ATTESTATION_ENV}: {bad!r}") from None
    if not scratch and attestation.get("scratch") is not False:
        raise ProvenanceError(
            "the job was launched as a scratch run, so it cannot make a normal run"
        )
    actual = source.digest_dir(start.parents[1])
    if actual != expected:
        raise ProvenanceError(
            f"the source running here ({actual}) is not the archive of commit "
            f"{commit} that the launcher made ({expected})"
        )
    return {
        "commit": commit,
        "dirty": "",
        "hidden_changes": [],
        "commit_on_main": not attestation.get("scratch"),
        "main_error": None,
        "git_error": str(error),
        "attestation": {
            **attestation,
            "job_name": job,
            "verified_source_sha256": actual,
        },
    }


def _check_code(scratch: bool, repo: Path | None) -> tuple[Path, dict[str, Any]]:
    """The repository root and its state; a normal run must be clean, on main."""
    start = (repo or _code_location()).resolve()
    state: dict[str, Any]
    try:
        root = _repo_root(start)
        state = _inspect(root, start)
    except GitError as error:
        attested = _attested_state(start, error, scratch)
        if attested is not None:
            return start.parents[1], attested
        if not scratch:
            raise
        root = start if repo else Path.cwd().resolve()
        state = {
            "commit": None,
            "dirty": None,
            "hidden_changes": None,
            "commit_on_main": None,
            "main_error": None,
            "git_error": str(error),
        }

    if not scratch:
        if state["dirty"]:
            raise ProvenanceError(
                "the working tree has uncommitted changes (including untracked "
                "files not in .gitignore). Commit them, or pass --scratch:\n"
                + state["dirty"]
            )
        if state["hidden_changes"]:
            raise ProvenanceError(
                "tracked files are hidden from git status (skip-worktree or "
                "assume-unchanged): " + ", ".join(state["hidden_changes"])
            )
        if state["main_error"]:
            raise ProvenanceError(state["main_error"])
        if not state["commit_on_main"]:
            raise ProvenanceError(
                f"HEAD {state['commit']} is not on origin/main. Merge it first, "
                "or pass --scratch. If it was merged recently, a `git fetch` "
                "may be needed."
            )
    return root, state


def _verify_data(
    scratch: bool,
    manifests: Path | None,
    data_root: Path | None,
    pinned: Sequence[checksums.Entry] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Read the manifests' data record; a normal run re-verifies the data."""
    record_path = manifests / "data_quality.json" if manifests else None
    try:
        if record_path is None:
            raise ValueError("no manifests directory was given")
        quality = json.loads(record_path.read_text())
        data, provenance = quality.get("data"), quality.get("provenance")
    except (OSError, ValueError, AttributeError) as error:
        if scratch:
            return None, None, None
        raise ProvenanceError(
            f"cannot read the manifests' data_quality.json ({record_path}: "
            f"{error}); a normal run needs manifests from a normal `prepare`. "
            "Pass --scratch to debug."
        ) from None
    if scratch:
        return data, provenance, None
    if not isinstance(provenance, dict):
        raise ProvenanceError(
            f"{record_path} has no provenance record; regenerate the manifests "
            "with the current `aerial-search prepare`."
        )
    if provenance.get("scratch") is not False:
        raise ProvenanceError(
            f"{record_path} comes from a scratch `prepare` (provenance.scratch "
            "is not false); regenerate the manifests from a clean tree on main, "
            "or pass --scratch."
        )
    if not isinstance(data, dict) or data.get("verified") is not True:
        raise ProvenanceError(
            f"{record_path} says its data was not verified against the pinned "
            "list (data.verified is not true); run `prepare` on the pinned dataset."
        )
    if data_root is None:
        raise ProvenanceError("a normal run needs the data root to verify it")
    entries = list(pinned) if pinned is not None else checksums.committed_list()
    try:
        difference = checksums.verify_directories(
            data_root, data.get("directories", []), entries
        )
    except ValueError as error:
        raise ProvenanceError(str(error)) from None
    if difference:
        raise ProvenanceError(
            f"{data_root} differs from the pinned list:\n"
            + "\n".join(difference.lines())
        )
    return data, provenance, datetime.now(UTC).isoformat()


def outputs_dir(*, scratch: bool, repo: Path | None = None) -> Path:
    """The `outputs/` directory `start_run` writes under, without starting a run.

    For a command that must check something in its output location before it
    creates the run. Refuses outside a git work tree unless `scratch`.
    """
    start = (repo or _code_location()).resolve()
    try:
        return _repo_root(start) / "outputs"
    except GitError:
        if not scratch:
            raise
        return (start if repo else Path.cwd().resolve()) / "outputs"


def code_state(*, scratch: bool, repo: Path | None = None) -> dict[str, Any]:
    """Commit and cleanliness of the running code, refusing like `start_run`.

    For other commands that write results meant for quoting, such as
    `aerial-search prepare`. Raises `ProvenanceError` unless `scratch` or the
    tree is clean with HEAD on `origin/main`. Returns `commit`, `dirty`,
    `hidden_changes`, `commit_on_main`, and `main_error` (or `git_error`).
    """
    return _check_code(scratch, repo)[1]


def start_run(
    name: str,
    config: dict[str, Any],
    seed: int,
    *,
    device: str,
    scratch: bool = False,
    argv: list[str] | None = None,
    repo: Path | None = None,
    inputs: Sequence[Path] = (),
    checkpoint: Path | None = None,
    fold: str | None = None,
    view: str | None = None,
    manifests: Path | None = None,
    data_root: Path | None = None,
    pinned: Sequence[checksums.Entry] | None = None,
) -> Path:
    """Check provenance, create `outputs/[scratch-]<name>/`, write `run.json`.

    `repo` is for tests; by default the repository holding this module is
    used. `fold` and `view` name the fold's test site-day and label view, and
    must match those of the checkpoint's run. Returns the run directory, the
    only place the experiment may write.
    Raises `ProvenanceError` if a normal run is not from a clean tree on main,
    if git cannot tell, if a checkpoint has no normal parent run, or if the
    directory exists.

    `manifests` is the directory `prepare` wrote and `data_root` the raw data
    it read. A normal run needs `manifests/data_quality.json` from a normal
    `prepare` that verified its data, and re-verifies the directories it
    names under `data_root` against the pinned list (`pinned`, default the
    committed list) before starting; any difference or missing record raises
    `ProvenanceError`. `run.json` gets the record's `data` and `provenance`
    and `data_verified_at`. A scratch run skips the verification and says so.
    """
    root, state = _check_code(scratch, repo)
    data_record, manifest_provenance, verified_at = _verify_data(
        scratch, manifests, data_root, pinned
    )

    input_paths = [*inputs, *([checkpoint] if checkpoint else [])]
    recorded_inputs = [
        {"path": str(path), "sha256": _sha256(path)} for path in input_paths
    ]
    parent = _parent(checkpoint, scratch, fold, view) if checkpoint else None

    directory = root / "outputs" / (f"{SCRATCH_PREFIX}{name}" if scratch else name)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ProvenanceError(
            f"{directory} exists; runs are never overwritten. Choose another name."
        ) from error

    record = {
        "run_name": directory.name,
        "status": "started",
        "scratch": scratch,
        **{k: v for k, v in state.items() if k != "main_error"},
        "main_error": state["main_error"],
        "fold": fold,
        "view": view,
        "config": config,
        "seed": seed,
        "inputs": recorded_inputs,
        "parent": parent,
        "command": list(sys.argv if argv is None else argv),
        "started_at": datetime.now(UTC).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.machine(),
            "python": platform.python_version(),
            "device": device,
            **_accelerator(device),
        },
        "packages": _package_versions(),
        "data": data_record,
        "provenance": manifest_provenance,
        "data_verified_at": verified_at,
        "data_verification_skipped": scratch,
    }
    _write(directory, record)
    return directory


def _write(directory: Path, record: dict[str, Any]) -> None:
    (directory / "run.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n"
    )


def update_run(directory: Path, **fields: Any) -> None:
    """Add or replace top-level fields of the run's `run.json`.

    For facts known only after `start_run`, such as where a run keeps its
    checkpoints and the step it resumed from. The status is not changed.
    """
    record = json.loads((directory / "run.json").read_text())
    if "status" in fields:
        raise ValueError("update_run does not change the status; use finish_run")
    record.update(fields)
    _write(directory, record)


def finish_run(directory: Path, error: BaseException | None = None) -> None:
    """Mark the run `completed`, or `failed` with the error's type and message."""
    record = json.loads((directory / "run.json").read_text())
    record["finished_at"] = datetime.now(UTC).isoformat()
    if error is None:
        record["status"] = "completed"
    else:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    _write(directory, record)
