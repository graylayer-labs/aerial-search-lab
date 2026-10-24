"""Package a commit and hash code, so a job can prove it runs that commit.

A SageMaker job has no `.git`. The launcher checks the tree and `origin/main`
on the laptop, archives exactly that commit with `git archive`, and sends the
archive's digest with the job. The job hashes the files it was given and
refuses to run on a difference. The digest covers each file's path, its
executable bit and its content, so a changed, added or removed file shows.
Bytecode, `__pycache__` and a top-level `outputs/` do not count: the job
creates them after unpacking.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import subprocess
import tarfile
from collections.abc import Iterable
from pathlib import Path

IGNORED_TOP = ("outputs",)
MARKER = "_DONE"  # written last when a directory is uploaded; not content


def _counts(path: str) -> bool:
    parts = Path(path).parts
    return not (
        parts[0] in IGNORED_TOP
        or "__pycache__" in parts
        or parts[-1].endswith(".pyc")
        or parts[-1] == MARKER
    )


def _digest_hashes(files: Iterable[tuple[str, bool, str]]) -> str:
    lines = sorted(
        f"{path}\0{'x' if executable else '-'}\0{sha256}\n"
        for path, executable, sha256 in files
        if _counts(path)
    )
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def digest_files(files: Iterable[tuple[str, bool, bytes]]) -> str:
    """SHA-256 over sorted (path, executable, content SHA-256) lines."""
    return _digest_hashes(
        (path, executable, hashlib.sha256(data).hexdigest())
        for path, executable, data in files
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def digest_paths(root: Path, relative: Iterable[str]) -> str:
    """The digest of these files under `root`, as `digest_dir` of a directory
    holding only them. Read in chunks: the feature cache is gigabytes."""
    return _digest_hashes(
        (
            name,
            bool((root / name).stat().st_mode & 0o111),
            _file_sha256(root / name),
        )
        for name in relative
    )


def digest_dir(root: Path) -> str:
    """The digest of the regular files under `root`."""
    files = []
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path.relative_to(root).as_posix())
    return digest_paths(root, files)


def archive_commit(repo: Path, commit: str) -> tuple[bytes, str]:
    """`git archive` of `commit` as a gzip tar, and its digest.

    The tar is gzipped with a fixed timestamp, so the same commit gives the
    same bytes.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git archive {commit} failed: {result.stderr.decode()}")
    files = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tar:
        for member in tar:
            if not member.isreg():
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            files.append((member.name, bool(member.mode & 0o111), handle.read()))
    return gzip.compress(result.stdout, mtime=0), digest_files(files)


EXPORT = (
    "uv",
    "export",
    "--frozen",
    "--no-hashes",
    "--no-dev",
    "--all-packages",
    "--no-emit-workspace",  # pip refuses the members' editable lines as constraints
)


def export_constraints(repo: Path) -> bytes:
    """The locked versions of every runtime dependency, from `uv.lock`.

    The job installs with these as pip constraints, so it runs the versions
    the laptop's tests ran, not whatever `>=` resolves to that day. The tree
    is clean when the launcher calls this, so the lock is the commit's.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "VIRTUAL_ENV"
    }
    try:
        result = subprocess.run(
            list(EXPORT), cwd=repo, capture_output=True, env=env, check=False
        )
    except OSError as error:
        raise RuntimeError(f"cannot run uv: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(EXPORT)}` failed: {result.stderr.decode()}")
    return result.stdout
