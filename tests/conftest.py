"""Shared fixtures."""

import subprocess
from pathlib import Path

import pytest


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
