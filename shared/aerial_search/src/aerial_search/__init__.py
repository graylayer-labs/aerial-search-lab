"""Shared code for aerial person detection: data, folds, metrics, provenance."""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_DIR = Path("shared", "aerial_search", "src", "aerial_search")
"""Where this package sits in the repository, and in a cloud job's code."""


def repo_root(package_dir: Path) -> Path:
    """The root of the repository (or job code) that holds `package_dir`."""
    return package_dir.parents[len(PACKAGE_DIR.parts) - 1]


REPO_ROOT = repo_root(Path(__file__).resolve().parent)
