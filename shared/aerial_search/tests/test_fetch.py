import zipfile
from pathlib import Path

import pytest

from aerial_search.data.fetch import _validate_member_paths


def test_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="Unsafe ZIP member path"),
    ):
        _validate_member_paths(archive, tmp_path / "output")


def test_accepts_archive_member_below_destination(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.zip"
    destination = tmp_path / "output"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/frame.jpg", "data")

    with zipfile.ZipFile(archive_path) as archive:
        _validate_member_paths(archive, destination)
