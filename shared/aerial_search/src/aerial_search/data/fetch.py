"""Fetch public source datasets without adding them to Git."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    """A downloadable dataset archive with basic integrity metadata."""

    name: str
    url: str
    filename: str
    expected_bytes: int | None
    minimum_bytes: int | None = None
    download_method: str = "urllib"


WISARD_SAMPLE = DatasetSource(
    name="wisard-sample",
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1uSgMXuZGVCrWM_151UcykyHejxxaVHOo&export=download&confirm=t"
    ),
    filename="WiSARD_Multi_Modal_Sample.zip",
    expected_bytes=1_018_780_195,
)

WISARD_FULL = DatasetSource(
    name="wisard-full",
    url="1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht",
    filename="WiSARD_Multi_Modal_Full.zip",
    expected_bytes=None,
    minimum_bytes=38_000_000_000,
    download_method="gdown",
)


def fetch_dataset(
    source: DatasetSource, data_root: Path, *, extract: bool = True
) -> Path:
    """Download, validate, and optionally extract a dataset archive."""
    archive_dir = data_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / source.filename

    if source.download_method == "gdown":
        _download_via_gdown(source, archive_path)
    else:
        _download_resumable(source, archive_path)

    _validate_archive(source, archive_path)

    if not extract:
        return archive_path

    destination = data_root / "raw" / source.name
    marker = destination / ".complete"
    if marker.exists():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _validate_member_paths(archive, destination)
        archive.extractall(destination)
    marker.touch()
    return destination


def _check_disk_space(source: DatasetSource, destination: Path) -> None:
    """Verify sufficient free space for the download."""
    required_bytes = source.expected_bytes or source.minimum_bytes
    if required_bytes is None:
        return
    current_bytes = destination.stat().st_size if destination.exists() else 0
    free_bytes = shutil.disk_usage(destination.parent).free
    remaining_bytes = required_bytes - current_bytes
    if free_bytes < remaining_bytes:
        raise OSError(
            f"Not enough free space: need {remaining_bytes:,} bytes, "
            f"have {free_bytes:,}"
        )


def _is_complete_zip(path: Path) -> bool:
    """Check if a file exists and is a valid ZIP archive."""
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _download_via_gdown(source: DatasetSource, destination: Path) -> None:
    """Download from Google Drive using gdown."""
    if _is_complete_zip(destination):
        return
    _check_disk_space(source, destination)
    import gdown

    gdown.download(id=source.url, output=str(destination), resume=True, quiet=False)


def _download_resumable(source: DatasetSource, destination: Path) -> None:
    current_bytes = destination.stat().st_size if destination.exists() else 0
    if source.expected_bytes and current_bytes == source.expected_bytes:
        return
    if source.expected_bytes and current_bytes > source.expected_bytes:
        raise ValueError(f"Existing archive is larger than expected: {destination}")

    _check_disk_space(source, destination)

    request = urllib.request.Request(source.url)
    if current_bytes:
        request.add_header("Range", f"bytes={current_bytes}-")

    with urllib.request.urlopen(request) as response:  # noqa: S310
        resumed = current_bytes > 0 and response.status == 206
        mode = "ab" if resumed else "wb"
        with destination.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _validate_archive(source: DatasetSource, archive_path: Path) -> None:
    actual_bytes = archive_path.stat().st_size
    if source.expected_bytes is not None:
        if actual_bytes != source.expected_bytes:
            raise ValueError(
                f"Unexpected archive size for {source.name}: "
                f"expected {source.expected_bytes:,}, got {actual_bytes:,}"
            )
    elif source.minimum_bytes is not None and actual_bytes < source.minimum_bytes:
        raise ValueError(
            f"Archive size for {source.name} below minimum: "
            f"expected >= {source.minimum_bytes:,}, got {actual_bytes:,}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"Corrupt ZIP member: {bad_member}")


def _validate_member_paths(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if not member_path.is_relative_to(root):
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
