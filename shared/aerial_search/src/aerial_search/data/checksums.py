"""Pin a dataset by checksum, check a copy against the pin, fetch what is absent.

The pin is a list with one line per file: relative path, size, SHA-256. It is
committed gzipped under `checksums/`, because 100,000 files of hashes come to
about 15 MB as text. Per-file hashes are kept rather than per-directory ones
so that a check can name every file that differs. The gzip is written with a
fixed timestamp, so the same list always gives the same bytes.

The root hash is the SHA-256 of the list's uncompressed text, lines sorted by
path. It names the dataset in `run.json` and `data_quality.json`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aerial_search import REPO_ROOT

DATASET = "wisard-full"
BUCKET_URIS = {
    DATASET: "s3://ssl-aerial-person-detection-data-eu-west1/wisard/raw/wisard-full"
}
LIST_DIR = Path(__file__).resolve().parent / "checksums"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
WORKERS = 8

Progress = Callable[[str], None]
_UNSET: Any = object()


class DataMismatchError(RuntimeError):
    """A directory differs from the pinned list."""


class FetchError(RuntimeError):
    """The aws command failed; the message carries its error text."""


@dataclass(frozen=True, order=True)
class Entry:
    path: str
    size: int
    sha256: str


@dataclass
class Differences:
    """How a local tree differs from the pinned list, by relative path."""

    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.changed or self.missing or self.added)

    def lines(self) -> list[str]:
        return [
            *(f"changed {p}" for p in self.changed),
            *(f"missing {p}" for p in self.missing),
            *(f"added   {p}" for p in self.added),
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_directory(directory: Path, prefix: str) -> list[Entry]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())

    def one(path: Path) -> Entry:
        relative = path.relative_to(directory).as_posix()
        return Entry(f"{prefix}/{relative}", path.stat().st_size, _sha256(path))

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(one, files))


def hash_tree(
    root: Path,
    only: Sequence[str] | None = None,
    progress: Progress | None = None,
) -> list[Entry]:
    """Hash every file under `root`, or under its directories named in `only`.

    Paths are relative to `root`, with `/`, and sorted. `progress` is called
    once per directory. Files directly in `root` are listed only when `only`
    is not given.
    """
    if only is None:
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        names = sorted(only)
    entries: list[Entry] = []
    for index, name in enumerate(names, 1):
        if progress:
            progress(f"[{index}/{len(names)}] {name}")
        entries.extend(_hash_directory(root / name, name))
    if only is None:
        for path in root.iterdir():
            if path.is_file():
                entries.append(Entry(path.name, path.stat().st_size, _sha256(path)))
    return sorted(entries)


def _text(entries: Iterable[Entry]) -> bytes:
    lines = []
    for entry in sorted(entries):
        if "\t" in entry.path or "\n" in entry.path:
            raise ValueError(
                f"cannot list a path with a tab or newline: {entry.path!r}"
            )
        lines.append(f"{entry.path}\t{entry.size}\t{entry.sha256}\n")
    return "".join(lines).encode()


def root_hash(entries: Iterable[Entry]) -> str:
    """SHA-256 of the list's text: one number that names the whole dataset."""
    return hashlib.sha256(_text(entries)).hexdigest()


def write_list(entries: Iterable[Entry], path: Path) -> None:
    """Write the list gzipped, with a fixed timestamp so the bytes repeat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as zipped,
    ):
        zipped.write(_text(entries))


def read_list(path: Path) -> list[Entry]:
    entries = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            relative, size, digest = line.rstrip("\n").split("\t")
            entries.append(Entry(relative, int(size), digest))
    return entries


def committed_path(dataset: str = DATASET) -> Path:
    return LIST_DIR / f"{dataset}.tsv.gz"


def committed_list(dataset: str = DATASET) -> list[Entry]:
    return read_list(committed_path(dataset))


def committed_root_hash(dataset: str = DATASET) -> str:
    return root_hash(committed_list(dataset))


def compare(actual: Iterable[Entry], expected: Iterable[Entry]) -> Differences:
    """Every file that differs. A file is changed if its size or hash differs."""
    have = {e.path: e for e in actual}
    want = {e.path: e for e in expected}
    return Differences(
        changed=sorted(p for p in have.keys() & want.keys() if have[p] != want[p]),
        missing=sorted(want.keys() - have.keys()),
        added=sorted(have.keys() - want.keys()),
    )


def check_tree(
    root: Path,
    expected: Sequence[Entry],
    only: Sequence[str] | None = None,
    progress: Progress | None = None,
) -> Differences:
    """Compare `root` with the list; `only` limits both to named directories.

    A directory named in `only` that is absent from `root` has all its files
    reported missing.
    """
    if only is None:
        return compare(hash_tree(root, None, progress), expected)
    wanted = set(only)
    unknown = sorted(wanted - {e.path.split("/")[0] for e in expected})
    if unknown:
        raise ValueError(f"not in the pinned list: {', '.join(unknown)}")
    present = [name for name in only if (root / name).is_dir()]
    subset = [e for e in expected if e.path.split("/")[0] in wanted]
    return compare(hash_tree(root, present, progress), subset)


def aws_profile(settings: Path = SETTINGS) -> tuple[str | None, str]:
    """The AWS profile to use and where it came from.

    The environment wins, then `.claude/settings.json`, then none: with no
    profile the command uses the default credential chain (for example an
    instance role).
    """
    if os.environ.get("AWS_PROFILE"):
        return os.environ["AWS_PROFILE"], "environment"
    try:
        profile = json.loads(settings.read_text()).get("env", {}).get("AWS_PROFILE")
    except (OSError, ValueError):
        profile = None
    if profile:
        return profile, "settings file"
    return None, "default credentials"


def verify_directories(
    root: Path,
    names: Sequence[str],
    expected: Sequence[Entry],
    progress: Progress | None = None,
) -> Differences:
    """Hash only the named directories of `root` and compare with the list.

    Raises ValueError for a name the list does not hold.
    """
    return check_tree(root, expected, list(names), progress)


def ensure_directory(
    name: str,
    data_root: Path,
    *,
    expected: Sequence[Entry] | None = None,
    dataset: str = DATASET,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    profile: str | None = _UNSET,
    profile_source: str | None = None,
    bucket_uri: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Make `data_root/<name>` available, fetching it from S3 if absent.

    The directory, whether found or fetched, is checked against the pinned
    list. A fetch goes to a temporary sibling and is moved into place only if
    it matches, so an unverified directory is never left at its real path. A
    directory that was already there and does not match is refused and kept.

    Raises ValueError for a name that is not a directory of the list, and
    DataMismatchError naming each differing file.
    """
    pinned = list(expected) if expected is not None else committed_list(dataset)
    if "/" in name or name not in {e.path.split("/")[0] for e in pinned}:
        raise ValueError(f"{name!r} is not a directory of the pinned list")
    mine = [e for e in pinned if e.path.startswith(f"{name}/")]
    final = data_root / name

    def verify(directory: Path) -> None:
        diff = compare(_hash_directory(directory, name), mine)
        if diff:
            raise DataMismatchError(
                f"{directory} differs from the pinned list:\n" + "\n".join(diff.lines())
            )

    if final.exists():
        verify(final)
        return final

    uri = (bucket_uri or BUCKET_URIS[dataset]).rstrip("/")
    if profile == _UNSET:
        profile, profile_source = aws_profile()
    (log or print_progress)(
        f"fetching {name} from {uri}/{name} with "
        + (
            f"AWS profile {profile} ({profile_source})"
            if profile
            else f"{profile_source}"
        )
    )
    data_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".fetch-{name}-", dir=data_root))
    command = [
        *("aws", "s3", "cp", "--recursive", f"{uri}/{name}", str(temporary)),
        *(("--profile", profile) if profile else ()),
    ]
    try:
        try:
            runner(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").strip() or "no error text"
            raise FetchError(
                f"aws failed (exit {error.returncode}): {detail}"
            ) from error
        if not any(temporary.rglob("*")):
            # `aws s3 cp --recursive` exits 0 on a prefix that holds nothing.
            raise DataMismatchError(
                f"nothing was fetched from {uri}/{name}; check the bucket prefix"
            )
        verify(temporary)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
