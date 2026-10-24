"""The source archive and its digest (#113)."""

import io
import tarfile
from pathlib import Path

from aerial_search import PACKAGE_DIR
from aerial_search.cloud import source


def extract(archive: bytes, where: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(where, filter="data")


def test_the_archive_digest_equals_the_digest_of_its_extracted_files(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    assert source.digest_dir(tmp_path / "code") == digest


def test_a_modified_file_changes_the_digest(repo: Path, tmp_path: Path) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    (tmp_path / "code" / PACKAGE_DIR / "a.py").write_text("x = 2\n")
    assert source.digest_dir(tmp_path / "code") != digest


def test_an_added_file_or_changed_mode_changes_the_digest(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    (tmp_path / "code" / "run.sh").chmod(0o644)
    assert source.digest_dir(tmp_path / "code") != digest
    (tmp_path / "code" / "run.sh").chmod(0o755)
    assert source.digest_dir(tmp_path / "code") == digest
    (tmp_path / "code" / "extra.py").write_text("")
    assert source.digest_dir(tmp_path / "code") != digest


def test_bytecode_and_the_outputs_directory_do_not_count(
    repo: Path, tmp_path: Path
) -> None:
    archive, digest = source.archive_commit(repo, "HEAD")
    extract(archive, tmp_path / "code")
    cache = tmp_path / "code" / PACKAGE_DIR / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.pyc").write_bytes(b"\0")
    (tmp_path / "code" / "outputs" / "r").mkdir(parents=True)
    (tmp_path / "code" / "outputs" / "r" / "run.json").write_text("{}")
    assert source.digest_dir(tmp_path / "code") == digest


def test_the_digest_of_chosen_files_equals_the_digest_of_a_copy_of_them(
    tmp_path: Path,
) -> None:
    # The launcher hashes part of the feature cache in place; the job hashes
    # the channel it received, which holds that part only (#191).
    whole, part = tmp_path / "whole", tmp_path / "part"
    files = {"cache.json": b"{}", "features/a/b.npy": b"\1" * (3 << 20)}
    for root, names in ((whole, [*files, "features/c/d.npy"]), (part, list(files))):
        for name in names:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(files.get(name, b"other"))
    assert source.digest_paths(whole, sorted(files)) == source.digest_dir(part)
    assert source.digest_dir(whole) != source.digest_dir(part)
