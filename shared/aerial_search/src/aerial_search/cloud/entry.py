"""What runs inside the SageMaker job (#113).

The same code path as the laptop: `aerial-search train-baseline` with the
launcher's flags, plus the job's own paths, `--checkpoint-dir
/opt/ml/checkpoints` and a per-attempt run name (`<run>-a<n>-<UTC time>`).
A CPU `train-head` job (#191) gets `--cache`, the `features` channel
verified against its digest and linked under the cache's own names, in
place of the checkpoint directory.

Provenance without git. The launcher verified the tree and `origin/main` on
the laptop and archived exactly that commit. It sent two digests in the job's
environment: of the archive's files and of the manifests. Here:

1. `prepare` hashes `/opt/ml/code`, the manifests channel and the
   constraints file (`uv export` of the lock, made on the laptop at launch)
   and refuses to run on any difference (modified, added or removed file).
   The job then installs the project's dependencies with those constraints,
   except torch and torchvision, which the container provides.
2. `train-baseline` then calls `start_run`, which, finding no git work tree
   and an `AERIAL_SEARCH_ATTESTATION` inside a SageMaker job
   (`TRAINING_JOB_NAME` set), hashes the code again and records the commit,
   instance type, spot or on-demand, and job name in `run.json`.
3. The data is verified as on the laptop: `start_run` hashes the directories
   the manifests name, read through the FastFile mount, against the pinned
   list in the archived commit. A normal run's `run.json` records
   `data_verified_at`; a scratch run records the verification skipped.

What this does not prove: that the digest the launcher sent is of the commit
it names. That rests on the launcher having made the archive itself with
`git archive`, on the laptop, after the clean-tree and origin/main checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from aerial_search import REPO_ROOT
from aerial_search.cloud import safe_name, source

CODE_ROOT = REPO_ROOT
DATASET = Path("/opt/ml/input/data/dataset")
MANIFESTS = Path("/opt/ml/input/data/manifests")
CONSTRAINTS = Path("/opt/ml/input/data/constraints/constraints.txt")
CHECKPOINTS = Path("/opt/ml/checkpoints")
OUTPUT = Path("/opt/ml/output/data")  # uploaded as output.tar.gz, even on failure
PROVIDED = ("torch", "torchvision")  # in the container; not reinstalled
FEATURES = Path("/opt/ml/input/data/features")  # train-head's feature cache
# Where the cache is linked under its own parent and directory names, so a
# run reads it as on the laptop. Outside the code's outputs/, which is copied
# out at the end and would copy the whole cache through the link.
FEATURE_LINKS = Path("/opt/ml/feature-links")
ATTESTATION_ENV = "AERIAL_SEARCH_ATTESTATION"
FEATURES_ENV = "AERIAL_SEARCH_FEATURES"  # as the launcher's
_PLAIN = re.compile(r"[A-Za-z0-9._-]+")


class JobRefused(RuntimeError):
    """The job will not run: its code or inputs are not what was launched."""


def attempt_run_name(
    logical_run: str, checkpoint_dir: Path, now: datetime | None = None
) -> str:
    """`<run>-a<n>-<UTC time>`, a name no earlier attempt of this run used.

    A restart must never reuse a run directory name (`start_run` refuses an
    existing one). `attempts.json` in the checkpoint directory, synced to S3,
    lists the earlier attempts' names and gives `n`; the time, to the
    microsecond, keeps two attempts apart even when a spot restart (same job
    name) finds `attempts.json` not yet synced. The checkpoint directory
    stays the same for every attempt of the run.
    """
    path = checkpoint_dir / "attempts.json"
    earlier = set(json.loads(path.read_text())) if path.exists() else set()
    base = safe_name(logical_run)
    n = len(earlier) + 1
    while any(e == f"{base}-a{n}" or e.startswith(f"{base}-a{n}-") for e in earlier):
        n += 1
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{base}-a{n}-{stamp}"


def prepare(
    environ: Mapping[str, str],
    *,
    code_root: Path,
    data_root: Path,
    manifests: Path,
    constraints: Path,
    checkpoint_dir: Path,
    features: Path = FEATURES,
    feature_links: Path = FEATURE_LINKS,
) -> list[str]:
    """Verify the code and inputs; return the `aerial-search` arguments.

    `train-baseline` gets the dataset, manifests and checkpoint directory;
    `train-head` gets the dataset, manifests and the verified feature cache,
    linked under its own names in `feature_links`."""
    try:
        attestation = json.loads(environ[ATTESTATION_ENV])
        baseline_args = json.loads(environ["AERIAL_SEARCH_BASELINE_ARGS"])
        logical_run = environ["AERIAL_SEARCH_LOGICAL_RUN"]
    except (KeyError, ValueError) as error:
        raise JobRefused(f"the job's environment is incomplete: {error!r}") from None
    if not environ.get("TRAINING_JOB_NAME"):
        raise JobRefused("not inside a SageMaker training job (no TRAINING_JOB_NAME)")
    command = environ.get("AERIAL_SEARCH_COMMAND", "train-baseline")
    if command not in ("train-baseline", "train-head"):
        raise JobRefused(
            f"the job has no inputs for {command!r}; only train-baseline and "
            "train-head run in a job"
        )
    if command == "train-head" and FEATURES_ENV not in environ:
        raise JobRefused(
            f"train-head needs its feature cache, but {FEATURES_ENV} is unset"
        )
    actual = source.digest_dir(code_root)
    if actual != attestation.get("source_sha256"):
        raise JobRefused(
            f"the source in {code_root} ({actual}) is not the archive the "
            f"launcher made ({attestation.get('source_sha256')})"
        )
    actual = source.digest_dir(manifests)
    if actual != attestation.get("manifests_sha256"):
        raise JobRefused(
            f"the manifests in {manifests} ({actual}) are not the ones the "
            f"launcher sent ({attestation.get('manifests_sha256')})"
        )
    try:
        actual = hashlib.sha256(constraints.read_bytes()).hexdigest()
    except OSError as error:
        raise JobRefused(
            f"cannot read the constraints {constraints}: {error}"
        ) from None
    if actual != attestation.get("constraints_sha256"):
        raise JobRefused(
            f"the constraints in {constraints} ({actual}) are not the ones the "
            f"launcher sent ({attestation.get('constraints_sha256')})"
        )
    if command == "train-head":
        cache = _linked_cache(environ, features, feature_links)
        return [
            "train-head",
            *baseline_args,
            "--data-root",
            str(data_root),
            "--manifests",
            str(manifests),
            "--cache",
            str(cache),
            "--run-name",
            attempt_run_name(logical_run, checkpoint_dir),
        ]
    return [
        "train-baseline",
        *baseline_args,
        "--data-root",
        str(data_root),
        "--manifests",
        str(manifests),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--run-name",
        attempt_run_name(logical_run, checkpoint_dir),
    ]


def _sent_features(environ: Mapping[str, str]) -> tuple[str, str]:
    """The features digest and cache names the launcher sent."""
    try:
        sent = json.loads(environ[FEATURES_ENV])
        digest, cache = str(sent["features_sha256"]), str(sent["cache"])
    except (KeyError, ValueError, TypeError) as error:
        raise JobRefused(f"unreadable {FEATURES_ENV}: {error!r}") from None
    parts = cache.split("/")
    if len(parts) != 2 or not all(_PLAIN.fullmatch(p) and p != ".." for p in parts):
        raise JobRefused(f"the cache name {cache!r} is not <parent>/<directory>")
    return digest, cache


def _linked_cache(
    environ: Mapping[str, str], features: Path, feature_links: Path
) -> Path:
    """Verify the features channel and link it as `<links>/<parent>/<cache>`."""
    digest, cache = _sent_features(environ)
    actual = source.digest_dir(features)
    if actual != digest:
        raise JobRefused(
            f"the features in {features} ({actual}) are not the ones the "
            f"launcher sent ({digest})"
        )
    link = feature_links / cache
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    link.symlink_to(features, target_is_directory=True)
    return link


def job_attestation(environ: Mapping[str, str]) -> str:
    """The attestation the run records: the launcher's, plus the features
    digest and cache names (sent apart, for the 512-character limit) once
    `prepare` has verified them."""
    attestation = json.loads(environ[ATTESTATION_ENV])
    if FEATURES_ENV in environ:
        digest, cache = _sent_features(environ)
        attestation |= {"features_sha256": digest, "features_cache": cache}
    return json.dumps(attestation)


_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def requirement_name(spec: str) -> str | None:
    """The PEP 503 normalised name of a PEP 508 requirement or a pinned
    `uv export` line; None for a comment or blank line."""
    match = _NAME.match(spec)
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _members(code_root: Path) -> list[Path]:
    """The uv workspace members, as the root's `[tool.uv.workspace]` lists them."""
    root = tomllib.loads((code_root / "pyproject.toml").read_text())
    workspace = root["tool"]["uv"]["workspace"]
    excluded = {p for g in workspace.get("exclude", []) for p in code_root.glob(g)}
    found = [
        member
        for pattern in workspace["members"]
        for member in sorted(code_root.glob(pattern))
        if member not in excluded and (member / "pyproject.toml").is_file()
    ]
    return list(dict.fromkeys(found))


def source_dirs(code_root: Path) -> list[Path]:
    """Each member's `src/`, which the job puts on the path (nothing is installed)."""
    return [m / "src" for m in _members(code_root) if (m / "src").is_dir()]


def _dependencies(code_root: Path) -> list[str]:
    """Every member's dependencies, less the members themselves and PROVIDED."""
    projects = [
        tomllib.loads((m / "pyproject.toml").read_text())["project"]
        for m in _members(code_root)
    ]
    skip = {*PROVIDED, *(requirement_name(p["name"]) for p in projects)}
    dependencies: list[str] = []
    for project in projects:
        for spec in project.get("dependencies", []):
            if requirement_name(spec) not in skip and spec not in dependencies:
                dependencies.append(spec)
    return dependencies


def without_provided(constraints: str) -> str:
    """The constraints minus torch and torchvision, so pip keeps the
    container's builds (CUDA-matched) instead of the laptop's."""
    return "".join(
        line
        for line in constraints.splitlines(keepends=True)
        if requirement_name(line) not in PROVIDED
    )


def _install(dependencies: list[str], constraints: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        pins = Path(scratch) / "constraints.txt"
        pins.write_text(without_provided(constraints.read_text()))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "-c",
                str(pins),
                *dependencies,
            ],
            check=True,
        )


def _installed(dependencies: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for spec in [*PROVIDED, *dependencies]:
        name = requirement_name(spec) or spec
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def main(environ: Mapping[str, str] | None = None) -> None:
    """Entry point: verify, install the few missing packages, run, hand back."""
    environ = os.environ if environ is None else environ
    try:
        argv = prepare(
            environ,
            code_root=CODE_ROOT,
            data_root=DATASET,
            manifests=MANIFESTS,
            constraints=CONSTRAINTS,
            checkpoint_dir=CHECKPOINTS,
            features=FEATURES,
            feature_links=FEATURE_LINKS,
        )
    except JobRefused as error:
        raise SystemExit(f"refusing to run: {error}") from error
    if FEATURES_ENV in environ:  # start_run reads the attestation from here
        os.environ[ATTESTATION_ENV] = job_attestation(environ)
    dependencies = _dependencies(CODE_ROOT)
    _install(dependencies, CONSTRAINTS)
    sys.path[:0] = [str(d) for d in source_dirs(CODE_ROOT)]
    from aerial_search import cli, run

    name = argv[argv.index("--run-name") + 1]
    outputs = CODE_ROOT / "outputs"
    try:
        cli.main(argv)
    finally:
        directory = outputs / name
        if (directory / "run.json").exists():
            run.update_run(directory, installed_packages=_installed(dependencies))
        if outputs.exists():
            OUTPUT.mkdir(parents=True, exist_ok=True)
            for item in outputs.iterdir():
                shutil.copytree(item, OUTPUT / item.name, dirs_exist_ok=True)
