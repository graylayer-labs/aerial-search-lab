"""Command-line entry point for project workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aerial_search.data import checksums
from aerial_search.data.fetch import WISARD_FULL, WISARD_SAMPLE, fetch_dataset
from aerial_search.data.folds import (
    PERCENTS,
    VALIDATION,
    ImageSize,
    check_folds,
    fold_table,
    train_file,
    view_dir,
    write_folds,
)
from aerial_search.data.folds import SEED as FOLD_SEED
from aerial_search.data.wisard import (
    WISARD_COLLECTIONS,
    prepare_manifests,
    select_collections,
)

DATASETS = {WISARD_SAMPLE.name: WISARD_SAMPLE, WISARD_FULL.name: WISARD_FULL}


SEED = 7  # both experiments seed with 7; recorded in run.json
VIEW = "paired"  # both experiments read the fold's paired-view manifests


@dataclass(frozen=True)
class ProjectCommand:
    """A command whose code lives in a project package (projects/<name>/)."""

    module: str  # defines main(argv); imported only when the command runs
    help: str


# Each approach's own commands. Shared code never imports a project: its module
# is imported only when its command runs, and is handed the rest of argv. The
# table is static, not entry points, because a SageMaker job puts the code on
# sys.path without installing it, so no entry point metadata exists there.
PROJECT_COMMANDS: dict[str, ProjectCommand] = {
    "cache-features": ProjectCommand(
        "frozen_head.cli",
        "run a frozen backbone once over one camera's images and save the "
        "patch features to outputs/features/",
    ),
    "train-head": ProjectCommand(
        "frozen_head.cli",
        "train the person-centre head on cached features: one fold, one "
        "camera, one label fraction, one seed (docs/detection-head-design.md)",
    ),
    "check-cache": ProjectCommand(
        "frozen_head.cli",
        "list frames the manifests reference that a feature cache lacks; exit 1 if any",
    ),
    "train-baseline": ProjectCommand(
        "finetuned_detector.cli",
        "fine-tune a COCO-pretrained torchvision detector on the same labels as "
        "train-head: one fold, camera, fraction, seed (#67)",
    ),
    "sagemaker-train": ProjectCommand(
        "finetuned_detector.cli",
        "run one train-baseline as one SageMaker spot training job on a commit "
        "of main, after a cost check (#113); takes train-baseline's flags",
    ),
}


def _run_project_command(name: str, argv: list[str]) -> None:
    module_name = PROJECT_COMMANDS[name].module
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and f"{module_name}.".startswith(f"{error.name}."):
            raise SystemExit(
                f"`aerial-search {name}` needs {module_name}, which is not "
                "installed; run `uv sync --dev`"
            ) from error
        raise
    module.main(argv)


def _part(text: str) -> tuple[int, int]:
    try:
        k, n = (int(v) for v in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not K/N, for example 1/2"
        ) from None
    if not 1 <= k <= n:
        raise argparse.ArgumentTypeError(f"{text!r}: K must be from 1 to N")
    return k, n


def _positive_hours(text: str) -> float:
    hours = float(text)
    if not hours > 0:
        raise argparse.ArgumentTypeError(f"{text!r}: must be more than 0 hours")
    return hours


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="debugging run: allowed from a dirty tree or off main, never quotable",
    )
    parser.add_argument(
        "--run-name", help="name under outputs/ (default: from command)"
    )


def _start_run(args: argparse.Namespace, default_name: str) -> Path:
    """Check provenance and write run.json; return the run directory.

    The run directory is the only place the experiment writes.
    """
    from aerial_search.models.components import get_device
    from aerial_search.run import ProvenanceError, start_run

    config = {k: v for k, v in vars(args).items() if k not in {"scratch", "run_name"}}
    view = _fold_manifests(args)
    inputs = [view / train_file(100), view / VALIDATION]
    try:
        return start_run(
            args.run_name or default_name,
            config,
            SEED,
            device=str(get_device()),
            scratch=args.scratch,
            inputs=inputs,
            checkpoint=getattr(args, "ssl_checkpoint", None),
            fold=args.fold,
            view=VIEW,
            manifests=args.manifests,
            data_root=args.data_root,
        )
    except ProvenanceError as error:
        raise SystemExit(f"refusing to start: {error}") from error


def _fold_manifests(args: argparse.Namespace) -> Path:
    """The paired-view manifests of the fold named by --fold."""
    return view_dir(args.manifests / "folds", args.fold, VIEW)


def _image_size(source: Path) -> ImageSize:
    from PIL import Image

    def size(path: str) -> tuple[int, int]:
        with Image.open(source / path) as image:
            return image.size

    return size


def _add_fold_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fold",
        required=True,
        metavar="SITE_DAY",
        help="test site-day of the fold to train on, e.g. 220109_Baker; reads "
        "<manifests>/folds/<SITE_DAY>/paired/",
    )


def _finish(directory: Path, error: BaseException | None = None) -> None:
    from aerial_search.run import finish_run

    finish_run(directory, error)


def _aws_session():
    import boto3

    return boto3.Session(region_name="eu-west-1")


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(prog="aerial-search")
    subcommands = parser.add_subparsers(dest="command", required=True)

    fetch = subcommands.add_parser("fetch", help="fetch a public dataset")
    fetch.add_argument("dataset", choices=list(DATASETS.keys()))
    fetch.add_argument("--data-root", type=Path, default=Path("data"))
    fetch.add_argument("--no-extract", action="store_true")

    checksum = subcommands.add_parser(
        "checksum", help="write the checksum list of a dataset directory"
    )
    checksum.add_argument("source", type=Path, help="e.g. data/raw/wisard-full")
    checksum.add_argument(
        "--output",
        type=Path,
        default=checksums.committed_path(),
        help="default: the committed list, under aerial_search/data/checksums/",
    )
    checksum.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing list; this re-pins the dataset",
    )
    check_data = subcommands.add_parser(
        "check-data", help="check a dataset directory against the committed list"
    )
    check_data.add_argument("source", type=Path, help="e.g. data/raw/wisard-full")
    check_data.add_argument("--list", type=Path, default=checksums.committed_path())
    check_data.add_argument(
        "--directory",
        action="append",
        metavar="NAME",
        help="check only this directory (repeat for several); default: all",
    )
    fetch_directory = subcommands.add_parser(
        "fetch-directory",
        help="fetch a WiSARD directory from S3 if absent, then verify it",
    )
    fetch_directory.add_argument("name")
    fetch_directory.add_argument(
        "--data-root", type=Path, default=Path("data/raw/wisard-full")
    )

    prepare = subcommands.add_parser("prepare", help="prepare WiSARD manifests")
    prepare.add_argument("source", type=Path)
    prepare.add_argument(
        "--output", type=Path, default=Path("data/manifests/wisard-sample")
    )
    prepare.add_argument(
        "--collection",
        action="append",
        metavar="ID",
        help="prepare only this collection (repeat for several); default: all",
    )

    prepare.add_argument(
        "--scratch",
        action="store_true",
        help="debugging: allowed from a dirty tree or off main, skips the data "
        "check; never quotable",
    )
    prepare.add_argument(
        "--force",
        action="store_true",
        help="let a scratch prepare overwrite manifests that are not scratch",
    )

    folds = subcommands.add_parser(
        "folds", help="write leave-one-site-day-out folds to <manifests>/folds"
    )
    check = subcommands.add_parser(
        "check-folds", help="check <manifests>/folds for leaks between splits"
    )
    for command in (folds, check):
        command.add_argument("source", type=Path, help="raw images, for image sizes")
        command.add_argument("manifests", type=Path, help="output of prepare")
    folds.add_argument("--seed", type=int, default=FOLD_SEED)

    train = subcommands.add_parser("train-ssl", help="run the paired SSL experiment")
    train.add_argument("--data-root", type=Path, default=Path("data/raw/wisard-sample"))
    train.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-sample")
    )
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    _add_fold_flag(train)
    _add_run_flags(train)

    detect = subcommands.add_parser("train-detector", help="run a detection baseline")
    detect.add_argument("modality", choices=["rgb", "thermal"])
    detect.add_argument("initialization", choices=["scratch", "ssl"])
    detect.add_argument(
        "--data-root", type=Path, default=Path("data/raw/wisard-sample")
    )
    detect.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-sample")
    )
    detect.add_argument("--epochs", type=int, default=5)
    detect.add_argument("--ssl-checkpoint", type=Path)
    _add_fold_flag(detect)
    _add_run_flags(detect)

    fetch = subcommands.add_parser(
        "sagemaker-fetch",
        help="copy a finished job's run directory from S3 into outputs/ and "
        "record its status and billable seconds (#113)",
    )
    fetch.add_argument("job", help="the training job's name")

    rescore = subcommands.add_parser(
        "rescore",
        help="score a finished detector run's test predictions file again, "
        "without the model; refuses a file missing any test frame (#112)",
    )
    rescore.add_argument("run", type=Path, help="the run directory")
    rescore.add_argument(
        "--by-size",
        action="store_true",
        help="also break the score down by person size: sqrt(box area) in "
        "source pixels, very_tiny <8, tiny 8-16, small 16-32, medium_plus >=32",
    )

    compare = subcommands.add_parser(
        "compare-table",
        help="head and baseline runs in one table, per fold with mean and "
        "spread; refuses runs that did not see the same frames",
    )
    compare.add_argument(
        "methods",
        nargs="+",
        metavar="NAME=DIR",
        help="e.g. head=outputs/detection-head/<cache> "
        "baseline=outputs/detection-baseline/fasterrcnn_resnet50_fpn-cache",
    )
    compare.add_argument("--json", type=Path, help="also write the summaries as JSON")

    table = subcommands.add_parser(
        "head-table",
        help="table of completed train-head runs: per camera, fold and fraction, "
        "with mean and spread over folds; refuses scratch or unfinished runs",
    )
    table.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="e.g. outputs/detection-head/siglip2-base-naflex-1024tok; with "
        "--ablation, one directory per input arm",
    )
    table.add_argument(
        "--ablation",
        action="store_true",
        help="the thermal input ablation (#68): one directory per arm, one table "
        "per camera and fraction with a row per arm",
    )
    table.add_argument("--camera", choices=["rgb", "thermal"], help="with --ablation")
    table.add_argument(
        "--percent",
        action="append",
        choices=[str(p) for p in PERCENTS],
        help="with --ablation: only this label fraction (repeat for several)",
    )
    table.add_argument("--json", type=Path, help="also write the summary as JSON")

    for name, command in PROJECT_COMMANDS.items():
        subcommands.add_parser(name, help=command.help, add_help=False)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a shared command, or hand a project's command to its package."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in PROJECT_COMMANDS:
        _run_project_command(argv[0], argv)
        return
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        source = DATASETS[args.dataset]
        destination = fetch_dataset(
            source,
            args.data_root,
            extract=not args.no_extract,
        )
        print(destination)
        return

    if args.command == "checksum":
        if args.output.exists() and not args.force:
            raise SystemExit(
                f"refusing to overwrite {args.output}: writing it re-pins the "
                "dataset. Pass --force if that is intended."
            )
        entries = checksums.hash_tree(args.source, progress=checksums.print_progress)
        checksums.write_list(entries, args.output)
        print(f"{len(entries):,} files, root hash {checksums.root_hash(entries)}")
        print(args.output)
        return

    if args.command == "check-data":
        expected = checksums.read_list(args.list)
        differences = checksums.check_tree(
            args.source, expected, args.directory, checksums.print_progress
        )
        if differences:
            for line in differences.lines():
                print(line)
            print(f"FAIL {args.source} differs from {args.list}")
            raise SystemExit(1)
        root = checksums.root_hash(expected)
        print(f"OK   {args.source} matches {args.list} (root hash {root})")
        return

    if args.command == "fetch-directory":
        try:
            print(checksums.ensure_directory(args.name, args.data_root))
        except (
            checksums.DataMismatchError,
            checksums.FetchError,
            ValueError,
        ) as error:
            raise SystemExit(f"FAIL {error}") from error
        return

    if args.command == "prepare":
        collections = (
            select_collections(args.collection)
            if args.collection
            else WISARD_COLLECTIONS
        )
        from aerial_search import run

        try:
            state = run.code_state(scratch=args.scratch)
        except run.ProvenanceError as error:
            raise SystemExit(f"refusing to prepare: {error}") from error
        provenance: dict[str, object] = {
            "commit": state["commit"],
            "scratch": args.scratch,
        }
        if args.scratch:
            provenance["dirty"] = state.get("dirty")
        try:
            counts = prepare_manifests(
                args.source,
                args.output,
                collections=collections,
                provenance=provenance,
                verify=not args.scratch,
                force=args.force,
            )
        except (checksums.DataMismatchError, ValueError) as error:
            raise SystemExit(f"refusing to prepare: {error}") from error
        print(", ".join(f"{name}: {count}" for name, count in counts.items()))
        return

    if args.command == "folds":
        summary = write_folds(
            args.manifests,
            args.manifests / "folds",
            _image_size(args.source),
            seed=args.seed,
        )
        print(fold_table(summary))
        return

    if args.command == "check-folds":
        report = check_folds(
            args.manifests, args.manifests / "folds", _image_size(args.source)
        )
        if report.problems:
            # The summary lines describe a check that did not pass; printing
            # them beside the failures would read as passes.
            for problem in report.problems:
                print(f"FAIL {problem}")
            raise SystemExit(1)
        for line in report.lines:
            print(f"OK   {line}")
        return

    if args.command == "sagemaker-fetch":
        from aerial_search import run
        from aerial_search.cloud import launcher

        try:
            directories = launcher.fetch(
                args.job,
                _aws_session(),
                run.outputs_dir(scratch=False),
            )
        except (OSError, run.ProvenanceError) as error:
            raise SystemExit(f"cannot fetch: {error}") from error
        print("\n".join(str(d) for d in directories))
        return

    if args.command == "rescore":
        from aerial_search.evaluation import predictions

        try:
            record = json.loads((args.run / "run.json").read_text())
            report = predictions.rescore(args.run)
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"refusing to re-score: {error}") from error
        out: dict[str, Any] = {
            "run": str(args.run),
            "status": record.get("status"),
            "scratch": record.get("scratch"),
            "overall": report.overall.to_dict(),
        }
        if args.by_size:
            out["by_size"] = {k: m.to_dict() for k, m in report.by_size.items()}
            out["size_buckets_sqrt_area_px"] = report.config[
                "size_buckets_sqrt_area_px"
            ]
        print(json.dumps(out, indent=2))
        return

    if args.command == "compare-table":
        from aerial_search.evaluation import head_table

        try:
            runs = {}
            for item in args.methods:
                name, _, directory = item.partition("=")
                if not directory:
                    raise head_table.TableError(f"{item!r} is not NAME=DIR")
                runs[name] = head_table.collect(Path(directory))
            head_table.check_same_frames(runs)
            summaries = {n: head_table.summarise(r) for n, r in runs.items()}
        except head_table.TableError as error:
            raise SystemExit(f"refusing to tabulate: {error}") from error
        print(head_table.markdown_compare(summaries, head_table.compare_caveat(runs)))
        if args.json:
            args.json.write_text(json.dumps(summaries, indent=2) + "\n")
        return

    if args.command == "head-table":
        from aerial_search.evaluation import head_table

        try:
            if args.ablation:
                by_arm = head_table.collect_arms(
                    args.runs, camera=args.camera, percents=args.percent
                )
                summary = head_table.summarise_arms(by_arm)
                print(head_table.markdown_arms(summary, head_table.arm_notes(by_arm)))
                if args.json:
                    args.json.write_text(json.dumps(summary, indent=2) + "\n")
                return
            if len(args.runs) != 1:
                raise head_table.TableError("several directories need --ablation")
            summary = head_table.summarise(head_table.collect(args.runs[0]))
        except head_table.TableError as error:
            raise SystemExit(f"refusing to tabulate: {error}") from error
        print(head_table.markdown(summary))
        if args.json:
            args.json.write_text(json.dumps(summary, indent=2) + "\n")
        return

    if args.command == "train-detector":
        directory = _start_run(args, f"detector-{args.modality}-{args.initialization}")
        from aerial_search.experiments.detection_experiment import (
            run_detection_experiment,
        )

        try:
            result = run_detection_experiment(
                args.data_root,
                _fold_manifests(args),
                directory,
                modality=args.modality,
                initialization=args.initialization,
                ssl_checkpoint=args.ssl_checkpoint,
                epochs=args.epochs,
                seed=SEED,
            )
        except BaseException as error:
            _finish(directory, error)
            raise
        _finish(directory)
        print(json.dumps(asdict(result), indent=2))
        return

    from aerial_search.experiments.ssl_experiment import run_experiment

    directory = _start_run(args, "ssl")
    try:
        result = run_experiment(
            args.data_root,
            _fold_manifests(args),
            directory,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=SEED,
        )
    except BaseException as error:
        _finish(directory, error)
        raise
    _finish(directory)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
