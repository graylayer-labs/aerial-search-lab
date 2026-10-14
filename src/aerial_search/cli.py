"""Command-line entry point for project workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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


# train-baseline flags that sagemaker-train passes on to the job, in order
BASELINE_FLAG_NAMES = [
    "percent",
    "seed",
    "arch",
    "input",
    "steps",
    "batch_size",
    "eval_every",
    "val_frames",
    "top_k",
    "workers",
    "tile_overlap",
    "frame_top_k",
    "min_visible",
    "empty_ratio",
    "learning_rate",
    "limit_frames",
    "device",
    "checkpoint_every",
    "fold",
]


def _baseline_args(args: argparse.Namespace) -> list[str]:
    """The `train-baseline` arguments the job runs, from `sagemaker-train`'s."""
    out = [args.camera]
    for name in BASELINE_FLAG_NAMES:
        if getattr(args, name) is not None:  # an unset flag keeps its default
            out += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return [*out, "--scratch"] if args.scratch else out


def _aws_session():
    import boto3

    return boto3.Session(region_name="eu-west-1")


def _add_baseline_flags(base: argparse.ArgumentParser, *, cloud: bool) -> None:
    """The `train-baseline` flags; `cloud` leaves out the paths the job sets."""
    base.add_argument("camera", choices=["rgb", "thermal"])
    base.add_argument("--percent", type=int, choices=PERCENTS, required=True)
    base.add_argument("--seed", type=int, default=SEED)
    if not cloud:
        base.add_argument(
            "--data-root", type=Path, default=Path("data/raw/wisard-full")
        )
    base.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-full")
    )
    from aerial_search.experiments.baseline_experiment import (
        ARCHITECTURES,
        INPUTS,
        BaselineRecipe,
    )

    base_defaults = BaselineRecipe()
    base.add_argument("--arch", choices=ARCHITECTURES, default=base_defaults.arch)
    base.add_argument(
        "--input",
        choices=INPUTS,
        default=base_defaults.input,
        help="cache: the frozen backbone's input size (the fair row); "
        "native: no resize; tiled: overlapping native-resolution tiles of the "
        "cache's input size, boxes stitched back to source pixels (#96)",
    )
    int_flags = ("steps", "batch_size", "eval_every", "val_frames", "top_k", "workers")
    for name in (*int_flags, "tile_overlap", "frame_top_k"):
        flag = "--" + name.replace("_", "-")
        base.add_argument(flag, type=int, default=getattr(base_defaults, name))
    for name in ("min_visible", "empty_ratio"):
        flag = "--" + name.replace("_", "-")
        base.add_argument(flag, type=float, default=getattr(base_defaults, name))
    base.add_argument(
        "--learning-rate", type=float, default=base_defaults.learning_rate
    )
    base.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="scratch only: keep this many frames per split (timing check)",
    )
    base.add_argument(
        "--device",
        default="cuda" if cloud else "cpu",
        help="cpu or cuda for a normal run (MPS diverged); the cloud job's "
        "default is cuda",
    )
    if not cloud:
        base.add_argument(
            "--checkpoint-dir",
            type=Path,
            help="where checkpoints and test predictions go as they are made "
            "(default: checkpoints/ in the run directory). If it holds a previous "
            "attempt of the same run (config, seed, commit, manifests), that "
            "attempt is resumed in a new run directory; another run's is refused. "
            "On SageMaker: /opt/ml/checkpoints (#112)",
        )
    from aerial_search.experiments.baseline_experiment import CHECKPOINT_EVERY

    base.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="training steps between checkpoints (also one when training ends)",
    )
    _add_fold_flag(base)
    _add_run_flags(base)


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
        help="default: the committed list, under src/aerial_search/data/checksums/",
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

    cache = subcommands.add_parser(
        "cache-features",
        help="run a frozen backbone once over one camera's images and save the "
        "patch features to outputs/features/",
    )
    check_cache = subcommands.add_parser(
        "check-cache",
        help="list frames the manifests reference that a feature cache lacks; "
        "exit 1 if any",
    )
    check_cache.add_argument("cache", type=Path, help="a features/<model>-<n>tok dir")
    check_cache.add_argument("manifests", type=Path, help="output of prepare")
    check_cache.add_argument(
        "--camera",
        choices=["rgb", "thermal"],
        help="check this camera only, for a cache that holds one (default: both)",
    )

    cache.add_argument("model", choices=["siglip2-base-naflex"])
    cache.add_argument("--camera", choices=["rgb", "thermal"], required=True)
    cache.add_argument(
        "--collection",
        action="append",
        metavar="ID",
        help="only this collection (repeat for several), to keep one session "
        "under 30 minutes; default: all",
    )
    cache.add_argument(
        "--part",
        type=_part,
        metavar="K/N",
        help="the Kth of N contiguous chunks of the selected images, for example "
        "1/2 and 2/2 to halve a session",
    )
    cache.add_argument("--token-budget", type=int, default=1024)
    cache.add_argument(
        "--thermal-input",
        choices=["replicate", "equalise"],
        default="replicate",
        help="how thermal frames reach the backbone (#68); equalise builds a "
        "separate cache, outputs/features/<model>-<n>tok-equalise; thermal only",
    )
    cache.add_argument(
        "--pooling", type=int, default=2, help="k of a k x k average pool; 1 is none"
    )
    cache.add_argument(
        "--verify",
        type=int,
        default=8,
        help="finished images to recompute and compare when resuming",
    )
    cache.add_argument("--limit", type=int, help="first N images only; needs --scratch")
    cache.add_argument("--data-root", type=Path, default=Path("data/raw/wisard-full"))
    cache.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-full")
    )
    _add_run_flags(cache)

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

    head = subcommands.add_parser(
        "train-head",
        help="train the person-centre head on cached features: one fold, one "
        "camera, one label fraction, one seed (docs/detection-head-design.md)",
    )
    head.add_argument("camera", choices=["rgb", "thermal"])
    head.add_argument("--percent", type=int, choices=PERCENTS, required=True)
    head.add_argument("--seed", type=int, default=SEED)
    head.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/features/siglip2-base-naflex-1024tok"),
        help="feature cache directory; only read",
    )
    head.add_argument("--data-root", type=Path, default=Path("data/raw/wisard-full"))
    head.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-full")
    )
    from aerial_search.experiments.head_experiment import Recipe
    from aerial_search.experiments.stem_experiment import StemRecipe

    defaults = Recipe()
    for name in ("steps", "batch_size", "eval_every", "upsample", "hidden", "top_k"):
        flag = "--" + name.replace("_", "-")
        head.add_argument(flag, type=int, default=getattr(defaults, name))
    head.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    head.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    head.add_argument("--device", help="default: mps if available, else cpu")
    head.add_argument(
        "--input-handling",
        choices=["replicate", "equalise", "stem"],
        help="thermal input arm (#68). replicate and equalise are read from "
        "--cache and this only checks it; stem trains a learned input stem "
        "before the frozen backbone and takes its geometry from the replicate "
        "--cache (thermal only; slow: see docs/thermal-input-ablation.md)",
    )
    head.add_argument(
        "--init-head",
        type=Path,
        metavar="RUN_DIR",
        help="warm start (#68): begin from the trained head of this finished "
        "arm-A run (same fold, percent, camera and seed; it must have saved "
        "head.pt). Use it with --steps 120 for the stem arm and for its matched "
        "control, so the two differ only in the stem",
    )
    stem_defaults = StemRecipe()
    for name in ("micro_batch", "eval_batch", "stem_hidden", "validation_stride"):
        flag = "--" + name.replace("_", "-")
        head.add_argument(
            flag, type=int, default=getattr(stem_defaults, name), help="stem arm only"
        )
    head.add_argument(
        "--loss-scale",
        type=float,
        default=stem_defaults.loss_scale,
        help="stem arm only",
    )
    _add_fold_flag(head)
    _add_run_flags(head)

    base = subcommands.add_parser(
        "train-baseline",
        help="fine-tune a COCO-pretrained torchvision detector on the same "
        "labels as train-head: one fold, camera, fraction, seed (#67)",
    )
    _add_baseline_flags(base, cloud=False)

    cloud = subcommands.add_parser(
        "sagemaker-train",
        help="run one train-baseline as one SageMaker spot training job on a "
        "commit of main, after a cost check (#113); takes train-baseline's flags",
    )
    _add_baseline_flags(cloud, cloud=True)
    cloud.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print the whole job request and the cost estimate; "
        "call no AWS API",
    )
    cloud.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="submission number of this run, in the job name; after a failed "
        "job use 2: the same run name resumes from the run's S3 checkpoints",
    )
    cloud.add_argument(
        "--on-demand", action="store_true", help="not spot (default: managed spot)"
    )
    cloud.add_argument("--max-runtime-hours", type=float, default=12.0)
    cloud.add_argument(
        "--max-wait-hours",
        type=float,
        default=18.0,
        help="spot only: runtime plus time spent waiting for capacity",
    )
    cloud.add_argument(
        "--expected-hours",
        type=float,
        help="for the estimate (default: from the measured laptop figures)",
    )
    cloud.add_argument(
        "--spot-price-usd", type=float, help="per hour, for the estimate"
    )
    cloud.add_argument("--eur-per-usd", type=float, help="the guard's rate")
    cloud.add_argument(
        "--month-to-date-eur",
        type=float,
        help="this month's cloud spend so far (default: read from Cost Explorer; "
        "refused if unreadable)",
    )
    cloud.add_argument("--image-uri", help="training container (default: pinned)")
    cloud.add_argument("--role-arn", help="execution role (default: the project's)")

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

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a project data command."""
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

    if args.command == "check-cache":
        from aerial_search.experiments import feature_cache

        cameras = (args.camera,) if args.camera else ("rgb", "thermal")
        missing = feature_cache.missing_frames(args.cache, args.manifests, cameras)
        for camera, paths in missing.items():
            print(f"{camera}: {len(paths)} missing")
            for path in paths:
                print(f"  {path}")
        if any(missing.values()):
            raise SystemExit(1)
        return

    if args.command == "cache-features":
        from aerial_search import run
        from aerial_search.experiments import feature_cache

        try:
            summary = feature_cache.cache_features(
                model=args.model,
                camera=args.camera,
                data_root=args.data_root,
                manifests=args.manifests,
                collections=args.collection,
                limit=args.limit,
                part=args.part,
                token_budget=args.token_budget,
                pooling=args.pooling,
                verify=args.verify,
                scratch=args.scratch,
                session=args.run_name,
                input_handling=args.thermal_input,
            )
        except (feature_cache.CacheError, ValueError, run.ProvenanceError) as error:
            raise SystemExit(f"refusing to cache: {error}") from error
        print(json.dumps(asdict(summary), indent=2))
        return

    if args.command == "train-head":
        import torch

        from aerial_search import run
        from aerial_search.experiments import feature_cache, head_experiment
        from aerial_search.models.components import get_device

        names = ("steps", "batch_size", "learning_rate", "weight_decay", "eval_every")
        names += ("upsample", "hidden", "top_k")
        recipe = head_experiment.Recipe(**{n: getattr(args, n) for n in names})
        stem = args.input_handling == "stem"
        try:
            if stem:
                from aerial_search.experiments import stem_experiment

                extra = ("micro_batch", "eval_batch", "stem_hidden", "loss_scale")
                extra += ("validation_stride",)
                summary = stem_experiment.run_stem_head(
                    cache_dir=args.cache,
                    manifests=args.manifests,
                    data_root=args.data_root,
                    fold=args.fold,
                    camera=args.camera,
                    percent=args.percent,
                    seed=args.seed,
                    recipe=stem_experiment.StemRecipe(
                        **asdict(recipe), **{n: getattr(args, n) for n in extra}
                    ),
                    scratch=args.scratch,
                    run_name=args.run_name,
                    device=torch.device(args.device) if args.device else get_device(),
                    init_head=args.init_head,
                )
                print(json.dumps(summary, indent=2))
                return
            summary = head_experiment.run_head(
                cache_dir=args.cache,
                manifests=args.manifests,
                data_root=args.data_root,
                fold=args.fold,
                camera=args.camera,
                percent=args.percent,
                seed=args.seed,
                recipe=recipe,
                scratch=args.scratch,
                run_name=args.run_name,
                device=torch.device(args.device) if args.device else get_device(),
                input_handling=args.input_handling,
                init_head=args.init_head,
            )
        except (
            feature_cache.CacheError,
            head_experiment.LeakError,
            run.ProvenanceError,
            ValueError,
        ) as error:
            raise SystemExit(f"refusing to start: {error}") from error
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-baseline":
        import torch

        from aerial_search import run
        from aerial_search.experiments import baseline_experiment

        names = ("arch", "input", "steps", "batch_size", "learning_rate")
        names += ("eval_every", "val_frames", "top_k", "workers")
        names += ("tile_overlap", "min_visible", "empty_ratio", "frame_top_k")
        recipe = baseline_experiment.BaselineRecipe(
            **{n: getattr(args, n) for n in names}
        )
        try:
            summary = baseline_experiment.run_baseline(
                manifests=args.manifests,
                data_root=args.data_root,
                fold=args.fold,
                camera=args.camera,
                percent=args.percent,
                seed=args.seed,
                recipe=recipe,
                scratch=args.scratch,
                run_name=args.run_name,
                device=torch.device(args.device),
                limit_frames=args.limit_frames,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
            )
        except (run.ProvenanceError, ValueError) as error:
            raise SystemExit(f"refusing to start: {error}") from error
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sagemaker-train":
        from aerial_search import run
        from aerial_search.cloud import cost, launcher
        from aerial_search.experiments import baseline_experiment

        recipe = baseline_experiment.BaselineRecipe(
            arch=args.arch, input=args.input, tile_overlap=args.tile_overlap
        )
        logical = args.run_name or baseline_experiment.default_run_name(
            recipe, args.camera, args.fold, args.percent, args.seed
        )
        baseline_args = _baseline_args(args)
        spot_price = args.spot_price_usd or cost.SPOT_USD_PER_HOUR
        try:
            plan = launcher.launch(
                repo=None,
                logical_run=logical,
                baseline_args=baseline_args,
                manifests=args.manifests,
                scratch=args.scratch,
                attempt=args.attempt,
                spot=not args.on_demand,
                max_runtime_hours=args.max_runtime_hours,
                max_wait_hours=args.max_wait_hours,
                expected_hours=args.expected_hours
                or cost.default_expected_hours(args.input, args.camera),
                eur_per_usd=args.eur_per_usd or cost.EUR_PER_USD,
                month_to_date_eur=args.month_to_date_eur,
                role_arn=args.role_arn,
                image_uri=args.image_uri,
                dry_run=args.dry_run,
                session_factory=_aws_session,
                spot_price_usd=spot_price,
            )
        except (run.ProvenanceError, cost.CostRefused, ValueError) as error:
            raise SystemExit(f"refusing to launch: {error}") from error
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "estimate": asdict(plan.estimate),
                    "request": plan.request,
                },
                indent=2,
            )
        )
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
        from aerial_search.experiments import predictions

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
        from aerial_search.experiments import head_table

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
        from aerial_search.experiments import head_table

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
