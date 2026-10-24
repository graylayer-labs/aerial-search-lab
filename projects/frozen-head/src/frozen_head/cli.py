"""The frozen head's commands, run through `aerial-search` (#160)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aerial_search import cli as shared

HELP = shared.PROJECT_COMMANDS
CPU_INSTANCE = "ml.m5.xlarge"  # the only CPU instance with a verified price
# train-head flags that sagemaker-train-head passes on to the job, in order
HEAD_FLAG_NAMES = [
    "percent",
    "seed",
    "steps",
    "batch_size",
    "eval_every",
    "upsample",
    "hidden",
    "top_k",
    "learning_rate",
    "weight_decay",
    "device",
    "input_handling",
    "micro_batch",
    "eval_batch",
    "stem_hidden",
    "validation_stride",
    "loss_scale",
    "fold",
]


def head_args(args: argparse.Namespace) -> list[str]:
    """The `train-head` arguments the job runs, from `sagemaker-train-head`'s."""
    out = [args.camera]
    for name in HEAD_FLAG_NAMES:
        if getattr(args, name) is not None:  # an unset flag keeps its default
            out += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return [*out, "--scratch"] if args.scratch else out


def _rate(text: str) -> float:
    """A EUR per USD rate no lower than the guard's, which errs high on purpose:
    a lower one would let the guard price a job below what it may cost."""
    from aerial_search.cloud import cost

    value = float(text)
    if not value >= cost.EUR_PER_USD:  # also refuses nan
        raise argparse.ArgumentTypeError(
            f"{text} is below the guard's {cost.EUR_PER_USD} EUR per USD"
        )
    return value


def _spend(text: str) -> float:
    """Spend so far, in euros: zero or more."""
    value = float(text)
    if not value >= 0:  # also refuses nan
        raise argparse.ArgumentTypeError(f"{text} EUR spent is not zero or more")
    return value


def _add_cloud_flags(cloud: argparse.ArgumentParser) -> None:
    cloud.add_argument(
        "--dry-run",
        action="store_true",
        help="build and print the whole job request and the cost estimate; "
        "call no AWS API",
    )
    cloud.add_argument("--attempt", type=int, default=1, help="as sagemaker-train")
    cloud.add_argument(
        "--max-runtime-hours", type=float, default=2.0, help="at most 2 on CPU"
    )
    cloud.add_argument("--max-wait-hours", type=float, default=4.0)
    cloud.add_argument(
        "--expected-hours", type=shared._positive_hours, help="for the estimate"
    )
    cloud.add_argument(
        "--eur-per-usd", type=_rate, help="the guard's rate; no lower than 0.95"
    )
    cloud.add_argument(
        "--month-to-date-eur",
        type=_spend,
        help="this month's SageMaker spend so far (default: Cost Explorer)",
    )
    cloud.add_argument("--image-uri", help="training container (default: pinned)")
    cloud.add_argument("--role-arn", help="execution role (default: the project's)")


def add_head_flags(head: argparse.ArgumentParser, *, cloud: bool) -> None:
    """The `train-head` flags; `cloud` leaves out the paths the job would set
    (`--manifests` and `--cache` stay: the launcher reads them here)."""
    head.add_argument("camera", choices=["rgb", "thermal"])
    head.add_argument("--percent", type=int, choices=shared.PERCENTS, required=True)
    head.add_argument("--seed", type=int, default=shared.SEED)
    head.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/features/siglip2-base-naflex-1024tok"),
        help="feature cache directory; only read",
    )
    if not cloud:
        head.add_argument(
            "--data-root", type=Path, default=Path("data/raw/wisard-full")
        )
    head.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-full")
    )
    from frozen_head.head_experiment import Recipe
    from frozen_head.stem_experiment import StemRecipe

    defaults = Recipe()
    for name in ("steps", "batch_size", "eval_every", "upsample", "hidden", "top_k"):
        flag = "--" + name.replace("_", "-")
        head.add_argument(flag, type=int, default=getattr(defaults, name))
    head.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    head.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    head.add_argument(
        "--device",
        default="cpu" if cloud else None,
        help="default: mps if available, else cpu; the CPU job's is cpu",
    )
    head.add_argument(
        "--input-handling",
        choices=["replicate", "equalise", "stem"],
        help="thermal input arm (#68). replicate and equalise are read from "
        "--cache and this only checks it; stem trains a learned input stem "
        "before the frozen backbone and takes its geometry from the replicate "
        "--cache (thermal only; slow: see docs/thermal-input-ablation.md)",
    )
    if not cloud:
        head.add_argument(
            "--init-head",
            type=Path,
            metavar="RUN_DIR",
            help="warm start (#68): begin from the trained head of this finished "
            "arm-A run (same fold, percent, camera and seed; it must have saved "
            "head.pt). Use it with --steps 120 for the stem arm and for its "
            "matched control, so the two differ only in the stem",
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
    shared._add_fold_flag(head)
    shared._add_run_flags(head)


def build_parser() -> argparse.ArgumentParser:
    """The parser of this project's commands, as `aerial-search` runs them."""
    parser = argparse.ArgumentParser(prog="aerial-search")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache-features", help=HELP["cache-features"].help)
    check_cache = commands.add_parser("check-cache", help=HELP["check-cache"].help)
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
        type=shared._part,
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
    shared._add_run_flags(cache)
    head = commands.add_parser("train-head", help=HELP["train-head"].help)
    add_head_flags(head, cloud=False)
    cloud = commands.add_parser(
        "sagemaker-train-head", help=HELP["sagemaker-train-head"].help
    )
    add_head_flags(cloud, cloud=True)
    _add_cloud_flags(cloud)
    return parser


def main(argv: list[str]) -> None:
    """Run one of this project's commands; `argv` starts with its name."""
    args = build_parser().parse_args(argv)
    if args.command == "check-cache":
        from frozen_head import feature_cache

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
        from frozen_head import feature_cache

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
        from aerial_search.data import records
        from aerial_search.models.components import get_device
        from frozen_head import feature_cache, head_experiment

        names = ("steps", "batch_size", "learning_rate", "weight_decay", "eval_every")
        names += ("upsample", "hidden", "top_k")
        recipe = head_experiment.Recipe(**{n: getattr(args, n) for n in names})
        stem = args.input_handling == "stem"
        try:
            if stem:
                from frozen_head import stem_experiment

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
            records.LeakError,
            run.ProvenanceError,
            ValueError,
        ) as error:
            raise SystemExit(f"refusing to start: {error}") from error
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sagemaker-train-head":
        from aerial_search import run
        from aerial_search.cloud import cost, launcher
        from aerial_search.data import records
        from frozen_head import feature_cache, head_experiment

        logical = args.run_name or head_experiment.default_run_name(
            args.cache, args.camera, args.fold, args.percent, args.seed
        )
        try:
            files = head_experiment.job_files(
                args.cache,
                args.manifests,
                fold=args.fold,
                camera=args.camera,
                percent=args.percent,
                scratch=args.scratch,
            )
            features = launcher.Features(args.cache, files)
            plan = launcher.launch(
                repo=None,
                logical_run=logical,
                baseline_args=head_args(args),
                manifests=args.manifests,
                scratch=args.scratch,
                attempt=args.attempt,
                spot=True,
                max_runtime_hours=args.max_runtime_hours,
                max_wait_hours=args.max_wait_hours,
                expected_hours=args.expected_hours or cost.EXPECTED_HOURS_OTHER,
                eur_per_usd=args.eur_per_usd or cost.EUR_PER_USD,
                month_to_date_eur=args.month_to_date_eur,
                role_arn=args.role_arn,
                image_uri=args.image_uri,
                dry_run=args.dry_run,
                session_factory=shared._aws_session,
                instance_type=CPU_INSTANCE,
                command="train-head",
                features=features,
            )
        except (
            feature_cache.CacheError,
            records.LeakError,
            run.ProvenanceError,
            cost.CostRefused,
            ValueError,
        ) as error:
            raise SystemExit(f"refusing to launch: {error}") from error
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "features": {"cache": features.cache, "files": len(files)},
                    "estimate": asdict(plan.estimate),
                    "request": plan.request,
                },
                indent=2,
            )
        )
