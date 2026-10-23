"""The fine-tuned detector's commands: train-baseline and sagemaker-train.

One definition of train-baseline's flags serves both: sagemaker-train takes
the same flags, less the paths the job sets, and forwards them to the job,
which runs train-baseline with them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aerial_search import cli as shared
from finetuned_detector import experiment as baseline_experiment

HELP = shared.PROJECT_COMMANDS

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


def baseline_args(args: argparse.Namespace) -> list[str]:
    """The `train-baseline` arguments the job runs, from `sagemaker-train`'s."""
    out = [args.camera]
    for name in BASELINE_FLAG_NAMES:
        if getattr(args, name) is not None:  # an unset flag keeps its default
            out += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return [*out, "--scratch"] if args.scratch else out


def add_baseline_flags(base: argparse.ArgumentParser, *, cloud: bool) -> None:
    """The `train-baseline` flags; `cloud` leaves out the paths the job sets."""
    base.add_argument("camera", choices=["rgb", "thermal"])
    base.add_argument("--percent", type=int, choices=shared.PERCENTS, required=True)
    base.add_argument("--seed", type=int, default=shared.SEED)
    if not cloud:
        base.add_argument(
            "--data-root", type=Path, default=Path("data/raw/wisard-full")
        )
    base.add_argument(
        "--manifests", type=Path, default=Path("data/manifests/wisard-full")
    )
    from finetuned_detector.experiment import (
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
    from finetuned_detector.experiment import CHECKPOINT_EVERY

    base.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="training steps between checkpoints (also one when training ends)",
    )
    shared._add_fold_flag(base)
    shared._add_run_flags(base)


def build_parser() -> argparse.ArgumentParser:
    """The parser of this project's commands, as `aerial-search` runs them."""
    parser = argparse.ArgumentParser(prog="aerial-search")
    commands = parser.add_subparsers(dest="command", required=True)
    base = commands.add_parser("train-baseline", help=HELP["train-baseline"].help)
    add_baseline_flags(base, cloud=False)

    cloud = commands.add_parser("sagemaker-train", help=HELP["sagemaker-train"].help)
    add_baseline_flags(cloud, cloud=True)
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
    cloud.add_argument("--max-runtime-hours", type=float, default=12.0)
    cloud.add_argument(
        "--max-wait-hours",
        type=float,
        default=18.0,
        help="spot only: runtime plus time spent waiting for capacity",
    )
    cloud.add_argument(
        "--expected-hours",
        type=shared._positive_hours,
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
    return parser


def main(argv: list[str]) -> None:
    """Run `train-baseline` or `sagemaker-train`; `argv` starts with the name."""
    args = build_parser().parse_args(argv)
    if args.command == "train-baseline":
        import torch

        from aerial_search import run

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

        recipe = baseline_experiment.BaselineRecipe(
            arch=args.arch, input=args.input, tile_overlap=args.tile_overlap
        )
        logical = args.run_name or baseline_experiment.default_run_name(
            recipe, args.camera, args.fold, args.percent, args.seed
        )
        forwarded = baseline_args(args)
        spot_price = args.spot_price_usd or cost.SPOT_USD_PER_HOUR
        try:
            plan = launcher.launch(
                repo=None,
                logical_run=logical,
                baseline_args=forwarded,
                manifests=args.manifests,
                scratch=args.scratch,
                attempt=args.attempt,
                spot=True,
                max_runtime_hours=args.max_runtime_hours,
                max_wait_hours=args.max_wait_hours,
                expected_hours=cost.default_expected_hours(args.input, args.camera)
                if args.expected_hours is None
                else args.expected_hours,
                eur_per_usd=args.eur_per_usd or cost.EUR_PER_USD,
                month_to_date_eur=args.month_to_date_eur,
                role_arn=args.role_arn,
                image_uri=args.image_uri,
                dry_run=args.dry_run,
                session_factory=shared._aws_session,
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
