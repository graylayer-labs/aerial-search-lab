"""The fine-tuned detector's commands: train-baseline and sagemaker-train.

One definition of train-baseline's flags serves both: sagemaker-train takes
the same flags, less the paths the job sets, and forwards them to the job,
which runs train-baseline with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerial_search import cli as shared

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
