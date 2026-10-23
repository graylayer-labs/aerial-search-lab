"""The frozen head's commands, run through `aerial-search` (#160)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aerial_search import cli as shared

HELP = shared.PROJECT_COMMANDS


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
