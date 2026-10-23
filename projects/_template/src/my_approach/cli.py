"""This approach's commands, run through `aerial-search`.

Register each command in `aerial_search.cli.PROJECT_COMMANDS`, for example
`"train-my-approach": ProjectCommand("my_approach.cli", "<one line>")`. The
shared CLI then hands the whole command line, name first, to `main`.
"""

from __future__ import annotations

import argparse

from aerial_search import cli as shared


def build_parser() -> argparse.ArgumentParser:
    """The parser of this project's commands, as `aerial-search` runs them."""
    parser = argparse.ArgumentParser(prog="aerial-search")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-my-approach", help="<one line>")
    train.add_argument("camera", choices=["rgb", "thermal"])
    train.add_argument("--percent", type=int, choices=shared.PERCENTS, required=True)
    train.add_argument("--seed", type=int, default=shared.SEED)
    shared._add_fold_flag(train)  # one site-day is the test set
    shared._add_run_flags(train)  # --scratch and --run-name, as every run has
    return parser


def main(argv: list[str]) -> None:
    """Run one of this project's commands; `argv` starts with its name."""
    args = build_parser().parse_args(argv)
    if args.command == "train-my-approach":
        raise SystemExit("train-my-approach is not written yet")
