"""The results table of the detection-head runs (#66).

Reads every `run.json` under a directory, refuses any that is scratch or not
completed, and reports per camera and label fraction: each fold's test score,
and the mean and sample standard deviation over folds. `ap_iou50` is always
shown beside `ap_iou25` (docs/evaluation-metric-review.md).
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aerial_search.evaluation.detection import METRICS_FILENAME, SIZE_BUCKETS

FPPI = ("0.1", "1.0")
METRICS = ("ap_iou25", "ap_iou50", *(f"recall_at_{f}fppi" for f in FPPI))


SELECTION_CAVEAT = (
    "Every run picks its best step on the fold's full labelled validation "
    'split, whatever its label fraction: "1% of labels" means 1% of the '
    "training labels plus all validation labels (RGB, test site-day MtErie: "
    "70 training frames at 1%, 1,339 validation frames)."
)


def compare_caveat(runs_by_method: dict[str, list[dict[str, Any]]]) -> str:
    """SELECTION_CAVEAT, plus how many validation frames each method that
    subsamples them (a recipe with `val_frames`) picked its step on."""
    notes = []
    for method, runs in runs_by_method.items():
        counts = sorted(
            {r["recipe"]["val_frames"] for r in runs if "val_frames" in r["recipe"]}
        )
        if counts:
            shown = ", ".join("all" if c == 0 else str(c) for c in counts)
            notes.append(
                f"{method} picked its step on {shown} evenly spaced validation frames"
            )
    if not notes:
        return SELECTION_CAVEAT
    return SELECTION_CAVEAT + " But " + "; ".join(notes) + " (the head uses all)."


class TableError(Exception):
    """A run cannot be quoted, or two runs claim the same cell."""


def _scores(metrics: dict[str, Any]) -> dict[str, float]:
    def value(v: Any) -> float:
        return math.nan if v is None else float(v)

    out = {
        "ap_iou25": value(metrics["ap_iou25"]),
        "ap_iou50": value(metrics["ap_iou50"]),
    }
    for f in FPPI:
        out[f"recall_at_{f}fppi"] = value(metrics["recall_at_fppi"].get(f))
    return out


def collect(root: Path, *, ablation: bool = False) -> list[dict[str, Any]]:
    """Every run under `root`, with its test scores. Raises `TableError` on a
    scratch or unfinished run, or a completed run without its report. Outside
    the ablation (`ablation=False`, the plain table) a run that is warm-started
    or not the replicate arm is refused: its numbers are not a #66-style cold
    result."""
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("run.json")):
        record = json.loads(path.read_text())
        name = str(path.parent.relative_to(root))
        if record.get("scratch") is not False:
            raise TableError(f"{name} is a scratch run; it cannot be quoted")
        if record.get("status") != "completed":
            raise TableError(
                f"{name} has status {record.get('status')!r}, not completed"
            )
        report = path.parent / METRICS_FILENAME
        if not report.exists():
            raise TableError(f"{name} is completed but has no {METRICS_FILENAME}")
        metrics = json.loads(report.read_text())
        config = record["config"]
        arm = config.get("input_handling", "replicate")
        if not ablation and (arm != "replicate" or config.get("warm_start")):
            raise TableError(
                f"{name} is a {arm!r} run"
                f"{' started from a trained head' if config.get('warm_start') else ''}"
                "; tabulate it with --ablation, not as a cold table"
            )
        runs.append(
            {
                "name": name,
                "camera": config["camera"],
                "fold": config["fold"],
                "percent": str(config["percent"]),
                # #66 runs predate the field and are the replicate arm; a run
                # started from a trained head is its own arm, "<arm>-warm"
                "arm": config.get("input_handling", "replicate")
                + ("-warm" if config.get("warm_start") else ""),
                "commit": record.get("commit"),
                "recipe": config.get("recipe", {}),
                "made_with": _made_with(record),
                "manifests": _manifests(record),
                "overall": _scores(metrics["overall"]),
                "by_size": {k: _scores(m) for k, m in metrics["by_size"].items()},
            }
        )
    for run in runs[1:]:
        first = runs[0]
        for field, value in run["made_with"].items():
            if value != first["made_with"].get(field):
                raise TableError(
                    f"{field} differs: {first['made_with'].get(field)!r} in "
                    f"{first['name']}, {value!r} in {run['name']}; one table "
                    "holds runs made the same way"
                )
    return runs


def _manifests(record: dict[str, Any]) -> dict[str, str]:
    """SHA-256 of each manifest a run read, keyed by `<fold>/<camera>/<file>`."""
    return {
        "/".join(i["path"].split("/")[-3:]): i["sha256"]
        for i in record.get("inputs", [])
        if i["path"].endswith(".jsonl")
    }


SPLIT_FILES = ("validation.jsonl", "test.jsonl")


def check_same_frames(runs_by_method: dict[str, list[dict[str, Any]]]) -> None:
    """Refuse a comparison unless, in every cell two methods share, both read
    the train, validation and test manifests with identical bytes (so the same
    frames, labels included), and unless every method has every cell."""
    cells: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for method, runs in runs_by_method.items():
        cells[method] = {(r["camera"], r["fold"], r["percent"]): r for r in runs}
    every = set().union(*(set(c) for c in cells.values())) if cells else set()
    for method, own in cells.items():
        missing = sorted(every - set(own))
        if missing:
            listed = ", ".join("/".join(c) for c in missing)
            raise TableError(f"{method} has no run for: {listed}")
    methods = list(cells)
    for cell in sorted(every):
        first = cells[methods[0]][cell]
        for method in methods[1:]:
            run = cells[method][cell]
            wanted = [
                f"{cell[1]}/{cell[0]}/{f}"
                for f in (f"train_{cell[2]}pct.jsonl", *SPLIT_FILES)
            ]
            for name in wanted:
                a, b = first["manifests"].get(name), run["manifests"].get(name)
                if a is None or b is None or a != b:
                    raise TableError(
                        f"{name} is {'missing' if None in (a, b) else 'different'}"
                        f" between {methods[0]} ({first['name']}) and {method} "
                        f"({run['name']}): they did not see the same frames"
                    )


def _made_with(record: dict[str, Any]) -> dict[str, Any]:
    """What must be equal across the runs of one table: every recipe field
    (steps, batch, learning rate, head size...), the seed, and the cache,
    by the SHA-256 of its cache.json."""
    recipe = record["config"].get("recipe", {})
    # the cache's settings but for the input handling, which is the arm itself:
    # an equalised cache made with other pooling, budget or weights shows up
    settings = {
        k: v
        for k, v in record["config"].get("cache", {}).get("settings", {}).items()
        if k != "input_handling"
    }
    cache = [
        i["sha256"]
        for i in record.get("inputs", [])
        if i["path"].endswith("/cache.json")
    ]
    return {
        **{f"recipe.{k}": v for k, v in sorted(recipe.items())},
        "seed": record.get("seed"),
        "cache settings": settings,
        "cache.json sha256": cache[0] if cache else None,
    }


def _spread(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    mean = float(np.mean(finite)) if finite else math.nan
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
    return mean, std


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """camera -> percent -> folds, mean, std, n_folds (and the same by size)."""
    cells: dict[tuple[str, str], dict[str, dict]] = {}
    for run in runs:
        folds = cells.setdefault((run["camera"], run["percent"]), {})
        if run["fold"] in folds:
            raise TableError(
                f"{run['camera']} fold {run['fold']} at {run['percent']}% appears "
                f"twice ({folds[run['fold']]['name']} and {run['name']})"
            )
        folds[run["fold"]] = run
    commits = sorted({str(r["commit"]) for r in runs})
    if len(commits) > 1:
        warnings.warn(
            f"runs come from {len(commits)} commits ({', '.join(commits)}); "
            "check that nothing that changes a result differs between them",
            stacklevel=2,
        )
    summary: dict[str, Any] = {}
    for (camera, percent), folds in sorted(cells.items()):
        entry: dict[str, Any] = {
            "folds": {f: r["overall"] for f, r in sorted(folds.items())},
            "n_folds": len(folds),
            "commits": sorted({str(r["commit"]) for r in folds.values()}),
            "fold_commits": {f: r["commit"] for f, r in sorted(folds.items())},
            "mean": {},
            "std": {},
            "by_size": {},
        }
        for metric in METRICS:
            entry["mean"][metric], entry["std"][metric] = _spread(
                [r["overall"][metric] for r in folds.values()]
            )
        for bucket in SIZE_BUCKETS:
            per_fold = [
                r["by_size"][bucket] for r in folds.values() if bucket in r["by_size"]
            ]
            # a fold with no people of this size has no score and is left out
            entry["by_size"][bucket] = {
                **{
                    metric: _spread([s[metric] for s in per_fold])
                    for metric in ("ap_iou25", "ap_iou50")
                },
                "n": sum(not math.isnan(s["ap_iou25"]) for s in per_fold),
            }
        summary.setdefault(camera, {})[percent] = entry
    return summary


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _pm(mean: float, std: float) -> str:
    return f"{_fmt(mean)} ± {_fmt(std)}"


def markdown(summary: dict[str, Any]) -> str:
    """One table per camera, then one of mean ap by person size."""
    lines: list[str] = []
    for camera, by_percent in summary.items():
        percents = sorted(by_percent, key=int)
        folds = sorted({f for e in by_percent.values() for f in e["folds"]})
        lines += [
            f"### {camera}: test ap_iou25 / ap_iou50 per fold, mean ± sd over "
            "folds (labels = share of training labels; the step is chosen on "
            "the full validation split)",
            "",
            "| labels | "
            + " | ".join(folds)
            + " | ap_iou25 | ap_iou50 | recall @0.1 FPPI | recall @1 FPPI | folds |",
            "|---" * (len(folds) + 6) + "|",
        ]
        for percent in percents:
            e = by_percent[percent]
            cells = [
                f"{_fmt(e['folds'][f]['ap_iou25'])} / {_fmt(e['folds'][f]['ap_iou50'])}"
                if f in e["folds"]
                else "-"
                for f in folds
            ]
            mean, std = e["mean"], e["std"]
            stats = [_pm(mean[m], std[m]) for m in METRICS]
            lines.append(
                f"| {percent}% | "
                + " | ".join([*cells, *stats, str(e["n_folds"])])
                + " |"
            )
        lines += [
            "",
            SELECTION_CAVEAT,
            "",
            f"### {camera}: test ap_iou25 / ap_iou50 by person size, mean over "
            "the n folds with people of that size",
            "",
            "| labels | " + " | ".join(SIZE_BUCKETS) + " |",
            "|---" * (len(SIZE_BUCKETS) + 1) + "|",
        ]
        for percent in percents:
            sizes = by_percent[percent]["by_size"]
            cells = [
                f"{_fmt(sizes[b]['ap_iou25'][0])} / {_fmt(sizes[b]['ap_iou50'][0])}"
                f" (n={sizes[b]['n']})"
                for b in SIZE_BUCKETS
            ]
            lines.append(f"| {percent}% | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# --- the thermal input ablation (#68) -----------------------------------------

CONTROL = "replicate"
WARM_CONTROL = "replicate-warm"
CONTROLS = (CONTROL, WARM_CONTROL)
ARM_ORDER = ("replicate", "equalise", "stem", WARM_CONTROL, "stem-warm")


def control_of(arm: str) -> str:
    """The arm to compare `arm` with: warm arms against the warm control (the
    same trained head, the same extra steps), cold arms against the cold one."""
    return WARM_CONTROL if arm.endswith("-warm") else CONTROL


def collect_arms(
    roots: Sequence[Path],
    *,
    camera: str | None = None,
    percents: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Runs by input arm, from one directory per arm.

    Each directory goes through `collect` (so scratch or unfinished runs and
    runs made differently within a directory are refused). A directory holds
    one arm and an arm comes from one directory. All arms must share the
    seed: that is the one thing that must not differ. Other differences (a
    shorter recipe for the stem arm, say) are allowed but listed by
    `arm_notes` under the table.
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for root in roots:
        runs = [
            r
            for r in collect(root, ablation=True)
            if (camera is None or r["camera"] == camera)
            and (percents is None or r["percent"] in percents)
        ]
        arms = {r["arm"] for r in runs}
        if len(arms) > 1:
            raise TableError(
                f"{root} holds several arms ({sorted(arms)}); one arm per directory"
            )
        for arm in arms:
            if arm in by_arm:
                raise TableError(f"arm {arm!r} appears twice (second in {root})")
            by_arm[arm] = runs
    seeds = {
        arm: {r["made_with"]["seed"] for r in runs} for arm, runs in by_arm.items()
    }
    if len({seed for v in seeds.values() for seed in v}) > 1:
        raise TableError(f"seed differs between arms: {seeds}")
    return by_arm


def arm_notes(by_arm: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Every way an arm's recipe or cache differs from its control's, one line each."""
    notes: list[str] = []
    for arm in ARM_ORDER:
        control = control_of(arm)
        if arm == control or arm not in by_arm or control not in by_arm:
            continue
        base = by_arm[control][0]["made_with"]
        made = by_arm[arm][0]["made_with"]
        for field in sorted(set(base) | set(made)):
            if field == "cache.json sha256":
                continue  # differs by design; the settings are compared instead
            if base.get(field) != made.get(field):
                notes.append(
                    f"{arm}: {field} is {made.get(field)!r}, {control} has "
                    f"{base.get(field)!r}"
                )
    return notes


def summarise_arms(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """camera -> percent -> arm -> `summarise` entry, plus for each non-control
    arm `delta`: mean, sample sd and count of the per-fold change of each
    metric against the control, over the folds both have."""
    if not any(c in by_arm for c in CONTROLS):
        raise TableError(f"no {CONTROL!r} arm: the ablation needs its control")
    commits = sorted({str(r["commit"]) for runs in by_arm.values() for r in runs})
    if len(commits) > 1:
        warnings.warn(
            f"the arms come from {len(commits)} commits ({', '.join(commits)}); "
            "check that nothing that changes a result differs between them",
            stacklevel=2,
        )
    per_arm = {arm: summarise(runs) for arm, runs in by_arm.items()}
    out: dict[str, Any] = {}
    for arm in (a for a in ARM_ORDER if a in per_arm):
        for camera, by_percent in per_arm[arm].items():
            for percent, entry in by_percent.items():
                out.setdefault(camera, {}).setdefault(percent, {})[arm] = entry
    for by_percent in out.values():
        for arms in by_percent.values():
            for arm, entry in arms.items():
                control = arms.get(control_of(arm))
                if arm == control_of(arm) or control is None:
                    continue
                common = sorted(set(entry["folds"]) & set(control["folds"]))
                entry["delta"] = {
                    metric: (
                        *_spread(
                            [
                                entry["folds"][f][metric] - control["folds"][f][metric]
                                for f in common
                            ]
                        ),
                        len(common),
                    )
                    for metric in ("ap_iou25", "ap_iou50")
                }
    return out


def markdown_arms(summary: dict[str, Any], notes: Sequence[str] = ()) -> str:
    """One table per camera and label fraction: a row per arm, each fold's
    ap_iou25 / ap_iou50, the mean and sample sd over folds, and the mean
    per-fold change of ap_iou25 against the control."""
    lines: list[str] = []
    for camera, by_percent in summary.items():
        for percent in sorted(by_percent, key=int):
            arms = by_percent[percent]
            folds = sorted({f for e in arms.values() for f in e["folds"]})
            lines += [
                f"### {camera}, {percent}% of labels: test ap_iou25 / ap_iou50 "
                "per fold, mean ± sd over folds, and the mean per-fold change "
                f"of ap_iou25 against the control (replicate; replicate-warm for the "
                "warm arms)",
                "",
                "| arm | "
                + " | ".join(folds)
                + " | ap_iou25 | ap_iou50 | change in ap_iou25 | folds |",
                "|---" * (len(folds) + 5) + "|",
            ]
            for arm, e in arms.items():
                cells = [
                    f"{_fmt(e['folds'][f]['ap_iou25'])} / "
                    f"{_fmt(e['folds'][f]['ap_iou50'])}"
                    if f in e["folds"]
                    else "-"
                    for f in folds
                ]
                delta = e.get("delta")
                change = (
                    "-"
                    if delta is None
                    else f"{delta['ap_iou25'][0]:+.3f} ± {_fmt(delta['ap_iou25'][1])}"
                )
                stats = [
                    _pm(e["mean"]["ap_iou25"], e["std"]["ap_iou25"]),
                    _pm(e["mean"]["ap_iou50"], e["std"]["ap_iou50"]),
                    change,
                    str(e["n_folds"]),
                ]
                lines.append(f"| {arm} | " + " | ".join([*cells, *stats]) + " |")
            lines.append("")
    lines += [SELECTION_CAVEAT, ""]
    if notes:
        lines += ["The arms' recipes differ:", "", *(f"- {n}" for n in notes), ""]
    return "\n".join(lines)


def markdown_compare(
    summaries: dict[str, dict[str, Any]], caveat: str = SELECTION_CAVEAT
) -> str:
    """One table per camera: a row per label fraction and method, a column per
    fold, then mean and sd over folds. `summaries` maps a method name to its
    `summarise` output; methods are listed in the order given."""
    cameras = sorted({c for s in summaries.values() for c in s})
    lines: list[str] = []
    for camera in cameras:
        by_method = {m: s[camera] for m, s in summaries.items() if camera in s}
        percents = sorted({p for e in by_method.values() for p in e}, key=int)
        folds = sorted(
            {f for e in by_method.values() for x in e.values() for f in x["folds"]}
        )
        lines += [
            f"### {camera}: test ap_iou25 / ap_iou50 per fold, "
            "mean ± sd over folds, by method",
            "",
            "| labels | method | "
            + " | ".join(folds)
            + " | ap_iou25 | ap_iou50 | recall @0.1 FPPI | recall @1 FPPI | folds |",
            "|---" * (len(folds) + 7) + "|",
        ]
        for percent in percents:
            for method, e_by_percent in by_method.items():
                e = e_by_percent.get(percent)
                if e is None:
                    continue
                cells = [
                    " / ".join(_fmt(e["folds"][f][m]) for m in METRICS[:2])
                    if f in e["folds"]
                    else "-"
                    for f in folds
                ]
                stats = [_pm(e["mean"][m], e["std"][m]) for m in METRICS]
                row = [*cells, *stats, str(e["n_folds"])]
                lines.append(f"| {percent}% | {method} | " + " | ".join(row) + " |")
        lines += ["", caveat, ""]
    return "\n".join(lines)
