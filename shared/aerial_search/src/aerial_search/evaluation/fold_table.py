"""Markdown tables of what each fold holds, from a folds.json summary."""

from __future__ import annotations

from aerial_search.evaluation.detection import SIZE_BUCKETS


def fold_table(summary: dict) -> str:
    """Markdown tables of what each fold holds, from a folds.json summary."""
    per_fold = summary["per_fold"]
    lines = []
    for view in summary["views"]:
        cameras = ["rgb", "thermal"] if view == "paired" else [view]
        boxes = " | ".join(
            f"{c.upper() if c == 'rgb' else c.title()} boxes" for c in cameras
        )
        lines += [
            f"### View: {view}",
            "",
            f"| Fold (test site-day) | Split | Site-days | Clips | Frames | {boxes} |",
            "|---|---|---|---|---|" + "---|" * len(cameras),
        ]
        for fold in summary["folds"]:
            parts = per_fold[fold]["views"][view]
            for split in ("train_100pct", "validation", "test"):
                part = parts[split]
                counts = " | ".join(f"{part[f'{c}_boxes']:,}" for c in cameras)
                lines.append(
                    f"| {_short(fold)} | {split.removesuffix('_100pct')} | "
                    f"{', '.join(_short(s) for s in part['site_days'])} | "
                    f"{part['clips']} | {part['frames']:,} | {counts} |"
                )
        lines += ["", "Validation source per training site-day:", ""]
        for fold in summary["folds"]:
            sources = per_fold[fold]["views"][view]["validation_sources"]
            described = "; ".join(
                f"{_short(site)}: {_source(source)}" for site, source in sources.items()
            )
            lines.append(f"- fold {_short(fold)}: {described}")
        buckets = list(SIZE_BUCKETS)
        lines += [
            "",
            "Test boxes by size (side of the box in pixels, "
            + ", ".join(f"{b} [{lo:g}, {hi:g})" for b, (lo, hi) in SIZE_BUCKETS.items())
            + "):",
            "",
            "| Fold | Camera | " + " | ".join(buckets) + " |",
            "|---|---|" + "---|" * len(buckets),
        ]
        for fold in summary["folds"]:
            test = per_fold[fold]["views"][view]["test"]
            for camera in cameras:
                counts = test[f"{camera}_size_buckets"]
                lines.append(
                    f"| {_short(fold)} | {camera} | "
                    + " | ".join(f"{counts[b]:,}" for b in buckets)
                    + " |"
                )
        percents = summary["percents"]
        lines += [
            "",
            "Label fractions, frames / "
            + " + ".join(f"{c} boxes" for c in cameras)
            + " / site-days:",
            "",
            "| Fold | " + " | ".join(f"{p}%" for p in percents) + " |",
            "|---|" + "---|" * len(percents),
        ]
        for fold in summary["folds"]:
            cells = []
            for percent in percents:
                part = per_fold[fold]["views"][view][f"train_{percent}pct"]
                box_counts = " + ".join(f"{part[f'{c}_boxes']:,}" for c in cameras)
                cells.append(
                    f"{part['frames']:,} / {box_counts} / {len(part['site_days'])}"
                )
            lines.append(f"| {_short(fold)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "### Unlabelled pool",
        "",
        "| Fold | Site-days | Clips | Pairs |",
        "|---|---|---|---|",
    ]
    for fold in summary["folds"]:
        pool = per_fold[fold]["unlabelled"]
        lines.append(
            f"| {_short(fold)} | {', '.join(_short(s) for s in pool['site_days'])} | "
            f"{pool['clips']} | {pool['frames']:,} |"
        )
    return "\n".join(lines) + "\n"


def _short(site: str) -> str:
    return site.split("_", 1)[1]


def _source(source: dict) -> str:
    clip = str(source["clip"]).rsplit("_", 1)[1]
    if source["kind"] == "whole clip":
        return f"whole clip {clip}"
    return (
        f"clip {clip} from frame {source['validation_from_frame']}, "
        f"{source['gap_frames']} frames dropped before it"
    )
