"""Draw the off-the-shelf bar figure (#69): head versus fine-tuned detector.

Reads completed, non-scratch runs with `aerial_search.evaluation.head_table`
(the same readers as `aerial-search compare-table`), refuses the pair unless
both methods saw the same frames, and writes one hand-made SVG with a panel
per camera: test `ap_iou25` against label fraction, mean over folds as a line,
each fold as a faint dot, and the mean `ap_iou50` dashed beside it. Prints the
plotted numbers so the figure can be checked against `compare-table`.

    uv run python tools/bar_figure.py \
        outputs/detection-head/siglip2-base-naflex-1024tok \
        outputs/detection-baseline/fasterrcnn_mobilenet_v3_large_fpn-cache

Writes docs/figures/off-the-shelf-bar.svg by default (`--output`).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from aerial_search.evaluation import head_table

SERIES = (
    ("head", "frozen SigLIP 2 + head", "#3b6fd8"),
    ("baseline", "fine-tuned detector (same frames)", "#e07020"),
)
CAMERAS = (("rgb", "RGB"), ("thermal", "Thermal"))
PERCENTS = ("1", "5", "10", "100")
TEXT = "#6e7681"  # readable on white and on #0d1117
GRID = "#8b949e"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
WIDTH, HEIGHT = 900, 380
PANELS = ((65, 430), (505, 870))  # x range of each panel's plot area
TOP, BOTTOM = 96, 290


def nice_ceiling(value: float) -> float:
    """The smallest multiple of 0.05 at or above `value`."""
    return max(0.05, math.ceil(value / 0.05 - 1e-9) * 0.05)


def _num(x: float) -> str:
    return f"{x:.1f}"


def render(summaries: dict[str, dict[str, Any]], commits: dict[str, list[str]]) -> str:
    top = nice_ceiling(
        max(
            f["ap_iou25"]
            for s in summaries.values()
            for cam in s.values()
            for entry in cam.values()
            for f in entry["folds"].values()
        )
    )
    ticks = [round(i * 0.05, 2) for i in range(round(top / 0.05) + 1)]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}" fill="{TEXT}" '
        'role="img" aria-labelledby="t">',
        '<title id="t">Mean test AP over four held-out site-days: frozen SigLIP 2 '
        "head against a fine-tuned detector, RGB and thermal</title>",
        f'<text x="{WIDTH / 2}" y="26" font-size="17" font-weight="600" '
        'text-anchor="middle">Mean test AP over four held-out site-days</text>',
    ]
    x = 60.0
    for _, label, colour in SERIES:
        out.append(
            f'<line x1="{x}" y1="52" x2="{x + 26}" y2="52" stroke="{colour}" '
            'stroke-width="2.5"/>'
            f'<circle cx="{x + 13}" cy="52" r="4" fill="{colour}"/>'
            f'<text x="{x + 34}" y="56" font-size="13">{label}</text>'
        )
        x += 34 + 7.0 * len(label) + 24
    out.append(
        f'<line x1="{x}" y1="52" x2="{x + 26}" y2="52" stroke="{TEXT}" '
        'stroke-width="2.5"/>'
        f'<text x="{x + 32}" y="56" font-size="13">ap_iou25</text>'
        f'<line x1="{x + 100}" y1="52" x2="{x + 126}" y2="52" stroke="{TEXT}" '
        'stroke-width="2" stroke-dasharray="5 4"/>'
        f'<text x="{x + 132}" y="56" font-size="13">ap_iou50</text>'
    )
    for (camera, title), (x0, x1) in zip(CAMERAS, PANELS, strict=True):
        step = (x1 - x0) / len(PERCENTS)

        def px(i: int, x0: float = x0, step: float = step) -> float:
            return x0 + step * (i + 0.5)

        def py(v: float) -> float:
            return BOTTOM - (BOTTOM - TOP) * v / top

        out.append(
            f'<text x="{(x0 + x1) / 2}" y="82" font-size="14" font-weight="600" '
            f'text-anchor="middle">{title}</text>'
        )
        for t in ticks:
            out.append(
                f'<line x1="{x0}" y1="{_num(py(t))}" x2="{x1}" y2="{_num(py(t))}" '
                f'stroke="{GRID}" stroke-opacity="0.3"/>'
                f'<text x="{x0 - 8}" y="{_num(py(t) + 4)}" font-size="11" '
                f'text-anchor="end">{t:.2f}</text>'
            )
        out.append(
            f'<line x1="{x0}" y1="{BOTTOM}" x2="{x1}" y2="{BOTTOM}" stroke="{GRID}"/>'
        )
        for i, p in enumerate(PERCENTS):
            out.append(
                f'<text x="{_num(px(i))}" y="{BOTTOM + 18}" font-size="12" '
                f'text-anchor="middle">{p}%</text>'
            )
        out.append(
            f'<text x="{(x0 + x1) / 2}" y="{BOTTOM + 38}" font-size="12" '
            'text-anchor="middle">label fraction</text>'
        )
        if camera == "rgb":
            out.append(
                f'<text transform="translate(14 {(TOP + BOTTOM) / 2}) rotate(-90)" '
                'font-size="12" text-anchor="middle">test AP (IoU 0.25 solid, '
                "0.50 dashed)</text>"
            )
        for key, _, colour in SERIES:
            cells = [summaries[key][camera][p] for p in PERCENTS]
            for i, e in enumerate(cells):
                for fold in e["folds"].values():
                    out.append(
                        f'<circle cx="{_num(px(i))}" cy="{_num(py(fold["ap_iou25"]))}" '
                        f'r="2.5" fill="{colour}" fill-opacity="0.35"/>'
                    )
            for metric, dash in (
                ("ap_iou50", ' stroke-dasharray="5 4"'),
                ("ap_iou25", ""),
            ):
                pts = " ".join(
                    f"{_num(px(i))},{_num(py(e['mean'][metric]))}"
                    for i, e in enumerate(cells)
                )
                width = 2.5 if metric == "ap_iou25" else 1.8
                out.append(
                    f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                    f'stroke-width="{width}"{dash}/>'
                )
            for i, e in enumerate(cells):
                out.append(
                    f'<circle cx="{_num(px(i))}" '
                    f'cy="{_num(py(e["mean"]["ap_iou25"]))}" '
                    f'r="4.5" fill="{colour}"/>'
                )
    shown = "; ".join(
        f"{short} {', '.join(commits[k])}"
        for k, short in (("head", "head"), ("baseline", "detector"))
    )
    out.append(
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 14}" font-size="11" text-anchor="middle">'
        f"Faint dots: the four site-days. Commits read: {shown}.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = ["camera   labels  method    ap_iou25  ap_iou50  per-fold ap_iou25"]
    for camera, _ in CAMERAS:
        for p in PERCENTS:
            for key, _, _ in SERIES:
                e = summaries[key][camera][p]
                folds = " ".join(f"{f['ap_iou25']:.3f}" for f in e["folds"].values())
                lines.append(
                    f"{camera:<8} {p + '%':<7} {key:<9} {e['mean']['ap_iou25']:.3f}"
                    f"     {e['mean']['ap_iou50']:.3f}     {folds}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("head", type=Path, help="detection-head runs directory")
    parser.add_argument("baseline", type=Path, help="detection-baseline runs directory")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/off-the-shelf-bar.svg")
    )
    args = parser.parse_args()
    try:
        runs = {
            "head": head_table.collect(args.head),
            "baseline": head_table.collect(args.baseline),
        }
        head_table.check_same_frames(runs)
        summaries = {k: head_table.summarise(v) for k, v in runs.items()}
    except head_table.TableError as error:
        raise SystemExit(f"refusing to draw: {error}") from error
    commits = {k: sorted({str(r["commit"])[:7] for r in v}) for k, v in runs.items()}
    args.output.write_text(render(summaries, commits))
    print(table(summaries))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
