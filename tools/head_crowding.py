"""Count test people the detection head's decoder cannot all find.

The head keeps one peak per sub-cell and decodes with a 3 x 3 peak test
(docs/detection-head-design.md). For each camera and fold this counts, over
the fold's test manifest, the people who share a sub-cell with another
(dropped at target assignment), those whose sub-cell touches another
person's, and the fewest of those a 3 x 3 peak test must lose
(`head_experiment.crowding`). Reads only the manifests and the cache index.

    uv run python tools/head_crowding.py \
        outputs/features/siglip2-base-naflex-1024tok data/manifests/wisard-full

Takes a few seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerial_search.data.folds import TEST, view_dir
from frozen_head.feature_cache import read_index
from frozen_head.head_experiment import (
    Geometry,
    Recipe,
    crowding,
    record_boxes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", type=Path)
    parser.add_argument("manifests", type=Path)
    parser.add_argument("--upsample", type=int, default=Recipe().upsample)
    args = parser.parse_args()

    index = read_index(args.cache)
    folds = sorted(p.name for p in (args.manifests / "folds").iterdir() if p.is_dir())
    print("| camera | fold | people | share a sub-cell | adjacent | unreachable |")
    print("|---|---|---|---|---|---|")
    for camera in ("rgb", "thermal"):
        total = dict.fromkeys(("people", "shared", "adjacent", "unreachable"), 0)
        for fold in folds:
            path = view_dir(args.manifests / "folds", fold, camera) / TEST
            counts = dict.fromkeys(total, 0)
            for line in path.read_text().splitlines():
                record = json.loads(line)
                geometry = Geometry.from_entry(index[record["image"]], args.upsample)
                boxes = record_boxes(record)
                c = crowding(geometry, boxes)
                counts["people"] += len(boxes)
                counts["shared"] += c.shared
                counts["adjacent"] += c.adjacent
                counts["unreachable"] += c.unreachable
            for key in total:
                total[key] += counts[key]
            print(_row(camera, fold, counts))
        print(_row(camera, "all", total))


def _row(camera: str, fold: str, counts: dict[str, int]) -> str:
    people = counts["people"]

    def share(key: str) -> str:
        return f"{counts[key]:,} ({100 * counts[key] / people:.1f}%)"

    cells = [share(k) for k in ("shared", "adjacent", "unreachable")]
    return f"| {camera} | {fold} | {people:,} | " + " | ".join(cells) + " |"


if __name__ == "__main__":
    main()
