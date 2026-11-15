"""What a whole head run writes, pinned to a stored fixture (#216).

A refactor of the run wiring, the training loop or the evaluation must leave
`run.json`, `training.json`, the test report and the saved head unchanged for
the same inputs and seed. The stem arm's run is pinned in
`test_stem_experiment.py`, beside its tiny tower.
"""

import json
from pathlib import Path

import numpy as np
import torch
from frozen_head import head_experiment as he

FOLD, OTHER = "220109_Baker", "210417_MtErie"
ENTRY = {
    "camera": "thermal",
    "image_size": [640, 512],
    "resized_size": [560, 448],
    "patch_grid": [28, 35],
    "pooling": 2,
    "grid": [14, 18],
    "dim": 8,
}


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def setup(tmp_path: Path) -> Path:
    """Fold manifests and a scratch cache of random features.

    Validation and test frames repeat training frames (features and people)
    under other names, so the head can learn them and the validation score
    moves between steps.
    """
    rng = np.random.default_rng(5)

    def frames(site: str, n: int) -> list[dict]:
        out = []
        for k in range(n):
            # rounded: the last bit of a uniform draw differs between platforms
            x, y = (round(float(v), 6) for v in rng.uniform(0.1, 0.9, 2))
            people = [{"x_center": x, "y_center": y, "width": 0.1, "height": 0.12}]
            image = f"{site}_Enterprise_IR_1/{k}.jpg"
            out.append(
                {
                    "site_day": site,
                    "collection_id": f"{site}_Enterprise_IR_1",
                    "camera": "thermal",
                    "image": image,
                    "size": [640, 512],
                    "boxes": people if k % 3 else [],
                }
            )
        return out

    train = frames(OTHER, 6)

    def again(r: dict, site: str, name: str) -> dict:
        collection = f"{site}_Enterprise_IR_1"
        image = f"{collection}/{name}{Path(r['image']).name}"
        return {**r, "site_day": site, "collection_id": collection, "image": image}

    validation = [again(r, OTHER, "v") for r in train[:4]]
    test = [again(r, FOLD, "t") for r in train[2:]]
    view = tmp_path / "manifests" / "folds" / FOLD / "thermal"
    write(view / "train_100pct.jsonl", train)
    write(view / "validation.jsonl", validation)
    write(view / "test.jsonl", test)
    cache = tmp_path / "outputs" / "scratch-features" / "m-1024tok"
    (cache / "runs" / "s1").mkdir(parents=True)
    (cache / "cache.json").write_text(json.dumps({"settings": {"model": "m"}}))
    (cache / "runs" / "s1" / "run.json").write_text(
        json.dumps({"scratch": True, "status": "completed"})
    )
    lines, features = [], {}
    for r in [*train, *validation, *test]:
        path = cache / "features" / Path(r["image"]).with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        key = Path(r["image"]).name.lstrip("vt")
        features.setdefault(key, rng.normal(size=(14, 18, 8)).astype("f2"))
        np.save(path, features[key])
        lines.append(json.dumps({**ENTRY, "path": r["image"], "run": "s1"}))
    (cache / "index.jsonl").write_text("\n".join(lines) + "\n")
    return cache


def test_a_head_run_writes_exactly_the_pinned_outputs(tmp_path, golden):
    cache = setup(tmp_path)

    summary = he.run_head(
        cache_dir=cache,
        manifests=tmp_path / "manifests",
        data_root=tmp_path / "raw",
        fold=FOLD,
        camera="thermal",
        percent=100,
        seed=11,
        recipe=he.Recipe(
            steps=24, batch_size=4, learning_rate=0.03, eval_every=4, hidden=8, top_k=20
        ),
        scratch=True,
        run_name=None,  # the default name is part of what is pinned
        device=torch.device("cpu"),
        repo=tmp_path / "repo",
        argv=["aerial-search", "train-head"],
    )

    directory = Path(summary["run"])
    golden.check(
        "head-run",
        {
            "run_dir": str(directory.relative_to(tmp_path)),
            "files": sorted(p.name for p in directory.iterdir()),
            "run.json": golden.run_record(directory),
            "training.json": golden.read_json(directory / "training.json"),
            "detection_metrics.json": golden.read_json(
                directory / "detection_metrics.json"
            ),
            "head.pt": golden.weights(torch.load(directory / "head.pt")),
            "summary": {k: v for k, v in summary.items() if k != "run"},
        },
    )
