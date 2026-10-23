"""The supervised baseline on tiles of the source frame (#96)."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aerial_search.data import tiling
from aerial_search.experiments import head_experiment as he
from finetuned_detector import experiment as be

FOLD, OTHER = "220109_Baker", "210417_MtErie"
SPEC = be.TileSpec(size=(32, 24), overlap=16, min_visible=0.5, empty_ratio=0.5)


def rec(site: str, name: str, boxes: list[dict]) -> dict:
    return {
        "site_day": site,
        "collection_id": f"{site}_Enterprise_1",
        "camera": "thermal",
        "image": f"{site}_Enterprise_IR_1/{name}.jpg",
        "size": [64, 48],
        "boxes": boxes,
    }


def box(cx: float, cy: float, w: float, h: float) -> dict:
    return {"x_center": cx, "y_center": cy, "width": w, "height": h}


def write_images(raw: Path, records: list[dict]) -> None:
    for r in records:
        path = raw / r["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((48, 64, 3), 90, dtype="u1")).save(path)


def test_tiled_input_is_native_pixels_cut_into_the_caches_input_size():
    assert be.input_size("rgb", "tiled") is None
    rgb = be.tile_spec("rgb", be.BaselineRecipe(input="tiled"))
    thermal = be.tile_spec("thermal", be.BaselineRecipe(input="tiled"))
    assert rgb is not None and rgb.size == (672, 384)
    assert thermal is not None and thermal.size == (560, 448)
    assert be.tile_spec("rgb", be.BaselineRecipe(input="cache")) is None
    custom = be.BaselineRecipe(input="tiled", tile_overlap=200)
    spec = be.tile_spec("rgb", custom)
    assert spec is not None and spec.overlap == 200


def test_training_tiles_hold_clipped_boxes_and_a_share_of_empty_tiles(tmp_path):
    record = rec(FOLD, "x0", [box(0.5, 0.5, 0.25, 0.25)])  # x 24..40, y 18..30
    write_images(tmp_path, [record])
    tiles = be.TiledTrainSet([record], tmp_path, SPEC, seed=0)
    grid = tiling.tile_grid(64, 48, SPEC.size, SPEC.overlap)
    truth = he.record_boxes(record)
    n_positive = sum(len(tiling.boxes_in_tile(truth, t, 0.5)[1]) > 0 for t in grid)
    assert 0 < n_positive < len(grid)
    seen = 0
    for k in range(len(tiles)):
        image, target, scale = tiles[k]
        assert image.shape == (3, 24, 32) and scale.tolist() == [1.0, 1.0]
        boxes = target["boxes"]
        assert (boxes >= 0).all()
        assert (boxes[:, [0, 2]] <= 32).all() and (boxes[:, [1, 3]] <= 24).all()
        seen += len(boxes)
    assert seen >= n_positive
    clear = sum(
        not tiling.touches_box(truth, t) for t in grid
    )  # tiles that can be true negatives
    assert len(tiles) == n_positive + min(int(np.ceil(0.5 * n_positive)), clear)


class Oracle(torch.nn.Module):
    """Sees every tile whole and reports the people in it, in tile pixels."""

    def __init__(self, source, grid, min_visible):
        super().__init__()
        self.source, self.grid, self.min_visible, self.n = source, grid, min_visible, 0

    def forward(self, images):
        out = []
        for _ in images:
            tile = self.grid[self.n % len(self.grid)]
            self.n += 1
            local, _ = tiling.boxes_in_tile(self.source, tile, self.min_visible)
            out.append(
                {
                    "boxes": torch.as_tensor(local, dtype=torch.float32),
                    "scores": torch.full((len(local),), 0.9),
                }
            )
        return out


def test_tiled_prediction_counts_a_person_in_an_overlap_once_in_source_pixels(
    tmp_path,
):
    # 10 x 8 px person at x 20..30, y 14..22: seen whole by several tiles.
    record = rec(FOLD, "x0", [box(25 / 64, 18 / 48, 10 / 64, 8 / 48)])
    write_images(tmp_path, [record])
    truth = he.record_boxes(record)
    grid = tiling.tile_grid(64, 48, SPEC.size, SPEC.overlap)
    assert sum(len(tiling.boxes_in_tile(truth, t, 1.0)[1]) > 0 for t in grid) >= 2
    frames = be.TiledFrames([record], tmp_path, SPEC)
    oracle = Oracle(truth, grid, min_visible=1.0)
    (prediction,) = be.predict(oracle, frames, torch.device("cpu"), batch_size=3)
    assert np.asarray(prediction.boxes).tolist() == [[20.0, 14.0, 30.0, 22.0]]
    assert np.asarray(prediction.scores).tolist() == [np.float32(0.9).item()]


def test_a_tiled_run_scores_whole_frames_on_the_test_site_day(tmp_path, monkeypatch):
    manifests, raw = tmp_path / "manifests", tmp_path / "raw"
    view = manifests / "folds" / FOLD / "thermal"
    view.mkdir(parents=True)
    splits = {
        "train_100pct.jsonl": [
            rec(OTHER, f"t{k}", [box(0.5, 0.5, 0.25, 0.25)]) for k in range(4)
        ],
        "validation.jsonl": [
            rec(OTHER, f"v{k}", [box(0.3, 0.6, 0.25, 0.25)]) for k in range(2)
        ],
        "test.jsonl": [
            rec(FOLD, f"x{k}", [box(0.4, 0.4, 0.25, 0.25)]) for k in range(3)
        ],
    }
    for file, records in splits.items():
        (view / file).write_text("".join(json.dumps(r) + "\n" for r in records))
        write_images(raw, records)
    real = be.build_model

    def tiny_model(arch, **kwargs):
        kwargs["pretrained"] = False  # no download in tests
        return real(arch, **kwargs)

    monkeypatch.setattr(be, "build_model", tiny_model)
    monkeypatch.setattr(be, "tile_spec", lambda camera, recipe: SPEC)
    recipe = be.BaselineRecipe(
        input="tiled",
        steps=2,
        batch_size=2,
        eval_every=1,
        val_frames=0,
        workers=0,
        warmup_steps=1,
    )
    summary = be.run_baseline(
        manifests=manifests,
        data_root=raw,
        fold=FOLD,
        camera="thermal",
        percent=100,
        seed=7,
        recipe=recipe,
        scratch=True,
        run_name="bl-tiled",
        device=torch.device("cpu"),
        repo=tmp_path / "repo",
        argv=["x"],
    )
    assert summary["test"]["n_images"] == 3  # frames, not tiles
    run_file = tmp_path / "repo" / "outputs" / "scratch-bl-tiled" / "run.json"
    assert json.loads(run_file.read_text())["config"]["tiling"]["overlap"] == 16
    training = json.loads((run_file.parent / "training.json").read_text())
    assert training["train_boxes"] == 4 and training["train_boxes_unlabelled"] == 0


class Flood(torch.nn.Module):
    """Reports 150 separate boxes on every tile."""

    def forward(self, images):
        xs = torch.arange(150, dtype=torch.float32) * 0.2
        boxes = torch.stack([xs, xs * 0, xs + 0.1, xs * 0 + 5], dim=1)
        scores = torch.linspace(0.9, 0.1, 150)
        return [{"boxes": boxes, "scores": scores} for _ in images]


def test_tiled_prediction_keeps_no_more_boxes_per_frame_than_the_untiled_cap(tmp_path):
    record = rec(FOLD, "x0", [])
    write_images(tmp_path, [record])
    assert be.BaselineRecipe().frame_cap == be.BaselineRecipe().top_k == 100
    assert be.BaselineRecipe(top_k=50).frame_cap == 50
    assert be.BaselineRecipe(top_k=50, frame_top_k=20).frame_cap == 20
    frames = be.TiledFrames([record], tmp_path, SPEC)
    (prediction,) = be.predict(Flood(), frames, torch.device("cpu"))
    assert 0 < len(np.asarray(prediction.scores)) <= 100


def test_boxes_labelled_in_no_tile_are_counted_and_slivers_are_not_negatives(tmp_path):
    # A person as wide as the frame is at most 50% visible in a 32 px tile.
    wide = rec(FOLD, "w0", [box(0.5, 0.5, 1.0, 0.25)])
    ok = rec(FOLD, "w1", [box(0.1, 0.1, 0.1, 0.1)])  # top-left corner only
    write_images(tmp_path, [wide, ok])
    strict = be.TileSpec((32, 24), 16, min_visible=0.6, empty_ratio=2.0)
    tiles = be.TiledTrainSet([wide, ok], tmp_path, strict, seed=0)
    assert tiles.n_boxes == 2 and tiles.n_unlabelled_boxes == 1
    empties = 0
    for r, tile, local in tiles.tiles:
        if len(local) == 0:  # an empty negative: it must not touch any box
            truth = he.record_boxes(tiles.records[r])
            assert not tiling.touches_box(truth, tile)
            empties += 1
    assert empties == 2  # only the corner frame has clear tiles; ratio 2.0 x 1


def test_empty_tiles_are_sampled_from_the_clear_ones_up_to_the_ratio(tmp_path):
    record = rec(FOLD, "c0", [box(0.1, 0.1, 0.1, 0.1)])
    write_images(tmp_path, [record])
    spec = be.TileSpec((32, 24), 16, min_visible=0.5, empty_ratio=2.0)
    tiles = be.TiledTrainSet([record], tmp_path, spec, seed=0)
    grid = tiling.tile_grid(64, 48, spec.size, spec.overlap)
    truth = he.record_boxes(record)
    n_positive = sum(len(tiling.boxes_in_tile(truth, t, 0.5)[1]) > 0 for t in grid)
    clear = sum(not tiling.touches_box(truth, t) for t in grid)
    assert n_positive >= 1 and clear > 2 * n_positive
    empties = sum(len(local) == 0 for _, _, local in tiles.tiles)
    assert empties == min(int(np.ceil(2.0 * n_positive)), clear)
