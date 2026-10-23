import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

from aerial_search.experiments import feature_cache as fc
from aerial_search.models.backbone import Extraction, NaflexExtractor, pool2

SETTINGS = fc.CacheSettings(
    model="fake",
    repo="fake/fake",
    revision="rev1",
    weights_sha256={"model.safetensors": "aa"},
    token_budget=64,
    pooling=2,
    dtype="float16",
    preprocessing_version=1,
)


class FakeExtractor:
    """Features that depend on the pixels, so a stale file is detectable."""

    def __init__(self) -> None:
        self.calls = 0

    def prepare(self, image: Image.Image) -> float:  # runs in a worker thread
        return float(np.asarray(image.convert("RGB"), dtype=np.float32).mean())

    def extract(self, prepared: float) -> Extraction:
        self.calls += 1
        return Extraction(
            np.full((2, 3, 4), prepared, dtype=np.float16),
            image_size=(32, 18),
            resized_size=(80, 48),
            patch_grid=(3, 5),
        )


def make_frames(root: Path, names: dict[str, str]) -> list[fc.Frame]:
    frames = []
    for i, (path, camera) in enumerate(names.items()):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 18), (10 * (i + 1),) * 3).save(root / path)
        frames.append(fc.Frame(path, camera, "c1", f"sha{i}"))
    return frames


@pytest.fixture
def frames(tmp_path: Path) -> list[fc.Frame]:
    return make_frames(
        tmp_path / "raw",
        {"VIS_1/a_0.jpeg": "rgb", "VIS_1/a_1.jpeg": "rgb", "IR_2/b_0.jpeg": "thermal"},
    )


def build(tmp_path, frames, extractor, settings=SETTINGS, **kw):
    return fc.build_cache(
        tmp_path / "cache",
        settings,
        frames,
        extractor,
        tmp_path / "raw",
        run="r1",
        commit="abc",
        **kw,
    )


def test_writes_fp16_features_per_image_with_camera_and_provenance(tmp_path, frames):
    summary = build(tmp_path, frames, FakeExtractor())

    assert summary.computed == 3
    cache = tmp_path / "cache"
    array = fc.read_features(cache, "VIS_1/a_0.jpeg")
    assert array.dtype == np.float16 and array.shape == (2, 3, 4)
    index = fc.read_index(cache)
    assert index["IR_2/b_0.jpeg"]["camera"] == "thermal"
    entry = index["VIS_1/a_0.jpeg"]
    assert entry["grid"] == [2, 3] and entry["sha256"] == "sha0"
    assert entry["run"] == "r1" and entry["commit"] == "abc"
    assert entry["image_size"] == [32, 18] and entry["resized_size"] == [80, 48]
    assert entry["patch_grid"] == [3, 5] and entry["pooling"] == 2
    assert json.loads((cache / "cache.json").read_text())["settings"]["model"] == "fake"


def test_each_stored_file_holds_its_own_images_features(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())

    for frame in frames:
        with Image.open(tmp_path / "raw" / frame.path) as image:
            expected = np.asarray(image.convert("RGB"), dtype=np.float32).mean()
        stored = fc.read_features(tmp_path / "cache", frame.path)
        assert np.all(stored == np.float16(expected)), frame.path
    values = {
        float(fc.read_features(tmp_path / "cache", f.path)[0, 0, 0]) for f in frames
    }
    assert len(values) == len(frames)  # the three images really differ


def test_second_run_skips_finished_images_and_computes_the_rest(tmp_path, frames):
    build(tmp_path, frames[:2], FakeExtractor())
    again = FakeExtractor()

    summary = build(tmp_path, frames, again, verify=0)

    assert (summary.skipped, summary.computed) == (2, 1)
    assert again.calls == 1


def test_resume_recomputes_a_sample_of_finished_images(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())
    again = FakeExtractor()

    summary = build(tmp_path, frames, again, verify=2)

    assert summary.verified == 2 and summary.computed == 0
    assert again.calls == 2


def test_resume_refuses_when_a_finished_file_no_longer_matches(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())
    path = tmp_path / "cache" / "features" / "VIS_1" / "a_0.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=np.float16))

    with pytest.raises(fc.CacheError, match="a_0"):
        build(tmp_path, frames, FakeExtractor(), verify=3)


def test_a_crash_between_file_and_index_is_recomputed(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())
    index = tmp_path / "cache" / "index.jsonl"
    lines = index.read_text().splitlines()
    index.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:10])  # torn last line

    summary = build(tmp_path, frames, FakeExtractor(), verify=0)

    assert summary.computed == 1 and summary.skipped == 2
    assert set(fc.read_index(tmp_path / "cache")) == {f.path for f in frames}


@pytest.mark.parametrize(
    "change",
    [
        {"token_budget": 128},
        {"pooling": 1},
        {"dtype": "float32"},
        {"weights_sha256": {"model.safetensors": "bb"}},
        {"repo": "other/model"},
        {"revision": "rev2"},
        {"preprocessing_version": 2},
    ],
)
def test_a_cache_refuses_settings_that_differ(tmp_path, frames, change):
    build(tmp_path, frames, FakeExtractor())
    other = fc.CacheSettings(**{**SETTINGS.__dict__, **change})

    with pytest.raises(fc.CacheError, match="different settings"):
        build(tmp_path, frames, FakeExtractor(), settings=other)
    with pytest.raises(fc.CacheError, match="different settings"):
        fc.check_settings(tmp_path / "cache", other)


def test_select_frames_by_camera_collection_and_limit(tmp_path):
    rows = [
        {"collection_id": "c1", "rgb_image": "V/1.jpeg", "thermal_image": "I/1.jpeg"},
        {"collection_id": "c1", "rgb_image": "V/2.jpeg", "thermal_image": "I/2.jpeg"},
        {"collection_id": "c2", "rgb_image": "V/3.jpeg", "thermal_image": "I/1.jpeg"},
    ]
    manifest = tmp_path / "all_pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    pinned = {"V/1.jpeg": "h1", "V/3.jpeg": "h3", "I/1.jpeg": "hi1"}

    rgb = fc.select_frames(manifest, "rgb", pinned)
    assert [f.path for f in rgb] == ["V/1.jpeg", "V/2.jpeg", "V/3.jpeg"]
    assert rgb[0].sha256 == "h1" and rgb[1].sha256 is None
    thermal = fc.select_frames(manifest, "thermal", pinned)
    assert [f.path for f in thermal] == ["I/1.jpeg", "I/2.jpeg"]  # de-duplicated
    assert thermal[0].camera == "thermal"
    only = fc.select_frames(manifest, "rgb", pinned, collections=["c2"], limit=5)
    assert [f.path for f in only] == ["V/3.jpeg"]
    assert len(fc.select_frames(manifest, "rgb", pinned, limit=2)) == 2
    with pytest.raises(fc.CacheError, match="no-such"):
        fc.select_frames(manifest, "rgb", pinned, collections=["no-such"])


def test_parts_split_the_frames_into_contiguous_chunks_that_cover_all(tmp_path):
    rows = [
        {"collection_id": "c", "rgb_image": f"V/{i}.jpeg", "thermal_image": "I/0.jpeg"}
        for i in range(7)
    ]
    manifest = tmp_path / "all_pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))

    parts = [fc.select_frames(manifest, "rgb", {}, part=(k, 3)) for k in (1, 2, 3)]

    assert [len(p) for p in parts] == [3, 3, 1]
    assert [f.path for p in parts for f in p] == [f"V/{i}.jpeg" for i in range(7)]
    with pytest.raises(fc.CacheError, match="part"):
        fc.select_frames(manifest, "rgb", {}, part=(4, 3))


def test_naflex_extractor_returns_a_pooled_fp16_grid_for_either_camera():
    config = Siglip2VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=16,
        num_patches=64,
    )
    model = Siglip2VisionModel(config).eval()
    processor = Siglip2ImageProcessor(patch_size=16)
    extractor = NaflexExtractor(model, processor, token_budget=64, pooling=2)
    rgb = Image.new("RGB", (160, 90), (200, 30, 30))
    thermal = Image.new("L", (160, 90), 128)  # a one-channel frame

    out = extractor(rgb)
    assert out.dtype == np.float16
    assert out.shape == (3, 5, 32)  # a 6 x 10 grid pooled 2 x 2
    assert extractor(thermal).shape == out.shape
    # thermal goes in as three identical channels
    same = extractor(Image.merge("RGB", [thermal] * 3))
    assert np.array_equal(extractor(thermal), same)


def test_pool2_keeps_odd_edges_and_averages_only_real_cells():
    x = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)

    pooled = pool2(x, 2)

    assert pooled.shape == (2, 3, 2)  # ceil, nothing dropped
    assert torch.allclose(pooled[0, 0], x[:2, :2].mean(dim=(0, 1)))
    assert torch.allclose(pooled[0, 2], x[:2, 4:].mean(dim=(0, 1)))  # edge column
    assert torch.allclose(pooled[1, 0], x[2:, :2].mean(dim=(0, 1)))  # edge row
    assert torch.equal(pooled[1, 2], x[2, 4])  # the corner is one real cell
    assert pool2(x, 1) is x


class PixelModel:
    """A stand-in tower whose token features are the patch pixels themselves, so
    a bright patch can be followed through to a pooled cell."""

    dtype = torch.float32

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, pixel_values, pixel_attention_mask, spatial_shapes):
        class Out:
            last_hidden_state = pixel_values

        return Out()


def test_naflex_extractor_places_a_bright_patch_in_its_cell_and_records_geometry():
    processor = Siglip2ImageProcessor(patch_size=16)
    extractor = NaflexExtractor(PixelModel(), processor, token_budget=64, pooling=2)
    image = Image.new("RGB", (112, 80), (0, 0, 0))  # a 7 x 9 patch grid
    # white in the bottom-right corner: inside patch (row 6, column 8)
    image.paste((255, 255, 255), (104, 70, 112, 80))
    prepared = extractor.prepare(image)
    tokens = prepared.inputs["pixel_values"][0, : 7 * 9].reshape(7, 9, -1)

    result = extractor.extract(prepared)

    assert result.array.dtype == np.float16
    assert result.array.shape == (4, 5, 768)  # ceil of 7 x 9 over 2 x 2
    assert result.image_size == (112, 80)
    assert result.resized_size == (144, 112)
    assert result.patch_grid == (7, 9)
    # the corner cell holds one real patch, so it equals that patch and is not
    # halved or quartered by padding
    assert np.allclose(result.array[3, 4], tokens[6, 8].numpy(), atol=1e-2)
    # an interior cell is a plain 2 x 2 mean
    assert np.allclose(
        result.array[1, 2], tokens[2:4, 4:6].mean(dim=(0, 1)).numpy(), atol=1e-2
    )
    # and it is the brightest cell
    brightest = np.unravel_index(result.array.mean(axis=2).argmax(), (4, 5))
    assert tuple(int(v) for v in brightest) == (3, 4)


def test_cell_boxes_cover_the_source_image_with_a_partial_edge_cell():
    entry = {
        "image_size": [112, 80],
        "resized_size": [144, 112],
        "patch_grid": [7, 9],
        "grid": [4, 5],
        "pooling": 2,
    }

    boxes = fc.cell_boxes(entry)

    assert boxes.shape == (4, 5, 4)
    sx, sy = 112 / 144, 80 / 112
    assert np.allclose(boxes[0, 0], [0, 0, 32 * sx, 32 * sy])
    assert np.allclose(boxes[1, 2], [64 * sx, 32 * sy, 96 * sx, 64 * sy])
    # the last row and column hold one patch row/column: half-size cells
    assert np.allclose(boxes[3, 4], [128 * sx, 96 * sy, 112, 80])
    assert boxes[:, :, 0].min() == 0 and boxes[:, :, 2].max() == 112
    assert boxes[:, :, 1].min() == 0 and boxes[:, :, 3].max() == 80
    # cells tile the image: each row's widths sum to the image width
    widths = boxes[0, :, 2] - boxes[0, :, 0]
    assert widths.sum() == pytest.approx(112)


@pytest.fixture
def dataset(tmp_path: Path):
    """Raw frames and an all_pairs manifest."""
    raw = tmp_path / "data" / "raw"
    names = {"V_1/a_0.jpeg": "rgb", "V_1/a_1.jpeg": "rgb", "I_2/b_0.jpeg": "thermal"}
    make_frames(raw, names)
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    T = "I_2/b_0.jpeg"
    rows = [
        {"collection_id": "c1", "rgb_image": "V_1/a_0.jpeg", "thermal_image": T},
        {"collection_id": "c1", "rgb_image": "V_1/a_1.jpeg", "thermal_image": T},
    ]
    (manifests / "all_pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return raw, manifests


def fake_loader(budget: int, pooling: int, transform=None) -> FakeExtractor:
    return FakeExtractor()


def run_features(
    tmp_path,
    dataset,
    session="s1",
    pooling=2,
    limit=None,
    scratch=True,
    weights=lambda repo, revision: {"model.safetensors": "aa"},
):
    raw, manifests = dataset
    (tmp_path / "repo").mkdir(exist_ok=True)
    return fc.cache_features(
        model="siglip2-base-naflex",
        camera="rgb",
        data_root=raw,
        manifests=manifests,
        collections=None,
        limit=limit,
        token_budget=1024,
        pooling=pooling,
        verify=8,
        scratch=scratch,
        session=session,
        repo=tmp_path / "repo",
        load_extractor=fake_loader,
        weights=weights,
    )


def test_a_run_writes_the_cache_and_a_run_record_inside_it(tmp_path, dataset):
    summary = run_features(tmp_path, dataset)

    cache = tmp_path / "repo/outputs/scratch-features/siglip2-base-naflex-1024tok"
    assert summary.computed == 2
    record = json.loads((cache / "runs/s1/run.json").read_text())
    assert record["status"] == "completed" and record["scratch"] is True
    assert record["config"]["settings"]["token_budget"] == 1024
    assert record["config"]["settings"]["weights_sha256"] == {"model.safetensors": "aa"}
    assert record["config"]["camera"] == "rgb"
    assert fc.read_index(cache)["V_1/a_1.jpeg"]["run"] == "s1"
    assert not list(dataset[1].glob("*.npy"))  # the manifests were only read


def test_a_run_with_other_settings_refuses_before_starting_a_run(tmp_path, dataset):
    run_features(tmp_path, dataset)
    cache = tmp_path / "repo/outputs/scratch-features/siglip2-base-naflex-1024tok"

    with pytest.raises(fc.CacheError, match="different settings"):
        run_features(tmp_path, dataset, session="s2", pooling=1)
    with pytest.raises(fc.CacheError, match="different settings"):
        other = {"model.safetensors": "zz"}
        run_features(
            tmp_path, dataset, session="s3", weights=lambda repo, revision: other
        )
    assert sorted(p.name for p in (cache / "runs").iterdir()) == ["s1"]


def test_a_limit_needs_a_scratch_run(tmp_path, dataset):
    with pytest.raises(fc.CacheError, match="scratch"):
        run_features(tmp_path, dataset, limit=1, scratch=False)
    assert run_features(tmp_path, dataset, limit=1).computed == 1


def test_a_changed_transformers_version_only_warns(tmp_path, frames, monkeypatch):
    build(tmp_path, frames, FakeExtractor())
    monkeypatch.setattr(fc, "_transformers_version", lambda: "9.9.9")

    with pytest.warns(UserWarning, match="transformers"):
        build(tmp_path, frames, FakeExtractor(), verify=0)
    record = json.loads((tmp_path / "cache" / "cache.json").read_text())
    assert record["environment"]["transformers"] != "9.9.9"  # the first one stays


def test_a_finished_image_whose_source_hash_changed_is_recomputed(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())
    changed = [
        fc.Frame(f.path, f.camera, f.collection_id, f.sha256 + "x") for f in frames[:1]
    ]
    again = FakeExtractor()

    summary = build(tmp_path, [*changed, *frames[1:]], again, verify=0)

    assert (summary.stale, summary.computed, summary.skipped) == (1, 1, 2)
    assert again.calls == 1
    assert fc.read_index(tmp_path / "cache")[frames[0].path]["sha256"] == "sha0x"


def test_resume_reports_the_measured_drift_and_refuses_beyond_the_limit(
    tmp_path, frames
):
    build(tmp_path, frames, FakeExtractor())
    cache = tmp_path / "cache"
    same = build(tmp_path, frames, FakeExtractor(), verify=3)
    assert same.max_drift == 0.0

    path = cache / "features" / "VIS_1" / "a_0.npy"
    stored = np.load(path)
    noisy = stored.copy()
    noisy[0, 0, 0] = np.nextafter(noisy[0, 0, 0], np.float16(1000))  # one step
    np.save(path, noisy)
    small = build(tmp_path, frames, FakeExtractor(), verify=3)
    assert 0 < small.max_drift < fc.VERIFY_TOLERANCE

    np.save(path, stored * np.float16(1.05))  # a 5% change is not noise
    with pytest.raises(fc.CacheError, match="a_0"):
        build(tmp_path, frames, FakeExtractor(), verify=3)


def test_two_paths_that_map_to_one_feature_file_are_refused(tmp_path):
    rows = [
        {"collection_id": "c", "rgb_image": "V/1.jpeg", "thermal_image": "I/1.jpeg"},
        {"collection_id": "c", "rgb_image": "V/1.jpg", "thermal_image": "I/2.jpeg"},
    ]
    manifest = tmp_path / "all_pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))

    with pytest.raises(fc.CacheError, match="same feature file"):
        fc.select_frames(manifest, "rgb", {})


def test_a_session_lock_blocks_a_second_session_and_is_released(tmp_path, dataset):
    cache = tmp_path / "repo/outputs/scratch-features/siglip2-base-naflex-1024tok"
    (tmp_path / "repo").mkdir()
    cache.mkdir(parents=True)
    (cache / "session.lock").write_text("pid 1 session other")

    with pytest.raises(fc.CacheError, match="session.lock"):
        run_features(tmp_path, dataset)
    assert not (cache / "runs").exists()  # refused before a run started

    (cache / "session.lock").unlink()
    run_features(tmp_path, dataset)
    assert not (cache / "session.lock").exists()
    run_features(tmp_path, dataset, session="s2")  # released, so a second works


def write_fold(manifests: Path, files: dict[str, list[dict]]) -> None:
    for name, records in files.items():
        path = manifests / "folds" / "siteday" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records))


UNPAIRED = {"site_day": "siteday", "collection_id": "c1", "camera": "rgb"}
CACHE = "repo/outputs/scratch-features/siglip2-base-naflex-1024tok"


def test_a_frame_only_in_a_fold_manifest_is_selected_after_the_paired_ones(
    tmp_path, dataset
):
    raw, manifests = dataset
    write_fold(
        manifests,
        {
            "rgb/train_100pct.jsonl": [{**UNPAIRED, "image": "V_1/only_train.jpeg"}],
            "rgb/test.jsonl": [
                {**UNPAIRED, "image": "V_1/only_test.jpeg"},
                {**UNPAIRED, "image": "V_1/a_0.jpeg"},  # also paired
            ],
            "thermal/test.jsonl": [{**UNPAIRED, "camera": "thermal", "image": "I/x"}],
            "unlabelled.jsonl": [
                {
                    "collection_id": "c1",
                    "rgb_image": "V_1/a_1.jpeg",
                    "thermal_image": "I_2/b_0.jpeg",
                }
            ],
        },
    )
    pinned = {"V_1/only_test.jpeg": "ht"}

    frames = fc.select_frames(
        manifests / "all_pairs.jsonl", "rgb", pinned, folds=manifests / "folds"
    )

    assert [f.path for f in frames] == [
        "V_1/a_0.jpeg",
        "V_1/a_1.jpeg",
        "V_1/only_test.jpeg",
        "V_1/only_train.jpeg",
    ]
    assert frames[2].sha256 == "ht" and frames[3].sha256 is None
    thermal = fc.select_frames(
        manifests / "all_pairs.jsonl", "thermal", {}, folds=manifests / "folds"
    )
    assert [f.path for f in thermal] == ["I_2/b_0.jpeg", "I/x"]


def test_a_resume_computes_only_the_frames_the_folds_add(tmp_path, dataset):
    raw, manifests = dataset
    run_features(tmp_path, dataset, session="s1")
    cache = tmp_path / CACHE
    before = fc.read_index(cache)
    make_frames(raw, {"V_1/extra.jpeg": "rgb"})
    write_fold(manifests, {"rgb/test.jsonl": [{**UNPAIRED, "image": "V_1/extra.jpeg"}]})

    summary = run_features(tmp_path, dataset, session="s2")

    after = fc.read_index(cache)
    assert summary.computed == 1 and summary.skipped == 2
    assert after["V_1/extra.jpeg"]["run"] == "s2"
    assert {p: after[p] for p in before} == before
    record = json.loads((cache / "runs/s2/run.json").read_text())
    assert any("folds/siteday/rgb/test.jsonl" in str(p) for p in record["inputs"])


def test_missing_frames_lists_what_the_manifests_reference_but_the_cache_lacks(
    tmp_path, dataset
):
    raw, manifests = dataset
    make_frames(raw, {"V_1/extra.jpeg": "rgb"})
    write_fold(manifests, {"rgb/test.jsonl": [{**UNPAIRED, "image": "V_1/extra.jpeg"}]})
    run_features(tmp_path, dataset, session="s1")  # selects the extra frame too
    cache = tmp_path / CACHE

    assert fc.missing_frames(cache, manifests) == {
        "rgb": [],
        "thermal": ["I_2/b_0.jpeg"],  # this run cached rgb only
    }

    index = cache / "index.jsonl"
    kept = [ln for ln in index.read_text().splitlines() if "extra" not in ln]
    index.write_text("\n".join(kept) + "\n")
    assert fc.missing_frames(cache, manifests)["rgb"] == ["V_1/extra.jpeg"]
    # a cache that holds one camera is checked for that camera only
    assert fc.missing_frames(cache, manifests, ("rgb",)).keys() == {"rgb"}


# --- the thermal input arm is part of the cache's identity (#68) ---------------


def other_arm() -> fc.CacheSettings:
    return dataclasses.replace(SETTINGS, input_handling="equalise")


def test_the_default_input_handling_is_replicate_and_names_the_existing_cache():
    assert SETTINGS.input_handling == "replicate"
    assert SETTINGS.name() == "fake-64tok"
    assert other_arm().name() == "fake-64tok-equalise"


def test_a_cache_made_before_the_field_existed_is_the_replicate_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    old = {k: v for k, v in SETTINGS.__dict__.items() if k != "input_handling"}
    (cache / "cache.json").write_text(
        json.dumps(
            {
                "settings": old,
                "environment": {"transformers": fc._transformers_version()},
            }
        )
    )

    fc.check_settings(cache, SETTINGS)  # the same cache
    with pytest.raises(fc.CacheError, match="input_handling"):
        fc.check_settings(cache, other_arm())


def test_a_cache_refuses_a_run_with_another_input_handling(tmp_path, frames):
    build(tmp_path, frames, FakeExtractor())

    with pytest.raises(fc.CacheError, match="input_handling"):
        build(tmp_path, frames, FakeExtractor(), settings=other_arm())


def run_arm(tmp_path, dataset, camera, input_handling, loader):
    raw, manifests = dataset
    (tmp_path / "repo").mkdir(exist_ok=True)
    return fc.cache_features(
        model="siglip2-base-naflex",
        camera=camera,
        data_root=raw,
        manifests=manifests,
        collections=None,
        limit=None,
        token_budget=1024,
        pooling=2,
        verify=0,
        scratch=True,
        session="s1",
        repo=tmp_path / "repo",
        load_extractor=loader,
        weights=lambda repo, revision: {"model.safetensors": "aa"},
        input_handling=input_handling,
    )


def test_equalisation_is_refused_for_the_rgb_camera(tmp_path, dataset):
    with pytest.raises(ValueError, match="thermal"):
        run_arm(tmp_path, dataset, "rgb", "equalise", lambda b, p, t=None: None)


def test_an_equalised_thermal_cache_lives_in_its_own_directory_and_records_why(
    tmp_path, dataset
):
    seen = []

    def loader(budget, pooling, transform=None):
        seen.append(transform)
        return FakeExtractor()

    run_arm(tmp_path, dataset, "thermal", "equalise", loader)

    top = tmp_path / "repo/outputs/scratch-features"
    cache = top / "siglip2-base-naflex-1024tok-equalise"
    assert not (top / "siglip2-base-naflex-1024tok").exists()
    stored = json.loads((cache / "cache.json").read_text())
    assert stored["settings"]["input_handling"] == "equalise"
    record = json.loads((cache / "runs/s1/run.json").read_text())
    assert "equalis" in record["config"]["preprocessing"]
    assert seen == [fc.input_transform("equalise", "thermal")]
