"""The thermal input ablation (#68): what differs between arms, and nothing else."""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

from aerial_search.models import thermal_input as ti
from aerial_search.models.backbone import NaflexExtractor, pool2


def ramp(low: int = 90, high: int = 130) -> Image.Image:
    """A low-contrast grey frame stored as three identical channels, as the
    dataset's thermal JPEGs are."""
    row = np.linspace(low, high, 64).astype(np.uint8)
    grey = np.tile(row, (32, 1))
    return Image.fromarray(np.stack([grey] * 3, axis=-1))


def test_equalise_spreads_a_narrow_histogram_and_keeps_three_equal_channels():
    out = np.asarray(ti.equalise(ramp()))

    assert out.shape == (32, 64, 3)
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 0] == out[..., 2]).all()
    assert out.max() - out.min() > 200  # was 40 wide


def test_equalise_is_deterministic():
    a, b = np.asarray(ti.equalise(ramp())), np.asarray(ti.equalise(ramp()))

    assert np.array_equal(a, b)


def test_equalisation_is_thermal_only_and_replicate_changes_nothing():
    assert ti.input_transform("replicate", "thermal") is None
    assert ti.input_transform("replicate", "rgb") is None
    assert ti.input_transform("equalise", "thermal") is ti.equalise
    with pytest.raises(ValueError, match="thermal"):
        ti.input_transform("equalise", "rgb")
    with pytest.raises(ValueError, match="unknown"):
        ti.input_transform("sharpen", "thermal")


def tiny_model() -> Siglip2VisionModel:
    config = Siglip2VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=16,
        num_patches=64,
    )
    return Siglip2VisionModel(config).eval()


def test_a_transform_changes_the_features_and_none_matches_the_legacy_path():
    model, processor = tiny_model(), Siglip2ImageProcessor(patch_size=16)
    plain = NaflexExtractor(model, processor, token_budget=64, pooling=2)
    same = NaflexExtractor(model, processor, 64, 2, transform=None)
    equalised = NaflexExtractor(model, processor, 64, 2, transform=ti.equalise)
    frame = ramp()

    assert np.array_equal(plain(frame), same(frame))
    assert not np.array_equal(plain(frame), equalised(frame))
    # the transform is applied to pixels before the processor, nothing else
    direct = plain(ti.equalise(frame))
    assert np.array_equal(equalised(frame), direct)


def test_patchify_inverts_the_processors_patch_layout():
    processor = Siglip2ImageProcessor(patch_size=16)
    prepared = processor(
        images=[ramp().resize((160, 90))], return_tensors="pt", max_num_patches=64
    )
    h, w = (int(v) for v in prepared["spatial_shapes"][0])
    patches = prepared["pixel_values"]  # (1, 64, 768), padded

    image = ti.unpatchify(patches[:, : h * w], h, w, 16)

    assert image.shape == (1, 3, h * 16, w * 16)
    assert torch.equal(ti.patchify(image, 16), patches[:, : h * w])


def test_a_fresh_stem_is_the_identity():
    stem = ti.ThermalStem()
    x = torch.randn(2, 3, 32, 48)

    assert torch.equal(stem(x), x)


def test_the_stem_trains_through_a_frozen_backbone_and_the_backbone_does_not_move():
    torch.manual_seed(0)
    model, processor = tiny_model().float(), Siglip2ImageProcessor(patch_size=16)
    for p in model.parameters():
        p.requires_grad_(False)
    before = [p.clone() for p in model.parameters()]
    stem = ti.ThermalStem()
    backbone = ti.StemBackbone(model, stem, patch=16, pooling=2)
    prepared = processor(
        images=[ramp().resize((160, 90))] * 2, return_tensors="pt", max_num_patches=64
    )

    grid = backbone(prepared)

    h, w = (int(v) for v in prepared["spatial_shapes"][0])
    assert grid.shape == (2, (h + 1) // 2, (w + 1) // 2, 32)
    grid.square().mean().backward()
    assert stem.out.weight.grad is not None and stem.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    optimiser = torch.optim.AdamW(stem.parameters(), lr=1e-2)
    optimiser.step()
    assert not torch.equal(backbone(prepared), grid)  # the stem moved
    assert all(
        torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )


def test_a_fresh_stem_backbone_matches_the_extractor_features():
    """Arm C starts as arm A: same pixels, same patches, same pooling."""
    torch.manual_seed(0)
    model, processor = tiny_model(), Siglip2ImageProcessor(patch_size=16)
    extractor = NaflexExtractor(model, processor, token_budget=64, pooling=2)
    prepared = extractor.prepare(ramp().resize((160, 90)))
    backbone = ti.StemBackbone(model, ti.ThermalStem(), patch=16, pooling=2)

    with torch.no_grad():
        got = backbone(prepared.inputs)[0]
    h, w = (int(v) for v in prepared.inputs["spatial_shapes"][0])
    tokens = model(**prepared.inputs).last_hidden_state[0, : h * w]
    want = pool2(tokens.reshape(h, w, -1), 2)

    assert torch.allclose(got, want.float(), atol=1e-5)
