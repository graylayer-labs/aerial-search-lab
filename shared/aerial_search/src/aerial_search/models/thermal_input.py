"""How a one-channel thermal frame is fed to the frozen RGB backbone (#68).

Three arms, and only these differ between them:

- `replicate`: the frame as stored. The dataset's thermal JPEGs already hold
  three identical channels, so this is what the cache has always done.
- `equalise`: histogram equalisation of the grey values, then the same three
  identical channels. Changes pixels, so it needs its own feature cache.
- `stem`: a small learned convolutional stem in front of the backbone, trained
  with the head. It sits before the frozen backbone, so no cache can serve it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import Tensor, nn

from aerial_search.models.backbone import pool2

ARMS = ("replicate", "equalise", "stem")
CACHED_ARMS = ("replicate", "equalise")
DEFAULT_ARM = "replicate"


def equalise(image: Image.Image) -> Image.Image:
    """Histogram-equalise the grey values; three identical channels out."""
    return ImageOps.equalize(image.convert("L")).convert("RGB")


def input_transform(
    arm: str, camera: str
) -> Callable[[Image.Image], Image.Image] | None:
    """The pixel transform a cached arm applies before the processor, if any.

    Equalisation is for thermal frames only: asking for it on another camera
    raises, so an RGB cache can never be built with it by mistake.
    """
    if arm not in CACHED_ARMS:
        raise ValueError(f"unknown cached input handling {arm!r}; use {CACHED_ARMS}")
    if arm == "replicate":
        return None
    if camera != "thermal":
        raise ValueError(f"{arm!r} input handling is for the thermal camera only")
    return equalise


def unpatchify(patches: Tensor, h: int, w: int, patch: int) -> Tensor:
    """(b, h*w, patch*patch*3) in the NaFlex processor's layout to (b, 3, H, W)."""
    b = patches.shape[0]
    x = patches.reshape(b, h, w, patch, patch, 3)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, h * patch, w * patch)


def patchify(images: Tensor, patch: int) -> Tensor:
    """The inverse of `unpatchify`, the processor's `convert_image_to_patches`."""
    b, c, height, width = images.shape
    h, w = height // patch, width // patch
    x = images.reshape(b, c, h, patch, w, patch).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, h * w, patch * patch * c)


class ThermalStem(nn.Module):
    """Two 3x3 convolutions with a skip: image + conv(GELU(conv(image))).

    Works on the normalised image the processor produces (3 channels), keeps
    its size, and starts as the identity because the output layer is zero
    initialised, so training begins exactly at arm `replicate`.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.inp = nn.Conv2d(3, hidden, 3, padding=1)
        self.act = nn.GELU()
        self.out = nn.Conv2d(hidden, 3, 3, padding=1)
        bias = self.out.bias
        assert bias is not None
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(bias)

    def forward(self, image: Tensor) -> Tensor:
        return image + self.out(self.act(self.inp(image)))


class StemBackbone(nn.Module):
    """Stem, frozen tower and pooling as one differentiable map to features.

    Takes the processor's output for a batch of frames that share one grid,
    runs the stem on the unpatchified image in float32, repatches, runs the
    (frozen, possibly half-precision) tower, and pools the patch grid like the
    cache does. Gradients reach the stem through the tower; the tower's
    weights must have `requires_grad` off.
    """

    def __init__(self, model: nn.Module, stem: ThermalStem, patch: int, pooling: int):
        super().__init__()
        self.model, self.stem = model, stem
        self.patch, self.pooling = patch, pooling

    def forward(self, inputs: Mapping[str, Any]) -> Tensor:
        shapes = inputs["spatial_shapes"]
        if not (shapes == shapes[0]).all():
            raise ValueError("a batch must share one patch grid")
        h, w = (int(v) for v in shapes[0])
        n = h * w
        pixels = inputs["pixel_values"]
        device = next(self.stem.parameters()).device
        image = unpatchify(pixels[:, :n].to(device).float(), h, w, self.patch)
        patches = patchify(self.stem(image), self.patch)
        padded = torch.zeros_like(pixels, dtype=patches.dtype, device=device)
        padded[:, :n] = patches
        dtype = next(self.model.parameters()).dtype
        tokens = self.model(
            pixel_values=padded.to(dtype),
            pixel_attention_mask=inputs["pixel_attention_mask"].to(device),
            spatial_shapes=shapes.to(device),
        ).last_hidden_state
        return torch.stack(
            [pool2(t[:n].reshape(h, w, -1), self.pooling).float() for t in tokens]
        )
