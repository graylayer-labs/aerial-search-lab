"""Frozen vision backbones: loading, weight checksums and patch features.

Shared by `tools/forward_pass_cost.py` (#64) and the feature cache (#65). The
weights come from the Hugging Face hub through `transformers` and are never
modified. `NaflexExtractor` turns one image into a grid of patch features and
is the only thing the cache needs from a model, so tests replace it with a
fake.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, Siglip2VisionModel, SiglipVisionModel


@dataclass(frozen=True)
class Spec:
    repo: str
    kind: str  # "siglip", "naflex" or "dinov2"
    patch: int
    tile_px: int  # side of one forward pass for the fixed-resolution models
    # (label, patch budget) for naflex, one forward pass per image
    budgets: tuple[tuple[str, int], ...] = ()
    # hub commit of the weights; loading and checksumming both use it, so a
    # moved branch on the hub cannot change what a cache was made with
    revision: str = "main"


SPECS = {
    "siglip2-base-512": Spec(
        "google/siglip2-base-patch16-512",
        "siglip",
        16,
        512,
        revision="a89f5c5093f902bf39d3cd4d81d2c09867f0724b",
    ),
    "siglip2-large-512": Spec(
        "google/siglip2-large-patch16-512",
        "siglip",
        16,
        512,
        revision="49488218e80259885f3be61d7a9455faf833b7a8",
    ),
    "siglip2-base-naflex": Spec(
        "google/siglip2-base-patch16-naflex",
        "naflex",
        16,
        0,
        (("~512", 1024), ("~1024", 4096)),
        revision="b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
    ),
    "dinov2-base": Spec(
        "facebook/dinov2-base",
        "dinov2",
        14,
        518,
        revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    ),
}


def load(spec: Spec, device: str) -> torch.nn.Module:
    """The vision tower in half precision on `device`, in eval mode."""
    if spec.kind == "siglip":
        model = SiglipVisionModel.from_pretrained(
            spec.repo, dtype=torch.float16, revision=spec.revision
        )
    elif spec.kind == "naflex":
        model = Siglip2VisionModel.from_pretrained(
            spec.repo, dtype=torch.float16, revision=spec.revision
        )
    else:
        model = AutoModel.from_pretrained(
            spec.repo, dtype=torch.float16, revision=spec.revision
        )
    module: torch.nn.Module = model
    return module.to(device).eval()


def weight_checksums(repo: str, revision: str = "main") -> dict[str, str]:
    """SHA-256 of each `.safetensors` file of `repo` at `revision`, by name."""
    from huggingface_hub import snapshot_download

    try:  # the cache first: a stalled hub connection must not hang a run
        root = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        root = Path(
            snapshot_download(repo, revision=revision, allow_patterns=["*.safetensors"])
        )
    out = {}
    for path in sorted(root.glob("*.safetensors")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1 << 24):
                digest.update(block)
        out[path.name] = digest.hexdigest()
    return out


def pool2(grid_features: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Average-pool an h x w x d grid by k x k without dropping anything.

    The output is ceil(h / k) x ceil(w / k). A cell on an odd edge covers fewer
    than k x k patches and is the mean of those real patches only; padding is
    never averaged in. `k` of 1 returns the grid unchanged.
    """
    if k == 1:
        return grid_features
    x = grid_features.permute(2, 0, 1).unsqueeze(0).float()
    pooled = F.avg_pool2d(x, k, ceil_mode=True, count_include_pad=False)
    return pooled.squeeze(0).permute(1, 2, 0).to(grid_features.dtype)


# Bump when `prepare`, `extract` or `pool2` change what a feature file holds.
PREPROCESSING_VERSION = 1


@dataclass(frozen=True)
class Prepared:
    inputs: dict[str, torch.Tensor]
    image_size: tuple[int, int]  # source width, height in pixels


@dataclass(frozen=True)
class Extraction:
    """Pooled features and the geometry that maps their cells back to pixels."""

    array: np.ndarray  # ceil(h/k) x ceil(w/k) x d, float16
    image_size: tuple[int, int]  # source width, height in pixels
    resized_size: tuple[int, int]  # width, height after the processor's resize
    patch_grid: tuple[int, int]  # h, w in patches, before pooling


class NaflexExtractor:
    """One image to a pooled grid of patch features and its geometry.

    The image keeps its aspect ratio but is stretched to a whole number of
    patches in each direction (`resized_size`), so the patch grid covers the
    entire source image. The processor picks the grid that fits `token_budget`
    patches. Any image mode is converted to RGB first, so a one-channel thermal
    frame is replicated to three identical channels. The features are the last
    hidden state of the vision tower, not the pooled embedding.
    """

    def __init__(
        self,
        model,
        processor,
        token_budget: int,
        pooling: int,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.token_budget = token_budget
        self.pooling = pooling
        # applied to the pixels first; None is the unchanged image (#68)
        self.transform = transform

    def prepare(self, image: Image.Image) -> Prepared:
        """Resize and patchify on the CPU; safe to call from worker threads."""
        if self.transform is not None:
            image = self.transform(image)
        inputs = self.processor(
            images=[image.convert("RGB")],
            return_tensors="pt",
            max_num_patches=self.token_budget,
        )
        return Prepared(inputs, (image.width, image.height))

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract(self.prepare(image)).array

    def extract(self, prepared: Prepared) -> Extraction:
        """Run the vision tower on prepared inputs and pool the patch grid."""
        inputs = prepared.inputs
        device = next(self.model.parameters()).device
        feed = {
            k: v.to(device, self.model.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        h, w = (int(v) for v in inputs["spatial_shapes"][0])
        with torch.no_grad():
            tokens = self.model(**feed).last_hidden_state[0]
            grid = pool2(tokens[: h * w].reshape(h, w, -1), self.pooling)
            host = grid.to(torch.float16).cpu()
        if device.type == "mps":
            torch.mps.synchronize()
        patch = int(self.processor.patch_size)
        return Extraction(
            host.numpy(), prepared.image_size, (w * patch, h * patch), (h, w)
        )
