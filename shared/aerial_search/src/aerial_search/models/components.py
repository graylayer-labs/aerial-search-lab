"""Shared model components and utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import functional as vision


def load_image(path: Path) -> Tensor:
    """Load an image from disk as a PyTorch tensor."""
    return decode_image(str(path), mode=ImageReadMode.RGB)


def prepare_image(image: Tensor, size: int = 224) -> Tensor:
    """Preprocess an image: resize, normalize, and convert to float."""
    image = vision.resize(image, [size, size], antialias=True)
    tensor = image.float() / 255
    return vision.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)


def get_device() -> torch.device:
    """Get the best available device (MPS on Mac, CPU fallback)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
