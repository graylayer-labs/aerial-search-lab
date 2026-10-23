"""Self-supervised contrastive encoder pair."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn
from torchvision.models import resnet18


class ContrastiveEncoders(nn.Module):
    """Separate RGB and thermal encoders with a shared embedding size."""

    def __init__(self, embedding_size: int = 128) -> None:
        super().__init__()
        self.rgb = _make_encoder(embedding_size)
        self.thermal = _make_encoder(embedding_size)

    def forward(self, rgb: Tensor, thermal: Tensor) -> tuple[Tensor, Tensor]:
        return functional.normalize(self.rgb(rgb)), functional.normalize(
            self.thermal(thermal)
        )


def _make_encoder(embedding_size: int) -> nn.Sequential:
    """Create a ResNet-18 encoder with embedding head."""
    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()
    return nn.Sequential(
        backbone,
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, embedding_size),
    )
