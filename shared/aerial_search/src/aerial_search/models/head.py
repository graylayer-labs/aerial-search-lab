"""A CenterNet-style person-centre head on a frozen feature grid (#66).

The design and its precedent are in docs/detection-head-design.md. The head
reads one image's pooled patch features (h x w x d) and predicts, on a grid
`upsample` times finer, a person-centre logit, the centre's offset inside its
sub-cell, and the log of the box size in units of one full feature cell.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

PRIOR = 0.01  # initial centre probability, so early training is not swamped


class CentreHead(nn.Module):
    """LayerNorm, 1x1 then 3x3 convolution, and a sub-pixel output layer."""

    def __init__(self, dim: int = 768, hidden: int = 256, upsample: int = 4) -> None:
        super().__init__()
        self.upsample = upsample
        self.norm = nn.LayerNorm(dim)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        # 5 maps (centre, 2 offsets, 2 log sizes), each split into u*u sub-cells
        self.out = nn.Conv2d(hidden, 5 * upsample * upsample, 1)
        self.shuffle = nn.PixelShuffle(upsample)
        with torch.no_grad():
            bias = self.out.bias
            assert bias is not None
            bias.zero_()
            bias[: upsample * upsample] = -math.log((1 - PRIOR) / PRIOR)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """features (b, h, w, d) -> centre logits (b, H, W), offsets in [0, 1]
        (b, 2, H, W) and log sizes (b, 2, H, W), with H = u*h and W = u*w."""
        x = self.norm(features.float()).permute(0, 3, 1, 2)
        maps = self.shuffle(self.out(self.body(x)))
        return maps[:, 0], maps[:, 1:3].sigmoid(), maps[:, 3:5]


def centre_loss(
    outputs: tuple[Tensor, Tensor, Tensor], targets: dict[str, Tensor]
) -> Tensor:
    """CenterNet's loss with a one-cell peak per person.

    Focal loss on the centre map (alpha 2; with no Gaussian splat the
    beta term is 1 everywhere), plus L1 on the offset and on the log size at
    each person's sub-cell, all divided by the number of people in the batch.
    `targets` holds `heat` (b, H, W) in {0, 1}, `offset` and `size` (b, 2, H, W).
    """
    heat, offset, size = outputs
    target = targets["heat"].float()
    positive = target == 1
    count = positive.sum().clamp(min=1)
    p = heat.sigmoid()
    log_p = functional.logsigmoid(heat)
    log_not_p = functional.logsigmoid(-heat)
    focal = (
        -(torch.where(positive, (1 - p) ** 2 * log_p, p**2 * log_not_p)).sum() / count
    )
    mask = positive[:, None].expand_as(offset)
    off = (offset - targets["offset"].float()).abs()[mask].sum() / count
    siz = (size - targets["size"].float()).abs()[mask].sum() / count
    return focal + off + siz
