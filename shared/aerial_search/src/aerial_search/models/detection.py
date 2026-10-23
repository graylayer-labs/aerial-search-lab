"""Person-localization detector on a grid."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import resnet18

GRID_SIZE = 14


class GridDetector(nn.Module):
    """ResNet-18 features with a person-centre grid head."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.backbone = backbone
        self.head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone.conv1(images)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        return self.head(features).squeeze(1)
