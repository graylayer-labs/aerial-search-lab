"""Person-centre localization on a grid (development benchmark)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aerial_search.models.components import get_device, load_image, prepare_image
from aerial_search.models.detection import GRID_SIZE, GridDetector

Modality = Literal["rgb", "thermal"]
Initialization = Literal["scratch", "ssl"]


class LocationGrids(Dataset[tuple[Tensor, Tensor]]):
    """Images paired with a grid containing person-centre locations."""

    def __init__(self, manifest: Path, data_root: Path, modality: Modality) -> None:
        self.records = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.data_root = data_root
        self.modality = modality

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        image = load_image(self.data_root / record[f"{self.modality}_image"])
        image = prepare_image(image, size=224)
        target = torch.zeros(GRID_SIZE, GRID_SIZE)
        for box in record[f"{self.modality}_boxes"]:
            column = min(int(box["x_center"] * GRID_SIZE), GRID_SIZE - 1)
            row = min(int(box["y_center"] * GRID_SIZE), GRID_SIZE - 1)
            target[row, column] = 1
        return image, target


@dataclass(frozen=True)
class DetectionResult:
    """Summary metrics for one localization run."""

    modality: str
    initialization: str
    device: str
    epochs: int
    train_images: int
    validation_images: int
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    final_loss: float


def run_detection_experiment(
    data_root: Path,
    manifests: Path,
    output: Path,
    *,
    modality: Modality,
    initialization: Initialization,
    ssl_checkpoint: Path | None = None,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    score_threshold: float = 0.5,
    seed: int = 7,
) -> DetectionResult:
    """Train and evaluate a lightweight person-centre locator."""
    torch.manual_seed(seed)
    device = get_device()
    train_data = LocationGrids(manifests / "train_100pct.jsonl", data_root, modality)
    validation_data = LocationGrids(manifests / "validation.jsonl", data_root, modality)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=16)
    model = GridDetector()
    if initialization == "ssl":
        if ssl_checkpoint is None:
            raise ValueError("SSL initialization requires a checkpoint")
        _load_ssl_backbone(model, ssl_checkpoint, modality)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positive_weight = torch.tensor(25.0, device=device)

    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(images.to(device))
            loss = functional.binary_cross_entropy_with_logits(
                predictions, targets.to(device), pos_weight=positive_weight
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(images)
        final_loss = running_loss / len(train_data)
        print(f"epoch {epoch + 1:02d}/{epochs}: loss={final_loss:.4f}", flush=True)

    true_positives, false_positives, false_negatives = _evaluate(
        model, validation_loader, device, score_threshold
    )
    result = DetectionResult(
        modality=modality,
        initialization=initialization,
        device=str(device),
        epochs=epochs,
        train_images=len(train_data),
        validation_images=len(validation_data),
        precision=true_positives / max(true_positives + false_positives, 1),
        recall=true_positives / max(true_positives + false_negatives, 1),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        final_loss=final_loss,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _load_ssl_backbone(
    model: GridDetector, checkpoint: Path, modality: Modality
) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = f"{modality}.0."
    backbone_state = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    incompatible = model.backbone.load_state_dict(backbone_state, strict=False)
    unexpected = incompatible.unexpected_keys
    if unexpected:
        raise ValueError(f"Unexpected SSL backbone keys: {unexpected}")


@torch.no_grad()
def _evaluate(
    model: GridDetector,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    score_threshold: float,
) -> tuple[int, int, int]:
    model.eval()
    true_positives = false_positives = false_negatives = 0
    for images, targets in loader:
        probabilities = model(images.to(device)).sigmoid().cpu()
        for probability, target in zip(probabilities, targets, strict=True):
            predicted = (probability >= score_threshold).nonzero()
            expected = target.nonzero()
            matched, missed = _match_locations(predicted, expected)
            true_positives += matched
            false_positives += len(predicted) - matched
            false_negatives += missed
    return true_positives, false_positives, false_negatives


def _match_locations(predicted: Tensor, expected: Tensor) -> tuple[int, int]:
    matched_expected: set[int] = set()
    matches = 0
    for location in predicted:
        distances = (expected - location).abs().amax(dim=1)
        for index in distances.argsort().tolist():
            if distances[index] <= 1 and index not in matched_expected:
                matched_expected.add(index)
                matches += 1
                break
    return matches, len(expected) - matches
