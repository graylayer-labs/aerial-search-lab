"""Paired RGB–thermal contrastive learning experiment."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aerial_search.models.components import get_device, load_image, prepare_image
from aerial_search.models.ssl import ContrastiveEncoders


class PairedImages(Dataset[tuple[Tensor, Tensor]]):
    """RGB–thermal pairs described by a prepared JSONL manifest."""

    def __init__(self, manifest: Path, data_root: Path, *, augment: bool) -> None:
        self.records = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.data_root = data_root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        rgb = load_image(self.data_root / record["rgb_image"])
        thermal = load_image(self.data_root / record["thermal_image"])
        if self.augment and random.random() < 0.5:
            from torchvision.transforms import functional as vision

            rgb = vision.hflip(rgb)
            thermal = vision.hflip(thermal)
        return prepare_image(rgb), prepare_image(thermal)


@dataclass(frozen=True)
class ExperimentResult:
    """Metrics from one contrastive run."""

    device: str
    epochs: int
    train_pairs: int
    validation_pairs: int
    random_top1: float
    random_top5: float
    trained_top1: float
    trained_top5: float
    final_loss: float


def run_experiment(
    data_root: Path,
    manifests: Path,
    output: Path,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 7,
) -> ExperimentResult:
    """Train paired encoders and evaluate cross-modal retrieval."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    train_data = PairedImages(manifests / "train_100pct.jsonl", data_root, augment=True)
    validation_data = PairedImages(
        manifests / "validation.jsonl", data_root, augment=False
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    model = ContrastiveEncoders().to(device)
    random_top1, random_top5 = _retrieval(model, validation_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for rgb, thermal in train_loader:
            optimizer.zero_grad()
            rgb_embedding, thermal_embedding = model(rgb.to(device), thermal.to(device))
            loss = _contrastive_loss(rgb_embedding, thermal_embedding)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rgb)
        final_loss = total_loss / len(train_data)
        print(f"epoch {epoch + 1:02d}/{epochs}: loss={final_loss:.4f}", flush=True)

    trained_top1, trained_top5 = _retrieval(model, validation_loader, device)
    result = ExperimentResult(
        device=str(device),
        epochs=epochs,
        train_pairs=len(train_data),
        validation_pairs=len(validation_data),
        random_top1=random_top1,
        random_top5=random_top5,
        trained_top1=trained_top1,
        trained_top5=trained_top5,
        final_loss=final_loss,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def _contrastive_loss(rgb: Tensor, thermal: Tensor, temperature: float = 0.1) -> Tensor:
    logits = rgb @ thermal.T / temperature
    labels = torch.arange(len(rgb), device=rgb.device)
    return (
        functional.cross_entropy(logits, labels)
        + functional.cross_entropy(logits.T, labels)
    ) / 2


@torch.no_grad()
def _retrieval(
    model: ContrastiveEncoders,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    rgb_embeddings = []
    thermal_embeddings = []
    for rgb, thermal in loader:
        rgb_embedding, thermal_embedding = model(rgb.to(device), thermal.to(device))
        rgb_embeddings.append(rgb_embedding.cpu())
        thermal_embeddings.append(thermal_embedding.cpu())
    similarities = torch.cat(rgb_embeddings) @ torch.cat(thermal_embeddings).T
    targets = torch.arange(len(similarities)).unsqueeze(1)
    ranking = similarities.argsort(dim=1, descending=True)
    top1 = (ranking[:, :1] == targets).any(dim=1).float().mean().item()
    top5 = (ranking[:, :5] == targets).any(dim=1).float().mean().item()
    return top1, top5
