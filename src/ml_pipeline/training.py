"""Training, weighted evaluation metrics and portable CPU checkpoints."""

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline.data import Standardizer, TableDataset, build_loaders
from ml_pipeline.models import TabularClassifier


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.01
    valid_fraction: float = 0.2
    seed: int = 42
    embed_dim: int = 8
    num_heads: int = 2
    num_classes: int = 2

    def __post_init__(self):
        if self.epochs <= 0 or self.batch_size <= 0 or self.seed < 0:
            raise ValueError("epochs and batch_size must be positive; seed must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0 < self.valid_fraction < 1:
            raise ValueError("valid_fraction must be in (0, 1)")
        if self.embed_dim <= 0 or self.num_heads <= 0 or self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be positive and divisible by positive num_heads")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")


@dataclass
class TrainingResult:
    model: TabularClassifier
    scaler: Standardizer
    config: TrainConfig
    history: list[dict[str, float | int]]
    initial_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("training loss is not finite")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    if not total:
        raise ValueError("training loader is empty")
    return total_loss / total


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    try:
        with torch.inference_mode():
            for features, labels in loader:
                logits = model(features)
                total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)
    finally:
        model.train(was_training)
    if not total:
        raise ValueError("evaluation loader is empty")
    return {"loss": total_loss / total, "accuracy": correct / total}


def _validate_classes(dataset: TableDataset, config: TrainConfig) -> None:
    if dataset.labels.max().item() >= config.num_classes:
        raise ValueError("dataset labels exceed the configured number of classes")


def run_training(dataset: TableDataset, config: TrainConfig | None = None) -> TrainingResult:
    config = config or TrainConfig()
    _validate_classes(dataset, config)
    seed_everything(config.seed)
    split = build_loaders(dataset, config.batch_size, config.valid_fraction, config.seed)
    model = TabularClassifier(
        dataset.features.shape[1], config.embed_dim, config.num_heads, config.num_classes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_loss = evaluate_model(model, split.valid)["loss"]
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, split.train, optimizer)
        valid = evaluate_model(model, split.valid)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid["loss"],
                "valid_accuracy": valid["accuracy"],
            }
        )
    model.eval()
    return TrainingResult(model, split.scaler, config, history, initial_loss)


def predict(result: TrainingResult, features) -> torch.Tensor:
    features = torch.as_tensor(features, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("prediction features must be finite")
    result.model.eval()
    with torch.inference_mode():
        return result.model(result.scaler.transform(features)).argmax(dim=-1)


def evaluate(result: TrainingResult, dataset: TableDataset) -> dict[str, float]:
    _validate_classes(dataset, result.config)
    normalized = result.scaler.transform(dataset.features)
    loader = DataLoader(
        TensorDataset(normalized, dataset.labels), batch_size=result.config.batch_size
    )
    return evaluate_model(result.model, loader)


def save_checkpoint(result: TrainingResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "num_features": result.model.num_features,
            "config": asdict(result.config),
            "model_state": result.model.state_dict(),
            "scaler_mean": result.scaler.mean,
            "scaler_scale": result.scaler.scale,
            "history": result.history,
            "initial_loss": result.initial_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> TrainingResult:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format version")
    config = TrainConfig(**checkpoint["config"])
    model = TabularClassifier(
        checkpoint["num_features"], config.embed_dim, config.num_heads, config.num_classes
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return TrainingResult(
        model,
        Standardizer(checkpoint["scaler_mean"], checkpoint["scaler_scale"]),
        config,
        checkpoint["history"],
        checkpoint["initial_loss"],
    )
