"""Validated tabular data, train-only normalization and deterministic loaders."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TableDataset(Dataset):
    def __init__(self, features, labels):
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float64)
        if features.ndim != 2 or 0 in features.shape or not np.isfinite(features).all():
            raise ValueError("features must be a nonempty finite 2D matrix")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must be a vector matching the number of samples")
        if (
            not np.isfinite(labels).all()
            or (labels < 0).any()
            or (labels >= 2**63).any()
            or (labels != np.floor(labels)).any()
        ):
            raise ValueError("labels must be finite nonnegative integers")
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_csv(cls, path: str | Path, target: str = "target") -> "TableDataset":
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            if target not in header or len(header) < 2 or len(set(header)) != len(header):
                raise ValueError("CSV requires unique feature columns and a target column")
            target_index = header.index(target)
            features, labels = [], []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError("CSV row length does not match header")
                try:
                    numbers = [float(cell) for cell in row]
                except ValueError as exc:
                    raise ValueError("CSV values must be numeric") from exc
                labels.append(numbers.pop(target_index))
                features.append(numbers)
        return cls(features, labels)


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "Standardizer":
        if features.ndim != 2 or 0 in features.shape or not torch.isfinite(features).all():
            raise ValueError("standardizer requires a nonempty finite 2D matrix")
        scale = features.std(dim=0, correction=0)
        return cls(features.mean(dim=0), torch.where(scale > 1e-6, scale, torch.ones_like(scale)))

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("features do not match the fitted standardizer")
        return (features - self.mean) / self.scale


@dataclass
class DataSplit:
    train: DataLoader
    valid: DataLoader
    scaler: Standardizer
    train_indices: list[int]
    valid_indices: list[int]


def build_loaders(
    dataset: TableDataset, batch_size: int = 32, valid_fraction: float = 0.2, seed: int = 42
) -> DataSplit:
    if batch_size <= 0 or not 0 < valid_fraction < 1 or seed < 0:
        raise ValueError("batch_size must be positive, valid_fraction in (0, 1), seed nonnegative")
    if len(dataset) < 2:
        raise ValueError("splitting requires at least two samples")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    valid_size = max(1, min(len(dataset) - 1, round(len(dataset) * valid_fraction)))
    valid_indices, train_indices = indices[:valid_size], indices[valid_size:]
    scaler = Standardizer.fit(dataset.features[train_indices])

    def subset_loader(selected, shuffle):
        subset = TableDataset(
            scaler.transform(dataset.features[selected]), dataset.labels[selected]
        )
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )

    return DataSplit(
        subset_loader(train_indices, True),
        subset_loader(valid_indices, False),
        scaler,
        train_indices,
        valid_indices,
    )


def make_classification(n_samples: int = 256, n_features: int = 4, seed: int = 42) -> TableDataset:
    """Generate a local binary task; no downloads or shared mutable RNG state."""
    if n_samples < 2 or n_features < 2 or seed < 0:
        raise ValueError("n_samples and n_features must be at least 2; seed must be nonnegative")
    features = np.random.default_rng(seed).normal(size=(n_samples, n_features)).astype(np.float32)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(np.int64)
    return TableDataset(features, labels)
