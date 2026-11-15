"""Feature tokenization, attention and classification logits."""

import torch
from torch import nn

from ml_pipeline.blocks import AttentionBlock


class TabularClassifier(nn.Module):
    def __init__(
        self, num_features: int, embed_dim: int = 8, num_heads: int = 2, num_classes: int = 2
    ):
        super().__init__()
        if num_features <= 0 or num_classes < 2:
            raise ValueError("num_features must be positive and num_classes must be at least 2")
        self.num_features = num_features
        self.block = AttentionBlock(embed_dim, num_heads)
        self.feature_weight = nn.Parameter(torch.empty(num_features, embed_dim))
        self.feature_bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        nn.init.normal_(self.feature_weight, std=0.2)
        self.head = nn.Linear(num_features * embed_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError(f"features must have shape [batch, {self.num_features}]")
        tokens = features.unsqueeze(-1) * self.feature_weight + self.feature_bias
        return self.head(self.block(tokens).flatten(start_dim=1))
