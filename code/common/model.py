"""Shared MLP model definition."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            hidden = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the representation immediately before the final linear layer."""
        layers = list(self.net.children())
        if not layers:
            return x
        for layer in layers[:-1]:
            x = layer(x)
        return x
