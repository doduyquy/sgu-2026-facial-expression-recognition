"""
Baseline model for bags of subgraph descriptors.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SubgraphMLPBaseline(nn.Module):
    """
    descriptor -> shared MLP encoder -> masked mean pooling -> classifier

    Input:
        x    : [B, K, D]
        mask : [B, K] or None
    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        dims = [input_dim] + list(hidden_dims)
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, K, D], got {tuple(x.shape)}")

        z = self.encoder(x)

        if mask is None:
            h_img = z.mean(dim=1)
        else:
            if mask.ndim != 2:
                raise ValueError(f"Expected mask to have shape [B, K], got {tuple(mask.shape)}")
            mask = mask.to(dtype=z.dtype).unsqueeze(-1)
            masked_sum = (z * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            h_img = masked_sum / denom

        logits = self.classifier(h_img)
        return logits
