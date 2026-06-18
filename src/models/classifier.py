

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SemanticStateEncoder(nn.Module):
    """Project region embeddings into interpretable semantic facial state space."""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim // 2, state_dim * 2)
        self.state_dim = state_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, region_embeddings: torch.Tensor) -> torch.Tensor:
        raw_state = self.proj(region_embeddings)
        gate = self.gate(region_embeddings)
        # Fix 2: pure gating — gate actually controls information flow.
        # Original `raw_state * gate + raw_state` = `raw_state * (gate + 1)`,
        # making the Sigmoid gate a mere scaling factor with no off-switch.
        semantic_state = self.norm(raw_state * gate)
        return semantic_state


class SemanticEmotionClassifier(nn.Module):
    """Classify emotion from semantic latent facial representation."""

    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
