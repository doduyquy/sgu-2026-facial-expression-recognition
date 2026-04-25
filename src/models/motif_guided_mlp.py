"""Motif-guided image-level MLP over selected subgraph descriptors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    masked_scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_scores, dim=dim) * mask.to(dtype=scores.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights / denom


class MotifGuidedMLP(nn.Module):
    """
    Encode motif-selected subgraph descriptors and classify the whole image.

    Expected batch keys:
        x, mask, match_scores, matched_class, motif_score_vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3,
        use_motif_score_vector: bool = True,
        use_match_score_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.use_motif_score_vector = bool(use_motif_score_vector)
        self.use_match_score_weighting = bool(use_match_score_weighting)

        node_input_dim = self.input_dim + 1 + self.num_classes
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        final_dim = hidden_dim + (self.num_classes if self.use_motif_score_vector else 0)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _unpack(self, batch_or_x, **kwargs):
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            return (
                batch["x"],
                batch.get("mask"),
                batch.get("match_scores"),
                batch.get("matched_class"),
                batch.get("motif_score_vector"),
            )
        return (
            batch_or_x,
            kwargs.get("mask"),
            kwargs.get("match_scores"),
            kwargs.get("matched_class"),
            kwargs.get("motif_score_vector"),
        )

    def forward(self, batch_or_x, **kwargs) -> torch.Tensor:
        x, mask, match_scores, matched_class, motif_score_vector = self._unpack(batch_or_x, **kwargs)
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, K, D], got {tuple(x.shape)}")
        B, K, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        device = x.device
        if mask is None:
            mask = torch.ones(B, K, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device).bool()
        if match_scores is None:
            match_scores = torch.zeros(B, K, dtype=x.dtype, device=device)
        else:
            match_scores = match_scores.to(device=device, dtype=x.dtype)
        if matched_class is None:
            matched_class = torch.zeros(B, K, dtype=torch.long, device=device)
        else:
            matched_class = matched_class.to(device=device).long()
        if motif_score_vector is None:
            motif_score_vector = torch.zeros(B, self.num_classes, dtype=x.dtype, device=device)
        else:
            motif_score_vector = motif_score_vector.to(device=device, dtype=x.dtype)

        class_idx = matched_class.clamp(min=0, max=self.num_classes - 1)
        class_one_hot = F.one_hot(class_idx, num_classes=self.num_classes).to(dtype=x.dtype)
        class_one_hot = class_one_hot * mask.unsqueeze(-1).to(dtype=x.dtype)
        node_input = torch.cat([x, match_scores.unsqueeze(-1), class_one_hot], dim=-1)

        z = self.node_encoder(node_input)
        z = z * mask.unsqueeze(-1).to(dtype=z.dtype)

        if self.use_match_score_weighting:
            weights = _masked_softmax(match_scores, mask, dim=1).unsqueeze(-1)
            h = (z * weights).sum(dim=1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=z.dtype)
            h = (z * mask.unsqueeze(-1).to(dtype=z.dtype)).sum(dim=1) / denom

        if self.use_motif_score_vector:
            h = torch.cat([h, motif_score_vector], dim=-1)

        logits = self.classifier(h)
        return logits
