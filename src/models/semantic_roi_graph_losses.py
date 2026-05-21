"""Standalone loss helpers for Semantic ROI Graph FER."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def motif_diversity_loss(motifs: torch.Tensor) -> torch.Tensor:
    # motifs: (C, M, R, D)
    c, m, r, d = motifs.shape
    motifs = motifs.view(c, m, r * d)
    motifs = F.normalize(motifs, dim=-1)
    sim = torch.einsum("cmd,cnd->cmn", motifs, motifs)
    identity = torch.eye(m, device=sim.device).unsqueeze(0)
    off_diag = sim * (1.0 - identity)
    return (off_diag ** 2).mean()


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # embeddings: (B, R, D)
    pooled = embeddings.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)
    sim = torch.matmul(pooled, pooled.t()) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()


def region_consistency_loss(region_embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # region_embeddings: (B, R, D)
    labels = labels.view(-1)
    loss = 0.0
    count = 0
    for cls in labels.unique():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_embeddings = region_embeddings[mask]
        mean = cls_embeddings.mean(dim=0, keepdim=True)
        loss = loss + ((cls_embeddings - mean) ** 2).mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=region_embeddings.device)
    return loss / count
