"""Scoring helpers for prototype motif mining and matching."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def check_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise a clear error if a tensor contains NaN or Inf."""
    if tensor is None:
        return
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).sum().item()
        raise ValueError(f"{name} contains {bad} NaN/Inf values")


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the last dimension."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    check_finite_tensor("l2_normalize.input", x)
    return F.normalize(x, p=2, dim=-1, eps=eps)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between all rows of a and b.

    Args:
        a: Tensor [N, D]
        b: Tensor [M, D]

    Returns:
        Tensor [N, M]
    """
    if not torch.is_tensor(a):
        a = torch.as_tensor(a)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got a={tuple(a.shape)}, b={tuple(b.shape)}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Descriptor dims differ: a={a.shape[1]} vs b={b.shape[1]}")

    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    sim = a_n @ b_n.t()
    check_finite_tensor("cosine_similarity_matrix.output", sim)
    return sim


def _top_fraction_mean(values: torch.Tensor, top_fraction: float = 0.2) -> float:
    if values.numel() == 0:
        return 0.0
    top_fraction = float(top_fraction)
    if top_fraction <= 0 or top_fraction > 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    k = max(1, int(math.ceil(values.numel() * top_fraction)))
    top_values = torch.topk(values.flatten(), k=k, largest=True).values
    return float(top_values.mean().item())


def compute_intra_score(
    prototype: torch.Tensor,
    class_descriptors: torch.Tensor,
    top_fraction: float = 0.2,
) -> float:
    """Mean top-p cosine similarity between prototype and descriptors of its class."""
    if class_descriptors.numel() == 0:
        return 0.0
    proto = torch.as_tensor(prototype).float().view(1, -1)
    desc = torch.as_tensor(class_descriptors).float()
    sim = cosine_similarity_matrix(proto, desc).squeeze(0)
    return _top_fraction_mean(sim, top_fraction=top_fraction)


def compute_inter_score(
    prototype: torch.Tensor,
    other_descriptors: torch.Tensor,
    top_fraction: float = 0.2,
) -> float:
    """Mean top-p cosine similarity between prototype and descriptors from other classes."""
    if other_descriptors.numel() == 0:
        return 0.0
    proto = torch.as_tensor(prototype).float().view(1, -1)
    desc = torch.as_tensor(other_descriptors).float()
    sim = cosine_similarity_matrix(proto, desc).squeeze(0)
    return _top_fraction_mean(sim, top_fraction=top_fraction)


def compute_discriminative_score(intra: float, inter: float, alpha: float = 0.5) -> float:
    """Class-discriminative motif score."""
    return float(intra) - float(alpha) * float(inter)
