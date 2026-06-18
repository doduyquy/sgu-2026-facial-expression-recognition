"""
utils.py — Shared helper functions for the semantic_roi_graph package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """A numerically stable softmax that prevents NaN when vectors are fully masked."""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    # Handle the case where x was all -inf (which results in NaN after subtraction)
    # or if the user used a very large negative number (like -1e9) which resolves to 0.
    all_invalid = torch.isinf(x_shifted).all(dim=dim, keepdim=True) | torch.isnan(x_shifted).all(dim=dim, keepdim=True)
    x_shifted = torch.where(all_invalid, torch.zeros_like(x_shifted), x_shifted)
    return F.softmax(x_shifted, dim=dim)
