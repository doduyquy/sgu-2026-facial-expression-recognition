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

def apply_horizontal_flip_tta(model, images, bboxes=None, region_mask=None, region_confidence=None):
    """Run model with Horizontal Flip TTA."""
    # 1. Normal forward
    logits_normal = model(images, bboxes, region_mask, region_confidence)
    
    # 2. Flipped forward
    flipped_images = torch.flip(images, dims=[-1])
    
    flipped_bboxes = None
    swap_pairs = [(1, 2), (4, 5), (7, 8)]
    
    if bboxes is not None:
        flipped_bboxes = bboxes.clone()
        # x1_new = 47.0 - x2, x2_new = 47.0 - x1
        flipped_bboxes[:, :, 0] = 47.0 - bboxes[:, :, 2]
        flipped_bboxes[:, :, 2] = 47.0 - bboxes[:, :, 0]
        
        # Swap symmetric regions
        for i, j in swap_pairs:
            tmp = flipped_bboxes[:, i].clone()
            flipped_bboxes[:, i] = flipped_bboxes[:, j]
            flipped_bboxes[:, j] = tmp
            
    flipped_region_mask = None
    flipped_region_confidence = None
    if region_mask is not None:
        flipped_region_mask = region_mask.clone()
        for i, j in swap_pairs:
            tmp = flipped_region_mask[:, i].clone()
            flipped_region_mask[:, i] = flipped_region_mask[:, j]
            flipped_region_mask[:, j] = tmp
            
    if region_confidence is not None:
        flipped_region_confidence = region_confidence.clone()
        for i, j in swap_pairs:
            tmp = flipped_region_confidence[:, i].clone()
            flipped_region_confidence[:, i] = flipped_region_confidence[:, j]
            flipped_region_confidence[:, j] = tmp
            
    logits_flipped = model(flipped_images, flipped_bboxes, flipped_region_mask, flipped_region_confidence)
    
    # 3. Average
    result = {}
    for k in logits_normal:
        if isinstance(logits_normal[k], torch.Tensor) and torch.is_floating_point(logits_normal[k]):
            result[k] = (logits_normal[k] + logits_flipped[k]) / 2.0
        else:
            result[k] = logits_normal[k]
    return result
