"""
utils.py — Shared helper functions for the semantic_roi_graph package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """A numerically stable softmax that prevents NaN when vectors are fully masked or have large values."""
    x_max = x.max(dim=dim, keepdim=True)[0].detach()
    x_max = torch.where(torch.isneginf(x_max) | torch.isnan(x_max), torch.zeros_like(x_max), x_max)
    x_shifted = (x - x_max).clamp(min=-50.0, max=0.0)
    exp_x = torch.exp(x_shifted)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return exp_x / denom

def apply_multi_scale_tta(model, images, bboxes=None, region_mask=None, region_confidence=None, scale=1.05):
    """Run model with Multi-scale TTA: Original, Flipped, Scaled (1.05x)+Flipped."""
    # 1. Normal forward
    logits_normal = model(images, bboxes, region_mask, region_confidence)
    
    swap_pairs = [(1, 2), (4, 5), (7, 8)]
    
    def flip_bboxes_fn(boxes):
        if boxes is None: return None
        fb = boxes.clone()
        fb[:, :, 0] = 47.0 - boxes[:, :, 2]
        fb[:, :, 2] = 47.0 - boxes[:, :, 0]
        for i, j in swap_pairs:
            tmp = fb[:, i].clone()
            fb[:, i] = fb[:, j]
            fb[:, j] = tmp
        return fb
        
    def flip_meta_fn(meta):
        if meta is None: return None
        fm = meta.clone()
        for i, j in swap_pairs:
            tmp = fm[:, i].clone()
            fm[:, i] = fm[:, j]
            fm[:, j] = tmp
        return fm
        
    # 2. Flipped forward
    flipped_images = torch.flip(images, dims=[-1])
    flipped_bboxes = flip_bboxes_fn(bboxes)
    flipped_region_mask = flip_meta_fn(region_mask)
    flipped_region_confidence = flip_meta_fn(region_confidence)
    
    logits_flipped = model(flipped_images, flipped_bboxes, flipped_region_mask, flipped_region_confidence)
    
    # 3. Scaled (1.05x) and Flipped forward
    h, w = images.shape[2:]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled_images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    top = (new_h - h) // 2
    left = (new_w - w) // 2
    scaled_images = scaled_images[:, :, top:top+h, left:left+w]
    
    scaled_flipped_images = torch.flip(scaled_images, dims=[-1])
    
    scaled_flipped_bboxes = None
    if bboxes is not None:
        scaled_bboxes = bboxes.clone()
        scaled_bboxes[:, :, 0] = bboxes[:, :, 0] * (new_w / w) - left
        scaled_bboxes[:, :, 2] = bboxes[:, :, 2] * (new_w / w) - left
        scaled_bboxes[:, :, 1] = bboxes[:, :, 1] * (new_h / h) - top
        scaled_bboxes[:, :, 3] = bboxes[:, :, 3] * (new_h / h) - top
        
        scaled_bboxes[:, :, [0, 2]] = torch.clamp(scaled_bboxes[:, :, [0, 2]], 0.0, float(w - 1))
        scaled_bboxes[:, :, [1, 3]] = torch.clamp(scaled_bboxes[:, :, [1, 3]], 0.0, float(h - 1))
        
        scaled_flipped_bboxes = flip_bboxes_fn(scaled_bboxes)
        
    logits_scaled_flipped = model(scaled_flipped_images, scaled_flipped_bboxes, flipped_region_mask, flipped_region_confidence)
    
    # 4. Average 3 predictions
    result = {}
    for k in logits_normal:
        if isinstance(logits_normal[k], torch.Tensor) and torch.is_floating_point(logits_normal[k]):
            result[k] = (logits_normal[k] + logits_flipped[k] + logits_scaled_flipped[k]) / 3.0
        else:
            result[k] = logits_normal[k]
    return result