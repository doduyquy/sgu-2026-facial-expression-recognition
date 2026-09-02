"""
utils.py — Shared helper functions for the semantic_roi_graph package.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F


def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """A numerically stable softmax that prevents NaN when vectors are fully masked or have large values."""
    x_max = x.max(dim=dim, keepdim=True)[0].detach()
    x_shifted = torch.clamp(x - x_max, min=-50.0, max=0.0)
    exp_x = torch.exp(x_shifted)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return exp_x / denom


def _flip_bboxes(bboxes: torch.Tensor, width: float = 48.0) -> torch.Tensor:
    flipped_bboxes = bboxes.clone()
    flipped_bboxes[..., 0] = (width - 1.0) - bboxes[..., 2]
    flipped_bboxes[..., 2] = (width - 1.0) - bboxes[..., 0]
    swap_pairs = [(1, 2), (4, 5), (7, 8)]
    for i, j in swap_pairs:
        tmp = flipped_bboxes[:, i].clone()
        flipped_bboxes[:, i] = flipped_bboxes[:, j]
        flipped_bboxes[:, j] = tmp
    return flipped_bboxes


def _flip_mask_or_conf(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    flipped = tensor.clone()
    swap_pairs = [(1, 2), (4, 5), (7, 8)]
    for i, j in swap_pairs:
        tmp = flipped[:, i].clone()
        flipped[:, i] = flipped[:, j]
        flipped[:, j] = tmp
    return flipped


def _forward_model(model: Any, images: torch.Tensor, bboxes: Optional[torch.Tensor] = None,
                   region_mask: Optional[torch.Tensor] = None, region_confidence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if hasattr(model, "_forward_single"):
        return model._forward_single(images, bboxes, region_mask, region_confidence)
    return model(images, bboxes, region_mask=region_mask, region_confidence=region_confidence)


def apply_multi_scale_tta(
    model: Any,
    images: torch.Tensor,
    bboxes: Optional[torch.Tensor] = None,
    region_mask: Optional[torch.Tensor] = None,
    region_confidence: Optional[torch.Tensor] = None,
    scales: Tuple[float, ...] = (1.0, 1.05, 0.95),
) -> Dict[str, torch.Tensor]:
    """
    Multi-Scale & Multi-View Test-Time Augmentation (TTA) with Softmax Probability Averaging.
    Generates original + flipped + scaled views with synchronized ROI bbox transforms.
    """
    views_images = []
    views_bboxes = []
    views_masks = []
    views_confs = []

    B, C, H, W = images.shape

    for s in scales:
        if s == 1.0:
            img_s = images
            bbox_s = bboxes
        elif s > 1.0:
            # Zoom in: crop center and resize back to (H, W)
            crop_h, crop_w = int(round(H / s)), int(round(W / s))
            off_h, off_w = (H - crop_h) / 2.0, (W - crop_w) / 2.0
            start_y, start_x = int(round(off_h)), int(round(off_w))
            img_cropped = images[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]
            img_s = F.interpolate(img_cropped, size=(H, W), mode='bilinear', align_corners=False)
            if bboxes is not None:
                bbox_s = bboxes.clone()
                bbox_s[..., [0, 2]] = (bbox_s[..., [0, 2]] - off_w) * (float(W) / float(crop_w))
                bbox_s[..., [1, 3]] = (bbox_s[..., [1, 3]] - off_h) * (float(H) / float(crop_h))
                bbox_s[..., 0::2] = bbox_s[..., 0::2].clamp(0.0, float(W - 1.0))
                bbox_s[..., 1::2] = bbox_s[..., 1::2].clamp(0.0, float(H - 1.0))
            else:
                bbox_s = None
        else:
            # Zoom out: resize smaller and pad to (H, W)
            target_h, target_w = int(round(H * s)), int(round(W * s))
            pad_h = (H - target_h) / 2.0
            pad_w = (W - target_w) / 2.0
            img_resized = F.interpolate(images, size=(target_h, target_w), mode='bilinear', align_corners=False)
            pad_left = int(round(pad_w))
            pad_right = W - target_w - pad_left
            pad_top = int(round(pad_h))
            pad_bottom = H - target_h - pad_top
            img_s = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom))
            if bboxes is not None:
                bbox_s = bboxes.clone()
                bbox_s[..., [0, 2]] = bbox_s[..., [0, 2]] * s + pad_left
                bbox_s[..., [1, 3]] = bbox_s[..., [1, 3]] * s + pad_top
                bbox_s[..., 0::2] = bbox_s[..., 0::2].clamp(0.0, float(W - 1.0))
                bbox_s[..., 1::2] = bbox_s[..., 1::2].clamp(0.0, float(H - 1.0))
            else:
                bbox_s = None

        # View A: Standard orientation
        views_images.append(img_s)
        views_bboxes.append(bbox_s)
        views_masks.append(region_mask)
        views_confs.append(region_confidence)

        # View B: Horizontally Flipped
        img_flipped = torch.flip(img_s, dims=[-1])
        bbox_flipped = _flip_bboxes(bbox_s, width=float(W)) if bbox_s is not None else None
        mask_flipped = _flip_mask_or_conf(region_mask)
        conf_flipped = _flip_mask_or_conf(region_confidence)

        views_images.append(img_flipped)
        views_bboxes.append(bbox_flipped)
        views_masks.append(mask_flipped)
        views_confs.append(conf_flipped)

    # Forward all views and aggregate probabilities via Softmax soft-voting
    prob_list = []
    fused_prob_list = []
    motif_prob_list = []

    for img_v, box_v, mask_v, conf_v in zip(views_images, views_bboxes, views_masks, views_confs):
        out_v = _forward_model(model, img_v, box_v, mask_v, conf_v)
        logits_v = out_v["logits"] if isinstance(out_v, dict) else out_v
        prob_list.append(F.softmax(logits_v, dim=-1))

        if isinstance(out_v, dict) and "logits_fused" in out_v:
            fused_prob_list.append(F.softmax(out_v["logits_fused"], dim=-1))
        if isinstance(out_v, dict) and "logits_motif" in out_v:
            motif_prob_list.append(F.softmax(out_v["logits_motif"], dim=-1))

    avg_probs = torch.stack(prob_list, dim=0).mean(dim=0)
    final_logits = torch.log(avg_probs.clamp_min(1e-7))

    result = {"logits": final_logits}
    if fused_prob_list:
        avg_fused = torch.stack(fused_prob_list, dim=0).mean(dim=0)
        result["logits_fused"] = torch.log(avg_fused.clamp_min(1e-7))
    if motif_prob_list:
        avg_motif = torch.stack(motif_prob_list, dim=0).mean(dim=0)
        result["logits_motif"] = torch.log(avg_motif.clamp_min(1e-7))

    return result


def apply_horizontal_flip_tta(
    model: Any,
    images: torch.Tensor,
    bboxes: Optional[torch.Tensor] = None,
    region_mask: Optional[torch.Tensor] = None,
    region_confidence: Optional[torch.Tensor] = None,
    use_multi_scale: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Standard evaluation TTA function. Defaults to high-precision 6-view Multi-Scale TTA.
    """
    if use_multi_scale:
        return apply_multi_scale_tta(model, images, bboxes, region_mask, region_confidence)
    return apply_multi_scale_tta(model, images, bboxes, region_mask, region_confidence, scales=(1.0,))