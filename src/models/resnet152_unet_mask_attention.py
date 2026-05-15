"""
ResNet152 + U-Net / face-parsing mask-guided region attention.

This reuses the landmark-attention architecture but treats the masks as
precomputed U-Net / face-parsing soft region masks rather than dlib landmarks.
"""

import torch

from .resnet152_landmark_attention import (
    LandmarkGuidedAlignment,
    ResNet152LandmarkAttentionFER,
)


class UNetMaskGuidedAlignment(LandmarkGuidedAlignment):
    """
    Cross-attention with a tunable soft mask bias.

    The mask is injected before softmax as:
        attention_score += alpha * log(clamp(mask, floor, 1.0))
    """

    def __init__(
        self,
        embed_dim=512,
        visual_dim=1024,
        num_heads=4,
        dropout=0.1,
        mask_attention_alpha=0.5,
        mask_floor=0.05,
    ):
        super().__init__(
            embed_dim=embed_dim,
            visual_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mask_attention_alpha = float(mask_attention_alpha)
        self.mask_floor = float(mask_floor)
        if self.mask_attention_alpha < 0.0:
            raise ValueError("model.mask_attention_alpha must be >= 0.")
        if not 0.0 < self.mask_floor <= 1.0:
            raise ValueError("model.mask_floor must be in (0, 1].")

    def _build_log_mask(self, region_masks, num_heads):
        masks = region_masks.clamp(min=self.mask_floor, max=1.0)
        log_mask = self.mask_attention_alpha * torch.log(masks + 1e-6)
        return log_mask.repeat_interleave(num_heads, dim=0)


class ResNet152UNetMaskAttentionFER(ResNet152LandmarkAttentionFER):
    """
    Same training surface as landmark attention, but the mask source is
    precomputed U-Net / face-parsing region masks.
    """

    def __init__(self, config, channels=3):
        super().__init__(config=config, channels=channels)
        model_cfg = config.get("model", {})
        self.mask_attention_alpha = float(model_cfg.get("mask_attention_alpha", 0.5))
        self.mask_floor = float(model_cfg.get("mask_floor", 0.05))

        self.alignment = UNetMaskGuidedAlignment(
            embed_dim=self.embed_dim,
            visual_dim=self.visual_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            mask_attention_alpha=self.mask_attention_alpha,
            mask_floor=self.mask_floor,
        )
        print(
            "--> [UNetMaskAttention] mask-guided attention enabled: "
            f"alpha={self.mask_attention_alpha}, floor={self.mask_floor}"
        )

