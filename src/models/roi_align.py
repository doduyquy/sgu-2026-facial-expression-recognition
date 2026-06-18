
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class SemanticRoiAlign(nn.Module):
    """ROIAlign over semantic regions (batch-aware)."""

    def __init__(self, roi_grid: int = 4, bbox_input_size: int = 48, feature_out_size: int = 12):
        super().__init__()
        self.roi_grid = int(roi_grid)
        self.bbox_input_size = int(bbox_input_size)
        self.feature_out_size = int(feature_out_size)

    @staticmethod
    def _canonical_region_boxes(bbox_input_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Fallback semantic ROIs for 9 regions in 48x48 space."""
        boxes = torch.tensor(
            [
                [8, 0, 40, 10],   # forehead
                [5, 8, 18, 18],   # left_eyebrow
                [30, 8, 43, 18],  # right_eyebrow
                [18, 12, 30, 22], # glabella
                [6, 16, 20, 30],  # left_eye
                [28, 16, 42, 30], # right_eye
                [14, 20, 34, 38], # nose
                [8, 30, 22, 43],  # left_mouth_corner
                [26, 30, 40, 43], # right_mouth_corner
            ],
            device=device,
            dtype=dtype,
        )
        scale = float(bbox_input_size) / 48.0
        return boxes * scale

    def validate_bboxes(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Clamp and repair invalid bbox coordinates while preserving batch/region count."""
        bboxes = bboxes.float().clone()
        bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0.0, float(self.bbox_input_size - 1))
        bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0.0, float(self.bbox_input_size - 1))

        x1 = torch.minimum(bboxes[..., 0], bboxes[..., 2])
        y1 = torch.minimum(bboxes[..., 1], bboxes[..., 3])
        x2 = torch.maximum(bboxes[..., 0], bboxes[..., 2])
        y2 = torch.maximum(bboxes[..., 1], bboxes[..., 3])

        x2 = torch.maximum(x2, x1 + 2.0)
        y2 = torch.maximum(y2, y1 + 2.0)

        x2 = torch.clamp(x2, max=float(self.bbox_input_size - 1))
        y2 = torch.clamp(y2, max=float(self.bbox_input_size - 1))
        x1 = torch.clamp(x1, max=float(self.bbox_input_size - 3))
        y1 = torch.clamp(y1, max=float(self.bbox_input_size - 3))

        repaired = torch.stack([x1, y1, x2, y2], dim=-1)
        too_small = ((repaired[..., 2] - repaired[..., 0]) < 2.0) | ((repaired[..., 3] - repaired[..., 1]) < 2.0)
        if too_small.any():
            repaired[too_small] = self._canonical_region_boxes(self.bbox_input_size, repaired.device, repaired.dtype)[None, :, :].expand_as(repaired)[too_small]
        return repaired

    def forward(self, feature_map: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        # feature_map: (B, C, H, W)
        # bboxes: (B, R, 4) in image coords (0..bbox_input_size-1)
        b, _, h, _ = feature_map.shape
        if bboxes.dim() != 3 or bboxes.size(-1) != 4:
            raise ValueError("bboxes must have shape (B, R, 4)")

        batch_size, num_regions, _ = bboxes.shape
        if batch_size != b:
            raise ValueError(f"bboxes batch {batch_size} does not match feature_map batch {b}")

        bboxes = self.validate_bboxes(bboxes)

        batch_indices = torch.arange(b, device=bboxes.device, dtype=bboxes.dtype).view(b, 1, 1)
        batch_indices = batch_indices.expand(b, num_regions, 1)
        rois = torch.cat([batch_indices, bboxes], dim=-1).reshape(-1, 5)

        # ROIAlign expects a single spatial_scale that maps input-image coordinates
        # to feature-map coordinates. For 48x48 inputs and 12x12 feature maps, this is 0.25.
        spatial_scale = float(h) / float(self.bbox_input_size)

        roi_features = roi_align(
            feature_map,
            rois,
            output_size=(self.roi_grid, self.roi_grid),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        # (B*R, C, G, G) -> (B, R, G*G, C)
        roi_features = roi_features.view(b, -1, feature_map.shape[1], self.roi_grid * self.roi_grid)
        roi_features = roi_features.permute(0, 1, 3, 2).contiguous()
        return roi_features
