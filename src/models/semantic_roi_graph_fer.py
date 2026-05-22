"""
Semantic ROI Graph FER model (2-tier: micro + macro) without ArcFace.

This module implements:
- SemanticBackbone (ResNet18-based, high-res feature map)
- SemanticRoIAlign (ROIAlign per region)
- MicroGraphReasoner (intra-region graph attention + pooling)
- MacroGraphReasoner (inter-region graph attention)
- Learnable Semantic Motif Bank + matcher
- Global branch + adaptive fusion
- Optional losses (motif diversity, supervised contrastive, region consistency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align

# Loss helpers moved to src/models/semantic_roi_graph_losses.py

DEFAULT_SEMANTIC_REGIONS = (
    "forehead",
    "left_eyebrow",
    "right_eyebrow",
    "glabella",
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth_corner",
    "right_mouth_corner",
)


@dataclass
class SemanticRoiGraphConfig:
    name: str = "semantic_roi_graph_fer"
    num_classes: int = 7
    num_regions: int = 9
    name: str = "semantic_roi_graph_fer"
    roi_grid: int = 4
    feature_dim: int = 256
    motif_per_class: int = 4
    micro_motifs_per_region: int = 8
    macro_motifs_per_class: int = 4
    use_pretrained: bool = True
    backbone_out_size: int = 12
    bbox_input_size: int = 48
    micro_layers: int = 2
    macro_layers: int = 2
    attn_heads: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.0
    relation_temperature: float = 0.07
    region_confidence_threshold: float = 0.3


class SemanticBackbone(nn.Module):
    """ResNet18 backbone with high spatial resolution output."""

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if use_pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)

        # Keep high resolution by removing early downsampling.
        resnet.conv1.stride = (1, 1)
        resnet.maxpool = nn.Identity()

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 48 -> 48
        self.layer2 = resnet.layer2  # 48 -> 24
        self.layer3 = resnet.layer3  # 24 -> 12
        self.layer4 = resnet.layer4  # 12 -> 6

        if feature_dim == 256:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3)
            self.out_channels = 256
            self.use_layer4 = False
        elif feature_dim == 512:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
            self.out_channels = 512
            self.use_layer4 = True
        else:
            raise ValueError("feature_dim must be 256 or 512")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.output_layer(x)
        if self.use_layer4:
            x = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)
        return x


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


class GATBlock(nn.Module):
    """Multi-head graph attention with learnable adjacency bias and locality prior."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        num_nodes: Optional[int] = None,
        use_locality: bool = False,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.adj_bias = None
        if num_nodes is not None:
            self.adj_bias = nn.Parameter(torch.zeros(1, 1, num_nodes, num_nodes))
            nn.init.normal_(self.adj_bias, mean=0.0, std=0.01)

        self.locality_bias = None
        if use_locality and num_nodes is not None:
            side = int(num_nodes ** 0.5)
            if side * side == num_nodes:
                coords_1d = torch.arange(side, dtype=torch.float32)
                grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
                coords = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)
            else:
                coords = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(-1)
            dist = torch.cdist(coords, coords)
            dist = dist / (dist.max().clamp(min=1e-6))
            self.register_buffer("locality_bias", -dist.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_prior: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, N, D)
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim ** 0.5)
        if self.adj_bias is not None:
            attn = attn + self.adj_bias
        if self.locality_bias is not None:
            attn = attn + self.locality_bias
        if edge_prior is not None:
            if edge_prior.dim() == 2:
                edge_prior = edge_prior.unsqueeze(0)
            if edge_prior.size(0) == 1 and b > 1:
                edge_prior = edge_prior.expand(b, -1, -1)
            edge_prior = edge_prior.clamp_min(1e-6)
            attn = attn + torch.log(edge_prior).unsqueeze(1)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        return out


class GatedPooling(nn.Module):
    """Attention-based gated pooling."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        weights = torch.sigmoid(self.gate(x))
        weighted = x * weights
        pooled = weighted.sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return pooled


class MicroGraphReasoner(nn.Module):
    """Intra-region reasoning with graph attention."""

    def __init__(self, dim: int, num_nodes: int, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATBlock(dim, heads=heads, dropout=dropout, num_nodes=num_nodes) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.pool = GatedPooling(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, R, N, D)
        b, r, n, d = x.shape
        x = x.view(b * r, n, d)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        pooled = self.pool(x).view(b, r, d)
        x = x.view(b, r, n, d)
        return x, pooled


class MacroGraphReasoner(nn.Module):
    """Inter-region reasoning across semantic nodes."""

    def __init__(self, dim: int, num_regions: int, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATBlock(dim, heads=heads, dropout=dropout, num_nodes=num_regions) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, R, D)
        if region_confidence is not None:
            x = x * region_confidence.unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x), edge_prior=adj, attn_mask=region_mask)
        return x


class MicroMotifBank(nn.Module):
    """Learnable micro motifs per semantic region."""

    def __init__(self, num_regions: int, motifs_per_region: int, dim: int):
        super().__init__()
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.dim = dim
        self.motifs = nn.Parameter(torch.randn(num_regions, motifs_per_region, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.motifs


class MicroMotifMatcher(nn.Module):
    """Match region embeddings to region-specific micro motifs."""

    def __init__(self, num_regions: int, motifs_per_region: int, dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.dim = dim
        self.temperature = float(temperature)
        self.token_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, region_embeddings: torch.Tensor, motif_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # region_embeddings: (B, R, D)
        # motif_bank: (R, K, D)
        region_norm = F.normalize(region_embeddings, dim=-1)
        bank_norm = F.normalize(motif_bank, dim=-1)

        sim = torch.einsum("brd,rkd->brk", region_norm, bank_norm) / self.temperature
        attn = F.softmax(sim, dim=-1)
        tokens = torch.einsum("brk,rkd->brd", attn, motif_bank)
        tokens = self.token_proj(tokens)

        gate = torch.sigmoid(self.fusion_gate)
        region_motif_tokens = region_embeddings + gate * tokens
        return attn, region_motif_tokens


class MacroMotifBank(nn.Module):
    """Learnable macro motifs per emotion class."""

    def __init__(self, num_classes: int, motifs_per_class: int, num_regions: int, dim: int):
        super().__init__()
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, num_regions, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.motifs


class MacroMotifMatcher(nn.Module):
    """Match macro graph embeddings to class-level topology motifs."""

    def __init__(self, num_classes: int, motifs_per_class: int, num_regions: int, dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_regions = num_regions
        self.dim = dim
        self.temperature = float(temperature)

    @staticmethod
    def relation_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (..., R, D) -> (..., R, R)
        embeddings = F.normalize(embeddings, dim=-1)
        return torch.einsum("...id,...jd->...ij", embeddings, embeddings)

    def forward(self, macro_embeddings: torch.Tensor, motif_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # macro_embeddings: (B, R, D)
        # motif_bank: (C, M, R, D)
        rel_macro = self.relation_matrix(macro_embeddings)  # (B, R, R)
        rel_motif = self.relation_matrix(motif_bank)  # (C, M, R, R)

        rel_macro_flat = F.normalize(rel_macro.reshape(rel_macro.shape[0], -1), dim=-1)
        rel_motif_flat = F.normalize(rel_motif.reshape(rel_motif.shape[0], rel_motif.shape[1], -1), dim=-1)

        sim = torch.einsum("bd,cmd->bcm", rel_macro_flat, rel_motif_flat) / self.temperature
        attn = F.softmax(sim, dim=-1)
        logits_motif = (attn * sim).sum(dim=-1)
        return logits_motif, attn, rel_macro


# Backward-compatible aliases for older code paths.
SemanticMotifBank = MacroMotifBank
SemanticMotifMatcher = MacroMotifMatcher


class SemanticROIGraphFER(nn.Module):
    """End-to-end semantic ROI graph FER model without ArcFace."""

    def __init__(self, config: SemanticRoiGraphConfig):
        super().__init__()
        self.config = config

        self.backbone = SemanticBackbone(
            feature_dim=config.feature_dim,
            use_pretrained=config.use_pretrained,
        )
        self.roi_align = SemanticRoiAlign(
            roi_grid=config.roi_grid,
            bbox_input_size=config.bbox_input_size,
            feature_out_size=config.backbone_out_size,
        )

        self.micro_reasoner = MicroGraphReasoner(
            dim=config.feature_dim,
            num_nodes=config.roi_grid * config.roi_grid,
            layers=config.micro_layers,
            heads=config.attn_heads,
            dropout=config.dropout,
        )

        self.region_proj = nn.Linear(config.feature_dim, config.feature_dim)

        self.micro_motif_bank = MicroMotifBank(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            dim=config.feature_dim,
        )
        self.micro_motif_matcher = MicroMotifMatcher(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            dim=config.feature_dim,
            temperature=config.relation_temperature,
        )

        self.macro_reasoner = MacroGraphReasoner(
            dim=config.feature_dim,
            num_regions=config.num_regions,
            layers=config.macro_layers,
            heads=config.attn_heads,
            dropout=config.dropout,
        )

        self.macro_motif_bank = MacroMotifBank(
            num_classes=config.num_classes,
            motifs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            dim=config.feature_dim,
        )
        self.macro_motif_matcher = MacroMotifMatcher(
            num_classes=config.num_classes,
            motifs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            dim=config.feature_dim,
            temperature=config.relation_temperature,
        )

        # Backward-compatible aliases for older checkpoints and callers.
        self.motif_bank = self.macro_motif_bank
        self.motif_matcher = self.macro_motif_matcher

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.num_classes),
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.edge_importance = nn.Parameter(torch.eye(config.num_regions))
        nn.init.eye_(self.edge_importance)
        self.missing_region_token = nn.Parameter(torch.randn(config.feature_dim) * 0.02)
        self.region_reliability_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.region_dropout_prob = 0.05

    def _canonical_bboxes(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        boxes = SemanticRoiAlign._canonical_region_boxes(self.config.bbox_input_size, device, dtype)
        return boxes.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _prepare_regions(
        self,
        bboxes: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return repaired boxes, region mask, confidence and invalid indices."""
        if bboxes is None:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.ones(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.full_like(region_mask, 0.95)
            invalid_indices = torch.empty((0, 2), device=device, dtype=torch.long)
            return repaired, region_mask, region_confidence, invalid_indices

        bboxes = bboxes.to(device=device, dtype=dtype)
        if bboxes.dim() != 3 or bboxes.size(-1) != 4:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.zeros(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.zeros_like(region_mask)
            invalid_indices = torch.nonzero(torch.ones_like(region_mask, dtype=torch.bool), as_tuple=False)
            return repaired, region_mask, region_confidence, invalid_indices

        valid_shape = bboxes.size(0) == batch_size and bboxes.size(1) == self.config.num_regions
        if not valid_shape:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.zeros(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.zeros_like(region_mask)
            invalid_indices = torch.nonzero(torch.ones_like(region_mask, dtype=torch.bool), as_tuple=False)
            return repaired, region_mask, region_confidence, invalid_indices

        finite_mask = torch.isfinite(bboxes).all(dim=-1)
        x1 = bboxes[..., 0]
        y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]
        y2 = bboxes[..., 3]
        size_mask = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
        order_mask = (x2 > x1) & (y2 > y1)
        region_mask = (finite_mask & size_mask & order_mask).to(dtype=dtype)

        repaired = self.roi_align.validate_bboxes(bboxes)
        canonical = self._canonical_bboxes(batch_size, device, dtype)
        repaired = torch.where(region_mask.unsqueeze(-1).bool(), repaired, canonical)

        width = (repaired[..., 2] - repaired[..., 0]).clamp(min=1.0)
        height = (repaired[..., 3] - repaired[..., 1]).clamp(min=1.0)
        area = (width * height) / float(self.config.bbox_input_size * self.config.bbox_input_size)
        area_conf = area.clamp(0.0, 1.0)
        region_confidence = torch.where(region_mask > 0, 0.5 + 0.5 * area_conf, torch.full_like(area_conf, 0.05))

        invalid_indices = torch.nonzero(region_mask == 0, as_tuple=False)
        return repaired, region_mask, region_confidence, invalid_indices

    def forward(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # image: (B, 1, 48, 48) -> expand to 3 channels for ResNet
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        batch_size = image.size(0)
        feature_map = self.backbone(image)
        bboxes, computed_mask, computed_confidence, invalid_indices = self._prepare_regions(
            bboxes,
            batch_size=batch_size,
            device=image.device,
            dtype=image.dtype,
        )

        if region_mask is None:
            region_mask = computed_mask
        else:
            region_mask = region_mask.to(device=image.device, dtype=image.dtype)
        if region_confidence is None:
            region_confidence = computed_confidence
        else:
            region_confidence = region_confidence.to(device=image.device, dtype=image.dtype)

        if self.training:
            drop_mask = (torch.rand(batch_size, self.config.num_regions, device=image.device) > self.region_dropout_prob).to(image.dtype)
            region_mask = region_mask * drop_mask
            region_confidence = region_confidence * drop_mask

        roi_nodes = self.roi_align(feature_map, bboxes)
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes)
        region_embeddings = self.region_proj(region_embeddings)

        missing_token = self.missing_region_token.view(1, 1, -1)
        region_valid_mask = region_mask.unsqueeze(-1) > 0
        region_embeddings = torch.where(region_valid_mask, region_embeddings, missing_token.expand_as(region_embeddings))

        predicted_confidence = self.region_reliability_predictor(region_embeddings).squeeze(-1)
        region_confidence = torch.clamp(0.5 * region_confidence + 0.5 * predicted_confidence, 0.0, 1.0)
        region_confidence = region_confidence * region_mask

        suppression_gate = (region_confidence > float(self.config.region_confidence_threshold)).float().unsqueeze(-1)
        region_embeddings = (
            suppression_gate * region_embeddings
            + (1.0 - suppression_gate) * missing_token.expand_as(region_embeddings)
        )

        micro_motif_bank = self.micro_motif_bank()
        micro_motif_attention, region_motif_tokens = self.micro_motif_matcher(region_embeddings, micro_motif_bank)
        region_motif_tokens = (
            suppression_gate * region_motif_tokens
            + (1.0 - suppression_gate) * missing_token.expand_as(region_motif_tokens)
        )

        # Global facial prior: learnable symmetric adjacency over semantic regions.
        # Symmetrize first, then sigmoid to map to [0, 1] as a soft prior.
        adj_sym = (self.edge_importance + self.edge_importance.transpose(0, 1)) / 2.0
        adj_prior = torch.sigmoid(adj_sym)
        macro_adj = adj_prior.unsqueeze(0).expand(image.size(0), -1, -1)

        macro_embeddings = self.macro_reasoner(
            region_motif_tokens,
            adj=macro_adj,
            region_confidence=region_confidence,
            region_mask=region_mask,
        )

        macro_motif_bank = self.macro_motif_bank()
        logits_motif, macro_motif_attention, topology_matrix = self.macro_motif_matcher(
            macro_embeddings,
            macro_motif_bank,
        )

        pooled = self.global_pool(feature_map).flatten(1)
        logits_global = self.global_head(pooled)

        fusion_weight = torch.sigmoid(self.alpha)
        logits = logits_motif + fusion_weight * logits_global

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_global": logits_global,
            "micro_node_features": micro_node_features,
            "micro_motif_attention": micro_motif_attention,
            "region_motif_tokens": region_motif_tokens,
            "region_embeddings": region_embeddings,
            "region_mask": region_mask,
            "region_confidence": region_confidence,
            "invalid_region_indices": invalid_indices,
            "macro_embeddings": macro_embeddings,
            "macro_motif_attention": macro_motif_attention,
            "topology_matrix": topology_matrix,
            "micro_motif_bank": micro_motif_bank,
            "macro_motif_bank": macro_motif_bank,
        }
