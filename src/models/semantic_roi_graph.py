from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import roi_align
import timm
import numpy as np


# Loss helpers moved to src/models/semantic_roi_graph_losses.py

def safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """A numerically stable softmax that prevents NaN when vectors are fully masked or have large values."""
    x_max = x.max(dim=dim, keepdim=True)[0].detach()
    x_shifted = torch.clamp(x - x_max, min=-50.0, max=0.0)
    exp_x = torch.exp(x_shifted)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return exp_x / denom


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
    backbone_type: str = "resnet50"
    num_classes: int = 7
    num_regions: int = 9
    roi_grid: int = 4
    feature_dim: int = 256
    motif_per_class: int = 4
    micro_motifs_per_region: int = 8
    macro_motifs_per_class: int = 4
    # Bug 5 fix: defaults now match semantic_roi_graph.yaml
    cross_region_compositions: int = 8   # was 4
    semantic_state_dim: int = 128         # was 9 — must be divisible by semantic_attn_heads
    semantic_latent_dim: int = 256        # was 128
    semantic_attn_heads: int = 4          # was 3 — must divide semantic_state_dim
    hyperedge_count: int = 4
    router_hidden_dim: int = 256          # was 64
    use_pretrained: bool = True
    backbone_out_size: int = 24
    bbox_input_size: int = 48
    micro_layers: int = 2
    macro_layers: int = 2
    attn_heads: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.0
    relation_temperature: float = 0.07
    region_confidence_threshold: float = 0.3
    fusion_scale: float = 0.25
    region_dropout_prob: float = 0.0
    program_dim: int = 128
    programs_per_class: int = 4
    use_facial_attention_neck: bool = True
    neck_groups: int = 4
    enable_logit_alignment: bool = True
    enable_manifold_mixup: bool = True
    manifold_mixup_prob: float = 0.3
    manifold_mixup_alpha: float = 0.2


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone with projection for high spatial resolution Graph input."""

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)

        # Keep high resolution by removing early downsampling.
        resnet.conv1.stride = (1, 1)
        resnet.maxpool = nn.Identity()

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 48 -> 48
        self.layer2 = resnet.layer2  # 48 -> 24
        self.layer3 = resnet.layer3  # 24 -> 12
        
        # ResNet50 layer3 outputs 1024 channels. Project to feature_dim (256)
        self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3)
        self.proj = nn.Sequential(
            nn.Conv2d(1024, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        self.out_channels = feature_dim

        # Free layer4 - not used in forward, but would waste GPU memory
        del resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.output_layer(x)
        return self.proj(x)


class HRNetBackbone(nn.Module):
    """HRNet-W18 backbone with multi-resolution fusion and spatial resolution preservation."""
    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True, out_size: int = 24):
        super().__init__()
        self.out_size = int(out_size)
        self.hrnet = timm.create_model('hrnet_w18', pretrained=use_pretrained, features_only=True)
        
        if hasattr(self.hrnet, 'conv1'):
            self.hrnet.conv1.stride = (1, 1)
            
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 48, 48)
            feats = self.hrnet(dummy)
            total_channels = sum([f.shape[1] for f in feats])
            
        self.proj = nn.Sequential(
            nn.Conv2d(total_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        self.out_channels = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        feats = self.hrnet(x)
        # Fuse multi-resolution features by resizing all to target_size (e.g. 24x24 or 12x12)
        target_size = (self.out_size, self.out_size)
        fused = []
        for f in feats:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            fused.append(f)
            
        x = torch.cat(fused, dim=1)
        return self.proj(x)


class ChannelShuffle(nn.Module):
    """Channel Shuffle operation for Group Convolution."""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = int(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.groups <= 1 or c % self.groups != 0:
            return x
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Facial Deformation Perception.
    Decomposes spatial pooling into horizontal (1xW) and vertical (Hx1) features
    to capture directional Action Units (smile width, jaw height, eye squinting).
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.shape

        x_h = self.pool_h(x)                       # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)           # (N, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


class AsymmetricERFBlock(nn.Module):
    """
    Asymmetric Group Residual Convolution Block (ERFNet-inspired).
    Decomposes 3x3 into 1x3 (horizontal) + 3x1 (vertical) with Group Convolutions,
    reducing FLOPs by 33% and enhancing directional facial edge feature extraction.
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.conv_1x3 = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1),
            groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()

        self.conv_3x1 = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
            groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()

        self.shuffle = ChannelShuffle(groups)

        # Weak bottleneck pointwise projection
        self.conv_pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        # Zero-init residual projection so neck starts as clean identity mapping
        nn.init.zeros_(self.bn3.weight)
        nn.init.zeros_(self.bn3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv_1x3(x)))
        out = self.act2(self.bn2(self.conv_3x1(out)))
        out = self.shuffle(out)
        out = self.bn3(self.conv_pointwise(out))
        return residual + out


class FacialDeformationAttentionNeck(nn.Module):
    """
    Neck module combining Asymmetric Group Residual Blocks and Coordinate Attention.
    Enhances spatial resolution quality and directional facial deformations
    prior to Semantic ROI Alignment and Global Context pooling.
    Toggle on/off via config: use_facial_attention_neck: true/false
    """

    def __init__(self, channels: int = 256, groups: int = 4):
        super().__init__()
        self.asym_block = AsymmetricERFBlock(channels=channels, groups=groups)
        self.coord_att = CoordinateAttention(in_channels=channels, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.asym_block(x)
        x = self.coord_att(x)
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
        attn = safe_softmax(attn, dim=-1)
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


class SemanticStateEncoder(nn.Module):
    """Project region embeddings into interpretable semantic facial state space."""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim // 2, state_dim * 2)
        self.state_dim = state_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, region_embeddings: torch.Tensor) -> torch.Tensor:
        raw_state = self.proj(region_embeddings)
        gate = self.gate(region_embeddings)
        # Fix 2: pure gating — gate actually controls information flow.
        # Original `raw_state * gate + raw_state` = `raw_state * (gate + 1)`,
        # making the Sigmoid gate a mere scaling factor with no off-switch.
        semantic_state = self.norm(raw_state * gate)
        return semantic_state


class MicroSemanticMotifBank(nn.Module):
    """Learnable local semantic motifs in semantic state space."""

    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int):
        super().__init__()
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim
        self.motifs = nn.Parameter(torch.randn(num_regions, motifs_per_region, state_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.motifs


class MicroSemanticMotifMatcher(nn.Module):
    """Match semantic region states to interpretable local semantic motifs."""

    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim
        self.temperature = float(temperature)
        self.token_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    def forward(self, semantic_states: torch.Tensor, motif_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_norm = F.normalize(semantic_states, dim=-1)
        bank_norm = F.normalize(motif_bank, dim=-1)
        sim = torch.einsum("brs,rks->brk", state_norm, bank_norm) / self.temperature
        attn = safe_softmax(sim, dim=-1)
        tokens = torch.einsum("brk,rks->brs", attn, motif_bank)
        tokens = self.token_proj(tokens)
        semantic_tokens = semantic_states + tokens
        return attn, semantic_tokens


class SemanticInteractionBlock(nn.Module):
    """Learned semantic interaction reasoning for pairwise facial coordination."""

    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, dropedge_rate: float = 0.5):
        super().__init__()
        self.dropedge_rate = dropedge_rate
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        pair_input_dim = state_dim * 4
        self.edge_gate = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.edge_message = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, semantic_states: torch.Tensor, region_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, r, s = semantic_states.shape
        left = semantic_states.unsqueeze(2).expand(b, r, r, s)
        right = semantic_states.unsqueeze(1).expand(b, r, r, s)
        pair_input = torch.cat([left, right, left - right, left * right], dim=-1)
        
        gates = self.edge_gate(pair_input).squeeze(-1) + 0.1
        
        # Kịch bản 2: Graph DropEdge
        # Randomly sever connections between facial regions during training
        # to prevent over-smoothing and force robust path discovery.
        if self.dropedge_rate > 0.0:
            gates = F.dropout(gates, p=self.dropedge_rate, training=self.training)
        
        # Computational fix: Mask out invalid regions from interaction
        if region_mask is not None:
            pair_mask = region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)
            gates = gates * pair_mask

        messages = self.edge_message(pair_input)
        interaction_tensor = gates.unsqueeze(-1) * messages
        interaction_summary = interaction_tensor.sum(dim=2) / (gates.sum(dim=2, keepdim=True).clamp_min(1e-4))
        updated_states = self.norm(semantic_states + interaction_summary)
        return updated_states, interaction_tensor, gates


class CrossRegionCompositionGraph(nn.Module):
    """Learn higher-order semantic compositions across facial regions."""

    def __init__(
        self,
        state_dim: int,
        num_compositions: int,
        attn_heads: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if state_dim % attn_heads != 0:
            raise ValueError("state_dim must be divisible by attn_heads")

        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.num_compositions = num_compositions
        self.composition_queries = nn.Parameter(torch.randn(num_compositions, state_dim) * 0.02)
        self.pair_encoder = nn.Sequential(
            nn.Linear(state_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.pair_router = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.composition_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.composition_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        semantic_states: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, r, d = semantic_states.shape
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence.unsqueeze(-1)

        left = tokens.unsqueeze(2).expand(b, r, r, d)
        right = tokens.unsqueeze(1).expand(b, r, r, d)
        pair_input = torch.cat([left, right, left - right, left * right], dim=-1)
        pair_tokens = self.pair_encoder(pair_input)
        pair_scores = self.pair_router(pair_tokens).squeeze(-1)

        if region_mask is not None:
            pair_mask = region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)
            pair_scores = pair_scores.masked_fill(pair_mask <= 0, -1e9)

        pair_attention = safe_softmax(pair_scores.reshape(b, -1), dim=-1).reshape(b, r, r)
        pair_sequence = pair_tokens.reshape(b, r * r, d)

        key_padding_mask = None
        if region_mask is not None:
            key_padding_mask = (pair_mask <= 0).reshape(b, r * r)

        composition_queries = self.composition_queries.unsqueeze(0).expand(b, -1, -1)
        cross_region_tokens, composition_attn = self.composition_attn(
            composition_queries,
            pair_sequence,
            pair_sequence,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        cross_region_tokens = self.composition_norm(cross_region_tokens)

        return {
            "cross_region_tokens": cross_region_tokens,
            "composition_attn": composition_attn,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention,
        }


class SemanticHypergraphReasoner(nn.Module):
    """Compose multi-region semantic programs with learned hyperedge routing."""

    def __init__(self, state_dim: int, latent_dim: int, hyperedge_count: int, attn_heads: int, router_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        if state_dim % attn_heads != 0:
            raise ValueError("state_dim must be divisible by semantic_attn_heads")

        self.hyperedge_count = hyperedge_count
        self.hyperedge_queries = nn.Parameter(torch.randn(hyperedge_count, state_dim) * 0.02)
        self.hyperedge_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.region_back_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.router = nn.Sequential(
            nn.Linear(state_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )
        self.latent_projector = nn.Sequential(
            nn.Linear(state_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        semantic_states: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence.unsqueeze(-1)

        key_padding_mask = None
        if region_mask is not None:
            key_padding_mask = region_mask <= 0

        batch_size = tokens.size(0)
        hyper_queries = self.hyperedge_queries.unsqueeze(0).expand(batch_size, -1, -1)
        hyperedge_tokens, hyperedge_attn = self.hyperedge_attn(
            hyper_queries,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        region_context, region_back_attn = self.region_back_attn(
            tokens,
            hyperedge_tokens,
            hyperedge_tokens,
            need_weights=True,
            average_attn_weights=False,
        )

        composed_states = tokens + region_context
        routing_logits = self.router(composed_states).squeeze(-1)
        if region_mask is not None:
            routing_logits = routing_logits.masked_fill(region_mask <= 0, -1e9)
        routing_weights = safe_softmax(routing_logits, dim=1)
        if region_mask is not None:
            routing_weights = routing_weights * region_mask
            routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_state = torch.sum(routing_weights.unsqueeze(-1) * composed_states, dim=1)
        hyper_summary = hyperedge_tokens.mean(dim=1)
        emotion_latent = self.latent_projector(torch.cat([pooled_state, hyper_summary], dim=-1))
        emotion_latent = self.latent_norm(emotion_latent)

        return {
            "composed_states": composed_states,
            "hyperedge_tokens": hyperedge_tokens,
            "hyperedge_attn": hyperedge_attn,
            "region_back_attn": region_back_attn,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "emotion_latent": emotion_latent,
        }


class SemanticCompositionalProgramBank(nn.Module):
    """Learn structured semantic facial programs and their topology."""

    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.programs = nn.Parameter(torch.randn(num_classes, programs_per_class, num_regions, state_dim) * 0.02)
        self.topology_logits = nn.Parameter(torch.randn(num_classes, programs_per_class, num_regions, num_regions) * 0.02)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.programs, torch.sigmoid(self.topology_logits)


class SemanticProgramExecutor(nn.Module):
    """Execute semantic facial programs against observed region states."""

    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.temperature = float(temperature)
        self.program_summary_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

        # Kịch bản 12: Trọng số cấu trúc thích ứng (Adaptive Semantic Structure)
        self.sim_weights = nn.Parameter(torch.ones(1, num_classes, 1, 3))
        with torch.no_grad():
            self.sim_weights[..., 0] = 1.0   # region_sim
            self.sim_weights[..., 1] = 0.5   # topology_sim
            self.sim_weights[..., 2] = 0.25  # composition_sim

    def forward(
        self,
        semantic_states: torch.Tensor,
        cross_region_tokens: torch.Tensor,
        program_bank: torch.Tensor,
        program_topology: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        interaction_gates: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        state_norm = F.normalize(semantic_states, dim=-1)
        program_norm = F.normalize(program_bank, dim=-1)

        # 1. Compute valid region similarity
        region_sims = torch.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            region_sim = (region_sims * routing_weights.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        elif region_mask is not None:
            valid_mask = region_mask.unsqueeze(1).unsqueeze(1)
            region_sims = region_sims * valid_mask
            region_sim = region_sims.sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1.0)
        else:
            region_sim = region_sims.mean(dim=-1)

        # 2. Compute valid topology similarity (1.0 - MSE)
        if interaction_gates is not None:
            observed_topology = interaction_gates.unsqueeze(1).unsqueeze(1)
            topology_mse = (observed_topology - program_topology.unsqueeze(0)) ** 2
            if region_mask is not None:
                pair_mask = (region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)).unsqueeze(1).unsqueeze(1)
                topology_mse = topology_mse * pair_mask
                topology_sim = 1.0 - (topology_mse.sum(dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp_min(1.0))
            else:
                topology_sim = 1.0 - topology_mse.mean(dim=(-1, -2))
        else:
            topology_sim = torch.ones_like(region_sim)

        # 3. Compute valid composition similarity
        # cross_region_tokens has shape (B, num_compositions, D) where num_compositions is 8.
        # It's already robust to region_mask because the attention that produces it masks invalid pairs.
        composition_summary = cross_region_tokens.mean(dim=1)
        composition_summary = self.program_summary_proj(composition_summary)
        
        program_summary = self.program_summary_proj(program_bank.mean(dim=2))
        composition_sim = torch.einsum("bd,cmd->bcm", F.normalize(composition_summary, dim=-1), F.normalize(program_summary, dim=-1))

        # Kịch bản 12: Sử dụng trọng số động thay vì hằng số cứng nhắc
        w = F.softplus(self.sim_weights) # Đảm bảo trọng số dương
        total_sim = w[..., 0] * region_sim + w[..., 1] * topology_sim + w[..., 2] * composition_sim
        
        # Save pre-temperature scaled versions for auxiliary loss logging consistency
        region_score = region_sim / self.temperature
        topology_score = topology_sim / self.temperature
        composition_score = composition_sim / self.temperature
        
        # Fix: Gradient Explosion during Temperature Scaling.
        # Clamp compatibility to avoid logsumexp gradient blowup while preserving relative order
        compatibility = (total_sim / self.temperature).clamp(-30.0, 30.0)
        
        program_attention = safe_softmax(compatibility, dim=-1)
        class_scores = torch.logsumexp(compatibility, dim=-1)
        program_tokens = torch.einsum("bcm,cmd->bcd", program_attention, program_summary)

        if routing_weights is not None:
            routing_entropy = -(routing_weights.clamp_min(1e-6) * routing_weights.clamp_min(1e-6).log()).sum(dim=-1)
        else:
            routing_entropy = torch.zeros(semantic_states.size(0), device=semantic_states.device)

        return {
            "program_scores": class_scores,
            "program_attention": program_attention,
            "program_tokens": program_tokens,
            "compatibility": compatibility,
            "region_score": region_score,
            "topology_score": topology_score,
            "composition_score": composition_score,
            "routing_entropy": routing_entropy,
        }


# Backward-compatible aliases for callers and checkpoints.
MacroSemanticProgramBank = SemanticCompositionalProgramBank
MacroSemanticProgramMatcher = SemanticProgramExecutor
MacroMotifBank = SemanticCompositionalProgramBank
MacroMotifMatcher = SemanticProgramExecutor
SemanticMotifBank = SemanticCompositionalProgramBank
SemanticMotifMatcher = SemanticProgramExecutor


class SemanticEmotionClassifier(nn.Module):
    """Classify emotion from semantic latent facial representation."""

    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backward-compatible aliases for callers and checkpoints.
MicroMotifBank = MicroSemanticMotifBank
MicroMotifMatcher = MicroSemanticMotifMatcher
MacroMotifBank = MacroSemanticProgramBank
MacroMotifMatcher = MacroSemanticProgramMatcher
SemanticMotifBank = MacroSemanticProgramBank
SemanticMotifMatcher = MacroSemanticProgramMatcher


class SemanticROIGraphFER(nn.Module):
    """End-to-end semantic compositional facial reasoning model."""

    def __init__(self, config: SemanticRoiGraphConfig):
        super().__init__()
        self.config = config

        if getattr(config, "backbone_type", "resnet50") == "hrnet_w18":
            self.backbone = HRNetBackbone(
                feature_dim=config.feature_dim,
                use_pretrained=config.use_pretrained,
                out_size=getattr(config, "backbone_out_size", 24),
            )
        else:
            self.backbone = ResNet50Backbone(
                feature_dim=config.feature_dim,
                use_pretrained=config.use_pretrained,
            )

        # Facial Deformation Attention Neck (toggle: use_facial_attention_neck)
        self.use_facial_attention_neck = getattr(config, "use_facial_attention_neck", True)
        if self.use_facial_attention_neck:
            neck_groups = getattr(config, "neck_groups", 4)
            self.facial_attention_neck = FacialDeformationAttentionNeck(
                channels=config.feature_dim,
                groups=neck_groups,
            )
        else:
            self.facial_attention_neck = nn.Identity()

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

        # Fix 3: region_proj (feature_dim→feature_dim) removed — a same-dimension
        # linear projection adds no representational capacity and wastes ~65K params.
        self.semantic_state_encoder = SemanticStateEncoder(
            input_dim=config.feature_dim,
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.feature_dim // 2, config.semantic_state_dim * 2),
            dropout=config.dropout,
        )
        # Breakthrough 2: Learnable Anatomical Region Positional Embedding
        # Injects 2D facial topology coordinates into the 9 region state tokens
        self.region_pos_embed = nn.Parameter(torch.zeros(1, config.num_regions, config.semantic_state_dim))
        nn.init.trunc_normal_(self.region_pos_embed, std=0.02)

        self.semantic_interaction_block = SemanticInteractionBlock(
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
            dropedge_rate=0.5,
        )

        self.micro_motif_bank = MicroSemanticMotifBank(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
        )
        self.micro_motif_matcher = MicroSemanticMotifMatcher(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
            temperature=config.relation_temperature,
        )

        self.semantic_compositional_reasoner = SemanticHypergraphReasoner(
            state_dim=config.semantic_state_dim,
            latent_dim=config.semantic_latent_dim,
            hyperedge_count=config.hyperedge_count,
            attn_heads=config.semantic_attn_heads,
            router_hidden_dim=config.router_hidden_dim,
            dropout=config.dropout,
        )

        self.cross_region_composition_graph = CrossRegionCompositionGraph(
            state_dim=config.semantic_state_dim,
            num_compositions=config.cross_region_compositions,
            attn_heads=config.semantic_attn_heads,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
        )

        self.semantic_program_bank = SemanticCompositionalProgramBank(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
        )
        self.semantic_program_executor = SemanticProgramExecutor(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
            temperature=config.relation_temperature,
        )

        self.semantic_classifier = SemanticEmotionClassifier(
            latent_dim=config.semantic_latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(config.feature_dim, config.semantic_latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.global_fusion = nn.Sequential(
            nn.Linear(config.semantic_latent_dim * 2, config.semantic_latent_dim),
            nn.LayerNorm(config.semantic_latent_dim),
            nn.GELU(),
        )

        # Per-class gate: each emotion class learns its own graph-vs-global balance.
        # Init with -0.5 → sigmoid(-0.5) ≈ 0.38, slightly below 0.5 to favour the graph branch early.
        self.semantic_structure_gate = nn.Parameter(torch.full((config.num_classes,), -0.5))

        # Solution 1: Logit Scale Alignment (LayerNorm per branch + learnable logit scale)
        self.enable_logit_alignment = bool(getattr(config, "enable_logit_alignment", True))
        if self.enable_logit_alignment:
            self.fused_logit_norm = nn.LayerNorm(config.num_classes)
            self.motif_logit_norm = nn.LayerNorm(config.num_classes)
            self.logit_scale = nn.Parameter(torch.tensor(8.0))

        # Solution 2: Semantic Manifold Mixup (Feature-level graph mixup, 100% bbox-safe)
        self.enable_manifold_mixup = bool(getattr(config, "enable_manifold_mixup", True))
        self.manifold_mixup_prob = float(getattr(config, "manifold_mixup_prob", 0.3))
        self.manifold_mixup_alpha = float(getattr(config, "manifold_mixup_alpha", 0.2))

        # Backward-compatible aliases for older checkpoints and callers.
        self.macro_motif_bank = self.semantic_program_bank
        self.macro_motif_matcher = self.semantic_program_executor
        self.motif_bank = self.semantic_program_bank
        self.motif_matcher = self.semantic_program_executor

        self.missing_region_token = nn.Parameter(torch.randn(config.feature_dim) * 0.02)
        self.region_reliability_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.region_dropout_prob = float(getattr(config, "region_dropout_prob", 0.05))

    def load_state_dict(self, state_dict, strict=False):
        """Backward-compatible: upgrade scalar semantic_structure_gate from old checkpoints."""
        key = "semantic_structure_gate"
        if key in state_dict:
            old = state_dict[key]
            if old.ndim == 0 or old.numel() == 1:
                state_dict = dict(state_dict)  # don't mutate the original
                state_dict[key] = old.detach().view(1).expand(self.config.num_classes).clone()
        return super().load_state_dict(state_dict, strict=strict)

    def set_training_progress(self, progress: float) -> None:
        """Track training progress (0.0 to 1.0) while maintaining constant mixup regularization."""
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
        self.manifold_mixup_prob = float(getattr(self.config, "manifold_mixup_prob", 0.3))

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
        """Public forward: dispatches to TTA or single-image path.
        
        NOTE: Built-in TTA removed. TTA is now handled exclusively by
        apply_multi_scale_tta() in src/models/utils.py during evaluation.
        This ensures clean (non-inflated) val_acc during training and
        halves validation compute time.
        """
        if image.dim() == 5:
            return self._forward_tta(image, bboxes, region_mask, region_confidence)

        return self._forward_single(image, bboxes, region_mask, region_confidence)

    def _forward_tta(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TTA path: image is (B, T, C, H, W); averages logits over T crops."""
        B, T, C, H, W = image.shape
        # Flatten crops into batch dimension
        flat_image = image.reshape(B * T, C, H, W)

        # Expand bbox / mask tensors from (B, R, *) -> (B*T, R, *)
        flat_bboxes = None
        if bboxes is not None:
            flat_bboxes = bboxes.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, bboxes.size(1), bboxes.size(2))
        flat_region_mask = None
        if region_mask is not None:
            flat_region_mask = region_mask.unsqueeze(1).expand(B, T, -1).reshape(B * T, region_mask.size(1))
        flat_region_confidence = None
        if region_confidence is not None:
            flat_region_confidence = region_confidence.unsqueeze(1).expand(B, T, -1).reshape(B * T, region_confidence.size(1))

        outputs = self._forward_single(flat_image, flat_bboxes, flat_region_mask, flat_region_confidence)

        # Average the classification scores over T crops
        _avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
        for key in _avg_keys:
            if key in outputs and torch.is_tensor(outputs[key]):
                x = outputs[key]
                if x.size(0) == B * T:
                    outputs[key] = x.reshape(B, T, *x.shape[1:]).mean(dim=1)

        # For non-averaged keys that still have B*T batch size, keep center-crop (index 4)
        center_idx = 4 if T > 4 else T // 2
        for key, val in outputs.items():
            if key in _avg_keys:
                continue
            if torch.is_tensor(val) and val.dim() >= 1 and val.size(0) == B * T:
                outputs[key] = val.reshape(B, T, *val.shape[1:])[:, center_idx]

        return outputs

    def _forward_single(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Core forward for a regular (B, C, H, W) batch."""
        # image: (B, 1, 48, 48) -> expand to 3 channels for ResNet
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        batch_size = image.size(0)
        feature_map = self.backbone(image)
        feature_map = self.facial_attention_neck(feature_map)
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

        if self.training and self.region_dropout_prob > 0.0:
            drop_prob = min(float(self.region_dropout_prob), 0.3)
            drop_mask = (torch.rand(batch_size, self.config.num_regions, device=image.device) > drop_prob).to(image.dtype)
            # Guarantee at least 4 regions remain active per sample to prevent graph collapse & NaN
            num_active = (region_mask * drop_mask).sum(dim=-1, keepdim=True)
            needs_rescue = num_active < 4.0
            if needs_rescue.any():
                drop_mask = torch.where(needs_rescue, torch.ones_like(drop_mask), drop_mask)
            region_mask = region_mask * drop_mask
            region_confidence = region_confidence * drop_mask

        roi_nodes = self.roi_align(feature_map, bboxes)
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes)

        missing_token = self.missing_region_token.view(1, 1, -1)
        region_valid_mask = region_mask.unsqueeze(-1) > 0
        region_embeddings = torch.where(region_valid_mask, region_embeddings, missing_token.expand_as(region_embeddings))

        predicted_confidence = self.region_reliability_predictor(region_embeddings).squeeze(-1)
        region_confidence = torch.clamp(0.5 * region_confidence + 0.5 * predicted_confidence, 0.0, 1.0)
        region_confidence = region_confidence * region_mask

        semantic_state_tokens = self.semantic_state_encoder(region_embeddings)
        # Breakthrough 2: Add Learnable Region Positional Embedding
        semantic_state_tokens = semantic_state_tokens + self.region_pos_embed
        global_semantic_context = self.global_context(feature_map)

        # Solution 2: Semantic Manifold Mixup (Feature-Level Mixup, 100% bbox-safe)
        mixup_lam = 1.0
        mixup_perm = None
        if self.training and self.enable_manifold_mixup and self.manifold_mixup_prob > 0.0:
            if torch.rand(1).item() < self.manifold_mixup_prob:
                alpha = self.manifold_mixup_alpha
                mixup_lam = float(np.random.beta(alpha, alpha))
                if mixup_lam < 0.5:
                    mixup_lam = 1.0 - mixup_lam
                mixup_perm = torch.randperm(batch_size, device=image.device)

                semantic_state_tokens = mixup_lam * semantic_state_tokens + (1.0 - mixup_lam) * semantic_state_tokens[mixup_perm]
                global_semantic_context = mixup_lam * global_semantic_context + (1.0 - mixup_lam) * global_semantic_context[mixup_perm]

        micro_motif_bank = self.micro_motif_bank()
        micro_motif_attention, semantic_motif_tokens = self.micro_motif_matcher(semantic_state_tokens, micro_motif_bank)

        # Step 1: Pairwise region interaction (local semantic coordination).
        interaction_states, semantic_interaction_tensor, semantic_interaction_gates = self.semantic_interaction_block(
            semantic_motif_tokens,
            region_mask=region_mask,
        )

        # Step 2: Higher-order cross-region composition on interaction-enriched states.
        cross_region_outputs = self.cross_region_composition_graph(
            interaction_states,
            region_mask=region_mask,
            region_confidence=region_confidence,
        )
        cross_region_tokens = cross_region_outputs["cross_region_tokens"]
        cross_region_attention = cross_region_outputs["composition_attn"]
        cross_region_pair_tokens = cross_region_outputs["pair_tokens"]
        cross_region_pair_scores = cross_region_outputs["pair_scores"]
        cross_region_pair_attention = cross_region_outputs["pair_attention"]

        # Step 3: Enrich interaction states with higher-order composition context.
        composition_summary = cross_region_tokens.mean(dim=1, keepdim=True)
        hypergraph_input = interaction_states + composition_summary.expand_as(interaction_states)

        compositional_outputs = self.semantic_compositional_reasoner(
            hypergraph_input,
            region_mask=region_mask,
            region_confidence=region_confidence,
        )
        composed_states = compositional_outputs["composed_states"]
        hyperedge_tokens = compositional_outputs["hyperedge_tokens"]
        routing_weights = compositional_outputs["routing_weights"]
        semantic_latent_embedding = compositional_outputs["emotion_latent"]

        semantic_program_bank, semantic_program_topology = self.semantic_program_bank()
        semantic_program_outputs = self.semantic_program_executor(
            composed_states,
            cross_region_tokens,
            semantic_program_bank,
            semantic_program_topology,
            region_mask=region_mask,
            interaction_gates=semantic_interaction_gates,
            routing_weights=routing_weights,
        )
        semantic_program_scores = semantic_program_outputs["program_scores"]
        semantic_program_attention = semantic_program_outputs["program_attention"]
        semantic_program_tokens = semantic_program_outputs["program_tokens"]
        semantic_program_compatibility = semantic_program_outputs["compatibility"]
        semantic_program_region_scores = semantic_program_outputs["region_score"]
        semantic_program_topology_scores = semantic_program_outputs["topology_score"]
        semantic_program_composition_scores = semantic_program_outputs["composition_score"]
        semantic_program_routing_entropy = semantic_program_outputs["routing_entropy"]

        fused_latent = self.global_fusion(torch.cat([semantic_latent_embedding, global_semantic_context], dim=-1))
        logits_fused = self.semantic_classifier(fused_latent)

        # Per-class gate: shape (1, num_classes) — each emotion learns its own balance
        structure_gate = torch.sigmoid(self.semantic_structure_gate).view(1, -1)
        logits_motif = semantic_program_scores

        # Solution 1: Logit Scale Alignment (Standardized blending)
        if self.enable_logit_alignment:
            logits_fused_norm = self.fused_logit_norm(logits_fused)
            logits_motif_norm = self.motif_logit_norm(logits_motif)
            blended_norm = (1.0 - structure_gate) * logits_fused_norm + structure_gate * logits_motif_norm
            logits = self.logit_scale * blended_norm
        else:
            logits = (1.0 - structure_gate) * logits_fused + structure_gate * logits_motif

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_fused": logits_fused,
            "structure_gate": structure_gate,
            "mixup_lam": mixup_lam,
            "mixup_perm": mixup_perm,
            
            "micro_node_features": micro_node_features,
            "micro_motif_attention": micro_motif_attention,
            "region_motif_tokens": semantic_motif_tokens,
            "region_embeddings": region_embeddings,
            "semantic_state_tokens": semantic_state_tokens,
            "semantic_motif_tokens": semantic_motif_tokens,
            "cross_region_tokens": cross_region_tokens,
            "cross_region_attention": cross_region_attention,
            "cross_region_pair_tokens": cross_region_pair_tokens,
            "cross_region_pair_scores": cross_region_pair_scores,
            "cross_region_pair_attention": cross_region_pair_attention,
            "semantic_interaction_tensor": semantic_interaction_tensor,
            "semantic_interaction_gates": semantic_interaction_gates,
            "semantic_routing_weights": routing_weights,
            "hyperedge_tokens": hyperedge_tokens,
            "semantic_program_scores": semantic_program_scores,
            "semantic_program_attention": semantic_program_attention,
            "semantic_program_tokens": semantic_program_tokens,
            "semantic_program_compatibility": semantic_program_compatibility,
            "semantic_program_region_scores": semantic_program_region_scores,
            "semantic_program_topology_scores": semantic_program_topology_scores,
            "semantic_program_composition_scores": semantic_program_composition_scores,
            "semantic_program_routing_entropy": semantic_program_routing_entropy,
            "semantic_program_bank": semantic_program_bank,
            "semantic_program_topology": semantic_program_topology,
            "semantic_latent_embedding": semantic_latent_embedding,
            "fused_latent_embedding": fused_latent,
            "region_mask": region_mask,
            "region_confidence": region_confidence,
            "invalid_region_indices": invalid_indices,
            "macro_embeddings": composed_states,
            "macro_motif_attention": semantic_program_attention,
            "micro_motif_bank": micro_motif_bank,
            "macro_motif_bank": semantic_program_bank,
            "aux_losses": {
                "semantic_consistency": semantic_motif_tokens.new_tensor(0.0),
            },
        }