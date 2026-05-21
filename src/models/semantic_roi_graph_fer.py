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


@dataclass
class SemanticRoiGraphConfig:
    num_classes: int = 7
    num_regions: int = 9
    roi_grid: int = 4
    feature_dim: int = 256
    motif_per_class: int = 4
    use_pretrained: bool = True
    backbone_out_size: int = 12
    bbox_input_size: int = 48
    micro_layers: int = 2
    macro_layers: int = 2
    attn_heads: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.0


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

    def forward(self, feature_map: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        # feature_map: (B, C, H, W)
        # bboxes: (B, R, 4) in image coords (0..bbox_input_size-1)
        b, _, h, w = feature_map.shape
        scale = float(self.feature_out_size) / float(self.bbox_input_size)

        bboxes = bboxes.float().clone()
        bboxes[..., 0::2] = bboxes[..., 0::2] * scale
        bboxes[..., 1::2] = bboxes[..., 1::2] * scale

        bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0.0, float(w - 1))
        bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0.0, float(h - 1))

        rois = []
        for batch_index in range(b):
            roi = bboxes[batch_index]
            batch_col = torch.full((roi.shape[0], 1), float(batch_index), device=roi.device)
            rois.append(torch.cat([batch_col, roi], dim=1))
        rois = torch.cat(rois, dim=0)

        roi_features = roi_align(
            feature_map,
            rois,
            output_size=(self.roi_grid, self.roi_grid),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        # (B*R, C, G, G) -> (B, R, G*G, C)
        roi_features = roi_features.view(b, -1, feature_map.shape[1], self.roi_grid * self.roi_grid)
        roi_features = roi_features.permute(0, 1, 3, 2).contiguous()
        return roi_features


class GATBlock(nn.Module):
    """Multi-head graph attention with learnable adjacency bias."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1, num_nodes: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.adj_bias = None
        if num_nodes is not None:
            self.adj_bias = nn.Parameter(torch.zeros(1, 1, num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) / (d // self.heads) ** 0.5
        if self.adj_bias is not None:
            attn = attn + self.adj_bias
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, R, D)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return x


class SemanticMotifBank(nn.Module):
    """Learnable motifs per class."""

    def __init__(self, num_classes: int, motifs_per_class: int, num_regions: int, dim: int):
        super().__init__()
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, num_regions, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.motifs


class SemanticMotifMatcher(nn.Module):
    """Match macro graph embeddings to motif bank using relation tensors."""

    def __init__(self, num_classes: int, motifs_per_class: int, num_regions: int, dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_regions = num_regions
        self.dim = dim
        self.scale = dim ** -0.5

    def relation_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (..., R, D) -> (..., R, R)
        embeddings = F.normalize(embeddings, dim=-1)
        return torch.einsum("...id,...jd->...ij", embeddings, embeddings)

    def forward(self, macro_embeddings: torch.Tensor, motif_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # macro_embeddings: (B, R, D)
        # motif_bank: (C, M, R, D)
        rel_macro = self.relation_matrix(macro_embeddings)  # (B, R, R)
        rel_motif = self.relation_matrix(motif_bank)  # (C, M, R, R)

        rel_macro_flat = F.normalize(rel_macro.reshape(rel_macro.shape[0], -1), dim=-1)
        rel_motif_flat = F.normalize(rel_motif.reshape(rel_motif.shape[0], rel_motif.shape[1], -1), dim=-1)

        sim = torch.einsum("bd,cmd->bcm", rel_macro_flat, rel_motif_flat)
        attn = F.softmax(sim, dim=-1)
        logits_motif = (attn * sim).sum(dim=-1)
        return logits_motif, attn


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

        self.macro_reasoner = MacroGraphReasoner(
            dim=config.feature_dim,
            num_regions=config.num_regions,
            layers=config.macro_layers,
            heads=config.attn_heads,
            dropout=config.dropout,
        )

        self.motif_bank = SemanticMotifBank(
            num_classes=config.num_classes,
            motifs_per_class=config.motif_per_class,
            num_regions=config.num_regions,
            dim=config.feature_dim,
        )
        self.motif_matcher = SemanticMotifMatcher(
            num_classes=config.num_classes,
            motifs_per_class=config.motif_per_class,
            num_regions=config.num_regions,
            dim=config.feature_dim,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.num_classes),
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, image: torch.Tensor, bboxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # image: (B, 1, 48, 48) -> expand to 3 channels for ResNet
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        feature_map = self.backbone(image)
        roi_nodes = self.roi_align(feature_map, bboxes)
        roi_nodes, region_embeddings = self.micro_reasoner(roi_nodes)
        region_embeddings = self.region_proj(region_embeddings)
        macro_embeddings = self.macro_reasoner(region_embeddings)

        motif_bank = self.motif_bank()
        logits_motif, motif_attention = self.motif_matcher(macro_embeddings, motif_bank)

        pooled = self.global_pool(feature_map).flatten(1)
        logits_global = self.global_head(pooled)

        fusion_weight = torch.sigmoid(self.alpha)
        logits = logits_motif + fusion_weight * logits_global

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_global": logits_global,
            "motif_attention": motif_attention,
            "region_embeddings": region_embeddings,
            "macro_embeddings": macro_embeddings,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        temperature: float = 0.07,
        contrastive_weight: float = 0.1,
        diversity_weight: float = 0.05,
        consistency_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)

        diversity_loss = self.motif_diversity_loss()
        contrastive_loss = self.supervised_contrastive_loss(outputs["macro_embeddings"], labels, temperature=temperature)
        consistency_loss = self.region_consistency_loss(outputs["region_embeddings"], labels)

        total = ce_loss
        total = total + diversity_weight * diversity_loss
        total = total + contrastive_weight * contrastive_loss
        total = total + consistency_weight * consistency_loss

        return {
            "loss": total,
            "loss_ce": ce_loss,
            "loss_motif_diversity": diversity_loss,
            "loss_contrastive": contrastive_loss,
            "loss_region_consistency": consistency_loss,
        }

    def motif_diversity_loss(self) -> torch.Tensor:
        motifs = self.motif_bank()  # (C, M, R, D)
        c, m, r, d = motifs.shape
        motifs = motifs.view(c, m, r * d)
        motifs = F.normalize(motifs, dim=-1)
        sim = torch.einsum("cmd,cnd->cmn", motifs, motifs)
        identity = torch.eye(m, device=sim.device).unsqueeze(0)
        off_diag = sim * (1.0 - identity)
        return (off_diag ** 2).mean()

    def supervised_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        # embeddings: (B, R, D) -> pooled
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

    def region_consistency_loss(self, region_embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
