import torch
import torch.nn as nn
from .backbones import FacialBackbone
from .spatial_attention import MultiHeadSpatialAttention
from .latent_graph import LatentGraphReasoner
from .scn_head import SCNHead
from .mask_guided_dynamic_graph import (
    ClasswiseResidualGate,
    MaskGuidedDynamicRegionGraph,
)


class AttentiveSCNFER(nn.Module):
    """
    Pure Image-Based Attentive Self-Cure Network with Latent Dynamic Graph Reasoning for FER.
    Zero dependency on bounding boxes, landmarks, or pre-extracted masks.

    When ``use_m4_graph`` is enabled, the legacy spatial-attention/latent-graph
    path is replaced by a mask-guided competitive tokenizer, a stack of sparse
    dynamic edge-aware graph blocks, and class-conditioned residual logits.
    
    Architecture Pipeline:
        Input: Raw grayscale images [B, 1, 48, 48]
          │
          ▼
        FacialBackbone (ResNet50 / ResNet34 adapted for 48x48)
          │ Feature Map F ∈ R^{B × C × 12 × 12}
          ├──► Global Average Pooling + Linear ──► f_global ∈ R^{B × D}
          │
          └──► MultiHeadSpatialAttention ────────► M Soft Regional Tokens {h_m} + Attention Maps
                 │
                 ▼
               Latent Dynamic Graph Reasoner ──► Dual-factor Dynamic Adjacency + Residual GCN
                 │
                 ▼
               Node Importance Readout ─────────► f_graph ∈ R^{B × D}
          │
          ▼
        Fusion: LayerNorm(f_global + gate * f_graph) ──► f_fused ∈ R^{B × D}
          │
          ▼
        SCNHead:
          ├──► Logits z ∈ R^{B × 7}
          └──► Sample Confidence alpha ∈ (0, 1)^{B × 1}
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 7,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_attn_heads: int = 8,
        use_latent_graph: bool = True,
        use_spatial_attention: bool = True,
        use_m4_graph: bool = False,
        m4_num_nodes: int = 8,
        m4_graph_depth: int = 3,
        m4_top_k: int = 3,
        m4_edge_heads: int = 4,
        m4_gate_init: float = 0.0,
        dropout: float = 0.25,
        classifier_type: str = "linear",
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.20,
        use_pretrained: bool = True,
        pretrained_weights_path: str = "",
        stem_init: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.use_m4_graph = use_m4_graph
        self.use_spatial_attention = use_spatial_attention and not use_m4_graph
        self.use_latent_graph = use_latent_graph and not use_m4_graph
        if self.use_latent_graph and not self.use_spatial_attention:
            raise ValueError("use_latent_graph requires use_spatial_attention=true")

        # 1. Backbone adapted for 48x48
        self.backbone = FacialBackbone(
            backbone_name=backbone_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            pretrained_weights_path=pretrained_weights_path,
            target_feat_size=12,
            stem_init=stem_init,
        )

        backbone_out_ch = self.backbone.out_channels

        # 2. Global Stream Projector
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Regional reasoning branch.  M4 replaces the original attention and
        # single-layer graph when enabled; legacy configurations remain intact.
        if self.use_m4_graph:
            self.spatial_attention = None
            self.latent_graph = None
            self.m4_graph = MaskGuidedDynamicRegionGraph(
                in_channels=backbone_out_ch,
                embed_dim=embed_dim,
                num_nodes=m4_num_nodes,
                num_classes=num_classes,
                graph_depth=m4_graph_depth,
                top_k=m4_top_k,
                edge_heads=m4_edge_heads,
                dropout=dropout,
            )
            self.m4_gate = ClasswiseResidualGate(
                embed_dim=embed_dim,
                num_classes=num_classes,
                dropout=dropout,
                init_bias=m4_gate_init,
            )
        else:
            self.m4_graph = None
            self.m4_gate = None
            self.spatial_attention = (
                MultiHeadSpatialAttention(
                    in_channels=backbone_out_ch,
                    embed_dim=embed_dim,
                    num_heads=num_attn_heads,
                    dropout=dropout,
                ) if self.use_spatial_attention else None
            )
            self.latent_graph = (
                LatentGraphReasoner(
                    embed_dim=embed_dim,
                    num_nodes=num_attn_heads,
                    dropout=dropout,
                ) if self.use_latent_graph else None
            )

        # 4. Legacy feature fusion.  M4 instead performs class-wise residual
        # fusion after the SCN base classifier.
        self.fusion_norm = nn.LayerNorm(embed_dim) if not self.use_m4_graph else None
        self.fusion_gate = (
            nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
            if self.use_spatial_attention and not self.use_m4_graph
            else None
        )

        # 6. SCN Head (classifier + confidence weight)
        self.scn_head = SCNHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            classifier_type=classifier_type,
            cosface_scale=cosface_scale,
            cosface_margin=cosface_margin,
            init_confidence_bias=1.5,
        )

    def _forward_single(self, x: torch.Tensor, targets=None, targets_b=None, lam=1.0):
        # Feature map: [B, C, 12, 12]
        feat_map = self.backbone(x)

        # Global feature: [B, D]
        f_global = self.global_proj(feat_map)

        if self.use_m4_graph:
            graph_outputs = self.m4_graph(feat_map)
            base_logits, alpha = self.scn_head(
                f_global, targets=targets, targets_b=targets_b, lam=lam
            )
            graph_gate = self.m4_gate(
                f_global, graph_outputs["graph_feature"]
            )
            residual_logits = graph_outputs["residual_logits"]
            logits = base_logits + graph_gate * residual_logits

            return {
                "logits": logits,
                "base_logits": base_logits,
                "residual_logits": residual_logits,
                "graph_gate": graph_gate,
                "alpha": alpha,
                "attn_maps": graph_outputs["attn_maps"],
                "mask_map": graph_outputs["mask_map"],
                "diversity_loss": graph_outputs["diversity_loss"],
                "adj_matrix": graph_outputs["adj_matrix"],
                "adj_matrices": graph_outputs["adj_matrices"],
                "sparsity_loss": graph_outputs["sparsity_loss"],
                "class_attention": graph_outputs["class_attention"],
                "region_geometry": graph_outputs["region_geometry"],
                "features": f_global,
                "graph_features": graph_outputs["graph_feature"],
            }

        # Local spatial feature & attention maps & soft node tokens: [B, M, D]
        if self.spatial_attention is not None:
            f_local, attn_maps, div_loss, head_feats = self.spatial_attention(feat_map)
        else:
            f_local = torch.zeros_like(f_global)
            attn_maps, head_feats = None, None
            div_loss = torch.zeros((), device=x.device)

        # Latent Dynamic Graph Reasoning over soft tokens
        if self.use_latent_graph and self.latent_graph is not None:
            graph_feats, adj_matrix, sparsity_loss = self.latent_graph(head_feats, attn_maps)
            f_rep = f_local + graph_feats
        else:
            f_rep = f_local
            adj_matrix = None
            sparsity_loss = torch.tensor(0.0, device=x.device)

        # Gated residual fusion
        gate = torch.sigmoid(self.fusion_gate) if self.fusion_gate is not None else 0.0
        f_fused = self.fusion_norm(f_global + gate * f_rep)

        # SCN Head classifier
        logits, alpha = self.scn_head(f_fused, targets=targets, targets_b=targets_b, lam=lam)

        return {
            "logits": logits,
            "alpha": alpha,
            "attn_maps": attn_maps,
            "diversity_loss": div_loss,
            "adj_matrix": adj_matrix,
            "sparsity_loss": sparsity_loss,
            "features": f_fused,
        }

    @staticmethod
    def _average_view_outputs(view_outputs):
        """Average predictions while retaining first-view graph diagnostics."""
        averaged = dict(view_outputs[0])
        for key in (
            "logits",
            "alpha",
            "base_logits",
            "residual_logits",
            "graph_gate",
        ):
            values = [output.get(key) for output in view_outputs]
            if all(isinstance(value, torch.Tensor) for value in values):
                averaged[key] = torch.stack(values, dim=0).mean(dim=0)
        return averaged

    def forward(self, x: torch.Tensor, targets=None, targets_b=None, lam=1.0, use_tta=None):

        """
        Forward pass with Test-Time Augmentation (TTA).
        Modes for use_tta:
          - False / None during training: Single image standard forward
          - True / 'flip': 2-crop Horizontal Flip TTA (average original + flipped)
          - 'multiscale' / 'multi_scale': 4-crop Multi-Scale TTA (original, flipped, zoom 1.05x, zoom flipped)
        """
        if use_tta is None:
            use_tta = not self.training

        if not self.training and use_tta:
            if use_tta in ("multiscale", "multi_scale"):
                # 4-crop Multi-Scale TTA
                x_orig = x
                x_flip = torch.flip(x, dims=[-1])
                # Zoom 1.05x and center crop back to the original resolution.
                # Works for both the 48x48 and 96x96 ablations.
                height, width = x.shape[-2:]
                zoom_h, zoom_w = max(height + 2, round(height * 1.05)), max(width + 2, round(width * 1.05))
                top, left = (zoom_h - height) // 2, (zoom_w - width) // 2
                x_zoom = torch.nn.functional.interpolate(
                    x, size=(zoom_h, zoom_w), mode="bilinear", align_corners=False
                )[:, :, top:top + height, left:left + width]
                x_zoom_flip = torch.flip(x_zoom, dims=[-1])

                out1 = self._forward_single(x_orig)
                out2 = self._forward_single(x_flip)
                out3 = self._forward_single(x_zoom)
                out4 = self._forward_single(x_zoom_flip)

                return self._average_view_outputs([out1, out2, out3, out4])
            else:
                # 2-crop Horizontal Flip TTA
                out_orig = self._forward_single(x)
                x_flipped = torch.flip(x, dims=[-1])
                out_flipped = self._forward_single(x_flipped)

                return self._average_view_outputs([out_orig, out_flipped])
        else:
            return self._forward_single(x, targets=targets, targets_b=targets_b, lam=lam)
