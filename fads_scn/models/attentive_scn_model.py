import torch
import torch.nn as nn
from .backbones import FacialBackbone
from .spatial_attention import MultiHeadSpatialAttention
from .latent_graph import LatentGraphReasoner
from .multiscale_fusion import AdaptiveBranchFusion, MultiScaleDualPooling
from .scn_head import SCNHead


class AttentiveSCNFER(nn.Module):
    """
    Pure Image-Based Attentive Self-Cure Network with Latent Dynamic Graph Reasoning for FER.
    Zero dependency on bounding boxes, landmarks, or pre-extracted masks.
    
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
        Optional M1: ConvNeXt C1-C4 + dual pooling ──► f_multi ∈ R^{B × D}
          │
          ▼
        Fusion: legacy residual gate, or M1 adaptive branch/channel weights
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
        dropout: float = 0.25,
        classifier_type: str = "linear",
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.20,
        use_pretrained: bool = True,
        pretrained_weights_path: str = "",
        stem_init: str = "mean",
        use_multiscale_fusion: bool = False,
        multiscale_se_reduction: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.use_spatial_attention = use_spatial_attention
        self.use_multiscale_fusion = use_multiscale_fusion
        if use_latent_graph and not use_spatial_attention:
            raise ValueError("use_latent_graph requires use_spatial_attention=true")
        self.use_latent_graph = use_latent_graph

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
        if self.use_multiscale_fusion and self.backbone.backbone_type != "convnext":
            raise ValueError("use_multiscale_fusion currently requires a ConvNeXt backbone")

        # 2. Global Stream Projector
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Local Spatial Attention Stream (Unsupervised discovery of Action Units)
        self.spatial_attention = (
            MultiHeadSpatialAttention(
                in_channels=backbone_out_ch,
                embed_dim=embed_dim,
                num_heads=num_attn_heads,
                dropout=dropout,
            ) if self.use_spatial_attention else None
        )

        # 4. Latent Dynamic Graph Reasoner (Message passing between soft semantic nodes)
        if self.use_latent_graph:
            self.latent_graph = LatentGraphReasoner(
                embed_dim=embed_dim,
                num_nodes=num_attn_heads,
                dropout=dropout,
            )
        else:
            self.latent_graph = None

        # 5. Fusion Layer. The legacy path keeps its original computation and
        # state-dict keys when multi-scale fusion is disabled.
        if self.use_multiscale_fusion:
            self.multiscale_pool = MultiScaleDualPooling(
                feature_channels=self.backbone.feature_channels,
                embed_dim=embed_dim,
                dropout=dropout,
                se_reduction=multiscale_se_reduction,
            )
            self.adaptive_fusion = AdaptiveBranchFusion(
                embed_dim=embed_dim,
                num_branches=3,
                dropout=dropout,
            )
            self.fusion_norm = None
            self.fusion_gate = None
        else:
            self.multiscale_pool = None
            self.adaptive_fusion = None
            self.fusion_norm = nn.LayerNorm(embed_dim)
            self.fusion_gate = (nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
                                if self.use_spatial_attention else None)

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
        if self.use_multiscale_fusion:
            stage_features = self.backbone(x, return_multiscale=True)
            feat_map = stage_features[-1]
        else:
            stage_features = None
            feat_map = self.backbone(x)

        # Global feature: [B, D]
        f_global = self.global_proj(feat_map)

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

        # M1: adaptive per-image/per-channel fusion across global,
        # local/graph, and multi-scale descriptors.
        if self.use_multiscale_fusion:
            f_multi, scale_weights = self.multiscale_pool(stage_features)
            f_fused, fusion_weights = self.adaptive_fusion(
                (f_global, f_rep, f_multi)
            )
        else:
            scale_weights = None
            fusion_weights = None
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
            "scale_weights": scale_weights,
            "fusion_weights": fusion_weights,
        }

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

                avg_logits = 0.25 * (out1["logits"] + out2["logits"] + out3["logits"] + out4["logits"])
                avg_alpha = 0.25 * (out1["alpha"] + out2["alpha"] + out3["alpha"] + out4["alpha"])

                return {
                    "logits": avg_logits,
                    "alpha": avg_alpha,
                    "attn_maps": out1["attn_maps"],
                    "diversity_loss": out1["diversity_loss"],
                    "adj_matrix": out1["adj_matrix"],
                    "sparsity_loss": out1["sparsity_loss"],
                    "features": out1["features"],
                    "scale_weights": out1["scale_weights"],
                    "fusion_weights": out1["fusion_weights"],
                }
            else:
                # 2-crop Horizontal Flip TTA
                out_orig = self._forward_single(x)
                x_flipped = torch.flip(x, dims=[-1])
                out_flipped = self._forward_single(x_flipped)

                avg_logits = 0.5 * (out_orig["logits"] + out_flipped["logits"])
                avg_alpha = 0.5 * (out_orig["alpha"] + out_flipped["alpha"])

                return {
                    "logits": avg_logits,
                    "alpha": avg_alpha,
                    "attn_maps": out_orig["attn_maps"],
                    "diversity_loss": out_orig["diversity_loss"],
                    "adj_matrix": out_orig["adj_matrix"],
                    "sparsity_loss": out_orig["sparsity_loss"],
                    "features": out_orig["features"],
                    "scale_weights": out_orig["scale_weights"],
                    "fusion_weights": out_orig["fusion_weights"],
                }
        else:
            return self._forward_single(x, targets=targets, targets_b=targets_b, lam=lam)
