import torch
import torch.nn as nn
from .backbones import FacialBackbone
from .spatial_attention import MultiHeadSpatialAttention
from .scn_head import SCNHead


class AttentiveSCNFER(nn.Module):
    """
    Pure Image-Based Attentive Self-Cure Network for Facial Expression Recognition (FER).
    Zero dependency on bounding boxes, landmarks, or pre-extracted masks.
    
    Architecture Pipeline:
        Input: Raw grayscale images [B, 1, 48, 48]
          │
          ▼
        FacialBackbone (ResNet50 / ResNet34 adapted for 48x48)
          │ Feature Map F ∈ R^{B × C × 12 × 12}
          ├──► Global Average Pooling + Linear ──► f_global ∈ R^{B × D}
          │
          └──► MultiHeadSpatialAttention ────────► f_local ∈ R^{B × D} + Attention Maps
          │
          ▼
        Fusion: LayerNorm(f_global + f_local) ──► f_fused ∈ R^{B × D}
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
        num_attn_heads: int = 4,
        dropout: float = 0.25,
        use_pretrained: bool = True,
        pretrained_weights_path: str = "",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 1. Backbone adapted for 48x48
        self.backbone = FacialBackbone(
            backbone_name=backbone_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            pretrained_weights_path=pretrained_weights_path,
            target_feat_size=12,
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

        # 3. Local Spatial Attention Stream (Unsupervised discovery of Action Units)
        self.spatial_attention = MultiHeadSpatialAttention(
            in_channels=backbone_out_ch,
            embed_dim=embed_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
        )

        # 4. Fusion Layer
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_gate = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        # 5. SCN Head (Classifier + Confidence Weight)
        self.scn_head = SCNHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            init_confidence_bias=1.5,
        )

    def _forward_single(self, x: torch.Tensor):
        # Feature map: [B, C, 12, 12]
        feat_map = self.backbone(x)

        # Global feature: [B, D]
        f_global = self.global_proj(feat_map)

        # Local spatial feature & attention maps: [B, D]
        f_local, attn_maps, div_loss = self.spatial_attention(feat_map)

        # Gated residual fusion
        gate = torch.sigmoid(self.fusion_gate)
        f_fused = self.fusion_norm(f_global + gate * f_local)

        # SCN Head
        logits, alpha = self.scn_head(f_fused)

        return {
            "logits": logits,
            "alpha": alpha,
            "attn_maps": attn_maps,
            "diversity_loss": div_loss,
            "features": f_fused,
        }

    def forward(self, x: torch.Tensor, use_tta: bool = None):
        """
        Forward pass with automatic Horizontal Flip Test-Time Augmentation (TTA).
        When in eval mode (and use_tta is not explicitly False), automatically
        computes the average of logits from the original image and horizontally flipped image.
        """
        if use_tta is None:
            use_tta = not self.training

        if not self.training and use_tta:
            # Forward original
            out_orig = self._forward_single(x)
            # Forward flipped
            x_flipped = torch.flip(x, dims=[-1])
            out_flipped = self._forward_single(x_flipped)

            # Average logits for superior generalization
            avg_logits = 0.5 * (out_orig["logits"] + out_flipped["logits"])
            avg_alpha = 0.5 * (out_orig["alpha"] + out_flipped["alpha"])

            return {
                "logits": avg_logits,
                "alpha": avg_alpha,
                "attn_maps": out_orig["attn_maps"],
                "diversity_loss": out_orig["diversity_loss"],
                "features": out_orig["features"],
            }
        else:
            return self._forward_single(x)
