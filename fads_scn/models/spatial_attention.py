import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSpatialAttention(nn.Module):
    """
    Unsupervised Multi-Head Spatial Attention (Distract-free / Self-Attention style).
    Automatically discovers and focuses on salient facial Action Unit regions
    (e.g., left eye, right eye, brow wrinkles, mouth corner) without any bounding boxes or landmarks.
    
    Given Feature Map F of shape [B, C, H, W]:
    1. Generates M spatial attention maps A_m (m = 1..M) via 2D Conv.
    2. Uses Softmax across spatial dimensions (H x W) so each head forms a smooth probability distribution.
    3. Aggregates local features per head and projects to embed_dim.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Feature value projector V: [B, C, H, W] -> [B, embed_dim, H, W]
        self.value_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 2. Attention map generator: [B, C, H, W] -> [B, num_heads, H, W]
        # Uses a depthwise-separable conv block for expressive spatial receptive field
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, num_heads, kernel_size=1, bias=True),
        )

        # 3. Aggregation projector
        self.out_proj = nn.Sequential(
            nn.Linear(num_heads * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feat_map: torch.Tensor):
        """
        Args:
            feat_map: [B, C, H, W]
        Returns:
            f_local: [B, embed_dim]
            attn_maps: [B, num_heads, H, W] (normalized spatial attention weights)
            diversity_loss: scalar tensor measuring head overlap (encourages heads to attend to different regions)
        """
        B, C, H, W = feat_map.shape

        # Values: [B, embed_dim, H, W]
        values = self.value_proj(feat_map)

        # Raw attention maps: [B, num_heads, H, W]
        raw_attn = self.attn_conv(feat_map)

        # Spatial softmax: normalize over H * W for each head
        raw_flat = raw_attn.view(B, self.num_heads, H * W)
        attn_weights = F.softmax(raw_flat, dim=-1)  # [B, num_heads, H * W]
        attn_maps = attn_weights.view(B, self.num_heads, H, W)

        # Weighted spatial pooling per head
        # values_flat: [B, embed_dim, H * W]
        values_flat = values.view(B, self.embed_dim, H * W)
        
        # head_feats: [B, num_heads, embed_dim]
        # einsum: b m s, b d s -> b m d (m = num_heads, d = embed_dim, s = H*W)
        head_feats = torch.einsum("bms,bds->bmd", attn_weights, values_flat)
        
        # Flatten across heads: [B, num_heads * embed_dim]
        f_concat = head_feats.view(B, self.num_heads * self.embed_dim)
        f_local = self.out_proj(f_concat)

        # Head Diversity Regularizer: penalize overlap between different attention heads
        # normalized cross-head similarity: [B, num_heads, num_heads]
        weights_norm = F.normalize(attn_weights, p=2, dim=-1)
        sim_matrix = torch.bmm(weights_norm, weights_norm.transpose(1, 2))  # [B, M, M]
        identity = torch.eye(self.num_heads, device=feat_map.device).unsqueeze(0)
        # Off-diagonal elements should be close to zero
        diversity_loss = ((sim_matrix - identity) ** 2).sum(dim=(-1, -2)).mean() / (self.num_heads * (self.num_heads - 1) + 1e-6)

        return f_local, attn_maps, diversity_loss, head_feats
