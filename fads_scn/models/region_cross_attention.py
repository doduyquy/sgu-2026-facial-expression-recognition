from typing import Dict

import torch
import torch.nn as nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest practical GroupNorm group count that divides channels."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TopKRegionCrossAttention(nn.Module):
    """
    Discover spatially separated C3 regions and enrich them with C4 context.

    C3 preserves fine facial details, while C4 supplies semantic context. A
    learned saliency map selects top-k locations with non-maximum suppression;
    their local C3 descriptors act as queries over all C4 spatial tokens.
    """

    def __init__(
        self,
        detail_channels: int,
        context_channels: int,
        embed_dim: int = 256,
        num_regions: int = 6,
        num_heads: int = 4,
        suppression_radius: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if detail_channels <= 0 or context_channels <= 0 or embed_dim <= 0:
            raise ValueError("Feature channels and embed_dim must be positive")
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        if suppression_radius < 0:
            raise ValueError("suppression_radius must be non-negative")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.num_heads = num_heads
        self.suppression_radius = suppression_radius

        saliency_channels = max(detail_channels // 4, 32)
        self.saliency = nn.Sequential(
            nn.Conv2d(
                detail_channels,
                saliency_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(saliency_channels), saliency_channels),
            nn.GELU(),
            nn.Conv2d(saliency_channels, 1, kernel_size=1),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(detail_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(context_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.GELU(),
        )
        self.coordinate_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.region_readout = nn.Linear(embed_dim, 1)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _coordinates(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
        return coordinates.expand(batch_size, -1, -1)

    def _select_spaced_topk(self, saliency_logits: torch.Tensor) -> torch.Tensor:
        """Select top-k indices while suppressing neighboring locations."""
        batch_size, height, width = saliency_logits.shape
        spatial_size = height * width
        if self.num_regions > spatial_size:
            raise ValueError(
                f"num_regions={self.num_regions} exceeds spatial size {height}x{width}"
            )

        original = saliency_logits.detach().flatten(1)
        ranking = original.clone()
        selected = []
        grid_y = torch.arange(height, device=saliency_logits.device).view(1, height, 1)
        grid_x = torch.arange(width, device=saliency_logits.device).view(1, 1, width)

        for _ in range(self.num_regions):
            # A large suppression radius can exhaust all candidates. In that
            # case, fall back to the best location not selected exactly before.
            has_candidate = torch.isfinite(ranking).any(dim=1)
            if selected:
                fallback = original.clone()
                fallback.scatter_(1, torch.stack(selected, dim=1), -torch.inf)
                ranking = torch.where(has_candidate.unsqueeze(1), ranking, fallback)

            index = ranking.argmax(dim=1)
            selected.append(index)

            center_y = torch.div(index, width, rounding_mode="floor")
            center_x = index.remainder(width)
            suppress = (
                (grid_y - center_y[:, None, None]).abs() <= self.suppression_radius
            ) & (
                (grid_x - center_x[:, None, None]).abs() <= self.suppression_radius
            )
            ranking = ranking.masked_fill(suppress.flatten(1), -torch.inf)

        return torch.stack(selected, dim=1)

    def forward(
        self,
        detail_map: torch.Tensor,
        context_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if detail_map.ndim != 4 or context_map.ndim != 4:
            raise ValueError("detail_map and context_map must be 4D feature maps")
        if detail_map.shape[0] != context_map.shape[0]:
            raise ValueError("detail_map and context_map must share the batch size")

        batch_size, _, detail_h, detail_w = detail_map.shape
        _, _, context_h, context_w = context_map.shape

        saliency_logits = self.saliency(detail_map).squeeze(1)
        region_indices = self._select_spaced_topk(saliency_logits)

        detail_tokens = self.detail_proj(detail_map).flatten(2).transpose(1, 2)
        context_tokens = self.context_proj(context_map).flatten(2).transpose(1, 2)
        detail_coordinates = self._coordinates(
            batch_size, detail_h, detail_w, detail_map.device, detail_tokens.dtype
        )
        context_coordinates = self._coordinates(
            batch_size, context_h, context_w, context_map.device, context_tokens.dtype
        )
        detail_tokens = detail_tokens + self.coordinate_proj(detail_coordinates)
        context_tokens = context_tokens + self.coordinate_proj(context_coordinates)

        gather_index = region_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        region_queries = torch.gather(detail_tokens, dim=1, index=gather_index)
        region_locations = torch.gather(
            detail_coordinates,
            dim=1,
            index=region_indices.unsqueeze(-1).expand(-1, -1, 2),
        )

        attended_regions, context_attention = self.cross_attention(
            query=region_queries,
            key=context_tokens,
            value=context_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        region_tokens = self.attention_norm(
            region_queries + self.attention_dropout(attended_regions)
        )
        region_tokens = self.ffn_norm(region_tokens + self.ffn(region_tokens))

        selected_saliency = torch.gather(
            saliency_logits.flatten(1), dim=1, index=region_indices
        )
        readout_logits = selected_saliency + self.region_readout(region_tokens).squeeze(-1)
        region_weights = torch.softmax(readout_logits, dim=1)
        region_feature = torch.sum(region_weights.unsqueeze(-1) * region_tokens, dim=1)
        region_feature = self.output_proj(region_feature)

        return {
            "features": region_feature,
            "indices": region_indices,
            "locations": region_locations,
            "weights": region_weights,
            "attention_maps": context_attention.view(
                batch_size, self.num_regions, context_h, context_w
            ),
            "saliency_map": torch.softmax(saliency_logits.flatten(1), dim=1).view(
                batch_size, detail_h, detail_w
            ),
        }
