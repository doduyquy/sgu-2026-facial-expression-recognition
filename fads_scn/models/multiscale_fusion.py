from typing import Sequence, Tuple

import torch
import torch.nn as nn


class SqueezeExcitation2d(nn.Module):
    """Lightweight channel recalibration for one backbone stage."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class MultiScaleDualPooling(nn.Module):
    """
    Convert ConvNeXt stage maps into one expression descriptor.

    Each stage is channel-recalibrated, summarized with average and max
    pooling, projected to a shared embedding space, then combined with
    sample-adaptive scale attention.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        embed_dim: int,
        dropout: float = 0.0,
        se_reduction: int = 16,
    ):
        super().__init__()
        if len(feature_channels) < 2:
            raise ValueError("MultiScaleDualPooling requires at least two feature stages")

        self.feature_channels = tuple(feature_channels)
        self.num_scales = len(self.feature_channels)
        self.channel_attention = nn.ModuleList(
            SqueezeExcitation2d(channels, reduction=se_reduction)
            for channels in self.feature_channels
        )
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(2 * channels),
                nn.Linear(2 * channels, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for channels in self.feature_channels
        )

        self.scale_embedding = nn.Parameter(
            torch.zeros(1, self.num_scales, embed_dim)
        )
        score_dim = max(embed_dim // 4, 16)
        self.scale_score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, score_dim),
            nn.GELU(),
            nn.Linear(score_dim, 1),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.scale_embedding, std=0.02)

    def forward(
        self, feature_maps: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(feature_maps) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps, got {len(feature_maps)}"
            )

        scale_tokens = []
        for index, (feature_map, expected_channels, channel_attention, projection) in enumerate(
            zip(
                feature_maps,
                self.feature_channels,
                self.channel_attention,
                self.projections,
            )
        ):
            if feature_map.ndim != 4 or feature_map.shape[1] != expected_channels:
                raise ValueError(
                    f"Stage {index} must have shape [B, {expected_channels}, H, W], "
                    f"got {tuple(feature_map.shape)}"
                )

            attended = channel_attention(feature_map)
            avg_pool = torch.mean(attended, dim=(-2, -1))
            max_pool = torch.amax(attended, dim=(-2, -1))
            scale_tokens.append(projection(torch.cat((avg_pool, max_pool), dim=1)))

        tokens = torch.stack(scale_tokens, dim=1)
        scale_weights = torch.softmax(
            self.scale_score(tokens + self.scale_embedding).squeeze(-1), dim=1
        )
        fused = torch.sum(scale_weights.unsqueeze(-1) * tokens, dim=1)
        return self.output_norm(fused), scale_weights


class AdaptiveBranchFusion(nn.Module):
    """Fuse global, local/graph, and multi-scale branches per sample/channel."""

    def __init__(
        self,
        embed_dim: int,
        num_branches: int = 3,
        dropout: float = 0.0,
        initial_weights: Sequence[float] = (0.56, 0.35, 0.09),
    ):
        super().__init__()
        if len(initial_weights) != num_branches:
            raise ValueError("initial_weights must contain one value per branch")
        if any(weight <= 0 for weight in initial_weights):
            raise ValueError("initial_weights must be strictly positive")

        self.embed_dim = embed_dim
        self.num_branches = num_branches
        self.router = nn.Sequential(
            nn.LayerNorm(num_branches * embed_dim),
            nn.Linear(num_branches * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_branches * embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

        # Begin close to the legacy global/local ratio while giving the new
        # multi-scale branch a small, non-zero path that can learn immediately.
        final_linear = self.router[-1]
        nn.init.zeros_(final_linear.weight)
        priors = torch.as_tensor(initial_weights, dtype=final_linear.bias.dtype)
        priors = priors / priors.sum()
        bias = priors.log().view(num_branches, 1).expand(num_branches, embed_dim)
        with torch.no_grad():
            final_linear.bias.copy_(bias.reshape(-1))

    def forward(
        self, branch_features: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(branch_features) != self.num_branches:
            raise ValueError(
                f"Expected {self.num_branches} branches, got {len(branch_features)}"
            )
        reference_shape = branch_features[0].shape
        if len(reference_shape) != 2 or reference_shape[1] != self.embed_dim:
            raise ValueError(
                f"Branch features must have shape [B, {self.embed_dim}]"
            )
        if any(feature.shape != reference_shape for feature in branch_features[1:]):
            raise ValueError("All branch features must have the same shape")

        branches = torch.stack(tuple(branch_features), dim=1)
        routing_logits = self.router(torch.cat(tuple(branch_features), dim=1))
        routing_logits = routing_logits.view(-1, self.num_branches, self.embed_dim)
        channel_weights = torch.softmax(routing_logits, dim=1)
        fused = torch.sum(channel_weights * branches, dim=1)

        # Mean branch contribution is returned only as an interpretable diagnostic.
        branch_weights = channel_weights.mean(dim=-1)
        return self.output_norm(fused), branch_weights
