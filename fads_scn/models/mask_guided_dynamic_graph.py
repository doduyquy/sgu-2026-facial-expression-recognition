"""Mask-guided dynamic region graph used by the M4 FER model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest useful GroupNorm divisor up to ``max_groups``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualMaskRefiner(nn.Module):
    """A compact bottom-up/top-down mask branch for the final feature map.

    The refiner keeps an identity path and applies the learned mask as
    ``F_refined = F * (1 + sigmoid(mask_logits))``.  A negative mask-head bias
    makes the module begin close to the identity mapping.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        groups = _group_count(embed_dim)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, embed_dim),
            nn.GELU(),
        )
        self.enc = DepthwiseSeparableBlock(embed_dim)
        self.down1 = DepthwiseSeparableBlock(embed_dim, stride=2)
        self.down2 = DepthwiseSeparableBlock(embed_dim, stride=2)
        self.bottleneck = DepthwiseSeparableBlock(embed_dim)
        self.decode1 = DepthwiseSeparableBlock(embed_dim)
        self.decode2 = DepthwiseSeparableBlock(embed_dim)
        self.mask_head = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)

        nn.init.normal_(self.mask_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.mask_head.bias, -2.0)

    def forward(self, feat_map: torch.Tensor):
        base = self.input_proj(feat_map)
        skip_high = self.enc(base)
        skip_mid = self.down1(skip_high)
        low = self.down2(skip_mid)
        low = self.bottleneck(low)

        decoded = F.interpolate(
            low, size=skip_mid.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decode1(decoded + skip_mid)
        decoded = F.interpolate(
            decoded, size=skip_high.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decode2(decoded + skip_high)

        channel_mask = torch.sigmoid(self.mask_head(decoded))
        refined = base * (1.0 + channel_mask)
        spatial_mask = channel_mask.mean(dim=1, keepdim=True)
        return refined, spatial_mask


class CompetitiveRegionTokenizer(nn.Module):
    """Convert a feature map into mutually competitive, landmark-free nodes."""

    def __init__(self, embed_dim: int, num_nodes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.coord_proj = nn.Linear(2, embed_dim, bias=False)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.node_norm = nn.LayerNorm(embed_dim)
        self.node_queries = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.trunc_normal_(self.node_queries, std=0.02)

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)

    def forward(self, feat_map: torch.Tensor):
        batch_size, channels, height, width = feat_map.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Expected {self.embed_dim} feature channels, received {channels}"
            )

        features = feat_map.flatten(2).transpose(1, 2)  # [B, S, D]
        coords = self._coordinate_grid(
            height, width, feat_map.device, feat_map.dtype
        )  # [S, 2]

        keys = self.key_proj(features) + self.coord_proj(coords).unsqueeze(0)
        keys = self.key_norm(keys)
        queries = self.query_norm(self.node_queries)
        assignment_logits = torch.einsum("bsd,kd->bks", keys, queries)
        assignment_logits = assignment_logits / math.sqrt(self.embed_dim)

        # Competition is across nodes for every spatial location.  The second
        # normalization turns assignments into stable weighted pooling maps.
        assignments = F.softmax(assignment_logits, dim=1)  # [B, K, S]
        region_mass = assignments.mean(dim=-1)  # [B, K]
        spatial_weights = assignments / assignments.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        values = self.value_proj(features)
        nodes = torch.einsum("bks,bsd->bkd", spatial_weights, values)
        nodes = self.node_norm(nodes + self.node_queries.unsqueeze(0))

        centers = torch.einsum("bks,sd->bkd", spatial_weights, coords)
        centered_coords = coords.view(1, 1, height * width, 2) - centers.unsqueeze(2)
        variance = torch.einsum(
            "bks,bksd->bkd", spatial_weights, centered_coords.square()
        )
        spread = torch.sqrt(variance.clamp_min(1e-6))
        geometry = torch.cat([centers, spread, region_mass.unsqueeze(-1)], dim=-1)

        normalized_maps = F.normalize(spatial_weights, p=2, dim=-1)
        similarity = torch.bmm(normalized_maps, normalized_maps.transpose(1, 2))
        identity = torch.eye(self.num_nodes, device=feat_map.device, dtype=feat_map.dtype)
        off_diagonal = similarity - identity.unsqueeze(0)
        overlap_loss = off_diagonal.square().sum(dim=(-1, -2)).mean()
        overlap_loss = overlap_loss / max(self.num_nodes * (self.num_nodes - 1), 1)

        target_mass = 1.0 / self.num_nodes
        balance_loss = ((region_mass - target_mass) / target_mass).square().mean()
        diversity_loss = overlap_loss + 0.1 * balance_loss

        attention_maps = spatial_weights.view(
            batch_size, self.num_nodes, height, width
        )
        return nodes, geometry, attention_maps, diversity_loss


class DynamicEdgeGraphBlock(nn.Module):
    """Sparse relation-aware message passing with input-dependent topology."""

    def __init__(
        self,
        embed_dim: int,
        top_k: int,
        num_heads: int,
        dropout: float,
        geometry_dim: int = 5,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by the graph head count")

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.geometry_dim = geometry_dim
        self.edge_geometry_dim = 5 + 2 * (geometry_dim - 2)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.semantic_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.topology_bias = nn.Sequential(
            nn.Linear(self.edge_geometry_dim, max(embed_dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(embed_dim // 4, 16), 1),
        )

        edge_input_dim = 3 * embed_dim + self.edge_geometry_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_attention = nn.Linear(embed_dim, num_heads)
        self.edge_value = nn.Linear(embed_dim, embed_dim)
        self.message_out = nn.Linear(embed_dim, embed_dim)
        self.message_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _pair_geometry(self, geometry: torch.Tensor) -> torch.Tensor:
        num_nodes = geometry.size(1)
        centers = geometry[..., :2]
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)  # c_j - c_i
        abs_delta = delta.abs()
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

        extras = geometry[..., 2:]
        center_extras = extras.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        neighbor_extras = extras.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        return torch.cat(
            [delta, abs_delta, distance, center_extras, neighbor_extras], dim=-1
        )

    def forward(self, nodes: torch.Tensor, geometry: torch.Tensor):
        batch_size, num_nodes, embed_dim = nodes.shape
        if num_nodes < 2:
            raise ValueError("Dynamic graph reasoning requires at least two nodes")
        top_k = min(self.top_k, num_nodes - 1)

        normalized_nodes = self.norm1(nodes)
        semantic = F.normalize(self.semantic_proj(normalized_nodes), p=2, dim=-1)
        semantic_similarity = torch.bmm(semantic, semantic.transpose(1, 2))

        pair_geometry = self._pair_geometry(geometry)
        topology_scores = semantic_similarity + self.topology_bias(pair_geometry).squeeze(-1)
        diagonal = torch.eye(num_nodes, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        topology_scores = topology_scores.masked_fill(diagonal, torch.finfo(nodes.dtype).min)
        neighbor_scores, neighbor_indices = torch.topk(
            topology_scores, k=top_k, dim=-1
        )

        all_neighbors = normalized_nodes.unsqueeze(1).expand(
            batch_size, num_nodes, num_nodes, embed_dim
        )
        gather_nodes = neighbor_indices.unsqueeze(-1).expand(
            batch_size, num_nodes, top_k, embed_dim
        )
        neighbor_nodes = torch.gather(all_neighbors, dim=2, index=gather_nodes)
        center_nodes = normalized_nodes.unsqueeze(2).expand(
            batch_size, num_nodes, top_k, embed_dim
        )

        gather_geometry = neighbor_indices.unsqueeze(-1).expand(
            batch_size, num_nodes, top_k, self.edge_geometry_dim
        )
        selected_geometry = torch.gather(
            pair_geometry, dim=2, index=gather_geometry
        )
        edge_input = torch.cat(
            [
                center_nodes,
                neighbor_nodes - center_nodes,
                neighbor_nodes * center_nodes,
                selected_geometry,
            ],
            dim=-1,
        )
        edge_features = self.edge_encoder(edge_input)

        attention_logits = self.edge_attention(edge_features)
        attention_logits = attention_logits + neighbor_scores.unsqueeze(-1)
        attention = F.softmax(attention_logits, dim=2)  # [B, N, k, heads]

        edge_values = self.edge_value(edge_features).view(
            batch_size, num_nodes, top_k, self.num_heads, self.head_dim
        )
        message = (attention.unsqueeze(-1) * edge_values).sum(dim=2)
        message = message.reshape(batch_size, num_nodes, embed_dim)
        nodes = nodes + self.message_dropout(self.message_out(message))
        nodes = nodes + self.ffn(self.norm2(nodes))

        mean_attention = attention.mean(dim=-1)
        adjacency = nodes.new_zeros(batch_size, num_nodes, num_nodes)
        adjacency.scatter_add_(dim=-1, index=neighbor_indices, src=mean_attention)

        safe_attention = mean_attention.clamp_min(torch.finfo(nodes.dtype).eps)
        entropy = -(mean_attention * safe_attention.log()).sum(dim=-1)
        entropy_loss = entropy.mean() / math.log(max(top_k, 2))
        return nodes, adjacency, entropy_loss


class ClassConditionedGraphReadout(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.node_norm = nn.LayerNorm(embed_dim)
        self.class_queries = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.class_weights = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.graph_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        nn.init.trunc_normal_(self.class_weights, std=0.02)

    def forward(self, nodes: torch.Tensor):
        normalized_nodes = self.node_norm(nodes)
        class_attention_logits = torch.einsum(
            "bnd,cd->bcn", normalized_nodes, self.class_queries
        ) / math.sqrt(self.embed_dim)
        class_attention = F.softmax(class_attention_logits, dim=-1)
        class_features = torch.einsum(
            "bcn,bnd->bcd", class_attention, normalized_nodes
        )
        residual_logits = (
            class_features * self.class_weights.unsqueeze(0)
        ).sum(dim=-1) / math.sqrt(self.embed_dim)
        residual_logits = residual_logits + self.class_bias.unsqueeze(0)
        graph_feature = self.graph_norm(class_features.mean(dim=1))
        return residual_logits, graph_feature, class_attention


class ClasswiseResidualGate(nn.Module):
    """Predict an image- and class-dependent contribution for graph logits."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float,
        init_bias: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(embed_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(
        self, global_feature: torch.Tensor, graph_feature: torch.Tensor
    ) -> torch.Tensor:
        gate_logits = self.net(torch.cat([global_feature, graph_feature], dim=-1))
        return torch.sigmoid(gate_logits)


class MaskGuidedDynamicRegionGraph(nn.Module):
    """Complete M4 branch: residual mask, tokenization, graph, and readout."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_nodes: int,
        num_classes: int,
        graph_depth: int = 3,
        top_k: int = 3,
        edge_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if graph_depth < 1:
            raise ValueError("graph_depth must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.num_nodes = num_nodes
        self.mask_refiner = ResidualMaskRefiner(in_channels, embed_dim)
        self.tokenizer = CompetitiveRegionTokenizer(embed_dim, num_nodes)
        self.graph_blocks = nn.ModuleList(
            [
                DynamicEdgeGraphBlock(
                    embed_dim=embed_dim,
                    top_k=top_k,
                    num_heads=edge_heads,
                    dropout=dropout,
                )
                for _ in range(graph_depth)
            ]
        )
        self.readout = ClassConditionedGraphReadout(embed_dim, num_classes)

    def forward(self, feat_map: torch.Tensor):
        refined, mask_map = self.mask_refiner(feat_map)
        nodes, geometry, attention_maps, diversity_loss = self.tokenizer(refined)

        adjacency = None
        adjacency_history = []
        entropy_losses = []
        for graph_block in self.graph_blocks:
            nodes, adjacency, entropy_loss = graph_block(nodes, geometry)
            adjacency_history.append(adjacency)
            entropy_losses.append(entropy_loss)

        sparsity_loss = torch.stack(entropy_losses).mean()
        residual_logits, graph_feature, class_attention = self.readout(nodes)
        return {
            "residual_logits": residual_logits,
            "graph_feature": graph_feature,
            "attn_maps": attention_maps,
            "mask_map": mask_map,
            "adj_matrix": adjacency,
            "adj_matrices": torch.stack(adjacency_history, dim=1),
            "class_attention": class_attention,
            "region_geometry": geometry,
            "diversity_loss": diversity_loss,
            "sparsity_loss": sparsity_loss,
        }
