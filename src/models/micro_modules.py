
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_softmax


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


class MicroSemanticMotifBank(nn.Module):

    def __init__(self, num_regions: int, motifs_per_region: int, state_dim: int):
        super().__init__()
        self.num_regions = num_regions
        self.motifs_per_region = motifs_per_region
        self.state_dim = state_dim
        self.motifs = nn.Parameter(torch.randn(num_regions, motifs_per_region, state_dim) * 0.02)

    def forward(self) -> torch.Tensor:
        return self.motifs


class MicroSemanticMotifMatcher(nn.Module):

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
