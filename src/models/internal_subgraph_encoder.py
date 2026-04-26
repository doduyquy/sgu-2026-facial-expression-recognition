"""Dense internal GNN encoder for small pixel subgraphs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGraphSAGELayer(nn.Module):
    """GraphSAGE mean aggregation on dense per-subgraph adjacency tensors."""

    def __init__(self, hidden_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # h: [M, N, H], adj: [M, N, N], node_mask: [M, N]
        mask_f = node_mask.unsqueeze(-1).to(dtype=h.dtype)
        adj = adj.to(dtype=h.dtype)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        adj = adj * pair_mask.to(dtype=h.dtype)

        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        h_neigh = torch.bmm(adj / deg, h)
        out = self.lin(torch.cat([h, h_neigh], dim=-1))
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = (out + h) * mask_f
        return out


class InternalPixelSubgraphEncoder(nn.Module):
    """
    Encode K selected pixel subgraphs per image.

    Input:
        sub_x:         [B, K, N, F]
        sub_adj:       [B, K, N, N]
        sub_node_mask: [B, K, N]

    Output:
        z: [B, K, out_dim]
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        readout: str = "mean_max",
        use_edge_attr: bool = False,
    ) -> None:
        super().__init__()
        if use_edge_attr:
            raise NotImplementedError("internal_use_edge_attr is reserved for a later version.")
        readout = str(readout).lower()
        if readout not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported internal_readout={readout!r}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.readout = readout

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [DenseGraphSAGELayer(self.hidden_dim, dropout=dropout) for _ in range(max(0, int(num_layers)))]
        )
        if self.readout == "mean_max":
            self.readout_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.readout_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, sub_x: torch.Tensor, sub_adj: torch.Tensor, sub_node_mask: torch.Tensor) -> torch.Tensor:
        if sub_x.ndim != 4:
            raise ValueError(f"Expected sub_x [B, K, N, F], got {tuple(sub_x.shape)}")
        if sub_adj.ndim != 4:
            raise ValueError(f"Expected sub_adj [B, K, N, N], got {tuple(sub_adj.shape)}")
        if sub_node_mask.ndim != 3:
            raise ValueError(f"Expected sub_node_mask [B, K, N], got {tuple(sub_node_mask.shape)}")

        B, K, N, F_in = sub_x.shape
        if F_in != self.input_dim:
            raise ValueError(f"Expected internal input_dim={self.input_dim}, got {F_in}")
        if sub_adj.shape[:3] != (B, K, N) or sub_adj.shape[3] != N:
            raise ValueError(
                f"sub_adj must match [B, K, N, N]={B, K, N, N}, got {tuple(sub_adj.shape)}"
            )
        if sub_node_mask.shape != (B, K, N):
            raise ValueError(f"sub_node_mask must match [B, K, N]={B, K, N}, got {tuple(sub_node_mask.shape)}")

        flat_x = sub_x.reshape(B * K, N, F_in)
        flat_adj = sub_adj.reshape(B * K, N, N)
        flat_mask = sub_node_mask.reshape(B * K, N).bool()

        h = self.input_proj(flat_x)
        h = h * flat_mask.unsqueeze(-1).to(dtype=h.dtype)
        flat_adj = torch.nan_to_num(flat_adj, nan=0.0, posinf=0.0, neginf=0.0)
        for layer in self.layers:
            h = layer(h, flat_adj, flat_mask)

        mean = self._masked_mean(h, flat_mask)
        if self.readout == "mean":
            pooled = mean
        else:
            max_pooled = self._masked_max(h, flat_mask)
            pooled = max_pooled if self.readout == "max" else torch.cat([mean, max_pooled], dim=-1)

        z = self.readout_proj(pooled)
        return z.reshape(B, K, self.out_dim)

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (h * mask_f).sum(dim=1) / denom

    @staticmethod
    def _masked_max(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
        values = masked_h.max(dim=1).values
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))
