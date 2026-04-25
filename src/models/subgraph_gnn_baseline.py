"""
src/models/subgraph_gnn_baseline.py
GNN baseline trên subgraph-level graph.

Kiến trúc:
    x [K, D] + edge_index
    → 2 lớp GraphSAGE (mean aggregation) hoặc GCN (tùy config)
    → global mean pool (dùng mask)
    → Linear classifier 7 lớp

Không dùng PyTorch Geometric để tránh dependency nặng.
Tự triển khai GraphSAGE mean và GCN (D → H → H → num_classes).

Input batch (từ collate_fn_gnn):
    x          : [B, K, D]
    edge_index  : [B, 2, E]   — mỗi sample có E edges (zero-padded nếu cần)
    edge_mask   : [B, E]      — 1 = edge hợp lệ
    mask        : [B, K]      — 1 = node hợp lệ
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# GraphSAGE layer (mean aggregation, không dùng PyG)
# ────────────────────────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE mean aggregation layer.
    h_i' = ReLU( W_self * h_i  +  W_neigh * mean(h_j for j in N(i)) )
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=True)
        self.norm      = nn.LayerNorm(out_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # [B, K, D]
        edge_index: torch.Tensor,    # [B, 2, E]
        edge_valid: torch.Tensor,    # [B, E] bool
        node_mask: torch.Tensor,     # [B, K] float (1=valid)
    ) -> torch.Tensor:               # [B, K, out_dim]
        B, K, D = x.shape

        # ── Neighbour aggregation ──────────────────────────────────────────
        # Dùng sparse scatter: với mỗi batch, duyệt edges và cộng source vào dest
        # Cách dense: build adj matrix [B, K, K] → matmul
        # Đây là cách dense, phù hợp với K nhỏ (32-64)

        # edge_index: [B, 2, E] → src=[B,E], dst=[B,E]
        E = edge_index.shape[2]
        src = edge_index[:, 0, :]   # [B, E]
        dst = edge_index[:, 1, :]   # [B, E]

        # Build dense adjacency [B, K, K]
        adj = torch.zeros(B, K, K, device=x.device, dtype=x.dtype)
        if E > 0:
            # Chỉ set edge hợp lệ
            ev = edge_valid.float()           # [B, E]
            b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, E)
            adj[b_idx, dst, src] = ev          # dst nhận từ src

        # Chuẩn hóa theo degree
        deg = adj.sum(dim=2, keepdim=True).clamp_min(1.0)   # [B, K, 1]
        adj_norm = adj / deg                                   # [B, K, K]

        # Aggregate: [B, K, D] = [B, K, K] @ [B, K, D]
        h_neigh = torch.bmm(adj_norm, x)   # [B, K, D]

        # ── Combine ──────────────────────────────────────────────────────
        out = self.lin_self(x) + self.lin_neigh(h_neigh)   # [B, K, out_dim]
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Mask padding nodes thành 0
        out = out * node_mask.unsqueeze(-1)
        return out


# ────────────────────────────────────────────────────────────────────────────
# SubgraphGNNBaseline
# ────────────────────────────────────────────────────────────────────────────

class SubgraphGNNBaseline(nn.Module):
    """
    GNN baseline trên subgraph-level graph.

    Input batch keys:
        x          : [B, K, D]
        edge_index  : [B, 2, E]
        edge_valid  : [B, E]
        mask        : [B, K]

    Output:
        logits : [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        gnn_hidden  = hidden_dims[0] if hidden_dims else 64

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, gnn_hidden),
            nn.LayerNorm(gnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── GNN layers ───────────────────────────────────────────────────
        gnn_layers = max(1, gnn_layers)
        self.gnn = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn.append(GraphSAGELayer(gnn_hidden, gnn_hidden, dropout=dropout))

        # ── Classifier ───────────────────────────────────────────────────
        dims = [gnn_hidden] + hidden_dims[1:] + [num_classes]
        clf_layers = []
        for i in range(len(dims) - 1):
            clf_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                clf_layers.append(nn.ReLU())
                clf_layers.append(nn.Dropout(dropout))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x          : [B, K, D]
        edge_index : [B, 2, E]
        edge_valid : [B, E]   bool or float
        mask       : [B, K]   float (1=valid, 0=pad)
        """
        B, K, _ = x.shape

        if mask is None:
            mask = torch.ones(B, K, device=x.device, dtype=x.dtype)
        else:
            mask = mask.float()

        # Input projection
        h = self.input_proj(x)                       # [B, K, gnn_hidden]
        h = h * mask.unsqueeze(-1)

        # GNN message passing
        ev = edge_valid.float() if edge_valid.dtype != torch.float32 else edge_valid
        for layer in self.gnn:
            h = layer(h, edge_index, ev, mask)       # [B, K, gnn_hidden]

        # Global mean pool (masked)
        denom  = mask.sum(dim=1, keepdim=True).clamp_min(1.0)   # [B, 1]
        h_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / denom     # [B, gnn_hidden]

        logits = self.classifier(h_pool)             # [B, num_classes]
        return logits
