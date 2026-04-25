"""Motif-guided GraphSAGE-style model over selected subgraph nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.motif_guided_mlp import _masked_softmax


class MotifGraphSAGELayer(nn.Module):
    """Dense GraphSAGE mean aggregation for small fixed-K motif-filtered graphs."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3,
        use_edge_attr: bool = False,
        edge_attr_dim: int = 3,
    ) -> None:
        super().__init__()
        self.use_edge_attr = bool(use_edge_attr)
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_mlp = (
            nn.Sequential(nn.Linear(edge_attr_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
            if self.use_edge_attr
            else None
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor | None,
        node_mask: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, H = h.shape
        E = int(edge_index.shape[2]) if edge_index is not None else 0
        adj = torch.zeros(B, K, K, device=h.device, dtype=h.dtype)

        if E > 0:
            src = edge_index[:, 0, :].long().clamp(0, K - 1)
            dst = edge_index[:, 1, :].long().clamp(0, K - 1)
            if edge_valid is None:
                ev = torch.ones(B, E, device=h.device, dtype=h.dtype)
            else:
                ev = edge_valid.to(device=h.device, dtype=h.dtype)
            if self.use_edge_attr and edge_attr is not None:
                gate = torch.sigmoid(self.edge_mlp(edge_attr.to(device=h.device, dtype=h.dtype))).squeeze(-1)
                ev = ev * gate
            b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, E)
            adj[b_idx, dst, src] = ev

        deg = adj.sum(dim=2, keepdim=True).clamp_min(1.0)
        h_neigh = torch.bmm(adj / deg, h)
        out = self.lin(torch.cat([h, h_neigh], dim=-1))
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = (out + h) * node_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class MotifGuidedGNN(nn.Module):
    """
    GraphSAGE-style classifier over K motif-selected subgraph nodes.

    Expected batch keys:
        x, edge_index, edge_attr, edge_valid, mask, match_scores,
        matched_class, motif_score_vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gnn_hidden_dim: int | None = None,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        use_edge_attr: bool = False,
        edge_attr_dim: int = 3,
        use_motif_score_vector: bool = True,
        use_match_score_feature: bool = True,
        use_match_score_weighting: bool = True,
        pooling: str = "motif_attention",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(gnn_hidden_dim or hidden_dim)
        self.num_classes = int(num_classes)
        self.use_motif_score_vector = bool(use_motif_score_vector)
        self.use_match_score_feature = bool(use_match_score_feature)
        self.use_match_score_weighting = bool(use_match_score_weighting)
        self.pooling = pooling

        node_input_dim = self.input_dim + (1 if self.use_match_score_feature else 0) + self.num_classes
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList(
            [
                MotifGraphSAGELayer(
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                    use_edge_attr=use_edge_attr,
                    edge_attr_dim=edge_attr_dim,
                )
                for _ in range(max(0, int(num_layers)))
            ]
        )

        attn_dim = self.hidden_dim + (1 if self.use_match_score_weighting else 0)
        self.attn = nn.Linear(attn_dim, 1)
        final_dim = self.hidden_dim + (self.num_classes if self.use_motif_score_vector else 0)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _unpack(self, batch_or_x, **kwargs):
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            return (
                batch["x"],
                batch.get("edge_index"),
                batch.get("edge_attr"),
                batch.get("edge_valid"),
                batch.get("mask"),
                batch.get("match_scores"),
                batch.get("matched_class"),
                batch.get("motif_score_vector"),
            )
        return (
            batch_or_x,
            kwargs.get("edge_index"),
            kwargs.get("edge_attr"),
            kwargs.get("edge_valid"),
            kwargs.get("mask"),
            kwargs.get("match_scores"),
            kwargs.get("matched_class"),
            kwargs.get("motif_score_vector"),
        )

    def forward(self, batch_or_x, **kwargs) -> torch.Tensor:
        (
            x,
            edge_index,
            edge_attr,
            edge_valid,
            mask,
            match_scores,
            matched_class,
            motif_score_vector,
        ) = self._unpack(batch_or_x, **kwargs)

        if x.ndim != 3:
            raise ValueError(f"Expected x [B, K, D], got {tuple(x.shape)}")
        B, K, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")
        if edge_index is None:
            raise ValueError("MotifGuidedGNN requires edge_index")
        if edge_index.ndim != 3 or edge_index.shape[0] != B or edge_index.shape[1] != 2:
            raise ValueError(f"Expected edge_index [B, 2, E], got {tuple(edge_index.shape)}")

        device = x.device
        edge_index = edge_index.to(device=device)
        edge_attr = edge_attr.to(device=device, dtype=x.dtype) if edge_attr is not None else None
        edge_valid = edge_valid.to(device=device).bool() if edge_valid is not None else None

        if mask is None:
            mask = torch.ones(B, K, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device).bool()
        if match_scores is None:
            match_scores = torch.zeros(B, K, dtype=x.dtype, device=device)
        else:
            match_scores = match_scores.to(device=device, dtype=x.dtype)
        if matched_class is None:
            matched_class = torch.zeros(B, K, dtype=torch.long, device=device)
        else:
            matched_class = matched_class.to(device=device).long()
        if motif_score_vector is None:
            motif_score_vector = torch.zeros(B, self.num_classes, dtype=x.dtype, device=device)
        else:
            motif_score_vector = motif_score_vector.to(device=device, dtype=x.dtype)

        class_idx = matched_class.clamp(min=0, max=self.num_classes - 1)
        class_one_hot = F.one_hot(class_idx, num_classes=self.num_classes).to(dtype=x.dtype)
        class_one_hot = class_one_hot * mask.unsqueeze(-1).to(dtype=x.dtype)
        node_parts = [x]
        if self.use_match_score_feature:
            node_parts.append(match_scores.unsqueeze(-1))
        node_parts.append(class_one_hot)
        node_input = torch.cat(node_parts, dim=-1)

        h = self.node_encoder(node_input)
        h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_valid, mask, edge_attr=edge_attr)

        if self.pooling == "motif_attention":
            if self.use_match_score_weighting:
                attn_in = torch.cat([h, match_scores.unsqueeze(-1)], dim=-1)
            else:
                attn_in = h
            attn_scores = self.attn(attn_in).squeeze(-1)
            weights = _masked_softmax(attn_scores, mask, dim=1).unsqueeze(-1)
            h_graph = (h * weights).sum(dim=1)
        elif self.use_match_score_weighting:
            weights = _masked_softmax(match_scores, mask, dim=1).unsqueeze(-1)
            h_graph = (h * weights).sum(dim=1)
        elif self.pooling == "max":
            masked_h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
            h_graph = masked_h.max(dim=1).values
            h_graph = torch.where(torch.isfinite(h_graph), h_graph, torch.zeros_like(h_graph))
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
            h_graph = (h * mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1) / denom

        if self.use_motif_score_vector:
            h_graph = torch.cat([h_graph, motif_score_vector], dim=-1)

        logits = self.classifier(h_graph)
        return logits
