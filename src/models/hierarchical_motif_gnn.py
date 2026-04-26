"""Hierarchical motif GNN: internal pixel-subgraph GNN plus motif-level GNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.internal_subgraph_encoder import InternalPixelSubgraphEncoder
from src.models.motif_guided_gnn import MotifGraphSAGELayer
from src.models.motif_guided_mlp import _masked_softmax


class HierarchicalMotifGNN(nn.Module):
    """
    Encode selected pixel subgraphs internally, then classify with motif-level GraphSAGE.

    Expected batch keys:
        x, sub_x, sub_adj, sub_node_mask, mask, edge_index, edge_valid,
        match_scores, matched_class, matched_disc_score, motif_score_vector
    """

    def __init__(
        self,
        num_classes: int = 7,
        internal_input_dim: int = 7,
        internal_hidden_dim: int = 64,
        internal_out_dim: int = 128,
        internal_num_layers: int = 2,
        internal_dropout: float = 0.25,
        internal_readout: str = "mean_max",
        internal_use_edge_attr: bool = False,
        use_descriptor: bool = True,
        descriptor_dim: int = 41,
        use_match_score_feature: bool = True,
        use_disc_score_feature: bool = True,
        use_matched_class_onehot: bool = True,
        motif_hidden_dim: int = 128,
        motif_num_layers: int = 2,
        motif_dropout: float = 0.3,
        motif_use_edge_attr: bool = False,
        motif_edge_attr_dim: int = 3,
        pooling: str = "motif_attention",
        use_motif_score_vector: bool = True,
        use_match_score_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.internal_out_dim = int(internal_out_dim)
        self.use_descriptor = bool(use_descriptor)
        self.descriptor_dim = int(descriptor_dim)
        self.use_match_score_feature = bool(use_match_score_feature)
        self.use_disc_score_feature = bool(use_disc_score_feature)
        self.use_matched_class_onehot = bool(use_matched_class_onehot)
        self.use_motif_score_vector = bool(use_motif_score_vector)
        self.use_match_score_weighting = bool(use_match_score_weighting)
        self.pooling = pooling
        self.hidden_dim = int(motif_hidden_dim)

        self.internal_encoder = InternalPixelSubgraphEncoder(
            input_dim=internal_input_dim,
            hidden_dim=internal_hidden_dim,
            out_dim=internal_out_dim,
            num_layers=internal_num_layers,
            dropout=internal_dropout,
            readout=internal_readout,
            use_edge_attr=internal_use_edge_attr,
        )

        node_input_dim = self.internal_out_dim
        if self.use_descriptor:
            node_input_dim += self.descriptor_dim
        if self.use_match_score_feature:
            node_input_dim += 1
        if self.use_disc_score_feature:
            node_input_dim += 1
        if self.use_matched_class_onehot:
            node_input_dim += self.num_classes
        self.node_input_dim = int(node_input_dim)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(motif_dropout),
        )
        self.gnn_layers = nn.ModuleList(
            [
                MotifGraphSAGELayer(
                    hidden_dim=self.hidden_dim,
                    dropout=motif_dropout,
                    use_edge_attr=motif_use_edge_attr,
                    edge_attr_dim=motif_edge_attr_dim,
                )
                for _ in range(max(0, int(motif_num_layers)))
            ]
        )
        attn_dim = self.hidden_dim + (1 if self.use_match_score_weighting else 0)
        self.attn = nn.Linear(attn_dim, 1)
        final_dim = self.hidden_dim + (self.num_classes if self.use_motif_score_vector else 0)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(motif_dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, batch_or_x, return_aux: bool = False, **kwargs):
        batch = batch_or_x if isinstance(batch_or_x, dict) else {"x": batch_or_x, **kwargs}

        x = batch.get("x")
        sub_x = batch.get("sub_x")
        sub_adj = batch.get("sub_adj")
        sub_node_mask = batch.get("sub_node_mask")
        edge_index = batch.get("edge_index")
        edge_attr = batch.get("edge_attr")
        edge_valid = batch.get("edge_valid")
        mask = batch.get("mask")
        match_scores = batch.get("match_scores")
        matched_class = batch.get("matched_class")
        matched_disc_score = batch.get("matched_disc_score")
        motif_score_vector = batch.get("motif_score_vector")

        if sub_x is None or sub_adj is None or sub_node_mask is None:
            raise KeyError("HierarchicalMotifGNN requires sub_x, sub_adj, and sub_node_mask in the batch.")
        if edge_index is None:
            raise ValueError("HierarchicalMotifGNN requires motif-level edge_index.")

        device = sub_x.device
        dtype = sub_x.dtype
        B, K = sub_x.shape[:2]

        if mask is None:
            mask = torch.ones(B, K, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device).bool()
        edge_index = edge_index.to(device=device)
        edge_attr = edge_attr.to(device=device, dtype=dtype) if edge_attr is not None else None
        edge_valid = edge_valid.to(device=device).bool() if edge_valid is not None else None

        z_internal = self.internal_encoder(sub_x, sub_adj, sub_node_mask)
        parts = [z_internal]

        if self.use_descriptor:
            if x is None:
                raise KeyError("use_descriptor=True requires x descriptors in the batch.")
            x = x.to(device=device, dtype=dtype)
            if x.ndim != 3 or x.shape[:2] != (B, K) or x.shape[2] != self.descriptor_dim:
                raise ValueError(f"Expected x [B, K, {self.descriptor_dim}], got {tuple(x.shape)}")
            parts.append(x)

        if match_scores is None:
            match_scores = torch.zeros(B, K, dtype=dtype, device=device)
        else:
            match_scores = match_scores.to(device=device, dtype=dtype)
        if self.use_match_score_feature:
            parts.append(match_scores.unsqueeze(-1))

        if self.use_disc_score_feature:
            if matched_disc_score is None:
                disc_scores = torch.zeros(B, K, dtype=dtype, device=device)
            else:
                disc_scores = matched_disc_score.to(device=device, dtype=dtype)
            parts.append(disc_scores.unsqueeze(-1))

        if matched_class is None:
            matched_class = torch.zeros(B, K, dtype=torch.long, device=device)
        else:
            matched_class = matched_class.to(device=device).long()
        if self.use_matched_class_onehot:
            class_idx = matched_class.clamp(min=0, max=self.num_classes - 1)
            class_one_hot = F.one_hot(class_idx, num_classes=self.num_classes).to(dtype=dtype)
            class_one_hot = class_one_hot * mask.unsqueeze(-1).to(dtype=dtype)
            parts.append(class_one_hot)

        motif_x = torch.cat(parts, dim=-1)
        motif_x = torch.nan_to_num(motif_x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.node_encoder(motif_x)
        h = h * mask.unsqueeze(-1).to(dtype=h.dtype)

        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_valid, mask, edge_attr=edge_attr)

        attn_weights = None
        if self.pooling == "motif_attention":
            attn_in = torch.cat([h, match_scores.unsqueeze(-1)], dim=-1) if self.use_match_score_weighting else h
            attn_scores = self.attn(attn_in).squeeze(-1)
            attn_weights = _masked_softmax(attn_scores, mask, dim=1).unsqueeze(-1)
            h_graph = (h * attn_weights).sum(dim=1)
        elif self.use_match_score_weighting:
            attn_weights = _masked_softmax(match_scores, mask, dim=1).unsqueeze(-1)
            h_graph = (h * attn_weights).sum(dim=1)
        elif self.pooling == "max":
            masked_h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
            h_graph = masked_h.max(dim=1).values
            h_graph = torch.where(torch.isfinite(h_graph), h_graph, torch.zeros_like(h_graph))
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
            h_graph = (h * mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1) / denom

        if self.use_motif_score_vector:
            if motif_score_vector is None:
                motif_score_vector = torch.zeros(B, self.num_classes, dtype=h_graph.dtype, device=device)
            else:
                motif_score_vector = motif_score_vector.to(device=device, dtype=h_graph.dtype)
            h_graph = torch.cat([h_graph, motif_score_vector], dim=-1)

        logits = self.classifier(h_graph)
        if not return_aux:
            return logits
        return logits, {
            "internal_embeddings": z_internal,
            "motif_node_features": motif_x,
            "attention_weights": attn_weights,
        }
