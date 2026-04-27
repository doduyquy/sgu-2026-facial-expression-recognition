"""Learnable slot candidate motif GNN for full candidate selection-free FER."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedGraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_valid: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, H = h.shape
        agg = torch.zeros_like(h)
        deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)
        for b in range(B):
            valid = edge_valid[b].bool()
            if not valid.any():
                continue
            src = edge_index[b, 0, valid].long().clamp(0, N - 1)
            dst = edge_index[b, 1, valid].long().clamp(0, N - 1)
            msg = h[b, src]
            agg[b].index_add_(0, dst, msg)
            deg[b].index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        out = self.lin(torch.cat([h, agg], dim=-1))
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.norm(h + out)
        return out * node_mask.unsqueeze(-1).to(dtype=out.dtype)


class FullyConnectedSlotSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[1] <= 1:
            neigh = h
        else:
            total = h.sum(dim=1, keepdim=True)
            neigh = (total - h) / max(1, h.shape[1] - 1)
        out = self.lin(torch.cat([h, neigh], dim=-1))
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class LearnableSlotCandidateMotifGNN(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 7,
        descriptor_dim: int = 41,
        hidden_dim: int = 128,
        candidate_gnn_layers: int = 2,
        slot_gnn_layers: int = 2,
        num_slots: int = 32,
        slot_iterations: int = 1,
        dropout: float = 0.2,
        use_geometry_features: bool = True,
        use_motif_metadata_features: bool = False,
        pooling: str = "class_conditioned_slot_attention",
        use_global_candidate_pooling: bool = False,
        global_pooling_type: str = "mean_max",
        slot_attention_entropy_weight: float = 0.0,
        slot_diversity_weight: float = 0.0,
        class_attention_diversity_weight: float = 0.0,
        combined_attention_diversity_weight: float = 0.0,
        combined_attention_diversity_margin: float = 0.65,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.slot_iterations = max(1, int(slot_iterations))
        self.use_geometry_features = bool(use_geometry_features)
        self.use_motif_metadata_features = bool(use_motif_metadata_features)
        self.pooling = str(pooling)
        self.use_global_candidate_pooling = bool(use_global_candidate_pooling)
        self.global_pooling_type = str(global_pooling_type)
        self.slot_attention_entropy_weight = float(slot_attention_entropy_weight)
        self.slot_diversity_weight = float(slot_diversity_weight)
        self.class_attention_diversity_weight = float(class_attention_diversity_weight)
        self.combined_attention_diversity_weight = float(combined_attention_diversity_weight)
        self.combined_attention_diversity_margin = float(combined_attention_diversity_margin)
        if self.global_pooling_type not in {"mean", "mean_max"}:
            raise ValueError(
                f"Unknown global_pooling_type={global_pooling_type!r}; expected 'mean' or 'mean_max'"
            )

        feature_dim = self.descriptor_dim
        if self.use_geometry_features:
            feature_dim += 2 + 4 + 1
        if self.use_motif_metadata_features:
            feature_dim += 1 + 1 + self.num_classes

        self.candidate_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_layers = nn.ModuleList(
            [BatchedGraphSAGELayer(hidden_dim, dropout=dropout) for _ in range(candidate_gnn_layers)]
        )

        self.slot_queries = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.slot_q = nn.Linear(hidden_dim, hidden_dim)
        self.cand_k = nn.Linear(hidden_dim, hidden_dim)
        self.cand_v = nn.Linear(hidden_dim, hidden_dim)
        self.slot_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.slot_norm = nn.LayerNorm(hidden_dim)

        self.slot_layers = nn.ModuleList(
            [FullyConnectedSlotSAGELayer(hidden_dim, dropout=dropout) for _ in range(slot_gnn_layers)]
        )

        global_dim = 0
        if self.use_global_candidate_pooling:
            global_dim = hidden_dim * (2 if self.global_pooling_type == "mean_max" else 1)

        if self.pooling == "class_conditioned_slot_attention":
            self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
            self.class_q = nn.Linear(hidden_dim, hidden_dim)
            self.slot_k = nn.Linear(hidden_dim, hidden_dim)
            self.slot_v = nn.Linear(hidden_dim, hidden_dim)
            self.class_logit = nn.Linear(hidden_dim + global_dim, 1)
        elif self.pooling == "attention":
            self.pool_score = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim + global_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim + global_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown pooling={pooling!r}")

    def _features(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch["candidate_x"].float()
        mask = batch["candidate_mask"].bool()
        feats = [x]
        if self.use_geometry_features:
            feats.extend(
                [
                    batch["candidate_centers"].float(),
                    batch["candidate_bbox"].float(),
                    batch["candidate_radius"].float().unsqueeze(-1),
                ]
            )
        if self.use_motif_metadata_features:
            match_score = batch.get("candidate_match_score")
            disc_score = batch.get("candidate_disc_score")
            matched_class = batch.get("candidate_matched_class")
            B, M, _ = x.shape
            if match_score is None:
                match_score = x.new_zeros(B, M)
            if disc_score is None:
                disc_score = x.new_zeros(B, M)
            if matched_class is None:
                one_hot = x.new_zeros(B, M, self.num_classes)
            else:
                cls = matched_class.long().clamp(0, self.num_classes - 1)
                one_hot = F.one_hot(cls, num_classes=self.num_classes).float()
                one_hot = one_hot * (matched_class.long().ge(0).unsqueeze(-1).float())
            feats.extend([match_score.float().unsqueeze(-1), disc_score.float().unsqueeze(-1), one_hot])
        return torch.cat(feats, dim=-1), mask

    def _slot_attention(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, M, H = h.shape
        slots = self.slot_queries.unsqueeze(0).expand(B, -1, -1)
        attn = None
        k = self.cand_k(h)
        v = self.cand_v(h)
        for _ in range(self.slot_iterations):
            q = self.slot_q(slots)
            logits = torch.einsum("bkh,bmh->bkm", q, k) / math.sqrt(H)
            logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
            attn = torch.softmax(logits, dim=-1)
            slot_update = torch.einsum("bkm,bmh->bkh", attn, v)
            slots = self.slot_update(
                slot_update.reshape(B * self.num_slots, H),
                slots.reshape(B * self.num_slots, H),
            ).view(B, self.num_slots, H)
            slots = self.slot_norm(slots)
        return slots, attn

    def _global_candidate_pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        if self.global_pooling_type == "mean":
            return mean_pool
        max_pool = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min).amax(dim=1)
        max_pool = torch.where(mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def _class_conditioned_pool(
        self,
        slots: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, H = slots.shape
        q = self.class_q(self.class_queries).unsqueeze(0).expand(B, -1, -1)
        k = self.slot_k(slots)
        v = self.slot_v(slots)
        logits = torch.einsum("bch,bkh->bck", q, k) / math.sqrt(H)
        class_attn = torch.softmax(logits, dim=-1)
        z = torch.einsum("bck,bkh->bch", class_attn, v)
        if global_context is not None:
            z = torch.cat([z, global_context.unsqueeze(1).expand(-1, self.num_classes, -1)], dim=-1)
        logits = self.class_logit(z).squeeze(-1)
        return logits, class_attn

    def _attention_pool(
        self,
        slots: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        weight = torch.softmax(self.pool_score(slots).squeeze(-1), dim=-1)
        image_emb = torch.einsum("bk,bkh->bh", weight, slots)
        if global_context is not None:
            image_emb = torch.cat([image_emb, global_context], dim=-1)
        return self.classifier(image_emb), None

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        return -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()

    @staticmethod
    def _attention_diversity(attn: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(attn, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        K = sim.shape[-1]
        eye = torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
        return sim.masked_select(~eye).mean() if K > 1 else sim.sum() * 0.0

    @staticmethod
    def _combined_attention_diversity(attn: torch.Tensor, margin: float) -> torch.Tensor:
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        norm = F.normalize(attn, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        C = sim.shape[-1]
        if C <= 1:
            return sim.sum() * 0.0
        eye = torch.eye(C, device=sim.device, dtype=torch.bool).unsqueeze(0)
        offdiag = sim.masked_select(~eye)
        return F.relu(offdiag - float(margin)).mean()

    def forward(self, batch: dict) -> dict:
        feat, mask = self._features(batch)
        h = self.candidate_proj(feat) * mask.unsqueeze(-1).to(dtype=feat.dtype)
        edge_index = batch["candidate_edge_index"]
        edge_valid = batch["candidate_edge_valid"]
        for layer in self.candidate_layers:
            h = layer(h, edge_index=edge_index, edge_valid=edge_valid, node_mask=mask)

        global_context = self._global_candidate_pool(h, mask) if self.use_global_candidate_pooling else None
        slots, candidate_attention = self._slot_attention(h, mask)
        for layer in self.slot_layers:
            slots = layer(slots)

        if self.pooling == "class_conditioned_slot_attention":
            logits, class_slot_attention = self._class_conditioned_pool(slots, global_context)
        else:
            logits, class_slot_attention = self._attention_pool(slots, global_context)

        combined_candidate_attention = None
        if class_slot_attention is not None:
            combined_candidate_attention = torch.matmul(class_slot_attention, candidate_attention)
            combined_candidate_attention = combined_candidate_attention.masked_fill(~mask.unsqueeze(1), 0.0)

        aux_loss = logits.new_tensor(0.0)
        if self.slot_attention_entropy_weight:
            aux_loss = aux_loss + self.slot_attention_entropy_weight * self._attention_entropy(candidate_attention)
        if self.slot_diversity_weight:
            aux_loss = aux_loss + self.slot_diversity_weight * self._attention_diversity(candidate_attention)
        if self.class_attention_diversity_weight and class_slot_attention is not None:
            aux_loss = aux_loss + self.class_attention_diversity_weight * self._attention_diversity(class_slot_attention)
        if self.combined_attention_diversity_weight and combined_candidate_attention is not None:
            aux_loss = aux_loss + self.combined_attention_diversity_weight * self._combined_attention_diversity(
                combined_candidate_attention,
                self.combined_attention_diversity_margin,
            )

        return {
            "logits": logits,
            "candidate_attention": candidate_attention,
            "class_slot_attention": class_slot_attention,
            "combined_candidate_attention": combined_candidate_attention,
            "slot_embeddings": slots,
            "aux_loss": aux_loss,
        }
