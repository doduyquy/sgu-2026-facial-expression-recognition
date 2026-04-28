"""D4A full pixel-graph adaptive motif slot GNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwarePixelGNNLayer(nn.Module):
    """Edge-aware message passing layer for batched full pixel graphs."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_edge_attr: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.use_edge_attr = bool(use_edge_attr)
        edge_input_dim = self.edge_dim if self.use_edge_attr and self.edge_dim > 0 else 1
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, H = h.shape
        src = edge_index[0].long().clamp(0, N - 1)
        dst = edge_index[1].long().clamp(0, N - 1)
        E = int(src.numel())
        out = torch.zeros_like(h)
        deg = torch.zeros(B, N, 1, device=h.device, dtype=h.dtype)

        if edge_attr is None or not self.use_edge_attr:
            edge_attr_batched = h.new_zeros(B, E, 1)
        elif edge_attr.ndim == 2:
            edge_attr_batched = edge_attr.to(device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1, -1)
        else:
            edge_attr_batched = edge_attr.to(device=h.device, dtype=h.dtype)

        for b in range(B):
            e = self.edge_encoder(edge_attr_batched[b])
            msg_in = torch.cat([h[b, src], e], dim=-1)
            msg = self.message_mlp(msg_in)
            out[b].index_add_(0, dst, msg)
            deg[b].index_add_(0, dst, torch.ones((E, 1), device=h.device, dtype=h.dtype))

        agg = out / deg.clamp_min(1.0)
        h = self.msg_norm(h + self.dropout(agg))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        return h * node_mask.unsqueeze(-1).to(dtype=h.dtype)


class FullGraphAdaptiveMotifSlotGNN(nn.Module):
    """Image -> full pixel graph -> adaptive motif slots -> emotion logits."""

    def __init__(
        self,
        *,
        node_dim: int = 7,
        edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_slots: int = 32,
        num_classes: int = 7,
        dropout: float = 0.2,
        use_edge_attr: bool = True,
        use_null_slot: bool = True,
        use_slot_gates: bool = True,
        readout_mode: str = "slots_global",
        lambda_smooth: float = 0.0,
        lambda_sparse: float = 0.0,
        lambda_diversity: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_slots = int(num_slots)
        self.num_classes = int(num_classes)
        self.use_null_slot = bool(use_null_slot)
        self.use_slot_gates = bool(use_slot_gates)
        self.readout_mode = str(readout_mode)
        if self.readout_mode not in {"slots_global", "slots_only", "global_only"}:
            raise ValueError(
                "readout_mode must be one of: 'slots_global', 'slots_only', 'global_only'; "
                f"got {self.readout_mode!r}"
            )
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_diversity = float(lambda_diversity)
        self.assignment_dim = self.num_slots + (1 if self.use_null_slot else 0)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gnn_layers = nn.ModuleList(
            [
                EdgeAwarePixelGNNLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    dropout=dropout,
                    use_edge_attr=use_edge_attr,
                )
                for _ in range(max(0, self.num_layers))
            ]
        )
        self.assignment_head = nn.Linear(self.hidden_dim, self.assignment_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        readout_dim = self.hidden_dim * 4 if self.readout_mode == "slots_global" else self.hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _unpack_batch(self, batch_or_x, **kwargs) -> dict:
        if isinstance(batch_or_x, dict):
            return batch_or_x
        return {"node_features": batch_or_x, **kwargs}

    def _masked_node_pool(self, h: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = node_mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        max_pool = h.masked_fill(~node_mask.unsqueeze(-1), torch.finfo(h.dtype).min).amax(dim=1)
        max_pool = torch.where(node_mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return mean_pool, max_pool

    def _slot_pool(
        self,
        h: torch.Tensor,
        assignments: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_motif = assignments[:, :, : self.num_slots] * node_mask.unsqueeze(-1).to(dtype=h.dtype)
        slot_mass = s_motif.sum(dim=1)
        slot_embeddings = torch.bmm(s_motif.transpose(1, 2), h) / slot_mass.clamp_min(1e-6).unsqueeze(-1)
        if self.use_slot_gates:
            slot_gates = torch.sigmoid(self.gate_mlp(slot_embeddings)).squeeze(-1)
            slot_embeddings = slot_embeddings * slot_gates.unsqueeze(-1)
        else:
            slot_gates = torch.ones(h.shape[0], self.num_slots, device=h.device, dtype=h.dtype)
        return slot_embeddings, slot_gates, slot_mass

    @staticmethod
    def _assignment_entropy(assignments: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        entropy = -(assignments.clamp_min(1e-8) * assignments.clamp_min(1e-8).log()).sum(dim=-1)
        mask_f = node_mask.to(dtype=entropy.dtype)
        return (entropy * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    @staticmethod
    def _slot_diversity(slot_embeddings: torch.Tensor) -> torch.Tensor:
        K = slot_embeddings.shape[1]
        if K <= 1:
            return slot_embeddings.sum() * 0.0
        norm = F.normalize(slot_embeddings, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        eye = torch.eye(K, device=slot_embeddings.device, dtype=torch.bool).unsqueeze(0)
        return sim.masked_select(~eye).mean()

    @staticmethod
    def _assignment_smoothness(
        assignments: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        valid = node_mask[:, src] & node_mask[:, dst]
        diff = assignments[:, src, :] - assignments[:, dst, :]
        per_edge = diff.pow(2).mean(dim=-1)
        return per_edge.masked_select(valid).mean() if valid.any() else assignments.sum() * 0.0

    def forward(self, batch_or_x, **kwargs) -> dict:
        batch = self._unpack_batch(batch_or_x, **kwargs)
        x = batch.get("node_features", batch.get("x")).float()
        edge_index = batch["edge_index"].to(device=x.device)
        edge_attr = batch.get("edge_attr")
        node_mask = batch.get("node_mask")
        if node_mask is None:
            node_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            node_mask = node_mask.to(device=x.device).bool()

        h = self.node_encoder(x) * node_mask.unsqueeze(-1).to(dtype=x.dtype)
        for layer in self.gnn_layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)

        assignment_logits = self.assignment_head(h)
        if not self.use_null_slot:
            assignment_logits = assignment_logits.masked_fill(~node_mask.unsqueeze(-1), -1e9)
        else:
            invalid = ~node_mask
            motif_logits = assignment_logits[:, :, : self.num_slots].masked_fill(invalid.unsqueeze(-1), -1e9)
            null_logits = torch.where(
                invalid,
                torch.zeros_like(assignment_logits[:, :, -1]),
                assignment_logits[:, :, -1],
            ).unsqueeze(-1)
            assignment_logits = torch.cat([motif_logits, null_logits], dim=-1)
        assignments = torch.softmax(assignment_logits, dim=-1)

        slot_embeddings, slot_gates, slot_mass = self._slot_pool(h, assignments, node_mask)
        slot_mean, slot_max = self._masked_node_pool(
            slot_embeddings,
            torch.ones(slot_embeddings.shape[:2], dtype=torch.bool, device=x.device),
        )
        global_mean, global_max = self._masked_node_pool(h, node_mask)
        if self.readout_mode == "slots_global":
            image_repr = torch.cat([slot_mean, slot_max, global_mean, global_max], dim=-1)
        elif self.readout_mode == "slots_only":
            image_repr = torch.cat([slot_mean, slot_max], dim=-1)
        else:
            image_repr = torch.cat([global_mean, global_max], dim=-1)
        logits = self.classifier(image_repr)

        if self.use_null_slot:
            null_values = assignments[:, :, -1]
            null_mass = (null_values * node_mask.to(dtype=null_values.dtype)).sum() / node_mask.sum().clamp_min(1)
        else:
            null_mass = logits.new_tensor(0.0)
        motif_mass_total = (
            assignments[:, :, : self.num_slots].sum(dim=-1) * node_mask.to(dtype=assignments.dtype)
        ).sum() / node_mask.sum().clamp_min(1)
        active_slot_count_soft = slot_gates.sum(dim=1).mean()

        aux_loss = logits.new_tensor(0.0)
        if self.lambda_sparse:
            aux_loss = aux_loss + self.lambda_sparse * self._assignment_entropy(assignments, node_mask)
        if self.lambda_diversity:
            aux_loss = aux_loss + self.lambda_diversity * self._slot_diversity(slot_embeddings)
        if self.lambda_smooth:
            aux_loss = aux_loss + self.lambda_smooth * self._assignment_smoothness(assignments, edge_index, node_mask)

        return {
            "logits": logits,
            "slot_assignments": assignments,
            "slot_embeddings": slot_embeddings,
            "slot_gates": slot_gates,
            "slot_mass": slot_mass,
            "null_mass": null_mass,
            "motif_mass_total": motif_mass_total,
            "active_slot_count_soft": active_slot_count_soft,
            "assignment_entropy": self._assignment_entropy(assignments, node_mask).detach(),
            "aux_loss": aux_loss,
        }
