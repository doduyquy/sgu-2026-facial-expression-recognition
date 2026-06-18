
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_softmax


class SemanticInteractionBlock(nn.Module):
    """Learned semantic interaction reasoning for pairwise facial coordination."""

    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, dropedge_rate: float = 0.5):
        super().__init__()
        self.dropedge_rate = dropedge_rate
        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        pair_input_dim = state_dim * 4
        self.edge_gate = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.edge_message = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, semantic_states: torch.Tensor, region_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, r, s = semantic_states.shape
        left = semantic_states.unsqueeze(2).expand(b, r, r, s)
        right = semantic_states.unsqueeze(1).expand(b, r, r, s)
        pair_input = torch.cat([left, right, left - right, left * right], dim=-1)

        gates = self.edge_gate(pair_input).squeeze(-1) + 0.1

        # Kịch bản 2: Graph DropEdge
        # Randomly sever connections between facial regions during training
        # to prevent over-smoothing and force robust path discovery.
        if self.dropedge_rate > 0.0:
            gates = F.dropout(gates, p=self.dropedge_rate, training=self.training)

        # Computational fix: Mask out invalid regions from interaction
        if region_mask is not None:
            pair_mask = region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)
            gates = gates * pair_mask

        messages = self.edge_message(pair_input)
        interaction_tensor = gates.unsqueeze(-1) * messages
        interaction_summary = interaction_tensor.sum(dim=2) / (gates.sum(dim=2, keepdim=True) + 1e-6)
        updated_states = self.norm(semantic_states + interaction_summary)
        return updated_states, interaction_tensor, gates


class CrossRegionCompositionGraph(nn.Module):
    """Learn higher-order semantic compositions across facial regions."""

    def __init__(
        self,
        state_dim: int,
        num_compositions: int,
        attn_heads: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if state_dim % attn_heads != 0:
            raise ValueError("state_dim must be divisible by attn_heads")

        hidden_dim = hidden_dim or max(state_dim * 2, 32)
        self.num_compositions = num_compositions
        self.composition_queries = nn.Parameter(torch.randn(num_compositions, state_dim) * 0.02)
        self.pair_encoder = nn.Sequential(
            nn.Linear(state_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )
        self.pair_router = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.composition_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.composition_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        semantic_states: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, r, d = semantic_states.shape
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence.unsqueeze(-1)

        left = tokens.unsqueeze(2).expand(b, r, r, d)
        right = tokens.unsqueeze(1).expand(b, r, r, d)
        pair_input = torch.cat([left, right, left - right, left * right], dim=-1)
        pair_tokens = self.pair_encoder(pair_input)
        pair_scores = self.pair_router(pair_tokens).squeeze(-1)

        if region_mask is not None:
            pair_mask = region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)
            pair_scores = pair_scores.masked_fill(pair_mask <= 0, -1e9)

        pair_attention = safe_softmax(pair_scores.reshape(b, -1), dim=-1).reshape(b, r, r)
        pair_sequence = pair_tokens.reshape(b, r * r, d)
        
        # FIX: Calculate key_padding_mask to strictly block invalid pairs from MHA
        key_padding_mask = None
        if region_mask is not None:
            key_padding_mask = (pair_mask <= 0).reshape(b, r * r)

        composition_queries = self.composition_queries.unsqueeze(0).expand(b, -1, -1)
        cross_region_tokens, composition_attn = self.composition_attn(
            composition_queries,
            pair_sequence,
            pair_sequence,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        cross_region_tokens = self.composition_norm(cross_region_tokens)

        return {
            "cross_region_tokens": cross_region_tokens,
            "composition_attn": composition_attn,
            "pair_tokens": pair_tokens,
            "pair_scores": pair_scores,
            "pair_attention": pair_attention,
        }


class SemanticHypergraphReasoner(nn.Module):
    """Compose multi-region semantic programs with learned hyperedge routing."""

    def __init__(self, state_dim: int, latent_dim: int, hyperedge_count: int, attn_heads: int, router_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        if state_dim % attn_heads != 0:
            raise ValueError("state_dim must be divisible by semantic_attn_heads")

        self.hyperedge_count = hyperedge_count
        self.hyperedge_queries = nn.Parameter(torch.randn(hyperedge_count, state_dim) * 0.02)
        self.hyperedge_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.region_back_attn = nn.MultiheadAttention(state_dim, attn_heads, dropout=dropout, batch_first=True)
        self.router = nn.Sequential(
            nn.Linear(state_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )
        self.latent_projector = nn.Sequential(
            nn.Linear(state_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        semantic_states: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        tokens = semantic_states
        if region_confidence is not None:
            tokens = tokens * region_confidence.unsqueeze(-1)

        key_padding_mask = None
        if region_mask is not None:
            key_padding_mask = region_mask <= 0

        batch_size = tokens.size(0)
        hyper_queries = self.hyperedge_queries.unsqueeze(0).expand(batch_size, -1, -1)
        hyperedge_tokens, hyperedge_attn = self.hyperedge_attn(
            hyper_queries,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        region_context, region_back_attn = self.region_back_attn(
            tokens,
            hyperedge_tokens,
            hyperedge_tokens,
            need_weights=True,
            average_attn_weights=False,
        )

        composed_states = tokens + region_context
        routing_logits = self.router(composed_states).squeeze(-1)
        if region_mask is not None:
            routing_logits = routing_logits.masked_fill(region_mask <= 0, -1e9)
        routing_weights = safe_softmax(routing_logits, dim=1)
        if region_mask is not None:
            routing_weights = routing_weights * region_mask
            routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_state = torch.sum(routing_weights.unsqueeze(-1) * composed_states, dim=1)
        hyper_summary = hyperedge_tokens.mean(dim=1)
        emotion_latent = self.latent_projector(torch.cat([pooled_state, hyper_summary], dim=-1))
        emotion_latent = self.latent_norm(emotion_latent)

        return {
            "composed_states": composed_states,
            "hyperedge_tokens": hyperedge_tokens,
            "hyperedge_attn": hyperedge_attn,
            "region_back_attn": region_back_attn,
            "routing_logits": routing_logits,
            "routing_weights": routing_weights,
            "emotion_latent": emotion_latent,
        }


class SemanticCompositionalProgramBank(nn.Module):
    """Learn structured semantic facial programs and their topology."""

    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.programs = nn.Parameter(torch.randn(num_classes, programs_per_class, num_regions, state_dim) * 0.02)
        self.topology_logits = nn.Parameter(torch.randn(num_classes, programs_per_class, num_regions, num_regions) * 0.02)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.programs, torch.sigmoid(self.topology_logits)


class SemanticProgramExecutor(nn.Module):
    """Execute semantic facial programs against observed region states."""

    def __init__(self, num_classes: int, programs_per_class: int, num_regions: int, state_dim: int, temperature: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.programs_per_class = programs_per_class
        self.num_regions = num_regions
        self.state_dim = state_dim
        self.temperature = float(temperature)
        self.program_summary_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

        # Kịch bản 12: Trọng số cấu trúc thích ứng (Adaptive Semantic Structure)
        self.sim_weights = nn.Parameter(torch.ones(1, num_classes, 1, 3))
        with torch.no_grad():
            self.sim_weights[..., 0] = 1.0   # region_sim
            self.sim_weights[..., 1] = 0.5   # topology_sim
            self.sim_weights[..., 2] = 0.25  # composition_sim

    def forward(
        self,
        semantic_states: torch.Tensor,
        cross_region_tokens: torch.Tensor,
        program_bank: torch.Tensor,
        program_topology: torch.Tensor,
        region_mask: Optional[torch.Tensor] = None,
        interaction_gates: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        state_norm = F.normalize(semantic_states, dim=-1)
        program_norm = F.normalize(program_bank, dim=-1)

        # 1. Compute valid region similarity
        region_sims = torch.einsum("brd,cmrd->bcmr", state_norm, program_norm)
        if routing_weights is not None:
            region_sim = (region_sims * routing_weights.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        elif region_mask is not None:
            valid_mask = region_mask.unsqueeze(1).unsqueeze(1)
            region_sims = region_sims * valid_mask
            region_sim = region_sims.sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1.0)
        else:
            region_sim = region_sims.mean(dim=-1)

        # 2. Compute valid topology similarity (1.0 - MSE)
        if interaction_gates is not None:
            observed_topology = interaction_gates.unsqueeze(1).unsqueeze(1)
            topology_mse = (observed_topology - program_topology.unsqueeze(0)) ** 2
            if region_mask is not None:
                pair_mask = (region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)).unsqueeze(1).unsqueeze(1)
                topology_mse = topology_mse * pair_mask
                topology_sim = 1.0 - (topology_mse.sum(dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp_min(1.0))
            else:
                topology_sim = 1.0 - topology_mse.mean(dim=(-1, -2))
        else:
            topology_sim = torch.ones_like(region_sim)

        # 3. Compute valid composition similarity
        # cross_region_tokens has shape (B, num_compositions, D) where num_compositions is 8.
        # It's already robust to region_mask because the attention that produces it masks invalid pairs.
        composition_summary = cross_region_tokens.mean(dim=1)
        composition_summary = self.program_summary_proj(composition_summary)

        program_summary = self.program_summary_proj(program_bank.mean(dim=2))
        composition_sim = torch.einsum("bd,cmd->bcm", F.normalize(composition_summary, dim=-1), F.normalize(program_summary, dim=-1))

        # Kịch bản 12: Sử dụng trọng số động thay vì hằng số cứng nhắc
        w = F.softplus(self.sim_weights)  # Đảm bảo trọng số dương
        total_sim = w[..., 0] * region_sim + w[..., 1] * topology_sim + w[..., 2] * composition_sim

        # Save pre-temperature scaled versions for auxiliary loss logging consistency
        region_score = region_sim / self.temperature
        topology_score = topology_sim / self.temperature
        composition_score = composition_sim / self.temperature

        # Fix: Gradient Explosion during Temperature Scaling.
        # Clamp compatibility to avoid logsumexp gradient blowup while preserving relative order
        compatibility = (total_sim / self.temperature).clamp(-50, 50)

        program_attention = safe_softmax(compatibility, dim=-1)
        class_scores = torch.logsumexp(compatibility, dim=-1)
        program_tokens = torch.einsum("bcm,cmd->bcd", program_attention, program_summary)

        if routing_weights is not None:
            routing_entropy = -(routing_weights.clamp_min(1e-6) * routing_weights.clamp_min(1e-6).log()).sum(dim=-1)
        else:
            routing_entropy = torch.zeros(semantic_states.size(0), device=semantic_states.device)

        return {
            "program_scores": class_scores,
            "program_attention": program_attention,
            "program_tokens": program_tokens,
            "compatibility": compatibility,
            "region_score": region_score,
            "topology_score": topology_score,
            "composition_score": composition_score,
            "routing_entropy": routing_entropy,
        }
