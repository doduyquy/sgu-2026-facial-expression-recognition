
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SemanticRoiGraphConfig
from .backbone import SemanticBackbone
from .roi_align import SemanticRoiAlign
from .micro_modules import (
    GATBlock,
    GatedPooling,
    MicroGraphReasoner,
    MicroSemanticMotifBank,
    MicroSemanticMotifMatcher,
)
from .macro_modules import (
    SemanticInteractionBlock,
    CrossRegionCompositionGraph,
    SemanticHypergraphReasoner,
    SemanticCompositionalProgramBank,
    SemanticProgramExecutor,
)
from .classifier import SemanticStateEncoder, SemanticEmotionClassifier

# ---------------------------------------------------------------------------
# Backward-compatible aliases for callers and older checkpoints.
# ---------------------------------------------------------------------------
MacroSemanticProgramBank   = SemanticCompositionalProgramBank
MacroSemanticProgramMatcher = SemanticProgramExecutor
MacroMotifBank             = SemanticCompositionalProgramBank
MacroMotifMatcher          = SemanticProgramExecutor
SemanticMotifBank          = SemanticCompositionalProgramBank
SemanticMotifMatcher       = SemanticProgramExecutor
MicroMotifBank             = MicroSemanticMotifBank
MicroMotifMatcher          = MicroSemanticMotifMatcher


class SemanticROIGraphFER(nn.Module):
    """End-to-end semantic compositional facial reasoning model."""

    def __init__(self, config: SemanticRoiGraphConfig):
        super().__init__()
        self.config = config

        self.backbone = SemanticBackbone(
            feature_dim=config.feature_dim,
            use_pretrained=config.use_pretrained,
        )
        self.roi_align = SemanticRoiAlign(
            roi_grid=config.roi_grid,
            bbox_input_size=config.bbox_input_size,
            feature_out_size=config.backbone_out_size,
        )

        self.micro_reasoner = MicroGraphReasoner(
            dim=config.feature_dim,
            num_nodes=config.roi_grid * config.roi_grid,
            layers=config.micro_layers,
            heads=config.attn_heads,
            dropout=config.dropout,
        )

        # Fix 3: region_proj (feature_dim→feature_dim) removed — a same-dimension
        # linear projection adds no representational capacity and wastes ~65K params.
        self.semantic_state_encoder = SemanticStateEncoder(
            input_dim=config.feature_dim,
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.feature_dim // 2, config.semantic_state_dim * 2),
            dropout=config.dropout,
        )
        self.semantic_interaction_block = SemanticInteractionBlock(
            state_dim=config.semantic_state_dim,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
            dropedge_rate=0.1,
        )

        self.micro_motif_bank = MicroSemanticMotifBank(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
        )
        self.micro_motif_matcher = MicroSemanticMotifMatcher(
            num_regions=config.num_regions,
            motifs_per_region=config.micro_motifs_per_region,
            state_dim=config.semantic_state_dim,
            temperature=config.relation_temperature,
        )

        self.semantic_compositional_reasoner = SemanticHypergraphReasoner(
            state_dim=config.semantic_state_dim,
            latent_dim=config.semantic_latent_dim,
            hyperedge_count=config.hyperedge_count,
            attn_heads=config.semantic_attn_heads,
            router_hidden_dim=config.router_hidden_dim,
            dropout=config.dropout,
        )

        self.cross_region_composition_graph = CrossRegionCompositionGraph(
            state_dim=config.semantic_state_dim,
            num_compositions=config.cross_region_compositions,
            attn_heads=config.semantic_attn_heads,
            hidden_dim=max(config.semantic_state_dim * 2, 32),
            dropout=config.dropout,
        )

        self.semantic_program_bank = SemanticCompositionalProgramBank(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
        )
        self.semantic_program_executor = SemanticProgramExecutor(
            num_classes=config.num_classes,
            programs_per_class=config.macro_motifs_per_class,
            num_regions=config.num_regions,
            state_dim=config.semantic_state_dim,
            temperature=0.15,
        )

        self.semantic_classifier = SemanticEmotionClassifier(
            latent_dim=config.semantic_latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(config.feature_dim, config.semantic_latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.global_fusion = nn.Sequential(
            nn.Linear(config.semantic_latent_dim * 2, config.semantic_latent_dim),
            nn.LayerNorm(config.semantic_latent_dim),
            nn.GELU(),
        )

        # Instance-aware gate: dynamically decides whether to trust the Graph branch or Global branch.
        self.semantic_structure_gate = nn.Sequential(
            nn.Linear(config.semantic_latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Backward-compatible aliases for older checkpoints and callers.
        self.macro_motif_bank    = self.semantic_program_bank
        self.macro_motif_matcher = self.semantic_program_executor
        self.motif_bank          = self.semantic_program_bank
        self.motif_matcher       = self.semantic_program_executor

        self.missing_region_token = nn.Parameter(torch.randn(config.feature_dim) * 0.02)
        self.region_reliability_predictor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.region_dropout_prob = float(getattr(config, "region_dropout_prob", 0.05))

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True):
        """Backward-compatible: upgrade scalar semantic_structure_gate from old checkpoints."""
        key = "semantic_structure_gate"
        if key in state_dict:
            old = state_dict[key]
            if old.ndim == 0 or old.numel() == 1:
                state_dict = dict(state_dict)  # don't mutate the original
                state_dict[key] = old.detach().view(1).expand(self.config.num_classes).clone()
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canonical_bboxes(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        boxes = SemanticRoiAlign._canonical_region_boxes(self.config.bbox_input_size, device, dtype)
        return boxes.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _prepare_regions(
        self,
        bboxes: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return repaired boxes, region mask, confidence and invalid indices."""
        if bboxes is None:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.ones(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.full_like(region_mask, 0.95)
            invalid_indices = torch.empty((0, 2), device=device, dtype=torch.long)
            return repaired, region_mask, region_confidence, invalid_indices

        bboxes = bboxes.to(device=device, dtype=dtype)
        if bboxes.dim() != 3 or bboxes.size(-1) != 4:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.zeros(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.zeros_like(region_mask)
            invalid_indices = torch.nonzero(torch.ones_like(region_mask, dtype=torch.bool), as_tuple=False)
            return repaired, region_mask, region_confidence, invalid_indices

        valid_shape = bboxes.size(0) == batch_size and bboxes.size(1) == self.config.num_regions
        if not valid_shape:
            repaired = self._canonical_bboxes(batch_size, device, dtype)
            region_mask = torch.zeros(batch_size, self.config.num_regions, device=device, dtype=dtype)
            region_confidence = torch.zeros_like(region_mask)
            invalid_indices = torch.nonzero(torch.ones_like(region_mask, dtype=torch.bool), as_tuple=False)
            return repaired, region_mask, region_confidence, invalid_indices

        finite_mask = torch.isfinite(bboxes).all(dim=-1)
        x1 = bboxes[..., 0]
        y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]
        y2 = bboxes[..., 3]
        size_mask  = ((x2 - x1) >= 2.0) & ((y2 - y1) >= 2.0)
        order_mask = (x2 > x1) & (y2 > y1)
        region_mask = (finite_mask & size_mask & order_mask).to(dtype=dtype)

        repaired  = self.roi_align.validate_bboxes(bboxes)
        canonical = self._canonical_bboxes(batch_size, device, dtype)
        repaired  = torch.where(region_mask.unsqueeze(-1).bool(), repaired, canonical)

        width  = (repaired[..., 2] - repaired[..., 0]).clamp(min=1.0)
        height = (repaired[..., 3] - repaired[..., 1]).clamp(min=1.0)
        area   = (width * height) / float(self.config.bbox_input_size * self.config.bbox_input_size)
        area_conf = area.clamp(0.0, 1.0)
        region_confidence = torch.where(region_mask > 0, 0.5 + 0.5 * area_conf, torch.full_like(area_conf, 0.05))

        invalid_indices = torch.nonzero(region_mask == 0, as_tuple=False)
        return repaired, region_mask, region_confidence, invalid_indices

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Public forward: dispatches to TTA or single-image path."""
        if image.dim() == 5:
            return self._forward_tta(image, bboxes, region_mask, region_confidence)

        if not self.training and bboxes is not None:
            # 1. Forward original image and bboxes
            outputs_orig = self._forward_single(image, bboxes, region_mask, region_confidence)

            # 2. Horizontal Flip TTA: flip image along width dimension (dim=-1)
            flipped_image = torch.flip(image, dims=[-1])

            # Flip bboxes: x1_new = (w - 1.0) - x2, x2_new = (w - 1.0) - x1
            w = float(self.config.bbox_input_size)
            flipped_bboxes = bboxes.clone()
            flipped_bboxes[..., 0] = (w - 1.0) - bboxes[..., 2]
            flipped_bboxes[..., 2] = (w - 1.0) - bboxes[..., 0]

            # Swap symmetric left/right regions:
            # 1 (left eyebrow) <-> 2 (right eyebrow)
            # 4 (left eye)     <-> 5 (right eye)
            # 7 (left mouth corner) <-> 8 (right mouth corner)
            swap_pairs = [(1, 2), (4, 5), (7, 8)]
            for idx_l, idx_r in swap_pairs:
                tmp = flipped_bboxes[:, idx_l].clone()
                flipped_bboxes[:, idx_l] = flipped_bboxes[:, idx_r]
                flipped_bboxes[:, idx_r] = tmp

            flipped_region_mask = None
            if region_mask is not None:
                flipped_region_mask = region_mask.clone()
                for idx_l, idx_r in swap_pairs:
                    tmp = flipped_region_mask[:, idx_l].clone()
                    flipped_region_mask[:, idx_l] = flipped_region_mask[:, idx_r]
                    flipped_region_mask[:, idx_r] = tmp

            flipped_region_confidence = None
            if region_confidence is not None:
                flipped_region_confidence = region_confidence.clone()
                for idx_l, idx_r in swap_pairs:
                    tmp = flipped_region_confidence[:, idx_l].clone()
                    flipped_region_confidence[:, idx_l] = flipped_region_confidence[:, idx_r]
                    flipped_region_confidence[:, idx_r] = tmp

            # 3. Forward flipped image and bboxes
            outputs_flipped = self._forward_single(
                flipped_image, flipped_bboxes, flipped_region_mask, flipped_region_confidence
            )

            # 4. Average predictions for logit/probability keys
            avg_outputs = {}
            _avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
            for k, val in outputs_orig.items():
                if k in _avg_keys and torch.is_tensor(val) and k in outputs_flipped:
                    avg_outputs[k] = 0.5 * (val + outputs_flipped[k])
                else:
                    avg_outputs[k] = val
            return avg_outputs

        return self._forward_single(image, bboxes, region_mask, region_confidence)

    def _forward_tta(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """TTA path: image is (B, T, C, H, W); averages logits over T crops."""
        B, T, C, H, W = image.shape
        # Flatten crops into batch dimension
        flat_image = image.reshape(B * T, C, H, W)

        # Expand bbox / mask tensors from (B, R, *) -> (B*T, R, *)
        flat_bboxes = None
        if bboxes is not None:
            flat_bboxes = bboxes.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, bboxes.size(1), bboxes.size(2))
        flat_region_mask = None
        if region_mask is not None:
            flat_region_mask = region_mask.unsqueeze(1).expand(B, T, -1).reshape(B * T, region_mask.size(1))
        flat_region_confidence = None
        if region_confidence is not None:
            flat_region_confidence = region_confidence.unsqueeze(1).expand(B, T, -1).reshape(B * T, region_confidence.size(1))

        outputs = self._forward_single(flat_image, flat_bboxes, flat_region_mask, flat_region_confidence)

        # Average the classification scores over T crops
        _avg_keys = ("logits", "logits_motif", "logits_fused", "semantic_program_scores")
        for key in _avg_keys:
            if key in outputs and torch.is_tensor(outputs[key]):
                x = outputs[key]
                if x.size(0) == B * T:
                    outputs[key] = x.reshape(B, T, *x.shape[1:]).mean(dim=1)

        # For non-averaged keys that still have B*T batch size, keep center-crop (index 4)
        center_idx = 4 if T > 4 else T // 2
        for key, val in outputs.items():
            if key in _avg_keys:
                continue
            if torch.is_tensor(val) and val.dim() >= 1 and val.size(0) == B * T:
                outputs[key] = val.reshape(B, T, *val.shape[1:])[:, center_idx]

        return outputs

    def _forward_single(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
        region_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Core forward for a regular (B, C, H, W) batch."""
        # image: (B, 1, 48, 48) -> expand to 3 channels for ResNet
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        batch_size = image.size(0)
        feature_map = self.backbone(image)
        bboxes, computed_mask, computed_confidence, invalid_indices = self._prepare_regions(
            bboxes,
            batch_size=batch_size,
            device=image.device,
            dtype=image.dtype,
        )

        if region_mask is None:
            region_mask = computed_mask
        else:
            region_mask = region_mask.to(device=image.device, dtype=image.dtype)
        if region_confidence is None:
            region_confidence = computed_confidence
        else:
            region_confidence = region_confidence.to(device=image.device, dtype=image.dtype)

        if self.training:
            drop_mask = (torch.rand(batch_size, self.config.num_regions, device=image.device) > self.region_dropout_prob).to(image.dtype)
            region_mask       = region_mask * drop_mask
            region_confidence = region_confidence * drop_mask

        roi_nodes = self.roi_align(feature_map, bboxes)
        micro_node_features, region_embeddings = self.micro_reasoner(roi_nodes)

        missing_token    = self.missing_region_token.view(1, 1, -1)
        region_valid_mask = region_mask.unsqueeze(-1) > 0
        region_embeddings = torch.where(region_valid_mask, region_embeddings, missing_token.expand_as(region_embeddings))

        predicted_confidence = self.region_reliability_predictor(region_embeddings).squeeze(-1)
        region_confidence    = torch.clamp(0.5 * region_confidence + 0.5 * predicted_confidence, 0.0, 1.0)
        region_confidence    = region_confidence * region_mask

        semantic_state_tokens = self.semantic_state_encoder(region_embeddings)
        micro_motif_bank = self.micro_motif_bank()
        micro_motif_attention, semantic_motif_tokens = self.micro_motif_matcher(semantic_state_tokens, micro_motif_bank)

        # Step 1: Pairwise region interaction (local semantic coordination).
        interaction_states, semantic_interaction_tensor, semantic_interaction_gates = self.semantic_interaction_block(
            semantic_motif_tokens,
            region_mask=region_mask,
        )

        # Step 2: Higher-order cross-region composition on interaction-enriched states.
        cross_region_outputs      = self.cross_region_composition_graph(
            interaction_states,
            region_mask=region_mask,
            region_confidence=region_confidence,
        )
        cross_region_tokens       = cross_region_outputs["cross_region_tokens"]
        cross_region_attention    = cross_region_outputs["composition_attn"]
        cross_region_pair_tokens  = cross_region_outputs["pair_tokens"]
        cross_region_pair_scores  = cross_region_outputs["pair_scores"]
        cross_region_pair_attention = cross_region_outputs["pair_attention"]

        # Step 3: Enrich interaction states with higher-order composition context.
        composition_summary = cross_region_tokens.mean(dim=1, keepdim=True)
        hypergraph_input    = interaction_states + composition_summary.expand_as(interaction_states)

        compositional_outputs       = self.semantic_compositional_reasoner(
            hypergraph_input,
            region_mask=region_mask,
            region_confidence=region_confidence,
        )
        composed_states              = compositional_outputs["composed_states"]
        hyperedge_tokens             = compositional_outputs["hyperedge_tokens"]
        routing_weights              = compositional_outputs["routing_weights"]
        semantic_latent_embedding    = compositional_outputs["emotion_latent"]

        semantic_program_bank, semantic_program_topology = self.semantic_program_bank()
        semantic_program_outputs = self.semantic_program_executor(
            composed_states,
            cross_region_tokens,
            semantic_program_bank,
            semantic_program_topology,
            region_mask=region_mask,
            interaction_gates=semantic_interaction_gates,
            routing_weights=routing_weights,
        )
        semantic_program_scores            = semantic_program_outputs["program_scores"]
        semantic_program_attention         = semantic_program_outputs["program_attention"]
        semantic_program_tokens            = semantic_program_outputs["program_tokens"]
        semantic_program_compatibility     = semantic_program_outputs["compatibility"]
        semantic_program_region_scores     = semantic_program_outputs["region_score"]
        semantic_program_topology_scores   = semantic_program_outputs["topology_score"]
        semantic_program_composition_scores = semantic_program_outputs["composition_score"]
        semantic_program_routing_entropy   = semantic_program_outputs["routing_entropy"]

        global_semantic_context = self.global_context(feature_map)
        fused_latent = self.global_fusion(torch.cat([semantic_latent_embedding, global_semantic_context], dim=-1))
        logits_fused = self.semantic_classifier(fused_latent)

        # Instance-aware gate: shape (Batch, 1) — dynamically balances based on global context
        structure_gate = self.semantic_structure_gate(global_semantic_context)
        logits_motif   = semantic_program_scores
        logits         = (1 - structure_gate) * logits_fused + structure_gate * logits_motif

        return {
            "logits": logits,
            "logits_motif": logits_motif,
            "logits_fused": logits_fused,
            "structure_gate": structure_gate,

            "micro_node_features": micro_node_features,
            "micro_motif_attention": micro_motif_attention,
            "region_motif_tokens": semantic_motif_tokens,
            "region_embeddings": region_embeddings,
            "semantic_state_tokens": semantic_state_tokens,
            "semantic_motif_tokens": semantic_motif_tokens,
            "cross_region_tokens": cross_region_tokens,
            "cross_region_attention": cross_region_attention,
            "cross_region_pair_tokens": cross_region_pair_tokens,
            "cross_region_pair_scores": cross_region_pair_scores,
            "cross_region_pair_attention": cross_region_pair_attention,
            "semantic_interaction_tensor": semantic_interaction_tensor,
            "semantic_interaction_gates": semantic_interaction_gates,
            "semantic_routing_weights": routing_weights,
            "hyperedge_tokens": hyperedge_tokens,
            "semantic_program_scores": semantic_program_scores,
            "semantic_program_attention": semantic_program_attention,
            "semantic_program_tokens": semantic_program_tokens,
            "semantic_program_compatibility": semantic_program_compatibility,
            "semantic_program_region_scores": semantic_program_region_scores,
            "semantic_program_topology_scores": semantic_program_topology_scores,
            "semantic_program_composition_scores": semantic_program_composition_scores,
            "semantic_program_routing_entropy": semantic_program_routing_entropy,
            "semantic_program_bank": semantic_program_bank,
            "semantic_program_topology": semantic_program_topology,
            "semantic_latent_embedding": semantic_latent_embedding,
            "fused_latent_embedding": fused_latent,
            "region_mask": region_mask,
            "region_confidence": region_confidence,
            "invalid_region_indices": invalid_indices,
            "macro_embeddings": composed_states,
            "macro_motif_attention": semantic_program_attention,
            "micro_motif_bank": micro_motif_bank,
            "macro_motif_bank": semantic_program_bank,
            "aux_losses": {
                "semantic_consistency": semantic_motif_tokens.new_tensor(0.0),
            },
        }
