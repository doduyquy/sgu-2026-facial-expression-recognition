"""
Loss functions for Semantic ROI Graph FER model.

This module provides standalone loss functions for the dual-level
semantic ROI graph architecture:
- micro motif diversity
- macro semantic program diversity
- semantic supervised contrastive loss
- semantic consistency loss
- compositional motif consistency
- semantic disentanglement
- region coordination regularization
"""

from typing import Dict

import torch
import torch.nn.functional as F


def _unwrap_model(model):
    return getattr(model, "module", model)


def _get_training_cfg(model) -> Dict:
    try:
        base_model = _unwrap_model(model)
        return getattr(base_model, "training_cfg", {})
    except Exception:
        return {}


def micro_motif_diversity_loss(motif_bank_fn) -> torch.Tensor:
    """Encourage diverse motifs within each semantic region bank."""
    motifs = motif_bank_fn()  # (R, K, D)
    r, k, d = motifs.shape
    motifs = F.normalize(motifs.view(r, k, d), dim=-1)
    sim = torch.einsum("rkd,rgd->rkg", motifs, motifs)
    identity = torch.eye(k, device=sim.device).unsqueeze(0)
    off_diag = sim * (1.0 - identity)
    return (off_diag ** 2).mean()


def macro_motif_diversity_loss(motif_bank_fn) -> torch.Tensor:
    """Encourage diverse macro motifs across class topology prototypes."""
    motifs = motif_bank_fn()  # (C, M, R, D)
    if isinstance(motifs, tuple):
        motifs = motifs[0]
    if motifs.dim() == 3:
        c, m, d = motifs.shape
        motifs = motifs.view(c, m, d)
    else:
        c, m, r, d = motifs.shape
        motifs = motifs.view(c, m, r * d)
    motifs = F.normalize(motifs, dim=-1)
    sim = torch.einsum("cmd,cnd->cmn", motifs, motifs)
    identity = torch.eye(m, device=sim.device).unsqueeze(0)
    off_diag = sim * (1.0 - identity)
    return (off_diag ** 2).mean()


def motif_diversity_loss(motif_bank_fn) -> torch.Tensor:
    """Backward-compatible alias for macro motif diversity."""
    motifs = motif_bank_fn()
    if isinstance(motifs, tuple):
        motifs = motifs[0]
    if motifs.dim() == 3:
        return micro_motif_diversity_loss(lambda: motifs)
    return macro_motif_diversity_loss(lambda: motifs)


def compositional_program_consistency_loss(program_scores: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    """Encourage the correct semantic facial program to dominate execution output."""
    if program_scores is None:
        return torch.tensor(0.0, device=labels.device)
    return F.cross_entropy(program_scores, labels)


def topology_alignment_loss(
    predicted_topology: torch.Tensor | None,
    program_topology: torch.Tensor | None,
    labels: torch.Tensor,
    program_attention: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align observed region coordination with the selected semantic program topology."""
    if predicted_topology is None or program_topology is None:
        device = predicted_topology.device if predicted_topology is not None else labels.device
        return torch.tensor(0.0, device=device)

    selected_topology = program_topology[labels]
    if program_attention is not None:
        selected_attention = program_attention[torch.arange(program_attention.size(0), device=labels.device), labels]
        selected_topology = (selected_attention.unsqueeze(-1).unsqueeze(-1) * selected_topology).sum(dim=1)
    else:
        selected_topology = selected_topology.mean(dim=1)

    if predicted_topology.dim() == 4:
        predicted_topology = predicted_topology.mean(dim=1)

    return F.mse_loss(predicted_topology, selected_topology)


def region_composition_contrastive_loss(
    cross_region_tokens: torch.Tensor | None,
    labels: torch.Tensor,
    region_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrast higher-order cross-region semantic compositions across emotions."""
    if cross_region_tokens is None:
        return torch.tensor(0.0, device=labels.device)
    return region_supervised_contrastive_loss(cross_region_tokens, labels, temperature=temperature, region_mask=None)


def semantic_program_sparsity_loss(
    program_attention: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
    cross_region_attention: torch.Tensor | None = None,
    mode: str = "l1",
) -> torch.Tensor:
    """Sparsity / load-balance loss for programs and routing.

    Args:
        mode: 'l1' to use L1 sparsity (encourages sparse activations),
              'entropy' to use entropy-maximization (encourage balanced use).
    """
    losses = []

    def _entropy(attn: torch.Tensor) -> torch.Tensor:
        attn = attn.clamp_min(1e-6)
        entropy = -(attn * attn.log()).sum(dim=-1)
        denom = torch.log(torch.tensor(float(attn.size(-1)), device=attn.device)).clamp_min(1e-6)
        return (entropy / denom).mean()

    def _l1(attn: torch.Tensor) -> torch.Tensor:
        return attn.abs().mean()

    if program_attention is not None:
        losses.append(program_attention)
    if routing_weights is not None:
        losses.append(routing_weights)
    if cross_region_attention is not None:
        if cross_region_attention.dim() == 4:
            attn = cross_region_attention.mean(dim=1)
        else:
            attn = cross_region_attention
        losses.append(attn)

    if not losses:
        if program_attention is not None:
            device = program_attention.device
        elif routing_weights is not None:
            device = routing_weights.device
        elif cross_region_attention is not None:
            device = cross_region_attention.device
        else:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    if mode == "entropy":
        vals = [_entropy(x) for x in losses]
        # We want to MAXIMIZE entropy to encourage load balancing.
        # Return negative entropy so that minimizing loss increases entropy.
        return -sum(vals) / float(len(vals))
    else:
        vals = [_l1(x) for x in losses]
        return sum(vals) / float(len(vals))


def program_diversity_loss(program_bank) -> torch.Tensor:
    """Encourage different semantic facial programs to specialize."""
    if callable(program_bank):
        program_bank = program_bank()
    if isinstance(program_bank, tuple):
        program_bank = program_bank[0]

    if program_bank.dim() == 4:
        summaries = program_bank.mean(dim=2)
    else:
        summaries = program_bank

    summaries = summaries.reshape(-1, summaries.size(-1))
    if summaries.size(0) < 2:
        return torch.tensor(0.0, device=summaries.device)

    summaries = F.normalize(summaries, dim=-1)
    sim = summaries @ summaries.t()
    identity = torch.eye(sim.size(0), device=sim.device)
    off_diag = sim * (1.0 - identity)
    return (off_diag ** 2).mean()


def semantic_consistency_loss(
    semantic_states: torch.Tensor,
    labels: torch.Tensor,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage samples from the same class to share similar semantic facial states."""
    if semantic_states.dim() == 3:
        if region_mask is not None:
            weights = region_mask.unsqueeze(-1).float()
            pooled = (semantic_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = semantic_states.mean(dim=1)
    else:
        pooled = semantic_states

    labels = labels.view(-1)
    loss = 0.0
    count = 0

    for cls in labels.unique():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_states = pooled[mask]
        center = cls_states.mean(dim=0, keepdim=True)
        loss = loss + ((cls_states - center) ** 2).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=pooled.device)

    return loss / count


def compositional_motif_consistency_loss(program_scores: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    """Align semantic latent emotion representations with the correct class program."""
    if program_scores is None:
        return torch.tensor(0.0, device=labels.device)
    return F.cross_entropy(program_scores, labels)


def semantic_disentanglement_loss(
    semantic_states: torch.Tensor,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce redundancy across semantic state channels."""
    if semantic_states.dim() == 3:
        if region_mask is not None:
            flat_mask = region_mask.reshape(-1) > 0
            tokens = semantic_states.reshape(-1, semantic_states.size(-1))[flat_mask]
        else:
            tokens = semantic_states.reshape(-1, semantic_states.size(-1))
    else:
        tokens = semantic_states.reshape(-1, semantic_states.size(-1))

    if tokens.size(0) < 2:
        return torch.tensor(0.0, device=tokens.device)

    centered = tokens - tokens.mean(dim=0, keepdim=True)
    cov = centered.t().mm(centered) / float(tokens.size(0) - 1)
    diag = torch.diag(torch.diag(cov))
    off_diag = cov - diag
    return (off_diag ** 2).mean()


def region_coordination_regularization(
    routing_weights: torch.Tensor | None,
    interaction_gates: torch.Tensor | None = None,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regularize how strongly regions coordinate through routing and interactions."""
    loss = None

    if routing_weights is not None:
        weights = routing_weights.clamp_min(1e-6)
        entropy = -(weights * weights.log()).sum(dim=1)
        denom = torch.log(torch.tensor(float(weights.size(1)), device=weights.device)).clamp_min(1e-6)
        loss = (entropy / denom).mean()

    if interaction_gates is not None:
        gates = interaction_gates
        if region_mask is not None:
            pair_mask = region_mask.unsqueeze(-1) * region_mask.unsqueeze(-2)
            gates = gates * pair_mask
        active_mean = gates.mean(dim=(-1, -2))
        gate_balance = ((active_mean - 0.35) ** 2).mean()
        gate_variance = gates.var(dim=(-1, -2)).mean()
        gate_loss = gate_balance + 0.05 * gate_variance
        loss = gate_loss if loss is None else loss + gate_loss

    if loss is None:
        if interaction_gates is not None:
            device = interaction_gates.device
        elif routing_weights is not None:
            device = routing_weights.device
        else:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    return loss


def relation_consistency_loss(
    topology_matrix: torch.Tensor,
    labels: torch.Tensor,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for semantic coordination regularization."""
    return region_coordination_regularization(topology_matrix, None, region_mask)


def topology_regularization_loss(topology_matrix: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for semantic disentanglement loss."""
    return semantic_disentanglement_loss(topology_matrix)


def region_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervised contrastive loss on pooled region embeddings."""
    if region_mask is not None:
        weights = region_mask.unsqueeze(-1).float()
        pooled = (embeddings * weights).sum(dim=1) / (weights.sum(dim=1).clamp_min(1.0))
    else:
        pooled = embeddings.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)
    sim = torch.matmul(pooled, pooled.t()) / float(temperature)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Backward-compatible alias for region supervised contrastive loss."""
    return region_supervised_contrastive_loss(embeddings, labels, temperature=temperature)


def region_consistency_loss(region_embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for semantic consistency loss."""
    return semantic_consistency_loss(region_embeddings, labels)


def compute_semantic_roi_graph_losses(
    model,
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    temperature: float | None = None,
    region_contrastive_weight: float | None = None,
    micro_diversity_weight: float | None = None,
    macro_diversity_weight: float | None = None,
    relation_consistency_weight: float | None = None,
    topology_reg_weight: float | None = None,
    semantic_consistency_weight: float | None = None,
    compositional_motif_weight: float | None = None,
    semantic_disentanglement_weight: float | None = None,
    region_coordination_weight: float | None = None,
    compositional_program_weight: float | None = None,
    topology_alignment_weight: float | None = None,
    region_composition_contrastive_weight: float | None = None,
    program_sparsity_weight: float | None = None,
    program_diversity_weight: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all losses for Semantic ROI Graph FER.
    
    Args:
        model: SemanticROIGraphFER instance (to access motif_bank)
        outputs: Dict with keys 'logits', 'macro_embeddings', 'region_embeddings'
        labels: (B,) class labels
        temperature: Temperature for contrastive loss
        region_contrastive_weight: Weight for region contrastive loss
        micro_diversity_weight: Weight for micro motif diversity
        macro_diversity_weight: Weight for macro motif diversity
        relation_consistency_weight: Weight for relation consistency
        topology_reg_weight: Weight for topology regularization
    
    Returns:
        Dict with loss components and total loss
    """
    logits = outputs["logits"]

    # Read weights/params from model.config if not explicitly provided
    training_cfg = _get_training_cfg(model)

    if temperature is None:
        temperature = float(training_cfg.get("contrastive_temperature", 0.07))
    if region_contrastive_weight is None:
        region_contrastive_weight = float(training_cfg.get("region_contrastive_weight", training_cfg.get("au_contrastive_weight", 0.1)))
    if micro_diversity_weight is None:
        micro_diversity_weight = float(training_cfg.get("micro_motif_diversity_weight", training_cfg.get("motif_diversity_weight", 0.05)))
    if macro_diversity_weight is None:
        macro_diversity_weight = float(training_cfg.get("macro_motif_diversity_weight", training_cfg.get("motif_diversity_weight", 0.05)))
    if relation_consistency_weight is None:
        relation_consistency_weight = float(training_cfg.get("region_coordination_weight", training_cfg.get("relation_consistency_weight", 0.1)))
    if topology_reg_weight is None:
        topology_reg_weight = float(training_cfg.get("semantic_disentanglement_weight", training_cfg.get("topology_reg_weight", 0.0)))
    if semantic_consistency_weight is None:
        semantic_consistency_weight = float(training_cfg.get("semantic_consistency_weight", training_cfg.get("region_consistency_weight", 0.1)))
    if compositional_motif_weight is None:
        compositional_motif_weight = float(training_cfg.get("compositional_motif_weight", training_cfg.get("macro_motif_consistency_weight", 0.1)))
    if semantic_disentanglement_weight is None:
        semantic_disentanglement_weight = float(training_cfg.get("semantic_disentanglement_weight", training_cfg.get("topology_reg_weight", 0.01)))
    if region_coordination_weight is None:
        region_coordination_weight = float(training_cfg.get("region_coordination_weight", training_cfg.get("relation_consistency_weight", 0.1)))
    if compositional_program_weight is None:
        compositional_program_weight = float(training_cfg.get("compositional_program_weight", 0.1))
    if topology_alignment_weight is None:
        topology_alignment_weight = float(training_cfg.get("topology_alignment_weight", training_cfg.get("region_coordination_weight", 0.1)))
    if region_composition_contrastive_weight is None:
        region_composition_contrastive_weight = float(training_cfg.get("region_composition_contrastive_weight", training_cfg.get("region_contrastive_weight", 0.1)))
    if program_sparsity_weight is None:
        program_sparsity_weight = float(training_cfg.get("program_sparsity_weight", 0.05))
    if program_diversity_weight is None:
        program_diversity_weight = float(training_cfg.get("program_diversity_weight", 0.05))

    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    try:
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, weight=class_weights)
    except TypeError:
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights)

    base_model = _unwrap_model(model)

    semantic_states = outputs.get("semantic_state_tokens")
    if semantic_states is None:
        semantic_states = outputs.get("region_embeddings")

    region_mask = outputs.get("region_mask")
    routing_weights = outputs.get("semantic_routing_weights")
    interaction_gates = outputs.get("semantic_interaction_gates")
    program_scores = outputs.get("semantic_program_scores")
    program_attention = outputs.get("semantic_program_attention")
    semantic_latent = outputs.get("semantic_latent_embedding")
    if semantic_latent is None:
        semantic_latent = outputs.get("macro_embeddings")
    cross_region_tokens = outputs.get("cross_region_tokens")
    cross_region_attention = outputs.get("cross_region_attention")
    program_topology = outputs.get("semantic_program_topology")

    micro_diversity_loss = micro_motif_diversity_loss(base_model.micro_motif_bank)
    macro_diversity_loss = macro_motif_diversity_loss(base_model.semantic_program_bank)
    contrastive_source = semantic_states if semantic_states is not None else semantic_latent
    contrastive_region_mask = region_mask if contrastive_source is not None and contrastive_source.dim() == 3 else None
    # Warn 4 fix: guard against both semantic_states and semantic_latent being None.
    if contrastive_source is not None:
        contrastive_loss = region_supervised_contrastive_loss(
            contrastive_source,
            labels,
            temperature=temperature,
            region_mask=contrastive_region_mask,
        )
    else:
        contrastive_loss = torch.tensor(0.0, device=labels.device)
    semantic_consistency = semantic_consistency_loss(semantic_states, labels, region_mask=region_mask)
    compositional_loss = compositional_program_consistency_loss(program_scores, labels)
    disentanglement_loss = semantic_disentanglement_loss(semantic_states, region_mask=region_mask)
    coordination_loss = region_coordination_regularization(routing_weights, interaction_gates, region_mask=region_mask)
    # Bug 4 note: `interaction_gates` (B, R, R) from SemanticInteractionBlock IS the
    # observed pairwise region coordination topology. Passing it as `predicted_topology`
    # to compare against the learnable program topology prototypes is intentional.
    topology_loss = topology_alignment_loss(interaction_gates, program_topology, labels, program_attention=program_attention)
    composition_contrastive_loss = region_composition_contrastive_loss(cross_region_tokens, labels, region_mask=region_mask, temperature=temperature)
    use_entropy = bool(training_cfg.get("use_entropy_sparsity", False))
    sparsity_mode = "entropy" if use_entropy else "l1"
    sparsity_loss = semantic_program_sparsity_loss(
        program_attention=program_attention,
        routing_weights=routing_weights,
        cross_region_attention=cross_region_attention,
        mode=sparsity_mode,
    )
    diversity_loss = program_diversity_loss(base_model.semantic_program_bank)

    total = ce_loss
    total = total + float(micro_diversity_weight) * micro_diversity_loss
    total = total + float(macro_diversity_weight) * macro_diversity_loss
    total = total + float(region_contrastive_weight) * contrastive_loss
    total = total + float(semantic_consistency_weight) * semantic_consistency
    total = total + float(compositional_program_weight) * compositional_loss
    total = total + float(semantic_disentanglement_weight) * disentanglement_loss
    total = total + float(region_coordination_weight) * coordination_loss
    total = total + float(topology_alignment_weight) * topology_loss
    total = total + float(region_composition_contrastive_weight) * composition_contrastive_loss
    total = total + float(program_sparsity_weight) * sparsity_loss
    total = total + float(program_diversity_weight) * diversity_loss
    
    return {
        "loss": total,
        "loss_ce": ce_loss,
        "loss_micro_motif_diversity": micro_diversity_loss,
        "loss_macro_motif_diversity": macro_diversity_loss,
        "loss_motif_diversity": micro_diversity_loss + macro_diversity_loss,
        "loss_contrastive": contrastive_loss,
        "loss_semantic_consistency": semantic_consistency,
        "loss_region_consistency": semantic_consistency,
        "loss_compositional_motif_consistency": compositional_loss,
        "loss_compositional_program_consistency": compositional_loss,
        "loss_semantic_disentanglement": disentanglement_loss,
        "loss_topology_reg": topology_loss,
        "loss_topology_alignment": topology_loss,
        "loss_region_composition_contrastive": composition_contrastive_loss,
        "loss_program_sparsity": sparsity_loss,
        "loss_program_diversity": diversity_loss,
        "loss_region_coordination": coordination_loss,
        "loss_relation_consistency": coordination_loss,
    }
