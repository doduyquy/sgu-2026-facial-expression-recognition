"""
Loss functions for Semantic ROI Graph FER model.

This module provides standalone loss functions for the dual-level
semantic ROI graph architecture:
- micro motif diversity
- macro motif diversity
- region supervised contrastive loss
- relation consistency loss
- optional topology regularization
"""

from typing import Dict

import torch
import torch.nn.functional as F


def _unwrap_model(model):
    return getattr(model, "module", model)


def _get_training_cfg(model) -> Dict:
    try:
        base_model = _unwrap_model(model)
        return dict(getattr(base_model, "config", {}).get("training", {}))
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
    if motifs.dim() == 3:
        return micro_motif_diversity_loss(lambda: motifs)
    return macro_motif_diversity_loss(lambda: motifs)


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


def relation_consistency_loss(
    topology_matrix: torch.Tensor,
    labels: torch.Tensor,
    region_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage topology matrices of the same class to stay close."""
    labels = labels.view(-1)
    loss = 0.0
    count = 0

    for cls in labels.unique():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_topology = topology_matrix[mask]
        if region_mask is not None:
            cls_mask = region_mask[mask].float()
            pair_mask = cls_mask.unsqueeze(-1) * cls_mask.unsqueeze(-2)
            cls_topology = cls_topology * pair_mask
        mean_topology = cls_topology.mean(dim=0, keepdim=True)
        loss = loss + ((cls_topology - mean_topology) ** 2).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=topology_matrix.device)

    return loss / count


def topology_regularization_loss(topology_matrix: torch.Tensor) -> torch.Tensor:
    """Light regularizer to avoid highly noisy topology tensors."""
    centered = topology_matrix - topology_matrix.mean(dim=(-1, -2), keepdim=True)
    return (centered ** 2).mean()


def region_consistency_loss(region_embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias that preserves previous region consistency behavior."""
    labels = labels.view(-1)
    loss = 0.0
    count = 0
    
    for cls in labels.unique():
        mask = labels == cls
        if mask.sum() < 2:
            continue
        cls_embeddings = region_embeddings[mask]
        mean = cls_embeddings.mean(dim=0, keepdim=True)
        loss = loss + ((cls_embeddings - mean) ** 2).mean()
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=region_embeddings.device)
    
    return loss / count


def compute_semantic_roi_graph_losses(
    model,
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    temperature: float | None = None,
    region_contrastive_weight: float | None = None,
    micro_diversity_weight: float | None = None,
    macro_diversity_weight: float | None = None,
    relation_consistency_weight: float | None = None,
    topology_reg_weight: float | None = None,
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
        relation_consistency_weight = float(training_cfg.get("relation_consistency_weight", training_cfg.get("region_consistency_weight", 0.1)))
    if topology_reg_weight is None:
        topology_reg_weight = float(training_cfg.get("topology_reg_weight", 0.0))

    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    try:
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    except TypeError:
        ce_loss = F.cross_entropy(logits, labels)

    base_model = _unwrap_model(model)

    micro_diversity_loss = micro_motif_diversity_loss(base_model.micro_motif_bank)
    macro_diversity_loss = macro_motif_diversity_loss(base_model.macro_motif_bank)
    contrastive_loss = region_supervised_contrastive_loss(
        outputs.get("macro_embeddings"),
        labels,
        temperature=temperature,
        region_mask=outputs.get("region_mask"),
    )
    topology_matrix = outputs.get("topology_matrix")
    if topology_matrix is None:
        topology_matrix = base_model.macro_motif_matcher.relation_matrix(outputs.get("macro_embeddings"))

    consistency_loss = relation_consistency_loss(
        topology_matrix,
        labels,
        region_mask=outputs.get("region_mask"),
    )
    topology_loss = topology_regularization_loss(topology_matrix)

    total = ce_loss
    total = total + float(micro_diversity_weight) * micro_diversity_loss
    total = total + float(macro_diversity_weight) * macro_diversity_loss
    total = total + float(region_contrastive_weight) * contrastive_loss
    total = total + float(relation_consistency_weight) * consistency_loss
    total = total + float(topology_reg_weight) * topology_loss
    
    return {
        "loss": total,
        "loss_ce": ce_loss,
        "loss_micro_motif_diversity": micro_diversity_loss,
        "loss_macro_motif_diversity": macro_diversity_loss,
        "loss_motif_diversity": micro_diversity_loss + macro_diversity_loss,
        "loss_contrastive": contrastive_loss,
        "loss_relation_consistency": consistency_loss,
        "loss_region_consistency": consistency_loss,
        "loss_topology_reg": topology_loss,
    }
