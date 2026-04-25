"""Loss functions for FER training, including motif-guided objectives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FER2013_TRAIN_COUNTS = [3995, 436, 4097, 7215, 4830, 3171, 4965]


def inception_loss(main_out, aux_out, targets, criterion=nn.CrossEntropyLoss(), aux_weight: float = 0.3):
    """Tính loss có auxiliary."""
    main_loss = criterion(main_out, targets)
    aux_loss = criterion(aux_out, targets)
    return main_loss + aux_weight * aux_loss


def compute_class_weights(
    class_counts,
    normalize_mean: bool = True,
    power: float = 1.0,
) -> torch.Tensor:
    """FER-style inverse-frequency weights, optionally softened by a power."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError(f"class_counts must be positive, got {class_counts}")
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights.pow(float(power))
    if normalize_mean:
        weights = weights / weights.mean().clamp_min(1e-8)
    return weights


class WeightedCrossEntropy(nn.Module):
    """Cross entropy with optional class weights and label smoothing."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce).clamp(1e-8, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


class MotifConsistencyLoss(nn.Module):
    """
    Encourage image motif-score vector to favor the ground-truth emotion bank.

    loss = mean relu(margin - score_true + max_other)
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, motif_score_vector: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if motif_score_vector.ndim != 2:
            raise ValueError(
                f"Expected motif_score_vector [B, C], got {tuple(motif_score_vector.shape)}"
            )
        B, C = motif_score_vector.shape
        if C < 2:
            return motif_score_vector.sum() * 0.0

        labels = labels.long()
        row = torch.arange(B, device=motif_score_vector.device)
        score_true = motif_score_vector[row, labels]
        other_scores = motif_score_vector.clone()
        other_scores[row, labels] = -1e9
        score_other = other_scores.max(dim=1).values
        return F.relu(self.margin - score_true + score_other).mean()


class CombinedMotifLoss(nn.Module):
    """Classification loss plus motif-consistency loss."""

    def __init__(
        self,
        cls_loss: nn.Module,
        lambda_motif: float = 0.1,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.cls_loss = cls_loss
        self.motif_loss = MotifConsistencyLoss(margin=margin)
        self.lambda_motif = float(lambda_motif)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch: Optional[dict] = None,
        motif_score_vector: Optional[torch.Tensor] = None,
    ) -> dict:
        cls_loss = self.cls_loss(logits, targets)
        if motif_score_vector is None and batch is not None:
            motif_score_vector = batch.get("motif_score_vector")
        if motif_score_vector is None:
            motif_loss = cls_loss.new_tensor(0.0)
        else:
            motif_score_vector = motif_score_vector.to(device=logits.device, dtype=logits.dtype)
            motif_loss = self.motif_loss(motif_score_vector, targets)
        total = cls_loss + self.lambda_motif * motif_loss
        return {
            "loss": total,
            "cls_loss": cls_loss,
            "motif_loss": motif_loss,
        }


def _resolve_loss_cfg(config: dict) -> dict:
    loss_cfg = dict(config.get("loss", {}) or {})
    training_cfg = config.get("training", {}) or {}
    if "name" not in loss_cfg:
        loss_cfg["name"] = training_cfg.get("loss", "cross_entropy")
    if "label_smoothing" not in loss_cfg:
        loss_cfg["label_smoothing"] = training_cfg.get("label_smoothing", 0.0)
    return loss_cfg


def _maybe_class_weights(loss_cfg: dict, class_weights=None) -> Optional[torch.Tensor]:
    if class_weights is not None:
        return class_weights.float()
    if not loss_cfg.get("use_class_weights", False):
        return None
    counts = loss_cfg.get("class_counts", FER2013_TRAIN_COUNTS)
    power = float(loss_cfg.get("class_weight_power", 1.0))
    return compute_class_weights(counts, normalize_mean=True, power=power)


def build_loss(config, class_weights=None):
    """Build a loss module from legacy training config or new loss config."""
    loss_cfg = _resolve_loss_cfg(config)
    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    weights = _maybe_class_weights(loss_cfg, class_weights=class_weights)

    if loss_name in {"cross_entropy", "ce"}:
        return WeightedCrossEntropy(class_weights=weights, label_smoothing=label_smoothing)

    if loss_name in {"weighted_ce", "weighted_cross_entropy"}:
        if weights is None:
            weights = compute_class_weights(
                loss_cfg.get("class_counts", FER2013_TRAIN_COUNTS),
                power=loss_cfg.get("class_weight_power", 1.0),
            )
        return WeightedCrossEntropy(class_weights=weights, label_smoothing=label_smoothing)

    if loss_name == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=None,
            label_smoothing=label_smoothing,
        )

    if loss_name == "weighted_focal":
        if weights is None:
            weights = compute_class_weights(
                loss_cfg.get("class_counts", FER2013_TRAIN_COUNTS),
                power=loss_cfg.get("class_weight_power", 1.0),
            )
        return FocalLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=weights,
            label_smoothing=label_smoothing,
        )

    if loss_name == "weighted_ce_motif":
        if weights is None:
            weights = compute_class_weights(
                loss_cfg.get("class_counts", FER2013_TRAIN_COUNTS),
                power=loss_cfg.get("class_weight_power", 1.0),
            )
        cls_loss = WeightedCrossEntropy(class_weights=weights, label_smoothing=label_smoothing)
        return CombinedMotifLoss(
            cls_loss=cls_loss,
            lambda_motif=loss_cfg.get("lambda_motif", 0.1),
            margin=loss_cfg.get("margin", 0.2),
        )

    if loss_name in {"ce_motif", "cross_entropy_motif"}:
        cls_loss = WeightedCrossEntropy(class_weights=None, label_smoothing=label_smoothing)
        return CombinedMotifLoss(
            cls_loss=cls_loss,
            lambda_motif=loss_cfg.get("lambda_motif", 0.1),
            margin=loss_cfg.get("margin", 0.2),
        )

    raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")


if __name__ == "__main__":
    config_default = {"training": {}}
    loss_fn = build_loss(config_default)
    print(f"Test 1 (Default): {type(loss_fn)}")

    config_motif = {"loss": {"name": "weighted_ce_motif", "use_class_weights": True}}
    loss_fn = build_loss(config_motif)
    print(f"Test 2 (Motif): {type(loss_fn)}")
