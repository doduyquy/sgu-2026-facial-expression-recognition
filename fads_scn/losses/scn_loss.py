import torch
import torch.nn as nn
import torch.nn.functional as F


class SCNLoss(nn.Module):
    """
    Self-Cure Network (SCN) Loss Suite for FER.
    Includes:
    1. Sample-Weighted Cross-Entropy Loss with Label Smoothing:
       L_SCN-CE = sum(alpha_i * CE_i) / (sum(alpha_i) + eps)
    2. Rank Regularization Loss:
       L_Rank = max(0, margin - (mean(alpha_clean) - mean(alpha_noisy)))
    3. Spatial Head Diversity Regularizer:
       L_Div penalizes spatial overlap between attention heads.
    """

    def __init__(
        self,
        num_classes: int = 7,
        label_smoothing: float = 0.05,
        margin: float = 0.15,
        clean_ratio: float = 0.70,
        rank_loss_weight: float = 0.10,
        div_loss_weight: float = 0.05,
        sparsity_loss_weight: float = 0.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.margin = margin
        self.clean_ratio = clean_ratio
        self.rank_loss_weight = rank_loss_weight
        self.div_loss_weight = div_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        current_epoch: int = 0,
        rank_warmup_epochs: int = 5,
    ):
        """
        Args:
            outputs: dict containing 'logits' [B, 7], 'alpha' [B, 1], 'diversity_loss' scalar
            targets: ground truth class indices [B]
            current_epoch: int current training epoch
            rank_warmup_epochs: int epochs before activating rank loss
        Returns:
            dict of losses
        """
        logits = outputs["logits"]
        alpha = outputs["alpha"].view(-1)  # [B]
        div_loss = outputs.get("diversity_loss", torch.tensor(0.0, device=logits.device))
        B = logits.shape[0]

        # 1. Per-sample Cross-Entropy Loss with Label Smoothing
        # ce_loss_per_sample: [B]
        ce_loss_per_sample = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        base_ce = ce_loss_per_sample.mean()

        # 2. SCN Weighted Cross-Entropy Loss
        # CRITICAL: Detach alpha so that minimizing classification loss does NOT pull alpha -> 0.
        # Alpha is exclusively trained by the Rank Regularization Loss.
        alpha_weights = alpha.detach()
        weighted_ce = (alpha_weights * ce_loss_per_sample).sum() / (alpha_weights.sum() + 1e-6)

        # Dual-anchor classification loss (base CE ensures constant gradient flow for all classes)
        cls_loss = 0.5 * base_ce + 0.5 * weighted_ce

        # 3. Rank Regularization Loss
        # Enforces that clean samples (low CE loss) have higher alpha than noisy samples (high CE loss)
        if B > 4:
            sorted_indices = torch.argsort(ce_loss_per_sample.detach())
            k_clean = max(1, int(B * self.clean_ratio))
            k_noisy = B - k_clean

            clean_indices = sorted_indices[:k_clean]
            noisy_indices = sorted_indices[k_clean:]

            mean_alpha_clean = alpha[clean_indices].mean()
            mean_alpha_noisy = alpha[noisy_indices].mean() if k_noisy > 0 else torch.tensor(0.0, device=alpha.device)

            rank_loss = F.relu(self.margin - (mean_alpha_clean - mean_alpha_noisy))
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)

        # Total multi-objective loss
        sparsity_loss = outputs.get("sparsity_loss", torch.tensor(0.0, device=logits.device))
        total_loss = cls_loss + (self.rank_loss_weight * rank_loss) + (self.div_loss_weight * div_loss)
        if self.sparsity_loss_weight > 0 and sparsity_loss is not None:
            total_loss = total_loss + (self.sparsity_loss_weight * sparsity_loss)

        return {
            "loss": total_loss,
            "weighted_ce": weighted_ce.item(),
            "base_ce": base_ce.item(),
            "rank_loss": rank_loss.item() if isinstance(rank_loss, torch.Tensor) else float(rank_loss),
            "div_loss": div_loss.item() if isinstance(div_loss, torch.Tensor) else float(div_loss),
            "sparsity_loss": sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) and sparsity_loss is not None else 0.0,
            "mean_alpha": alpha.mean().item(),
            "ce_per_sample": ce_loss_per_sample.detach(),
        }
