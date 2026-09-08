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
        use_scn: bool = True,
        rank_mode: str = "global",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.margin = margin
        self.clean_ratio = clean_ratio
        self.rank_loss_weight = rank_loss_weight
        self.div_loss_weight = div_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight
        self.use_scn = use_scn
        self.rank_mode = rank_mode
        if not 0 < clean_ratio < 1:
            raise ValueError("clean_ratio must be between 0 and 1")
        if rank_mode not in ("global", "classwise"):
            raise ValueError("rank_mode must be global or classwise")

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        targets_b: torch.Tensor = None,
        lam: float = 1.0,
        current_epoch: int = 0,
        rank_warmup_epochs: int = 5,
    ):
        """
        Args:
            outputs: dict containing 'logits' [B, 7], 'alpha' [B, 1], 'diversity_loss' scalar
            targets: ground truth class indices [B]
            targets_b: optional second ground truth indices for Mixup [B]
            lam: float mixing ratio for Mixup in [0, 1]
            current_epoch: int current training epoch
            rank_warmup_epochs: int epochs before activating rank loss
        Returns:
            dict of losses
        """
        logits = outputs["logits"].float()
        alpha = outputs["alpha"].view(-1).float()  # [B]
        div_loss = outputs.get("diversity_loss", torch.tensor(0.0, device=logits.device))
        B = logits.shape[0]
        mixup_active = targets_b is not None and lam < 1.0

        # 1. Per-sample Cross-Entropy Loss with Label Smoothing & Mixup
        scn_active = (
            self.use_scn and self.rank_loss_weight > 0 and not mixup_active
            and current_epoch >= rank_warmup_epochs and B > 4
        )
        if mixup_active:
            ce_a = F.cross_entropy(
                logits,
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            ce_b = F.cross_entropy(
                logits,
                targets_b,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            ce_loss_per_sample = lam * ce_a + (1.0 - lam) * ce_b
        else:
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
        if not scn_active:
            # No sample weighting before warmup, on Mixup, or when SCN is disabled.
            weighted_ce = base_ce
            cls_loss = base_ce
        else:
            alpha_weights = alpha.detach()
            weighted_ce = (alpha_weights * ce_loss_per_sample).sum() / (alpha_weights.sum() + 1e-6)

            # Dual-anchor classification loss (base CE ensures constant gradient flow for all classes)
            cls_loss = 0.5 * base_ce + 0.5 * weighted_ce

        # 3. Rank Regularization Loss
        # Enforces that clean samples (low CE loss) have higher alpha than noisy samples (high CE loss)
        rank_is_active = scn_active
        noisy_mask = torch.zeros(B, dtype=torch.bool, device=logits.device)
        ranked_mask = torch.zeros_like(noisy_mask)
        rank_loss = logits.new_zeros(())
        if rank_is_active:
            # Class balancing and smoothing must not define label reliability.
            rank_scores = F.cross_entropy(logits.detach(), targets, reduction="none")
            groups = [torch.arange(B, device=logits.device)] if self.rank_mode == "global" else [
                (targets == cls).nonzero(as_tuple=True)[0] for cls in range(self.num_classes)
            ]
            terms = []
            for indices in groups:
                if indices.numel() < 2:
                    continue
                ordered = indices[torch.argsort(rank_scores[indices], stable=True)]
                k = min(len(ordered) - 1, max(1, int(len(ordered) * self.clean_ratio)))
                clean, noisy = ordered[:k], ordered[k:]
                ranked_mask[indices] = True
                noisy_mask[noisy] = True
                terms.append(F.relu(self.margin - (alpha[clean].mean() - alpha[noisy].mean())))
            if terms:
                rank_loss = torch.stack(terms).mean()

        # Total multi-objective loss
        sparsity_loss = outputs.get("sparsity_loss", torch.tensor(0.0, device=logits.device))
        total_loss = cls_loss + (self.rank_loss_weight * rank_loss) + (self.div_loss_weight * div_loss)
        if self.sparsity_loss_weight > 0 and sparsity_loss is not None:
            total_loss = total_loss + (self.sparsity_loss_weight * sparsity_loss)

        return {
            "loss": total_loss,
            "weighted_ce": weighted_ce.detach(),
            "base_ce": base_ce.detach(),
            "rank_loss": rank_loss.detach(),
            "div_loss": torch.as_tensor(div_loss).detach(),
            "sparsity_loss": torch.as_tensor(sparsity_loss if sparsity_loss is not None else 0.0).detach(),
            "mean_alpha": alpha.mean().detach(),
            "rank_active": bool(rank_is_active),
            "scn_active": bool(scn_active),
            "noisy_mask": noisy_mask,
            "ranked_mask": ranked_mask,
            "mixup_active": bool(mixup_active),
            "ce_per_sample": ce_loss_per_sample.detach(),
        }
