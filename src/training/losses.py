import torch
import torch.nn as nn
import torch.nn.functional as F


def confidence_soft_target_loss(
    logits,
    targets,
    *,
    max_mix=0.2,
    min_confidence=0.55,
    confidence_power=1.0,
    label_smoothing=0.0,
    class_weights=None,
):
    """
    Cross-entropy against a confidence-adaptive soft target.

    The target starts from the usual smoothed hard label, then blends in the
    model's detached prediction only when the prediction is sufficiently
    confident. This keeps early / uncertain updates anchored to ground truth.
    """
    if logits.dim() != 2:
        raise ValueError("confidence_soft_target_loss expects logits shaped [B, C].")
    if not 0.0 <= max_mix <= 1.0:
        raise ValueError("max_mix must be in [0, 1].")
    if not 0.0 <= min_confidence < 1.0:
        raise ValueError("min_confidence must be in [0, 1).")
    if confidence_power <= 0.0:
        raise ValueError("confidence_power must be > 0.")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")

    num_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)

    with torch.no_grad():
        teacher_probs = F.softmax(logits.detach(), dim=-1)
        confidence = teacher_probs.max(dim=-1).values
        confidence = (
            (confidence - min_confidence)
            / max(1.0 - min_confidence, 1e-8)
        ).clamp_(0.0, 1.0)
        mix = (max_mix * confidence.pow(confidence_power)).unsqueeze(1)

        hard_targets = torch.full_like(log_probs, label_smoothing / num_classes)
        hard_targets.scatter_(
            1,
            targets.unsqueeze(1),
            1.0 - label_smoothing + label_smoothing / num_classes,
        )
        soft_targets = (1.0 - mix) * hard_targets + mix * teacher_probs

    loss_matrix = -(soft_targets * log_probs)
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
        weighted_loss = (loss_matrix * weights).sum(dim=1)
        normalizer = (soft_targets * weights).sum().clamp_min(1e-8)
        return weighted_loss.sum() / normalizer

    return loss_matrix.sum(dim=1).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.
    Addresses class imbalance by down-weighting well-classified examples.
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Lấy xác suất đúng của class mục tiêu (pt) CHUẨN XÁC, không bị nhiễu bởi label_smoothing hay class_weights
        log_probs = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        
        # Tính Cross Entropy phân bổ rải rác (chưa tính trung bình)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        
        # Focal weights
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Trả về kết quả CHUẨN theo style của PyTorch với Weight
        if self.weight is not None:
            # Pytorch CrossEntropyLoss (reduction='mean') sẽ chia tổng loss cho TỔNG weight của mẻ batch
            batch_weights = self.weight.gather(0, targets)
            return focal_loss.sum() / batch_weights.sum().clamp(min=1e-8)
        else:
            return focal_loss.mean()

# Loss -> auxiliary (training)
def inception_loss(main_out, aux_out, targets,
                   criterion=nn.CrossEntropyLoss(),
                   aux_weight: float = 0.3):
    """Tính loss có auxiliary.
    total_loss = main_loss + aux_weight * aux_loss
    """
    main_loss = criterion(main_out, targets)
    aux_loss  = criterion(aux_out,  targets)
    return main_loss + aux_weight * aux_loss
 
def build_loss(config, class_weights=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
            class_weight=None: apply class weight or not?
    """
    loss_name = config['training'].get('loss', 'cross_entropy')
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if loss_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    elif loss_name == 'focal':
        gamma = config['training'].get('focal_gamma', 2.0)
        loss = FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)
    else: 
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    return loss

if __name__ == "__main__":
    config_default = {'training': {}}
    loss_fn = build_loss(config_default)
    print(f"Test 1 (Default): {type(loss_fn)}") 

    config_explicit = {'training': {'loss': 'focal'}}
    loss_fn = build_loss(config_explicit)
    print(f"Test 2 (Focal): {type(loss_fn)}")
