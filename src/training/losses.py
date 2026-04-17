import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FocalLoss(nn.Module):
    """Multi-class focal loss.
    alpha can be a tensor of per-class weights (same shape as class weights).
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal = alpha[targets] * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
 
def build_loss(config, class_weights=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
            class_weight=None: apply class weight or not?
    """
    loss_name = config['training'].get('loss', 'cross_entropy')
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if loss_name == 'cross_entropy':
        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == 'focal':
        gamma = config['training'].get('focal_gamma', 2.0)
        use_alpha = config['training'].get('focal_use_class_weights', True)
        alpha = class_weights if (use_alpha and class_weights is not None) else None
        loss = FocalLoss(gamma=gamma, alpha=alpha)
    
    else: 
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    return loss


if __name__ == "__main__":
    config_default = {'training': {}}
    loss_fn = build_loss(config_default)
    print(f"Test 1 (Default): {type(loss_fn)}") 
    # Expect: <class 'torch.nn.modules.loss.CrossEntropyLoss'>

    config_explicit = {'training': {'loss': 'cross_entropy'}}
    loss_fn = build_loss(config_explicit)
    print(f"Test 2 (Explicit): {type(loss_fn)}")
    # Expect: <class 'torch.nn.modules.loss.CrossEntropyLoss'>
    # Ok
