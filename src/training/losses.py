import torch.nn as nn 
from .loss import FocalLoss

def build_loss(config, class_weights=None):
    """ Define loss for training.

    Supported losses:
        - cross_entropy (default)
        - focal_loss

    Args:
        config: dict config loaded from yaml
        class_weights (Tensor, optional): class weights tensor

    Returns:
        loss function (nn.Module)
    """

    loss_name = config['training'].get('loss', 'cross_entropy').lower()

    # =========================
    # Cross Entropy Loss
    # =========================
    if loss_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(weight=class_weights)

    # =========================
    # Focal Loss
    # =========================
    elif loss_name == 'focal_loss':
        gamma = config['training'].get('gamma', 2.0)
        reduction = config['training'].get('reduction', 'mean')

        loss = FocalLoss(
            weight=class_weights,
            gamma=gamma,
            reduction=reduction
        )

    # =========================
    # Not supported
    # =========================
    else:
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    print(f"[Loss] Using loss: {loss_name}")
    if loss_name == "focal_loss":
        print(f"[Loss] gamma = {gamma}, reduction = {reduction}")
    if class_weights is not None:
        print(f"[Loss] class_weights = {class_weights}")

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
