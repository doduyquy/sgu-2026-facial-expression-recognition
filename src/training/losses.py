import torch.nn as nn 
from .loss import FocalLoss

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
    loss_name = config['training'].get('loss', 'cross_entropy')
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    loss_name = config['training'].get('loss', 'cross_entropy').lower()

    # =========================
    # Cross Entropy Loss
    # =========================
    if loss_name == 'cross_entropy':
        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
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
