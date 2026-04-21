import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # inputs: [B, C], targets: [B]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
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
        loss = FocalLoss(gamma=2.0, weight=class_weights, label_smoothing=label_smoothing)
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
