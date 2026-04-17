import torch
import torch.nn as nn 



def build_loss(config, class_weights=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
            class_weight=None: apply class weight or not?
    """
    loss_name = config['training'].get('loss', 'cross_entropy')

    if loss_name == 'cross_entropy':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_name == 'focal':
        # Simple focal loss implementation wrapper
        gamma = config['training'].get('focal_gamma', 2.0)
        alpha = config['training'].get('focal_alpha', None)

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
                self.reduction = reduction

            def forward(self, inputs, targets):
                ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
                p_t = torch.exp(-ce)
                loss = ((1 - p_t) ** self.gamma) * ce
                if self.alpha is not None:
                    at = self.alpha[targets]
                    loss = at * loss
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                return loss

        alpha_tensor = None
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        loss = FocalLoss(gamma=gamma, alpha=alpha_tensor)

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