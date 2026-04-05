import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(
        self,
        class_priors,
        tau=0.2,
        weight=None,
        label_smoothing=0.0,
        temperature=1.0,
        dynamic_tau=False,
        dynamic_tau_k=None,
        dynamic_tau_min=0.0,
    ):
        super().__init__()
        eps = 1e-12
        self.register_buffer("log_priors", torch.log(class_priors.clamp_min(eps)))
        self.base_tau = float(tau)
        self.current_tau = float(tau)
        self.label_smoothing = label_smoothing
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.dynamic_tau = bool(dynamic_tau)
        self.dynamic_tau_k = float(dynamic_tau_k) if dynamic_tau_k is not None else float(tau)
        self.dynamic_tau_min = float(dynamic_tau_min)
        if weight is not None:
            self.register_buffer("class_weights", weight)
        else:
            self.class_weights = None

    def set_epoch(self, epoch):
        if self.dynamic_tau:
            epoch = max(int(epoch), 1)
            self.current_tau = max(self.dynamic_tau_min, self.dynamic_tau_k / (epoch ** 0.5))
        else:
            self.current_tau = self.base_tau

    def forward(self, logits, targets):
        adjusted_logits = logits - self.current_tau * self.log_priors.unsqueeze(0)
        adjusted_logits = adjusted_logits / self.temperature
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


def build_loss(config, class_weights=None, class_priors=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
            class_weight=None: apply class weight or not?
    """
    loss_name = config['training'].get('loss', 'cross_entropy')
    use_logit_adjustment = config['training'].get('use_logit_adjustment', False)
    logit_adjustment_tau = config['training'].get('logit_adjustment_tau', 0.2)
    logit_adjustment_temperature = config['training'].get('logit_adjustment_temperature', 1.0)
    logit_adjustment_dynamic_tau = config['training'].get('logit_adjustment_dynamic_tau', False)
    logit_adjustment_dynamic_k = config['training'].get('logit_adjustment_dynamic_k', logit_adjustment_tau)
    logit_adjustment_dynamic_tau_min = config['training'].get('logit_adjustment_dynamic_tau_min', 0.0)
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if loss_name == 'cross_entropy':
        if use_logit_adjustment:
            if class_priors is None:
                raise ValueError("class_priors is required when use_logit_adjustment=True")
            loss = LogitAdjustedCrossEntropy(
                class_priors=class_priors,
                tau=logit_adjustment_tau,
                weight=class_weights,
                label_smoothing=label_smoothing,
                temperature=logit_adjustment_temperature,
                dynamic_tau=logit_adjustment_dynamic_tau,
                dynamic_tau_k=logit_adjustment_dynamic_k,
                dynamic_tau_min=logit_adjustment_dynamic_tau_min,
            )
        elif class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
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
