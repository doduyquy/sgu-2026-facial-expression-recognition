import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn):
        # sp: (B, K) positive similarity
        # sn: (B, L) negative similarity
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        return loss.mean()

class ClassBalancedSupConLoss(nn.Module):
    """
    Supervised Contrastive Learning with Class Balancing to mitigate noisy labels.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        # features: (B, D)
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask for positive samples
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # remove self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 1).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class balancing weight (1 / count of positive samples)
        pos_counts = mask.sum(1)
        # Avoid division by zero
        pos_counts = torch.max(pos_counts, torch.ones_like(pos_counts))
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

class EMOGNPCombinedLoss(nn.Module):
    def __init__(self, config, class_weights=None):
        super().__init__()
        training_cfg = config.get('training', {})
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.ce_weight = training_cfg.get('ce_weight', 1.0)
        
        # Circle Loss
        self.use_circle = training_cfg.get('circle_weight', 0.0) > 0
        if self.use_circle:
            self.circle = CircleLoss(
                m=training_cfg.get('circle_margin', 0.25),
                gamma=training_cfg.get('circle_gamma', 256)
            )
            self.circle_weight = training_cfg.get('circle_weight', 0.3)
            
        # SupCon Loss
        self.use_supcon = training_cfg.get('supcon_weight', 0.0) > 0
        if self.use_supcon:
            self.supcon = ClassBalancedSupConLoss()
            self.supcon_weight = training_cfg.get('supcon_weight', 0.5)
            
        # Prototype Repulsion
        self.repulsion_weight = training_cfg.get('repulsion_weight', 0.1)

    def forward(self, logits, targets, model=None):
        loss = self.ce_weight * self.ce(logits, targets)
        
        if model is not None:
            # Prototype repulsion loss
            aux_losses = model.get_aux_losses()
            if 'repulsion_loss' in aux_losses:
                loss = loss + self.repulsion_weight * aux_losses['repulsion_loss']
                
            # Class-balanced SupCon Loss
            if self.use_supcon and hasattr(model, '_latest_features'):
                features = model._latest_features
                l_supcon = self.supcon(features, targets)
                loss = loss + self.supcon_weight * l_supcon
                
        return loss

def build_loss(config, class_weights=None):
    loss_name = config['training'].get('loss', 'cross_entropy')

    if loss_name == 'cross_entropy':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    elif loss_name == 'emo_gnp_combined':
        loss = EMOGNPCombinedLoss(config, class_weights)

    else:
        # Fallback to simple cross entropy
        loss = nn.CrossEntropyLoss(weight=class_weights)

    return loss