import torch
import torch.nn as nn 
import torch.nn.functional as F

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=7, label_smoothing=0.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, pred, labels):
        # pred: (B, num_classes), labels: (B,)
        # Standard Cross Entropy (ce)
        ce = F.cross_entropy(pred, labels, weight=self.weight, label_smoothing=self.label_smoothing)

        # Reverse Cross Entropy (rce)
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        
        # One-hot labels
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
        
        rce_per_sample = -1.0 * torch.sum(pred_softmax * torch.log(one_hot), dim=1)
        if self.weight is not None:
            rce = (rce_per_sample * self.weight[labels]).sum() / self.weight[labels].sum()
        else:
            rce = rce_per_sample.mean()

        return self.alpha * ce + self.beta * rce


class MotifConsistencyLoss(nn.Module):
    def __init__(self, num_classes=7, motifs_per_class=8, tau=0.1, margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.tau = tau
        self.margin = margin

    def forward(self, scores, top_k_idx, targets, reduction='mean'):
        """
        scores: (B, num_candidates, Total_Motifs)
        top_k_idx: (B, top_k)
        targets: (B,)   
        """
        B, num_cands, Total_Motifs = scores.shape
        top_k = top_k_idx.shape[1]
        
        # Get scores for selected subgraphs
        batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, top_k)
        selected_scores = scores[batch_idx, top_k_idx] # (B, top_k, Total_Motifs)
        
        # Create mask for correct class motifs
        mask = torch.zeros(B, Total_Motifs, device=scores.device)
        for i in range(B):
            c = int(targets[i].item())
            mask[i, c*self.motifs_per_class : (c+1)*self.motifs_per_class] = 1.0
        mask = mask.unsqueeze(1) # (B, 1, Total_Motifs)
        
        # 1. Similarity to SAME class motifs (Positive)
        pos_scores = selected_scores.masked_fill(mask == 0, -1e9)
        log_sum_exp_pos = torch.logsumexp(pos_scores / self.tau, dim=-1)
        
        # 2. Similarity to ALL motifs
        log_sum_exp_all = torch.logsumexp(selected_scores / self.tau, dim=-1)
        
        # Intra-class loss per sample (InfoNCE style)
        # Average over top-k subgraphs
        loss_intra = -(log_sum_exp_pos - log_sum_exp_all).mean(dim=1) # (B,)
        
        # 3. Inter-class Separation (Contrastive/Triplet style)
        # We want Avg(pos_scores) > Avg(neg_scores) + margin
        pos_avg = (selected_scores * mask).sum(dim=-1) / self.motifs_per_class
        neg_avg = (selected_scores * (1 - mask)).sum(dim=-1) / (Total_Motifs - self.motifs_per_class)
        
        # Contrastive margin loss per sample
        loss_inter = F.relu(self.margin + neg_avg - pos_avg).mean(dim=1) # (B,)
        
        total_loss = loss_intra + loss_inter
        
        if reduction == 'mean':
            return total_loss.mean()
        return total_loss


def build_loss(config, class_weights=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
            class_weights=None: apply class weight or not?
    """
    loss_name = config['training'].get('loss', 'cross_entropy')

    if loss_name == 'cross_entropy':
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        if class_weights is not None:
            loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_name == 'sce':
        sce_alpha = config['training'].get('sce_alpha', 1.0)
        sce_beta = config['training'].get('sce_beta', 1.0)
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        num_classes = config['model'].get('num_classes', 7)
        loss = SymmetricCrossEntropy(
            alpha=sce_alpha, beta=sce_beta, 
            num_classes=num_classes, label_smoothing=label_smoothing,
            weight=class_weights
        )

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

    elif loss_name == 'motif_combined':
        # Combined (CE or SCE) and MotifConsistencyLoss
        alpha_weight = config['training'].get('motif_loss_weight', 0.5)
        
        base_loss_name = config['training'].get('base_loss', 'cross_entropy')
        use_sce_base = config['training'].get('use_sce_base', False)
        if base_loss_name == 'sce' or use_sce_base:
            sce_alpha = config['training'].get('sce_alpha', 1.0)
            sce_beta = config['training'].get('sce_beta', 1.0)
            label_smoothing = config['training'].get('label_smoothing', 0.0)
            num_classes = config['model'].get('num_classes', 7)
            ce_loss = SymmetricCrossEntropy(
                alpha=sce_alpha, beta=sce_beta, 
                num_classes=num_classes, label_smoothing=label_smoothing,
                weight=class_weights
            )
        else:
            label_smoothing = config['training'].get('label_smoothing', 0.0)
            if class_weights is not None:
                ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            else:
                ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
        motif_loss = MotifConsistencyLoss(
            num_classes=config['model'].get('num_classes', 7),
            motifs_per_class=config['model'].get('motifs_per_class', 8),
            tau=config['training'].get('motif_tau', 0.1),
            margin=config['training'].get('motif_margin', 0.5)
        )
        
        class CombinedMotifLoss(nn.Module):
            def __init__(self, ce, motif, weight, div_weight=0.1):
                super().__init__()
                self.ce = ce
                self.motif = motif
                self.weight = weight
                self.div_weight = div_weight
            
            def forward(self, logits, targets, scores=None, top_k_idx=None, model=None):
                l_ce = self.ce(logits, targets)
                
                if scores is not None and top_k_idx is not None:
                    l_motif = self.motif(scores, top_k_idx, targets)
                    loss = l_ce + self.weight * l_motif
                else:
                    loss = l_ce
                
                if model is not None and hasattr(model, 'compute_motif_diversity_loss'):
                    l_div = model.compute_motif_diversity_loss()
                    loss = loss + self.div_weight * l_div
                return loss

        loss = CombinedMotifLoss(
            ce_loss, motif_loss, alpha_weight, 
            div_weight=config['training'].get('motif_div_weight', 0.1)
        )

    else: 
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    return loss


if __name__ == "__main__":
    # Test block
    config_default = {'training': {}}
    loss_fn = build_loss(config_default)
    print(f"Test 1 (Default): {type(loss_fn)}") 
    # Expect: <class 'torch.nn.modules.loss.CrossEntropyLoss'>

    config_explicit = {'training': {'loss': 'cross_entropy'}}
    loss_fn = build_loss(config_explicit)
    print(f"Test 2 (Explicit): {type(loss_fn)}")
    # Expect: <class 'torch.nn.modules.loss.CrossEntropyLoss'>

    config_sce = {'training': {'loss': 'sce', 'sce_alpha': 1.0, 'sce_beta': 1.0}, 'model': {'num_classes': 7}}
    loss_fn = build_loss(config_sce)
    print(f"Test 3 (SCE standalone): {type(loss_fn)}")
    # Expect: <class '__main__.SymmetricCrossEntropy'>
    
    config_motif_sce = {
        'training': {
            'loss': 'motif_combined', 
            'base_loss': 'sce', 
            'motif_loss_weight': 0.5,
            'motif_margin': 0.5
        },
        'model': {
            'num_classes': 7,
            'motifs_per_class': 8
        }
    }
    loss_fn = build_loss(config_motif_sce)
    print(f"Test 4 (Motif combined with SCE): {type(loss_fn)}")
    # Expect: CombinedMotifLoss
    
    dummy_logits = torch.randn(4, 7)
    dummy_targets = torch.tensor([0, 1, 2, 3])
    loss_val = loss_fn(dummy_logits, dummy_targets)
    print(f"Test 4 loss val: {loss_val.item():.4f}")