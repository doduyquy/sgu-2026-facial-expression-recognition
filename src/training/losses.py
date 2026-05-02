import torch
import torch.nn as nn 
import torch.nn.functional as F

# Import confusion loss components
from .confusion_loss import ConfusionMatrixLoss, ContrastiveConfusionLoss, CombinedConfusionLoss


def prototype_orthogonal_loss(prototypes):
    """
    FIX 6.4: Regularize prototypes to be orthogonal (prevent collapse)
    
    Args:
        prototypes: (num_classes, motifs_per_class, num_nodes, feat_dim)
                   or (total_motifs, feat_dim)
    
    Returns:
        loss: scalar
    """
    if prototypes.dim() == 4:
        # (C, M, N, D) → (C*M, N*D) → (C*M, D) after flatten
        C, M, N, D = prototypes.shape
        p = prototypes.reshape(C * M, -1)
    else:
        p = prototypes
    
    # Normalize: (num_motifs, feat_dim)
    p_norm = F.normalize(p, dim=-1)
    
    # Similarity matrix
    sim = torch.matmul(p_norm, p_norm.t())  # (num_motifs, num_motifs)
    
    # Orthogonal penalty: off-diagonal elements should be 0
    eye = torch.eye(sim.size(0), device=sim.device)
    orth_loss = ((sim - eye) ** 2).mean()
    
    return orth_loss

class MotifConsistencyLoss(nn.Module):
    def __init__(self, num_classes=7, motifs_per_class=8, tau=0.1, simplified=False):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.tau = tau
        self.simplified = simplified  # Option to simplify (remove margin loss)

    def forward(self, scores, top_k_idx, targets, reduction='mean'):
        """
        scores: (B, num_candidates, Total_Motifs)
        top_k_idx: (B, top_k)
        targets: (B,)
        
        FIXED (6.1):
        - Vectorized mask creation (no Python loop)
        - Normalized averaging with mask.sum() (no hardcode)
        """
        B, num_cands, Total_Motifs = scores.shape
        top_k = top_k_idx.shape[1]
        
        # Get scores for selected subgraphs
        batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, top_k)
        selected_scores = scores[batch_idx, top_k_idx]  # (B, top_k, Total_Motifs)
        
        # FIX 6.1: Vectorized mask creation (no Python loop) ✓
        # targets: (B,) → (B, 1) → (B, motifs_per_class) indices
        idx = (targets.unsqueeze(1) * self.motifs_per_class + 
               torch.arange(self.motifs_per_class, device=scores.device))  # (B, motifs_per_class)
        
        mask = torch.zeros(B, Total_Motifs, device=scores.device, dtype=torch.float32)
        mask.scatter_(1, idx, 1.0)  # Vectorized assignment
        mask = mask.unsqueeze(1)  # (B, 1, Total_Motifs)
        
        # 1. Similarity to SAME class motifs (Positive) - InfoNCE style
        pos_scores = selected_scores.masked_fill(mask == 0, -1e9)
        log_sum_exp_pos = torch.logsumexp(pos_scores / self.tau, dim=-1)
        
        # 2. Similarity to ALL motifs
        log_sum_exp_all = torch.logsumexp(selected_scores / self.tau, dim=-1)
        
        # Intra-class loss per sample (InfoNCE)
        loss_intra = -(log_sum_exp_pos - log_sum_exp_all).mean(dim=1)  # (B,)
        
        if self.simplified:
            # FIX 6.6: Simplified version (no margin loss)
            total_loss = loss_intra
        else:
            # 3. Inter-class Separation - with fixed averaging ✓
            # FIX 6.2: Use mask.sum() instead of hardcoded numbers
            pos_count = mask.sum(dim=-1) + 1e-8  # (B, 1)
            neg_count = (1 - mask).sum(dim=-1) + 1e-8  # (B, 1)
            
            pos_avg = (selected_scores * mask).sum(dim=-1) / pos_count
            neg_avg = (selected_scores * (1 - mask)).sum(dim=-1) / neg_count
            
            # Contrastive margin loss per sample
            margin = 0.2
            loss_inter = F.relu(margin + neg_avg - pos_avg).mean(dim=1)  # (B,)
            
            total_loss = loss_intra + loss_inter
        
        if reduction == 'mean':
            return total_loss.mean()
        return total_loss




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

    elif loss_name == 'motif_combined':
        # Combined CrossEntropy and MotifConsistencyLoss
        alpha_weight = config['training'].get('motif_loss_weight', 0.5)
        ce_loss = nn.CrossEntropyLoss()
        motif_loss = MotifConsistencyLoss(
            num_classes=config['model'].get('num_classes', 7),
            motifs_per_class=config['model'].get('motifs_per_class', 8),
            tau=config['training'].get('motif_tau', 0.1)
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
                
                # FIX 6.4: Add prototype orthogonal regularization
                if model is not None:
                    if hasattr(model, 'compute_motif_diversity_loss'):
                        l_div = model.compute_motif_diversity_loss()
                        loss = loss + self.div_weight * l_div
                    
                    # FIX 6.5: Extract and add auxiliary losses (diversity, sparsity)
                    if hasattr(model, 'get_aux_losses'):
                        aux_losses = model.get_aux_losses()
                        if aux_losses:
                            for aux_name, aux_loss in aux_losses.items():
                                aux_weight = 0.1  # Default auxiliary weight
                                if 'diversity' in aux_name:
                                    aux_weight = 0.1
                                elif 'sparsity' in aux_name or 'entropy' in aux_name:
                                    aux_weight = 0.05
                                loss = loss + aux_weight * aux_loss
                
                return loss

        loss = CombinedMotifLoss(
            ce_loss, motif_loss, alpha_weight, 
            div_weight=config['training'].get('motif_div_weight', 0.1)
        )

    elif loss_name == 'confusion_combined':
        # Combined CrossEntropy and Confusion Loss
        confusion_weight = config['training'].get('confusion_loss_weight', 0.6)
        confusion_margin = config['training'].get('confusion_margin', 0.5)
        label_smoothing = config['training'].get('label_smoothing', 0.1)
        
        loss = CombinedConfusionLoss(
            num_classes=config['model'].get('num_classes', 7),
            confusion_weight=confusion_weight,
            confusion_margin=confusion_margin,
            label_smoothing=label_smoothing,
            class_weights=class_weights
        )

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