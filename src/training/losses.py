import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss.
    Focuses only on the top K% hardest samples in each batch.
    """
    def __init__(self, base_loss_fn, ratio=0.7):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.ratio = ratio

    def forward(self, logits, targets):
        # Compute per-sample loss (no reduction)
        loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Select top K% samples
        num_samples = loss.size(0)
        num_hard = max(int(num_samples * self.ratio), 1)
        
        hard_loss, _ = torch.topk(loss, num_hard)
        return hard_loss.mean()

class MotifConsistencyLoss(nn.Module):
    """
    Ensures motifs for the same class have consistent activations.
    """
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, scores, top_k_idx, targets):
        # scores: (B, num_cands, Total_Motifs)
        B, num_cands, Total_Motifs = scores.shape
        num_classes = Total_Motifs // (Total_Motifs // 7) # Basic estimate
        motifs_per_class = Total_Motifs // 7
        
        # Create mask for ground truth class motifs
        # targets: (B,)
        mask = torch.zeros(B, Total_Motifs, device=scores.device)
        for i in range(B):
            c = targets[i]
            mask[i, c*motifs_per_class : (c+1)*motifs_per_class] = 1.0
            
        # loss = -log( sum(exp(pos_scores)) / sum(exp(all_scores)) )
        log_sum_exp_all = torch.logsumexp(scores / self.tau, dim=-1) # (B, num_cands)
        
        pos_scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        log_sum_exp_pos = torch.logsumexp(pos_scores / self.tau, dim=-1) # (B, num_cands)
        
        loss = -(log_sum_exp_pos - log_sum_exp_all).mean()
        return loss

def build_loss(config):
    loss_name = config['training'].get('loss', 'cross_entropy')
    use_ohem = config['training'].get('use_ohem', True)
    
    # Base CrossEntropy
    base_ce = nn.CrossEntropyLoss()
    ce_loss = OHEMLoss(base_ce, ratio=0.7) if use_ohem else base_ce
    
    if loss_name == 'cross_entropy':
        return ce_loss
        
    elif loss_name == 'motif_combined':
        motif_loss = MotifConsistencyLoss()
        alpha_weight = config['training'].get('motif_loss_weight', 0.5)
        div_weight = config['training'].get('motif_div_weight', 0.1)
        
        class CombinedMotifLoss(nn.Module):
            def __init__(self, ce, motif, weight, div_w):
                super().__init__()
                self.ce = ce
                self.motif = motif
                self.weight = weight
                self.div_weight = div_w
            
            def forward(self, logits, targets, scores=None, top_k_idx=None, model=None):
                l_ce = self.ce(logits, targets)
                
                loss = l_ce
                if scores is not None:
                    l_motif = self.motif(scores, None, targets)
                    loss = loss + self.weight * l_motif
                
                if model is not None:
                    aux = model.get_aux_losses()
                    if "motif_diversity" in aux:
                        loss = loss + self.div_weight * aux["motif_diversity"]
                    if "offset_reg" in aux:
                        loss = loss + 0.05 * aux["offset_reg"]
                return loss

        return CombinedMotifLoss(ce_loss, motif_loss, alpha_weight, div_weight)
    
    else: 
        raise ValueError(f"Unsupported loss: {loss_name}")

if __name__ == "__main__":
    config = {'training': {'use_ohem': True}}
    loss_fn = build_loss(config)
    print("Loss built successfully")