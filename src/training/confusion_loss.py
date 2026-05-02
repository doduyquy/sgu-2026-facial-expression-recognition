"""
Confusion Matrix Loss: Margin-based loss for hard emotion pairs
Purpose: Penalize confusion between similar emotions (Fear↔Sad, Sad↔Anger, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionMatrixLoss(nn.Module):
    """
    Margin-based loss for hard-to-distinguish emotion pairs.
    
    Encourages: logit_true > logit_false + margin
    With higher weight for known confusion pairs.
    """
    
    def __init__(self, num_classes=7, margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        
        # Define confusion pairs and their weights
        # (class_i, class_j, weight) -> penalize confusion between i and j
        self.confusion_pairs = [
            (3, 5, 2.5),   # Fear (3) ↔ Sad (5): weight 2.5×
            (5, 0, 2.0),   # Sad (5) ↔ Angry (0): weight 2.0×
            (0, 1, 1.8),   # Angry (0) ↔ Disgust (1): weight 1.8×
            (1, 2, 1.6),   # Disgust (1) ↔ Fear (2): weight 1.6×
            (4, 6, 1.2),   # Neutral (4) ↔ Surprise (6): weight 1.2×
        ]
        # Classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sad, 6=Surprise
    
    def forward(self, logits, labels, reduction='mean'):
        """
        Args:
            logits: (B, num_classes) - raw model outputs
            labels: (B,) - ground truth labels
            reduction: 'mean' or 'none'
        
        Returns:
            loss: scalar or (B,) tensor depending on reduction
        """
        B = logits.shape[0]
        losses = []
        
        for b in range(B):
            logit_b = logits[b]  # (num_classes,)
            label_b = labels[b].item()  # scalar
            
            # Initialize as Tensor, NOT float (critical!)
            pair_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            
            # Apply margin loss for each confusion pair
            for class_i, class_j, weight in self.confusion_pairs:
                # When true label is class_i, penalize high score for class_j
                if label_b == class_i:
                    score_true = logit_b[class_i]
                    score_false = logit_b[class_j]
                    # Loss: max(0, margin + score_false - score_true)
                    pair_loss_ij = weight * F.relu(self.margin + score_false - score_true)
                    # IMPORTANT: Use explicit assignment (=) not in-place (+=)
                    pair_loss = pair_loss + pair_loss_ij
                
                # When true label is class_j, penalize high score for class_i
                if label_b == class_j:
                    score_true = logit_b[class_j]
                    score_false = logit_b[class_i]
                    pair_loss_ji = weight * F.relu(self.margin + score_false - score_true)
                    pair_loss = pair_loss + pair_loss_ji
            
            losses.append(pair_loss)
        
        # Stack losses - all must be Tensors now (no type mismatch)
        losses = torch.stack(losses)
        
        if reduction == 'mean':
            return losses.mean()
        elif reduction == 'sum':
            return losses.sum()
        else:
            return losses


class ContrastiveConfusionLoss(nn.Module):
    """
    Alternative: Contrastive learning approach for confusion pairs.
    Pulls together similar classes while pushing apart dissimilar ones.
    """
    
    def __init__(self, num_classes=7, margin=0.5, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.temperature = temperature
        
        # Define which classes should be similar/dissimilar
        self.confusion_pairs = [
            (3, 5, 2.5),   # Fear ↔ Sad
            (5, 0, 2.0),   # Sad ↔ Angry
            (0, 1, 1.8),   # Angry ↔ Disgust
        ]
    
    def forward(self, logits, labels, reduction='mean'):
        """Contrastive loss for confusion pairs"""
        B = logits.shape[0]
        losses = []
        
        for b in range(B):
            logit_b = logits[b]
            label_b = labels[b].item()
            
            loss_b = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            
            for class_i, class_j, weight in self.confusion_pairs:
                if label_b == class_i or label_b == class_j:
                    # Get true class logit
                    true_logit = logit_b[label_b]
                    
                    # Get confusion class logit
                    confusion_class = class_j if label_b == class_i else class_i
                    confusion_logit = logit_b[confusion_class]
                    
                    # Contrastive: penalize small gap
                    gap = true_logit - confusion_logit
                    contrastive_loss = weight * F.relu(self.margin - gap)
                    loss_b = loss_b + contrastive_loss
            
            losses.append(loss_b)
        
        losses = torch.stack(losses)
        
        if reduction == 'mean':
            return losses.mean()
        else:
            return losses


class CombinedConfusionLoss(nn.Module):
    """
    Combined loss: CrossEntropy + Confusion Loss
    Used as unified loss function in training
    """
    
    def __init__(self, num_classes=7, confusion_weight=0.6, confusion_margin=0.5, 
                 label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.confusion_weight = confusion_weight
        self.ce_weight = 1.0 - confusion_weight
        
        # Cross-entropy component
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # Confusion component
        self.confusion_loss = ConfusionMatrixLoss(
            num_classes=num_classes,
            margin=confusion_margin
        )
    
    def forward(self, logits, labels, reduction='mean'):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,)
            reduction: 'mean' or 'none'
        
        Returns:
            Combined loss = CE_weight * CE + Confusion_weight * Confusion
        """
        # Compute both losses
        ce = self.ce_loss(logits, labels)
        confusion = self.confusion_loss(logits, labels, reduction='mean')
        
        # Combine
        total_loss = self.ce_weight * ce + self.confusion_weight * confusion
        
        return total_loss
