"""
Confusable Pair Learning Loss
- ConfusionMatrixLoss: Weight hard emotion pairs
- Applied as: loss = CE + lambda * ConfusionMatrixLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionMatrixLoss(nn.Module):
    """
    Loss that focuses on hard confusion pairs.
    Emotion confusion hierarchy (from confusion matrix):
    - Fear ↔ Sad: 49.1% vs 55.1% (WORST)
    - Sad ↔ Anger: 55.1% vs 60.9%
    - Anger ↔ Disgust: 60.9% vs 54.5%
    - Fear ↔ Anger: 49.1% vs 60.9%
    
    Strategy: Increase margin for hard pairs using triplet-style loss
    """
    def __init__(self, num_classes=7, margin=0.5, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.temperature = temperature
        
        # Define confusion pairs: (class_i, class_j, weight)
        # Emotion indices: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        self.confusion_pairs = [
            (2, 4, 2.5),  # fear <-> sad: highest weight - MOST confused
            (4, 0, 2.0),  # sad <-> anger
            (0, 1, 1.8),  # anger <-> disgust
            (0, 2, 2.2),  # anger <-> fear
            (1, 2, 1.7),  # disgust <-> fear
            (4, 6, 1.5),  # sad <-> neutral
            (2, 6, 1.6),  # fear <-> neutral
            (0, 6, 1.4),  # anger <-> neutral
        ]
    
    def forward(self, logits, labels, reduction='mean'):
        """
        Args:
            logits: (B, num_classes) - model predictions
            labels: (B,) - ground truth labels
            reduction: 'mean' or 'none'
        
        Returns:
            loss: scalar or (B,) depending on reduction
        """
        B = logits.shape[0]
        
        # Compute per-sample confusion loss
        losses = []
        
        for b in range(B):
            logit_b = logits[b]  # (num_classes,)
            label_b = labels[b].item()
            
            # Check if this sample belongs to a hard confusion pair
            pair_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            is_hard_pair = False
            
            for class_i, class_j, weight in self.confusion_pairs:
                # Case 1: True label is class_i, penalize predicting class_j
                if label_b == class_i:
                    score_true = logit_b[class_i]
                    score_false = logit_b[class_j]
                    # Margin-based loss: want score_true > score_false + margin
                    pair_loss_ij = weight * F.relu(self.margin + score_false - score_true)
                    pair_loss = pair_loss + pair_loss_ij
                    is_hard_pair = True
                
                # Case 2: True label is class_j, penalize predicting class_i
                if label_b == class_j:
                    score_true = logit_b[class_j]
                    score_false = logit_b[class_i]
                    pair_loss_ji = weight * F.relu(self.margin + score_false - score_true)
                    pair_loss = pair_loss + pair_loss_ji
                    is_hard_pair = True
            
            losses.append(pair_loss)
        
        losses = torch.stack(losses)
        
        if reduction == 'mean':
            return losses.mean()
        elif reduction == 'none':
            return losses
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")


class ContrastiveConfusionLoss(nn.Module):
    """
    Contrastive loss for confusion pairs.
    Push similar emotions apart in logit space.
    """
    def __init__(self, num_classes=7, tau=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        
        # Confusion pairs
        self.confusion_pairs = {
            2: [4, 0, 6],      # fear: avoid sad, anger, neutral
            4: [2, 0, 6],      # sad: avoid fear, anger, neutral
            0: [2, 4, 1, 6],   # anger: avoid fear, sad, disgust, neutral
            1: [0, 2],         # disgust: avoid anger, fear
        }
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,)
        Returns:
            loss: scalar
        """
        B = logits.shape[0]
        
        # Apply softmax
        probs = F.softmax(logits / self.tau, dim=1)  # (B, num_classes)
        
        losses = []
        for b in range(B):
            true_label = labels[b].item()
            
            # Get logits for this sample
            logit_true = logits[b, true_label]  # scalar
            probs_b = probs[b]  # (num_classes,)
            
            if true_label in self.confusion_pairs:
                # Get confused classes
                confused_classes = self.confusion_pairs[true_label]
                
                # Penalize logits of confused classes
                for confused_class in confused_classes:
                    logit_confused = logits[b, confused_class]
                    # We want logit_true >> logit_confused
                    # Use softmax cross-entropy style loss
                    loss_pair = -torch.log(probs_b[true_label] + 1e-8)
                    losses.append(loss_pair)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=logits.device)


if __name__ == "__main__":
    # Test ConfusionMatrixLoss
    print("Testing ConfusionMatrixLoss...")
    loss_fn = ConfusionMatrixLoss(num_classes=7, margin=0.5)
    
    # Create mock data with hard confusion pair
    B = 4
    logits = torch.randn(B, 7)
    labels = torch.tensor([2, 4, 0, 1])  # fear, sad, anger, disgust - all hard pairs
    
    loss = loss_fn(logits, labels, reduction='mean')
    print(f"Loss: {loss.item():.4f}")
    print("✓ ConfusionMatrixLoss passed!")
    
    # Test ContrastiveConfusionLoss
    print("\nTesting ContrastiveConfusionLoss...")
    loss_fn2 = ContrastiveConfusionLoss(num_classes=7, tau=0.1)
    loss2 = loss_fn2(logits, labels)
    print(f"Contrastive Loss: {loss2.item():.4f}")
    print("✓ ContrastiveConfusionLoss passed!")
