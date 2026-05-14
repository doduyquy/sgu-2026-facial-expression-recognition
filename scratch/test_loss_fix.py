import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from src.training.losses import MotifConsistencyLoss

def test_motif_consistency_loss():
    # Setup
    num_classes = 7
    motifs_per_class = 8
    tau = 0.1
    loss_fn = MotifConsistencyLoss(num_classes, motifs_per_class, tau)
    
    B = 10
    Total_Motifs = num_classes * motifs_per_class
    top_k = 4
    
    # Test case 1: Matching sizes
    scores = torch.randn(B, 20, Total_Motifs)
    top_k_idx = torch.randint(0, 20, (B, top_k))
    targets = torch.randint(0, num_classes, (B,))
    
    print("Testing matching sizes...")
    loss = loss_fn(scores, top_k_idx, targets)
    print(f"Loss: {loss.item()}")
    
    # Test case 2: TenCrop mismatch (targets=10, scores=100)
    B_expanded = 100
    scores_expanded = torch.randn(B_expanded, 20, Total_Motifs)
    top_k_idx_expanded = torch.randint(0, 20, (B_expanded, top_k))
    targets_small = torch.randint(0, num_classes, (B,))
    
    print("\nTesting TenCrop mismatch (targets size 10, scores size 100)...")
    loss_expanded = loss_fn(scores_expanded, top_k_idx_expanded, targets_small)
    print(f"Loss Expanded: {loss_expanded.item()}")
    
    print("\nTests completed successfully!")

if __name__ == "__main__":
    test_motif_consistency_loss()
