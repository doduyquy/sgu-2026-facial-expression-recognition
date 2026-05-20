import torch

B = 4
num_cands = 6
Total_Motifs = 112
top_k = 3

scores = torch.randn(B, num_cands, Total_Motifs)
top_k_idx = torch.randint(0, num_cands, (B, top_k))

batch_idx = torch.arange(B).unsqueeze(1).expand(-1, top_k)
selected_scores = scores[batch_idx, top_k_idx]

print("scores shape:", scores.shape)
print("top_k_idx shape:", top_k_idx.shape)
print("batch_idx shape:", batch_idx.shape)
print("selected_scores shape:", selected_scores.shape)
