import sys
sys.path.insert(0, '.')

import torch
from src.models.motif_graph_fer import MotifGraphModel

cfg = {
    'feat_dim': 64,
    'num_classes': 7,
    'motifs_per_class': 16,
    'top_k': 4,
}
device = torch.device('cpu')
model = MotifGraphModel(cfg).to(device)

# Test 4D forward pass without targets (inference mode)
dummy_img = torch.randn(4, 1, 48, 48)
logits_inf = model(dummy_img)
print("Inference logits shape:", logits_inf.shape)
assert logits_inf.shape == (4, 7), "Inference shape mismatch!"

# Test 4D forward pass with targets (training mode)
targets = torch.tensor([0, 1, 2, 3])
logits_train = model(dummy_img, targets=targets)
print("Training logits shape :", logits_train.shape)
assert logits_train.shape == (4, 7), "Training shape mismatch!"

# Test auxiliary losses collection
aux = model.get_aux_losses()
print("Auxiliary Losses:")
for name, loss in aux.items():
    print(f"  - {name}: {loss.item():.4f}" if isinstance(loss, torch.Tensor) else f"  - {name}: {loss}")

assert 'motif_diversity' in aux, "motif_diversity loss missing!"
assert 'attn_entropy' in aux, "attn_entropy loss missing!"
assert 'offset_reg' in aux, "offset_reg loss missing!"
assert 'au_contrastive' in aux, "au_contrastive loss missing!"

print("ALL MODEL SANITY CHECKS PASSED!")
