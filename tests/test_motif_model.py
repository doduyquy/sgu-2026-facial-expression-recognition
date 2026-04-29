import sys
sys.path.insert(0, '.')

import torch
from src.models.motif_graph_fer import MotifGraphModel

cfg = {
    'feat_dim': 64, 'num_classes': 7, 'k_neighbors': 8,
    'num_motifs': 16, 'gat_heads': 4, 'motif_tau': 0.1,
    'dropout': 0.3, 'motif_div_weight': 0.2,
}
device = torch.device('cpu')
model  = MotifGraphModel(cfg).to(device)

dummy  = torch.randn(4, 1, 48, 48)
logits = model(dummy)
print("Output shape    :", logits.shape)           # expect (4, 7)

aux = model.get_aux_losses()
div = aux['motif_diversity'].item()
print("motif_diversity :", round(div, 4))

model.freeze_for_phase1()
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print("Phase-1 trainable:", n_train, "/", n_total)

model.unfreeze_all()
n_train2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Phase-2 trainable:", n_train2, "/", n_total)

assert logits.shape == (4, 7), "Shape mismatch!"
assert n_train < n_total,      "Phase-1 freeze failed!"
assert n_train2 == n_total,    "Phase-2 unfreeze failed!"
print("ALL CHECKS PASSED")
