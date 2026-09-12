import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.losses.scn_loss import SCNLoss
from fads_scn.training.trainer import AttentiveSCNTrainer


class CountingToyFER(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 7)
        self.alpha_head = nn.Linear(4, 1)
        self.forward_calls = 0

    def forward(self, images, targets=None, targets_b=None, lam=1.0, use_tta=False):
        self.forward_calls += 1
        features = images.view(images.size(0), -1)
        return {
            "logits": self.classifier(features),
            "alpha": 0.1 + 0.9 * torch.sigmoid(self.alpha_head(features)),
            "diversity_loss": torch.zeros((), device=images.device),
            "sparsity_loss": torch.zeros((), device=images.device),
        }


def test_mixaugment_trains_real_and_virtual_views_in_one_update(tmp_path):
    images = torch.randn(6, 1, 2, 2)
    labels = torch.arange(6) % 7
    indices = torch.arange(6)
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=6, shuffle=False)
    model = CountingToyFER()
    trainer = AttentiveSCNTrainer(
        model=model,
        criterion=SCNLoss(num_classes=7),
        train_loader=loader,
        val_loader=loader,
        cfg={
            "data": {
                "use_mixup": True,
                "mixup_alpha": 0.2,
                "mixup_prob": 1.0,
                "mix_mode": "mixaugment",
                "mixaugment_real_weight": 0.5,
            },
            "training": {"epochs": 1, "use_ema": False, "output_dir": str(tmp_path)},
            "scn": {"rank_warmup_epochs": 0, "enable_relabel": False},
        },
        device="cpu",
    )

    loss, accuracy, relabelled = trainer.train_one_epoch(epoch=0)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= accuracy <= 1.0
    assert relabelled == 0
    assert model.forward_calls == 2, "MixAugment must forward real and virtual views"
