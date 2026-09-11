import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fads_scn.evaluation.weighted_flip_tta import WeightedHorizontalFlipTTASweep


class TinyFlipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_modes = []

    def forward(self, images, use_tta=False):
        self.seen_modes.append(use_tta)
        labels = images[:, 0, 0, :].max(dim=-1).values.long()
        logits = torch.full((images.size(0), 7), -4.0, device=images.device)
        # The image itself encodes the label. A flipped image swaps the first
        # and last column, so the model can recognize whether it is the TTA view.
        is_flipped = images[:, 0, 0, 0] < 0
        predictions = torch.where(is_flipped, labels, torch.zeros_like(labels))
        logits.scatter_(1, predictions.unsqueeze(1), 4.0)
        return {"logits": logits}


def test_weighted_horizontal_flip_sweep_selects_on_val_then_applies_to_test():
    labels = torch.tensor([0, 1, 0, 1])
    images = torch.zeros(4, 1, 2, 2)
    images[:, 0, 0, 0] = labels
    images[:, 0, 0, -1] = -1
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    model = TinyFlipModel()

    result = WeightedHorizontalFlipTTASweep.sweep_and_apply(
        model,
        loader,
        loader,
        device="cpu",
        flip_weights=[0.0, 0.5, 1.0],
        selection_metric="accuracy",
    )

    assert result["selected_flip_weight"] == 1.0
    assert result["validation_selected"]["accuracy"] == 1.0
    assert result["test_metrics"]["accuracy"] == 1.0
