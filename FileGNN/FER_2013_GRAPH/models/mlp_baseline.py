import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Baseline MLP cho graph-level vector.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims=(64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.classifier(h)
        return logits