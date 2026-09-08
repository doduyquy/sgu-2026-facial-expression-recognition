import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 7, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, features, targets=None, targets_b=None, lam=1.0):
        return self.net(features)


class SCNHead(nn.Module):
    """
    Self-Cure Network Head (SCN, CVPR 2020 style).
    Simultaneously produces:
    1. Emotion classification logits z in R^7
    2. Sample confidence / importance weight alpha in [0.10, 1.00]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.25,
        init_confidence_bias: float = 1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = LinearClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Self-Cure Importance Weight gate alpha in (0, 1)
        # alpha_i predicts the likelihood that sample i has a clean, reliable label.
        self.importance_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Initialize gate bias positively so model begins by trusting samples (sigmoid(1.5) ~ 0.82)
        # and selectively learns to suppress noisy labels as training proceeds.
        with torch.no_grad():
            self.importance_gate[2].bias.fill_(init_confidence_bias)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor = None,
        targets_b: torch.Tensor = None,
        lam: float = 1.0,
    ):
        """
        Args:
            features: [B, embed_dim]
            targets: ground truth class indices [B] (optional, accepted for API compatibility)
            targets_b: second targets for Mixup [B] (optional)
            lam: Mixup ratio in [0, 1]
        Returns:
            logits: [B, num_classes]
            alpha: [B, 1] sample confidence weights in safe range [0.10, 1.00]
        """
        logits = self.classifier(features, targets=targets, targets_b=targets_b, lam=lam)
        raw_alpha = self.importance_gate(features)
        # Bounded in [0.10, 1.00] to strictly prevent gradient vanishing or mode collapse
        alpha = 0.10 + 0.90 * raw_alpha
        return logits, alpha
