import torch
import torch.nn as nn


class SCNHead(nn.Module):
    """
    Self-Cure Network Head (SCN, CVPR 2020 style).
    Simultaneously produces:
    1. Emotion classification logits z in R^7
    2. Sample confidence / importance weight alpha in (0, 1)
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

        # Emotion classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
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

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [B, embed_dim]
        Returns:
            logits: [B, num_classes]
            alpha: [B, 1] sample confidence weights in range (0, 1)
        """
        logits = self.classifier(features)
        alpha = self.importance_gate(features)
        return logits, alpha
