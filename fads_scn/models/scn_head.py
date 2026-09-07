import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceClassifier(nn.Module):
    """
    Large Margin Cosine Loss (CosFace / AM-Softmax) Classifier.
    Normalizes features and class weights to unit hypersphere,
    applying additive angular margin m to target classes:
        cos(theta_y) - m
    Scaled by temperature/scale factor s.
    
    Effect: Enforces high inter-class variance and compact intra-class clusters,
    preventing confusion between easily confused negative emotions (Sad, Fear, Neutral, Angry).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        scale: float = 30.0,
        margin: float = 0.20,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.GELU(),
        )
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features // 2))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor = None,
        targets_b: torch.Tensor = None,
        lam: float = 1.0,
    ):
        # 1. Feature projection and normalization
        x = self.drop(features)
        x = self.proj(x)
        features_norm = F.normalize(x, p=2, dim=-1)
        weight_norm = F.normalize(self.weight, p=2, dim=-1)

        # 2. Cosine similarity: [B, num_classes]
        cosine = F.linear(features_norm, weight_norm)

        # 3. Additive margin during training
        mixup_active = targets_b is not None and lam < 1.0
        if self.training and targets is not None and not mixup_active:
            one_hot_a = F.one_hot(targets, self.num_classes).float()
            margin_mask = one_hot_a

            margin_cosine = cosine - (margin_mask * self.margin)
            margin_cosine = torch.clamp(margin_cosine, -1.0, 1.0)
            logits = self.scale * margin_cosine
        else:
            logits = self.scale * cosine

        return logits


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
    Self-Cure Network Head (SCN, CVPR 2020 style) with CosFace Angular Margin support.
    Simultaneously produces:
    1. Emotion classification logits z in R^7 (CosFace angular margin or Linear)
    2. Sample confidence / importance weight alpha in [0.10, 1.00]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.25,
        classifier_type: str = "cosface",
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.20,
        init_confidence_bias: float = 1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classifier_type = classifier_type

        if classifier_type == "cosface":
            self.classifier = CosFaceClassifier(
                in_features=embed_dim,
                num_classes=num_classes,
                scale=cosface_scale,
                margin=cosface_margin,
                dropout=dropout,
            )
        else:
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
            targets: ground truth class indices [B] (optional, used for CosFace margin in training)
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
