
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# CBAM is now at the same level (src/models/CBAM.py)
from .CBAM import CBAM


class SemanticBackbone(nn.Module):
    """ResNet18 backbone with high spatial resolution output."""

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if use_pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)

        # Keep high resolution by removing early downsampling.
        resnet.conv1.stride = (1, 1)
        resnet.maxpool = nn.Identity()

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 48 -> 48
        self.layer2 = resnet.layer2  # 48 -> 24
        self.layer3 = resnet.layer3  # 24 -> 12
        self.layer4 = resnet.layer4  # 12 -> 6

        if feature_dim == 256:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3)
            self.out_channels = 256
            self.use_layer4 = False
            # Free layer4 — not used in forward, but would waste ~8MB GPU memory
            # if kept as a registered submodule with its pretrained weights.
            del self.layer4
        elif feature_dim == 512:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
            self.out_channels = 512
            self.use_layer4 = True
        else:
            raise ValueError("feature_dim must be 256 or 512")

        self.spatial_attention = CBAM(channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.output_layer(x)
        if self.use_layer4:
            x = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)

        # Apply Spatial Attention (CBAM) to filter out background noise
        # x = self.spatial_attention(x)
        return x
