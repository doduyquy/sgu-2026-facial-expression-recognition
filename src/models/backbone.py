
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# CBAM is no longer used, replaced by SpatialResidualMasking
# from .CBAM import CBAM

class SpatialResidualMasking(nn.Module):
    """
    Lightweight Spatial Residual Masking Block.
    Generates a spatial attention mask and applies it via a residual connection.
    This suppresses background noise and highlights micro-expressions.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Bottleneck to reduce parameters
        reduced_channels = max(in_channels // 4, 16)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mask = self.mask_generator(x)
        # Residual masking: x' = x + x * M
        return x + x * mask


class SemanticBackbone(nn.Module):
    """ResNet50 backbone with high spatial resolution output."""

    def __init__(self, feature_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if use_pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)

        # Keep high resolution by removing early downsampling.
        resnet.conv1.stride = (1, 1)
        resnet.maxpool = nn.Identity()

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1  # 48 -> 48 (256 channels)
        self.layer2 = resnet.layer2  # 48 -> 24 (512 channels)
        self.layer3 = resnet.layer3  # 24 -> 12 (1024 channels)
        self.layer4 = resnet.layer4  # 12 -> 6 (2048 channels)

        if feature_dim == 256:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3)
            self.out_channels = 256
            self.use_layer4 = False
            self.proj = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            del self.layer4
        elif feature_dim == 512:
            self.output_layer = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
            self.out_channels = 512
            self.use_layer4 = True
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError("feature_dim must be 256 or 512")

        self.spatial_attention = SpatialResidualMasking(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.output_layer(x)
        if self.use_layer4:
            x = F.interpolate(x, size=(12, 12), mode="bilinear", align_corners=False)
            
        x = self.proj(x)

        # Apply Spatial Residual Masking to amplify micro-expressions without losing global features
        x = self.spatial_attention(x)
        return x
