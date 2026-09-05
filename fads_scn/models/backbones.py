import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models


class FacialBackbone(nn.Module):
    """
    Backbone modified for 48x48 facial images.
    Modifications:
    1. First conv modified to accept in_channels (1 for grayscale).
    2. Initial 7x7 stride-2 conv replaced with 3x3 stride-1 conv, maxpool removed.
       This prevents spatial collapse on small 48x48 images, preserving a rich
       feature map of size (C, 12, 12) or (C, 6, 6).
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        in_channels: int = 1,
        use_pretrained: bool = True,
        pretrained_weights_path: str = "",
        target_feat_size: int = 12,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.in_channels = in_channels

        if "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            base = models.resnet50(weights=weights)
            self.out_channels = 2048
        elif "resnet34" in self.backbone_name:
            weights = models.ResNet34_Weights.DEFAULT if use_pretrained else None
            base = models.resnet34(weights=weights)
            self.out_channels = 512
        elif "resnet18" in self.backbone_name:
            weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
            base = models.resnet18(weights=weights)
            self.out_channels = 512
        else:
            # Fallback to resnet34
            weights = models.ResNet34_Weights.DEFAULT if use_pretrained else None
            base = models.resnet34(weights=weights)
            self.out_channels = 512

        # Adapt first conv for small 48x48 grayscale face images
        orig_conv1 = base.conv1
        new_conv1 = nn.Conv2d(
            in_channels,
            orig_conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Transfer pretrained weights to 1-channel conv
        if use_pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    new_conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
                elif in_channels == 3:
                    # Resize 7x7 kernel to 3x3 or take center 3x3
                    new_conv1.weight.copy_(orig_conv1.weight[:, :, 2:5, 2:5])

        self.conv1 = new_conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        # Remove early maxpool to preserve facial detail
        self.layer1 = base.layer1  # 48x48
        self.layer2 = base.layer2  # 24x24
        self.layer3 = base.layer3  # 12x12
        self.layer4 = base.layer4  # 6x6 -> if stride 1 in layer4: 12x12

        # Modify layer4 first block stride to 1 to maintain 12x12 feature map
        if target_feat_size == 12:
            self._set_layer_stride1(self.layer4)

        # Load custom facial pre-trained weights if provided
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            self._load_custom_weights(pretrained_weights_path)

    def _set_layer_stride1(self, layer):
        """Set stride to 1 for the first block in layer to maintain 12x12 resolution."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)
                break
            if hasattr(module, "downsample") and module.downsample is not None:
                for sub in module.downsample.modules():
                    if isinstance(sub, nn.Conv2d) and sub.stride == (2, 2):
                        sub.stride = (1, 1)

    def _load_custom_weights(self, path: str):
        try:
            state = torch.load(path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            msg = self.load_state_dict(state, strict=False)
            print(f"[FacialBackbone] Loaded custom facial weights from {path}: {msg}")
        except Exception as e:
            print(f"[FacialBackbone] Warning: Could not load custom weights from {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, in_channels, 48, 48]
        Output: Feature Map F of shape [B, out_channels, 12, 12]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # [B, C1, 48, 48]
        x = self.layer2(x)  # [B, C2, 24, 24]
        x = self.layer3(x)  # [B, C3, 12, 12]
        x = self.layer4(x)  # [B, C4, 12, 12]
        return x
