import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        if "convnext_tiny" in self.backbone_name or "convnext_t" in self.backbone_name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.out_channels = 768
            self.backbone_type = "convnext"
        elif "convnext_small" in self.backbone_name or "convnext_s" in self.backbone_name:
            weights = models.ConvNeXt_Small_Weights.DEFAULT if use_pretrained else None
            base = models.convnext_small(weights=weights)
            self.out_channels = 768
            self.backbone_type = "convnext"
        elif "densenet121" in self.backbone_name:
            weights = models.DenseNet121_Weights.DEFAULT if use_pretrained else None
            base = models.densenet121(weights=weights)
            self.out_channels = 1024
            self.backbone_type = "densenet"
        elif "densenet169" in self.backbone_name:
            weights = models.DenseNet169_Weights.DEFAULT if use_pretrained else None
            base = models.densenet169(weights=weights)
            self.out_channels = 1664
            self.backbone_type = "densenet"
        elif "densenet201" in self.backbone_name:
            weights = models.DenseNet201_Weights.DEFAULT if use_pretrained else None
            base = models.densenet201(weights=weights)
            self.out_channels = 1920
            self.backbone_type = "densenet"
        elif "resnet50" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            base = models.resnet50(weights=weights)
            self.out_channels = 2048
            self.backbone_type = "resnet"
        elif "resnet34" in self.backbone_name:
            weights = models.ResNet34_Weights.DEFAULT if use_pretrained else None
            base = models.resnet34(weights=weights)
            self.out_channels = 512
            self.backbone_type = "resnet"
        elif "resnet18" in self.backbone_name:
            weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
            base = models.resnet18(weights=weights)
            self.out_channels = 512
            self.backbone_type = "resnet"
        else:
            # Default fallback to densenet121
            weights = models.DenseNet121_Weights.DEFAULT if use_pretrained else None
            base = models.densenet121(weights=weights)
            self.out_channels = 1024
            self.backbone_type = "densenet"

        if self.backbone_type == "convnext":
            # Adapt ConvNeXt for small 48x48 facial images
            features = base.features
            orig_conv0 = features[0][0]
            new_conv0 = nn.Conv2d(
                in_channels,
                orig_conv0.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=orig_conv0.bias is not None,
            )
            if use_pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        kernel_1ch_4x4 = orig_conv0.weight.mean(dim=1, keepdim=True)
                        kernel_1ch_3x3 = F.interpolate(kernel_1ch_4x4, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv0.weight.copy_(kernel_1ch_3x3)
                    elif in_channels == 3:
                        kernel_3ch_3x3 = F.interpolate(orig_conv0.weight, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv0.weight.copy_(kernel_3ch_3x3)
                    if orig_conv0.bias is not None:
                        new_conv0.bias.copy_(orig_conv0.bias)
            features[0][0] = new_conv0

            # Modify Stage 6 downsample to stride 1 to keep 12x12 feature map
            if target_feat_size == 12:
                orig_ds6 = features[6][1]
                new_ds6 = nn.Conv2d(
                    orig_ds6.in_channels,
                    orig_ds6.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=orig_ds6.bias is not None,
                )
                if use_pretrained:
                    with torch.no_grad():
                        kernel_3x3 = F.interpolate(orig_ds6.weight, size=(3, 3), mode='bilinear', align_corners=False)
                        new_ds6.weight.copy_(kernel_3x3)
                        if orig_ds6.bias is not None:
                            new_ds6.bias.copy_(orig_ds6.bias)
                features[6][1] = new_ds6

            self.features = features
        elif self.backbone_type == "densenet":
            # Adapt DenseNet for small 48x48 facial images
            features = base.features
            orig_conv0 = features.conv0
            new_conv0 = nn.Conv2d(
                in_channels,
                orig_conv0.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            if use_pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        kernel_1ch_7x7 = orig_conv0.weight.mean(dim=1, keepdim=True)
                        kernel_1ch_3x3 = F.interpolate(kernel_1ch_7x7, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv0.weight.copy_(kernel_1ch_3x3)
                    elif in_channels == 3:
                        kernel_3ch_3x3 = F.interpolate(orig_conv0.weight, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv0.weight.copy_(kernel_3ch_3x3)

            features.conv0 = new_conv0
            features.pool0 = nn.Identity()  # Remove early maxpool to preserve facial detail

            # Modify transition3 pool to identity to keep 12x12 feature map
            if target_feat_size == 12 and hasattr(features, "transition3"):
                features.transition3.pool = nn.Identity()

            self.features = features
        else:
            # Adapt ResNet for small 48x48 facial images
            orig_conv1 = base.conv1
            new_conv1 = nn.Conv2d(
                in_channels,
                orig_conv1.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

            # Transfer pretrained weights to new 3x3 conv1
            if use_pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        kernel_1ch_7x7 = orig_conv1.weight.mean(dim=1, keepdim=True)
                        kernel_1ch_3x3 = F.interpolate(kernel_1ch_7x7, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv1.weight.copy_(kernel_1ch_3x3)
                    elif in_channels == 3:
                        kernel_3ch_3x3 = F.interpolate(orig_conv1.weight, size=(3, 3), mode='bilinear', align_corners=False)
                        new_conv1.weight.copy_(kernel_3ch_3x3)

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
        if self.backbone_type == "convnext":
            return self.features(x)
        elif self.backbone_type == "densenet":
            out = self.features(x)
            out = F.relu(out, inplace=True)
            return out
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)  # [B, C1, 48, 48]
            x = self.layer2(x)  # [B, C2, 24, 24]
            x = self.layer3(x)  # [B, C3, 12, 12]
            x = self.layer4(x)  # [B, C4, 12, 12]
            return x
