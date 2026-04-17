import torch
import torch.nn as nn
from .CBAM import CBAM


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.cbam = CBAM(f3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.cbam(x)
        x = self.relu(x + shortcut)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.cbam = CBAM(f3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f3),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.cbam(x)

        x = self.relu(x + shortcut)
        return x




class ResNetDualBranch(nn.Module):
    def __init__(
        self,
        num_classes=7,
        use_cbam_stage34=True,
        cbam_reduction=16,
        cbam_kernel_size=7,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        # Hai conv1 riêng biệt cho ảnh gốc và sobel
        self.conv1_goc = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_goc = nn.BatchNorm2d(32)
        self.conv1_sobel = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_sobel = nn.BatchNorm2d(32)

        # Layer2 riêng cho từng nhánh
        self.layer2_goc = nn.Sequential(
            ConvBlock(32, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256]),
        )
        self.layer2_sobel = nn.Sequential(
            ConvBlock(32, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256]),
        )

        # Attention gate symmetry: 2 conv cho alpha, beta
        self.gate_conv_alpha = nn.Conv2d(512, 256, kernel_size=1)
        self.gate_conv_beta = nn.Conv2d(512, 256, kernel_size=1)

        # Sau khi fusion: 256 channels
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, img_goc, img_sobel, return_features=False):
        # conv1 + pool
        x1 = self.relu(self.bn1_goc(self.conv1_goc(img_goc)))
        x1 = self.pool(x1)
        x2 = self.relu(self.bn1_sobel(self.conv1_sobel(img_sobel)))
        x2 = self.pool(x2)
        # layer2 riêng
        x1 = self.layer2_goc(x1)
        x2 = self.layer2_sobel(x2)
        # concat feature để tính attention gate
        x_cat = torch.cat([x1, x2], dim=1)  # [B, 512, H, W]
        alpha = torch.sigmoid(self.gate_conv_alpha(x_cat))  # [B, 256, H, W]
        beta = torch.sigmoid(self.gate_conv_beta(x_cat))   # [B, 256, H, W]
        # symmetry fusion
        x = alpha * x1 + beta * x2
        # tiếp tục backbone
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        feat = torch.flatten(self.avgpool(x4), 1)
        out = self.fusion_fc(feat)
        if return_features:
            # Trả về output, feature fusion (sau attention), x1, x2, alpha, beta
            return out, x, x1, x2, alpha, beta
        return out

    def set_training_progress(self, progress):
        # No-op for dual branch (no landmark branch)
        pass

    def get_current_prior_strength(self):
        # No-op for dual branch (no landmark branch)
        return None
        _ = landmark_mask
        input_image = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)

        x3 = self.layer3(x)
        feat3 = torch.flatten(self.avgpool(x3), 1)

        x4 = self.layer4(x3)
        feat4 = torch.flatten(self.avgpool(x4), 1)

        if not self.use_learned_landmark_branch:
            feat = torch.cat([feat3, feat4], dim=1)
            self._latest_aux_losses = {}
            self._latest_landmark_heatmaps = None
            self._latest_landmark_coords = None
            return self.fusion_fc(feat)

        landmark_src = x3 if self.landmark_from_stage == 3 else x4
        heatmaps, coords, feat_k, aux = self.learned_landmark_branch(landmark_src, input_image=input_image)
        fused = torch.cat([feat4, feat_k], dim=1)
        logits = self.landmark_fusion_fc(fused)

        self._latest_aux_losses = aux
        self._latest_landmark_heatmaps = heatmaps
        self._latest_landmark_coords = coords

        return logits
