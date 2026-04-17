import torch
import torch.nn as nn
from .CBAM import CBAM
from .learned_landmark import LearnedLandmarkBranch


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

        # Attention gate giữa 2 nhánh (learnable)
        self.gate_conv = nn.Conv2d(64, 32, kernel_size=1)  # Đầu ra 32 kênh (gating cho x1)

        # Sau khi attention fusion: 32 channels
        self.layer2 = nn.Sequential(
            ConvBlock(32, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256]),
        )

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
        # img_goc, img_sobel: [B, 1, H, W]
        x1 = self.relu(self.bn1_goc(self.conv1_goc(img_goc)))  # [B, 32, H, W]
        x2 = self.relu(self.bn1_sobel(self.conv1_sobel(img_sobel)))
        x_cat = torch.cat([x1, x2], dim=1)  # [B, 64, H, W]
        # Attention gate: học trọng số cho x1, x2
        alpha = torch.sigmoid(self.gate_conv(x_cat))  # [B, 32, H, W], giá trị (0,1)
        x = alpha * x1 + (1 - alpha) * x2  # attention fusion
        x = self.pool(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        feat = torch.flatten(self.avgpool(x4), 1)
        out = self.fusion_fc(feat)
        if return_features:
            # Trả về feature map cuối của từng nhánh đầu vào (sau conv1)
            return out, x1, x2, alpha
        return out

    def set_training_progress(self, progress):
        # No-op for dual branch (no landmark branch)
        pass

    def get_current_prior_strength(self):
        # No-op for dual branch (no landmark branch)
        return None

        self.layer2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256]),
        )

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

        # Baseline classifier (no landmark branch).
        self.fusion_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        landmark_in_channels = 512 if landmark_from_stage == 3 else 1024

        self.learned_landmark_branch = LearnedLandmarkBranch(
            in_channels=landmark_in_channels,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            feature_dropout_p=landmark_feature_dropout_p,
            head_dropout_p=landmark_head_dropout_p,
            edge_guidance_beta=landmark_edge_guidance_beta,
            edge_alpha=landmark_edge_alpha,
            edge_feat_guidance_beta=landmark_edge_feat_guidance_beta,
            edge_dropout_prob=landmark_edge_dropout_prob,
            edge_head_scale_std=landmark_edge_head_scale_std,
            edge_mask_threshold=landmark_edge_mask_threshold,
            edge_gamma=landmark_edge_gamma,
        )

        fusion_in_dim = 1024 + ((landmark_num_points + 1) * landmark_in_channels)
        self.landmark_fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def get_aux_losses(self):
        return self._latest_aux_losses

    def get_landmark_outputs(self):
        return self._latest_landmark_heatmaps, self._latest_landmark_coords

    def set_training_progress(self, progress):
        setter = getattr(self.learned_landmark_branch, "set_training_progress", None)
        if callable(setter):
            setter(progress)

    def get_current_prior_strength(self):
        getter = getattr(self.learned_landmark_branch, "get_current_prior_strength", None)
        if callable(getter):
            return getter()
        return None

    def forward(self, x, landmarks=None, landmark_mask=None):
        _ = landmarks
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
