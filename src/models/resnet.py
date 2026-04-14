import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes=7,
        in_channels=1,
        use_cbam_stage34=True,
        cbam_reduction=16,
        cbam_kernel_size=7,
        use_learned_landmark_branch=True,
        landmark_num_points=12,
        landmark_tau=0.07,
        landmark_feature_dropout_p=0.3,
        landmark_head_dropout_p=0.2,
        landmark_logit_noise_std=0.1,
        use_multiscale_landmark_branch=True,
        landmark_tau_start=0.5,
        landmark_tau_mid=0.2,
        landmark_tau_end=0.05,

        landmark_from_stage=3,
    ):
        super().__init__()
        self.use_learned_landmark_branch = use_learned_landmark_branch
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.landmark_from_stage = landmark_from_stage
        self.use_multiscale_landmark_branch = use_multiscale_landmark_branch

        self._latest_aux_losses = {}
        self._latest_landmark_heatmaps = None
        self._latest_landmark_coords = None

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

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

        self.learned_landmark_branch3 = LearnedLandmarkBranch(
            in_channels=512,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            feature_dropout_p=landmark_feature_dropout_p,
            head_dropout_p=landmark_head_dropout_p,
            logit_noise_std=landmark_logit_noise_std,
            tau_start=landmark_tau_start,
            tau_mid=landmark_tau_mid,
            tau_end=landmark_tau_end,
        )

        self.learned_landmark_branch4 = LearnedLandmarkBranch(
            in_channels=1024,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            feature_dropout_p=landmark_feature_dropout_p,
            head_dropout_p=landmark_head_dropout_p,
            logit_noise_std=landmark_logit_noise_std,
            tau_start=landmark_tau_start,
            tau_mid=landmark_tau_mid,
            tau_end=landmark_tau_end,
        )

        if self.use_multiscale_landmark_branch:
            fusion_in_dim = 1024 + ((landmark_num_points + 1) * 512) + ((landmark_num_points + 1) * 1024)
        else:
            landmark_in_channels = 512 if landmark_from_stage == 3 else 1024
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
        self.learned_landmark_branch3.set_training_progress(progress)
        self.learned_landmark_branch4.set_training_progress(progress)

    def get_current_prior_strength(self):
        return None

    def forward(self, x, landmarks=None, landmark_mask=None):
        _ = landmarks
        _ = landmark_mask

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

        if self.use_multiscale_landmark_branch:
            heatmaps3, coords3, feat_k3, aux3 = self.learned_landmark_branch3(x3)
            heatmaps4, coords4, feat_k4, aux4 = self.learned_landmark_branch4(x4)
            fused = torch.cat([feat4, feat_k3, feat_k4], dim=1)
            logits = self.landmark_fusion_fc(fused)

            heatmaps4_up = F.interpolate(
                heatmaps4,
                size=heatmaps3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            self._latest_landmark_heatmaps = torch.cat([heatmaps3, heatmaps4_up], dim=1)
            self._latest_landmark_coords = torch.cat([coords3, coords4], dim=1)

            self._latest_aux_losses = {
                "landmark_diversity": 0.5 * (aux3["landmark_diversity"] + aux4["landmark_diversity"]),
                "landmark_entropy": 0.5 * (aux3["landmark_entropy"] + aux4["landmark_entropy"]),
                "landmark_coord_separation": 0.5 * (aux3["landmark_coord_separation"] + aux4["landmark_coord_separation"]),
                "landmark_balance": 0.5 * (aux3["landmark_balance"] + aux4["landmark_balance"]),
            }
            return logits

        landmark_src = x3 if self.landmark_from_stage == 3 else x4
        branch = self.learned_landmark_branch3 if self.landmark_from_stage == 3 else self.learned_landmark_branch4
        heatmaps, coords, feat_k, aux = branch(landmark_src)
        fused = torch.cat([feat4, feat_k], dim=1)
        logits = self.landmark_fusion_fc(fused)

        self._latest_aux_losses = aux
        self._latest_landmark_heatmaps = heatmaps
        self._latest_landmark_coords = coords

        return logits
