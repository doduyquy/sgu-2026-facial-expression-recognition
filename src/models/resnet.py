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
        landmark_tau=0.03,
        landmark_feature_dropout_p=0.3,
        landmark_mask_prob=0.2,
        landmark_prior_strength=0.05,
        landmark_prior_sigma=0.22,
        landmark_keypoint_dropout_p=0.1,
        landmark_prior_min_strength=0.0,
        landmark_prior_anneal_power=1.5,
        landmark_part_mask_expand=0.08,
        landmark_part_target_inside=0.35,
        landmark_prior_disable_after_progress=0.3,
        landmark_use_cross_keypoint_competition=False,
        landmark_post_softmax_sharpness=1.3,
        landmark_use_soft_face_mask=True,
        landmark_face_mask_strength=0.15,
        landmark_from_stage=3,
    ):
        super().__init__()
        self.use_learned_landmark_branch = use_learned_landmark_branch
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.landmark_from_stage = landmark_from_stage

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

        landmark_in_channels = 512 if landmark_from_stage == 3 else 1024

        self.learned_landmark_branch = LearnedLandmarkBranch(
            in_channels=landmark_in_channels,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            feature_dropout_p=landmark_feature_dropout_p,
            heatmap_mask_prob=landmark_mask_prob,
            prior_strength=landmark_prior_strength,
            prior_sigma=landmark_prior_sigma,
            keypoint_dropout_p=landmark_keypoint_dropout_p,
            prior_min_strength=landmark_prior_min_strength,
            prior_anneal_power=landmark_prior_anneal_power,
            part_mask_expand=landmark_part_mask_expand,
            part_target_inside=landmark_part_target_inside,
            prior_disable_after_progress=landmark_prior_disable_after_progress,
            use_cross_keypoint_competition=landmark_use_cross_keypoint_competition,
            post_softmax_sharpness=landmark_post_softmax_sharpness,
            use_soft_face_mask=landmark_use_soft_face_mask,
            face_mask_strength=landmark_face_mask_strength,
        )

        fusion_in_dim = 1024 + (landmark_num_points * landmark_in_channels)
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
        heatmaps, coords, feat_k, aux = self.learned_landmark_branch(landmark_src)
        fused = torch.cat([feat4, feat_k], dim=1)
        logits = self.landmark_fusion_fc(fused)

        self._latest_aux_losses = aux
        self._latest_landmark_heatmaps = heatmaps
        self._latest_landmark_coords = coords

        return logits
