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
        landmark_tau=0.1,
    ):
        super().__init__()
        self.use_learned_landmark_branch = use_learned_landmark_branch
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau

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

        # Learned soft-landmark branch.
        self.landmark_heatmap_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, landmark_num_points, kernel_size=1),
        )

        self.landmark_fusion_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _soft_argmax(heatmaps):
        # heatmaps: (B, K, H, W)
        bsz, keypoints, h, w = heatmaps.shape
        flat = heatmaps.view(bsz, keypoints, -1)
        probs = torch.softmax(flat, dim=-1)

        xs = torch.linspace(0, 1, w, device=heatmaps.device, dtype=heatmaps.dtype)
        ys = torch.linspace(0, 1, h, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x = (probs * grid_x).sum(dim=-1)
        y = (probs * grid_y).sum(dim=-1)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _diversity_loss(heatmaps):
        # Encourage keypoints to attend to different regions.
        bsz, keypoints, h, w = heatmaps.shape
        if keypoints <= 1:
            return heatmaps.new_tensor(0.0)

        flat = heatmaps.view(bsz, keypoints, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = torch.bmm(flat, flat.transpose(1, 2))

        eye = torch.eye(keypoints, device=heatmaps.device, dtype=heatmaps.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return off_diag.pow(2).mean()

    @staticmethod
    def _sparsity_loss(heatmaps):
        # Lower entropy-like pressure for sharper heatmaps.
        return heatmaps.mean()

    def get_aux_losses(self):
        return self._latest_aux_losses

    def get_landmark_outputs(self):
        return self._latest_landmark_heatmaps, self._latest_landmark_coords

    def _compute_learned_landmarks(self, feat_map):
        # feat_map from stage4: (B,1024,H,W)
        logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = logits.shape

        scaled = logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        probs = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)
        coords = self._soft_argmax(probs)

        attn = probs.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        feat_attn = feat_map * attn

        aux = {
            "landmark_diversity": self._diversity_loss(probs),
            "landmark_sparsity": self._sparsity_loss(probs),
        }
        return probs, coords, feat_attn, aux

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

        heatmaps, coords, feat_attn_map, aux = self._compute_learned_landmarks(x4)
        feat_attn = torch.flatten(self.avgpool(feat_attn_map), 1)

        fused = torch.cat([feat4, feat_attn], dim=1)
        logits = self.landmark_fusion_fc(fused)

        self._latest_aux_losses = aux
        self._latest_landmark_heatmaps = heatmaps
        self._latest_landmark_coords = coords

        return logits
