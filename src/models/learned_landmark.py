import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLandmarkBranch(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        landmark_num_points=12,
        landmark_tau=0.03,
        feature_dropout_p=0.3,
        heatmap_mask_prob=0.2,
        prior_strength=0.15,
        prior_sigma=0.22,
        keypoint_dropout_p=0.2,
    ):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.feature_dropout_p = feature_dropout_p
        self.heatmap_mask_prob = heatmap_mask_prob
        self.prior_strength = prior_strength
        self.prior_sigma = prior_sigma
        self.keypoint_dropout_p = keypoint_dropout_p

        self.landmark_heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, landmark_num_points, kernel_size=1),
        )

    def _build_keypoint_priors(self, h, w, device, dtype):
        # Face-aware anchors for FER: brows/eyes, nose, mouth region.
        base_centers = [
            [0.28, 0.24],  # left brow/eye
            [0.72, 0.24],  # right brow/eye
            [0.22, 0.38],  # left eye side
            [0.78, 0.38],  # right eye side
            [0.50, 0.50],  # nose
            [0.34, 0.74],  # left mouth
            [0.66, 0.74],  # right mouth
            [0.50, 0.84],  # lower lip/chin transition
        ]
        centers = []
        for i in range(self.landmark_num_points):
            centers.append(base_centers[i % len(base_centers)])
        centers = torch.tensor(centers, device=device, dtype=dtype)

        gx = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        gy = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0)

        diff = grid.unsqueeze(0) - centers.unsqueeze(-1).unsqueeze(-1)
        dist2 = (diff ** 2).sum(dim=1)
        prior = -dist2 / max(2.0 * (self.prior_sigma ** 2), 1e-6)
        return prior

    @staticmethod
    def _soft_argmax(probs):
        # probs: (B, K, H, W), normalize per keypoint map for numerical stability
        bsz, keypoints, h, w = probs.shape
        flat = probs.view(bsz, keypoints, -1)
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        xs = torch.linspace(0, 1, w, device=probs.device, dtype=probs.dtype)
        ys = torch.linspace(0, 1, h, device=probs.device, dtype=probs.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x = (flat * grid_x).sum(dim=-1)
        y = (flat * grid_y).sum(dim=-1)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _diversity_loss(heatmaps):
        bsz, keypoints, _, _ = heatmaps.shape
        if keypoints <= 1:
            return heatmaps.new_tensor(0.0)

        flat = heatmaps.view(bsz, keypoints, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = torch.bmm(flat, flat.transpose(1, 2))

        eye = torch.eye(keypoints, device=heatmaps.device, dtype=heatmaps.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return off_diag.pow(2).mean()

    @staticmethod
    def _entropy_loss(heatmaps):
        eps = 1e-6
        p = heatmaps.clamp(min=eps)
        return -(p * torch.log(p)).sum(dim=[2, 3]).mean()

    @staticmethod
    def _separation_loss(coords):
        # coords: (B, K, 2)
        bsz, keypoints, _ = coords.shape
        if keypoints <= 1:
            return coords.new_tensor(0.0)

        dist = torch.cdist(coords, coords)
        eye = torch.eye(keypoints, device=coords.device, dtype=coords.dtype).unsqueeze(0)
        dist = dist + eye
        inv_dist = 1.0 / dist.clamp(min=1e-6)
        off_diag = inv_dist * (1.0 - eye)
        return off_diag.sum() / ((bsz * keypoints * (keypoints - 1)) + 1e-6)

    @staticmethod
    def _build_part_masks(h, w, device, dtype):
        # Coarse face-region masks for FER-aligned faces.
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        left_eye = ((grid_x >= 0.10) & (grid_x <= 0.45) & (grid_y >= 0.14) & (grid_y <= 0.45)).to(dtype)
        right_eye = ((grid_x >= 0.55) & (grid_x <= 0.90) & (grid_y >= 0.14) & (grid_y <= 0.45)).to(dtype)
        nose = ((grid_x >= 0.35) & (grid_x <= 0.65) & (grid_y >= 0.35) & (grid_y <= 0.68)).to(dtype)
        mouth = ((grid_x >= 0.20) & (grid_x <= 0.80) & (grid_y >= 0.58) & (grid_y <= 0.95)).to(dtype)

        return [left_eye, right_eye, nose, mouth]

    def _part_prior_loss(self, probs):
        # Encourage each keypoint to stay in meaningful facial parts.
        bsz, keypoints, h, w = probs.shape
        part_masks = self._build_part_masks(h, w, probs.device, probs.dtype)

        part_ids = []
        for k in range(keypoints):
            if k < 2:
                part_ids.append(k)  # left_eye, right_eye
            elif k == 2:
                part_ids.append(2)  # nose
            else:
                part_ids.append(3)  # mouth

        penalties = []
        for k, part_id in enumerate(part_ids):
            mask = part_masks[part_id].unsqueeze(0)
            inside = (probs[:, k] * mask).sum(dim=[1, 2])
            penalties.append((1.0 - inside).mean())
        return torch.stack(penalties).mean()

    def forward(self, feat_map):
        logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = logits.shape

        if self.prior_strength > 0.0:
            prior_logits = self._build_keypoint_priors(h, w, logits.device, logits.dtype)
            logits = logits + (self.prior_strength * prior_logits.unsqueeze(0))

        scaled = logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        probs = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)
        base_probs = probs

        if self.training and self.heatmap_mask_prob > 0.0:
            keep_mask = (torch.rand_like(probs) > self.heatmap_mask_prob).to(probs.dtype)
            probs = probs * keep_mask
            spatial_sum = probs.sum(dim=[2, 3], keepdim=True)
            probs = torch.where(spatial_sum > 1e-6, probs / spatial_sum.clamp(min=1e-6), base_probs)

        if self.training and self.keypoint_dropout_p > 0.0 and keypoints > 1:
            kp_keep = (torch.rand(bsz, keypoints, 1, 1, device=probs.device) > self.keypoint_dropout_p).to(probs.dtype)
            # Ensure at least one keypoint remains active per sample.
            has_any = kp_keep.sum(dim=1, keepdim=True) > 0
            fallback = torch.ones_like(kp_keep[:, :1])
            kp_keep = torch.where(has_any, kp_keep, torch.cat([fallback, torch.zeros_like(kp_keep[:, 1:])], dim=1))
            probs = probs * kp_keep
            spatial_sum = probs.sum(dim=[2, 3], keepdim=True)
            probs = torch.where(spatial_sum > 1e-6, probs / spatial_sum.clamp(min=1e-6), base_probs)

        # Competition across K heatmaps to avoid all points collapsing to one area.
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-6)
        # Re-normalize per keypoint after competition for stable coords/features.
        probs = probs / probs.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        coords = self._soft_argmax(probs)

        # Keep per-keypoint feature paths, no SUM over K.
        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = probs.unsqueeze(2)
        feat_k = (feat_expanded * heat_expanded).mean(dim=[3, 4])
        feat_k = feat_k.view(bsz, -1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        aux = {
            "landmark_diversity": self._diversity_loss(probs),
            "landmark_entropy": self._entropy_loss(probs),
            "landmark_separation": self._separation_loss(coords),
            "landmark_part_prior": self._part_prior_loss(probs),
        }
        return probs, coords, feat_k, aux
