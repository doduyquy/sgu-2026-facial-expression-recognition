import torch
import torch.nn as nn


class LearnedLandmarkBranch(nn.Module):
    def __init__(self, in_channels=1024, landmark_num_points=12, landmark_tau=0.03):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau

        self.landmark_heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, landmark_num_points, kernel_size=1),
        )

    @staticmethod
    def _soft_argmax(probs):
        # probs: (B, K, H, W), already normalized
        bsz, keypoints, h, w = probs.shape
        flat = probs.view(bsz, keypoints, -1)

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

    def forward(self, feat_map):
        logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = logits.shape

        scaled = logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        probs = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)

        # Competition across K heatmaps to avoid all points collapsing to one area.
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-6)

        coords = self._soft_argmax(probs)

        # Keep per-keypoint feature paths, no SUM over K.
        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = probs.unsqueeze(2)
        feat_k = (feat_expanded * heat_expanded).mean(dim=[3, 4])
        feat_k = feat_k.view(bsz, -1)

        aux = {
            "landmark_diversity": self._diversity_loss(probs),
            "landmark_entropy": self._entropy_loss(probs),
        }
        return probs, coords, feat_k, aux
