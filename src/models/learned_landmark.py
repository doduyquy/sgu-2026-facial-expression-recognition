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
        use_topk_pooling=True,
        topk_pool_k=5,
    ):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.feature_dropout_p = feature_dropout_p
        self.use_topk_pooling = use_topk_pooling
        self.topk_pool_k = topk_pool_k

        self.landmark_heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, landmark_num_points, kernel_size=1),
        )

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

    def _sparsify_topk(self, probs):
        if (not self.use_topk_pooling) or self.topk_pool_k <= 0:
            return probs

        bsz, keypoints, h, w = probs.shape
        flat = probs.view(bsz, keypoints, -1)
        k = min(self.topk_pool_k, h * w)

        top_vals, top_idx = torch.topk(flat, k=k, dim=-1)
        sparse_flat = torch.zeros_like(flat)
        sparse_flat.scatter_(dim=-1, index=top_idx, src=top_vals)
        sparse = sparse_flat.view(bsz, keypoints, h, w)
        sparse = sparse / sparse.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        return sparse

    def forward(self, feat_map):
        logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = logits.shape
        scaled = logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        probs = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)
        probs = probs / probs.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        coords = self._soft_argmax(probs)
        probs_pool = self._sparsify_topk(probs)

        # Weighted SUM pooling keeps strong local evidence better than mean pooling.
        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = probs_pool.unsqueeze(2)
        feat_k = (feat_expanded * heat_expanded).sum(dim=[3, 4])
        feat_k = feat_k.view(bsz, -1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        aux = {
            "landmark_diversity": self._diversity_loss(probs),
            "landmark_entropy": self._entropy_loss(probs),
            "landmark_separation": self._separation_loss(coords),
        }
        return probs, coords, feat_k, aux
