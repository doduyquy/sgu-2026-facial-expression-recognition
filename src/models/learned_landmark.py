import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLandmarkBranch(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        landmark_num_points=6,
        landmark_tau=0.1,
        feature_dropout_p=0.3,
        head_dropout_p=0.2,
    ):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.feature_dropout_p = feature_dropout_p
        self.head_dropout_p = head_dropout_p

        # Multi-head spatial attention maps: one head = one attended facial region.
        self.attn_head = nn.Conv2d(in_channels, landmark_num_points, kernel_size=1)

    @staticmethod
    def _soft_argmax(attn):
        # attn: (B, K, H, W)
        bsz, keypoints, h, w = attn.shape
        flat = attn.view(bsz, keypoints, -1)
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        xs = torch.linspace(0, 1, w, device=attn.device, dtype=attn.dtype)
        ys = torch.linspace(0, 1, h, device=attn.device, dtype=attn.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x = (flat * grid_x).sum(dim=-1)
        y = (flat * grid_y).sum(dim=-1)
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _diversity_loss(attn):
        bsz, keypoints, _, _ = attn.shape
        if keypoints <= 1:
            return attn.new_tensor(0.0)

        flat = attn.view(bsz, keypoints, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = torch.bmm(flat, flat.transpose(1, 2))

        eye = torch.eye(keypoints, device=attn.device, dtype=attn.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return off_diag.pow(2).mean()

    @staticmethod
    def _entropy_loss(attn):
        eps = 1e-6
        p = attn.clamp(min=eps)
        return -(p * torch.log(p)).sum(dim=[2, 3]).mean()

    def forward(self, feat_map):
        attn_logits = self.attn_head(feat_map)
        bsz, keypoints, h, w = attn_logits.shape
        scaled = attn_logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        attn = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)

        if self.training and self.head_dropout_p > 0.0 and keypoints > 1:
            keep = (torch.rand(bsz, keypoints, 1, 1, device=attn.device) > self.head_dropout_p).to(attn.dtype)
            has_any = keep.sum(dim=1, keepdim=True) > 0
            keep = torch.where(has_any, keep, torch.ones_like(keep))
            attn = attn * keep
            attn = attn / attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        coords = self._soft_argmax(attn)

        feat = feat_map.unsqueeze(1)
        attn_exp = attn.unsqueeze(2)
        feat_k = (feat * attn_exp).sum(dim=[3, 4])
        feat_k = feat_k.view(bsz, -1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        aux = {
            "landmark_diversity": self._diversity_loss(attn),
            "landmark_entropy": self._entropy_loss(attn),
        }
        return attn, coords, feat_k, aux
