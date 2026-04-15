import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLandmarkBranch(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        landmark_num_points=6,
        landmark_tau=0.07,
        feature_dropout_p=0.3,
        head_dropout_p=0.2,
        edge_guidance_beta=1.0,
        edge_alpha=6.0,
    ):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.feature_dropout_p = feature_dropout_p
        self.head_dropout_p = head_dropout_p
        self.edge_guidance_beta = edge_guidance_beta
        self.edge_alpha = edge_alpha
        self.current_edge_weight = 1.0

        self.landmark_heatmap_head = nn.Conv2d(in_channels, landmark_num_points, kernel_size=1)

    def set_training_progress(self, progress):
        # Strong edge guidance at start, then let attention self-organize later.
        progress = float(max(0.0, min(1.0, progress)))
        self.current_edge_weight = max(0.0, 1.0 - progress)

    def get_current_prior_strength(self):
        return float(self.current_edge_weight)

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
    def _diversity_loss(attn):
        bsz, keypoints, _, _ = attn.shape
        if keypoints <= 1:
            return attn.new_tensor(0.0)

        flat = attn.view(bsz, keypoints, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gram = torch.bmm(flat, flat.transpose(1, 2))
        eye = torch.eye(keypoints, device=attn.device, dtype=attn.dtype).unsqueeze(0)
        return (gram - eye).pow(2).mean()

    @staticmethod
    def _entropy_loss(attn):
        eps = 1e-6
        p = attn.clamp(min=eps)
        return -(p * torch.log(p)).sum(dim=[2, 3]).mean()

    def _build_edge_attention(self, image, h, w):
        if image is None or image.ndim != 4:
            return None

        # Edge map is used as prior only; detach to avoid unnecessary gradient graph.
        gray = image.detach().mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=gray.device,
            dtype=gray.dtype,
        ).unsqueeze(1)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=gray.device,
            dtype=gray.dtype,
        ).unsqueeze(1)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        edge = F.interpolate(edge, size=(h, w), mode="bilinear", align_corners=False)
        edge = edge / edge.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        # Strong edges -> values close to 1, weak edges -> values close to 0.
        return torch.sigmoid((edge - 0.5) * self.edge_alpha)

    def forward(self, feat_map, input_image=None):
        attn_logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = attn_logits.shape
        edge_attn = self._build_edge_attention(input_image, h, w)

        # Lightweight cross-head competition to avoid same hotspot collapse.
        attn_logits = attn_logits - attn_logits.mean(dim=1, keepdim=True)

        if edge_attn is not None and self.edge_guidance_beta > 0.0:
            # Apply edge guidance at logit level for stable and effective softmax shaping.
            guide = self.edge_guidance_beta * self.current_edge_weight
            attn_logits = attn_logits + (guide * edge_attn)

        scaled = attn_logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        attn = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)

        if self.training and self.head_dropout_p > 0.0 and keypoints > 1:
            kp_keep = (torch.rand(bsz, keypoints, 1, 1, device=attn.device) > self.head_dropout_p).to(attn.dtype)
            has_any = kp_keep.sum(dim=1, keepdim=True) > 0
            kp_keep = torch.where(has_any, kp_keep, torch.ones_like(kp_keep))
            attn = attn * kp_keep

        attn = attn / attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        coords = self._soft_argmax(attn)

        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = attn.unsqueeze(2)
        feat_k = (feat_expanded * heat_expanded).sum(dim=[3, 4])

        global_attn = attn.mean(dim=1, keepdim=True)
        feat_global = (feat_map * global_attn).sum(dim=[2, 3])

        feat_k = feat_k.view(bsz, -1)
        feat_k = torch.cat([feat_k, feat_global], dim=1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        aux = {
            "landmark_diversity": self._diversity_loss(attn),
            "landmark_entropy": self._entropy_loss(attn),
            "landmark_edge_align": (
                # Penalize only attention mass that lies outside edge regions.
                (attn * (1.0 - (edge_attn > 0.3).to(attn.dtype))).sum(dim=[2, 3]).mean()
                if edge_attn is not None
                else attn.new_tensor(0.0)
            ),
        }
        return attn, coords, feat_k, aux