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
        # Learnable per-keypoint edge extractor (replaces fixed Sobel)
        # Input: grayscale image (1 channel) -> Output: K channels (one per keypoint)
        self.edge_conv = nn.Conv2d(1, landmark_num_points, kernel_size=3, padding=1, bias=False)
        # Initialize edge_conv with Sobel-like kernel to give a sensible prior
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        sobel_mag = torch.sqrt(sobel_x.pow(2) + sobel_y.pow(2)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            # replicate sobel_mag for each output channel
            self.edge_conv.weight.data.copy_(sobel_mag.repeat(landmark_num_points, 1, 1, 1))

        # Per-keypoint multiplicative strength (beta) and additive bias on logits
        self.edge_beta = nn.Parameter(torch.full((landmark_num_points,), float(edge_guidance_beta)))
        self.landmark_bias = nn.Parameter(torch.zeros(landmark_num_points))
        # Keep a fixed Sobel kernel for consistency target (not learnable)
        self.register_buffer("_fixed_sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("_fixed_sobel_y", sobel_y.view(1, 1, 3, 3))

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
        # Convert to grayscale and run learnable conv -> per-keypoint edge logits
        gray = image.mean(dim=1, keepdim=True)
        edge = self.edge_conv(gray)
        edge = F.interpolate(edge, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize per-sample, per-keypoint to [0,1]
        denom = edge.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        edge = edge / denom

        # Stabilize values and map to (0,1) range with sharpness controlled by edge_alpha
        return torch.sigmoid((edge - 0.5) * self.edge_alpha)

    def _build_sobel_target(self, image, h, w):
        # Fixed Sobel target (single channel) duplicated to K channels
        if image is None or image.ndim != 4:
            return None
        gray = image.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self._fixed_sobel_x, padding=1)
        grad_y = F.conv2d(gray, self._fixed_sobel_y, padding=1)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        edge = F.interpolate(edge, size=(h, w), mode="bilinear", align_corners=False)
        edge = edge / edge.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        return torch.sigmoid((edge - 0.5) * self.edge_alpha)

    def forward(self, feat_map, input_image=None):
        attn_logits = self.landmark_heatmap_head(feat_map)
        bsz, keypoints, h, w = attn_logits.shape
        edge_attn = self._build_edge_attention(input_image, h, w)
        sobel_target = self._build_sobel_target(input_image, h, w)

        # Lightweight cross-head competition to avoid same hotspot collapse.
        attn_logits = attn_logits - attn_logits.mean(dim=1, keepdim=True)

        # If edge guidance is available, add the learned edge_pred as a prior to logits
        if edge_attn is not None:
            # edge_beta: (K,) -> (1,K,1,1); landmark_bias: (K,) -> (1,K,1,1)
            beta = (self.edge_beta.view(1, keypoints, 1, 1) * self.current_edge_weight).to(attn_logits.dtype)
            bias = self.landmark_bias.view(1, keypoints, 1, 1).to(attn_logits.dtype)
            attn_logits = attn_logits + beta * edge_attn + bias

        scaled = attn_logits.view(bsz, keypoints, -1) / max(self.landmark_tau, 1e-6)
        attn = torch.softmax(scaled, dim=-1).view(bsz, keypoints, h, w)

        if self.training and self.head_dropout_p > 0.0 and keypoints > 1:
            kp_keep = (torch.rand(bsz, keypoints, 1, 1, device=attn.device) > self.head_dropout_p).to(attn.dtype)
            has_any = kp_keep.sum(dim=1, keepdim=True) > 0
            kp_keep = torch.where(has_any, kp_keep, torch.ones_like(kp_keep))
            attn = attn * kp_keep

        # Normalize attention maps
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

        # Edge consistency per keypoint: compare predicted attn maps with fixed Sobel target
        if sobel_target is not None:
            # duplicate sobel_target to K channels
            sobel_k = sobel_target.repeat(1, keypoints, 1, 1)
            edge_consistency = F.l1_loss(attn, sobel_k, reduction="mean")
            # legacy alignment metric (global mean vs edge mean using sobel)
            edge_align = ((attn.mean(dim=1, keepdim=True) - sobel_k) ** 2).mean()
        else:
            edge_consistency = attn.new_tensor(0.0)
            edge_align = attn.new_tensor(0.0)

        # Regularize the learnable edge_conv weights to stay close to Sobel initialization
        with torch.no_grad():
            fixed = torch.sqrt(self._fixed_sobel_x.pow(2) + self._fixed_sobel_y.pow(2))
        # fixed shape: (1,1,3,3) -> expand to (K,1,3,3)
        fixed_rep = fixed.repeat(keypoints, 1, 1, 1).to(self.edge_conv.weight.device)
        edge_conv_reg = F.mse_loss(self.edge_conv.weight, fixed_rep, reduction="mean")

        # Total variation (smoothness) regularizer on learned edge_pred
        if edge_attn is not None:
            e = edge_attn
            tv_h = torch.abs(e[:, :, 1:, :] - e[:, :, :-1, :]).mean()
            tv_w = torch.abs(e[:, :, :, 1:] - e[:, :, :, :-1]).mean()
            edge_tv = (tv_h + tv_w) * 0.5
        else:
            edge_tv = attn.new_tensor(0.0)

        aux = {
            "landmark_diversity": self._diversity_loss(attn),
            "landmark_entropy": self._entropy_loss(attn),
            "landmark_edge_align": edge_align,
            "landmark_edge_consistency": edge_consistency,
            "landmark_edge_conv_reg": edge_conv_reg,
            "landmark_edge_tv": edge_tv,
        }

        return attn, coords, feat_k, aux