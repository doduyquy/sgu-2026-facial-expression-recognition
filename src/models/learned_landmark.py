import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLandmarkBranch(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        landmark_num_points=6,
        landmark_tau=0.07,
        diversity_margin=0.1,
        feature_dropout_p=0.3,
        head_dropout_p=0.2,
        edge_guidance_beta=1.0,
        edge_alpha=6.0,
    ):
        super().__init__()
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.feature_dropout_p = feature_dropout_p
        # default safer dropout for small K; will only be enabled late in training
        self.head_dropout_p = head_dropout_p
        self._training_progress = 0.0
        self.edge_guidance_beta = edge_guidance_beta
        self.edge_alpha = edge_alpha
        self.current_edge_weight = 1.0
        # stronger default margin for low-res images (48x48)
        self.diversity_margin = float(diversity_margin)

        self.landmark_heatmap_head = nn.Conv2d(in_channels, landmark_num_points, kernel_size=1)
        # For low-res FER, remove learnable edge extractor and heavy sobel targets
        # Keep simple attention-only branch (soft regions), so do not create edge conv/bias
        self.edge_beta = None
        self.landmark_bias = None

    def set_training_progress(self, progress):
        # Strong edge guidance at start, then let attention self-organize later.
        progress = float(max(0.0, min(1.0, progress)))
        # use squared decay to let prior fade smoothly toward end of training
        self.current_edge_weight = float(max(0.0, (1.0 - progress) ** 2))
        # store progress so we can enable certain noisy ops only late in training
        self._training_progress = progress

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

    def _diversity_loss(self, attn, coords=None):
        # Encourage landmark coordinates to be far apart using pairwise distances
        # coords: (B, K, 2) in normalized [0,1] space. If not provided, fallback to previous gram-based loss.
        bsz, keypoints, _, _ = attn.shape
        if keypoints <= 1:
            return attn.new_tensor(0.0)

        if coords is None:
            # fallback: original feature-space diversity (weakened)
            flat = attn.view(bsz, keypoints, -1)
            flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            gram = torch.bmm(flat, flat.transpose(1, 2))
            eye = torch.eye(keypoints, device=attn.device, dtype=attn.dtype).unsqueeze(0)
            return (gram - eye).pow(2).mean()

        # coords: (B, K, 2)
        try:
            # pairwise distances per sample: (B, K, K)
            d = torch.cdist(coords, coords, p=2)
            # mask out diagonal
            mask = ~torch.eye(keypoints, device=d.device, dtype=torch.bool).unsqueeze(0)
            d_masked = d[mask].view(bsz, keypoints * (keypoints - 1))
            # stronger margin-based diversity: penalize when points closer than margin
            margin = getattr(self, 'diversity_margin', 0.05)
            # use relu(margin - distance) to push points apart when too close
            loss_per_sample = F.relu(margin - d_masked).mean(dim=1)
            return loss_per_sample.mean()
        except Exception:
            # safe fallback
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
        # Removed: edge guidance is unreliable for 48x48 FER. Return None so parent skips edge priors.
        return None

    def _build_sobel_target(self, image, h, w):
        # removed: no sobel target for low-res FER
        return None

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

        # Head dropout: keep conservative by enabling only late in training
        effective_head_p = 0.0
        try:
            if self.training and keypoints > 1:
                # enable head dropout only in refinement phase (progress >= 0.7)
                if getattr(self, '_training_progress', 0.0) >= 0.7:
                    effective_head_p = float(self.head_dropout_p)
        except Exception:
            effective_head_p = 0.0

        if effective_head_p > 0.0:
            kp_keep = (torch.rand(bsz, keypoints, 1, 1, device=attn.device) > effective_head_p).to(attn.dtype)
            has_any = kp_keep.sum(dim=1, keepdim=True) > 0
            kp_keep = torch.where(has_any, kp_keep, torch.ones_like(kp_keep))
            attn = attn * kp_keep

        # Normalize attention maps
        attn = attn / attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        coords = self._soft_argmax(attn)

        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = attn.unsqueeze(2)
        # pooled per-keypoint features: (B, K, C)
        feat_k = (feat_expanded * heat_expanded).sum(dim=[3, 4])

        # Preserve keypoint structure by flattening K*C instead of mean-pooling
        # This keeps relative information between eyes/mouth/eyebrows
        bsz, K, C = feat_k.shape
        feat_k_flat = feat_k.view(bsz, K * C)  # (B, K*C)

        # global pooled feature (B, C)
        global_attn = attn.mean(dim=1, keepdim=True)
        feat_global = (feat_map * global_attn).sum(dim=[2, 3])  # (B, C)

        # concat flattened per-keypoint features and global pooled -> (B, K*C + C)
        feat_k = torch.cat([feat_k_flat, feat_global], dim=1)
        # normalize fused landmark features to avoid backbone domination
        feat_k = F.normalize(feat_k, dim=1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        # No edge consistency on low-res FER (unreliable)
        edge_consistency = attn.new_tensor(0.0)
        edge_align = attn.new_tensor(0.0)

        # removed edge_conv_reg and edge_tv (to avoid toxic gradients)
        edge_conv_reg = attn.new_tensor(0.0)
        edge_tv = attn.new_tensor(0.0)

        # Peaky constraint: encourage each keypoint map to have a strong peak
        # max_val: (B,K) -> encourage values near 1.0
        max_val = attn.amax(dim=[2, 3])
        peak_loss = ((1.0 - max_val) ** 2 ).mean()

        # Keep only lightweight, non-toxic auxiliaries for low-res FER
        aux = {
            "landmark_diversity": self._diversity_loss(attn),
            # use peak-based loss instead of raw entropy to encourage sharp attention
            "landmark_entropy": peak_loss,
            # keep edge consistency (soft guidance) but avoid heavy alignment/conv/TV penalties
            "landmark_edge_consistency": edge_consistency,
            "landmark_edge_align": attn.new_tensor(0.0),
            "landmark_edge_conv_reg": attn.new_tensor(0.0),
            "landmark_edge_tv": attn.new_tensor(0.0),
        }

        return attn, coords, feat_k, aux