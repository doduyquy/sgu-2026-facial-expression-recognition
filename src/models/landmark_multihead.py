import torch
import torch.nn as nn
import torch.nn.functional as F

from .learned_landmark import LearnedLandmarkBranch


class MultiHeadLandmarkBranch(LearnedLandmarkBranch):
    """Multi-head wrapper around LearnedLandmarkBranch heatmap logic.

    Produces `landmark_num_points` maps by averaging per-head maps.
    This keeps the existing priors/regularizers while adding head diversity.
    """
    def __init__(
        self,
        in_channels=1024,
        landmark_num_points=6,
        num_heads=4,
        landmark_tau=0.07,
        feature_dropout_p=0.3,
        kp_proj_dim=64,
        head_dropout_p=0.1,
        edge_guidance_beta=1.0,
        edge_alpha=6.0,
    ):
        super().__init__(
            in_channels=in_channels,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            kp_proj_dim=kp_proj_dim,
            feature_dropout_p=feature_dropout_p,
            head_dropout_p=head_dropout_p,
            edge_guidance_beta=edge_guidance_beta,
            edge_alpha=edge_alpha,
        )

        self.num_heads = int(max(1, num_heads))
        # learnable head weights so model can prefer certain heads (not plain averaging)
        self.head_weight = nn.Parameter(torch.ones(self.num_heads))
        # replace single-head heatmap with multi-head heatmap conv
        self.landmark_heatmap_head = nn.Conv2d(
            in_channels, landmark_num_points * self.num_heads, kernel_size=1
        )

    def forward(self, feat_map, input_image=None):
        # apply positional encoding to feature map to help separate facial regions
        try:
            feat_map = self._apply_pos_encoding(feat_map)
        except Exception:
            pass

        # attn_logits_heads: (B, num_heads*K, H, W) -> (B, num_heads, K, H, W)
        attn_logits_heads = self.landmark_heatmap_head(feat_map)
        bsz, hk, h, w = attn_logits_heads.shape
        K = self.landmark_num_points
        nh = self.num_heads
        attn_logits_heads = attn_logits_heads.view(bsz, nh, K, h, w)

        # Edge guidance removed for low-res FER; parent methods will return None
        edge_attn = None
        sobel_target = None

        # Per-head processing: center logits per-head to encourage competition
        attn_logits_heads = attn_logits_heads - attn_logits_heads.mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)

        # If edge guidance available, add it to each head's logits
        if edge_attn is not None:
            beta = (self.edge_beta.view(1, K, 1, 1) * self.current_edge_weight).to(attn_logits_heads.dtype)
            bias = self.landmark_bias.view(1, K, 1, 1).to(attn_logits_heads.dtype)
            # expand to heads: (1, nh, K, 1, 1)
            beta_h = beta.unsqueeze(1)
            bias_h = bias.unsqueeze(1)
            edge_h = edge_attn.unsqueeze(1)
            attn_logits_heads = attn_logits_heads + beta_h * edge_h + bias_h

        # softmax per-head & per-keypoint over spatial dims
        flat = attn_logits_heads.view(bsz, nh, K, -1) / max(self.landmark_tau, 1e-6)
        attn_heads = torch.softmax(flat, dim=-1).view(bsz, nh, K, h, w)

        # head dropout: enable conservatively only late in training
        effective_head_p = 0.0
        try:
            if self.training and K > 1:
                if getattr(self, '_training_progress', 0.0) >= 0.7:
                    effective_head_p = float(self.head_dropout_p)
        except Exception:
            effective_head_p = 0.0

        if effective_head_p > 0.0:
            keep = (torch.rand(bsz, nh, K, 1, 1, device=attn_heads.device) > effective_head_p).to(attn_heads.dtype)
            has_any = keep.sum(dim=2, keepdim=True) > 0
            keep = torch.where(has_any, keep, torch.ones_like(keep))
            attn_heads = attn_heads * keep

        # Inter-head similarity penalty: encourage heads to be different
        try:
            flat_heads = attn_heads.view(bsz, nh, K, -1)  # (B, nh, K, D)
            # normalize over spatial dim
            flat_norm = flat_heads / (flat_heads.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            # move keypoint dim forward: (B, K, nh, D)
            flat_perm = flat_norm.permute(0, 2, 1, 3)
            # sim: (B, K, nh, nh)
            sim = torch.matmul(flat_perm, flat_perm.transpose(-1, -2))
            # remove diagonal similarity
            with torch.no_grad():
                eye_h = torch.eye(nh, device=sim.device, dtype=sim.dtype).view(1, 1, nh, nh)
            sim_off = sim * (1.0 - eye_h)
            inter_head_loss = (sim_off.pow(2)).sum() / max(sim_off.numel(), 1)
        except Exception:
            inter_head_loss = attn_heads.new_tensor(0.0)

        # aggregate heads with learned weights (softmax over heads)
        try:
            w = torch.softmax(self.head_weight, dim=0)
            attn = (attn_heads * w.view(1, nh, 1, 1, 1)).sum(dim=1)
        except Exception:
            attn = attn_heads.mean(dim=1)

        # normalize aggregated attention maps
        attn = attn / attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        # coordinates via soft-argmax on aggregated maps
        coords = self._soft_argmax(attn)

        # pooled per-keypoint features
        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = attn.unsqueeze(2)
        # pooled per-keypoint features: (B, K, C)
        feat_k = (feat_expanded * heat_expanded).sum(dim=[3, 4])

        # reduce per-keypoint dim if projection available to avoid massive flatten
        bsz, Kp, C = feat_k.shape
        if getattr(self, 'kp_proj', None) is not None:
            feat_k_reduced = self.kp_proj(feat_k)  # (B, K, kp_proj_dim)
            feat_k_flat = feat_k_reduced.view(bsz, Kp * self.kp_proj_dim)
        else:
            feat_k_flat = feat_k.view(bsz, Kp * C)

        global_attn = attn.mean(dim=1, keepdim=True)
        feat_global = (feat_map * global_attn).sum(dim=[2, 3])

        feat_k = torch.cat([feat_k_flat, feat_global], dim=1)
        feat_k = F.normalize(feat_k, dim=1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        # No edge guidance for low-res FER
        edge_consistency = attn.new_tensor(0.0)
        edge_align = attn.new_tensor(0.0)

        # Reduce toxic regularizers for low-res FER: keep only soft edge_consistency
        edge_conv_reg = attn.new_tensor(0.0)
        edge_tv = attn.new_tensor(0.0)

        # peaky loss
        max_val = attn.amax(dim=[2, 3])
        peak_loss = ((1.0 - max_val) ** 2 ).mean()

        # normalize pooled features and return compact aux set
        feat_k = F.normalize(feat_k, dim=1)
        aux = {
            "landmark_diversity": self._diversity_loss(attn, coords=coords),
            "landmark_entropy": peak_loss,
            "landmark_inter_head_similarity": inter_head_loss,
            "landmark_edge_consistency": edge_consistency,
            "landmark_edge_align": attn.new_tensor(0.0),
            "landmark_edge_conv_reg": edge_conv_reg,
            "landmark_edge_tv": edge_tv,
        }

        return attn, coords, feat_k, aux
