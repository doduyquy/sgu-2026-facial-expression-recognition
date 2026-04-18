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
        head_dropout_p=0.2,
        edge_guidance_beta=1.0,
        edge_alpha=6.0,
    ):
        super().__init__(
            in_channels=in_channels,
            landmark_num_points=landmark_num_points,
            landmark_tau=landmark_tau,
            feature_dropout_p=feature_dropout_p,
            head_dropout_p=head_dropout_p,
            edge_guidance_beta=edge_guidance_beta,
            edge_alpha=edge_alpha,
        )

        self.num_heads = int(max(1, num_heads))
        # replace single-head heatmap with multi-head heatmap conv
        self.landmark_heatmap_head = nn.Conv2d(
            in_channels, landmark_num_points * self.num_heads, kernel_size=1
        )

    def forward(self, feat_map, input_image=None):
        # attn_logits_heads: (B, num_heads*K, H, W) -> (B, num_heads, K, H, W)
        attn_logits_heads = self.landmark_heatmap_head(feat_map)
        bsz, hk, h, w = attn_logits_heads.shape
        K = self.landmark_num_points
        nh = self.num_heads
        attn_logits_heads = attn_logits_heads.view(bsz, nh, K, h, w)

        # Reuse edge guidance from parent
        edge_attn = self._build_edge_attention(input_image, h, w)
        sobel_target = self._build_sobel_target(input_image, h, w)

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

        # head dropout (randomly drop entire heads per keypoint during training)
        if self.training and self.head_dropout_p > 0.0 and K > 1:
            keep = (torch.rand(bsz, nh, K, 1, 1, device=attn_heads.device) > self.head_dropout_p).to(attn_heads.dtype)
            has_any = keep.sum(dim=2, keepdim=True) > 0
            keep = torch.where(has_any, keep, torch.ones_like(keep))
            attn_heads = attn_heads * keep

        # aggregate heads by simple average (could be learned later)
        attn = attn_heads.mean(dim=1)

        # normalize aggregated attention maps
        attn = attn / attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        # coordinates via soft-argmax on aggregated maps
        coords = self._soft_argmax(attn)

        # pooled per-keypoint features
        feat_expanded = feat_map.unsqueeze(1)
        heat_expanded = attn.unsqueeze(2)
        feat_k = (feat_expanded * heat_expanded).sum(dim=[3, 4])

        global_attn = attn.mean(dim=1, keepdim=True)
        feat_global = (feat_map * global_attn).sum(dim=[2, 3])

        feat_k = feat_k.view(bsz, -1)
        feat_k = torch.cat([feat_k, feat_global], dim=1)
        feat_k = F.dropout(feat_k, p=self.feature_dropout_p, training=self.training)

        # Edge consistency computed on aggregated maps
        if sobel_target is not None:
            sobel_k = sobel_target.repeat(1, K, 1, 1)
            edge_consistency = (attn * sobel_k).mean()
            edge_align = ((attn.mean(dim=1, keepdim=True) - sobel_target) ** 2).mean()
        else:
            edge_consistency = attn.new_tensor(0.0)
            edge_align = attn.new_tensor(0.0)

        # Regularizers: keep original edge_conv_reg and TV from parent
        with torch.no_grad():
            fixed = torch.sqrt(self._fixed_sobel_x.pow(2) + self._fixed_sobel_y.pow(2))
        fixed_rep = fixed.repeat(K, 1, 1, 1).to(self.edge_conv.weight.device)
        edge_conv_reg = F.mse_loss(self.edge_conv.weight, fixed_rep, reduction="mean")

        if edge_attn is not None:
            e = edge_attn
            tv_h = torch.abs(e[:, :, 1:, :] - e[:, :, :-1, :]).mean()
            tv_w = torch.abs(e[:, :, :, 1:] - e[:, :, :, :-1]).mean()
            edge_tv = (tv_h + tv_w) * 0.5
        else:
            edge_tv = attn.new_tensor(0.0)

        # peaky loss
        max_val = attn.amax(dim=[2, 3])
        peak_loss = (1.0 - max_val).mean()

        aux = {
            "landmark_diversity": self._diversity_loss(attn, coords=coords),
            "landmark_entropy": peak_loss,
            "landmark_edge_align": edge_align,
            "landmark_edge_consistency": edge_consistency,
            "landmark_edge_conv_reg": edge_conv_reg,
            "landmark_edge_tv": edge_tv,
        }

        return attn, coords, feat_k, aux
