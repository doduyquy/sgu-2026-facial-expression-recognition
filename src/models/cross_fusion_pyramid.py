import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CrossFusionScaleBlock(nn.Module):
    """One cross-fusion block for a single scale.

    Image stream:  Attention(img) = Softmax(Q_lm K_img^T) V_img
    Landmark stream: Attention(lm) = Softmax(Q_img K_lm^T) V_lm
    """

    def __init__(
        self,
        img_in_channels,
        lm_base_dim,
        attn_dim,
        num_heads,
        out_dim=256,
        base_hw=(6, 6),
        topk_tokens=128,
        token_selection_mode="topk_softmax",
        use_sinusoidal_pos=True,
        use_geometry_encoding=True,
        use_geometry_angle=False,
        use_relative_pos_bias=True,
        align_spread=0.03,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.img_proj = nn.Conv2d(img_in_channels, attn_dim, kernel_size=1)
        self.lm_proj = nn.Linear(lm_base_dim, attn_dim)
        self.topk_tokens = topk_tokens
        self.token_selection_mode = token_selection_mode
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.use_geometry_encoding = use_geometry_encoding
        self.use_geometry_angle = use_geometry_angle
        self.use_relative_pos_bias = use_relative_pos_bias
        self.base_hw = base_hw
        self.dropout = nn.Dropout(dropout_rate)
        self.mask_token_bias = nn.Parameter(torch.tensor(-5.0))
        self.self_attn_heads = num_heads
        self.align_spread = align_spread
        self.temperature = nn.Parameter(torch.tensor(1.0))

        geom_in_dim = 4 if use_geometry_angle else 3
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_in_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, attn_dim),
        )
        self.geom_score_mlp = nn.Linear(attn_dim, 1)
        self.align_norm = nn.LayerNorm(attn_dim)
        self.rel_pos_scale = nn.Parameter(torch.tensor(1.0))

        # Landmark self-attention lets the model learn geometry consistency.
        self.lm_self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.lm_self_attn_2 = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cross_lm_to_img = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_img_to_lm = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Attention pooling replaces plain mean to keep token importance.
        self.lm_pool = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.img_pool = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.lm_pool_query = nn.Parameter(torch.randn(1, 1, attn_dim))
        self.img_pool_query = nn.Parameter(torch.randn(1, 1, attn_dim))

        self.lm_norm = nn.LayerNorm(attn_dim)
        self.lm_self_norm_2 = nn.LayerNorm(attn_dim)
        self.lm_self_ffn_norm = nn.LayerNorm(attn_dim)
        self.lm_cross_norm = nn.LayerNorm(attn_dim)
        self.lm_cross_ffn_norm = nn.LayerNorm(attn_dim)
        self.img_norm = nn.LayerNorm(attn_dim)
        self.img_cross_norm = nn.LayerNorm(attn_dim)
        self.img_cross_ffn_norm = nn.LayerNorm(attn_dim)

        self.lm_self_ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.ReLU(),
            nn.Linear(attn_dim * 4, attn_dim),
        )
        self.lm_cross_ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.ReLU(),
            nn.Linear(attn_dim * 4, attn_dim),
        )
        self.img_cross_ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.ReLU(),
            nn.Linear(attn_dim * 4, attn_dim),
        )
        self.out_proj = nn.Linear(attn_dim * 2, out_dim)

    @staticmethod
    def _build_key_padding_mask(mask, token_count):
        if mask is None:
            return None, None

        m = mask[:, :token_count] <= 0.5
        all_masked = m.all(dim=1)
        if all_masked.any():
            m[all_masked, 0] = False
        return m, all_masked

    @staticmethod
    def _build_img_key_padding_mask(batch_size, seq_len, device):
        return torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    def _apply_temperature(self, q):
        temp = self.temperature.clamp(0.1, 10.0)
        return q * temp

    @staticmethod
    def _build_2d_sinusoidal_pos_embed(h, w, dim, device, dtype):
        if dim < 4:
            return torch.zeros((1, h * w, dim), device=device, dtype=dtype)

        quarter = max(dim // 4, 1)
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()

        omega = torch.arange(quarter, device=device).float()
        omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))

        x_proj = torch.einsum("n,d->nd", x, omega)
        y_proj = torch.einsum("n,d->nd", y, omega)

        pos = torch.cat([torch.sin(x_proj), torch.cos(x_proj), torch.sin(y_proj), torch.cos(y_proj)], dim=1)
        if pos.size(1) < dim:
            pad = torch.zeros((pos.size(0), dim - pos.size(1)), device=device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=1)
        elif pos.size(1) > dim:
            pos = pos[:, :dim]

        return pos.unsqueeze(0).to(dtype=dtype)

    def _add_position(self, img_tokens, img_map):
        if not self.use_sinusoidal_pos:
            return img_tokens
        h, w = img_map.shape[-2], img_map.shape[-1]
        pos_tokens = self._build_2d_sinusoidal_pos_embed(
            h=h,
            w=w,
            dim=img_tokens.size(-1),
            device=img_tokens.device,
            dtype=img_tokens.dtype,
        )
        return img_tokens + pos_tokens

    def _select_topk_tokens(self, img_tokens):
        if self.token_selection_mode == "softmax":
            scores = img_tokens.norm(dim=-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return img_tokens * weights

        if self.token_selection_mode == "sigmoid":
            scores = img_tokens.norm(dim=-1)
            weights = torch.sigmoid(scores).unsqueeze(-1)
            return img_tokens * weights

        if self.token_selection_mode == "topk_softmax":
            if self.topk_tokens is None or self.topk_tokens <= 0:
                return img_tokens
            seq_len = img_tokens.size(1)
            if seq_len <= self.topk_tokens:
                return img_tokens

            scores = img_tokens.norm(dim=-1)
            topk_scores, topk_idx = torch.topk(scores, k=self.topk_tokens, dim=1)
            gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, img_tokens.size(-1))
            selected = torch.gather(img_tokens, dim=1, index=gather_idx)
            weights = torch.softmax(topk_scores, dim=1).unsqueeze(-1)
            return selected * weights

        if self.topk_tokens is None or self.topk_tokens <= 0:
            return img_tokens
        seq_len = img_tokens.size(1)
        if seq_len <= self.topk_tokens:
            return img_tokens

        scores = img_tokens.norm(dim=-1)
        topk_idx = torch.topk(scores, k=self.topk_tokens, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, img_tokens.size(-1))
        return torch.gather(img_tokens, dim=1, index=gather_idx)

    @staticmethod
    def _normalize_landmark_points(landmark_points, landmark_mask):
        pts = landmark_points
        if landmark_mask is None:
            center = pts.mean(dim=1, keepdim=True)
            diff = pts - center
            scale = diff.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp(min=1e-6)
            return diff / scale

        valid = (landmark_mask > 0.5).float().unsqueeze(-1)
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        center = (pts * valid).sum(dim=1, keepdim=True) / count
        diff = (pts - center) * valid
        scale = torch.sqrt((diff.pow(2).sum(dim=(1, 2), keepdim=True) / (count * 2.0)).clamp(min=1e-6))
        return diff / scale

    def _build_geometry_features(self, landmark_points, landmark_mask):
        # landmark_points: (B, P, 2)
        pts = self._normalize_landmark_points(landmark_points, landmark_mask)
        pi = pts.unsqueeze(2)
        pj = pts.unsqueeze(1)
        rel = pi - pj
        dist = torch.norm(rel, dim=-1, keepdim=True)

        if self.use_geometry_angle:
            # Triplet-aware angle feature: average_k cos((p_i-p_j), (p_k-p_j)).
            rel_norm = rel / (torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6))
            cos_triplet = torch.einsum("bijc,bkjc->bijk", rel_norm, rel_norm)
            if landmark_mask is None:
                angle = cos_triplet.mean(dim=3, keepdim=True)
            else:
                valid = (landmark_mask > 0.5).float()
                k_mask = valid.unsqueeze(1).unsqueeze(1)
                angle_num = (cos_triplet * k_mask).sum(dim=3, keepdim=True)
                angle_den = k_mask.sum(dim=3, keepdim=True).clamp(min=1e-6)
                angle = angle_num / angle_den
            geom_input = torch.cat([rel, dist, angle], dim=-1)
        else:
            geom_input = torch.cat([rel, dist], dim=-1)
        geom_pair = self.geom_mlp(geom_input)

        geom_scores = self.geom_score_mlp(geom_pair).squeeze(-1)

        if landmark_mask is None:
            geom_attn = torch.softmax(geom_scores, dim=2)
            return (geom_attn.unsqueeze(-1) * geom_pair).sum(dim=2)

        valid = (landmark_mask > 0.5).float()
        pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1)
        geom_scores = geom_scores.masked_fill(pair_mask <= 0, -1e4)
        geom_attn = torch.softmax(geom_scores, dim=2)
        geom_attn = geom_attn * pair_mask
        denom = geom_attn.sum(dim=2, keepdim=True).clamp(min=1e-6)
        geom_attn = geom_attn / denom
        return (geom_attn.unsqueeze(-1) * geom_pair).sum(dim=2)

    def _build_rel_pos_attn_mask(self, landmark_points, landmark_mask):
        if (not self.use_relative_pos_bias) or landmark_points is None:
            return None

        rel_dist = torch.cdist(landmark_points, landmark_points, p=2)
        bias = -self.rel_pos_scale * rel_dist

        if landmark_mask is not None:
            pair_mask = (landmark_mask > 0.5).unsqueeze(2) * (landmark_mask > 0.5).unsqueeze(1)
            bias = bias.masked_fill(~pair_mask, -1e4)

        # MultiheadAttention expects (B * num_heads, L, S) for per-batch bias.
        return bias.repeat_interleave(self.self_attn_heads, dim=0)

    def _sample_aligned_img_features(self, img_feat_map, landmark_points, landmark_mask):
        # img_feat_map: (B, D, H, W), landmark_points: (B, P, 2) in [0,1]
        if landmark_points is None:
            return None

        offsets = torch.tensor(
            [
                [0.0, 0.0],
                [self.align_spread, 0.0],
                [-self.align_spread, 0.0],
                [0.0, self.align_spread],
                [0.0, -self.align_spread],
            ],
            device=img_feat_map.device,
            dtype=landmark_points.dtype,
        )
        offset_weights = torch.tensor([1.0, 0.6, 0.6, 0.6, 0.6], device=img_feat_map.device, dtype=landmark_points.dtype)
        offset_weights = offset_weights / offset_weights.sum()

        samples = []
        for k in range(offsets.size(0)):
            pts = landmark_points + offsets[k].view(1, 1, 2)
            pts = pts.clamp(0.0, 1.0)
            gx = (pts[..., 0] * 2.0) - 1.0
            gy = (pts[..., 1] * 2.0) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B, P, 1, 2)
            sampled_k = F.grid_sample(
                img_feat_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled_k = sampled_k.squeeze(-1).transpose(1, 2)  # (B, P, D)
            samples.append(sampled_k * offset_weights[k])

        sampled = torch.stack(samples, dim=0).sum(dim=0)

        if landmark_mask is not None:
            sampled = sampled * landmark_mask.unsqueeze(-1)
        return sampled

    def forward(self, img_map, lm_base_tokens, landmark_mask=None, landmark_points=None):
        lm_tokens = self.lm_proj(lm_base_tokens)
        img_map_proj = self.img_proj(img_map)

        if landmark_mask is not None:
            lm_tokens = lm_tokens * landmark_mask.unsqueeze(-1)
            lm_tokens = lm_tokens + (1.0 - landmark_mask).unsqueeze(-1) * self.mask_token_bias

        if self.use_geometry_encoding and landmark_points is not None:
            geom_feat = self._build_geometry_features(landmark_points, landmark_mask)
            lm_tokens = lm_tokens + geom_feat

        aligned_img_feat = self._sample_aligned_img_features(img_map_proj, landmark_points, landmark_mask)
        if aligned_img_feat is not None:
            lm_tokens = self.align_norm(lm_tokens + aligned_img_feat)

        token_count = lm_tokens.size(1)
        lm_kpm, all_missing = self._build_key_padding_mask(landmark_mask, token_count)
        lm_attn_mask = self._build_rel_pos_attn_mask(landmark_points, landmark_mask)

        lm_self, _ = self.lm_self_attn(
            query=self._apply_temperature(lm_tokens),
            key=lm_tokens,
            value=lm_tokens,
            attn_mask=lm_attn_mask,
            key_padding_mask=lm_kpm,
        )
        lm_tokens = self.lm_norm(lm_tokens + self.dropout(lm_self))

        lm_self_2, _ = self.lm_self_attn_2(
            query=self._apply_temperature(lm_tokens),
            key=lm_tokens,
            value=lm_tokens,
            attn_mask=lm_attn_mask,
            key_padding_mask=lm_kpm,
        )
        lm_tokens = self.lm_self_norm_2(lm_tokens + self.dropout(lm_self_2))
        lm_tokens = self.lm_self_ffn_norm(lm_tokens + self.dropout(self.lm_self_ffn(lm_tokens)))

        if landmark_mask is not None:
            lm_tokens = lm_tokens * landmark_mask.unsqueeze(-1)

        img_tokens = img_map_proj.flatten(2).transpose(1, 2)
        img_tokens = self._add_position(img_tokens, img_map)
        img_tokens = self._select_topk_tokens(img_tokens)

        img_kpm = self._build_img_key_padding_mask(
            batch_size=img_tokens.size(0),
            seq_len=img_tokens.size(1),
            device=img_tokens.device,
        )

        lm_query_img_ctx, _ = self.cross_lm_to_img(
            query=self._apply_temperature(lm_tokens),
            key=img_tokens,
            value=img_tokens,
        )
        lm_query_img_ctx = self.dropout(lm_query_img_ctx)
        if landmark_mask is not None:
            lm_query_img_ctx = lm_query_img_ctx * landmark_mask.unsqueeze(-1)
        lm_tokens = self.lm_cross_norm(lm_tokens + lm_query_img_ctx)
        lm_tokens = self.lm_cross_ffn_norm(lm_tokens + self.dropout(self.lm_cross_ffn(lm_tokens)))
        if landmark_mask is not None:
            lm_tokens = lm_tokens * landmark_mask.unsqueeze(-1)

        img_query_lm_ctx, _ = self.cross_img_to_lm(
            query=self._apply_temperature(img_tokens),
            key=lm_tokens,
            value=lm_tokens,
            key_padding_mask=lm_kpm,
        )
        img_query_lm_ctx = self.dropout(img_query_lm_ctx)
        img_tokens = self.img_cross_norm(img_tokens + img_query_lm_ctx)
        img_tokens = self.img_cross_ffn_norm(img_tokens + self.dropout(self.img_cross_ffn(img_tokens)))

        lm_query = self.lm_pool_query.expand(lm_query_img_ctx.size(0), -1, -1)
        img_query = self.img_pool_query.expand(img_query_lm_ctx.size(0), -1, -1)

        lm_ctx, _ = self.lm_pool(
            query=self._apply_temperature(lm_query),
            key=lm_tokens,
            value=lm_tokens,
            key_padding_mask=lm_kpm,
        )
        img_ctx, _ = self.img_pool(
            query=self._apply_temperature(img_query),
            key=img_tokens,
            value=img_tokens,
            key_padding_mask=img_kpm,
        )

        lm_ctx = lm_ctx.squeeze(1)
        img_ctx = self.img_norm(img_ctx).squeeze(1)
        if all_missing is not None and all_missing.any():
            img_ctx[all_missing] = 0.0

        return self.out_proj(torch.cat([lm_ctx, img_ctx], dim=1))


class PyramidCrossFusionTransformer(nn.Module):
    """Three-scale cross-fusion transformer wrapper (small/medium/large)."""

    def __init__(
        self,
        lm_base_dim=128,
        topk_tokens=128,
        use_sinusoidal_pos=True,
        use_geometry_encoding=True,
        use_geometry_angle=False,
        use_relative_pos_bias=True,
        token_selection_mode="topk_softmax",
        align_spread=0.03,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.small = CrossFusionScaleBlock(
            img_in_channels=256,
            lm_base_dim=lm_base_dim,
            attn_dim=128,
            num_heads=4,
            out_dim=256,
            base_hw=(24, 24),
            topk_tokens=topk_tokens,
            token_selection_mode=token_selection_mode,
            use_sinusoidal_pos=use_sinusoidal_pos,
            use_geometry_encoding=use_geometry_encoding,
            use_geometry_angle=use_geometry_angle,
            use_relative_pos_bias=use_relative_pos_bias,
            align_spread=align_spread,
            dropout_rate=dropout_rate,
        )
        self.medium = CrossFusionScaleBlock(
            img_in_channels=512,
            lm_base_dim=lm_base_dim,
            attn_dim=256,
            num_heads=8,
            out_dim=256,
            base_hw=(12, 12),
            topk_tokens=topk_tokens,
            token_selection_mode=token_selection_mode,
            use_sinusoidal_pos=use_sinusoidal_pos,
            use_geometry_encoding=use_geometry_encoding,
            use_geometry_angle=use_geometry_angle,
            use_relative_pos_bias=use_relative_pos_bias,
            align_spread=align_spread,
            dropout_rate=dropout_rate,
        )
        self.large = CrossFusionScaleBlock(
            img_in_channels=1024,
            lm_base_dim=lm_base_dim,
            attn_dim=512,
            num_heads=8,
            out_dim=256,
            base_hw=(6, 6),
            topk_tokens=topk_tokens,
            token_selection_mode=token_selection_mode,
            use_sinusoidal_pos=use_sinusoidal_pos,
            use_geometry_encoding=use_geometry_encoding,
            use_geometry_angle=use_geometry_angle,
            use_relative_pos_bias=use_relative_pos_bias,
            align_spread=align_spread,
            dropout_rate=dropout_rate,
        )

        # Final inter-scale interaction attention.
        self.final_fusion_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(256)
        self.final_dropout = nn.Dropout(dropout_rate)

    def forward(self, x2, x3, x4, lm_base_tokens, landmark_mask=None, landmark_points=None):
        x_fuse_small = self.small(x2, lm_base_tokens, landmark_mask, landmark_points)
        x_fuse_medium = self.medium(x3, lm_base_tokens, landmark_mask, landmark_points)
        x_fuse_large = self.large(x4, lm_base_tokens, landmark_mask, landmark_points)

        final_tokens = torch.stack([x_fuse_small, x_fuse_medium, x_fuse_large], dim=1)
        final_tokens_attn, _ = self.final_fusion_attn(
            query=final_tokens,
            key=final_tokens,
            value=final_tokens,
        )
        final_tokens = self.final_norm(final_tokens + self.final_dropout(final_tokens_attn))
        return final_tokens[:, 0, :], final_tokens[:, 1, :], final_tokens[:, 2, :]
