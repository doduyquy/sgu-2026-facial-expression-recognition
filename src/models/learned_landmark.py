import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLandmarkBranch(nn.Module):
    def __init__(
    self,
    in_channels=1024,
    landmark_num_points=6,
    landmark_tau=0.07,
    diversity_margin=0.2,
    kp_proj_dim=64,
        feature_dropout_p=0.3,
        head_dropout_p=0.1,
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
        # positional encoding: a lightweight 2-channel (x,y) coord projector
        self.use_pos_encoding = True
        try:
            self.pos_proj = nn.Conv2d(2, in_channels, kernel_size=1)
            self.pos_scale = nn.Parameter(torch.tensor(0.1))
        except Exception:
            self.use_pos_encoding = False
        # For low-res FER, remove learnable edge extractor and heavy sobel targets
        # Keep simple attention-only branch (soft regions), so do not create edge conv/bias
        self.edge_beta = None
        self.landmark_bias = None
        # per-keypoint projection to reduce C -> kp_proj_dim before flattening
        self.kp_proj_dim = int(kp_proj_dim)
        try:
            self.kp_proj = nn.Linear(in_channels, self.kp_proj_dim)
        except Exception:
            self.kp_proj = None
        # per-keypoint gating to weight keypoints by importance
        try:
            if self.kp_proj is not None:
                self.kp_gate = nn.Linear(self.kp_proj_dim, 1)
            else:
                self.kp_gate = None
        except Exception:
            self.kp_gate = None
        # optional light positional supervision: indices expected in normalized y
        # upper_idxs: keypoint indices that should stay in upper-face (y <= 0.5)
        # lower_idxs: keypoint indices that should stay in lower-face (y >= 0.5)
        # sensible defaults for FER keypoint grouping (can be overridden by config)
        self.upper_idxs = [0, 1, 2]
        self.lower_idxs = [3, 4, 5]
        self.pos_supervision_weight = 0.05

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

    def _apply_pos_encoding(self, feat_map):
        # feat_map: (B, C, H, W)
        if not getattr(self, 'use_pos_encoding', False):
            return feat_map
        bsz, C, H, W = feat_map.shape
        # normalized coords in [0,1]
        xs = torch.linspace(0, 1, W, device=feat_map.device, dtype=feat_map.dtype)
        ys = torch.linspace(0, 1, H, device=feat_map.device, dtype=feat_map.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
        pos = pos.repeat(bsz, 1, 1, 1)
        try:
            delta = self.pos_proj(pos) * torch.sigmoid(self.pos_scale)
            return feat_map + delta
        except Exception:
            return feat_map
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
        # --- Debug instrumentation per user request: check NaN and shapes ---
        try:
            try:
                if torch.isnan(attn).any():
                    print("NaN in attn")
            except Exception:
                pass
            try:
                if torch.isnan(coords).any():
                    print("NaN in coords")
            except Exception:
                pass
        except Exception:
            pass

        try:
            # sanitize coords on CPU to avoid triggering CUDA device-side asserts
            coords_cpu = coords.detach().cpu()
            if not coords_cpu.is_floating_point():
                coords_cpu = coords_cpu.float()
            if not torch.isfinite(coords_cpu).all():
                coords_cpu = torch.nan_to_num(coords_cpu, nan=0.5, posinf=1.0, neginf=0.0)
            coords_cpu = coords_cpu.clamp(0.0, 1.0)

            # promote spread term to encourage global coverage (variance of keypoints)
            try:
                center_cpu = coords_cpu.mean(dim=1, keepdim=True)  # (B,1,2)
                spread_cpu = ((coords_cpu - center_cpu) ** 2).sum(dim=-1).mean(dim=1)  # (B,)
            except Exception:
                spread_cpu = coords_cpu.new_tensor(0.0)

            # move sanitized coords back to original device for fast cdist; if cdist fails, fallback to CPU result
            coords_s = coords_cpu.to(coords.device)
            try:
                d = torch.cdist(coords_s, coords_s, p=2)
            except Exception:
                d = torch.cdist(coords_cpu, coords_cpu, p=2).to(coords.device)
            # mask out diagonal
            mask = ~torch.eye(keypoints, device=d.device, dtype=torch.bool).unsqueeze(0)
            d_masked = d[mask].view(bsz, keypoints * (keypoints - 1))
            # stronger margin-based diversity: penalize when points closer than margin
            margin = getattr(self, 'diversity_margin', 0.05)
            # use relu(margin - distance) to push points apart when too close
            loss_per_sample = F.relu(margin - d_masked).mean(dim=1)
            # encourage spread (higher spread reduces the loss)
            try:
                loss_per_sample = loss_per_sample - (0.1 * spread_cpu.to(loss_per_sample.device))
            except Exception:
                pass
            return loss_per_sample.mean()
        except Exception:
            # safe fallback: sanitize attention maps and compute fallback on CPU if GPU ops fail
            try:
                # perform sanitization on CPU to avoid CUDA asserts, then compute gram on device when safe
                flat_cpu = attn.detach().cpu().view(bsz, keypoints, -1)
                flat_cpu = flat_cpu.float()
                if not torch.isfinite(flat_cpu).all():
                    flat_cpu = torch.nan_to_num(flat_cpu, nan=0.0, posinf=1.0, neginf=0.0)
                flat_cpu = flat_cpu / flat_cpu.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                gram = torch.bmm(flat_cpu, flat_cpu.transpose(1, 2)).to(attn.device)
            except Exception:
                # last-resort safe value
                eye = torch.eye(keypoints, device=attn.device, dtype=attn.dtype).unsqueeze(0)
                return (eye - eye).pow(2).mean()
            eye = torch.eye(keypoints, device=gram.device, dtype=gram.dtype).unsqueeze(0)
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
        # add positional encoding to help landmark separation (eyes/mouth etc.)
        try:
            feat_map = self._apply_pos_encoding(feat_map)
        except Exception:
            pass

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

        # temperature schedule: softer early, sharper later to move from exploration -> exploitation
        try:
            tau = float(self.landmark_tau) * (1.0 - getattr(self, '_training_progress', 0.0) * 0.5)
        except Exception:
            tau = float(getattr(self, 'landmark_tau', 0.07))
        tau = max(tau, 1e-6)
        scaled = attn_logits.view(bsz, keypoints, -1) / tau
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

        # Preserve keypoint structure but reduce per-keypoint dims to avoid huge vectors
        bsz, K, C = feat_k.shape
        if getattr(self, 'kp_proj', None) is not None:
            # apply projection per keypoint
            feat_k_reduced = self.kp_proj(feat_k)  # (B, K, kp_proj_dim)
            feat_k_flat = feat_k_reduced.view(bsz, K * self.kp_proj_dim)
        else:
            feat_k_flat = feat_k.view(bsz, K * C)

        # global pooled feature (B, C)
        global_attn = attn.mean(dim=1, keepdim=True)
        feat_global = (feat_map * global_attn).sum(dim=[2, 3])  # (B, C)

        # Feature fusion with keypoint gating when kp_proj exists
        if getattr(self, 'kp_proj', None) is not None and 'feat_k_reduced' in locals():
            # feat_k_reduced: (B, K, D)
            try:
                if getattr(self, 'kp_gate', None) is not None:
                    gate_logits = self.kp_gate(feat_k_reduced)  # (B, K, 1)
                    gate_logits = gate_logits.squeeze(-1)
                    gate = torch.softmax(gate_logits, dim=1)  # (B, K)
                    # apply gate per-keypoint but preserve per-keypoint representation
                    feat_k_weighted = feat_k_reduced * gate.unsqueeze(-1)
                else:
                    # fallback: keep per-keypoint vectors as-is
                    feat_k_weighted = feat_k_reduced
            except Exception:
                feat_k_weighted = feat_k_reduced

            # flatten weighted per-keypoint features to (B, K*D) to match expected pre-reduction layout
            try:
                feat_k_flat = feat_k_weighted.view(bsz, K * feat_k_weighted.size(-1))
            except Exception:
                feat_k_flat = feat_k_reduced.view(bsz, K * self.kp_proj_dim)

            # concat flattened per-keypoint features and global pooled -> (B, K*kp_proj_dim + C)
            feat_k = torch.cat([feat_k_flat, feat_global], dim=1)
        else:
            # no per-keypoint projection available: fall back to concatenation
            feat_k = torch.cat([feat_k_flat, feat_global], dim=1)

        # normalize fused feature using layer-norm (preserve scale relations across dims)
        try:
            feat_k = F.layer_norm(feat_k, feat_k.shape[1:])
        except Exception:
            pass
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

        # Heatmap overlap penalty: discourage different keypoints from overlapping
        flat_maps = attn.view(bsz, keypoints, -1)
        # normalize per-map for stable dot products
        flat_norm = flat_maps / (flat_maps.norm(dim=-1, keepdim=True).clamp(min=1e-6))
        sim_m = torch.bmm(flat_norm, flat_norm.transpose(1, 2))  # (B, K, K)
        # zero out diagonal (self-similarity) and average off-diagonals across batch
        eye = torch.eye(keypoints, device=attn.device, dtype=sim_m.dtype).unsqueeze(0)
        masked = sim_m * (1.0 - eye)
        denom = float(bsz * keypoints * max(1, (keypoints - 1)))
        if denom > 0:
            overlap_loss = masked.sum() / denom
        else:
            overlap_loss = attn.new_tensor(0.0)

        # Positional supervision: lightly nudge some keypoints to expected vertical halves
        pos_sup = attn.new_tensor(0.0)
        try:
            # respect explicit opt-out: if model-level pos weight is <= 0 skip entirely
            model_pos_weight = float(getattr(self, 'pos_supervision_weight', 0.0) or 0.0)
            if model_pos_weight <= 0.0:
                # skip computing positional supervision (avoids spurious warnings)
                pos_sup = attn.new_tensor(0.0)
            else:
                if len(self.upper_idxs) > 0 or len(self.lower_idxs) > 0:
                    # coords: (B, K, 2) with y in second dim
                    ys = coords[..., 1]
                    b_k = ys.shape
                    # defensive checks to avoid CUDA device-side asserts from OOB indices
                    K_actual = ys.shape[1] if len(ys.shape) > 1 else 0

                    # If user-supplied idx lists assume a larger K (common mismatch),
                    # auto-resolve by splitting available K into upper/lower halves.
                    try:
                        if (len(self.upper_idxs) + len(self.lower_idxs)) > 0 and (
                            max(self.upper_idxs or [0]) >= K_actual or max(self.lower_idxs or [0]) >= K_actual
                        ):
                            # compute a safe split for the current K_actual
                            mid = K_actual // 2
                            new_upper = list(range(0, mid))
                            new_lower = list(range(mid, K_actual))
                            print(f"[landmark pos_sup] Info: index mismatch detected. Auto-splitting K={K_actual} -> upper={new_upper}, lower={new_lower}")
                            upper_idxs_use = new_upper
                            lower_idxs_use = new_lower
                        else:
                            upper_idxs_use = list(self.upper_idxs)
                            lower_idxs_use = list(self.lower_idxs)
                    except Exception:
                        upper_idxs_use = list(self.upper_idxs)
                        lower_idxs_use = list(self.lower_idxs)

                    # filter valid indices as final safety net
                    valid_upper = [i for i in upper_idxs_use if 0 <= i < K_actual]
                    valid_lower = [i for i in lower_idxs_use if 0 <= i < K_actual]

                    penalties = []
                    if len(valid_upper) > 0:
                        up = ys[:, valid_upper]
                        penalties.append(F.relu(up - 0.5).mean())
                    if len(valid_lower) > 0:
                        lo = ys[:, valid_lower]
                        penalties.append(F.relu(0.5 - lo).mean())

                    if len(penalties) > 0:
                        pos_sup = sum(penalties) / len(penalties)
                        pos_sup = pos_sup * float(model_pos_weight)
        except Exception:
            pos_sup = attn.new_tensor(0.0)

        # Keep only lightweight, non-toxic auxiliaries for low-res FER
        aux = {
            "landmark_diversity": self._diversity_loss(attn, coords=coords),
            # disable entropy/peak auxiliary for small noisy FER images (no GT)
            "landmark_entropy": attn.new_tensor(0.0),
            # heatmap overlap penalty
            "landmark_overlap": overlap_loss,
            # light positional supervision penalty (upper/lower guidance)
            "landmark_pos_supervision": pos_sup,
            # keep edge consistency (soft guidance) but avoid heavy alignment/conv/TV penalties
            "landmark_edge_consistency": edge_consistency,
            "landmark_edge_align": attn.new_tensor(0.0),
            "landmark_edge_conv_reg": attn.new_tensor(0.0),
            "landmark_edge_tv": attn.new_tensor(0.0),
        }

        return attn, coords, feat_k, aux