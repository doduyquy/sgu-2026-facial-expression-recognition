import torch
import torch.nn as nn
from .CBAM import CBAM
from .cross_fusion_pyramid import PyramidCrossFusionTransformer


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.cbam = CBAM(f3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.cbam(x)
        x = self.relu(x + shortcut)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.cbam = CBAM(f3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f3),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.cbam(x)

        x = self.relu(x + shortcut)
        return x


class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes=7,
        in_channels=1,
        use_cbam_stage34=True,
        cbam_reduction=16,
        cbam_kernel_size=7,
        use_landmark_cross_fusion=False,
        landmark_num_points=12,
        use_pyramid_multi_scale=True,
        img_topk_tokens=128,
        token_selection_mode="topk_softmax",
        use_sinusoidal_pos=True,
        pyramid_dropout_rate=0.1,
        pyramid_depth=4,
        cross_attn_dim=256,
        cross_attn_heads=8,
        use_token_conv_mix=True,
    ):
        super().__init__()
        self.use_landmark_cross_fusion = use_landmark_cross_fusion
        self.use_pyramid_multi_scale = use_pyramid_multi_scale
        self.landmark_num_points = landmark_num_points
        self.cross_attn_dim = cross_attn_dim

        candidate_heads = [cross_attn_heads, 8, 4, 2, 1]
        self.poster_heads = 1
        for h in candidate_heads:
            if h <= 0:
                continue
            if (cross_attn_dim % h == 0) and ((cross_attn_dim // 2) % h == 0) and ((cross_attn_dim // 4) % h == 0):
                self.poster_heads = h
                break

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256]),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128, 128, 512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256, 256, 1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Landmark encoder.
        self.base_lm_dim = 128
        self.lm_point_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.base_lm_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.lm_spatial_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.lm_spatial_proj = nn.Linear(self.base_lm_dim * 8 * 8, self.base_lm_dim)
        self.lm_coord_proj = nn.Linear(2, self.base_lm_dim)
        self.landmark_index_embed = nn.Embedding(landmark_num_points, self.base_lm_dim)

        # Poster-like pyramid module.
        self.pyramid_cross_fusion = PyramidCrossFusionTransformer(
            lm_base_dim=self.base_lm_dim,
            topk_tokens=img_topk_tokens,
            token_selection_mode=token_selection_mode,
            use_sinusoidal_pos=use_sinusoidal_pos,
            dropout_rate=pyramid_dropout_rate,
            pyramid_depth=pyramid_depth,
            fusion_dim=self.cross_attn_dim,
            num_heads=self.poster_heads,
        )

        self.base_feat_proj = nn.Linear(1536, self.cross_attn_dim)
        self.final_global_fusion_attn = nn.MultiheadAttention(
            embed_dim=self.cross_attn_dim,
            num_heads=self.poster_heads,
            batch_first=True,
        )
        self.final_global_norm = nn.LayerNorm(self.cross_attn_dim)
        self.final_scale_gate = nn.Sequential(
            nn.Linear(self.cross_attn_dim, self.cross_attn_dim),
            nn.Sigmoid(),
        )
        self.final_scale_score = nn.Linear(self.cross_attn_dim, 1)

        self.fusion_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.poster_mlp_head = nn.Sequential(
            nn.Linear(self.cross_attn_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _encode_landmarks_base(self, landmarks):
        if landmarks is None:
            return None

        if landmarks.dim() == 4:
            # heatmap input: (B, P, H, W)
            bsz, pnt, h, w = landmarks.shape
            lm = landmarks.view(bsz * pnt, 1, h, w)
            tokens = self.lm_point_cnn(lm)
            tokens = self.lm_spatial_pool(tokens)
            tokens = tokens.flatten(1)
            tokens = self.lm_spatial_proj(tokens)
            tokens = tokens.view(bsz, pnt, self.base_lm_dim)
        elif landmarks.dim() == 3 and landmarks.size(-1) == 2:
            tokens = self.lm_coord_proj(landmarks)
        elif landmarks.dim() == 2:
            bsz, c = landmarks.shape
            pnt = c // 2
            coords = landmarks.view(bsz, pnt, 2)
            tokens = self.lm_coord_proj(coords)
        else:
            return None

        p = tokens.size(1)
        emb_len = min(p, self.landmark_num_points)
        idx = torch.arange(emb_len, device=tokens.device)
        emb = self.landmark_index_embed(idx).unsqueeze(0)
        tokens[:, :emb_len, :] = tokens[:, :emb_len, :] + emb
        return tokens

    def forward(self, x, landmarks=None, landmark_mask=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x2 = self.layer2(x)

        x3 = self.layer3(x2)
        feat3 = torch.flatten(self.avgpool(x3), 1)

        x4 = self.layer4(x3)
        feat4 = torch.flatten(self.avgpool(x4), 1)

        base_feat = torch.cat([feat3, feat4], dim=1)

        if self.use_landmark_cross_fusion and self.use_pyramid_multi_scale and landmarks is not None:
            lm_base = self._encode_landmarks_base(landmarks)
            if lm_base is not None:
                xs, xm, xl = self.pyramid_cross_fusion(
                    x2,
                    x3,
                    x4,
                    lm_base,
                    landmark_mask=landmark_mask,
                )

                base_token = self.base_feat_proj(base_feat)
                final_tokens = torch.stack([base_token, xs, xm, xl], dim=1)
                final_attn, _ = self.final_global_fusion_attn(
                    query=final_tokens,
                    key=final_tokens,
                    value=final_tokens,
                )
                final_tokens = self.final_global_norm(final_tokens + final_attn)
                final_tokens = final_tokens * self.final_scale_gate(final_tokens)
                scale_weights = torch.softmax(self.final_scale_score(final_tokens).squeeze(-1), dim=1)
                poster_feat = (final_tokens * scale_weights.unsqueeze(-1)).sum(dim=1)
                return self.poster_mlp_head(poster_feat)

        return self.fusion_fc(base_feat)
