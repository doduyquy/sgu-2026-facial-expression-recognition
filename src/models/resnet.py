import torch
import torch.nn as nn
from .CBAM import CBAM
from .cross_fusion_pyramid import PyramidCrossFusionTransformer

# Input:  (B, 1, 48, 48)
# conv1: 3x3, stride=1, pad=1           -> (B, 64, 48, 48)
# pool:  2x2, stride=2                  -> (B, 64, 24, 24)
# layer2: ConvBlock(s=1) + 2 IDs        -> (B, 256, 24, 24)
# layer3: ConvBlock(s=2) + 3 IDs        -> (B, 512, 12, 12)
# layer4: ConvBlock(s=2) + 3 IDs        -> (B, 1024, 6, 6)
# avgpool: AdaptiveAvgPool2d((1,1))     -> (B, 1024, 1, 1)
# flatten                               -> (B, 1024)
# fc                                    -> (B, num_classes)
#Hout = ((Hin + 2*pad - kernel_size) // stride) + 1

class IdentityBlock(nn.Module): #giữ nguyên kích thước không gian (H x W) và số kênh,tinh chỉnh đặc trưng rồi cộng tắt (residual) với đầu vào.
    def __init__(self, in_channels, filters, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super(IdentityBlock, self).__init__()
        F1,F2,F3 = filters # F1: số kênh của conv1, F2: số kênh của conv2, F3: số kênh của conv3
        #vd 256, [64,64,256]
        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1) #256, 64, kernel_size=1
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1) #64, 64, kernel_size=3, padding=1
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1) #64, 256, kernel_size=1
        self.bn3 = nn.BatchNorm2d(F3)
        self.cbam = CBAM(F3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        self.relu = nn.ReLU()
    def forward(self, x):
        shortcut = x    
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x=self.cbam(x)

        x += shortcut
        x = self.relu(x)

        return x
    
class ConvBlock(nn.Module): #thay đổi kích thước/ số kênh đặc trưng,đồng thời chiếu nhánh tắt (shortcut) để khớp kích thước trước khi cộng residual.
    def __init__(self, in_channels, filters, stride=2, use_cbam=False, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        F1, F2, F3 = filters

        self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1, F2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.cbam = CBAM(F3, reduction=cbam_reduction, kernel_size=cbam_kernel_size) if use_cbam else nn.Identity()

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, F3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(F3)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.cbam(x)

        x += shortcut
        x = self.relu(x)

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
        cross_attn_dim=256,
        cross_attn_heads=8,
        use_pyramid_multi_scale=True,
        img_topk_tokens=128,
        token_selection_mode="topk_softmax",
        use_sinusoidal_pos=True,
        use_geometry_encoding=True,
        use_geometry_angle=False,
        use_relative_pos_bias=True,
        align_spread=0.03,
        pyramid_dropout_rate=0.1,
    ):
        super().__init__()
        self.use_landmark_cross_fusion = use_landmark_cross_fusion
        self.use_pyramid_multi_scale = use_pyramid_multi_scale
        self.landmark_num_points = landmark_num_points
        self.cross_attn_dim = cross_attn_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        # Stage 2
        self.layer2 = nn.Sequential(
            ConvBlock(64, [64,64,256], stride=1),
            IdentityBlock(256, [64,64,256]),
            IdentityBlock(256, [64,64,256])
        )

        # Stage 3
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128,128,512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128,128,512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128,128,512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(512, [128,128,512], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size)
        )
        # Stage 4
        self.layer4 = nn.Sequential(
            ConvBlock(512, [256,256,1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256,256,1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),    
            IdentityBlock(1024, [256,256,1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size),
            IdentityBlock(1024, [256,256,1024], use_cbam=use_cbam_stage34, cbam_reduction=cbam_reduction, cbam_kernel_size=cbam_kernel_size)
        )
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
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

        self.pyramid_cross_fusion = PyramidCrossFusionTransformer(
            lm_base_dim=self.base_lm_dim,
            topk_tokens=img_topk_tokens,
            token_selection_mode=token_selection_mode,
            use_sinusoidal_pos=use_sinusoidal_pos,
            use_geometry_encoding=use_geometry_encoding,
            use_geometry_angle=use_geometry_angle,
            use_relative_pos_bias=use_relative_pos_bias,
            align_spread=align_spread,
            dropout_rate=pyramid_dropout_rate,
        )

        self.base_feat_proj = nn.Linear(1536, 256)
        self.final_global_fusion_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True,
        )
        self.final_global_norm = nn.LayerNorm(256)
        self.final_scale_gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )
        self.final_scale_score = nn.Linear(256, 1)

        fusion_dim = 1536
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.poster_mlp_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _encode_landmarks_base(self, landmarks):
        if landmarks is None:
            return None

        if landmarks.dim() == 4:
            # Heatmap representation: (B, P, H, W) -> (B, P, D)
            b, p, h, w = landmarks.shape
            lm = landmarks.view(b * p, 1, h, w)
            tokens = self.lm_point_cnn(lm)
            tokens = self.lm_spatial_pool(tokens)
            tokens = tokens.flatten(1)
            tokens = self.lm_spatial_proj(tokens)
            tokens = tokens.view(b, p, self.base_lm_dim)
        elif landmarks.dim() == 3 and landmarks.size(-1) == 2:
            # Coordinate representation: (B, P, 2) -> (B, P, D)
            tokens = self.lm_coord_proj(landmarks)
        elif landmarks.dim() == 2:
            # Flattened coordinates: (B, 2P) -> (B, P, D)
            b, c = landmarks.shape
            p = c // 2
            coords = landmarks.view(b, p, 2)
            tokens = self.lm_coord_proj(coords)
        else:
            return None

        p = tokens.size(1)
        emb_len = min(p, self.landmark_num_points)
        idx = torch.arange(emb_len, device=tokens.device)
        emb = self.landmark_index_embed(idx).unsqueeze(0)
        tokens[:, :emb_len, :] = tokens[:, :emb_len, :] + emb
        return tokens

    def _extract_landmark_points(self, landmarks):
        if landmarks is None:
            return None

        if landmarks.dim() == 3 and landmarks.size(-1) == 2:
            return landmarks

        if landmarks.dim() == 2:
            b, c = landmarks.shape
            p = c // 2
            return landmarks.view(b, p, 2)

        if landmarks.dim() == 4:
            # Heatmaps: (B, P, H, W) -> argmax points (B, P, 2) normalized to [0,1].
            b, p, h, w = landmarks.shape
            flat = landmarks.view(b, p, -1)
            idx = torch.argmax(flat, dim=-1)
            py = idx // w
            px = idx % w
            px = px.float() / max(w - 1, 1)
            py = py.float() / max(h - 1, 1)
            return torch.stack([px, py], dim=-1)

        return None

    def forward(self, x, landmarks=None, landmark_mask=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)
        x2 = x

        x = self.layer3(x)
        x3 = x
        feat3 = self.avgpool(x)
        feat3 = torch.flatten(feat3, 1)   # (B, 512)

        x = self.layer4(x)
        x4 = x
        feat4 = self.avgpool(x)
        feat4 = torch.flatten(feat4, 1)   # (B, 1024)

        # Base concat from multi-scale image features.
        base_feat = torch.cat([feat3, feat4], dim=1)  # (B, 1536)

        if self.use_landmark_cross_fusion and self.use_pyramid_multi_scale and landmarks is not None:
            lm_base = self._encode_landmarks_base(landmarks)
            lm_points = self._extract_landmark_points(landmarks)
            if lm_base is not None:
                x_fuse_small, x_fuse_medium, x_fuse_large = self.pyramid_cross_fusion(
                    x2,
                    x3,
                    x4,
                    lm_base,
                    landmark_mask,
                    lm_points,
                )

                base_token = self.base_feat_proj(base_feat)
                final_tokens = torch.stack([base_token, x_fuse_small, x_fuse_medium, x_fuse_large], dim=1)
                final_tokens_attn, _ = self.final_global_fusion_attn(
                    query=final_tokens,
                    key=final_tokens,
                    value=final_tokens,
                )
                final_tokens = self.final_global_norm(final_tokens + final_tokens_attn)
                final_tokens = final_tokens * self.final_scale_gate(final_tokens)
                scale_weights = torch.softmax(self.final_scale_score(final_tokens).squeeze(-1), dim=1)
                poster_feat = (final_tokens * scale_weights.unsqueeze(-1)).sum(dim=1)
                return self.poster_mlp_head(poster_feat)

        out = self.fusion_fc(base_feat)

        return out