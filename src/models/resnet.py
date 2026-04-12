import torch
import torch.nn as nn
from .CBAM import CBAM

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
    ):
        super().__init__()
        self.use_landmark_cross_fusion = use_landmark_cross_fusion
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
        self.img_token_proj = nn.Conv2d(1024, cross_attn_dim, kernel_size=1)

        self.lm_point_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cross_attn_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lm_coord_proj = nn.Linear(2, cross_attn_dim)
        self.landmark_index_embed = nn.Embedding(landmark_num_points, cross_attn_dim)

        self.cross_lm_to_img = nn.MultiheadAttention(
            embed_dim=cross_attn_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
        )
        self.cross_img_to_lm = nn.MultiheadAttention(
            embed_dim=cross_attn_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
        )

        fusion_dim = 1536 + (2 * cross_attn_dim if use_landmark_cross_fusion else 0)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _encode_landmarks(self, landmarks):
        if landmarks is None:
            return None

        if landmarks.dim() == 4:
            # Heatmap representation: (B, P, H, W) -> (B, P, D)
            b, p, h, w = landmarks.shape
            lm = landmarks.view(b * p, 1, h, w)
            tokens = self.lm_point_cnn(lm)
            tokens = tokens.flatten(1).view(b, p, self.cross_attn_dim)
        elif landmarks.dim() == 3 and landmarks.size(-1) == 2:
            # Coordinate representation: (B, P, 2) -> (B, P, D)
            tokens = self.lm_coord_proj(landmarks)
            b, p, _ = tokens.shape
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

    @staticmethod
    def _masked_mean(tokens, mask):
        if mask is None:
            return tokens.mean(dim=1)

        valid_mask = (mask > 0.5).float().unsqueeze(-1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        return (tokens * valid_mask).sum(dim=1) / denom

    @staticmethod
    def _build_key_padding_mask(mask, token_count):
        if mask is None:
            return None, None

        m = mask[:, :token_count] <= 0.5
        # MultiheadAttention cannot handle rows where all keys are masked.
        all_masked = m.all(dim=1)
        if all_masked.any():
            m[all_masked, 0] = False
        return m, all_masked

    def forward(self, x, landmarks=None, landmark_mask=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)

        x = self.layer3(x)
        feat3 = self.avgpool(x)
        feat3 = torch.flatten(feat3, 1)   # (B, 512)

        x = self.layer4(x)
        x_map = x
        feat4 = self.avgpool(x)
        feat4 = torch.flatten(feat4, 1)   # (B, 1024)

        # Base concat from multi-scale image features.
        feat = torch.cat([feat3, feat4], dim=1)  # (B, 1536)

        if self.use_landmark_cross_fusion and landmarks is not None:
            lm_tokens = self._encode_landmarks(landmarks)
            if lm_tokens is not None:
                img_tokens = self.img_token_proj(x_map)  # (B, D, 6, 6)
                img_tokens = img_tokens.flatten(2).transpose(1, 2)  # (B, 36, D)

                token_count = lm_tokens.size(1)
                lm_kpm, all_missing = self._build_key_padding_mask(landmark_mask, token_count)

                # Q from landmarks, K/V from image.
                lm_query_img_ctx, _ = self.cross_lm_to_img(
                    query=lm_tokens,
                    key=img_tokens,
                    value=img_tokens,
                )

                # Q from image, K/V from landmarks.
                img_query_lm_ctx, _ = self.cross_img_to_lm(
                    query=img_tokens,
                    key=lm_tokens,
                    value=lm_tokens,
                    key_padding_mask=lm_kpm,
                )

                lm_ctx = self._masked_mean(lm_query_img_ctx, landmark_mask[:, :token_count] if landmark_mask is not None else None)
                img_ctx = img_query_lm_ctx.mean(dim=1)
                if all_missing is not None and all_missing.any():
                    img_ctx[all_missing] = 0.0
                cross_feat = torch.cat([lm_ctx, img_ctx], dim=1)
                feat = torch.cat([feat, cross_feat], dim=1)

        out = self.fusion_fc(feat)

        return out