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
        use_landmark_branch=False,
        landmark_token_mode="learnable",
        landmark_num_points=12,
        landmark_embed_dim=128,
    ):
        super().__init__()
        self.use_landmark_branch = use_landmark_branch
        self.landmark_token_mode = landmark_token_mode
        self.landmark_num_points = landmark_num_points
        self.landmark_embed_dim = landmark_embed_dim

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
        self.fc = nn.Linear(1024, num_classes)
        self.fusion_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Optional landmark branch (learnable/input/hybrid).
        self.learnable_landmark_tokens = nn.Parameter(
            torch.randn(1, landmark_num_points, landmark_embed_dim)
        )
        self.hybrid_landmark_gate = nn.Parameter(torch.tensor(0.0))
        self.lm_heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lm_heatmap_proj = nn.Linear(32, landmark_embed_dim)
        self.lm_coords_proj = nn.Linear(2, landmark_embed_dim)
        self.lm_token_fuse = nn.Sequential(
            nn.Linear(1536 + landmark_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _resize_token_count(tokens, target_points):
        if tokens.size(1) == target_points:
            return tokens
        resized = torch.nn.functional.interpolate(
            tokens.transpose(1, 2),
            size=target_points,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    def _encode_input_landmarks(self, landmarks):
        if landmarks is None:
            return None

        if landmarks.dim() == 4:
            # (B, P, H, W) -> (B, P, D)
            bsz, pnt, h, w = landmarks.shape
            x = landmarks.view(bsz * pnt, 1, h, w)
            x = self.lm_heatmap_encoder(x).flatten(1)
            x = self.lm_heatmap_proj(x)
            x = x.view(bsz, pnt, -1)
            return x

        if landmarks.dim() == 3 and landmarks.size(-1) == 2:
            return self.lm_coords_proj(landmarks)

        if landmarks.dim() == 2:
            bsz, c = landmarks.shape
            pnt = c // 2
            coords = landmarks.view(bsz, pnt, 2)
            return self.lm_coords_proj(coords)

        return None

    def _resolve_landmark_feature(self, landmarks, landmark_mask, batch_size, device, dtype):
        learned = self.learnable_landmark_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        if self.landmark_token_mode == "learnable":
            return learned.mean(dim=1)

        inp = self._encode_input_landmarks(landmarks)
        if inp is not None:
            inp = self._resize_token_count(inp, self.landmark_num_points)

        if self.landmark_token_mode == "input":
            if inp is None:
                return learned.mean(dim=1)
            if landmark_mask is not None and landmark_mask.size(1) == inp.size(1):
                inp = inp * landmark_mask.unsqueeze(-1)
            return inp.mean(dim=1)

        # hybrid
        if inp is None:
            return learned.mean(dim=1)

        alpha = torch.sigmoid(self.hybrid_landmark_gate)
        mixed = alpha * inp + (1.0 - alpha) * learned
        if landmark_mask is not None and landmark_mask.size(1) == mixed.size(1):
            mixed = mixed * landmark_mask.unsqueeze(-1) + learned * (1.0 - landmark_mask).unsqueeze(-1)
        return mixed.mean(dim=1)

    def forward(self, x, landmarks=None, landmark_mask=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer2(x)

        x = self.layer3(x)
        feat3 = self.avgpool(x)
        feat3 = torch.flatten(feat3, 1)   # (B, 512)

        x = self.layer4(x)
        feat4 = self.avgpool(x)
        feat4 = torch.flatten(feat4, 1)   # (B, 1024)

        # concat
        # feat = torch.cat([feat3, feat4], dim=1)  # (B, 1536)
        feat = feat4

        if self.use_landmark_branch:
            lm_feat = self._resolve_landmark_feature(
                landmarks=landmarks,
                landmark_mask=landmark_mask,
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            feat = torch.cat([feat, lm_feat], dim=1)
            out = self.lm_token_fuse(feat)
            return out

        out = self.fusion_fc(feat)

        return out