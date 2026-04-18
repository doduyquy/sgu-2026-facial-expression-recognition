import torch
import torch.nn as nn
from .CBAM import CBAM
from .learned_landmark import LearnedLandmarkBranch
from .landmark_multihead import MultiHeadLandmarkBranch


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
        num_classes=7, #số class mặc định là 7 (theo FER2013), nhưng có thể override qua config
        in_channels=1, #số channel của input, mặc định 1 (grayscale), nhưng có thể override qua config
        use_cbam_stage34=True,
        cbam_reduction=16, #số lượng kênh giảm trong CBAM, mặc định 16, nhưng có thể override qua config
        cbam_kernel_size=7, #kích thước kernel trong CBAM, mặc định 7, nhưng có thể override qua config
        use_learned_landmark_branch=True, #Xác định dùng nhánh learned landmark
        landmark_num_points=6, #Số điểm landmark, mặc định 6, nhưng có thể override qua config
        landmark_tau=0.07, #Dùng để điều chỉnh độ mềm của heatmap
        landmark_feature_dropout_p=0.3, #Dùng để dropout trên feature map trước khi tạo heatmap để tăng tính generalization
        landmark_head_dropout_p=0.2, # Dùng để dropout trên từng head của multi-head để tăng tính diversity giữa các head
        landmark_edge_guidance_beta=1.0, #Dùng để điều chỉnh trọng số edge guidance trong loss của nhánh landmark
        landmark_edge_alpha=6.0, #Dùng để điều chỉnh độ nhạy của edge guidance (alpha trong exp(-alpha * edge_dist)) để làm mờ dần ảnh hưởng của các điểm xa cạnh

        landmark_from_stage=3,#Dùng để xác định lấy feature map từ stage nào để làm input cho nhánh landmark (3 hoặc 4)
        landmark_num_heads=1, #Dùng để xác định số head trong multi-head landmark branch (>=1, nếu =1 thì sẽ dùng single-head như trước)
        # Optionally upsample input images before backbone (e.g., (96,96) or (112,112))
        input_upsample=None,
    ):
        super().__init__()
        self.use_learned_landmark_branch = use_learned_landmark_branch
        self.landmark_num_points = landmark_num_points
        self.landmark_tau = landmark_tau
        self.landmark_from_stage = landmark_from_stage
        self.landmark_num_heads = max(1, int(landmark_num_heads))
        # optional input upsample size (H, W) or None
        self.input_upsample = tuple(input_upsample) if input_upsample is not None else None

        self._latest_aux_losses = {} #dùng để lưu trữ các loss phụ từ nhánh landmark để log lên wandb, tránh việc phải return nhiều giá trị từ forward
        self._latest_landmark_heatmaps = None #dùng để lưu trữ heatmap đầu ra từ nhánh landmark để log lên wandb
        self._latest_landmark_coords = None #dùng để lưu trữ tọa độ landmark đầu ra từ nhánh landmark để log lên wandb

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

        # Baseline classifier (no landmark branch).
        self.fusion_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        landmark_in_channels = 512 if landmark_from_stage == 3 else 1024

        # support optional multi-head landmark branch
        if self.landmark_num_heads > 1:
            self.learned_landmark_branch = MultiHeadLandmarkBranch(
                in_channels=landmark_in_channels,
                landmark_num_points=landmark_num_points,
                num_heads=self.landmark_num_heads,
                landmark_tau=landmark_tau,
                feature_dropout_p=landmark_feature_dropout_p,
                head_dropout_p=landmark_head_dropout_p,
                edge_guidance_beta=landmark_edge_guidance_beta,
                edge_alpha=landmark_edge_alpha,
            )
        else:
            self.learned_landmark_branch = LearnedLandmarkBranch(
                in_channels=landmark_in_channels,
                landmark_num_points=landmark_num_points,
                landmark_tau=landmark_tau,
                feature_dropout_p=landmark_feature_dropout_p,
                head_dropout_p=landmark_head_dropout_p,
                edge_guidance_beta=landmark_edge_guidance_beta,
                edge_alpha=landmark_edge_alpha,
            )

        #Dùng để fusion giữa feature map từ backbone và feature map từ nhánh landmark để đưa vào classifier cuối cùng. Kích thước đầu vào sẽ là tổng của 3 phần: feature map từ stage 3, feature map từ stage 4, và feature map được trích xuất từ heatmap landmark (sau khi flatten). Cụ thể là: 512 (stage 3) + 1024 (stage 4) + ((landmark_num_points + 1) * landmark_in_channels) (từ nhánh landmark, bao gồm cả điểm đặc trưng trung tâm). Tổng sẽ là 1536 nếu dùng 6 điểm landmark với input 512 kênh cho nhánh landmark.
        fusion_in_dim = 512 + 1024 + ((landmark_num_points + 1) * landmark_in_channels) #tính toán kích thước đầu vào cho fusion_fc dựa trên số điểm landmark và số kênh đầu vào cho nhánh landmark
        self.landmark_fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        #Dùng để auxiliary classification chỉ dựa trên feature map từ nhánh landmark để đảm bảo rằng các điểm landmark được học ra có tính phân biệt cao đối với nhiệm vụ chính. Kích thước đầu vào sẽ là kích thước của feature map được trích xuất từ heatmap landmark (sau khi flatten), cụ thể là (landmark_num_points + 1) * landmark_in_channels, trong đó +1 là cho điểm đặc trưng trung tâm.
        aux_in_dim = (landmark_num_points + 1) * landmark_in_channels #tính toán kích thước đầu vào cho landmark_aux_fc dựa trên số điểm landmark và số kênh đầu vào cho nhánh landmark
        self.landmark_aux_fc = nn.Sequential(
            nn.Linear(aux_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._latest_landmark_feat = None
        self._latest_landmark_aux_logits = None
        # learnable scale for landmark features to avoid hardcoded heuristics
        self.landmark_scale = nn.Parameter(torch.tensor(1.0))

        # small projection for landmark features can be added later; keep placeholder name
        self.landmark_proj = None

    def get_aux_losses(self):
        return self._latest_aux_losses

    def get_landmark_features(self):
        return self._latest_landmark_feat

    def get_landmark_aux_logits(self):
        return self._latest_landmark_aux_logits

    def get_landmark_outputs(self):
        return self._latest_landmark_heatmaps, self._latest_landmark_coords

    def set_training_progress(self, progress):
        setter = getattr(self.learned_landmark_branch, "set_training_progress", None)
        if callable(setter):
            setter(progress)

    def get_current_prior_strength(self):
        getter = getattr(self.learned_landmark_branch, "get_current_prior_strength", None)
        if callable(getter):
            return getter()
        return None

    def forward(self, x, landmarks=None, landmark_mask=None):
        _ = landmarks
        _ = landmark_mask
        input_image = x

        # optionally upsample input image to larger spatial size to give backbone more resolution
        if self.input_upsample is not None:
            try:
                input_image = nn.functional.interpolate(input_image, size=self.input_upsample, mode='bilinear', align_corners=False)
            except Exception:
                # fallback to original if interpolation fails
                input_image = x

        # run backbone on (possibly upsampled) input
        x = self.relu(self.bn1(self.conv1(input_image)))
        x = self.pool(x)

        x = self.layer2(x)

        x3 = self.layer3(x)
        feat3 = torch.flatten(self.avgpool(x3), 1)

        x4 = self.layer4(x3)
        feat4 = torch.flatten(self.avgpool(x4), 1)

        if not self.use_learned_landmark_branch:
            feat = torch.cat([feat3, feat4], dim=1)
            self._latest_aux_losses = {}
            self._latest_landmark_heatmaps = None
            self._latest_landmark_coords = None
            return self.fusion_fc(feat)

        landmark_src = x3 if self.landmark_from_stage == 3 else x4
        heatmaps, coords, feat_k, aux = self.learned_landmark_branch(landmark_src, input_image=input_image)
        # emphasize landmark features via a learnable scale to avoid brittle heuristics
        fused = torch.cat([feat3, feat4, self.landmark_scale * feat_k], dim=1)
        logits = self.landmark_fusion_fc(fused)

        # auxiliary classification from landmark features
        # allow gradient flow so landmark features can become discriminative
        aux_logits = self.landmark_aux_fc(feat_k)

        self._latest_aux_losses = aux
        self._latest_landmark_heatmaps = heatmaps
        self._latest_landmark_coords = coords
        self._latest_landmark_feat = feat_k
        self._latest_landmark_aux_logits = aux_logits

        return logits