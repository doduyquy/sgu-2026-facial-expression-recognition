import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeformableCoreMotifModule(nn.Module):
    """
    [Bí quyết chống Overfit] Khối Motif Cốt lõi cải tiến theo triết lý DAT.
    Học các bộ biểu cảm đặc trưng cục bộ (Semantic Prototypes) một cách tinh gọn.
    """
    def __init__(self, num_classes=7, motifs_per_class=4, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.feat_dim = feat_dim
        
        # Ngân hàng cấu hình cảm xúc tinh giản: (7, 4, 10, 128)
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, 10, feat_dim))
        nn.init.xavier_uniform_(self.motifs)
        
        # Nhiệt độ scale mềm distribution cho Cosine Similarity
        self.temperature = nn.Parameter(torch.ones(1) * -2.0)

    def forward(self, node_features):
        """
        Args: node_features: (B, 10, 128) - Đặc trưng đa quy mô, đa đầu cực nét
        """
        B, K, C = node_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        # Chuẩn hóa không gian L2 để đưa về Cosine Similarity nguyên bản
        node_features_norm = F.normalize(node_features, p=2, dim=-1)
        motifs_norm = F.normalize(self.motifs, p=2, dim=-1)
        
        tau = F.softplus(self.temperature).clamp(min=0.05)
        
        # 1. Soft Alignment qua Einstein Summation: (B, L, M, 10, 10)
        sim_matrix = torch.einsum('bic,lmjc->blmij', node_features_norm, motifs_norm)
        
        align_weights = F.softmax(sim_matrix / tau, dim=-1)
        aligned_motifs = torch.einsum('blmij,lmjc->blmic', align_weights, motifs_norm)
        aligned_motifs = F.normalize(aligned_motifs, p=2, dim=-1)
        
        # 2. Tính toán điểm đối sánh cấu trúc hình học giải phẫu
        S = torch.einsum('bic,blmic->blmi', node_features_norm, aligned_motifs).mean(dim=-1)
        
        # 3. Lựa chọn mềm (Smooth Selection) qua logsumexp và nhân trả lại scale tau chuẩn
        logits = tau * torch.logsumexp(S / tau, dim=-1)
        
        return logits, S


class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 4)     # Giảm xuống 4 chống học vẹt
        self.num_heads = config.get('num_heads', 4)                  # 4 đầu trinh sát tọa độ theo DAT
        self.offset_amplitude = float(config.get('offset_amplitude', 0.25))
        
        import torchvision.models as models
        # 1. BACKBONE RESNET18 CHUẨN SOTA HỌC ACADEMIC
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data[:, 2:3, 2:5, 2:5].mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 # Đầu ra Layer 3: (B, 256, 12, 12)
        self.layer4 = resnet.layer4 # Đầu ra Layer 4: (B, 512, 6, 6)
        
        # 2. TRIẾT LÝ DEFORMABLE DETR: Bộ nén chiều độc lập, không cộng gộp vụng về
        self.reducer_l3 = nn.Sequential(
            nn.Conv2d(256, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.reducer_l4 = nn.Sequential(
            nn.Conv2d(512, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. GLOBAL CONTEXT BRANCH
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        dropout = config.get('dropout', 0.5) # Ép Dropout mạnh tay để diệt Overfit
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, self.num_classes)
        )
        
        # 4. TRIẾT LÝ DAT: Bộ dự đoán độ lệch Đa đầu (Multi-Head Offsets Predictor)
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_heads * 2), # Xuất ra (num_heads * 2) tọa độ dịch chuyển
            nn.Tanh()
        )
        
        # Bộ trộn đặc trưng sau khi gộp đa tầng và đa đầu
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim), # Kết hợp kênh L3 + L4
            nn.ReLU(inplace=True)
        )
        
        # 5. MOTIF BLOCK 
        self.motif_module = DeformableCoreMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            feat_dim=self.feat_dim
        )
        
        # CỔNG HOÀ TRỘN ĐỘNG GATED FUSION (Global + Motif)
        self.gate = nn.Sequential(
            nn.Linear(self.num_classes * 2, self.num_classes),
            nn.Sigmoid()
        )

    def _extract_deformable_multiscale_multihead_nodes(self, feat_map_l3, feat_map_l4, glob_embed, landmarks):
        B = feat_map_l3.shape[0]
        H3, W3 = feat_map_l3.shape[2:]
        
        # Thiết lập tọa độ lưới nền trung bình giải phẫu (Fallback)
        base_y = torch.tensor([1, 3, 2, 2, 4, 4, 6, 9, 9, 11], device=feat_map_l3.device, dtype=torch.float) * (H3 / 12.0)
        base_x = torch.tensor([5, 5, 3, 7, 3, 7, 5, 3, 7, 5], device=feat_map_l3.device, dtype=torch.float) * (W3 / 12.0)
        default_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, -1, -1) # (B, 10, 2)
        
        if landmarks is None:
            base_grid = (default_grid / (W3 - 1)) * 2.0 - 1.0
        else:
            base_grid = (landmarks / 48.0) * 2.0 - 1.0 # Ánh xạ thẳng vào không gian lưới [-1.0, 1.0]
            
        # Jittering tọa độ lúc Train nhằm triệt tiêu hoàn toàn khả năng học vẹt vị trí
        if self.training:
            base_grid = base_grid + (torch.rand_like(base_grid) - 0.5) * 0.08
            
        # Lấy mẫu đặc trưng mồi hướng dẫn (Initial Sampling) từ tầng sắc nét L3
        mboi_feats = F.grid_sample(feat_map_l3, base_grid.unsqueeze(2), align_corners=True).squeeze(-1).transpose(1, 2)
        
        # Dự đoán Đa đầu lệch (Multi-Head Offsets)
        glob_expand = glob_embed.unsqueeze(1).expand(-1, 10, -1)
        combined = torch.cat([mboi_feats, glob_expand], dim=-1)
        
        raw_offsets = self.offset_predictor(combined) # (B, 10, num_heads * 2)
        offsets = raw_offsets.view(B, 10, self.num_heads, 2) * self.offset_amplitude
        self._latest_offsets = offsets
        
        # Phát tán lưới tọa độ đa vệ tinh (Multi-head coordination sampling map)
        final_grid = base_grid.unsqueeze(2) + offsets # (B, 10, num_heads, 2)
        final_grid_flat = final_grid.view(B, 10 * self.num_heads, 1, 2)
        
        # Thực hiện trích xuất dữ liệu song song từ 2 quy mô độc lập (Triết lý Deformable DETR)
        nodes_l3 = F.grid_sample(feat_map_l3, final_grid_flat, align_corners=True).squeeze(-1).transpose(1, 2)
        nodes_l3 = nodes_l3.view(B, 10, self.num_heads, self.feat_dim)
        
        nodes_l4 = F.grid_sample(feat_map_l4, final_grid_flat, align_corners=True).squeeze(-1).transpose(1, 2)
        nodes_l4 = nodes_l4.view(B, 10, self.num_heads, self.feat_dim)
        
        # Tiến hành ép gộp đa tầng và hòa trộn liên thông không gian
        multiscale_feats = torch.cat([nodes_l3, nodes_l4], dim=-1) # (B, 10, num_heads, feat_dim * 2)
        fused_head_feats = self.feature_fusion(multiscale_feats)   # (B, 10, num_heads, feat_dim)
        
        # Gom tụ thông tin từ các đầu trinh sát về Node đại diện cốt lõi bằng phép Mean
        final_nodes = fused_head_feats.mean(dim=2) # (B, 10, feat_dim)
        return final_nodes

    def forward(self, x, return_selection=False, targets=None, landmarks=None, statuses=None):
        if targets is not None:
            self._latest_targets = targets
            
        # Xử lý tự động bọc màng dữ liệu khi kích hoạt cấu trúc TenCrop (5D input tensor)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            logits_list = []
            self._tencrop_landmarks_active = True
            
            for t in range(T):
                crop_x = x[:, t, :, :, :]
                crop_landmarks = None
                if landmarks is not None:
                    crop_landmarks = landmarks.clone()
                    off_x = [0, 8, 0, 8, 4, 0, 8, 0, 8, 4][t]
                    off_y = [0, 0, 8, 8, 4, 0, 0, 8, 8, 4][t]
                    crop_landmarks[:, :, 0] = crop_landmarks[:, :, 0] - off_x
                    crop_landmarks[:, :, 1] = crop_landmarks[:, :, 1] - off_y
                    if t >= 5:
                        crop_landmarks[:, :, 0] = 39.0 - crop_landmarks[:, :, 0]
                        crop_landmarks[:, [2, 3, 4, 5, 7, 8], :] = crop_landmarks[:, [3, 2, 5, 4, 8, 7], :].clone()
                        
                out = self.forward(crop_x, return_selection=return_selection, targets=targets, landmarks=crop_landmarks, statuses=statuses)
                if return_selection:
                    logits_list.append(out[0])
                else:
                    logits_list.append(out)
                    
            self._tencrop_landmarks_active = False
            mean_logits = torch.stack(logits_list, dim=1).mean(dim=1)
            return (mean_logits, out[1], out[2], out[3]) if return_selection else mean_logits

        B = x.shape[0]
        
        # --- PHASE 1: CNN TRÍCH XUẤT ĐA TẦNG ---
        # Tận dụng tối đa xương sống ResNet18 đã được nạp trọng số chuẩn ImageNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) # (B, 256, 12, 12)
        x4 = self.layer4(x3) # (B, 512, 6, 6)
        
        # --- PHASE 2: NHÁNH TOÀN CỤC (GLOBAL CONTEXT) ---
        logits_global = self.global_fc(self.global_pool(x4))
        self._latest_logits_global = logits_global
        
        # --- PHASE 3: NHÁNH CỤC BỘ BIẾN DẠNG ĐA QUY MÔ ---
        feat_map_l3 = self.reducer_l3(x3) # Không gian chi tiết sắc nét (B, 128, 12, 12)
        feat_map_l4 = self.reducer_l4(x4) # Không gian ngữ nghĩa sâu (B, 128, 6, 6)
        
        # Chuẩn bị vector bối cảnh nền cho bộ định hướng
        glob_embed = feat_map_l4.mean(dim=(2, 3)) # (B, 128)
        
        # Trích xuất 10 hạt đặc trưng tinh khiết (Sparse Nodes)
        node_feats = self._extract_deformable_multiscale_multihead_nodes(
            feat_map_l3=feat_map_l3,
            feat_map_l4=feat_map_l4,
            glob_embed=glob_embed,
            landmarks=landmarks
        ) # Đầu ra chuẩn chỉnh: (B, 10, 128)
        
        # --- PHASE 4: ĐỐI SÁNH TRÚC LƯỚI MOTIF VÀNG ---
        logits_motif, S = self.motif_module(node_feats)
        self._latest_logits_motif = logits_motif
        self._latest_scores = S
        
        # --- PHASE 5: CỔNG HÒA TRỘN ĐỘNG GATED FUSION ---
        gate_input = torch.cat([logits_motif, logits_global], dim=-1)
        g = self.gate(gate_input)
        
        logits = g * logits_motif + (1.0 - g) * logits_global
        
        # Giữ lại các bộ đệm giả lập phục vụ cho khâu Logging/Visualization của Trainer
        self._latest_top_k = torch.zeros(B, self.num_classes, dtype=torch.long, device=x.device)
        
        if return_selection:
            return logits, self._latest_top_k, (None, None), self._latest_scores
            
        return logits

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
            
        # Phạt hình học L2 nhẹ nhàng nhằm ép các đầu lệch chỉ hoạt động quanh vùng cơ cơ bản
        l_off = torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        
        aux_dict = {
            "offset_reg": l_off,
            "logits_global": self._latest_logits_global,
            "logits_motif": self._latest_logits_motif
        }
        return aux_dict

    def get_landmark_outputs(self):
        return getattr(self, '_latest_scores', None), getattr(self, '_latest_top_k', None)


if __name__ == "__main__":
    config = {'feat_dim': 128, 'num_classes': 7, 'motifs_per_class': 4, 'num_heads': 4}
    model = MotifGraphModel(config)
    dummy_img = torch.randn(2, 1, 48, 48)
    out = model(dummy_img)
    print(f"MS-DCMN Output Shape: {out.shape}") # (2, 7)