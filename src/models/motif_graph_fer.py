import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeformableCoreMotifModule(nn.Module):
    def __init__(self, num_classes=7, motifs_per_class=4, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.feat_dim = feat_dim
        
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, 10, feat_dim))
        nn.init.xavier_uniform_(self.motifs)
        self.temperature = nn.Parameter(torch.ones(1) * -2.0)

    def forward(self, node_features):
        B, K, C = node_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        node_features_norm = F.normalize(node_features, p=2, dim=-1)
        motifs_norm = F.normalize(self.motifs, p=2, dim=-1)
        
        # [VŨ KHÍ 2]: MOTIF DIVERSITY LOSS (Inter-motif Decorrelation)
        flat_m = motifs_norm.view(L, M, -1) # (L, M, 10*C)
        sim_m = torch.einsum('lmc,lnc->lmn', flat_m, flat_m)
        mask = torch.eye(M, device=sim_m.device).unsqueeze(0).bool()
        # Tính trung bình sự trùng lặp (càng trùng lặp loss càng cao)
        self._latest_diversity = sim_m.masked_fill(mask, 0.0).sum() / (L * M * (M - 1))
        
        tau = F.softplus(self.temperature).clamp(min=0.05)
        sim_matrix = torch.einsum('bic,lmjc->blmij', node_features_norm, motifs_norm)
        
        # [SỬA LỖI VŨ KHÍ 1]: MOTIF DROPOUT CHUẨN TOÁN HỌC SOFTMAX
        if self.training:
            # Random drop khoảng 15% motif (True = Drop)
            drop_mask = torch.rand(L, M, device=node_features.device) < 0.15 
            # Reshape để khớp với sim_matrix (B, L, M, 10, 10)
            drop_mask = drop_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # Gán âm vô cùng để Softmax lờ đi hoàn toàn Motif bị drop
            sim_matrix = sim_matrix.masked_fill(drop_mask, -1e9)
            
        align_weights = F.softmax(sim_matrix / tau, dim=-1)
        aligned_motifs = torch.einsum('blmij,lmjc->blmic', align_weights, motifs_norm)
        aligned_motifs = F.normalize(aligned_motifs, p=2, dim=-1)
        
        S = torch.einsum('bic,blmic->blmi', node_features_norm, aligned_motifs).mean(dim=-1)
        logits = tau * torch.logsumexp(S / tau, dim=-1)
        
        return logits, S


class MotifBackbone(nn.Module):
    def __init__(self, pretrained_cnn_path="", in_channels=1, feat_dim=128):
        super().__init__()
        import torchvision.models as models
        import os

        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet18(pretrained=True)

        if pretrained_cnn_path and os.path.exists(pretrained_cnn_path):
            print(f"Loading pretrained CNN from {pretrained_cnn_path}...")
            try:
                ckpt = torch.load(pretrained_cnn_path, map_location='cpu')
                state = ckpt
                for key in ['state_dict', 'model_state_dict', 'net', 'model']:
                    if isinstance(ckpt, dict) and key in ckpt:
                        state = ckpt[key]
                        break
                
                model_dict = resnet.state_dict()
                pretrained_dict = {}
                
                for k, v in state.items():
                    name = k.replace('module.', '').replace('backbone.', '').replace('resnet.', '').replace('net.', '')
                    if name in model_dict:
                        if v.shape == model_dict[name].shape:
                            pretrained_dict[name] = v
                
                if len(pretrained_dict) > 0:
                    model_dict.update(pretrained_dict)
                    resnet.load_state_dict(model_dict)
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")

        old_w = resnet.conv1.weight
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            if old_w.shape[2:] == (7, 7):
                center = old_w[:, :, 2:5, 2:5]
            else:
                center = old_w
            self.conv1.weight.copy_(center.mean(dim=1, keepdim=True))

        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 8)    # TĂNG LÊN 8 ĐỂ COVER RỘNG MANIFOLD
        self.num_heads = config.get('num_heads', 4)                  
        self.offset_amplitude = float(config.get('offset_amplitude', 0.25))
        
        pretrained_cnn_path = config.get('pretrained_cnn_path', "")
        self.backbone = MotifBackbone(pretrained_cnn_path=pretrained_cnn_path, feat_dim=self.feat_dim)
        
        self.reducer_l2 = nn.Sequential(
            nn.Conv2d(128, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.reducer_l3 = nn.Sequential(
            nn.Conv2d(256, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # [VŨ KHÍ 5]: Bóp cổ chai nhánh Hình học để chặn rò rỉ danh tính (Identity Leakage)
        self.geo_proj = nn.Sequential(
            nn.Linear(100, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + 32, self.feat_dim), # Chỉ lấy 32 chiều hình học
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(self.feat_dim, self.num_classes)
        )
        
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_heads * 2), 
            nn.Tanh()
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.head_fusion = nn.Sequential(
            nn.Linear(self.feat_dim * self.num_heads, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # [VŨ KHÍ 6]: Đẩy mạnh Attention Dropout lên 0.5
        self.graph_interaction = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=4,
            dim_feedforward=self.feat_dim * 2,
            dropout=0.5, # TĂNG DROPOUT
            batch_first=True
        )
        
        self.motif_module = DeformableCoreMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            feat_dim=self.feat_dim
        )
        
        self.gate = nn.Sequential(
            nn.Linear(self.num_classes * 2 + 2, self.num_classes),
            nn.Sigmoid()
        )

    def _extract_deformable_multiscale_multihead_nodes(self, feat_map_l2, feat_map_l3, glob_embed, landmarks, img_h=48.0, img_w=48.0):
        B = feat_map_l3.shape[0]
        H3, W3 = feat_map_l3.shape[2:]
        
        base_y = torch.tensor([1, 3, 2, 2, 4, 4, 6, 9, 9, 11], device=feat_map_l3.device, dtype=torch.float) * (H3 / 12.0)
        base_x = torch.tensor([5, 5, 3, 7, 3, 7, 5, 3, 7, 5], device=feat_map_l3.device, dtype=torch.float) * (W3 / 12.0)
        default_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, -1, -1)
        
        if landmarks is None:
            base_grid = (default_grid / (W3 - 1)) * 2.0 - 1.0
        else:
            c_x = (landmarks[:, :, 0] / img_w) * 2.0 - 1.0
            c_y = (landmarks[:, :, 1] / img_h) * 2.0 - 1.0
            base_grid = torch.stack([c_x, c_y], dim=-1)
            
        if self.training:
            base_grid = base_grid + (torch.rand_like(base_grid) - 0.5) * 0.08
            
        mboi_feats = F.grid_sample(feat_map_l3, base_grid.unsqueeze(2), align_corners=True).squeeze(-1).transpose(1, 2)
        glob_expand = glob_embed.unsqueeze(1).expand(-1, 10, -1)
        combined = torch.cat([mboi_feats, glob_expand], dim=-1)
        
        raw_offsets = self.offset_predictor(combined) 
        offsets = raw_offsets.view(B, 10, self.num_heads, 2) * self.offset_amplitude
        self._latest_offsets = offsets
        
        final_grid = base_grid.unsqueeze(2) + offsets
        final_grid_flat = final_grid.view(B, 10 * self.num_heads, 1, 2)
        
        nodes_l2 = F.grid_sample(feat_map_l2, final_grid_flat, align_corners=True).squeeze(-1).transpose(1, 2)
        nodes_l2 = nodes_l2.view(B, 10, self.num_heads, self.feat_dim)
        
        nodes_l3 = F.grid_sample(feat_map_l3, final_grid_flat, align_corners=True).squeeze(-1).transpose(1, 2)
        nodes_l3 = nodes_l3.view(B, 10, self.num_heads, self.feat_dim)
        
        multiscale_feats = torch.cat([nodes_l2, nodes_l3], dim=-1)
        fused_head_feats = self.feature_fusion(multiscale_feats)   
        fused_head_flat = fused_head_feats.view(B, 10, self.num_heads * self.feat_dim)
        final_nodes = self.head_fusion(fused_head_flat) 
        
        return final_nodes

    def forward_single_crop(self, x, return_selection=False, targets=None, landmarks=None, statuses=None):
        B = x.shape[0]
        img_h, img_w = x.shape[2], x.shape[3]
        
        x2, x3, x4 = self.backbone(x)
        glob_cnn = self.global_pool(x4).view(B, 512)
        
        if landmarks is None:
            base_y = torch.tensor([1, 3, 2, 2, 4, 4, 6, 9, 9, 11], device=x.device, dtype=torch.float) * (img_h / 12.0)
            base_x = torch.tensor([5, 5, 3, 7, 3, 7, 5, 3, 7, 5], device=x.device, dtype=torch.float) * (img_w / 12.0)
            lm = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(B, -1, -1)
        else:
            lm = landmarks.clone()
            
        # [VŨ KHÍ 4]: Bơm nhiễu hình học lúc Train để tránh học thuộc lòng khoảng cách
        if self.training:
            lm = lm + torch.randn_like(lm) * 1.5 
            
        diff = lm.unsqueeze(2) - lm.unsqueeze(1)
        dist_matrix = torch.norm(diff, p=2, dim=-1) / 48.0 
        rel_geo = dist_matrix.view(B, 100)
        rel_geo = self.geo_proj(rel_geo) # Nén xuống 32 chiều
        
        glob_combined = torch.cat([glob_cnn, rel_geo], dim=-1)
        logits_global = self.global_fc(glob_combined)
        self._latest_logits_global = logits_global
        
        feat_map_l2 = self.reducer_l2(x2) 
        feat_map_l3 = self.reducer_l3(x3) 
        
        glob_embed = self.global_pool(x4).view(B, -1)
        glob_embed = F.adaptive_avg_pool1d(glob_embed.unsqueeze(1), self.feat_dim).squeeze(1) 
        
        node_feats_raw = self._extract_deformable_multiscale_multihead_nodes(
            feat_map_l2=feat_map_l2,
            feat_map_l3=feat_map_l3,
            glob_embed=glob_embed,
            landmarks=landmarks,
            img_h=img_h,
            img_w=img_w
        )
        
        # [VŨ KHÍ 3]: WEAK GRAPH INTERACTION (Residual Scaling 80/20)
        # Ép Graph chỉ làm nhiễu phụ (Modulation) thay vì ghi đè toàn bộ ngữ nghĩa
        node_feats_interacted = self.graph_interaction(node_feats_raw)
        node_feats = node_feats_raw * 0.8 + node_feats_interacted * 0.2
        
        logits_motif, S = self.motif_module(node_feats)
        self._latest_logits_motif = logits_motif
        self._latest_scores = S
        
        prob_mot = F.softmax(logits_motif, dim=-1)
        # SỬA LỖI: Bắt buộc phải thêm .detach() để tránh rò rỉ Gradient
        ent_mot = -(prob_mot * torch.log(prob_mot + 1e-8)).sum(dim=-1, keepdim=True).detach()
        prob_glob = F.softmax(logits_global, dim=-1)
        # SỬA LỖI: Bắt buộc phải thêm .detach() 
        ent_glob = -(prob_glob * torch.log(prob_glob + 1e-8)).sum(dim=-1, keepdim=True).detach()
        
        gate_input = torch.cat([logits_motif, logits_global, ent_mot, ent_glob], dim=-1)
        g = self.gate(gate_input)
        logits = g * logits_motif + (1.0 - g) * logits_global
        
        self._latest_top_k = torch.zeros(B, self.num_classes, dtype=torch.long, device=x.device)
        if return_selection:
            return logits, self._latest_top_k, (None, None), self._latest_scores
        return logits

    def forward(self, x, return_selection=False, targets=None, landmarks=None, statuses=None):
        if targets is not None:
            self._latest_targets = targets
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            logits_list = []
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
                        
                out = self.forward_single_crop(crop_x, return_selection=return_selection, targets=targets, landmarks=crop_landmarks, statuses=statuses)
                if return_selection:
                    logits_list.append(out[0])
                else:
                    logits_list.append(out)
            mean_logits = torch.stack(logits_list, dim=1).mean(dim=1)
            return (mean_logits, out[1], out[2], out[3]) if return_selection else mean_logits
        return self.forward_single_crop(x, return_selection, targets, landmarks, statuses)

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
        l_off = torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        
        # Bắt lấy Motif Diversity Loss
        l_div = getattr(self.motif_module, '_latest_diversity', 0.0)
        
        aux_dict = {
            "offset_reg": l_off,
            "motif_diversity": l_div, # CHUẨN BỊ CHO TRAINER
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