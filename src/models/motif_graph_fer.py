import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.training.losses import MotifConsistencyLoss

try:
    from .CBAM import CBAM
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.CBAM import CBAM


class MaskBlock(nn.Sequential): # Kế thừa nn.Sequential thay vì nn.Module để làm phẳng tên biến
    def __init__(self, in_channels):
        super().__init__(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

class MotifBackbone(nn.Module):
    """
    Advanced Backbone following Pham Qui Luan's ResMaskingNet structure.
    NÂNG CẤP LÊN RESNET152 ĐỂ KHỚP VỚI CHECKPOINT.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        
        # Load cấu trúc ResNet152 chuẩn từ PyTorch để hứng trọng số
        import torchvision.models as models
        resnet = models.resnet152(weights=None)
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.Identity() # Giữ nguyên để output là 6x6
        
        # ResNet152 Layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # PHAM QUI LUAN'S MASKING BLOCKS (Cập nhật số kênh cho ResNet152)
        self.mask1 = MaskBlock(256)   # Layer 1 của ResNet152 xuất ra 256
        self.mask2 = MaskBlock(512)   # Layer 2 xuất ra 512
        self.mask3 = MaskBlock(1024)  # Layer 3 xuất ra 1024
        self.mask4 = MaskBlock(2048)  # Layer 4 xuất ra 2048
        
        # Reducers for Multi-Scale Fusion (Sẽ được quản lý ở MotifGraphModel)
        # self.dim_reducer đã bị loại bỏ ở đây để chuyển sang MotifGraphModel

    def load_pretrained_cnn(self, checkpoint_path):
        import os
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: CNN checkpoint {checkpoint_path} not found. Skipping.")
            return

        print(f"Loading pretrained CNN from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
            
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint.get('net', checkpoint)))
            
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('backbone.', '').replace('resnet.', '').replace('net.', '')
            
            if name in model_dict:
                if v.shape == model_dict[name].shape:
                    pretrained_dict[name] = v
                    
                # BẢN VÁ LỖI CỨU LỚP CONV1 (Cắt 7x7 xuống 3x3 và ép về Grayscale)
                elif 'conv1.weight' in name and v.shape[1] == 3 and model_dict[name].shape[1] == 1:
                    if v.shape[2:] == (7, 7) and model_dict[name].shape[2:] == (3, 3):
                        print(f"[*] Extracting center 3x3 from 7x7 kernel and adapting to 1 channel for {name}...")
                        # Cắt lõi 3x3 từ tâm bộ lọc 7x7
                        center_v = v[:, :, 2:5, 2:5]
                        # Ép 3 kênh RGB về 1 kênh
                        pretrained_dict[name] = center_v.mean(dim=1, keepdim=True)
                    elif v.shape[2:] == model_dict[name].shape[2:]:
                        print(f"[*] Adapting {name} from 3 channels to 1 channel (Grayscale)...")
                        pretrained_dict[name] = v.mean(dim=1, keepdim=True)
                else:
                    pass
                    
        if len(pretrained_dict) == 0:
            print("WARNING: No matching keys found. Check your checkpoint architecture.")
        else:
            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} matching layers.")
            self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 + Mask 1
        x = self.layer1(x)
        m1 = self.mask1(x)
        x = x * (1 + m1)

        # Layer 2 + Mask 2
        x = self.layer2(x)
        m2 = self.mask2(x)
        x = x * (1 + m2)

        # Layer 3 + Mask 3 (Dùng cho Multi-scale)
        x = self.layer3(x)
        m3 = self.mask3(x)
        x3 = x * (1 + m3)

        # Layer 4 + Mask 4
        x = self.layer4(x)
        m4 = self.mask4(x)
        x4 = x * (1 + m4)
        
        return x3, x4

class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer (GAT) for small graphs.
    """
    def __init__(self, in_dim, out_dim, heads=8):
        super().__init__()
        self.heads = heads #Số lượng attention head để tăng khả năng biểu diễn
        self.d_k = out_dim // heads # Dimension per head, đảm bảo out_dim chia hết cho heads
        # UPDATE: increase heads for richer attention capacity
        
        self.q_lin = nn.Linear(in_dim, out_dim) #Linear layers có tác dụng biến đổi đặc trưng đầu vào thành không gian đặc trưng mới phù hợp cho attention
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(out_dim, out_dim)
        # UPDATE: edge-aware attention + learnable adjacency gating
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )
        # UPDATE: residual + norm + attention dropout
        self.attn_drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        # x: (B, N, in_dim) B: batch size,N: number of nodes,in_dim: input dimension, adj: (B, N, N) matrix kề của đồ thị, có thể là binary hoặc weighted
        B, N, _ = x.shape
        
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2) 
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) #matmul có tác dụng tính toán điểm số attention giữa các node, chia cho sqrt(d_k) để ổn định gradient khi d_k lớn

        # UPDATE: edge-aware bias + learnable gate per edge
        x_i = x.unsqueeze(2).expand(B, N, N, -1)
        x_j = x.unsqueeze(1).expand(B, N, N, -1)
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        edge_gate = torch.sigmoid(self.edge_gate(edge_feat)).squeeze(-1) # (B, N, N)
        edge_bias = self.edge_bias(edge_feat).squeeze(-1) # (B, N, N)

        if adj is not None:
            edge_gate = edge_gate * adj

        scores = scores + edge_bias.unsqueeze(1)
        scores = scores * edge_gate.unsqueeze(1)

        if adj is not None:
            # FIX: Dùng -1e4 thay vì -1e9 để tránh lỗi overflow khi dùng AMP (Float16)
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v) # (B, H, N, d_k)

        out = out.transpose(1, 2).contiguous().view(B, N, -1) #kết hợp các head lại với nhau bằng cách transpose và reshape
        out = self.out_lin(out)
        out = self.norm(out + x) # UPDATE: residual + norm
        return F.relu(out)

class GraphMotifModule(nn.Module):
    """
    Research-grade Structured Graph Matching Module.
    suitable for publication in CVPR/ICCV.
    
    Features:
    - Combined Node & Edge Structure Matching
    - Learnable weighting between Node/Edge similarity
    - Low-rank Factorized Motif Topology
    - Fully vectorized structure alignment using einsum
    - Interpretability via attention and activation maps
    """
    def __init__(self, num_classes, motifs_per_class, K, C, top_k=None, rank=4):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K = K  
        self.C = C  
        self.top_k = top_k
        
        # 1. Motif Representation: (Classes, Motifs, K, Dim)
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, C)) 
        # motifs là các mẫu con đồ thị học được, mỗi motif đại diện cho một cấu trúc đặc trưng có thể xuất hiện trong biểu cảm khuôn mặt, được tổ chức theo lớp và số lượng motif trên mỗi lớp
        nn.init.xavier_uniform_(self.motifs)
        # Khởi tạo các motif bằng phương pháp Xavier để đảm bảo phân phối hợp lý của trọng số, giúp quá trình huấn luyện ổn định và hiệu quả hơn.
        # 2. Factorized Motif Topology: (Classes, Motifs, K, Rank)
        # Motif edges A = U @ U^T
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
        # Khởi tạo ma trận low-rank để biểu diễn cấu trúc cạnh của motif, giúp giảm số lượng tham số và tăng khả năng tổng quát hóa của mô hình khi học các cấu trúc đồ thị phức tạp.
        nn.init.xavier_uniform_(self.motif_low_rank)
        
        # 3. Learnable weights for Node vs Edge similarity
        self.alpha = nn.Parameter(torch.zeros(1)) # Node similarity weight (logit scale)
        self.beta = nn.Parameter(torch.zeros(1))  # Edge similarity weight (logit scale)
        
        # 4. Stability parameters
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        # Temperature parameter để điều chỉnh độ mềm của phân phối attention, giúp quá trình huấn luyện ổn định hơn và tránh overfitting vào các motif cụ thể.
    def compute_diversity_loss(self):
        """
        Orthogonality constraint for motifs.
        L = || M M^T - I ||
        """
        m = self.motifs.view(self.num_classes, self.motifs_per_class, -1)
        m = F.normalize(m, dim=-1)
        sim = torch.matmul(m, m.transpose(1, 2))
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        return torch.norm(sim - eye, p='fro', dim=(1, 2)).mean()

    def forward(self, region_features, adj=None, return_attention=False):
        """
        Args:
            region_features: (B, K, C)
            adj: (B, K, K) input graph adjacency
            
        Returns:
            logits: (B, num_classes)
            motif_scores: (B, num_classes, motifs_per_class)
            metadata: dict containing attention and activation maps
        """
        B, K, C = region_features.shape # B: batch size, K: number of regions, C: feature dimension
        L, M = self.num_classes, self.motifs_per_class # L: number of classes, M: motifs per class
        
        # 1. Normalize Inputs
        region_features = F.normalize(region_features, p=2, dim=-1)
        motifs = F.normalize(self.motifs, p=2, dim=-1)
        
        # 2. Soft node alignment (cross-graph matching)
        sim_align = torch.einsum('bkc,lmjc->blmkj', region_features, motifs)
        align_weights = F.softmax(sim_align, dim=-1)
        aligned_motifs = torch.einsum('blmkj,lmjc->blmkc', align_weights, motifs)
        # UPDATE: Re-normalize aligned motifs to maintain true cosine similarity space
        aligned_motifs = F.normalize(aligned_motifs, p=2, dim=-1)
        node_sim = torch.einsum('bkc,blmkc->blmk', region_features, aligned_motifs)
        
        # 3. Edge Structure Matching (Pairwise differences) - Memory Efficient Formulation
        # Mathematically equivalent to: (Ri - Rj) * (Mi - Mj) = Ri*Mi + Rj*Mj - Ri*Mj - Rj*Mi
        # cross_sim: (B, L, M, K, K) where cross_sim[b,l,m,i,j] = Ri * Mj
        cross_sim = torch.einsum('bic,blmjc->blmij', region_features, aligned_motifs)
        node_sim_i = node_sim.unsqueeze(-1) # (B, L, M, K, 1) -> Ri*Mi
        node_sim_j = node_sim.unsqueeze(-2) # (B, L, M, 1, K) -> Rj*Mj
        
        # edge_sim_raw: (B, L, M, K, K)
        edge_sim_raw = node_sim_i + node_sim_j - cross_sim - cross_sim.transpose(-1, -2)
        # structure-preserving aggregation with node-attn outer product
        tau = F.softplus(self.temperature)
        node_attn = F.softmax(node_sim / tau.clamp(min=1e-3), dim=-1)
        edge_weights = node_attn.unsqueeze(-2) * node_attn.unsqueeze(-1)
        edge_weights = edge_weights / edge_weights.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        edge_sim = (edge_sim_raw * edge_weights).sum(dim=(-1, -2))
        
        # 4. Topology matching using Low-Rank Motif Edges
        # motif_adj: (L, M, K, K)
        motif_adj = torch.matmul(self.motif_low_rank, self.motif_low_rank.transpose(-1, -2))
        motif_adj = F.softmax(motif_adj, dim=-1)
        
        topo_sim = 0
        if adj is not None:
            topo_sim = (motif_adj.unsqueeze(0) * edge_weights).sum(dim=(-1, -2))
            
        # 5. Combined Similarity
        # s_node: (B, L, M, K)
        s_node = node_sim
        # s_struct: (B, L, M)
        s_struct = edge_sim + topo_sim
        
        # Aggregate node similarity per motif
        node_sim_agg = torch.sum(node_attn * s_node, dim=-1) # (B, L, M)
        
        # Final combined score: (B, L, M)
        # Learnable balance between node and structural information
        w_node = torch.sigmoid(self.alpha)
        w_edge = torch.sigmoid(self.beta)
        S = w_node * node_sim_agg + w_edge * s_struct
        
        # 6. Smooth Selection via logsumexp
        logits = torch.logsumexp(S / tau.clamp(min=1e-3), dim=-1)
        
        # 7. Entropy for stability
        entropy = -(node_attn * torch.log(node_attn + 1e-8)).sum(dim=-1).mean()
        self._latest_attn_entropy = entropy
        
        if return_attention:
            metadata = {
                "node_attention": node_attn,
                "motif_activations": S,
                "edge_sim_matrix": edge_sim_raw
            }
            return logits, S, metadata
        return logits, S

class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 16)  # UPDATE: more motif diversity
        self.top_k = config.get('top_k', 6)  # UPDATE: more candidate diversity for 4x4
        self.temperature = config.get('motif_tau', 0.1) 
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
        # A. MULTI-SCALE FEATURE FUSION COMPONENTS
        # Ép Layer 3 (1024 channels) và Layer 4 (2048 channels) về feat_dim
        self.reducer_l3 = nn.Sequential(
            nn.Conv2d(1024, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.reducer_l4 = nn.Sequential(
            nn.Conv2d(2048, self.feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # UPDATE: Load custom CNN checkpoint if provided
        pretrained_cnn_path = config.get('pretrained_cnn_path', "")
        if pretrained_cnn_path != "":
            self.backbone.load_pretrained_cnn(pretrained_cnn_path)
            
        # 4. Global Branch: Capture overall face context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        dropout = config.get('dropout', 0.3)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(dropout),  # BUG FIX: đọc từ config thay vì hardcode 0.3
            nn.Linear(128, self.num_classes)
        )
        
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_dim, self.feat_dim),
            GraphAttentionLayer(self.feat_dim, self.feat_dim)
        ])
        
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 64), # UPDATE: Global-Guided Input
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Tanh() 
        )
        self.offset_amplitude = float(config.get('offset_amplitude', 0.35))  # UPDATE: stabilize 4x4 offsets
        
        self.pos_embed = nn.Parameter(torch.randn(1, 16, self.feat_dim))  # UPDATE: 4x4 grid
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.register_buffer('grid_adj', self._generate_4x4_grid_adj())  # UPDATE: 4x4 grid
        
        self.motif_module = GraphMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            K=16, # UPDATE: 4x4 region nodes
            C=self.feat_dim,
            top_k=self.top_k
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 1.0)
        
        # CƠ CHẾ GATED FUSION TỪ BÀI BÁO
        self.gate = nn.Sequential(
            nn.Linear(self.num_classes * 2, self.num_classes),
            # Đã xóa LayerNorm để tránh nhiễu trên vector 7 chiều (num_classes=7)
            nn.Sigmoid()
        )
        self.gate_drop = nn.Dropout(dropout) # BUG FIX: đọc từ config thay vì hardcode 0.3
        
        # FIX 1: Bahdanau Feature Attention thay vi Static cand_query
        # cand_query cũ: vector tĩnh 7 chiều → luôn chọn điểm dự đoán Happy cao nhất, bất kể chất lượng ảnh
        # cand_attn_net mới: đánh giá UY TÍN từ Feature nguyên thủy → điểm nào “rõ nét” hơn được trọng số cao hơn
        self.cand_attn_net = nn.Sequential(
            nn.Linear(self.feat_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Khởi tạo Motif Consistency Loss
        self.motif_consistency_loss = MotifConsistencyLoss(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            tau=self.temperature
        )

    def compute_motif_diversity_loss(self):
        # Point 1: Replace motif_bank with motif_module
        m = self.motif_module.motifs 
        C, M, N, D = m.shape
        m_flat = m.view(C, M, -1) 
        m_flat = F.normalize(m_flat, dim=-1)
        
        sim_intra = torch.matmul(m_flat, m_flat.transpose(1, 2))
        eye = torch.eye(M, device=m.device).unsqueeze(0)
        l_intra = (torch.abs(sim_intra) * (1 - eye)).mean()
        
        class_centers = m_flat.mean(dim=1) 
        class_centers = F.normalize(class_centers, dim=-1)
        sim_inter = torch.matmul(class_centers, class_centers.transpose(0, 1))
        eye_c = torch.eye(C, device=m.device)
        l_inter = (sim_inter * (1 - eye_c)).mean()
        
        return l_intra + 1.0 * l_inter

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats, landmarks_48, statuses):
        B, C_feat, _, _ = feat_map.shape
        num_cands = 10 # BỘ 10 ĐIỂM VÀNG
        
        # Tọa độ neo sinh học mặc định trên lưới 12x12 (dùng khi failed)
        default_centers_y = torch.tensor([1, 3, 2, 2, 4, 4, 6, 9, 9, 11], device=feat_map.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        default_centers_x = torch.tensor([5, 5, 3, 7, 3, 7, 5, 3, 7, 5], device=feat_map.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        if landmarks_48 is None:
            centers_x = default_centers_x
            centers_y = default_centers_y
            centers_x_float = default_centers_x.float()
            centers_y_float = default_centers_y.float()
        else:
            # ĐẠI PHẪU: TÁCH ĐÔI KHÔNG GIAN (Discrete vs Continuous)
            # Vấn đề cũ: clamp() "thiến" tọa độ âm → 2 điểm đè lên nhau → cấu trúc hình học bị hủy
            
            # NHÁNH 1: FLOAT (KHÔNG CLAMP) — giữ nguyên hình học thực cho F.grid_sample
            # grid_sample xử lý số âm tự nhiên = Zero Padding (trả về vector 0 = "không nhìn thấy")
            csv_centers_x_float = landmarks_48[:, :, 0] / 48.0 * W
            csv_centers_y_float = landmarks_48[:, :, 1] / 48.0 * H
            
            # NHÁNH 2: LONG (CÓ CLAMP) — để torch.gather không văng IndexError
            # Chỉ dùng để lấy node features, không ảnh hưởng đến vùng sampling thực
            csv_centers_x_long = torch.clamp(csv_centers_x_float.long(), 0, W - 1)
            csv_centers_y_long = torch.clamp(csv_centers_y_float.long(), 0, H - 1)
            
            if statuses is not None:
                mask_success_float = statuses.view(B, 1).to(feat_map.device)
                mask_success_long  = statuses.view(B, 1).long().to(feat_map.device)
                
                # gather dùng bản LONG (an toàn index)
                centers_x = csv_centers_x_long * mask_success_long + default_centers_x * (1 - mask_success_long)
                centers_y = csv_centers_y_long * mask_success_long + default_centers_y * (1 - mask_success_long)
                
                # grid_sample dùng bản FLOAT (cho phép số âm bay ra ngoài → zero padding)
                centers_x_float = csv_centers_x_float * mask_success_float + default_centers_x.float() * (1 - mask_success_float)
                centers_y_float = csv_centers_y_float * mask_success_float + default_centers_y.float() * (1 - mask_success_float)
            else:
                centers_x = csv_centers_x_long
                centers_y = csv_centers_y_long
                centers_x_float = csv_centers_x_float
                centers_y_float = csv_centers_y_float
        
        # Tính index 1D cho từng ảnh trong batch, shape: (B, 10)
        center_indices = centers_y * W + centers_x
        
        # 2. Trích xuất Node Features cho TỪNG ẢNH bằng torch.gather
        # node_feats shape: (B, 144, 128)
        center_indices_expanded = center_indices.unsqueeze(-1).expand(-1, -1, C_feat)
        center_feats = torch.gather(node_feats, 1, center_indices_expanded) # (B, 10, 128)
        
        # 3. Global-Guided Offsets
        global_feat = feat_map.mean(dim=(2, 3)).unsqueeze(1).expand(-1, num_cands, -1) 
        combined_feats = torch.cat([center_feats, global_feat], dim=-1) 
        
        offsets = self.offset_predictor(combined_feats) * getattr(self, 'offset_amplitude', 0.2)
        self._latest_offsets = offsets.detach()  # BUG FIX: detach để tránh memory leak
        
        # BẢN VÁ LỊCH SỬ: Phân nhánh Success vs Failed
        if statuses is not None:
            mask_failed = (1.0 - statuses.view(B, 1, 1)).to(feat_map.device)
            offsets = offsets * mask_failed
        
        # 4. Tính toán Lưới lấy mẫu (Sampling Grid)
        rel_y, rel_x = torch.meshgrid(torch.linspace(-1.0, 1.0, 4), torch.linspace(-1.0, 1.0, 4), indexing='ij') 
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device).view(1, 1, 16, 2) 
        
        # Multi-Shape Graph Sampling
        scales_x = torch.tensor([1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 0.8, 1.0, 1.0, 0.8], device=feat_map.device, dtype=torch.float32)
        scales_y = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 1.2, 1.0, 1.0, 1.2], device=feat_map.device, dtype=torch.float32)
        scales = torch.stack([scales_x, scales_y], dim=-1).view(1, num_cands, 1, 2)
        
        # ĐẠI PHẪU: Dùng centers_x_float (KHÔNG CLAMP) để grid_sample nhận tọa độ thực
        # Nếu điểm nằm ngoài ảnh (y < 0, x < 0...): grid_sample tự trả về vector 0
        # → Mạng học được: "điểm này bị che khuất, không đáng tin" → dồn attention sang điểm khác
        c_x = (centers_x_float / (W - 1)) * 2 - 1.0
        c_y = (centers_y_float / (H - 1)) * 2 - 1.0
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(B, num_cands, 1, 2)
        
        patch_scale = 3.0 / W
        sampling_grid = centers_grid + offsets.unsqueeze(2) + (rel_grid * scales) * patch_scale
        # KHÔNG CLAMP sampling_grid nữa! grid_sample tự xử lý out-of-bounds = zero padding
        # (padding_mode='zeros' là default của F.grid_sample)
        sampling_grid = sampling_grid.view(B, num_cands * 16, 1, 2)
        
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 16).permute(0, 2, 3, 1) 
        
        # Update ma trận kề cho 10 điểm
        adj = self.grid_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)
        
        # Trả về tensor đầy đủ (B, 10) cho cả batch để dùng trong Bahdanau Attention
        return sampled_feats, adj, (centers_y, centers_x)

    def forward(self, x, return_selection=False, targets=None, landmarks=None, statuses=None):
        if targets is not None:
            self._latest_targets = targets
        else:
            self._latest_targets = None # BẮT BUỘC PHẢI THÊM DÒNG NÀY ĐỂ XÓA NHÃN CŨ
            
        # Handle TenCrop input: (B, 10, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            logits_list = []
            
            # XỬ LÝ CUỐN CHIẾU: Cho từng crop chạy qua model để chống Tràn RAM
            for t in range(T):
                crop_x = x[:, t, :, :, :] # Lấy crop thứ t, shape: (B, C, H, W)
                
                if targets is not None:
                    out = self.forward(crop_x, return_selection=return_selection, targets=targets, landmarks=landmarks, statuses=statuses)
                else:
                    out = self.forward(crop_x, return_selection=return_selection, landmarks=landmarks, statuses=statuses) 
                
                # Lưu lại kết quả
                if return_selection:
                    logits_list.append(out[0]) # out[0] là logits
                else:
                    logits_list.append(out)
                    
            # Tính trung bình dự đoán của 10 crop
            mean_logits = torch.stack(logits_list, dim=1).mean(dim=1)
            
            if return_selection:
                return mean_logits, out[1], out[2], out[3]
            else:
                return mean_logits

        # Handle TenCrop ... (logic TenCrop giữ nguyên)
        
        B = x.shape[0]
        
        # 1. TRÍCH XUẤT ĐA QUY MÔ (Multi-scale Extraction)
        x3, x4 = self.backbone(x) # x3: (B, 1024, 12, 12), x4: (B, 2048, 6, 6)
        
        # 2. FUSION LAYER 3 & 4 (ROI cao: Lấy thêm chi tiết từ L3)
        feat_map_l3 = self.reducer_l3(x3)                   # (B, feat_dim, 12, 12)
        feat_map_l4 = self.upsample(self.reducer_l4(x4))   # (B, feat_dim, 12, 12)
        feat_map = feat_map_l3 + feat_map_l4                # Fusion: (B, feat_dim, 12, 12)
        _, _, H, W = feat_map.shape
        
        # 3. Global Branch prediction (Vẫn dùng Layer 4 nguyên bản cho Global context)
        logits_global = self.global_fc(self.global_pool(x4))
        self._latest_logits_global = logits_global # Lưu cho DGS Loss
        
        # FIX 2: Bơm nhận thức không gian vào GAT (GPS cho Graph)
        # Cũ: cắt [:-2] bỏ tọa độ → GAT mù, không biết Trán khác Cằm
        # Mới: giữ nguyên (feat_dim+2), project xuống feat_dim → GAT biết “Bên trái, Trên, Dưới”
        nodes_with_coords, adj = self._get_global_graph(feat_map)
        node_feats = nodes_with_coords  # giữ nguyên cả tọa độ (feat_dim + 2)
        if not hasattr(self, 'proj_node_with_coords') or self.proj_node_with_coords.in_features != node_feats.shape[-1]:
            self.proj_node_with_coords = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
        node_feats = self.proj_node_with_coords(node_feats)
            
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
            
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(feat_map, H, W, node_feats, landmarks, statuses)
        num_cands = candidates.shape[1]
        
        # Advanced Motif Module Forward
        # 1. Prepare candidate subgraphs: (B*num_cands, 16, Dim)
        flat_cands = candidates.reshape(B * num_cands, 16, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed
        
        # 2. Prepare candidate adjacencies: (B*num_cands, 16, 16)
        flat_adjs = cand_adjs.reshape(B * num_cands, 16, 16)
        
        # 3. Match against Learnable Motifs (Research Grade)
        logits_cand, motif_scores_cand, metadata = self.motif_module(flat_cands, adj=flat_adjs, return_attention=True)
        
        # 4. Aggregate across all candidate subgraphs
        # logits_cand: (B*num_cands, num_classes)
        logits_cand = logits_cand.view(B, num_cands, self.num_classes)
        
        # FIX 1: Đánh giá uy tín từ Post-GNN Features của từng điểm landmark
        # centers = (centers_y, centers_x) — tensor (B, 10) mỗi cái
        centers_y_idx, centers_x_idx = centers  # (B, 10) each
        center_indices_attn = (centers_y_idx * W + centers_x_idx).long()  # (B, 10)
        center_indices_attn = center_indices_attn.unsqueeze(-1).expand(-1, -1, self.feat_dim)  # (B, 10, feat_dim)
        post_gnn_center_feats = torch.gather(node_feats, 1, center_indices_attn)  # (B, 10, feat_dim)
        
        cand_scores = self.cand_attn_net(post_gnn_center_feats).squeeze(-1)  # (B, 10)
        cand_tau = 1.0
        attn_weights = F.softmax(cand_scores / cand_tau, dim=1).unsqueeze(-1)
        
        logits_motif = torch.sum(logits_cand * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        self._latest_logits_motif = logits_motif # FIX: Lưu lại cho DGS Loss
        
        # GATED FUSION: Học cách kết hợp linh hoạt giữa đặc trưng cục bộ (Motif) và toàn cục (Global)
        gate_input = torch.cat([logits_motif, logits_global], dim=-1)
        gate_input = self.gate_drop(gate_input) # Dropout để ép mô hình chú ý cả 2 nhánh
        g = self.gate(gate_input) # Cổng ra quyết decision (B, num_classes)
        
        # Kết hợp có trọng số động
        logits = g * logits_motif + (1 - g) * logits_global
        
        # Point 2: Reshape for MotifConsistencyLoss (B, num_cands, num_classes * motifs_per_class)
        self._latest_scores = motif_scores_cand.view(B, num_cands, -1)
        # Relevance for Top-K visualization
        cand_relevance = cand_scores
        k = min(self.top_k, cand_relevance.size(1))
        if k > 0:
            _, top_k_idx = torch.topk(cand_relevance, k=k, dim=1)
        else:
            top_k_idx = torch.empty(B, 0, dtype=torch.long, device=cand_relevance.device)
        self._latest_top_k = top_k_idx
        self._latest_metadata = metadata
        
        if return_selection:
            return logits, top_k_idx, centers, self._latest_scores
            
        return logits

    def _get_global_graph(self, feat_map):
        B, C, H, W = feat_map.shape
        N = H * W
        
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        coords = torch.stack([x, y], dim=-1).to(feat_map.device).view(1, N, 2).expand(B, -1, -1)
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, N, C)
        nodes_with_coords = torch.cat([nodes, coords], dim=-1)
        
        nodes_norm = F.normalize(nodes, dim=-1)
        sim = torch.matmul(nodes_norm, nodes_norm.transpose(1, 2))
        
        k_neighbors = 4 
        topk_sim, topk_idx = torch.topk(sim, k=k_neighbors, dim=-1)
        
        adj = torch.zeros_like(sim)
        adj.scatter_(-1, topk_idx, topk_sim)
        
        return nodes_with_coords, adj

    def get_landmark_outputs(self):
        return getattr(self, '_latest_scores', None), getattr(self, '_latest_top_k', None)

    def get_landmark_aux_logits(self):
        return None

    def set_training_progress(self, progress):
        self.training_progress = progress
        
    def get_current_prior_strength(self):
        return 0.0

    def _generate_4x4_grid_adj(self):
        # UPDATE: 4x4 grid adjacency for K=16
        adj = torch.zeros(16, 16)
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4:
                            n_idx = ni * 4 + nj
                            adj[idx, n_idx] = 1.0
        return adj

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
            
        # 1. Motif Diversity (Orthogonality)
        l_div = self.motif_module.compute_diversity_loss()
        
        # 2. Attention Entropy (Prevent collapse)
        l_ent = getattr(self.motif_module, '_latest_attn_entropy', 0.0)
        
        # 3. Offset Regularization
        l_off = torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        
        aux_dict = {
            "motif_diversity": l_div,
            "attn_entropy": l_ent,
            "offset_reg": l_off,
            "logits_global": self._latest_logits_global, # Gửi cho Trainer tính DGS
            "logits_motif": self._latest_logits_motif    # Gửi cho Trainer tính DGS
        }
        
        # 4. Kích hoạt Motif Consistency Loss tại đây
        if hasattr(self, '_latest_targets') and self._latest_targets is not None:
            progress = getattr(self, 'training_progress', 1.0)
            # Tạm tắt Motif Consistency trong Phase 1 (progress <= 0.065) vì Mixup trộn nhãn
            if self.training and progress <= 0.005:  # BUG FIX: đồng bộ threshold với trainer.py Phase 1
                l_motif_consist = torch.tensor(0.0, device=self._latest_scores.device)
            else:
                l_motif_consist = self.motif_consistency_loss(
                    self._latest_scores, 
                    self._latest_top_k, 
                    self._latest_targets
                )
            aux_dict["motif_consistency"] = l_motif_consist
            
        return aux_dict


    def _get_grid_graph(self, feat_map):
        """ Vectorized version of graph building """
        B, C, H, W = feat_map.shape
        N = H * W
        
        # Node features
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        coords = torch.stack([x, y], dim=-1).to(feat_map.device).view(1, N, 2).expand(B, -1, -1)
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, N, C)
        nodes_with_coords = torch.cat([nodes, coords], dim=-1)
        
        # Adjacency using 8-neighborhood mask + vectorized similarity
        # 1. Spatial mask
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(N, 2)
        dist_spatial = torch.cdist(grid_coords.float(), grid_coords.float(), p=float('inf'))
        mask = (dist_spatial <= 1).float().to(feat_map.device)
        
        # 2. Feature similarity
        dist_feat = torch.cdist(nodes, nodes) / math.sqrt(C)
        sim = torch.exp(-dist_feat)
        
        adj = sim * mask.unsqueeze(0)
        return nodes_with_coords, adj

if __name__ == "__main__":
    config = {
        'feat_dim': 64,
        'num_classes': 7,
        'motifs_per_class': 4,
        'top_k': 4
    }
    model = MotifGraphModel(config)
    
    # Test 4D
    dummy_img_4d = torch.randn(2, 1, 48, 48)
    out_4d = model(dummy_img_4d)
    print(f"4D Output shape: {out_4d.shape}") # (2, 7)
    
    # Test 5D (TenCrop)
    dummy_img_5d = torch.randn(2, 10, 1, 40, 40)
    out_5d = model(dummy_img_5d)
    print(f"5D Output shape: {out_5d.shape}") # (2, 7)