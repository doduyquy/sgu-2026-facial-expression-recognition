import os
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

class SpatialResidualMasking(nn.Module):
    """
    Lightweight Spatial Residual Masking Block.
    Generates a spatial attention mask and applies it via a residual connection.
    This suppresses background noise and highlights micro-expressions.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Bottleneck to reduce parameters
        reduced_channels = in_channels // 4
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mask = self.mask_generator(x)
        # Residual masking: x' = x + x * M
        return x + x * mask
class MotifBackbone(nn.Module):
    """
    Advanced Backbone with Pretrained ResNet18 for stronger feature extraction.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        import torchvision.models as models
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet18(pretrained=True)
            
        # Keep original 3-channel pretrained conv1 but change stride to 1 to keep spatial size 6x6
        self.conv1 = resnet.conv1
        self.conv1.stride = (1, 1)
        self.conv1.padding = (3, 3)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        
        # Skip maxpool to keep spatial size 6x6 at the end (48->48->24->12->6)
        self.maxpool = nn.Identity()
        
        self.layer1 = resnet.layer1 # 48x48
        self.layer2 = resnet.layer2 # 24x24
        self.layer3 = resnet.layer3 # 12x12
        self.layer4 = resnet.layer4 # 6x6, 512 channels
        
        self.residual_masking = SpatialResidualMasking(768)
        
        # Reduce dimension to expected feat_dim (128)
        self.dim_reducer = nn.Sequential(
            nn.Conv2d(768, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

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
            
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
            
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Clean common prefixes from RMN or generic wrappers
            name = k.replace('module.', '').replace('backbone.', '').replace('resnet.', '').replace('net.', '')
            
            if name in model_dict:
                if v.shape == model_dict[name].shape:
                    pretrained_dict[name] = v
                else:
                    # Skip layers with shape mismatch (e.g., conv1 7x7 vs 3x3)
                    pass
                    
        if len(pretrained_dict) == 0:
            print("WARNING: No matching keys found. Checkpoint format might be unsupported.")
        else:
            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} matching layers from CNN checkpoint.")
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        # Repeat 1-channel input to 3 channels to match pretrained conv1 weight shape
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        
        # Downsample x3 để khớp không gian x4 một cách động (dynamic spatial matching)
        x3_down = F.adaptive_avg_pool2d(x3, x4.shape[2:])
        x_combined = torch.cat([x3_down, x4], dim=1) # (B, 768, H, W)
        
        x = self.residual_masking(x_combined)
        x = self.dim_reducer(x)
        return x

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
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)

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
    def __init__(self, num_classes, motifs_per_class, K, C, top_k=None, rank=4, clip_embedding_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K = K  
        self.C = C  
        self.top_k = top_k
        
        # 1. Motif Representation: (Classes, Motifs, K, Dim)
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, C)) 
        nn.init.xavier_uniform_(self.motifs)
        
        # Vision-Language Grounding Setup
        clip_embeds = None
        target_path = clip_embedding_path
        if target_path is not None and not os.path.exists(target_path):
            # Fallback checks for local workspace run
            potential_fallbacks = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'clip_au_embeddings.pt'),
                'dataset/clip_au_embeddings.pt'
            ]
            for fb in potential_fallbacks:
                if os.path.exists(fb):
                    target_path = fb
                    break
                    
        if target_path is not None and os.path.exists(target_path):
            try:
                clip_embeds = torch.load(target_path, map_location='cpu')
            except Exception as e:
                print(f"[WARNING] Could not load clip embeddings from {target_path}: {e}")
        
        if clip_embeds is not None:
            self.register_buffer('clip_au_embeds', clip_embeds) # (num_classes, M_ground, 512)
            self.text_projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, C),
                nn.LayerNorm(C)
            )
            # Khởi tạo ban đầu cho self.motifs:
            # Partial Grounding: mồi các motifs đầu bằng projected text embeddings, các motifs sau hoàn toàn tự do
            with torch.no_grad():
                proj_text = self.text_projection(clip_embeds) # (num_classes, M_ground, C)
                
                # Lưu mỏ neo CỐ ĐỊNH cho việc grounding
                self.register_buffer('anchor_motifs', proj_text.clone().detach())
                
                M_ground = clip_embeds.shape[1] # Số lượng semantic prompts (ví dụ: 8)
                proj_text_expanded = proj_text.unsqueeze(2).expand(-1, -1, K, -1)
                
                # Mồi M_ground motifs đầu
                self.motifs.data[:, :M_ground].copy_(proj_text_expanded + 0.05 * torch.randn_like(self.motifs.data[:, :M_ground]))
                # Các motifs sau được khởi tạo ngẫu nhiên (xavier_uniform_) hoàn toàn tự do
                nn.init.xavier_uniform_(self.motifs.data[:, M_ground:])
                
            # XÓA HOÀN TOÀN projection layer để dọn dẹp Graph và VRAM
            del self.text_projection
            self.text_projection = None
        else:
            self.register_buffer('clip_au_embeds', None)
            self.register_buffer('anchor_motifs', None)
            self.text_projection = None
            
        # 2. Factorized Motif Topology: (Classes, Motifs, K, Rank)
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
        nn.init.xavier_uniform_(self.motif_low_rank)
        
        # 3. Learnable weights for Node vs Edge similarity
        self.alpha = nn.Parameter(torch.zeros(1)) # Node similarity weight (logit scale)
        self.beta = nn.Parameter(torch.zeros(1))  # Edge similarity weight (logit scale)
        
        # 4. Trọng số học được cho K nodes (Khởi tạo bằng 0 để lúc đầu softmax ra đều nhau)
        self.node_importance = nn.Parameter(torch.zeros(1, 1, K))
        
        # 5. Stability parameters
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def compute_vision_language_grounding_loss(self):
        if getattr(self, 'anchor_motifs', None) is None:
            return torch.tensor(0.0, device=self.motifs.device)
            
        # Partial Grounding: Chỉ áp dụng grounding loss lên M_ground motifs đầu tiên
        M_ground = self.anchor_motifs.shape[1]
        motifs_to_ground = self.motifs[:, :M_ground, :, :] # (L, M_ground, K, C)
        anchors_to_ground = self.anchor_motifs # (L, M_ground, C)
        
        # Weighted Pooling thay vì Mean Pooling
        attn = F.softmax(self.node_importance, dim=-1) # (1, 1, K)
        motif_center = (motifs_to_ground * attn.unsqueeze(-1)).sum(dim=2) # (L, M_ground, C)
        
        sim = F.cosine_similarity(motif_center, anchors_to_ground.to(self.motifs.device), dim=-1)
        return (1.0 - sim).mean()

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
        
        # UPDATE: Load custom CNN checkpoint if provided
        pretrained_cnn_path = config.get('pretrained_cnn_path', "")
        if pretrained_cnn_path != "":
            self.backbone.load_pretrained_cnn(pretrained_cnn_path)
            
        # 4. Global Branch: Capture overall face context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        
        clip_embedding_path = config.get('clip_embedding_path', None)
        self.motif_module = GraphMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            K=16, # UPDATE: 4x4 region nodes
            C=self.feat_dim,
            top_k=self.top_k,
            clip_embedding_path=clip_embedding_path
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 1.0)
        # Weight for combining Motif and Global logits
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # Learnable query for candidate-level attention
        self.cand_query = nn.Parameter(torch.randn(1, 1, self.num_classes))
        nn.init.xavier_uniform_(self.cand_query)

        # Khởi tạo Motif Consistency Loss
        self.motif_consistency_loss = MotifConsistencyLoss(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            tau=self.temperature
        )

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        B, C_feat, _, _ = feat_map.shape
        
        # Base sampling grid (4 anatomical priors representing Eyebrow, Left Eye, Right Eye, Mouth)
        centers = [(1, 2), (2, 1), (2, 4), (4, 3)]
        center_indices = [i * W + j for i, j in centers]
        center_indices = torch.tensor(center_indices, device=feat_map.device)
        num_cands = len(center_indices)
        
        center_feats = node_feats[:, center_indices, :] 
        
        # UPDATE: Global-Guided Offsets
        global_feat = feat_map.mean(dim=(2, 3)) # Global Average Pooling (B, C)
        global_feat = global_feat.unsqueeze(1).expand(-1, num_cands, -1) # (B, num_cands, C)
        combined_feats = torch.cat([center_feats, global_feat], dim=-1) # (B, num_cands, 2C)
        
        offsets = self.offset_predictor(combined_feats) * self.offset_amplitude  # UPDATE: scale offsets
        # Point 3: Stabilize offset predictor with regularization
        self._latest_offsets = offsets
        
        rel_y, rel_x = torch.meshgrid(
            torch.linspace(-1.5, 1.5, 4),
            torch.linspace(-1.5, 1.5, 4),
            indexing='ij'
        )  # UPDATE: 4x4 grid offsets
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device) 
        rel_grid = rel_grid.view(1, 1, 16, 2) 
        
        c_y = (center_indices // W).float() / (H - 1) * 2 - 1
        c_x = (center_indices % W).float() / (W - 1) * 2 - 1
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(1, num_cands, 1, 2) 
        
        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W-1))
        sampling_grid = sampling_grid.view(B, num_cands * 16, 1, 2)
        
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 16).permute(0, 2, 3, 1) 
        
        adj = self.grid_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)
        
        centers_coords = []
        for idx in center_indices:
            centers_coords.append((idx // W, idx % W))
            
        return sampled_feats, adj, centers_coords

    def forward(self, x, return_selection=False, targets=None):
        if targets is not None:
            self._latest_targets = targets
            
        # Handle TenCrop input: (B, 10, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            # Recursive call to handle all crops (expand targets to match B*T)
            if targets is not None:
                targets_expanded = targets.unsqueeze(1).expand(-1, T).reshape(-1)
                logits = self.forward(x, targets=targets_expanded)
            else:
                logits = self.forward(x) 
            # Average predictions across all 10 crops
            return logits.view(B, T, -1).mean(dim=1)

        B = x.shape[0]
        
        feat_map = self.backbone(x) # (B, C, H, W)
        _, _, H, W = feat_map.shape
        
        # 4. Global Branch prediction
        logits_global = self.global_fc(self.global_pool(feat_map))
        
        # Motif Branch
        nodes_with_coords, adj = self._get_global_graph(feat_map)
        node_feats = nodes_with_coords[:, :, :-2]
        if node_feats.shape[-1] != self.feat_dim:
            if not hasattr(self, 'proj_node'):
                self.proj_node = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
            node_feats = self.proj_node(node_feats)
            
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
            
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(feat_map, H, W, node_feats)
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
        
        # Point 5: Candidate-level attention using learnable query
        cand_scores = (logits_cand * self.cand_query).sum(dim=-1) # (B, num_cands)
        cand_tau = 1.0
        attn_weights = F.softmax(cand_scores / cand_tau, dim=1).unsqueeze(-1) 
        
        logits_motif = torch.sum(logits_cand * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        
        # Final combined logits
        logits = logits_motif + torch.sigmoid(self.alpha) * logits_global
        
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
            "offset_reg": l_off
        }
        
        # 4. Kích hoạt Motif Consistency Loss tại đây
        if hasattr(self, '_latest_targets') and self._latest_targets is not None:
            progress = getattr(self, 'training_progress', 1.0)
            # Tạm tắt Motif Consistency trong Phase 1 (progress <= 0.05) vì Mixup trộn nhãn
            if self.training and progress <= 0.06:
                l_motif_consist = torch.tensor(0.0, device=self._latest_scores.device)
            else:
                l_motif_consist = self.motif_consistency_loss(
                    self._latest_scores, 
                    self._latest_top_k, 
                    self._latest_targets
                )
            aux_dict["motif_consistency"] = l_motif_consist
            
        # 5. Kích hoạt Vision-Language Grounding Loss
        if hasattr(self.motif_module, 'compute_vision_language_grounding_loss'):
            aux_dict["vision_language_grounding"] = self.motif_module.compute_vision_language_grounding_loss()
            
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
        'motifs_per_class': 16,
        'top_k': 4,
        'clip_embedding_path': 'dataset/clip_au_embeddings.pt'
    }
    model = MotifGraphModel(config)
    
    # Test 4D
    dummy_img_4d = torch.randn(2, 1, 48, 48)
    out_4d = model(dummy_img_4d, targets=torch.tensor([0, 3]))
    print(f"4D Output shape: {out_4d.shape}") # (2, 7)
    
    aux_losses = model.get_aux_losses()
    print("Auxiliary Losses:")
    for name, loss in aux_losses.items():
        print(f"  - {name}: {loss.item():.4f}" if isinstance(loss, torch.Tensor) else f"  - {name}: {loss}")
    
    # Test 5D (TenCrop)
    dummy_img_5d = torch.randn(2, 10, 1, 40, 40)
    out_5d = model(dummy_img_5d)
    print(f"5D Output shape: {out_5d.shape}") # (2, 7)