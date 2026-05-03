import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .CBAM import CBAM
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.CBAM import CBAM

class MotifBackbone(nn.Module):
    """
    Research-grade Multi-Scale Backbone (CVPR Style).
    Extracts features at 24x24, 12x12, 6x6 and fuses them via a feature pyramid.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        # Stage 0: 48x48 -> 24x24
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Stage 1: 24x24 (Mid-level textures)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128)
        )
        self.cbam1 = CBAM(128)
        
        # Stage 2: 12x12 (Object parts)
        self.down1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256)
        )
        self.cbam2 = CBAM(256)
        
        # Stage 3: 6x6 (Global shapes)
        self.down2 = nn.MaxPool2d(2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, feat_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, feat_dim)
        )
        self.cbam3 = CBAM(feat_dim)
        
        # Multi-scale Fusion Projections
        self.proj_mid = nn.Conv2d(256, feat_dim, kernel_size=1)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(feat_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f24 = self.stem(x)
        f24 = self.layer1(f24)
        f24 = self.cbam1(f24)
        
        f12 = self.down1(f24)
        f12 = self.layer2(f12)
        f12 = self.cbam2(f12)
        
        f6 = self.down2(f12)
        f6 = self.layer3(f6)
        f6 = self.cbam3(f6)
        
        # Multi-scale Fusion (Point 1)
        f12_proj = self.proj_mid(f12)
        # Upsample f6 to 12x12
        f6_up = F.interpolate(f6, size=f12.shape[2:], mode='bilinear', align_corners=True)
        
        # Gated fusion
        gate = self.fuse_gate(torch.cat([f12_proj, f6_up], dim=1))
        f_fused = gate * f12_proj + (1 - gate) * f6_up
        
        # Final output is matched to f6 resolution
        out = F.adaptive_avg_pool2d(f_fused, f6.shape[2:]) + f6
        return out

class GraphTransformerBlock(nn.Module):
    """
    Research-grade Graph Transformer Block (CVPR Style).
    Features: Pre-Norm, Residuals, FFN (GELU), and Relative Positional Encoding.
    """
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads
        
        # 1. Multi-head Attention Branch
        self.norm1 = nn.LayerNorm(dim)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        
        # Relative Positional Bias MLP (Point 3)
        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, heads)
        )
        
        self.attn_drop = nn.Dropout(dropout)
        self.out_lin = nn.Linear(dim, dim)
        
        # 2. Feed Forward Network Branch
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj, coords=None):
        # x: (B, N, dim), adj: (B, N, N), coords: (B, N, 2)
        B, N, C = x.shape
        
        # --- Multi-head Attention Branch (Pre-Norm) ---
        identity = x
        z = self.norm1(x)
        
        q = self.q_lin(z).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(z).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(z).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Relative Positional Encoding (Point 3)
        if coords is not None:
            # rel_pos: (B, N, N, 2)
            rel_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
            # bias: (B, N, N, H) -> (B, H, N, N)
            bias = self.pos_bias_mlp(rel_pos).permute(0, 3, 1, 2)
            scores = scores + bias
        
        if adj is not None:
            # Edge-aware masking (Point 5)
            # Use log-masking to maintain gradient flow
            scores = scores + torch.log(adj.unsqueeze(1) + 1e-9)
            
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_lin(out)
        
        x = identity + out # First Residual
        
        # --- FFN Branch (Pre-Norm) ---
        identity = x
        z = self.norm2(x)
        out = self.ffn(z)
        
        return identity + out # Second Residual

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
        nn.init.xavier_uniform_(self.motifs)
        
        # 2. Factorized Motif Topology: (Classes, Motifs, K, Rank)
        # Motif edges A = U @ U^T
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
        nn.init.xavier_uniform_(self.motif_low_rank)
        
        # 3. Learnable weights for Node vs Edge similarity
        self.alpha = nn.Parameter(torch.zeros(1)) # Node weight (logit scale)
        self.beta = nn.Parameter(torch.zeros(1))  # Edge weight (logit scale)
        
        # 4. Stability parameters
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
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
        B, K, C = region_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        # 1. Normalize Inputs
        region_features = F.normalize(region_features, p=2, dim=-1)
        motifs = F.normalize(self.motifs, p=2, dim=-1)
        
        # 2. Node Similarity matching: (B, L, M, K)
        node_sim = torch.einsum('bkc,lmkc->blmk', region_features, motifs)
        
        # 3. Edge Structure Matching (Pairwise differences)
        # diff_R: (B, K, K, C)
        diff_R = region_features.unsqueeze(2) - region_features.unsqueeze(1)
        # diff_M: (L, M, K, K, C)
        diff_M = motifs.unsqueeze(3) - motifs.unsqueeze(2)
        
        # Align structural relationships Ri-Rj with Mi-Mj
        # edge_sim_raw: (B, L, M, K, K)
        edge_sim_raw = torch.einsum('bijk,lmijk->blmij', diff_R, diff_M)
        edge_sim = edge_sim_raw.mean(dim=(-1, -2)) # (B, L, M)
        
        # 4. Topology matching using Low-Rank Motif Edges
        # motif_adj: (L, M, K, K)
        motif_adj = torch.matmul(self.motif_low_rank, self.motif_low_rank.transpose(-1, -2))
        motif_adj = F.softmax(motif_adj, dim=-1)
        
        topo_sim = 0
        if adj is not None:
            topo_sim = torch.einsum('bij,lmij->blm', adj, motif_adj)
            
        # 5. Combined Similarity
        # s_node: (B, L, M, K)
        s_node = node_sim
        # s_struct: (B, L, M)
        s_struct = edge_sim + topo_sim
        
        # Aggregate node similarity per motif
        # Point 5: Improved temperature usage
        tau = F.softplus(self.temperature).clamp(min=1e-3)
        
        node_attn = F.softmax(s_node / tau, dim=-1)
        node_sim_agg = torch.sum(node_attn * s_node, dim=-1) # (B, L, M)
        
        # Final combined score: (B, L, M)
        # Learnable balance between node and structural information
        w_node = torch.sigmoid(self.alpha)
        w_edge = torch.sigmoid(self.beta)
        S = w_node * node_sim_agg + w_edge * s_struct
        
        # 6. Smooth Selection via logsumexp
        logits = torch.logsumexp(S / tau, dim=-1)
        
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
        self.motifs_per_class = config.get('motifs_per_class', 8)
        self.top_k = config.get('top_k', 4) 
        self.temperature = config.get('motif_tau', 0.1) 
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
        # 4. Global Branch: Capture overall face context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3), # Point 6
            nn.Linear(self.feat_dim, self.num_classes)
        )
        
        self.gnn_layers = nn.ModuleList([
            GraphTransformerBlock(self.feat_dim, heads=4, dropout=0.2),
            GraphTransformerBlock(self.feat_dim, heads=4, dropout=0.2)
        ])
        
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim // 2, 2)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 9, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.register_buffer('grid_adj', self._generate_3x3_grid_adj())
        
        self.motif_module = GraphMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            K=9, # 3x3 region nodes
            C=self.feat_dim,
            top_k=self.top_k
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        # Weight for combining Motif and Global logits
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # Learnable query for candidate-level attention
        self.cand_query = nn.Parameter(torch.randn(1, 1, self.num_classes))
        nn.init.xavier_uniform_(self.cand_query)

    def compute_motif_diversity_loss(self):
        # Point 1: Replace motif_bank with motif_module
        m = self.motif_module.motifs 
        C, M, N, D = m.shape
        m_flat = m.view(C, M, -1) 
        m_flat = F.normalize(m_flat, dim=-1)
        
        sim_intra = torch.matmul(m_flat, m_flat.transpose(1, 2))
        eye = torch.eye(M, device=m.device).unsqueeze(0)
        l_intra = (sim_intra * (1 - eye)).mean()
        
        class_centers = m_flat.mean(dim=1) 
        class_centers = F.normalize(class_centers, dim=-1)
        sim_inter = torch.matmul(class_centers, class_centers.transpose(0, 1))
        eye_c = torch.eye(C, device=m.device)
        l_inter = (sim_inter * (1 - eye_c)).mean()
        
        return l_intra + 1.0 * l_inter

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        B, C_feat, _, _ = feat_map.shape
        
        center_indices = []
        for i in range(1, H-1):
            for j in range(1, W-1):
                center_indices.append(i * W + j)
        center_indices = torch.tensor(center_indices, device=feat_map.device)
        num_cands = len(center_indices)
        
        center_feats = node_feats[:, center_indices, :] 
        offsets = self.offset_predictor(center_feats) 
        # Point 3: Stabilize offset predictor with regularization
        self._latest_offsets = offsets
        
        rel_y, rel_x = torch.meshgrid(torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing='ij')
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device) 
        rel_grid = rel_grid.view(1, 1, 9, 2) 
        
        c_y = (center_indices // W).float() / (H - 1) * 2 - 1
        c_x = (center_indices % W).float() / (W - 1) * 2 - 1
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(1, num_cands, 1, 2) 
        
        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W-1))
        sampling_grid = sampling_grid.view(B, num_cands * 9, 1, 2)
        
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 9).permute(0, 2, 3, 1) 
        
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
            # Recursive call to handle all crops (targets already set)
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
        coords = nodes_with_coords[:, :, -2:] # (B, N, 2)
        
        if node_feats.shape[-1] != self.feat_dim:
            if not hasattr(self, 'proj_node'):
                self.proj_node = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
            node_feats = self.proj_node(node_feats)
            
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj, coords=coords)
            
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(feat_map, H, W, node_feats)
        num_cands = candidates.shape[1]
        
        # Advanced Motif Module Forward
        # 1. Prepare candidate subgraphs: (B*num_cands, 9, Dim)
        flat_cands = candidates.reshape(B * num_cands, 9, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed
        
        # 2. Prepare candidate adjacencies: (B*num_cands, 9, 9)
        flat_adjs = cand_adjs.reshape(B * num_cands, 9, 9)
        
        # 3. Match against Learnable Motifs (Research Grade)
        logits_cand, motif_scores_cand, metadata = self.motif_module(flat_cands, adj=flat_adjs, return_attention=True)
        
        # 4. Aggregate across all candidate subgraphs
        # logits_cand: (B*num_cands, num_classes)
        logits_cand = logits_cand.view(B, num_cands, self.num_classes)
        
        # Point 5: Candidate-level attention using learnable query
        cand_scores = (logits_cand * self.cand_query).sum(dim=-1) # (B, num_cands)
        cand_tau = 0.3
        attn_weights = F.softmax(cand_scores / cand_tau, dim=1).unsqueeze(-1) 
        
        logits_motif = torch.sum(logits_cand * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        
        # Final combined logits
        logits = logits_motif + torch.sigmoid(self.alpha) * logits_global
        
        # Point 2: Reshape for MotifConsistencyLoss (B, num_cands, num_classes * motifs_per_class)
        self._latest_scores = motif_scores_cand.view(B, num_cands, -1)
        # Relevance for Top-K visualization
        cand_relevance = cand_scores
        _, top_k_idx = torch.topk(cand_relevance, k=self.top_k, dim=1)
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
        
        # Point 4: Fix Top-K asymmetry
        adj = (adj + adj.transpose(-1, -2)) / 2
        
        # Add self-loops for numerical stability
        adj = adj + torch.eye(N, device=feat_map.device).unsqueeze(0)
        
        # Point 3: Symmetric Normalization A = D^-1/2 A D^-1/2
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d + 1e-9, -0.5)
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)
        
        return nodes_with_coords, adj

    def get_landmark_outputs(self):
        return getattr(self, '_latest_scores', None), getattr(self, '_latest_top_k', None)

    def get_landmark_aux_logits(self):
        return None

    def set_training_progress(self, progress):
        pass
        
    def get_current_prior_strength(self):
        return 0.0

    def _generate_3x3_grid_adj(self):
        adj = torch.zeros(9, 9)
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 3 and 0 <= nj < 3:
                            n_idx = ni * 3 + nj
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
        
        return {
            "motif_diversity": l_div,
            "attn_entropy": l_ent,
            "offset_reg": l_off
        }


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