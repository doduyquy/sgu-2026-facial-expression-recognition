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
    Advanced Backbone with Residual connections and CBAM.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 24x24
        )
        
        # Residual Block 1
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.cbam1 = CBAM(64)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 12x12
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Residual Block 2
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.cbam2 = CBAM(128)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, feat_dim, kernel_size=3, stride=2, padding=1), # 6x6
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )
        self.final_cbam = CBAM(feat_dim)

    def forward(self, x):
        x = self.conv1(x)
        
        identity = x
        x = self.res1(x)
        x = self.cbam1(x)
        x = F.relu(x + identity)
        
        x = self.down1(x)
        
        identity = x
        x = self.res2(x)
        x = self.cbam2(x)
        x = F.relu(x + identity)
        
        x = self.down2(x)
        x = self.final_cbam(x)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Edge-aware Graph Attention Layer (CVPR-level refactor).
    Incorporate edge-conditioned bias into self-attention.
    """
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        
        # Point 3: Learnable edge MLP to project adjacency into head-specific biases
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, heads),
            nn.LeakyReLU(0.2),
            nn.Linear(heads, heads)
        )
        
        self.out_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        # x: (B, N, in_dim), adj: (B, N, N)
        B, N, _ = x.shape
        
        # 1. Project nodes to multi-head queries, keys, values
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # 2. Content-based scores: (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Edge-aware bias (Point 1 & 2)
        if adj is not None:
            # bias: (B, N, N, H) -> (B, H, N, N)
            edge_bias = self.edge_mlp(adj.unsqueeze(-1)).permute(0, 3, 1, 2)
            scores = scores + edge_bias
            
            # Preserve hard masking for absolutely zero edges if desired
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)
            
        # 4. Attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (B, H, N, d_k)
        
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return F.relu(self.out_lin(out))

class GraphMotifModule(nn.Module):
    """
    Research-grade Structured Graph Matching Module.
    Advanced features: Sinkhorn Node Alignment, Symmetric Topology, Usage Regularization.
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
        
        # 3. Learnable weights for structural integration
        self.alpha = nn.Parameter(torch.zeros(1)) 
        self.beta = nn.Parameter(torch.zeros(1))  
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def sinkhorn(self, log_alpha, n_iters=3):
        """
        Approximate Sinkhorn normalization for node alignment (Requirement 1).
        log_alpha: (B, L, M, K, K)
        """
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return log_alpha.exp()

    def compute_diversity_loss(self):
        m = self.motifs.view(self.num_classes, self.motifs_per_class, -1)
        m = F.normalize(m, dim=-1)
        sim = torch.matmul(m, m.transpose(1, 2))
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        return torch.norm(sim - eye, p='fro', dim=(1, 2)).mean()

    def forward(self, region_features, adj=None, return_attention=False):
        B, K, C = region_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        # 1. Normalize Inputs
        region_features = F.normalize(region_features, p=2, dim=-1)
        motifs = F.normalize(self.motifs, p=2, dim=-1)
        
        # 2. Advanced Node Alignment using Sinkhorn (Requirement 1)
        # Compute all-to-all similarity: (B, L, M, K, K)
        # s_ij = <region_node_i, motif_node_j>
        node_sim_matrix = torch.einsum('bic,lmjc->blmij', region_features, motifs)
        
        tau = F.softplus(self.temperature).clamp(min=1e-3)
        # Perform approximate alignment
        P = self.sinkhorn(node_sim_matrix / tau, n_iters=3) # (B, L, M, K, K)
        
        # Aligned node similarity score
        node_sim_agg = torch.sum(P * node_sim_matrix, dim=(-1, -2)) # (B, L, M)
        
        # 3. Edge Structure Matching (Requirement 2 & 4)
        diff_R = region_features.unsqueeze(2) - region_features.unsqueeze(1)
        diff_M = motifs.unsqueeze(3) - motifs.unsqueeze(2)
        
        edge_sim_raw = torch.einsum('bijk,lmijk->blmij', diff_R, diff_M)
        edge_sim = edge_sim_raw.mean(dim=(-1, -2)) 
        
        # 4. Topology matching using Symmetric Motif Edges (Requirement 2)
        # Motif edges A = U @ U^T (Symmetric by design)
        motif_adj = torch.matmul(self.motif_low_rank, self.motif_low_rank.transpose(-1, -2))
        motif_adj = F.softmax(motif_adj, dim=-1)
        
        topo_sim = 0
        if adj is not None:
            topo_sim = torch.einsum('bij,lmij->blm', adj, motif_adj)
            
        # 5. Combined Similarity Selection
        S = torch.sigmoid(self.alpha) * node_sim_agg + torch.sigmoid(self.beta) * (edge_sim + topo_sim)
        
        # Final logits: smooth selection across motifs
        logits = torch.logsumexp(S / tau, dim=-1)
        
        # 6. Usage Regularization (Requirement 3)
        # Compute how often each motif is chosen per batch
        motif_usage = F.softmax(S / tau, dim=-1).mean(dim=0) # (L, M)
        usage_entropy = -(motif_usage * torch.log(motif_usage + 1e-8)).sum(dim=-1).mean()
        self._latest_usage_entropy = usage_entropy
        
        if return_attention:
            metadata = {
                "alignment_matrix": P,
                "motif_activations": S,
                "usage_entropy": usage_entropy
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
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Tanh() 
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
        if node_feats.shape[-1] != self.feat_dim:
            if not hasattr(self, 'proj_node'):
                self.proj_node = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
            node_feats = self.proj_node(node_feats)
            
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
            
        # Refactor: Create a graph-refined feature map for deformable sampling
        # This ensures motif matching depends on graph reasoning (Requirement 1)
        feat_map_refined = node_feats.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(feat_map_refined, H, W, node_feats)
        num_cands = candidates.shape[1]
        
        # Advanced Motif Module Forward
        # 1. Prepare candidate subgraphs: (B*num_cands, 9, Dim)
        flat_cands = candidates.reshape(B * num_cands, 9, -1)
        # Position embedding for subgraphs
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
        
        # 2. Motif Usage Regularization (Prevent collapse)
        l_usage = getattr(self.motif_module, '_latest_usage_entropy', 0.0)
        
        # 3. Offset Regularization
        l_off = torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        
        return {
            "motif_diversity": l_div,
            "usage_entropy": l_usage,
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