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
    Simple Graph Attention Layer (GAT) for small graphs.
    """
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        
        self.out_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        # x: (B, N, in_dim), adj: (B, N, N)
        B, N, _ = x.shape
        
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply adjacency mask (binary or weighted)
        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (B, H, N, d_k)
        
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return F.relu(self.out_lin(out))

class CrossAttentionMatching(nn.Module):
    """
    Matches subgraphs to motifs using Cross-Attention (Soft Alignment).
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.q_lin = nn.Linear(feat_dim, feat_dim)
        self.k_lin = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, candidates, motifs):
        B_c, N, D = candidates.shape
        M, _, _ = motifs.shape
        
        # Project
        q = self.q_lin(candidates) 
        k = self.k_lin(motifs)    
        
        # (3) Matching Normalization: Use Cosine Similarity for stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        sim_matrix = torch.einsum('bid,mjd->bmij', q, k) 
        
        # Soft alignment
        align_cand = sim_matrix.max(dim=-1)[0].mean(dim=-1)
        align_motif = sim_matrix.max(dim=-2)[0].mean(dim=-1)
        
        return (align_cand + align_motif) / 2.0

class MotifBank(nn.Module):
    def __init__(self, num_classes=7, motifs_per_class=8, num_nodes=9, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_nodes = num_nodes
        
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, num_nodes, feat_dim))
        nn.init.xavier_uniform_(self.motifs)
        
        adj = self._generate_3x3_grid_adj()
        self.register_buffer('motif_adj', adj)
        
        rel_coords = self._generate_3x3_rel_coords()
        self.register_buffer('rel_coords', rel_coords)

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

    def _generate_3x3_rel_coords(self):
        y, x = torch.meshgrid(torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), indexing='ij')
        return torch.stack([x, y], dim=-1).view(9, 2) 

    def get_motifs(self):
        flat_motifs = self.motifs.view(-1, self.num_nodes, self.motifs.shape[-1])
        Total_Motifs = flat_motifs.shape[0]
        coords = self.rel_coords.unsqueeze(0).expand(Total_Motifs, -1, -1)
        motifs_with_coords = torch.cat([flat_motifs, coords], dim=-1)
        return motifs_with_coords, self.motif_adj

class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 8)
        self.top_k = config.get('top_k', 4) 
        self.temperature = config.get('motif_tau', 0.1) 
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_dim, self.feat_dim),
            GraphAttentionLayer(self.feat_dim, self.feat_dim)
        ])
        
        # 3.2 Deformable: Predict offsets
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Tanh() 
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, 9, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.motif_bank = MotifBank(
            num_classes=self.num_classes, 
            motifs_per_class=self.motifs_per_class,
            num_nodes=9,
            feat_dim=self.feat_dim
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)

    def compute_motif_diversity_loss(self):
        # (4) Enhanced Diversity Loss: Intra-class and Inter-class
        m = self.motif_bank.motifs # (C, M, 9, d)
        C, M, N, D = m.shape
        m_flat = m.view(C, M, -1) # (C, M, N*D)
        m_flat = F.normalize(m_flat, dim=-1)
        
        # Intra-class diversity (Make motifs within same class different)
        # (C, M, M)
        sim_intra = torch.matmul(m_flat, m_flat.transpose(1, 2))
        eye = torch.eye(M, device=m.device).unsqueeze(0)
        l_intra = (sim_intra * (1 - eye)).mean()
        
        # Inter-class separation (Make class centers different)
        class_centers = m_flat.mean(dim=1) # (C, N*D)
        class_centers = F.normalize(class_centers, dim=-1)
        sim_inter = torch.matmul(class_centers, class_centers.transpose(0, 1))
        eye_c = torch.eye(C, device=m.device)
        l_inter = (sim_inter * (1 - eye_c)).mean()
        
        return l_intra + 0.75 * l_inter

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        """ (2) Real Deformable Sampling using grid_sample """
        B, C_feat, _, _ = feat_map.shape
        N_nodes = H * W
        
        # Center indices (interior)
        center_indices = []
        for i in range(1, H-1):
            for j in range(1, W-1):
                center_indices.append(i * W + j)
        center_indices = torch.tensor(center_indices, device=feat_map.device)
        num_cands = len(center_indices)
        
        center_feats = node_feats[:, center_indices, :] # (B, Num_Centers, D)
        offsets = self.offset_predictor(center_feats) # (B, Num_Centers, 2) in [-1, 1]
        
        # Base grid for 3x3 patches
        # Relative grid: 3x3 around (0,0)
        rel_y, rel_x = torch.meshgrid(torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing='ij')
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device) # (3, 3, 2)
        rel_grid = rel_grid.view(1, 1, 9, 2) # (1, 1, 9, 2)
        
        # Map center indices to normalized grid coordinates [-1, 1]
        c_y = (center_indices // W).float() / (H - 1) * 2 - 1
        c_x = (center_indices % W).float() / (W - 1) * 2 - 1
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(1, num_cands, 1, 2) # (1, num_cands, 1, 2)
        
        # Full grid for sampling: (B, num_cands, 9, 2)
        # Each candidate gets a 3x3 grid centered at (center + offset)
        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W-1))
        sampling_grid = sampling_grid.view(B, num_cands * 9, 1, 2)
        
        # Real sampling from feature map
        # Output: (B, C_feat, num_cands * 9, 1)
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 9).permute(0, 2, 3, 1) # (B, num_cands, 9, C_feat)
        
        # For adjacency, we stick to grid for now but with sampled features
        adj = self.motif_bank.motif_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)
        
        # Return centers for visualization
        centers_coords = []
        for idx in center_indices:
            centers_coords.append((idx // W, idx % W))
            
        return sampled_feats, adj, centers_coords

    def forward(self, x, return_selection=False, targets=None):
        B = x.shape[0]
        self._latest_targets = targets
        
        feat_map = self.backbone(x) # (B, C, H, W)
        _, _, H, W = feat_map.shape
        
        # 3.1 Sparse Semantic Graph
        nodes_with_coords, adj = self._get_global_graph(feat_map)
        
        # 3.4 Multi-layer GNN
        node_feats = nodes_with_coords[:, :, :-2]
        if node_feats.shape[-1] != self.feat_dim:
            if not hasattr(self, 'proj_node'):
                self.proj_node = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
            node_feats = self.proj_node(node_feats)
            
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
            
        # (2) Real Deformable Extraction
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(feat_map, H, W, node_feats)
        num_cands = candidates.shape[1]
        
        # Prepare Candidates
        flat_cands = candidates.reshape(B * num_cands, 9, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed
        
        # Prepare Motifs
        motifs_with_coords, _ = self.motif_bank.get_motifs()
        motif_feats = motifs_with_coords[:, :, :-2]
        if motif_feats.shape[-1] != self.feat_dim:
            motif_feats = self.proj_node(motif_feats)
        motif_feats = motif_feats + self.pos_embed
        
        # (3) Matching Logic with Normalization
        if not hasattr(self, 'matching_layer'):
            self.matching_layer = CrossAttentionMatching(self.feat_dim).to(x.device)
            
        scores = self.matching_layer(flat_cands, motif_feats).view(B, num_cands, -1)
        
        # 4. Prototype-based Classification
        class_motif_scores = scores.view(B, num_cands, self.num_classes, self.motifs_per_class)
        best_motif_per_cand_per_class = class_motif_scores.max(dim=-1)[0]
        
        # (5) Sharp Soft Selection: Sharpened Softmax
        cand_relevance = best_motif_per_cand_per_class.max(dim=-1)[0]
        # Use a temperature for selection sharpness
        selection_temp = 0.1 
        attn_weights = F.softmax(cand_relevance / selection_temp, dim=1).unsqueeze(-1) 
        
        logits = torch.sum(best_motif_per_cand_per_class * attn_weights, dim=1)
        logits = logits * self.logit_scale 
        
        self._latest_scores = scores
        _, top_k_idx = torch.topk(cand_relevance, k=self.top_k, dim=1)
        self._latest_top_k = top_k_idx
        
        if return_selection:
            return logits, top_k_idx, centers, scores
            
        return logits

    def _get_global_graph(self, feat_map):
        """ (1) Sparse Semantic Graph (k=4) """
        B, C, H, W = feat_map.shape
        N = H * W
        
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        coords = torch.stack([x, y], dim=-1).to(feat_map.device).view(1, N, 2).expand(B, -1, -1)
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, N, C)
        nodes_with_coords = torch.cat([nodes, coords], dim=-1)
        
        nodes_norm = F.normalize(nodes, dim=-1)
        sim = torch.matmul(nodes_norm, nodes_norm.transpose(1, 2))
        
        # Sparse Graph: k=4 to reduce noise
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

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
        l_div = self.compute_motif_diversity_loss()
        return {"motif_diversity": l_div}
        
        if return_selection:
            return logits, top_k_idx, centers, scores
            
        return logits

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
    dummy_img = torch.randn(2, 1, 48, 48)
    out = model(dummy_img)
    print(f"Output shape: {out.shape}") # Should be (2, 7)
