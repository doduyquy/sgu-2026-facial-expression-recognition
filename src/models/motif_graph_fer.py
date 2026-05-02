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

class GraphMotifModule(nn.Module):
    """
    Research-grade Graph Motif Learning Module (CVPR/ICCV Level).
    Models structured relationships between facial regions using learnable motifs
    and edge-aware structural matching.
    """
    def __init__(self, num_classes, motifs_per_class, K, C, top_k=None):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K = K  # number of regions (e.g., 9)
        self.C = C  # feature dimension
        self.top_k = top_k
        
        # Motif Representation: (Classes, Motifs, K, Dim)
        # Each motif is a structured pattern across K regions
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, C))
        nn.init.xavier_uniform_(self.motifs)
        
        # Edge-aware Motifs: (Classes, Motifs, K, K)
        # Learnable structural relationships for each motif
        self.motif_edges = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, K))
        nn.init.xavier_uniform_(self.motif_edges)
        
        # Learnable temperature for stable similarity scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def compute_diversity_loss(self):
        """
        Diversity regularization: encourages motifs to represent distinct patterns.
        """
        # (Classes, Motifs, K*Dim)
        m = self.motifs.view(self.num_classes, self.motifs_per_class, -1)
        m = F.normalize(m, dim=-1)
        
        # Intra-class diversity: push motifs of the same class apart
        sim_intra = torch.matmul(m, m.transpose(1, 2)) # (C, M, M)
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        l_intra = (sim_intra * (1 - eye)).mean()
        
        # Inter-class diversity: push class prototypes apart
        class_centers = m.mean(dim=1) # (C, K*D)
        class_centers = F.normalize(class_centers, dim=-1)
        sim_inter = torch.matmul(class_centers, class_centers.transpose(0, 1)) # (C, C)
        eye_c = torch.eye(self.num_classes, device=m.device)
        l_inter = (sim_inter * (1 - eye_c)).mean()
        
        return l_intra + l_inter

    def forward(self, region_features, adj=None, return_attention=False):
        """
        Args:
            region_features: (B, K, C)
            adj: (B, K, K) optional edge relationships
            return_attention: bool
            
        Returns:
            logits: (B, num_classes)
            motif_scores: (B, num_classes, motifs_per_class)
        """
        B, K, C = region_features.shape
        eps = 1e-8
        
        # Step 1: Normalize region features and motifs
        region_features = F.normalize(region_features, p=2, dim=-1) # (B, K, C)
        motifs = F.normalize(self.motifs, p=2, dim=-1) # (C_out, M, K, C)
        
        # Step 2: Node-wise Similarity (B, num_classes, motifs_per_class, K)
        # Computes similarity between each input region and corresponding motif region
        # Vectorized implementation using einsum
        similarity = torch.einsum('bkc,lmkc->blmk', region_features, motifs)
        
        # Step 3: Edge-aware matching (Structural similarity)
        if adj is not None:
            # Flatten structures for Frobenius-style similarity
            adj_flat = F.normalize(adj.reshape(B, -1), dim=-1)
            motif_edges_flat = F.normalize(self.motif_edges.reshape(self.num_classes, self.motifs_per_class, -1), dim=-1)
            # (B, Classes, Motifs)
            edge_sim = torch.einsum('bi,cmi->bcm', adj_flat, motif_edges_flat)
            # Combine node similarity with structural bias
            similarity = similarity + edge_sim.unsqueeze(-1)
            
        # Step 4: Subgraph selection (Optional Top-K)
        if self.top_k is not None:
            similarity, _ = torch.topk(similarity, k=min(self.top_k, K), dim=-1)
            
        # Step 5: Attention-weighted aggregation
        # temperature-scaled softmax over regions
        attn = F.softmax(similarity / self.temperature.clamp(min=1e-3), dim=-1)
        
        # motif_score: (B, num_classes, motifs_per_class)
        motif_scores = torch.sum(attn * similarity, dim=-1)
        
        # Step 6: Motif Selection (Max over motifs_per_class)
        # logits: (B, num_classes)
        logits, _ = torch.max(motif_scores, dim=-1)
        
        if return_attention:
            return logits, motif_scores, attn
        return logits, motif_scores

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

    def compute_motif_diversity_loss(self):
        m = self.motif_bank.motifs 
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
        
        # 3. Match against Learnable Motifs
        logits_cand, motif_scores_cand = self.motif_module(flat_cands, adj=flat_adjs)
        
        # 4. Aggregate across all candidate subgraphs
        # logits_cand: (B*num_cands, num_classes)
        logits_cand = logits_cand.view(B, num_cands, self.num_classes)
        
        # Compute selection weights based on max motif score per candidate
        # relevance: (B, num_cands)
        cand_relevance = logits_cand.max(dim=-1)[0]
        selection_temp = 0.3 
        attn_weights = F.softmax(cand_relevance / selection_temp, dim=1).unsqueeze(-1) 
        
        logits_motif = torch.sum(logits_cand * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        
        # Final combined logits
        logits = logits_motif + torch.sigmoid(self.alpha) * logits_global
        
        self._latest_scores = motif_scores_cand.view(B, num_cands, self.num_classes, self.motifs_per_class)
        _, top_k_idx = torch.topk(cand_relevance, k=self.top_k, dim=1)
        self._latest_top_k = top_k_idx
        
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
        l_div = self.motif_module.compute_diversity_loss()
        return {"motif_diversity": l_div}


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