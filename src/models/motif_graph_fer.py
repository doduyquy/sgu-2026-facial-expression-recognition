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
        """
        candidates: (B*Cands, 9, D)
        motifs: (Num_Motifs, 9, D)
        """
        # We want a similarity score between each candidate and each motif
        # Using vectorized cross-attention
        B_c, N, D = candidates.shape
        M, _, _ = motifs.shape
        
        # Project
        q = self.q_lin(candidates) # (B*Cands, 9, D)
        k = self.k_lin(motifs)    # (M, 9, D)
        
        # Compute all-to-all similarity for each candidate-motif pair
        # Reshape to (B*Cands, 1, 9, D) and (1, M, 9, D)
        # Dot product: (B*Cands, M, 9, 9)
        # sim[b, m, i, j] is similarity between node i of cand b and node j of motif m
        sim_matrix = torch.einsum('bid,mjd->bmij', q, k) / math.sqrt(D)
        
        # Soft alignment: for each node in cand, find best match in motif
        # (B*Cands, M, 9)
        align_cand = sim_matrix.max(dim=-1)[0].mean(dim=-1)
        # For each node in motif, find best match in cand
        align_motif = sim_matrix.max(dim=-2)[0].mean(dim=-1)
        
        # Symmetric similarity
        return (align_cand + align_motif) / 2.0

class MotifBank(nn.Module):
    """
    Stores learnable prototype subgraphs (motifs) for each class.
    """
    def __init__(self, num_classes=7, motifs_per_class=8, num_nodes=9, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_nodes = num_nodes
        
        # Learnable motif prototypes: (num_classes, motifs_per_class, num_nodes, feat_dim)
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, num_nodes, feat_dim))
        nn.init.xavier_uniform_(self.motifs)
        
        # Motif fixed structure (3x3 grid adjacency)
        adj = self._generate_3x3_grid_adj()
        self.register_buffer('motif_adj', adj)
        
        # Relative coordinates for 3x3 motif nodes (from 0 to 1)
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
        return torch.stack([x, y], dim=-1).view(9, 2) # (9, 2)

    def get_motifs(self):
        # Return motifs reshaped to (Total_Motifs, num_nodes, feat_dim)
        flat_motifs = self.motifs.view(-1, self.num_nodes, self.motifs.shape[-1])
        # Add relative coords
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
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        self.gnn = GraphAttentionLayer(self.feat_dim, self.feat_dim) 
        
        # Learnable positional encoding for 3x3 patch (9 nodes)
        self.pos_embed = nn.Parameter(torch.randn(1, 9, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.motif_bank = MotifBank(
            num_classes=self.num_classes, 
            motifs_per_class=self.motifs_per_class,
            num_nodes=9,
            feat_dim=self.feat_dim
        )
        
        # Subgraph encoder for classification
        self.subgraph_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3))
        )
        
        # Attention pooling: scores for K subgraphs
        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

    def compute_motif_diversity_loss(self):
        # Prevent motifs from becoming too similar (L_div = mean(cosine_sim))
        # self.motif_bank.motifs: (num_classes, motifs_per_class, 9, feat_dim)
        m = self.motif_bank.motifs # (C, M, 9, d)
        m = m.view(self.num_classes, self.motifs_per_class, -1) # (C, M, 9d)
        m = F.normalize(m, dim=-1)
        
        # Compute self-similarity for each class
        # (C, M, M)
        sim = torch.matmul(m, m.transpose(1, 2))
        # Remove diagonal
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        sim = sim * (1 - eye)
        
        return sim.mean()

    def _extract_candidate_subgraphs(self, nodes, adj, H, W):
        B, N, _ = nodes.shape
        subgraphs = []
        subgraph_adjs = []
        subgraph_centers = []
        
        # Extract 3x3 patches (sliding window)
        for i in range(1, H-1):
            for j in range(1, W-1):
                indices = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        indices.append((i + di) * W + (j + dj))
                
                indices = torch.tensor(indices, device=nodes.device)
                sub_nodes = nodes[:, indices, :] # (B, 9, C+2)
                sub_adj = adj[:, indices][:, :, indices] # (B, 9, 9)
                
                subgraphs.append(sub_nodes)
                subgraph_adjs.append(sub_adj)
                subgraph_centers.append((i, j))
                
        return torch.stack(subgraphs, dim=1), torch.stack(subgraph_adjs, dim=1), subgraph_centers

    def forward(self, x, return_selection=False):
        B = x.shape[0]
        feat_map = self.backbone(x)
        _, _, H, W = feat_map.shape
        
        # 1. Build Grid Graph
        nodes, adj = self._get_grid_graph(feat_map)
        
        # 2. Extract candidate subgraphs (3x3 patches)
        # candidates: (B, num_candidates, 9, C+2)
        candidates, cand_adjs, centers = self._extract_candidate_subgraphs(nodes, adj, H, W)
        num_cands = candidates.shape[1]
        
        # 3. Encode candidates using GAT and Positional Encoding
        flat_cands = candidates.view(B * num_cands, 9, -1)
        flat_adjs = cand_adjs.view(B * num_cands, 9, 9)
        
        if flat_cands.shape[-1] != self.feat_dim:
            if not hasattr(self, 'proj_node'):
                self.proj_node = nn.Linear(flat_cands.shape[-1], self.feat_dim).to(flat_cands.device)
            flat_cands = self.proj_node(flat_cands)
            
        flat_cands = flat_cands + self.pos_embed
        
        # Use GAT for better expressiveness
        cand_embeds = self.gnn(flat_cands, flat_adjs) 
        
        # 4. Motif Matching (Cross-Attention Matching)
        motifs_with_coords, motif_adj = self.motif_bank.get_motifs()
        num_motifs = motifs_with_coords.shape[0]
        
        if motifs_with_coords.shape[-1] != self.feat_dim:
            motifs_with_coords = self.proj_node(motifs_with_coords)
            
        motifs_with_coords = motifs_with_coords + self.pos_embed
        
        motif_embeds = self.gnn(motifs_with_coords, 
                                motif_adj.unsqueeze(0).expand(num_motifs, -1, -1)) 
        
        # Soft Matching instead of hard alignment
        if not hasattr(self, 'matching_layer'):
            self.matching_layer = CrossAttentionMatching(self.feat_dim).to(x.device)
            
        # scores: (B*num_cands, num_motifs)
        scores = self.matching_layer(cand_embeds, motif_embeds)
        scores = scores.view(B, num_cands, num_motifs)
        
        # 5. Subgraph Selection
        max_scores, _ = scores.max(dim=-1) # (B, num_cands)
        top_k_vals, top_k_idx = torch.topk(max_scores, k=self.top_k, dim=1)
        
        # 6. Aggregate and Classify
        cand_pooled = cand_embeds.mean(dim=1).view(B, num_cands, -1)
        
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_embeds = cand_pooled[batch_idx, top_k_idx] 
        
        selected_embeds = self.subgraph_encoder(selected_embeds)
        
        attn_logits = self.attention(selected_embeds)
        attn_weights = F.softmax(attn_logits, dim=1) 
        
        aggregated = torch.sum(selected_embeds * attn_weights, dim=1)
        logits = self.classifier(aggregated)
        
        # Store for Trainer access
        self._latest_scores = scores
        self._latest_top_k = top_k_idx
        
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
