import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    More stable than CBAM for low-resolution FER2013 (48x48).
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MotifBackbone(nn.Module):
    """
    Stabilized Backbone with SEBlocks and GroupNorm.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 24x24
        )
        
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64)
        )
        self.se1 = SEBlock(64)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 12x12
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )
        
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128)
        )
        self.se2 = SEBlock(128)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, feat_dim, kernel_size=3, stride=2, padding=1), # 6x6
            nn.GroupNorm(8, feat_dim),
            nn.ReLU(inplace=True)
        )
        self.final_se = SEBlock(feat_dim)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.res1(x)
        x = self.se1(x)
        x = F.relu(x + identity, inplace=True)
        
        x = self.down1(x)
        identity = x
        x = self.res2(x)
        x = self.se2(x)
        x = F.relu(x + identity, inplace=True)
        
        x = self.down2(x)
        return self.final_se(x)

class GraphAttentionLayer(nn.Module):
    """
    Structure-aware Graph Attention with learnable head-specific bias.
    """
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        
        # Learnable structural bias for 3x3 region nodes
        self.edge_bias = nn.Parameter(torch.zeros(heads, 9, 9))
        self.out_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        B, N, _ = x.shape
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Inject structural priority
        scores = scores + self.edge_bias.unsqueeze(0)
        
        if adj is not None:
            # Gating instead of masking to preserve weak signals
            scores = scores * torch.sigmoid(adj.unsqueeze(1))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return F.gelu(self.out_lin(out))

class GraphMotifModule(nn.Module):
    """
    Vectorized Motif Matching with Orthogonal Init and Similarity Standardization.
    """
    def __init__(self, num_classes, motifs_per_class, K, C, rank=4):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K, self.C = K, C
        
        # Orthogonal Init to prevent early collapse
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, C))
        nn.init.orthogonal_(self.motifs)
        
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
        nn.init.xavier_uniform_(self.motif_low_rank)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def compute_diversity_loss(self):
        m = self.motifs.view(self.num_classes, self.motifs_per_class, -1)
        m = F.normalize(m, dim=-1)
        sim = torch.matmul(m, m.transpose(1, 2))
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        return torch.norm(sim - eye, p='fro', dim=(1, 2)).mean()

    def forward(self, region_features, adj=None):
        B, K, C = region_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        region_features = F.normalize(region_features, p=2, dim=-1)
        motifs = F.normalize(self.motifs, p=2, dim=-1)
        
        node_sim = torch.einsum('bkc,lmkc->blmk', region_features, motifs)
        
        diff_R = region_features.unsqueeze(2) - region_features.unsqueeze(1)
        diff_M = motifs.unsqueeze(3) - motifs.unsqueeze(2)
        edge_sim = torch.einsum('bijk,lmijk->blmij', diff_R, diff_M).mean(dim=(-1, -2))
        
        motif_adj = torch.matmul(self.motif_low_rank, self.motif_low_rank.transpose(-1, -2))
        motif_adj = F.softmax(motif_adj, dim=-1)
        topo_sim = torch.einsum('bij,lmij->blm', adj, motif_adj) if adj is not None else 0
        
        tau = F.softplus(self.temperature).clamp(min=0.01)
        node_attn = F.softmax(node_sim / tau, dim=-1)
        node_sim_agg = torch.sum(node_attn * node_sim, dim=-1)
        
        S = torch.sigmoid(self.alpha) * node_sim_agg + torch.sigmoid(self.beta) * (edge_sim + topo_sim)
        # Standardization before selection
        S = (S - S.mean(dim=-1, keepdim=True)) / (S.std(dim=-1, keepdim=True) + 1e-6)
        
        logits = torch.logsumexp(S / tau, dim=-1)
        return logits, S

class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 8)
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_dim, self.feat_dim),
            GraphAttentionLayer(self.feat_dim, self.feat_dim)
        ])
        
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, 9, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.register_buffer('grid_adj', self._generate_3x3_grid_adj())
        self.motif_module = GraphMotifModule(self.num_classes, self.motifs_per_class, 9, self.feat_dim)
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.alpha_fuse = nn.Parameter(torch.ones(1) * 0.5)

    def _generate_3x3_grid_adj(self):
        adj = torch.zeros(9, 9)
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        adj[idx, ni * 3 + nj] = 1.0
        return adj

    def forward(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            logits = self.forward(x.view(B*T, C, H, W))
            return logits.view(B, T, -1).mean(dim=1)
            
        B, C, H, W = x.shape
        feat_map = self.backbone(x)
        logits_global = self.global_fc(self.global_pool(feat_map))
        
        # Build spatial graph
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        sim = torch.matmul(F.normalize(nodes, dim=-1), F.normalize(nodes, dim=-1).transpose(1, 2))
        _, topk_idx = torch.topk(sim, k=8, dim=-1)
        adj = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
        
        node_feats = nodes
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
            
        feat_map_refined = node_feats.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Deformable Subgraph Sampling
        center_idx = torch.tensor([i*W+j for i in range(1,H-1) for j in range(1,W-1)], device=x.device)
        num_cands = len(center_idx)
        offsets = self.offset_predictor(node_feats[:, center_idx, :])
        self._latest_offsets = offsets
        
        rel_grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,3), torch.linspace(-1,1,3), indexing='ij'), dim=-1).to(x.device).view(1,1,9,2)
        centers_grid = torch.stack([(center_idx % W).float()/(W-1)*2-1, (center_idx // W).float()/(H-1)*2-1], dim=-1).view(1, num_cands, 1, 2)
        sampling_grid = (centers_grid + offsets.unsqueeze(2) + rel_grid*(1.0/(W-1))).view(B, num_cands*9, 1, 2)
        
        candidates = F.grid_sample(feat_map_refined, sampling_grid, align_corners=True).view(B, -1, num_cands, 9).permute(0, 2, 3, 1)
        
        flat_cands = (candidates.reshape(B*num_cands, 9, -1) + self.pos_embed)
        flat_adjs = self.grid_adj.unsqueeze(0).expand(B*num_cands, -1, -1)
        
        logits_cand, motif_scores_cand = self.motif_module(flat_cands, adj=flat_adjs)
        logits_motif = logits_cand.view(B, num_cands, -1).mean(dim=1) * self.logit_scale
        
        self._latest_scores = motif_scores_cand.view(B, num_cands, -1)
        return logits_motif + torch.sigmoid(self.alpha_fuse) * logits_global

    def get_aux_losses(self):
        return {
            "motif_diversity": self.motif_module.compute_diversity_loss(),
            "offset_reg": torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        }

if __name__ == "__main__":
    config = {'feat_dim': 128, 'num_classes': 7, 'motifs_per_class': 8}
    model = MotifGraphModel(config)
    dummy = torch.randn(2, 1, 48, 48)
    print("Output shape:", model(dummy).shape)