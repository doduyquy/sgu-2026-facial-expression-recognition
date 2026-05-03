import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ECA(nn.Module):
    """ Efficient Channel Attention for early stages """
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(1, mip)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.gn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h

class MSBlock(nn.Module):
    """
    Improved Multi-Scale Block with Channel-wise Gating and LayerScale.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=2, drop_path=0.1):
        super().__init__()
        mid_channels = out_channels // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU()
        )
        
        # [1] Channel-wise gating
        self.gate = nn.Parameter(torch.ones(out_channels))
        
        # [3] Cross-channel interaction fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels)
        )
        
        # [2] LayerScale + DropPath for variance preservation
        self.gamma = nn.Parameter(torch.ones(out_channels) * 1e-2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        
        # Apply channel gating
        out = out * self.gate.view(1, -1, 1, 1).sigmoid()
        out = self.fuse(out)
        
        return self.shortcut(x) + self.drop_path(self.gamma.view(1, -1, 1, 1) * out)

class SobelStem(nn.Module):
    """ Stem with edge-aware inductive bias """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels // 2),
            nn.GELU()
        )
        self.edge_proj = nn.Conv2d(2, out_channels // 2, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        e = self.edge_proj(torch.cat([gx, gy], dim=1))
        return self.final(torch.cat([self.conv(x), e], dim=1))

class MotifBackbone(nn.Module):
    """
    Optimized Backbone for Motif Graph FER.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.stem = SobelStem(in_channels, 64)
        
        # Stage 1: Detail preservation (dilation=1) + ECA
        self.stage1 = nn.Sequential(
            MSBlock(64, 64, dilation=1),
            ECA(3)
        )
        
        # Stage 2 & 3: Context awareness + CoordAtt
        self.stage2 = nn.Sequential(
            MSBlock(64, 128, stride=2, dilation=2),
            CoordinateAttention(128)
        )
        self.stage3 = nn.Sequential(
            MSBlock(128, feat_dim, stride=2, dilation=2),
            CoordinateAttention(feat_dim)
        )
        
        # Cross-stage skip
        self.skip_1_3 = nn.Sequential(
            nn.Conv2d(64, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, feat_dim)
        )
        self.feat_stats = {}

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        
        # Spatial alignment skip
        s1_down = F.interpolate(s1, size=s3.shape[2:], mode='bilinear', align_corners=False)
        x = s3 + self.skip_1_3(s1_down)
        
        # Diagnostics
        if self.training:
            self.feat_stats = {
                'spatial_var': x.var(dim=(2,3)).mean().item(),
                'channel_var': x.var(dim=(0,2,3)).mean().item()
            }
        self.activation_map = x.detach()
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, num_nodes=36):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.edge_bias = nn.Parameter(torch.zeros(heads, num_nodes, num_nodes))
        self.out_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        B, N, _ = x.shape
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if N == self.edge_bias.shape[-1]:
            scores = scores + self.edge_bias.unsqueeze(0)
        if adj is not None:
            scores = scores * torch.sigmoid(adj.unsqueeze(1))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return F.gelu(self.out_lin(out))

class GraphMotifModule(nn.Module):
    def __init__(self, num_classes, motifs_per_class, K, C, rank=4):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K, self.C = K, C
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, C))
        nn.init.orthogonal_(self.motifs)
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
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
            GraphAttentionLayer(self.feat_dim, self.feat_dim, num_nodes=36), 
            GraphAttentionLayer(self.feat_dim, self.feat_dim, num_nodes=36)
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
                    if 0 <= ni < 3 and 0 <= nj < 3: adj[idx, ni * 3 + nj] = 1.0
        return adj

    def forward(self, x, targets=None):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            logits = self.forward(x.view(B*T, C, H, W))
            return logits.view(B, T, -1).mean(dim=1)
        
        B, C_in, H_in, W_in = x.shape
        feat_map = self.backbone(x)
        B, C_feat, H, W = feat_map.shape
        
        logits_global = self.global_fc(self.global_pool(feat_map))
        
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, H*W, C_feat)
        sim = torch.matmul(F.normalize(nodes, dim=-1), F.normalize(nodes, dim=-1).transpose(1, 2))
        _, topk_idx = torch.topk(sim, k=min(8, H*W), dim=-1)
        adj = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
        
        node_feats = nodes
        for gnn in self.gnn_layers: 
            node_feats = gnn(node_feats, adj)
            
        feat_map_refined = node_feats.view(B, H, W, C_feat).permute(0, 3, 1, 2).contiguous()
        
        center_idx = torch.tensor([i*W+j for i in range(1,H-1) for j in range(1,W-1)], device=x.device)
        num_cands = len(center_idx)
        
        offsets = self.offset_predictor(node_feats[:, center_idx, :])
        
        rel_grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,3), torch.linspace(-1,1,3), indexing='ij'), dim=-1).to(x.device).view(1,1,9,2)
        centers_grid = torch.stack([(center_idx % W).float()/(W-1)*2-1, (center_idx // W).float()/(H-1)*2-1], dim=-1).view(1, num_cands, 1, 2)
        sampling_grid = (centers_grid + offsets.unsqueeze(2) + rel_grid*(1.0/(W-1))).view(B, num_cands*9, 1, 2)
        
        candidates = F.grid_sample(feat_map_refined, sampling_grid, align_corners=True).view(B, C_feat, num_cands, 9).permute(0, 2, 3, 1)
        
        flat_cands = (candidates.reshape(B*num_cands, 9, C_feat) + self.pos_embed)
        logits_cand, motif_scores_cand = self.motif_module(flat_cands, adj=self.grid_adj.unsqueeze(0).expand(B*num_cands, -1, -1))
        
        logits_motif = logits_cand.view(B, num_cands, -1).mean(dim=1) * self.logit_scale
        
        # Store for consistency loss (InfoNCE)
        # S shape is (B*num_cands, num_classes, motifs_per_class)
        self._latest_scores = motif_scores_cand.view(B, num_cands, -1)
        self._latest_offsets = offsets
        
        # Select top-k subgraphs for landmark visualization/loss
        # Based on matching confidence of the most likely class
        max_scores, _ = motif_scores_cand.max(dim=-1) # (B_large, num_classes)
        max_scores, _ = max_scores.max(dim=-1) # (B_large,)
        max_scores = max_scores.view(B, num_cands)
        _, self._top_k_idx = torch.topk(max_scores, k=min(4, num_cands), dim=-1)
        
        return logits_motif + torch.sigmoid(self.alpha_fuse) * logits_global

    def compute_motif_diversity_loss(self):
        return self.motif_module.compute_diversity_loss()

    def get_landmark_outputs(self): 
        return getattr(self, '_latest_scores', None), getattr(self, '_top_k_idx', None)
    
    def get_landmark_aux_logits(self): return None
    def set_training_progress(self, progress): pass
    def get_current_prior_strength(self): return 0.0
    
    def get_aux_losses(self):
        return {
            "motif_diversity": self.compute_motif_diversity_loss(), 
            "offset_reg": torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        }