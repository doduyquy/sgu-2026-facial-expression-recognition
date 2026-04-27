import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18, ResNet18_Weights


class MotifBackbone(nn.Module):
    """
    Upgraded Backbone: ResNet18 with Pretrained Weight Transfer.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # (1) Knowledge Transfer: Average 3-channel weights to 1-channel
        if in_channels == 1:
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.model.conv1 = new_conv
            
        # Adjust strides for 48x48 input to get 6x6 output
        self.model.layer3[0].conv1.stride = (1, 1)
        self.model.layer3[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv1.stride = (1, 1)
        self.model.layer4[0].downsample[0].stride = (1, 1)
        
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1, # 12x12
            self.model.layer2, # 6x6
            self.model.layer3, # 6x6
            self.model.layer4  # 6x6
        )
        
        self.projection = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projection(self.features(x))

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        B, N, _ = x.shape
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # Multi-head Attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Combine with adjacency
        attn = attn.masked_fill(adj.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_lin(out)

class CrossAttentionMatching(nn.Module):
    """
    True Attention-based Matching with Relaxed Position Bias.
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.q_lin = nn.Linear(feat_dim, feat_dim)
        self.k_lin = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, candidates, motifs, cand_coords, motif_target_coords):
        B_c, N, D = candidates.shape
        M, _, _ = motifs.shape
        
        q = self.q_lin(candidates) 
        k = self.k_lin(motifs)    
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # sim_matrix: (B_c, M, 9, 9)
        sim_matrix = torch.einsum('bid,mjd->bmij', q, k) 
        
        # (2) Sharpened Matching: Top-5 nodes for better robustness
        # 3 was too sparse, making it sensitive to noise.
        top_k_sim = sim_matrix.max(dim=-1)[0].topk(k=5, dim=-1)[0].mean(dim=-1)
        feat_sim = top_k_sim
        
        # (3) Position Bias: Slightly tighter (0.4) to avoid eyes/mouth confusion
        dist = torch.cdist(cand_coords.unsqueeze(0), motif_target_coords.unsqueeze(0)).squeeze(0)
        pos_penalty = torch.sigmoid(-(dist - 0.4) * 5.0) 
        
        return feat_sim * pos_penalty

class MotifBank(nn.Module):
    # ... (rest of MotifBank stays same)
    def __init__(self, num_classes=7, motifs_per_class=8, num_nodes=9, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_nodes = num_nodes
        
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, num_nodes, feat_dim))
        nn.init.xavier_uniform_(self.motifs)
        
        targets = torch.tensor([
            [0.2, 0.2], [0.8, 0.2], # Eyes
            [0.5, 0.8], [0.5, 0.9], # Mouth
            [0.2, 0.5], [0.8, 0.5], # Cheeks
            [0.5, 0.4], [0.5, 0.6]  # Nose/Center
        ]).repeat(num_classes, 1)
        self.register_buffer('target_coords', targets)

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
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
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
        
        self.motif_bank = MotifBank(
            num_classes=self.num_classes, 
            motifs_per_class=self.motifs_per_class,
            num_nodes=9,
            feat_dim=self.feat_dim
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self._progress = 0.0 # Training progress [0, 1]

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
        
        # (5) Motif-Feature Alignment Loss:
        # Encourages motifs to be close to at least some real features
        l_align = torch.tensor(0.0, device=m.device)
        if hasattr(self, '_latest_scores'):
            # Max score per motif across all candidates in batch
            # scores: (B, num_cands, num_motifs)
            max_motif_match = self._latest_scores.max(dim=0)[0].max(dim=0)[0]
            l_align = 1.0 - max_motif_match.mean()
        
        return l_intra + 1.0 * l_inter + 0.5 * l_align

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        B, C_feat, _, _ = feat_map.shape
        y_c, x_c = torch.meshgrid(torch.arange(1, H-1), torch.arange(1, W-1), indexing='ij')
        center_indices = (y_c * W + x_c).flatten().to(feat_map.device)
        num_cands = len(center_indices)
        center_feats = node_feats[:, center_indices, :] 
        offsets = self.offset_predictor(center_feats) 
        rel_y, rel_x = torch.meshgrid(torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing='ij')
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device).view(1, 1, 9, 2)
        c_y = (center_indices // W).float() / (H - 1) * 2 - 1
        c_x = (center_indices % W).float() / (W - 1) * 2 - 1
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(1, num_cands, 1, 2) 
        abs_coords = (centers_grid + offsets.unsqueeze(2) + 1.0) / 2.0 
        abs_centers = abs_coords.squeeze(2) 
        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W-1))
        sampling_grid = sampling_grid.view(B, num_cands * 9, 1, 2)
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True, padding_mode='zeros')
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 9).permute(0, 2, 3, 1) 
        adj = self.motif_bank.motif_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)
        centers_coords = torch.stack([center_indices // W, center_indices % W], dim=-1)
        return sampled_feats, adj, centers_coords, abs_centers

    def forward(self, x, return_selection=False, targets=None):
        B = x.shape[0]
        feat_map = self.backbone(x) 
        _, _, H, W = feat_map.shape
        logits_global = self.global_fc(self.global_pool(feat_map))
        nodes_with_coords, adj = self._get_global_graph(feat_map)
        node_feats = nodes_with_coords[:, :, :-2]
        if not hasattr(self, 'proj_node'):
            self.proj_node = nn.Linear(node_feats.shape[-1], self.feat_dim).to(x.device)
        node_feats = self.proj_node(node_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adj)
        candidates, cand_adjs, centers, abs_centers = self._extract_deformable_subgraphs(feat_map, H, W, node_feats)
        num_cands = candidates.shape[1]
        flat_cands = candidates.reshape(B * num_cands, 9, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed
        flat_abs_centers = abs_centers.reshape(B * num_cands, 2)
        motifs_with_coords, _ = self.motif_bank.get_motifs()
        motif_feats = motifs_with_coords[:, :, :-2]
        if motif_feats.shape[-1] != self.feat_dim:
            motif_feats = self.proj_node(motif_feats)
        motif_feats = motif_feats + self.pos_embed
        if not hasattr(self, 'matching_layer'):
            self.matching_layer = CrossAttentionMatching(self.feat_dim).to(x.device)
        scores = self.matching_layer(
            flat_cands, motif_feats, 
            flat_abs_centers, self.motif_bank.target_coords
        ).view(B, num_cands, -1)
        self._latest_scores = scores
        class_motif_scores = scores.view(B, num_cands, self.num_classes, self.motifs_per_class)
        best_motif_per_cand_per_class = class_motif_scores.topk(k=min(2, self.motifs_per_class), dim=-1)[0].mean(dim=-1)
        
        # (4) Selection Annealing: Softmax -> Gumbel
        cand_relevance = best_motif_per_cand_per_class.max(dim=-1)[0]
        if self.training:
            # Phase 1 (Progress < 0.3): Softmax for exploration
            # Phase 2 (Progress >= 0.3): Gumbel for sharpening
            if self._progress < 0.3:
                attn_weights = F.softmax(cand_relevance / 0.5, dim=1).unsqueeze(-1)
            else:
                # Use hard=False to avoid losing multi-region information
                attn_weights = F.gumbel_softmax(cand_relevance, tau=0.3, hard=False).unsqueeze(-1)
        else:
            attn_weights = F.softmax(cand_relevance / 0.1, dim=1).unsqueeze(-1)
        
        logits_motif = torch.sum(best_motif_per_cand_per_class * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        logits = logits_motif + torch.sigmoid(self.alpha) * logits_global
        _, top_k_idx = torch.topk(cand_relevance, k=self.top_k, dim=1)
        self._latest_top_k = top_k_idx
        if return_selection:
            return logits, top_k_idx, centers, scores
        return logits

    def _get_global_graph(self, feat_map):
        """ (1) Top-K Sparse Softmax to prevent over-smoothing """
        B, C, H, W = feat_map.shape
        N = H * W
        y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        coords = torch.stack([x, y], dim=-1).to(feat_map.device).view(1, N, 2).expand(B, -1, -1)
        nodes = feat_map.permute(0, 2, 3, 1).reshape(B, N, C)
        nodes_with_coords = torch.cat([nodes, coords], dim=-1)
        nodes_norm = F.normalize(nodes, dim=-1)
        sim_feat = torch.matmul(nodes_norm, nodes_norm.transpose(1, 2))
        dist = torch.cdist(coords, coords) 
        sim_spatial = torch.exp(-dist**2 / (2 * 0.5**2)) 
        sim = sim_feat * sim_spatial
        
        # Sparse Softmax: Mask everything except top-k, then softmax
        # Increased k to 8 to regain facial context
        k = 8
        mask = torch.zeros_like(sim)
        topk_idx = torch.topk(sim, k=k, dim=-1)[1]
        mask.scatter_(-1, topk_idx, 1.0)
        
        adj = F.softmax(sim.masked_fill(mask == 0, -1e9) / 0.1, dim=-1)
        return nodes_with_coords, adj

    def get_landmark_outputs(self):
        return getattr(self, '_latest_scores', None), getattr(self, '_latest_top_k', None)
    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
        l_div = self.compute_motif_diversity_loss()
        return {"motif_diversity": l_div}
    def get_landmark_aux_logits(self):
        return None
    def set_training_progress(self, progress):
        self._progress = progress
    def get_current_prior_strength(self):
        return 0.0
