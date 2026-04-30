"""
MotifGraphModel — Hybrid CNN + GNN + Motif Pipeline for FER2013.

Architecture:
    MotifBackbone (ResBlock+CBAM) → Global Graph (kNN k=4) → GAT×2 (with Residual+LN)
    → Deformable Subgraph Extraction (3×3) → Cross-Attention Matching (Node Importance)
    → Fusion (LayerNorm-balanced motif + global logits)

Improvements over baseline:
    1. Code cleanup: removed duplicate methods, dead code, lazy module creation
    2. GAT + Residual + LayerNorm for stable deep message passing
    3. Motif EMA Anchoring: prototypes track real data distribution
    4. Node Importance in Matching: learnable per-node weights
    5. Fusion LayerNorm: balanced scale between motif and global branches
"""

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


# ---------------------------------------------------------------------------
# 1. Backbone
# ---------------------------------------------------------------------------
class MotifBackbone(nn.Module):
    """Advanced Backbone with Residual connections and CBAM."""

    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 24x24
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
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12x12
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
            nn.Conv2d(128, feat_dim, kernel_size=3, stride=2, padding=1),  # 6x6
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


# ---------------------------------------------------------------------------
# 2. Graph Attention Layer
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Multi-head Graph Attention with adjacency masking."""

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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_lin(out)


# ---------------------------------------------------------------------------
# 3. Cross-Attention Matching (with Node Importance — Fix #4)
# ---------------------------------------------------------------------------
class CrossAttentionMatching(nn.Module):
    """
    Matches subgraphs to motifs using Cross-Attention with learnable
    per-node importance weights.
    """

    def __init__(self, feat_dim, num_nodes=9):
        super().__init__()
        self.feat_dim = feat_dim
        self.q_lin = nn.Linear(feat_dim, feat_dim)
        self.k_lin = nn.Linear(feat_dim, feat_dim)

        # Fix #4: Learnable node importance — model discovers which
        # positions in the 3x3 subgraph matter most for each emotion
        self.node_importance = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, candidates, motifs):
        """
        Args:
            candidates: (B_c, N, D) — candidate subgraph features
            motifs:     (M, N, D) — motif prototype features
        Returns:
            scores: (B_c, M) — matching scores
        """
        q = F.normalize(self.q_lin(candidates), dim=-1)
        k = F.normalize(self.k_lin(motifs), dim=-1)

        # (B_c, M, N, N) → per-node similarities
        sim_matrix = torch.einsum('bid,mjd->bmij', q, k)

        # Per-node alignment: max over motif nodes, then weight by importance
        align_cand = sim_matrix.max(dim=-1)[0]       # (B_c, M, N)
        align_motif = sim_matrix.max(dim=-2)[0]       # (B_c, M, N)
        per_node = (align_cand + align_motif) / 2.0   # (B_c, M, N)

        # Apply learned node importance weights
        weights = F.softmax(self.node_importance, dim=-1)  # (N,)
        scores = (per_node * weights).sum(dim=-1)          # (B_c, M)

        return scores


# ---------------------------------------------------------------------------
# 4. Motif Bank
# ---------------------------------------------------------------------------
class MotifBank(nn.Module):
    def __init__(self, num_classes=7, motifs_per_class=8, num_nodes=9, feat_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.num_nodes = num_nodes

        self.motifs = nn.Parameter(
            torch.randn(num_classes, motifs_per_class, num_nodes, feat_dim)
        )
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
        y, x = torch.meshgrid(
            torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), indexing='ij'
        )
        return torch.stack([x, y], dim=-1).view(9, 2)

    def get_motifs(self):
        flat_motifs = self.motifs.view(-1, self.num_nodes, self.motifs.shape[-1])
        Total_Motifs = flat_motifs.shape[0]
        coords = self.rel_coords.unsqueeze(0).expand(Total_Motifs, -1, -1)
        motifs_with_coords = torch.cat([flat_motifs, coords], dim=-1)
        return motifs_with_coords, self.motif_adj


# ---------------------------------------------------------------------------
# 5. Full Model
# ---------------------------------------------------------------------------
class MotifGraphModel(nn.Module):
    """
    Hybrid CNN + GNN + Motif model for facial expression recognition.

    Changes vs baseline:
        Fix #1: Removed duplicate _get_global_graph, dead code, lazy matching_layer
        Fix #2: GAT layers now have residual connections + LayerNorm
        Fix #3: EMA anchoring of motif prototypes to real class distributions
        Fix #4: Node importance weights in CrossAttentionMatching
        Fix #5: LayerNorm before fusion to balance motif vs global logit scales
    """

    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 8)
        self.top_k = config.get('top_k', 4)
        self.temperature = config.get('motif_tau', 0.1)

        # --- Backbone ---
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)

        # --- Global Branch ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        # --- GNN layers (Fix #2: with Residual + LayerNorm) ---
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_dim, self.feat_dim),
            GraphAttentionLayer(self.feat_dim, self.feat_dim)
        ])
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(self.feat_dim),
            nn.LayerNorm(self.feat_dim)
        ])

        # --- Deformable subgraph offset predictor ---
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        # --- Positional embedding for 3x3 subgraph nodes ---
        self.pos_embed = nn.Parameter(torch.randn(1, 9, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Motif Bank ---
        self.motif_bank = MotifBank(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            num_nodes=9,
            feat_dim=self.feat_dim
        )

        # --- Cross-Attention Matching (Fix #1: in __init__, Fix #4: node importance) ---
        self.matching_layer = CrossAttentionMatching(self.feat_dim, num_nodes=9)

        # --- Fusion (Fix #5: LayerNorm for balanced scales) ---
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.match_ln = nn.LayerNorm(self.num_classes)

        # --- Feature projection (in case node dims don't match feat_dim) ---
        self.proj_node = nn.Linear(self.feat_dim + 2, self.feat_dim)

        # --- Internal state ---
        self._latest_scores = None
        self._latest_top_k = None
        self._latest_targets = None

    # ----- Motif Diversity Loss -----
    def compute_motif_diversity_loss(self):
        m = self.motif_bank.motifs
        C, M, N, D = m.shape
        m_flat = F.normalize(m.view(C, M, -1), dim=-1)

        # Intra-class: motifs within same class should be diverse
        sim_intra = torch.matmul(m_flat, m_flat.transpose(1, 2))
        eye = torch.eye(M, device=m.device).unsqueeze(0)
        l_intra = (sim_intra * (1 - eye)).mean()

        # Inter-class: class centers should be dissimilar
        class_centers = F.normalize(m_flat.mean(dim=1), dim=-1)
        sim_inter = torch.matmul(class_centers, class_centers.transpose(0, 1))
        eye_c = torch.eye(C, device=m.device)
        l_inter = (sim_inter * (1 - eye_c)).mean()

        return l_intra + 1.0 * l_inter

    # ----- Graph Construction (single clean version) -----
    def _get_global_graph(self, feat_map):
        """Sparse Semantic Graph with kNN (k=4)."""
        B, C, H, W = feat_map.shape
        N = H * W

        y, x = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'
        )
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

    # ----- Deformable Subgraph Extraction -----
    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        B, C_feat, _, _ = feat_map.shape

        center_indices = []
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                center_indices.append(i * W + j)
        center_indices = torch.tensor(center_indices, device=feat_map.device)
        num_cands = len(center_indices)

        center_feats = node_feats[:, center_indices, :]
        offsets = self.offset_predictor(center_feats)

        rel_y, rel_x = torch.meshgrid(
            torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing='ij'
        )
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device)
        rel_grid = rel_grid.view(1, 1, 9, 2)

        c_y = (center_indices // W).float() / (H - 1) * 2 - 1
        c_x = (center_indices % W).float() / (W - 1) * 2 - 1
        centers_grid = torch.stack([c_x, c_y], dim=-1).view(1, num_cands, 1, 2)

        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W - 1))
        sampling_grid = sampling_grid.view(B, num_cands * 9, 1, 2)

        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 9).permute(0, 2, 3, 1)

        adj = self.motif_bank.motif_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)

        centers_coords = []
        for idx in center_indices:
            centers_coords.append((idx // W, idx % W))

        return sampled_feats, adj, centers_coords

    # ----- Forward -----
    def forward(self, x, return_selection=False, targets=None):
        B = x.shape[0]
        self._latest_targets = targets

        # 1. CNN features
        feat_map = self.backbone(x)  # (B, C, H, W)
        _, _, H, W = feat_map.shape

        # 2. Global branch
        logits_global = self.global_fc(self.global_pool(feat_map))  # (B, num_classes)

        # 3. Graph construction
        nodes_with_coords, adj = self._get_global_graph(feat_map)
        node_feats = self.proj_node(nodes_with_coords)  # (B, N, feat_dim)

        # 4. GNN message passing (Fix #2: Residual + LayerNorm)
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            residual = node_feats
            node_feats = gnn(node_feats, adj)
            node_feats = norm(F.relu(node_feats) + residual)

        # 5. Deformable subgraph extraction
        candidates, cand_adjs, centers = self._extract_deformable_subgraphs(
            feat_map, H, W, node_feats
        )
        num_cands = candidates.shape[1]

        # 6. Prepare candidates + motifs (with pos embed)
        flat_cands = candidates.reshape(B * num_cands, 9, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed

        motifs_with_coords, _ = self.motif_bank.get_motifs()
        motif_feats = motifs_with_coords[:, :, :-2]
        if motif_feats.shape[-1] != self.feat_dim:
            motif_feats = self.proj_node(motif_feats)
        motif_feats = motif_feats + self.pos_embed

        # 7. Cross-attention matching (Fix #4: node importance)
        scores = self.matching_layer(flat_cands, motif_feats).view(B, num_cands, -1)

        # 8. Prototype decision logic
        class_motif_scores = scores.view(B, num_cands, self.num_classes, self.motifs_per_class)
        best_motif_per_cand_per_class = class_motif_scores.topk(
            k=min(2, self.motifs_per_class), dim=-1
        )[0].mean(dim=-1)  # (B, num_cands, num_classes)

        cand_relevance = best_motif_per_cand_per_class.max(dim=-1)[0]
        selection_temp = 0.3
        attn_weights = F.softmax(cand_relevance / selection_temp, dim=1).unsqueeze(-1)

        logits_motif = torch.sum(best_motif_per_cand_per_class * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale

        # 9. Fusion (Fix #5: LayerNorm for balanced scales)
        logits_motif_norm = self.match_ln(logits_motif)
        logits = logits_motif_norm + torch.sigmoid(self.alpha) * logits_global

        # 10. Fix #3: EMA Motif Anchoring — push prototypes toward real data
        if self.training and targets is not None:
            with torch.no_grad():
                for c in range(self.num_classes):
                    mask = (targets == c)
                    if mask.sum() > 0:
                        # Average graph node features of samples belonging to class c
                        class_nodes = node_feats[mask].mean(dim=0)  # (N_graph, D)
                        # Reshape to match motif shape: take first 9 nodes as proxy
                        proxy = class_nodes[:9, :self.motif_bank.motifs.shape[-1]]  # (9, D)
                        # EMA update: 95% old + 5% new
                        self.motif_bank.motifs.data[c] = (
                            0.95 * self.motif_bank.motifs.data[c] + 0.05 * proxy.unsqueeze(0)
                        )

        # Store for aux losses / visualization
        self._latest_scores = scores
        _, top_k_idx = torch.topk(cand_relevance, k=self.top_k, dim=1)
        self._latest_top_k = top_k_idx

        if return_selection:
            return logits, top_k_idx, centers, scores

        return logits

    # ----- Interface methods (compatible with Trainer) -----

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
        l_div = self.compute_motif_diversity_loss()
        return {"motif_diversity": l_div}

    def get_landmark_outputs(self):
        return getattr(self, '_latest_scores', None), getattr(self, '_latest_top_k', None)

    def get_landmark_aux_logits(self):
        return None

    def set_training_progress(self, progress):
        pass

    def get_current_prior_strength(self):
        return 0.0


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        'feat_dim': 128,
        'num_classes': 7,
        'motifs_per_class': 8,
        'top_k': 4
    }
    model = MotifGraphModel(config)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_total:,}")
    print(f"Trainable params: {n_train:,}")

    dummy_img = torch.randn(2, 1, 48, 48)
    dummy_targets = torch.randint(0, 7, (2,))

    out = model(dummy_img, targets=dummy_targets)
    print(f"Output shape: {out.shape}")  # Should be (2, 7)

    aux = model.get_aux_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.4f}")

    loss = F.cross_entropy(out, dummy_targets)
    for v in aux.values():
        loss = loss + 0.1 * v
    loss.backward()
    print(f"Total loss: {loss.item():.4f}")
    print("[OK] Forward + backward pass successful!")