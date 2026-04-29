"""
Hybrid CNN + Graph Neural Network + Motif Model for FER2013
===========================================================
Architecture:
  (A) CNNEncoder       : 3 Conv layers → patch feature map (B,64,6,6) + global feat (B,64)
  (B) build_knn_graph  : patch nodes → PyG Batch, k=8, correct batch tensor (no cross-image edges)
  (C) GNNEncoder       : GATConv×2 with residual → (N_total, 64)
  (D) MotifLayer       : K=16 motifs, cosine sim + temperature, attention pooling → (B,16)
  (E) NodeSelector     : MLP→sigmoid weight × node emb → global_add_pool → (B,64)
  (F) Fusion           : cat(CNN:64, Graph:64, Motif:16) = 144
  (G) Classifier       : 144→64→Dropout→7

Fixes applied (from review):
  ✅ kNN batch tensor prevents cross-image edges
  ✅ Attention pooling for motif aggregation (not mean)
  ✅ global_add_pool(x * w, batch) in NodeSelector
  ✅ F.normalize before motif diversity gram → true cosine penalty
  ✅ Phase-1 freezes GNN/Motif/NodeSelector only (classifier stays trainable)
  ✅ Residual connections in GNN
  ✅ Temperature scaling in motif cosine similarity
  ✅ Dropout(0.3) in classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── PyTorch Geometric imports ──────────────────────────────────────────────────
try:
    from torch_geometric.nn import GATConv, global_add_pool
    from torch_geometric.utils import scatter
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    import warnings
    warnings.warn(
        "[MotifGraphFER] torch_geometric not found. "
        "Install with: pip install torch_geometric\n"
        "Falling back to dense-batch GNN (no PyG).",
        stacklevel=2,
    )

    # ── Minimal scatter fallback (pure PyTorch) ───────────────────────────────
    def scatter(src, index, dim=0, reduce='sum', dim_size=None):
        """Minimal scatter implementation for non-PyG environments."""
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        expand_shape = [-1] * src.dim()
        expand_shape[dim] = -1
        idx = index
        # Expand index to match src dimensions
        for _ in range(src.dim() - 1):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)
        if reduce == 'sum':
            out.scatter_add_(dim, idx, src)
        elif reduce == 'max':
            # scatter_reduce_ is in-place and returns self (not a tuple)
            out.fill_(-1e9)
            out.scatter_reduce_(dim, idx, src, reduce='amax', include_self=True)
        return out

    def global_add_pool(x, batch, size=None):
        """Fallback global add pool."""
        if size is None:
            size = int(batch.max().item()) + 1
        return scatter(x, batch, dim=0, reduce='sum', dim_size=size)


# ══════════════════════════════════════════════════════════════════════════════
# (A) CNN ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class CNNEncoder(nn.Module):
    """
    3-layer CNN encoder.
    Input : (B, 1, 48, 48)
    Output:
      feat_map    : (B, feat_dim, 6, 6)  — patch-level features for graph
      cnn_global  : (B, feat_dim)         — global image feature via avg-pool
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim

        # Conv block 1: spatial 48 → 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 48→24
        )
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # Patch pool: (B, feat_dim, 24, 24) → (B, feat_dim, 6, 6)
        self.patch_pool = nn.AdaptiveAvgPool2d((6, 6))
        # Global pool: (B, feat_dim, 6, 6) → (B, feat_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)   # (B, 32, 24, 24)
        x = self.conv2(x)   # (B, 64, 24, 24)
        x = self.conv3(x)   # (B, feat_dim, 24, 24)

        patch_map  = self.patch_pool(x)                           # (B, feat_dim, 6, 6)
        cnn_global = self.global_pool(patch_map).flatten(1)       # (B, feat_dim)
        return patch_map, cnn_global


# ══════════════════════════════════════════════════════════════════════════════
# (B) GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_graph(patch_map: torch.Tensor, k: int = 8):
    """
    Native PyTorch implementation of kNN graph (no torch-cluster required).
    Optimized for constant N nodes per image.
    """
    B, C, H, W = patch_map.shape
    N = H * W
    
    # (B, C, H, W) -> (B, N, C)
    x = patch_map.permute(0, 2, 3, 1).reshape(B, N, C)
    
    # Tính khoảng cách Euclidean giữa các node trong từng ảnh
    # dist shape: (B, N, N)
    dist = torch.cdist(x, x)
    
    # Loại bỏ self-loop bằng cách đặt khoảng cách tới chính nó là vô hạn
    idx_eye = torch.arange(N, device=x.device)
    dist[:, idx_eye, idx_eye] = float('inf')
    
    # Lấy k láng giềng gần nhất
    # topk_indices shape: (B, N, k)
    _, topk_indices = dist.topk(k, dim=-1, largest=False)
    
    # Tạo edge_index (2, B * N * k)
    # Row indices: [0, 0, ..., 0, 1, 1, ..., N-1] cho mỗi batch
    row = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, k)
    col = topk_indices
    
    # Offset theo batch index để các node không bị nối nhầm sang ảnh khác
    batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
    row = (row + batch_offset).reshape(-1)
    col = (col + batch_offset).reshape(-1)
    
    edge_index = torch.stack([row, col], dim=0)
    
    # Flatten node features và tạo batch_idx
    node_feats = x.reshape(B * N, C)
    batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
    
    return node_feats, edge_index, batch_idx


# ══════════════════════════════════════════════════════════════════════════════
# (C) GNN ENCODER  (GATConv × 2 with residual)
# ══════════════════════════════════════════════════════════════════════════════

class GNNEncoder(nn.Module):
    """
    Two GATConv layers with residual connections.
    Input : node_feats (N_total, feat_dim), edge_index
    Output: node_emb   (N_total, feat_dim)
    """

    def __init__(self, feat_dim: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()

        if _HAS_PYG:
            # concat=False → output dim == feat_dim (not heads*feat_dim)
            self.gat1 = GATConv(feat_dim, feat_dim, heads=heads, concat=False,
                                dropout=dropout, add_self_loops=True)
            self.gat2 = GATConv(feat_dim, feat_dim, heads=heads, concat=False,
                                dropout=dropout, add_self_loops=True)
        else:
            # Fallback: simple linear layers acting as graph-free encoders
            self.gat1 = nn.Linear(feat_dim, feat_dim)
            self.gat2 = nn.Linear(feat_dim, feat_dim)

        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Layer 1 with residual: x = x + GATConv(x)
        if _HAS_PYG:
            h = self.gat1(x, edge_index)
        else:
            h = self.gat1(x)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.drop(h)
        x = x + h          # ✅ Residual

        # Layer 2 with residual
        if _HAS_PYG:
            h = self.gat2(x, edge_index)
        else:
            h = self.gat2(x)
        h = self.bn2(h)
        h = F.elu(h)
        x = x + h          # ✅ Residual

        return x   # (N_total, feat_dim)


# ══════════════════════════════════════════════════════════════════════════════
# (D) MOTIF LAYER
# ══════════════════════════════════════════════════════════════════════════════

class MotifLayer(nn.Module):
    """
    Learnable motif bank with attention-weighted aggregation.

    Parameters
    ----------
    num_motifs  : K = 16  learnable motif vectors
    feat_dim    : 64
    temperature : scaling for cosine logits (lower = sharper)

    Forward
    -------
    Input  : node_emb (N_total, feat_dim),  batch (N_total,),  num_graphs B
    Output : motif_vec (B, K),  sim_map (N_total, K)
    """

    def __init__(self, num_motifs: int = 16, feat_dim: int = 64,
                 temperature: float = 0.1):
        super().__init__()
        self.K   = num_motifs
        self.tau = temperature

        # Learnable motif bank: (K, feat_dim)
        # ✅ Xavier init without in-place view error
        init_data = torch.empty(1, num_motifs, feat_dim)
        nn.init.xavier_uniform_(init_data)
        self.motif_bank = nn.Parameter(init_data.squeeze(0))  # (K, feat_dim)

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        """
        node_emb : (N_total, D)
        batch    : (N_total,)  — which graph each node belongs to
        Returns:
          motif_vec : (B, K)
          sim_map   : (N_total, K)
        """
        # ── Cosine similarity with temperature ────────────────────────────────
        x_norm = F.normalize(node_emb, dim=1)               # (N_total, D)
        m_norm = F.normalize(self.motif_bank, dim=1)         # (K, D)
        sim_map = (x_norm @ m_norm.T) / self.tau            # (N_total, K)

        # ── Attention pooling per graph ────────────────────────────────────────
        # For each graph g and motif k:
        #   attn_{g,k} = softmax over nodes in g of sim[nodes_g, k]
        #   motif_vec[g, k] = sum_n (attn[n,k] * sim[n,k])
        #
        # ✅ Manual per-graph softmax (no extra dependency)
        # Subtract per-graph max for numerical stability
        sim_max = scatter(sim_map, batch, dim=0, reduce='max',
                          dim_size=num_graphs)                # (B, K)
        sim_shifted = sim_map - sim_max[batch]               # (N_total, K)
        exp_sim = torch.exp(sim_shifted)                     # (N_total, K)
        sum_exp = scatter(exp_sim, batch, dim=0, reduce='sum',
                          dim_size=num_graphs)                # (B, K)
        # attn for each node (N_total, K)
        attn = exp_sim / (sum_exp[batch] + 1e-8)

        # Weighted sum → motif_vec
        motif_vec = scatter(attn * sim_map, batch, dim=0, reduce='sum',
                            dim_size=num_graphs)              # (B, K)

        return motif_vec, sim_map

    def diversity_loss(self) -> torch.Tensor:
        """
        Penalise high off-diagonal cosine similarity between motifs.
        ✅ Uses F.normalize → true cosine gram.
        """
        m = F.normalize(self.motif_bank, dim=1)  # (K, D)
        gram = m @ m.T                            # (K, K) — cosine similarities
        eye  = torch.eye(self.K, device=m.device)
        # Only penalise off-diagonal entries
        off_diag = gram * (1.0 - eye)
        return off_diag.pow(2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# (E) NODE SELECTOR  (attention-weighted pooling)
# ══════════════════════════════════════════════════════════════════════════════

class NodeSelector(nn.Module):
    """
    MLP → sigmoid weight per node → weighted global pooling.
    ✅ Uses global_add_pool(x * w, batch) — respects PyG batch.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor,
                num_graphs: int) -> torch.Tensor:
        """
        node_emb  : (N_total, feat_dim)
        batch     : (N_total,)
        Returns   : graph_feat (B, feat_dim)
        """
        w = self.mlp(node_emb)              # (N_total, 1)
        weighted = node_emb * w             # (N_total, feat_dim)  element-wise scale

        if _HAS_PYG:
            # ✅ global_add_pool respects batch boundaries
            graph_feat = global_add_pool(weighted, batch,
                                         size=num_graphs)   # (B, feat_dim)
        else:
            # Fallback: manual scatter sum
            graph_feat = scatter(weighted, batch, dim=0, reduce='sum',
                                 dim_size=num_graphs)        # (B, feat_dim)

        return graph_feat


# ══════════════════════════════════════════════════════════════════════════════
# (F+G) FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MotifGraphModel(nn.Module):
    """
    Hybrid CNN + GNN + Motif model for FER2013.

    Fusion vector: cat(cnn_global:64, graph_feat:64, motif_vec:16) = 144
    Classifier:    144 → 64 → Dropout(0.3) → 7
    """

    def __init__(self, config: dict):
        super().__init__()
        # ── Hyperparameters ────────────────────────────────────────────────────
        self.feat_dim    = config.get('feat_dim',    64)
        self.num_classes = config.get('num_classes', 7)
        self.k_neighbors = config.get('k_neighbors', 8)
        self.num_motifs  = config.get('num_motifs',  16)
        self.gat_heads   = config.get('gat_heads',   4)
        self.motif_tau   = config.get('motif_tau',   0.1)
        self.dropout     = config.get('dropout',     0.3)
        self.motif_div_weight = config.get('motif_div_weight', 0.2)

        D  = self.feat_dim
        K  = self.num_motifs
        NC = self.num_classes

        # ── Blocks ────────────────────────────────────────────────────────────
        self.cnn_encoder   = CNNEncoder(in_channels=1, feat_dim=D)
        self.gnn_encoder   = GNNEncoder(feat_dim=D, heads=self.gat_heads,
                                        dropout=0.1)
        self.motif_layer   = MotifLayer(num_motifs=K, feat_dim=D,
                                        temperature=self.motif_tau)
        self.node_selector = NodeSelector(feat_dim=D)

        # Fusion + classifier: cat(D, D, K) = 2D+K
        fusion_dim = D + D + K   # 64 + 64 + 16 = 144
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),   # ✅ Dropout in classifier
            nn.Linear(64, NC),
        )

        # Internal state (for Trainer compatibility)
        self._latest_motif_div_loss: torch.Tensor | None = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, targets=None) -> torch.Tensor:
        """
        x       : (B, 1, 48, 48)
        targets : unused (kept for Trainer compatibility)
        Returns : logits (B, num_classes)
        """
        B = x.shape[0]

        # (A) CNN
        patch_map, cnn_global = self.cnn_encoder(x)
        # patch_map : (B, 64, 6, 6)
        # cnn_global: (B, 64)

        # (B) Graph construction
        node_feats, edge_index, batch_idx = build_knn_graph(
            patch_map, k=self.k_neighbors
        )

        # (C) GNN
        node_emb = self.gnn_encoder(node_feats, edge_index)
        # node_emb: (B*36, 64)

        # (D) Motif layer  → (B, K) + (N_total, K)
        motif_vec, sim_map = self.motif_layer(node_emb, batch_idx, B)

        # Cache diversity loss for get_aux_losses()
        self._latest_motif_div_loss = self.motif_layer.diversity_loss()

        # (E) Node selector → (B, 64)
        graph_feat = self.node_selector(node_emb, batch_idx, B)

        # (F) Fusion: cat(64, 64, 16) = 144
        fused = torch.cat([cnn_global, graph_feat, motif_vec], dim=1)

        # (G) Classifier
        logits = self.classifier(fused)   # (B, 7)
        return logits

    # ── Phase-1 helpers (freeze/unfreeze for staged training) ─────────────────

    def freeze_for_phase1(self):
        """
        Phase 1 — CNN warmup.
        ✅ Freeze GNN, Motif, NodeSelector.
        ✅ Keep CNNEncoder + Classifier trainable so CNN aligns with output.
        """
        for name, param in self.named_parameters():
            if any(name.startswith(k) for k in
                   ('gnn_encoder', 'motif_layer', 'node_selector')):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        print("[Phase 1] Frozen: gnn_encoder, motif_layer, node_selector. "
              "Trainable: cnn_encoder, classifier.")

    def unfreeze_all(self):
        """Phase 2 — full end-to-end training."""
        for param in self.parameters():
            param.requires_grad_(True)
        print("[Phase 2] All parameters unfrozen.")

    # ── Trainer-compatible interface ──────────────────────────────────────────

    def get_aux_losses(self) -> dict:
        """
        Called by Trainer after forward() to collect auxiliary losses.
        Key 'motif_diversity' is weighted by motif_div_weight in Trainer.
        """
        if self._latest_motif_div_loss is None:
            return {}
        return {"motif_diversity": self._latest_motif_div_loss}

    def get_landmark_outputs(self):
        """Stub — required by Trainer."""
        return None, None

    def get_landmark_aux_logits(self):
        """Stub — required by Trainer."""
        return None

    def set_training_progress(self, progress: float):
        """Stub — required by Trainer phase schedule."""
        pass

    def get_current_prior_strength(self) -> float:
        """Stub — required by Trainer logging."""
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = {
        'feat_dim': 64,
        'num_classes': 7,
        'k_neighbors': 8,
        'num_motifs': 16,
        'gat_heads': 4,
        'motif_tau': 0.1,
        'dropout': 0.3,
        'motif_div_weight': 0.2,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MotifGraphModel(cfg).to(device)

    dummy  = torch.randn(4, 1, 48, 48, device=device)
    logits = model(dummy)
    print(f"✅ Output shape  : {logits.shape}")   # (4, 7)

    aux = model.get_aux_losses()
    print(f"✅ motif_diversity: {aux['motif_diversity'].item():.4f}")

    # Phase-1 freeze check
    model.freeze_for_phase1()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"✅ Phase-1 trainable params: {n_train:,} / {n_total:,}")

    model.unfreeze_all()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Phase-2 trainable params: {n_train:,} / {n_total:,}")
