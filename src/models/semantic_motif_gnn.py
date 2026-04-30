"""
SemanticMotif-GNN: Semantic Graph + Structured Motif Model for FER.

Architecture:
    CNN Backbone (ResBlock+CBAM) → Semantic Node Extraction (Region Pooling)
    → Hybrid Adjacency (Anatomical + Learned) → GAT Layers
    → Structured MotifBank (symmetry/region constraints)
    → Contrastive Matching (Gumbel top-k + InfoNCE)
    → Combined Classification (motif_logits + α·global_logits)

Designed for FER2013 (48×48 grayscale), trainable on single GPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .CBAM import CBAM
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.CBAM import CBAM


# ---------------------------------------------------------------------------
# 1. CNN Backbone (reused from original, produces 6×6 feature maps)
# ---------------------------------------------------------------------------
class SemanticBackbone(nn.Module):
    """ResBlock + CBAM backbone. Input 48×48 → 6×6×feat_dim."""

    def __init__(self, in_channels=1, feat_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24×24
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
        )
        self.cbam1 = CBAM(64)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 12×12
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
        )
        self.cbam2 = CBAM(128)

        self.down2 = nn.Sequential(
            nn.Conv2d(128, feat_dim, 3, stride=2, padding=1),  # 6×6
            nn.BatchNorm2d(feat_dim), nn.ReLU(),
        )
        self.final_cbam = CBAM(feat_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.cbam1(self.res1(x)) + x)
        x = self.down1(x)
        x = F.relu(self.cbam2(self.res2(x)) + x)
        x = self.down2(x)
        x = self.final_cbam(x)
        return x  # (B, D, 6, 6)


# ---------------------------------------------------------------------------
# 2. Semantic Node Extraction — Region-Based Pooling
# ---------------------------------------------------------------------------
class RegionNodeExtractor(nn.Module):
    """
    Extracts 8 semantic facial-region nodes from a 6×6 feature map.

    Regions (on 6×6 grid):
        0: Left eyebrow   rows 0-1, cols 0-2
        1: Right eyebrow  rows 0-1, cols 3-5
        2: Left eye        rows 1-2, cols 0-2
        3: Right eye       rows 1-2, cols 3-5
        4: Nose bridge     rows 2-3, cols 2-3
        5: Nose / cheeks   rows 3-4, cols 1-4
        6: Left mouth      rows 4-5, cols 0-2
        7: Right mouth     rows 4-5, cols 3-5
    """

    def __init__(self, feat_dim, num_regions=8, H=6, W=6):
        super().__init__()
        self.num_regions = num_regions
        self.H, self.W = H, W

        # Learnable per-region attention (refines fixed masks)
        self.region_attn = nn.Parameter(torch.zeros(num_regions, H, W))

        # Fixed binary masks
        masks = torch.zeros(num_regions, H, W)
        region_defs = [
            (slice(0, 2), slice(0, 3)),   # 0 left eyebrow
            (slice(0, 2), slice(3, 6)),   # 1 right eyebrow
            (slice(1, 3), slice(0, 3)),   # 2 left eye
            (slice(1, 3), slice(3, 6)),   # 3 right eye
            (slice(2, 4), slice(2, 4)),   # 4 nose bridge
            (slice(3, 5), slice(1, 5)),   # 5 nose / cheeks
            (slice(4, 6), slice(0, 3)),   # 6 left mouth
            (slice(4, 6), slice(3, 6)),   # 7 right mouth
        ]
        for i, (rs, cs) in enumerate(region_defs):
            masks[i, rs, cs] = 1.0
        self.register_buffer("masks", masks)

        # Small projection to align pooled features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
        )

    def forward(self, feat_map):
        """
        Args:
            feat_map: (B, D, H, W)
        Returns:
            nodes: (B, num_regions, D)
        """
        B, D, H, W = feat_map.shape
        attn = torch.sigmoid(self.region_attn)              # (R, H, W)
        weighted = self.masks * attn                         # (R, H, W)
        weighted = weighted / (weighted.sum(dim=(1, 2), keepdim=True) + 1e-8)

        feat_flat = feat_map.view(B, D, H * W)              # (B, D, HW)
        masks_flat = weighted.view(self.num_regions, H * W)  # (R, HW)

        nodes = torch.einsum("bdn,rn->brd", feat_flat, masks_flat)  # (B, R, D)
        nodes = self.proj(nodes)
        return nodes


# ---------------------------------------------------------------------------
# 3. Hybrid Adjacency — Fixed Anatomical + Learned Edges
# ---------------------------------------------------------------------------
class HybridAdjacency(nn.Module):
    """
    Edge-wise gated fusion of anatomical prior and learned adjacency.

    Each edge (i, j) has its own gate value g_ij ∈ (0, 1) that decides:
        adj_ij = g_ij * A_fixed_ij + (1 - g_ij) * A_learned_ij

    This replaces the coarse scalar gate with a per-edge adaptive mechanism,
    letting the model independently control anatomical trust per relationship.
    """

    def __init__(self, num_nodes=8, feat_dim=64):
        super().__init__()
        self.num_nodes = num_nodes

        # ── Fixed anatomical adjacency ──
        A_fixed = torch.zeros(num_nodes, num_nodes)
        anatomical_edges = [
            (0, 2), (1, 3),        # eyebrow ↔ eye (same side)
            (0, 1),                # left ↔ right eyebrow
            (2, 3),                # left ↔ right eye
            (2, 4), (3, 4),        # eyes ↔ nose bridge
            (4, 5),                # nose bridge ↔ cheeks
            (5, 6), (5, 7),        # cheeks ↔ mouth
            (6, 7),                # left ↔ right mouth
        ]
        for i, j in anatomical_edges:
            A_fixed[i, j] = A_fixed[j, i] = 1.0
        A_fixed += torch.eye(num_nodes)  # self-loops
        self.register_buffer("A_fixed", A_fixed)

        # ── Edge-wise gate MLP ──
        # Decides per-edge: "trust anatomical prior (→1) or learned (→0)?"
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── Learned edge weight MLP ──
        # Produces the data-driven edge strength A_learned_ij ∈ (0, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._last_A_learned = None

    def forward(self, node_feats):
        """
        Args:
            node_feats: (B, N, D)
        Returns:
            adj: (B, N, N) row-normalized, symmetric weighted adjacency
        """
        B, N, D = node_feats.shape

        # Pairwise feature concatenation: (B, N, N, 2D)
        fi = node_feats.unsqueeze(2).expand(-1, -1, N, -1)
        fj = node_feats.unsqueeze(1).expand(-1, N, -1, -1)
        pairs = torch.cat([fi, fj], dim=-1)

        # Per-edge gate: g_ij ∈ (0, 1)  — high = trust anatomical
        gate = self.gate_mlp(pairs).squeeze(-1)       # (B, N, N)

        # Learned edge weights: A_learned_ij ∈ (0, 1)
        A_learned = self.edge_mlp(pairs).squeeze(-1)  # (B, N, N)

        # --- Step 3: Top-k edges sparsity ---
        # Only keep top 4 strongest learned connections per node
        k_edges = min(4, N)
        vals, idx = torch.topk(A_learned, k_edges, dim=-1)
        mask = torch.zeros_like(A_learned).scatter_(-1, idx, 1.0)
        A_learned = A_learned * mask
        
        # Save for entropy loss (Step 1)
        self._last_A_learned = A_learned

        # Edge-wise fusion
        A_fixed = self.A_fixed.unsqueeze(0)            # (1, N, N)
        adj = gate * A_fixed + (1 - gate) * A_learned  # (B, N, N)

        # Symmetrize: adj_ij = (adj_ij + adj_ji) / 2
        adj = (adj + adj.transpose(-2, -1)) * 0.5

        # Guarantee self-loops (clamp diagonal to >= 1)
        diag_mask = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj * (1 - diag_mask) + torch.clamp(adj * diag_mask, min=1.0)

        # Row-normalize for numerical stability
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj = adj / row_sum

        return adj


# ---------------------------------------------------------------------------
# 4. Graph Attention Layer (Multi-Head)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention with adjacency masking."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        assert out_dim % heads == 0

        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   (B, N, in_dim)
            adj: (B, N, N)
        Returns:
            out: (B, N, out_dim)
        """
        B, N, _ = x.shape
        residual = x

        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply adjacency mask
        if adj is not None:
            mask = (adj.unsqueeze(1) < 0.01)  # mask out near-zero edges
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_lin(out)

        # Residual + LayerNorm (if dims match)
        if residual.shape[-1] == out.shape[-1]:
            out = self.norm(out + residual)
        else:
            out = self.norm(out)

        return F.relu(out)


# ---------------------------------------------------------------------------
# 5. Structured MotifBank with Symmetry/Region Constraints
# ---------------------------------------------------------------------------
class StructuredMotifBank(nn.Module):
    """
    Emotion-specific motif prototypes aligned with semantic graph topology.
    Each motif has the same node count as the semantic graph (num_nodes=8).
    """

    def __init__(self, num_classes=7, motifs_per_class=4,
                 num_nodes=8, feat_dim=64):
        super().__init__()
        self.C = num_classes
        self.M = motifs_per_class
        self.N = num_nodes
        self.D = feat_dim

        # Learnable motif node features
        self.motif_feats = nn.Parameter(
            torch.randn(num_classes, motifs_per_class, num_nodes, feat_dim) * 0.02
        )

        # Region zones: 0=upper face (brows+eyes), 1=middle (nose), 2=lower (mouth)
        zone = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.register_buffer("region_zone", zone)

        # Symmetry pairs: (left_idx, right_idx)
        self.symmetry_pairs = [(0, 1), (2, 3), (6, 7)]

    def get_motifs(self):
        """Returns: (C, M, N, D)"""
        return self.motif_feats

    def symmetry_loss(self):
        """Enforce left-right consistency in motif node features."""
        m = self.motif_feats  # (C, M, N, D)
        loss = torch.tensor(0.0, device=m.device)
        for li, ri in self.symmetry_pairs:
            diff = (m[:, :, li, :] - m[:, :, ri, :]).pow(2).mean()
            loss = loss + diff
        return loss / max(len(self.symmetry_pairs), 1)

    def region_consistency_loss(self):
        """
        Nodes in same facial zone should be more similar than cross-zone.
        Uses margin-based contrastive: ReLU(margin + cross_sim - intra_sim).
        """
        m = self.motif_feats  # (C, M, N, D)
        m_norm = F.normalize(m, dim=-1)
        # (C, M, N, N) pairwise similarity
        sim = torch.einsum("cmid,cmjd->cmij", m_norm, m_norm)

        zone = self.region_zone  # (N,)
        # Intra-zone mask: same zone
        zone_eq = (zone.unsqueeze(0) == zone.unsqueeze(1)).float()  # (N, N)
        # Remove diagonal
        eye = torch.eye(self.N, device=m.device)
        intra_mask = zone_eq * (1 - eye)
        cross_mask = (1 - zone_eq)

        intra_count = intra_mask.sum().clamp(min=1)
        cross_count = cross_mask.sum().clamp(min=1)

        intra_sim = (sim * intra_mask).sum() / (self.C * self.M * intra_count)
        cross_sim = (sim * cross_mask).sum() / (self.C * self.M * cross_count)

        return F.relu(0.2 + cross_sim - intra_sim)

    def diversity_loss(self):
        """Encourage inter-class motif separation + intra-class diversity."""
        m = self.motif_feats.view(self.C, self.M, -1)  # (C, M, N*D)
        m_norm = F.normalize(m, dim=-1)

        # Inter-class: class centers should be dissimilar
        centers = m_norm.mean(dim=1)  # (C, N*D)
        centers = F.normalize(centers, dim=-1)
        sim_inter = torch.mm(centers, centers.t())
        eye = torch.eye(self.C, device=m.device)
        l_inter = (sim_inter * (1 - eye)).pow(2).mean()

        # Intra-class: motifs within same class should be diverse
        sim_intra = torch.bmm(m_norm, m_norm.transpose(1, 2))  # (C, M, M)
        eye_m = torch.eye(self.M, device=m.device).unsqueeze(0)
        l_intra = (sim_intra * (1 - eye_m)).pow(2).mean()

        return l_inter + 0.5 * l_intra


# ---------------------------------------------------------------------------
# 6. Contrastive Motif Matcher with Gumbel Noise
# ---------------------------------------------------------------------------
class ContrastiveMotifMatcher(nn.Module):
    """
    Matches graph nodes to motif prototypes using node-aligned cosine
    similarity with InfoNCE contrastive loss and Gumbel noise for
    differentiable hard selection.
    """

    def __init__(self, feat_dim, num_classes=7, motifs_per_class=4, tau=0.07, num_nodes=8):
        super().__init__()
        self.tau = tau
        self.C = num_classes
        self.M = motifs_per_class
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        # Learnable node importance for weighted similarity (attention over nodes)
        self.node_importance = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, graph_nodes, motif_bank, targets=None):
        """
        Args:
            graph_nodes: (B, N, D) — graph-enriched semantic node features
            motif_bank:  (C, M, N, D) — structured motif prototypes
            targets:     (B,) — class labels (optional, for contrastive loss)
        Returns:
            logits: (B, C) — class scores
            contrastive_loss: scalar or None
        """
        B, N, D = graph_nodes.shape
        C, M = self.C, self.M

        g = F.normalize(self.proj(graph_nodes), dim=-1)                 # (B, N, D)
        m = F.normalize(motif_bank.view(C * M, N, D), dim=-1)          # (CM, N, D)

        # Node-aligned cosine similarity: (B, CM, N)
        sim = torch.einsum("bnd,mnd->bmn", g, m)

        # Weighted node attention instead of simple average
        node_weight = F.softmax(self.node_importance, dim=-1)  # (N,)
        sim = (sim * node_weight).sum(dim=-1)                  # (B, CM)
        
        sim_scaled = sim / self.tau

        # Gumbel noise for exploration during training
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(sim_scaled) + 1e-8) + 1e-8)
            sim_scaled = sim_scaled + gumbel * 0.05

        # Reshape to (B, C, M)
        sim_class = sim_scaled.view(B, C, M)
        
        # Softmax pooling (logsumexp) instead of hard top-k
        # This provides smoother gradients and allows all motifs to learn
        match_sim = torch.logsumexp(sim_class, dim=-1)   # (B, C)

        # InfoNCE contrastive loss
        contrastive_loss = None
        if targets is not None:
            contrastive_loss = F.cross_entropy(match_sim, targets)

        return match_sim, contrastive_loss


# ---------------------------------------------------------------------------
# 7. Full Model
# ---------------------------------------------------------------------------
class SemanticMotifGNN(nn.Module):
    """
    SemanticMotif-GNN: complete model combining semantic graph construction,
    structured motif prototypes, and contrastive matching for FER.
    """

    def __init__(self, config):
        super().__init__()
        D = config.get("feat_dim", 64)
        C = config.get("num_classes", 7)
        M = config.get("motifs_per_class", 4)
        N = config.get("num_regions", 8)
        heads = config.get("gat_heads", 4)
        tau = config.get("match_tau", 0.07)
        dropout = config.get("dropout", 0.3)

        self.feat_dim = D
        self.num_classes = C

        # --- Backbone ---
        self.backbone = SemanticBackbone(in_channels=1, feat_dim=D)

        # --- Semantic Graph ---
        self.node_extractor = RegionNodeExtractor(D, num_regions=N)
        self.adjacency = HybridAdjacency(num_nodes=N, feat_dim=D)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(D, D, heads=heads, dropout=dropout),
            GraphAttentionLayer(D, D, heads=heads, dropout=dropout),
        ])

        # --- Structured Motifs ---
        self.motif_bank = StructuredMotifBank(C, M, N, D)

        # --- Contrastive Matcher ---
        self.matcher = ContrastiveMotifMatcher(D, C, M, tau=tau, num_nodes=N)

        # --- Global Branch ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(D, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, C),
        )

        # Fusion logic
        self.match_ln = nn.LayerNorm(C)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Internal state
        self._contrastive_loss = None
        self._latest_scores = None

    def forward(self, x, targets=None):
        B = x.shape[0]
        self._latest_targets = targets

        # 1. CNN feature extraction
        feat_map = self.backbone(x)                     # (B, D, 6, 6)

        # 2. Global branch
        logits_global = self.global_fc(self.global_pool(feat_map))  # (B, C)

        # 3. Semantic graph construction
        nodes = self.node_extractor(feat_map)            # (B, N, D)
        adj = self.adjacency(nodes)                      # (B, N, N)

        # 4. Graph message passing
        for gat in self.gat_layers:
            nodes = gat(nodes, adj)                      # (B, N, D)

        # 5. Motif matching
        motifs = self.motif_bank.get_motifs()             # (C, M, N, D)

        # EMA Prototype Update: push motifs towards the actual graph nodes of their class
        if self.training and targets is not None:
            with torch.no_grad():
                for c in range(self.num_classes):
                    mask = (targets == c)
                    if mask.sum() > 0:
                        class_nodes = nodes[mask].mean(dim=0)  # (N, D)
                        # 0.99 decay for stability
                        self.motif_bank.motif_feats.data[c] = \
                            0.95 * self.motif_bank.motif_feats.data[c] + 0.05 * class_nodes.unsqueeze(0)

        match_logits, contrastive_loss = self.matcher(
            nodes, motifs, targets
        )                                                 # (B, C)
        self._contrastive_loss = contrastive_loss
        self._latest_scores = match_logits.detach()

        # 6. Fusion
        gate = torch.sigmoid(self.alpha)
        # Scale match logits to match global_logits scale via LayerNorm
        match_logits_norm = self.match_ln(match_logits)
        logits = match_logits_norm + gate * logits_global

        return logits

    # ---- Interface methods expected by Trainer ----

    def get_aux_losses(self):
        losses = {}
        losses["motif_symmetry"] = self.motif_bank.symmetry_loss()
        losses["motif_region"] = self.motif_bank.region_consistency_loss()
        losses["motif_diversity"] = self.motif_bank.diversity_loss()
        
        # Binary Entropy on learned adjacency (Step 1)
        if hasattr(self.adjacency, "_last_A_learned") and self.adjacency._last_A_learned is not None:
            A = self.adjacency._last_A_learned
            eps = 1e-6
            entropy = -(A * torch.log(A + eps) + (1 - A) * torch.log(1 - A + eps))
            losses["adjacency_entropy"] = entropy.mean()

        if self._contrastive_loss is not None:
            losses["contrastive_match"] = self._contrastive_loss
        return losses

    def get_landmark_outputs(self):
        return self._latest_scores, None

    def get_landmark_aux_logits(self):
        return None

    def set_training_progress(self, progress):
        pass

    def get_current_prior_strength(self):
        return 0.0

    # ---- Two-phase training support (used by train_hybrid.py) ----

    def freeze_for_phase1(self):
        """
        Phase 1 — CNN warmup: freeze graph + motif components,
        train only backbone + global classifier.
        """
        # Freeze: node extractor, adjacency, GAT, motif bank, matcher
        frozen_modules = [
            self.node_extractor,
            self.adjacency,
            self.motif_bank,
            self.matcher,
        ]
        frozen_modules.extend(self.gat_layers)

        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        # Keep backbone + global_fc + alpha trainable (they are by default)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Phase1 Freeze] Trainable: {trainable:,} / {total:,} params")

    def unfreeze_all(self):
        """Phase 2 — unfreeze everything for end-to-end training."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Unfreeze All] Trainable: {trainable:,} params")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "feat_dim": 64,
        "num_classes": 7,
        "motifs_per_class": 4,
        "num_regions": 8,
        "gat_heads": 4,
        "match_tau": 0.07,
        "dropout": 0.3,
    }
    model = SemanticMotifGNN(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Forward pass
    dummy = torch.randn(4, 1, 48, 48)
    targets = torch.randint(0, 7, (4,))

    logits = model(dummy, targets=targets)
    print(f"Output shape: {logits.shape}")  # (4, 7)

    # Aux losses
    aux = model.get_aux_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.4f}")

    # Backward
    loss = F.cross_entropy(logits, targets)
    for v in aux.values():
        loss = loss + 0.1 * v
    loss.backward()
    print(f"Total loss: {loss.item():.4f}")
    print("[OK] Forward + backward pass successful!")
