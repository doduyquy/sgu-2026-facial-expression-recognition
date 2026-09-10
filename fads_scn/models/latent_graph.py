import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGraphReasoner(nn.Module):
    """
    Latent Dynamic Graph Reasoner for Facial Expression Recognition (FER).
    Operates directly on M soft semantic tokens produced by Spatial Attention.
    Zero dependency on bounding boxes or facial landmarks.
    
    Pipeline:
    1. Compute soft 2D centers of mass c_m in [0, 1]^2 from attention maps.
    2. Construct Dual-Factor Dynamic Adjacency Matrix:
       A_ij = Softmax( Semantic_Coactivation + Spatial_Distance_Prior )
    3. Gated Residual Graph Convolution (GAT / Graph Transformer style) with FFN.
    4. Self-Attentive Node Importance Readout:
       f_graph = sum( beta_m * h_m ) where beta_m focuses on most active AU deformations.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_nodes: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        init_geo_scale: float = 2.0,
        graph_mode: str = "dense",
        topk: int = 3,
        self_loop_bias: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        if graph_mode not in ("dense", "sparse", "reliability_sparse"):
            raise ValueError("graph_mode must be dense, sparse, or reliability_sparse")
        if graph_mode in ("sparse", "reliability_sparse") and not 1 <= topk < num_nodes:
            raise ValueError("sparse graph modes require 1 <= topk < num_nodes")
        self.graph_mode = graph_mode
        self.topk = topk
        self.self_loop_bias = self_loop_bias
        self.last_graph_gain = None

        # 1. Projections for Dual-Factor Adjacency Matrix
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learnable spatial distance sensitivity parameter
        self.geo_scale = nn.Parameter(torch.tensor([init_geo_scale], dtype=torch.float32))

        # 2. Graph Message Passing Layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 3. Graph Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 4. Self-Attentive Node Importance Readout Gate
        self.readout_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Created only for the proposed graph mode, preserving dense-model
        # checkpoint compatibility. It decides how much graph evidence to use
        # for each image; SCN reliability is detached before entering this gate.
        if self.graph_mode == "reliability_sparse":
            self.reliability_gate = nn.Sequential(
                nn.Linear(2 * embed_dim + 1, max(64, embed_dim // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(64, embed_dim // 2), 1),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                self.reliability_gate[3].bias.fill_(1.0)

    def _compute_soft_centers(self, attn_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute soft continuous 2D coordinates (c_x, c_y) in [0, 1] for each head.
        Args:
            attn_maps: [B, M, H, W]
        Returns:
            centers: [B, M, 2]
        """
        B, M, H, W = attn_maps.shape
        device = attn_maps.device

        # Normalized coordinates [0, 1]
        y_grid = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
        x_grid = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)

        # Weighted expectation of coordinates
        c_y = (attn_maps * y_grid).sum(dim=(-1, -2))  # [B, M]
        c_x = (attn_maps * x_grid).sum(dim=(-1, -2))  # [B, M]

        centers = torch.stack([c_x, c_y], dim=-1)  # [B, M, 2]
        return centers

    @staticmethod
    def edge_entropy_loss(adj_matrix: torch.Tensor) -> torch.Tensor:
        """Normalized edge entropy; uniform rows are penalized more than sharp rows."""
        num_nodes = adj_matrix.size(-1)
        safe_adj = adj_matrix.clamp_min(torch.finfo(adj_matrix.dtype).eps)
        entropy_terms = torch.where(
            adj_matrix > 0,
            adj_matrix * safe_adj.log(),
            torch.zeros_like(adj_matrix),
        )
        entropy = -entropy_terms.sum(dim=-1)
        return entropy.mean() / math.log(max(num_nodes, 2))

    def forward(
        self,
        node_tokens: torch.Tensor,
        attn_maps: torch.Tensor,
        global_features: torch.Tensor = None,
        reliability: torch.Tensor = None,
    ):
        """
        Args:
            node_tokens: [B, M, D] soft regional tokens from spatial attention
            attn_maps: [B, M, H, W] spatial attention distributions
        Returns:
            f_graph: [B, D] graph-level pooled representation
            adj_matrix: [B, M, M] dynamic adjacency matrix
            sparsity_loss: scalar sparsity loss for edges
        """
        B, M, D = node_tokens.shape

        # 1. Soft Spatial Centers
        centers = self._compute_soft_centers(attn_maps)  # [B, M, 2]

        # Pairwise Euclidean distance squared: ||c_i - c_j||^2
        # [B, M, 1, 2] - [B, 1, M, 2] -> [B, M, M, 2]
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)  # [B, M, M]
        geo_prior = -torch.abs(self.geo_scale) * dist_sq  # [B, M, M]

        # 2. Semantic Co-activation Similarity
        Q = self.q_proj(node_tokens)  # [B, M, D]
        K = self.k_proj(node_tokens)  # [B, M, D]
        sem_sim = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)  # [B, M, M]

        # 3. Dual-Factor Adjacency Matrix
        raw_adj = sem_sim + geo_prior
        if self.graph_mode in ("sparse", "reliability_sparse"):
            if self.graph_mode == "reliability_sparse" and (global_features is None or reliability is None):
                raise ValueError("reliability_sparse graph requires global_features and reliability")
            identity = torch.eye(M, dtype=torch.bool, device=node_tokens.device).unsqueeze(0).expand(B, -1, -1)
            # A diagonal prior retains each regional feature. Dynamic top-k
            # neighbors prevent all local regions from being indiscriminately mixed.
            scored_adj = raw_adj + self.self_loop_bias * identity.to(raw_adj.dtype)
            neighbor_scores = scored_adj.masked_fill(identity, torch.finfo(raw_adj.dtype).min)
            neighbor_indices = neighbor_scores.topk(self.topk, dim=-1).indices
            edge_mask = identity.clone()
            edge_mask.scatter_(2, neighbor_indices, True)
            adj_matrix = F.softmax(scored_adj.masked_fill(~edge_mask, torch.finfo(raw_adj.dtype).min), dim=-1)
        else:
            adj_matrix = F.softmax(raw_adj, dim=-1)  # [B, M, M], each row sums to 1.0

        # 4. Graph Message Passing
        V = self.v_proj(node_tokens)  # [B, M, D]
        message = torch.bmm(adj_matrix, V)  # [B, M, D]
        h1 = self.norm1(node_tokens + self.dropout1(message))

        # 5. FFN with Skip-Connection
        h2 = self.norm2(h1 + self.ffn(h1))  # [B, M, D]

        # 6. Self-Attentive Node Importance Readout
        readout_logits = self.readout_gate(h2)  # [B, M, 1]
        beta = F.softmax(readout_logits, dim=1)  # [B, M, 1]
        f_graph = (beta * h2).sum(dim=1)  # [B, D]

        if self.graph_mode == "reliability_sparse":
            # Alpha remains trained by SCN rank regularization, not by a shortcut
            # through classification. The learnable gate can still use visual context.
            alpha = reliability.detach().view(B, 1).to(node_tokens.dtype)
            gate_input = torch.cat([global_features, node_tokens.mean(dim=1), alpha], dim=-1)
            graph_gain = alpha * self.reliability_gate(gate_input)
            f_graph = f_graph * graph_gain
            self.last_graph_gain = graph_gain
        else:
            self.last_graph_gain = node_tokens.new_ones((B, 1))

        # 7. Sparsity regularizer (penalizes overly uniform/diffuse edges)
        sparsity_loss = self.edge_entropy_loss(adj_matrix)

        return f_graph, adj_matrix, sparsity_loss
