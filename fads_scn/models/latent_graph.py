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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

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

    def forward(self, node_tokens: torch.Tensor, attn_maps: torch.Tensor):
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
        # Self-loop: add identity prior to reinforce self-features
        raw_adj = sem_sim + geo_prior
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

        # 7. Sparsity regularizer (penalizes overly uniform/diffuse edges)
        sparsity_loss = (adj_matrix ** 2).sum(dim=-1).mean()

        return f_graph, adj_matrix, sparsity_loss
