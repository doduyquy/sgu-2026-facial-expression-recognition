import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.training.losses import MotifConsistencyLoss

try:
    from .CBAM import CBAM
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.CBAM import CBAM


class SpatialResidualMasking(nn.Module):
    """
    Lightweight Spatial Residual Masking Block.
    Generates a spatial attention mask and applies it via a residual connection.
    This suppresses background noise and highlights micro-expressions.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Bottleneck to reduce parameters
        reduced_channels = in_channels // 4
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mask = self.mask_generator(x)
        # Residual masking: x' = x + x * M
        return x + x * mask


class MotifBackbone(nn.Module):
    """
    Advanced Backbone with Pretrained ResNet18 for stronger feature extraction.
    """
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        import torchvision.models as models
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet18(pretrained=True)
            
        # Keep original 3-channel pretrained conv1 but change stride to 1 to keep spatial size 6x6
        self.conv1 = resnet.conv1
        self.conv1.stride = (1, 1)
        self.conv1.padding = (3, 3)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        
        # Skip maxpool to keep spatial size 6x6 at the end (48->48->24->12->6)
        self.maxpool = nn.Identity()
        
        self.layer1 = resnet.layer1 # 48x48
        self.layer2 = resnet.layer2 # 24x24
        self.layer3 = resnet.layer3 # 12x12
        self.layer4 = resnet.layer4 # 6x6, 512 channels
        
        self.residual_masking = SpatialResidualMasking(768)
        
        # Reduce dimension to expected feat_dim (128)
        self.dim_reducer = nn.Sequential(
            nn.Conv2d(768, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )

    def load_pretrained_cnn(self, checkpoint_path):
        import os
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: CNN checkpoint {checkpoint_path} not found. Skipping.")
            return

        print(f"Loading pretrained CNN from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
            
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
            
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            # Clean common prefixes from RMN or generic wrappers
            name = k.replace('module.', '').replace('backbone.', '').replace('resnet.', '').replace('net.', '')
            
            if name in model_dict:
                if v.shape == model_dict[name].shape:
                    pretrained_dict[name] = v
                else:
                    # Skip layers with shape mismatch (e.g., conv1 7x7 vs 3x3)
                    pass
                    
        if len(pretrained_dict) == 0:
            print("WARNING: No matching keys found. Checkpoint format might be unsupported.")
        else:
            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} matching layers from CNN checkpoint.")
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        # Repeat 1-channel input to 3 channels to match pretrained conv1 weight shape
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        
        # Upsample layer4 (6x6) to match layer3 (12x12) spatial size
        x4_up = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x_combined = torch.cat([x3, x4_up], dim=1) # (B, 768, 12, 12)
        
        x = self.residual_masking(x_combined)
        x = self.dim_reducer(x)
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer (GAT) for small graphs.
    """
    def __init__(self, in_dim, out_dim, heads=8):
        super().__init__()
        self.heads = heads # Number of attention heads
        self.d_k = out_dim // heads # Dimension per head
        
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(out_dim, out_dim)
        
        # Edge-aware attention + learnable adjacency gating
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )
        # Residual + norm + attention dropout
        self.attn_drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        B, N, C = x.shape
        
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2) 
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        # (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Decomposed memory-efficient computation of edge_gate and edge_bias
        # Deconstruct edge_gate[0] (nn.Linear(2 * C, C))
        gate_fc1 = self.edge_gate[0]
        gate_w_i = gate_fc1.weight[:, :C]
        gate_w_j = gate_fc1.weight[:, C:]
        gate_bias = gate_fc1.bias
        
        gate_feat_i = F.linear(x, gate_w_i) # (B, N, C)
        gate_feat_j = F.linear(x, gate_w_j) # (B, N, C)
        # Broadcast add to shape (B, N, N, C) without large concatenation
        gate_h = gate_feat_i.unsqueeze(2) + gate_feat_j.unsqueeze(1) + gate_bias.view(1, 1, 1, -1)
        gate_h = F.relu(gate_h)
        edge_gate = torch.sigmoid(self.edge_gate[2](gate_h)).squeeze(-1) # (B, N, N)

        # Deconstruct edge_bias[0] (nn.Linear(2 * C, C))
        bias_fc1 = self.edge_bias[0]
        bias_w_i = bias_fc1.weight[:, :C]
        bias_w_j = bias_fc1.weight[:, C:]
        bias_bias = bias_fc1.bias
        
        bias_feat_i = F.linear(x, bias_w_i) # (B, N, C)
        bias_feat_j = F.linear(x, bias_w_j) # (B, N, C)
        # Broadcast add to shape (B, N, N, C) without large concatenation
        bias_h = bias_feat_i.unsqueeze(2) + bias_feat_j.unsqueeze(1) + bias_bias.view(1, 1, 1, -1)
        bias_h = F.relu(bias_h)
        edge_bias = self.edge_bias[2](bias_h).squeeze(-1) # (B, N, N)

        if adj is not None:
            edge_gate = edge_gate * adj

        scores = scores + edge_bias.unsqueeze(1)
        scores = scores * edge_gate.unsqueeze(1)

        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v) # (B, H, N, d_k)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_lin(out)
        out = self.norm(out + x) # Residual + Norm
        return F.relu(out)


class RelationEncoder(nn.Module):
    """
    Priority 1: Relation Encoder.
    Computes a pairwise relation tensor for every pair of nodes (i, j).
    Formula: R_ij = MLP([x_i, x_j, x_i - x_j])
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Input is [x_i, x_j, x_i - x_j] -> size is 3 * in_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x shape: (B, K, C)
        B, K, C = x.shape
        x_i = x.unsqueeze(2).expand(B, K, K, C)  # (B, K, K, C)
        x_j = x.unsqueeze(1).expand(B, K, K, C)  # (B, K, K, C)
        x_diff = x_i - x_j                      # (B, K, K, C)
        
        # Concatenate: shape (B, K, K, 3*C)
        feat = torch.cat([x_i, x_j, x_diff], dim=-1)
        
        # Map relation features using the MLP
        out = self.mlp(feat)                    # (B, K, K, out_dim)
        return out


class ArcMarginProduct(nn.Module):
    """
    Priority 2: ArcFace (Additive Angular Margin) classification head.
    Formula: L = -\log \frac{e^{s(\cos(\theta_y + m))}}{e^{s(\cos(\theta_y + m))} + \sum_{j \neq y} e^{s \cos(\theta_j)}}
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # L2 normalize both the inputs and weights to get cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # During validation/inference or if targets are not provided, return s * cos(theta)
        if label is None or not self.training:
            return cosine * self.s
            
        # Clamp cosine to prevent NaN gradient when cosine approaches 1.0 or -1.0
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # One-hot target label mapping
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply the angular margin shift to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class GraphMotifModule(nn.Module):
    """
    Research-grade Structured Graph Matching Module.
    Upgraded to use pairwise Relation Tensor matching and learnable AU Contrastive loss.
    """
    def __init__(self, num_classes, motifs_per_class, K, C, top_k=None, rank=4, clip_embedding_path=None, au_tau=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.K = K  
        self.C = C  
        self.top_k = top_k
        self.au_tau = au_tau
        
        # 1. Relation Motif Representation: (Classes, Motifs, K, K, Dim)
        self.motifs = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, K, C)) 
        nn.init.normal_(self.motifs, std=1.0 / math.sqrt(C))
        
        # Relation encoder mapping candidate nodes to relation representations
        self.relation_encoder = RelationEncoder(in_dim=C, out_dim=C)
        
        # 1.1 Learnable Visual AU Tokens (Replacing CLIP completely)
        self.au_tokens = nn.Parameter(torch.randn(num_classes, motifs_per_class, C))
        nn.init.orthogonal_(self.au_tokens) # Orthogonal initialization to disperse tokens
        
        # 2. Factorized Motif Topology: (Classes, Motifs, K, Rank)
        self.motif_low_rank = nn.Parameter(torch.randn(num_classes, motifs_per_class, K, rank))
        nn.init.xavier_uniform_(self.motif_low_rank)
        
        # 3. Learnable weights for Node vs Edge similarity
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # 4. Node importance weighting weights (Initialized to 0 to yield uniform softmax at first)
        self.node_importance = nn.Parameter(torch.zeros(1, 1, K))
        
        # 5. Stability parameters
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def compute_au_contrastive_loss(self, region_features, targets):
        """
        InfoNCE Action Unit Contrastive Loss.
        Pulls region_features close to target au_tokens (Positive) 
        and pushes them away from other class au_tokens (Negative).
        """
        if targets is None or region_features is None:
            return torch.tensor(0.0, device=self.motifs.device)
            
        B = region_features.shape[0]
        # Flatten candidate region nodes: (B, num_cands * 16, C)
        feat_flat = region_features.reshape(B, -1, self.C) 
        feat_norm = F.normalize(feat_flat, p=2, dim=-1) # (B, N, C)
        
        # Normalize AU tokens: (L, M, C)
        au_norm = F.normalize(self.au_tokens, p=2, dim=-1)
        
        # Compute cosine similarity: (B, N, L, M) -> max pool over motifs M: (B, N, L)
        sim = torch.einsum('bnc,lmc->bnlm', feat_norm, au_norm)
        sim_max, _ = sim.max(dim=-1) 
        
        # InfoNCE temperature scaling
        tau = self.au_tau
        sim_max = sim_max / tau
        
        loss = 0.0
        for i in range(B):
            label = targets[i]
            # Compute log_softmax across classes L: (N, L)
            log_prob = F.log_softmax(sim_max[i], dim=-1)
            # Add negative log probability of target label
            loss += -log_prob[:, label].mean()
            
        return loss / B

    def compute_diversity_loss(self):
        """
        Orthogonality constraint for motifs.
        L = || M M^T - I ||
        """
        m = self.motifs.view(self.num_classes, self.motifs_per_class, -1)
        m = F.normalize(m, dim=-1)
        sim = torch.matmul(m, m.transpose(1, 2))
        eye = torch.eye(self.motifs_per_class, device=m.device).unsqueeze(0)
        return torch.norm(sim - eye, p='fro', dim=(1, 2)).mean()

    def forward(self, region_features, adj=None, return_attention=False):
        """
        Args:
            region_features: (B, K, C)
            adj: (B, K, K) input graph adjacency
            
        Returns:
            logits: (B, num_classes)
            motif_scores: (B, num_classes, motifs_per_class)
            metadata: dict containing attention and activation maps
        """
        B, K, C = region_features.shape
        L, M = self.num_classes, self.motifs_per_class
        
        # 1. Project candidate regions to Relation Space and normalize
        R_cand = self.relation_encoder(region_features)  # (B, K, K, C)
        R_cand = F.normalize(R_cand, p=2, dim=-1)
        
        motifs = F.normalize(self.motifs, p=2, dim=-1)   # (L, M, K, K, C)
        
        # 2. Match candidate relation matrix against the motifs using einsum
        # Output shape: (B, L, M, K, K)
        edge_sim_raw = torch.einsum('bijk,lmijk->blmij', R_cand, motifs)
        
        # 3. Retrieve Node Similarity from the diagonal (i == j) of the relation similarity matrix
        node_sim = torch.diagonal(edge_sim_raw, dim1=-2, dim2=-1)  # (B, L, M, K)
        
        # Aggregation weighting computation
        tau = F.softplus(self.temperature)
        node_attn = F.softmax(node_sim / tau.clamp(min=1e-3), dim=-1)
        edge_weights = node_attn.unsqueeze(-2) * node_attn.unsqueeze(-1)
        edge_weights = edge_weights / edge_weights.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        edge_sim = (edge_sim_raw * edge_weights).sum(dim=(-1, -2))
        
        # 4. Topology matching using Low-Rank Motif Edges
        motif_adj = torch.matmul(self.motif_low_rank, self.motif_low_rank.transpose(-1, -2))
        motif_adj = F.softmax(motif_adj, dim=-1)
        
        topo_sim = 0
        if adj is not None:
            topo_sim = (motif_adj.unsqueeze(0) * edge_weights).sum(dim=(-1, -2))
            
        # 5. Combined Node and Edge Similarity
        s_node = node_sim
        s_struct = edge_sim + topo_sim
        
        node_sim_agg = torch.sum(node_attn * s_node, dim=-1) # (B, L, M)
        
        w_node = torch.sigmoid(self.alpha)
        w_edge = torch.sigmoid(self.beta)
        S = w_node * node_sim_agg + w_edge * s_struct
        
        # 6. Selection via logsumexp
        logits = torch.logsumexp(S / tau.clamp(min=1e-3), dim=-1)
        
        # 7. Entropy calculation
        entropy = -(node_attn * torch.log(node_attn + 1e-8)).sum(dim=-1).mean()
        self._latest_attn_entropy = entropy
        
        if return_attention:
            metadata = {
                "node_attention": node_attn,
                "motif_activations": S,
                "edge_sim_matrix": edge_sim_raw
            }
            return logits, S, metadata
        return logits, S


class MotifGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.motifs_per_class = config.get('motifs_per_class', 16)
        self.top_k = config.get('top_k', 6)
        self.temperature = config.get('motif_tau', 0.1) 
        
        self.backbone = MotifBackbone(feat_dim=self.feat_dim)
        
        pretrained_cnn_path = config.get('pretrained_cnn_path', "")
        if pretrained_cnn_path != "":
            self.backbone.load_pretrained_cnn(pretrained_cnn_path)
            
        # 4. Global Feature Branch & ArcFace Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # ArcFace parameters loaded from config
        arc_s = float(config.get('arcface_s', 20.0))
        arc_m = float(config.get('arcface_m', 0.35))
        self.arc_face = ArcMarginProduct(in_features=128, out_features=self.num_classes, s=arc_s, m=arc_m)
        
        # Learnable Region Proposal Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feat_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_dim, self.feat_dim),
            GraphAttentionLayer(self.feat_dim, self.feat_dim)
        ])
        
        self.offset_predictor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Tanh() 
        )
        self.offset_amplitude = float(config.get('offset_amplitude', 0.35))
        
        self.pos_embed = nn.Parameter(torch.randn(1, 16, self.feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.register_buffer('grid_adj', self._generate_4x4_grid_adj())
        
        au_tau = float(config.get('au_tau', 0.07))
        self.motif_module = GraphMotifModule(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            K=16,
            C=self.feat_dim,
            top_k=self.top_k,
            clip_embedding_path=None,
            au_tau=au_tau
        )
        
        self.logit_scale = nn.Parameter(torch.ones(1) * arc_s)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        self.cand_query = nn.Parameter(torch.randn(1, 1, self.num_classes))
        nn.init.xavier_uniform_(self.cand_query)

        motif_margin = float(config.get('motif_margin', 0.4)) if 'motif_margin' in config else float(config.get('training', {}).get('motif_margin', 0.4))
        self.motif_consistency_loss = MotifConsistencyLoss(
            num_classes=self.num_classes,
            motifs_per_class=self.motifs_per_class,
            tau=self.temperature,
            margin=motif_margin
        )

    def _extract_deformable_subgraphs(self, feat_map, H, W, node_feats):
        B, C_feat, _, _ = feat_map.shape
        num_cands = self.top_k
        
        # 1. Compute spatial heatmap representing region importance: shape (B, 1, H, W)
        heatmap = self.spatial_attention(feat_map)
        
        # 2. Flatten spatial layout and extract Top-K peak indices: shape (B, num_cands)
        flat_heatmap = heatmap.view(B, -1)
        topk_scores, topk_indices = torch.topk(flat_heatmap, k=num_cands, dim=-1)
        
        # 3. Gather candidate center features from node_feats using vectorized indexing
        # node_feats shape: (B, H * W, C_feat)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C_feat)
        center_feats = torch.gather(node_feats, dim=1, index=gather_indices) # (B, num_cands, C_feat)
        
        # 4. Predict deformable offsets using spatial and global features
        global_feat = feat_map.mean(dim=(2, 3)) # (B, C_feat)
        global_feat = global_feat.unsqueeze(1).expand(-1, num_cands, -1) # (B, num_cands, C_feat)
        combined_feats = torch.cat([center_feats, global_feat], dim=-1) # (B, num_cands, 2 * C_feat)
        
        offsets = self.offset_predictor(combined_feats) * self.offset_amplitude
        self._latest_offsets = offsets
        
        # 5. Local grid offset setup
        rel_y, rel_x = torch.meshgrid(
            torch.linspace(-1.5, 1.5, 4),
            torch.linspace(-1.5, 1.5, 4),
            indexing='ij'
        )
        rel_grid = torch.stack([rel_x, rel_y], dim=-1).to(feat_map.device) 
        rel_grid = rel_grid.view(1, 1, 16, 2) 
        
        # 6. Convert 1D topk_indices back to relative 2D coordinates in range [-1, 1]
        y_indices = torch.div(topk_indices, W, rounding_mode='trunc') # (B, num_cands)
        x_indices = topk_indices % W                                  # (B, num_cands)
        
        c_y = y_indices.float() / (H - 1) * 2.0 - 1.0                # (B, num_cands)
        c_x = x_indices.float() / (W - 1) * 2.0 - 1.0                # (B, num_cands)
        
        # Stack to form centers grid of shape (B, num_cands, 1, 2)
        centers_grid = torch.stack([c_x, c_y], dim=-1).unsqueeze(2)
        
        # Combine center, offsets and local window grid to construct final sampling grid
        sampling_grid = centers_grid + offsets.unsqueeze(2) + rel_grid * (1.0 / (W-1))
        sampling_grid = sampling_grid.view(B, num_cands * 16, 1, 2)
        
        # Perform grid sampling to extract subgraphs
        sampled_feats = F.grid_sample(feat_map, sampling_grid, align_corners=True)
        sampled_feats = sampled_feats.view(B, C_feat, num_cands, 16).permute(0, 2, 3, 1) 
        
        # Soft attention scaling to route gradients back to spatial_attention
        sampled_feats = sampled_feats * topk_scores.unsqueeze(-1).unsqueeze(-1)
        
        adj = self.grid_adj.unsqueeze(0).unsqueeze(0).expand(B, num_cands, -1, -1)
        
        # For debug and visualization, compute center coordinates of the first batch element
        centers_coords = []
        first_batch_indices = topk_indices[0].detach().cpu().numpy()
        for idx in first_batch_indices:
            centers_coords.append((int(idx // W), int(idx % W)))
            
        return sampled_feats, adj, centers_coords

    def forward(self, x, return_selection=False, targets=None):
        if targets is not None:
            self._latest_targets = targets
            
        # Handle TenCrop input: (B, 10, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            # Recursive call to handle all crops (expand targets to match B*T)
            if targets is not None:
                targets_expanded = targets.unsqueeze(1).expand(-1, T).reshape(-1)
                logits = self.forward(x, targets=targets_expanded)
            else:
                logits = self.forward(x) 
            # Average predictions across all 10 crops
            return logits.view(B, T, -1).mean(dim=1)

        B = x.shape[0]
        
        feat_map = self.backbone(x) # (B, C, H, W)
        _, _, H, W = feat_map.shape
        
        # 4. Global Branch prediction with ArcFace Head
        feat_global = self.global_features(self.global_pool(feat_map))
        logits_global = self.arc_face(feat_global, targets)
        
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
        self._latest_candidates = candidates
        num_cands = candidates.shape[1]
        
        # Advanced Motif Module Forward
        # 1. Prepare candidate subgraphs: (B*num_cands, 16, Dim)
        flat_cands = candidates.reshape(B * num_cands, 16, -1)
        if flat_cands.shape[-1] != self.feat_dim:
            flat_cands = self.proj_node(flat_cands)
        flat_cands = flat_cands + self.pos_embed
        
        # 2. Prepare candidate adjacencies: (B*num_cands, 16, 16)
        flat_adjs = cand_adjs.reshape(B * num_cands, 16, 16)
        
        # 3. Match against Learnable Motifs (Research Grade)
        logits_cand, motif_scores_cand, metadata = self.motif_module(flat_cands, adj=flat_adjs, return_attention=True)
        
        # 4. Aggregate across all candidate subgraphs
        logits_cand = logits_cand.view(B, num_cands, self.num_classes)
        
        # Candidate-level attention using learnable query
        cand_scores = (logits_cand * self.cand_query).sum(dim=-1) # (B, num_cands)
        cand_tau = 1.0
        attn_weights = F.softmax(cand_scores / cand_tau, dim=1).unsqueeze(-1) 
        
        logits_motif = torch.sum(logits_cand * attn_weights, dim=1)
        logits_motif = logits_motif * self.logit_scale 
        
        # Final combined logits
        logits = logits_motif + torch.sigmoid(self.alpha) * logits_global
        
        # Reshape for MotifConsistencyLoss (B, num_cands, num_classes * motifs_per_class)
        self._latest_scores = motif_scores_cand.view(B, num_cands, -1)
        # Relevance for Top-K visualization
        cand_relevance = cand_scores
        k = min(self.top_k, cand_relevance.size(1))
        if k > 0:
            _, top_k_idx = torch.topk(cand_relevance, k=k, dim=1)
        else:
            top_k_idx = torch.empty(B, 0, dtype=torch.long, device=cand_relevance.device)
        self._latest_top_k = top_k_idx
        self._latest_metadata = metadata
        
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
        self.training_progress = progress
        
    def get_current_prior_strength(self):
        return 0.0

    def _generate_4x4_grid_adj(self):
        adj = torch.zeros(16, 16)
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4:
                            n_idx = ni * 4 + nj
                            adj[idx, n_idx] = 1.0
        return adj

    def get_aux_losses(self):
        if not hasattr(self, '_latest_scores') or self._latest_scores is None:
            return {}
            
        # 1. Motif Diversity (Orthogonality)
        l_div = self.motif_module.compute_diversity_loss()
        
        # 2. Attention Entropy (Prevent collapse)
        l_ent = getattr(self.motif_module, '_latest_attn_entropy', 0.0)
        
        # 3. Offset Regularization
        l_off = torch.norm(getattr(self, '_latest_offsets', 0.0), p=2, dim=-1).mean()
        
        aux_dict = {
            "motif_diversity": l_div,
            "attn_entropy": l_ent,
            "offset_reg": l_off
        }
        
        # 4. Motif Consistency Loss
        if hasattr(self, '_latest_targets') and self._latest_targets is not None:
            progress = getattr(self, 'training_progress', 1.0)
            # Disable consistency loss during Phase 1 (Mixup active)
            if self.training and progress <= 0.06:
                l_motif_consist = torch.tensor(0.0, device=self._latest_scores.device)
            else:
                l_motif_consist = self.motif_consistency_loss(
                    self._latest_scores, 
                    self._latest_top_k, 
                    self._latest_targets
                )
            aux_dict["motif_consistency"] = l_motif_consist
            
        # 5. Visual-Driven AU Contrastive Loss (Replacing CLIP Grounding)
        if hasattr(self.motif_module, 'compute_au_contrastive_loss'):
            targets = getattr(self, '_latest_targets', None)
            candidates = getattr(self, '_latest_candidates', None)
            if targets is not None and candidates is not None:
                aux_dict["au_contrastive"] = self.motif_module.compute_au_contrastive_loss(candidates, targets)
            else:
                aux_dict["au_contrastive"] = torch.tensor(0.0, device=self.logit_scale.device)
            
        return aux_dict


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
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(N, 2)
        dist_spatial = torch.cdist(grid_coords.float(), grid_coords.float(), p=float('inf'))
        mask = (dist_spatial <= 1).float().to(feat_map.device)
        
        dist_feat = torch.cdist(nodes, nodes) / math.sqrt(C)
        sim = torch.exp(-dist_feat)
        
        adj = sim * mask.unsqueeze(0)
        return nodes_with_coords, adj


if __name__ == "__main__":
    config = {
        'feat_dim': 64,
        'num_classes': 7,
        'motifs_per_class': 16,
        'top_k': 4,
    }
    model = MotifGraphModel(config)
    
    # Test 4D (with targets)
    dummy_img_4d = torch.randn(2, 1, 48, 48)
    out_4d = model(dummy_img_4d, targets=torch.tensor([0, 3]))
    print(f"4D Output shape: {out_4d.shape}") # (2, 7)
    
    aux_losses = model.get_aux_losses()
    print("Auxiliary Losses:")
    for name, loss in aux_losses.items():
        print(f"  - {name}: {loss.item():.4f}" if isinstance(loss, torch.Tensor) else f"  - {name}: {loss}")
    
    # Test 5D (TenCrop)
    dummy_img_5d = torch.randn(2, 10, 1, 40, 40)
    out_5d = model(dummy_img_5d)
    print(f"5D Output shape: {out_5d.shape}") # (2, 7)