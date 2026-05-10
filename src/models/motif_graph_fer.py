import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class LearnableStem(nn.Module):
    """
    Projects 1-channel grayscale to 3-channel RGB space dynamically.
    """
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.proj(x)

class ConvNeXtBackbone(nn.Module):
    def __init__(self, backbone_name='convnextv2_tiny', pretrained=True):
        super().__init__()
        self.model = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        # Using feature_info to dynamically get the last channel dimension
        self.out_channels = self.model.feature_info[-1]['num_chs']

    def forward(self, x):
        features = self.model(x)
        return features[-1] # Return the deepest feature map

class SemanticTokenExtractor(nn.Module):
    """
    Extracts K learnable semantic tokens from the feature map using Cross Attention.
    (DETR / Slot Attention style)
    """
    def __init__(self, in_dim=768, num_tokens=16, d_model=128, nhead=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        
        self.proj_v = nn.Conv2d(in_dim, d_model, 1)
        self.proj_k = nn.Conv2d(in_dim, d_model, 1)
        
        self.query_embed = nn.Parameter(torch.randn(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        v = self.proj_v(x).view(B, self.d_model, -1).transpose(1, 2) # (B, H*W, d_model)
        k = self.proj_k(x).view(B, self.d_model, -1).transpose(1, 2)
        
        q = self.query_embed.expand(B, -1, -1) # (B, K, d_model)
        
        # Cross Attention
        attn_out, attn_weights = self.cross_attn(q, k, v)
        q = self.norm1(q + attn_out)
        
        # FFN
        q = self.norm2(q + self.ffn(q))
        
        return q, attn_weights # q: (B, K, d_model)

class RelativePositionalGAT(nn.Module):
    """
    Graph Attention with Relative Positional Encoding for stable topology learning.
    """
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.k_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.out_lin = nn.Linear(out_dim, out_dim)
        
        # Relative positional encoding projection
        self.rel_pos_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, heads)
        )
        
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )
        
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, coords=None):
        B, N, _ = x.shape
        q = self.q_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, H, N, N)
        
        if coords is not None:
            # coords: (B, N, 2) -> relative diffs (B, N, N, 2)
            diffs = coords.unsqueeze(2) - coords.unsqueeze(1) 
            rel_bias = self.rel_pos_proj(diffs) # (B, N, N, heads)
            rel_bias = rel_bias.permute(0, 3, 1, 2) # (B, heads, N, N)
            scores = scores + rel_bias
            
        x_i = x.unsqueeze(2).expand(B, N, N, -1)
        x_j = x.unsqueeze(1).expand(B, N, N, -1)
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        edge_gate = torch.sigmoid(self.edge_gate(edge_feat)).squeeze(-1) # (B, N, N)
        
        scores = scores * edge_gate.unsqueeze(1)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(x + self.out_lin(out))
        return F.relu(out)

class HMPB(nn.Module):
    """
    Hierarchical Motif Prototype Bank with Inter-class Repulsion.
    """
    def __init__(self, num_classes=7, motifs_per_class=16, d_model=128):
        super().__init__()
        self.num_classes = num_classes
        self.motifs_per_class = motifs_per_class
        self.d_model = d_model
        
        self.prototypes = nn.Parameter(torch.randn(num_classes, motifs_per_class, d_model))
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        
        self.tau = nn.Parameter(torch.ones(1) * 0.1)

    def compute_repulsion_loss(self, margin=0.5):
        # L_rep = sum_{i != j} max(0, P_i^T P_j - margin)
        # We want to push inter-class prototypes apart.
        p = self.prototypes.view(self.num_classes * self.motifs_per_class, -1)
        p = F.normalize(p, dim=-1)
        sim = torch.matmul(p, p.t())
        
        mask = torch.ones_like(sim)
        for c in range(self.num_classes):
            start = c * self.motifs_per_class
            end = (c+1) * self.motifs_per_class
            mask[start:end, start:end] = 0 # ignore intra-class similarity
            
        rep_loss = F.relu(sim - (1 - margin)) * mask
        return rep_loss.mean()

    def forward(self, x):
        # x: Graph features (B, K, d_model)
        B, K, D = x.shape
        
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1) # (L, M, D)
        
        # sim: (B, K, L, M)
        sim = torch.einsum('bkd,lmd->bklm', x_norm, p_norm)
        
        # Max-pool over tokens to get motif matching score
        motif_scores, _ = sim.max(dim=1) # (B, L, M)
        
        tau = F.softplus(self.tau) + 1e-4
        logits = torch.logsumexp(motif_scores / tau, dim=-1) # (B, L)
        
        return logits, motif_scores, sim

class ConfidenceFusion(nn.Module):
    """
    Dynamic Reasoning Confidence Fusion.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, logits_cnn, logits_graph):
        p_cnn = F.softmax(logits_cnn, dim=-1)
        p_graph = F.softmax(logits_graph, dim=-1)
        
        ent_cnn = -(p_cnn * torch.log(p_cnn + 1e-8)).sum(dim=-1, keepdim=True)
        ent_graph = -(p_graph * torch.log(p_graph + 1e-8)).sum(dim=-1, keepdim=True)
        
        ent_features = torch.cat([ent_cnn, ent_graph], dim=-1)
        gate = torch.sigmoid(self.mlp(ent_features)) # (B, 1)
        
        fused = gate * logits_graph + (1 - gate) * logits_cnn
        return fused, gate

class MotifGraphModel(nn.Module):
    # Aliased to EMOGNP for backward compatibility with config loader
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.get('feat_dim', 128)
        self.num_classes = config.get('num_classes', 7)
        self.num_tokens = config.get('num_tokens', 16)
        self.motifs_per_class = config.get('motifs_per_class', 16)
        backbone_name = config.get('backbone_name', 'convnextv2_tiny')
        
        self.stem = LearnableStem(in_channels=config.get('in_channels', 1), out_channels=3)
        self.backbone = ConvNeXtBackbone(backbone_name=backbone_name, pretrained=True)
        
        cnn_out_dim = self.backbone.out_channels
        self.token_extractor = SemanticTokenExtractor(in_dim=cnn_out_dim, num_tokens=self.num_tokens, d_model=self.feat_dim)
        
        # Learnable virtual coordinates for tokens (for geometric priors)
        self.token_coords = nn.Parameter(torch.randn(1, self.num_tokens, 2))
        
        self.gnn = nn.Sequential(
            RelativePositionalGAT(self.feat_dim, self.feat_dim),
            RelativePositionalGAT(self.feat_dim, self.feat_dim)
        )
        
        self.hmpb = HMPB(self.num_classes, self.motifs_per_class, self.feat_dim)
        
        # Global Branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(256, self.num_classes)
        )
        
        self.fusion = ConfidenceFusion()

    def forward(self, x, targets=None, return_selection=False):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            logits = self.forward(x)
            return logits.view(B, T, -1).mean(dim=1)
            
        x_rgb = self.stem(x)
        feat_map = self.backbone(x_rgb) # (B, C, H, W)
        
        # Global CNN branch
        logits_global = self.global_fc(self.global_pool(feat_map))
        
        # Semantic Tokens
        tokens, attn_maps = self.token_extractor(feat_map) # tokens: (B, K, feat_dim)
        
        # Graph branch
        coords = self.token_coords.expand(x.shape[0], -1, -1)
        for layer in self.gnn:
            tokens = layer(tokens, coords)
            
        # Motif Matching
        logits_graph, motif_scores, sim_matrix = self.hmpb(tokens)
        
        # Adaptive Confidence Fusion
        logits, gate = self.fusion(logits_global, logits_graph)
        
        self._latest_rep_loss = self.hmpb.compute_repulsion_loss()
        self._latest_features = F.normalize(self.global_pool(feat_map).flatten(1), dim=-1) # for SupCon
        
        if return_selection:
            return logits, None, None, motif_scores
            
        return logits
        
    def get_aux_losses(self):
        return {
            "repulsion_loss": getattr(self, '_latest_rep_loss', 0.0)
        }

if __name__ == "__main__":
    config = {
        'feat_dim': 128,
        'num_classes': 7,
        'num_tokens': 16,
        'motifs_per_class': 16,
        'backbone_name': 'convnextv2_tiny'
    }
    model = MotifGraphModel(config)
    dummy_img = torch.randn(2, 1, 48, 48)
    out = model(dummy_img)
    print(f"Output shape: {out.shape}")