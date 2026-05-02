"""
Facial Emotion Recognition (FER2013) - Advanced Model
Architecture: CNN Backbone + Learnable Region Attention + Graph Module + Motif Learning

Key Components:
1. Lightweight CNN Backbone - Feature extraction
2. Learnable Region Attention - Soft region extraction (K regions)
3. Graph Attention Module - Relational modeling between regions
4. Motif Module - Prototype-based pattern learning
5. Fusion & Classification - Combine all signals

Target: ~73% accuracy on FER2013
Author: AI Assistant
Date: May 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. VGG BACKBONE (Replaces CNN Backbone)
# ============================================================================

class VggBackbone(nn.Module):
    """
    VGG-based backbone for 48x48 grayscale images.
    Following standard VGG architecture with 2 conv per block.
    
    Architecture:
    - Block 1: 2 Conv (64) + Pool → 24x24
    - Block 2: 2 Conv (128) + Pool → 12x12
    - Block 3: 2 Conv (256) + Pool → 6x6
    - Block 4: 2 Conv (512) → 6x6 (NO POOL - keep spatial size)
    - Project to feat_dim
    
    Output: (B, feat_dim, 6, 6) feature map for region attention
    """
    def __init__(self, feat_dim=128, in_channels=1):
        super().__init__()
        self.feat_dim = feat_dim
        
        # Pool layer (shared)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ========== Block 1: (B, 1, 48, 48) -> (B, 64, 24, 24) ==========
        self.conv1a = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)
        
        # ========== Block 2: (B, 64, 24, 24) -> (B, 128, 12, 12) ==========
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)
        
        # ========== Block 3: (B, 128, 12, 12) -> (B, 256, 6, 6) ==========
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)
        
        # ========== Block 4: (B, 256, 6, 6) -> (B, 512, 6, 6) ==========
        # NO POOLING - keep 6x6 spatial resolution for region attention
        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)
        
        # Project to feat_dim
        self.feat_project = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 48, 48) - Grayscale image
        
        Returns:
            feat_map: (B, feat_dim, 6, 6) - Feature map
        """
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)  # 48 -> 24
        
        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)  # 24 -> 12
        
        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)  # 12 -> 6
        
        # Block 4 (NO POOL - maintain 6x6)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        # NO: x = self.pool(x)  ← Removed to keep 6x6
        
        # Project to feat_dim
        x = self.feat_project(x)  # (B, feat_dim, 6, 6)
        return x


# ============================================================================
# 1. LIGHTWEIGHT CNN BACKBONE (Legacy - kept for compatibility)
# ============================================================================

class CNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for 48x48 grayscale images.
    
    Architecture:
    - Input: (B, 1, 48, 48)
    - Conv blocks with BatchNorm + ReLU + MaxPool
    - Output: (B, feat_dim, H_out, W_out)
    
    Design choice: Residual-like connections for better gradient flow
    """
    def __init__(self, feat_dim=128, in_channels=1):
        super().__init__()
        self.feat_dim = feat_dim
        
        # Block 1: (B, 1, 48, 48) -> (B, 64, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 2: (B, 64, 24, 24) -> (B, 128, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 3: (B, 128, 12, 12) -> (B, feat_dim, 6, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, 48, 48) - Grayscale image
        
        Returns:
            feat_map: (B, feat_dim, 6, 6) - Feature map
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# ============================================================================
# 2. LEARNABLE REGION ATTENTION MODULE
# ============================================================================

class RegionAttentionModule(nn.Module):
    """
    Learnable region attention - Soft extraction of K discriminative regions.
    
    Method:
    - Use 1x1 Conv to generate K attention maps (K = num_regions)
    - Apply softmax over spatial dimension for each region
    - Weighted pooling: region_feat = sum(attention_map * feat_map) / (sum(attention_map))
    
    Intuition:
    - Each region learns to focus on different parts of face
    - E.g., Region 1: mouth, Region 2: eyes, Region 3: whole face
    - Soft extraction allows gradient flow through attention
    
    Design choice:
    - Use spatial softmax (not channel-wise) to focus on WHERE
    - K regions per emotion helps with multiple aspects
    """
    def __init__(self, feat_dim=128, num_regions=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions
        
        # Generate K attention maps using 1x1 convolution
        self.attention_generator = nn.Conv2d(feat_dim, num_regions, kernel_size=1)
        
        # Optional: Refine attention with learned weights
        self.region_refiner = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim)
        )
    
    def forward(self, feat_map):
        """
        Args:
            feat_map: (B, feat_dim, H, W) - Feature map from backbone
        
        Returns:
            region_features: (B, num_regions, feat_dim) - Extracted region features
            attention_maps: (B, num_regions, H, W) - Attention maps (for visualization/regularization)
        """
        B, C, H, W = feat_map.shape
        
        # Generate attention maps: (B, num_regions, H, W)
        attention_logits = self.attention_generator(feat_map)  # (B, K, H, W)
        
        # Spatial softmax - normalize over spatial dimension for each region
        # Reshape for softmax: (B, K, H*W)
        attention_logits_flat = attention_logits.view(B, self.num_regions, -1)
        attention_maps_flat = F.softmax(attention_logits_flat, dim=-1)  # (B, K, H*W)
        attention_maps = attention_maps_flat.view(B, self.num_regions, H, W)
        
        # Weighted pooling: extract region features
        # feat_map_flat: (B, C, H*W)
        feat_map_flat = feat_map.view(B, C, -1)  # (B, C, H*W)
        
        # For each region: region_feat = sum_hw(attention[k,h,w] * feat[c,h,w])
        # (B, K, H*W) x (B, C, H*W) -> (B, K, C)
        region_features = torch.bmm(
            attention_maps_flat,  # (B, K, H*W)
            feat_map_flat.transpose(1, 2)  # (B, H*W, C)
        )  # -> (B, K, C)
        
        # Optional: Refine region features
        region_features = region_features + self.region_refiner(region_features)
        
        return region_features, attention_maps


# ============================================================================
# 3. MOTIF (PROTOTYPE) MODULE
# ============================================================================

class MotifModule(nn.Module):
    """
    Learnable motif (prototype) learning for emotion-specific patterns.
    
    Method:
    - Define learnable prototypes: (num_emotions, feat_dim)
    - For each region, compute similarity to all emotion prototypes
    - Output: similarity scores indicate which emotion patterns present
    
    Intuition:
    - Prototypes capture emotion-specific facial patterns
    - E.g., "fear prototype" learns universal fear expression features
    - Similarity scores act as "soft classification" per region
    - Helps model learn interpretable emotion patterns
    
    Design choice:
    - Cosine similarity (normalized) for stability
    - Learnable temperature parameter for controlling softness
    """
    def __init__(self, feat_dim=128, num_emotions=7, num_regions=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_emotions = num_emotions
        self.num_regions = num_regions
        
        # Learnable emotion prototypes
        self.prototypes = nn.Parameter(torch.randn(num_emotions, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Temperature for controlling softness of similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, region_features):
        """
        Args:
            region_features: (B, num_regions, feat_dim) - Region features from attention
        
        Returns:
            similarity_scores: (B, num_regions, num_emotions) - Similarity to emotion prototypes
            motif_features: (B, num_regions*num_emotions) - Flattened similarities (for classification)
        """
        B, K, C = region_features.shape
        
        # Normalize region features and prototypes for cosine similarity
        region_features_norm = F.normalize(region_features, dim=-1)  # (B, K, C)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)  # (num_emotions, C)
        
        # Compute cosine similarity: (B, K, C) x (C, num_emotions) -> (B, K, num_emotions)
        similarity_scores = torch.bmm(
            region_features_norm,  # (B, K, C)
            prototypes_norm.t().unsqueeze(0).expand(B, -1, -1)  # (B, C, num_emotions)
        )  # -> (B, K, num_emotions)
        
        # Scale by temperature (higher temp = softer similarities)
        similarity_scores = similarity_scores / (self.temperature + 1e-8)
        
        # Flatten for classification use
        motif_features = similarity_scores.reshape(B, -1)  # (B, K*num_emotions)
        
        return similarity_scores, motif_features


# ============================================================================
# 4. GRAPH ATTENTION MODULE
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer for modeling relationships between regions.
    
    Method:
    - Treat each region as a node in a fully connected graph
    - Learn edge attention weights using region features
    - Update node features by weighted aggregation of neighbors
    
    Intuition:
    - Regions are not independent - they relate to each other
    - E.g., mouth smile correlates with eye crinkle (happy emotion)
    - Graph attention learns these dependencies
    
    Design choice:
    - Multi-head attention for capturing different relationship types
    - Self-loops to maintain node information
    """
    def __init__(self, feat_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, node_features):
        """
        Args:
            node_features: (B, num_regions, feat_dim) - Region features
        
        Returns:
            updated_features: (B, num_regions, feat_dim) - Updated region features after graph interaction
        """
        B, N, C = node_features.shape
        
        # Project to Q, K, V
        Q = self.query(node_features)  # (B, N, C)
        K = self.key(node_features)     # (B, N, C)
        V = self.value(node_features)   # (B, N, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        scores = self.leaky_relu(scores)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, H, N, d)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous()  # (B, N, H, d)
        attended_values = attended_values.view(B, N, C)  # (B, N, C)
        
        # Output projection and residual connection
        output = self.out_proj(attended_values)
        output = output + node_features  # Residual connection
        
        return output


class GraphModule(nn.Module):
    """Multi-layer graph attention for relational modeling"""
    def __init__(self, feat_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(feat_dim, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, region_features):
        """
        Args:
            region_features: (B, num_regions, feat_dim)
        
        Returns:
            graph_features: (B, num_regions, feat_dim) - Updated after graph reasoning
        """
        x = region_features
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

class AttentionDiversityLoss(nn.Module):
    """
    Encourage different regions to focus on different parts of the face.
    
    Intuition: If all K regions focus on same area, they're redundant.
    Loss: Minimize pairwise similarity between attention maps.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_maps):
        """
        Args:
            attention_maps: (B, num_regions, H, W) - Attention maps from region module
        
        Returns:
            loss: Scalar - diversity penalty
        """
        B, K, H, W = attention_maps.shape
        
        # Flatten spatial dimensions: (B, K, H*W)
        attention_flat = attention_maps.view(B, K, -1)
        
        # Compute pairwise cosine similarity between regions
        attention_norm = F.normalize(attention_flat, dim=-1)  # (B, K, H*W)
        
        # (B, K, H*W) x (B, H*W, K) -> (B, K, K)
        similarity = torch.bmm(attention_norm, attention_norm.transpose(1, 2))
        
        # We want off-diagonal elements to be small (different regions)
        # Create mask for off-diagonal
        mask = (1 - torch.eye(K, device=attention_maps.device)).unsqueeze(0)
        
        # Loss = mean of squared off-diagonal similarities
        diversity_loss = (similarity * mask).pow(2).mean()
        
        return diversity_loss


class AttentionSparsityLoss(nn.Module):
    """
    Encourage attention maps to be sparse (concentrated on small regions).
    
    Intuition: Attention should focus, not be diffuse everywhere.
    Loss: Entropy regularization - minimize entropy of attention distribution.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_maps):
        """
        Args:
            attention_maps: (B, num_regions, H, W) - Attention maps
        
        Returns:
            loss: Scalar - sparsity penalty
        """
        B, K, H, W = attention_maps.shape
        
        # Flatten: (B, K, H*W)
        attention_flat = attention_maps.view(B, K, -1)
        
        # Entropy = -sum(p * log(p))
        # Lower entropy = more concentrated
        entropy = -(attention_flat * torch.log(attention_flat + 1e-8)).sum(dim=-1).mean()
        
        # Loss: we want LOW entropy, so return entropy as loss
        sparsity_loss = entropy
        
        return sparsity_loss


# ============================================================================
# 6. FULL MODEL
# ============================================================================

class FERAdvancedModel(nn.Module):
    """
    Complete Facial Emotion Recognition model with:
    - VGG Backbone (improved over CNN)
    - Learnable Region Attention
    - Graph Module for relational reasoning
    - Motif (Prototype) learning
    - Classification head
    
    Architecture flow:
    Input (48x48) 
      -> VGG Backbone -> feat_map (B, 128, 6, 6) - 36 spatial regions
      -> Region Attention -> regions (B, 3, 128)
      -> Graph Module -> updated_regions (B, 3, 128)
      -> Motif Module -> emotion_scores (B, 3, 7)
      -> Classifier -> logits (B, 7)
    """
    
    def __init__(self, 
                 feat_dim=128, 
                 num_emotions=7, 
                 num_regions=3,
                 num_graph_layers=2,
                 num_heads=4,
                 dropout=0.3,
                 use_vgg=True):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_emotions = num_emotions
        self.num_regions = num_regions
        
        # Components - Use VGG by default
        if use_vgg:
            self.backbone = VggBackbone(feat_dim=feat_dim, in_channels=1)
        else:
            self.backbone = CNNBackbone(feat_dim=feat_dim, in_channels=1)
        
        self.region_attention = RegionAttentionModule(feat_dim=feat_dim, num_regions=num_regions)
        self.graph_module = GraphModule(feat_dim=feat_dim, num_layers=num_graph_layers, num_heads=num_heads)
        self.motif_module = MotifModule(feat_dim=feat_dim, num_emotions=num_emotions, num_regions=num_regions)
        
        # Classifier
        # Input: (B, num_regions, feat_dim) from graph + (B, num_regions*num_emotions) from motif
        combined_dim = feat_dim + num_regions * num_emotions
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
        
        # Auxiliary losses
        self.diversity_loss = AttentionDiversityLoss()
        self.sparsity_loss = AttentionSparsityLoss()
    
    def forward(self, x, return_auxiliary=False):
        """
        Args:
            x: (B, 1, 48, 48) - Grayscale image
            return_auxiliary: If True, return attention maps and motif features for analysis
        
        Returns:
            logits: (B, num_emotions) - Emotion class logits
            auxiliary: dict with attention_maps, motif_features (if return_auxiliary=True)
        """
        # Step 1: Backbone feature extraction
        feat_map = self.backbone(x)  # (B, feat_dim, H, W)
        
        # Step 2: Learnable region attention
        region_features, attention_maps = self.region_attention(feat_map)  # (B, K, C), (B, K, H, W)
        
        # Cache attention maps for get_landmark_outputs() (trainer.py compatibility)
        self._last_attention_maps = attention_maps
        
        # Step 3: Graph module for relational reasoning
        graph_features = self.graph_module(region_features)  # (B, K, C)
        
        # Step 4: Motif module - emotion prototype matching
        motif_scores, motif_features = self.motif_module(graph_features)  # (B, K, E), (B, K*E)
        
        # Step 5: Fusion - combine graph features and motif features
        # Average graph features across regions: (B, K, C) -> (B, C)
        graph_pooled = graph_features.mean(dim=1)  # (B, C)
        
        # Concatenate: (B, C + K*E)
        combined_features = torch.cat([graph_pooled, motif_features], dim=1)
        
        # Step 6: Classification
        logits = self.classifier(combined_features)  # (B, num_emotions)
        
        # Compute auxiliary losses
        auxiliary = None
        if return_auxiliary:
            auxiliary = {
                'attention_maps': attention_maps,
                'motif_scores': motif_scores,
                'region_features': region_features,
                'graph_features': graph_features,
                'diversity_loss': self.diversity_loss(attention_maps),
                'sparsity_loss': self.sparsity_loss(attention_maps),
            }
        
        return logits, auxiliary
    
    def get_landmark_outputs(self):
        """
        Get last attention maps (for compatibility with trainer.py)
        Returns: (attention_maps, None) tuple
        
        Note: This requires a forward pass first to cache the attention maps
        """
        if hasattr(self, '_last_attention_maps'):
            return self._last_attention_maps, None
        else:
            return None, None
    
    def get_auxiliary_losses(self, attention_maps):
        """Compute auxiliary regularization losses"""
        diversity_loss = self.diversity_loss(attention_maps)
        sparsity_loss = self.sparsity_loss(attention_maps)
        return {
            'diversity': diversity_loss,
            'sparsity': sparsity_loss,
        }


# ============================================================================
# 7. TEST & EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FER Advanced Model - Test")
    print("=" * 70)
    
    # Initialize model
    model = FERAdvancedModel(
        feat_dim=128,
        num_emotions=7,
        num_regions=3,
        num_graph_layers=2,
        num_heads=4,
        dropout=0.3
    )
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 48, 48)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        logits, auxiliary = model(x, return_auxiliary=True)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits sample (first batch): {logits[0].cpu().numpy()}")
    
    # Auxiliary outputs
    print(f"\n--- Auxiliary Outputs ---")
    for key, val in auxiliary.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}: {val.shape if hasattr(val, 'shape') else val}")
        else:
            print(f"{key}: {val:.4f}")
    
    print("\n✓ Model test passed!")
    print("=" * 70)
