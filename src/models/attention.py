"""
Region Attention: Spatial and channel attention for FER
Purpose: Focus model on discriminative facial regions (mouth, eyes, eyebrows)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionAttention(nn.Module):
    """
    Generate spatial attention maps for K regions of interest.
    Applies softmax-normalized attention to extract region-specific features.
    """
    
    def __init__(self, in_channels=128, num_regions=3, reduction=16):
        """
        Args:
            in_channels: Number of input channels
            num_regions: Number of spatial regions to attend to (e.g., 3 for mouth/eyes/face)
            reduction: Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.num_regions = num_regions
        self.in_channels = in_channels
        
        # Generate K attention maps via 1×1 convolution
        # Each map focuses on different region
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, num_regions, kernel_size=1)
        )
        
        # Per-region feature refinement
        self.region_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - feature maps
        
        Returns:
            region_features: (B, num_regions, C) - attended region features
        """
        B, C, H, W = x.shape
        
        # Generate K attention maps: (B, K, H, W)
        attention_maps = self.attention_gen(x)  # (B, num_regions, H, W)
        
        # Normalize attention maps: softmax over spatial dimensions
        # Reshape for softmax: (B, K, H*W)
        attention_flat = attention_maps.view(B, self.num_regions, -1)
        attention_weights = F.softmax(attention_flat, dim=-1)  # Normalize over spatial locations
        
        # Extract region features: weighted pooling per region
        region_features = []
        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        
        for k in range(self.num_regions):
            # Weighted sum: (B, C, H*W) × (B, 1, H*W) = (B, C)
            weight_k = attention_weights[:, k:k+1, :]  # (B, 1, H*W)
            feat_k = (x_flat * weight_k).sum(dim=2)  # (B, C)
            region_features.append(feat_k)
        
        # Stack: (B, K, C)
        region_features = torch.stack(region_features, dim=1)
        
        return region_features, attention_maps


class SpatialAttentionMap(nn.Module):
    """
    Create predefined spatial attention for specific facial regions.
    Examples: mouth, eyes, eyebrows, face center
    """
    
    def __init__(self, region_type='mouth', image_size=48):
        """
        Args:
            region_type: 'mouth' | 'eyes' | 'eyebrows' | 'face'
            image_size: Input image resolution (e.g., 48×48)
        """
        super().__init__()
        self.region_type = region_type
        self.image_size = image_size
        
        # Create predefined attention masks
        mask = self._create_mask(image_size, region_type)
        self.register_buffer('mask', mask)
    
    @staticmethod
    def _create_mask(size, region_type):
        """Create Gaussian-like mask for region"""
        # (1, 1, size, size)
        y = torch.linspace(-1, 1, size).view(-1, 1).expand(size, size)
        x = torch.linspace(-1, 1, size).view(1, -1).expand(size, size)
        
        if region_type == 'mouth':
            # Lower part of face, center: y ∈ [0.3, 1.0], x ∈ [-0.6, 0.6]
            mask = torch.exp(-((y - 0.6)**2 / 0.15 + (x**2) / 0.4))
        
        elif region_type == 'eyes':
            # Upper part, wide: y ∈ [-1.0, -0.2], x ∈ [-1.0, 1.0]
            mask = torch.exp(-(((y + 0.6)**2 / 0.2 + (x**2) / 0.8)))
        
        elif region_type == 'eyebrows':
            # Top part, narrow: y ∈ [-1.0, -0.6]
            mask = torch.exp(-((y + 0.8)**2 / 0.12 + (x**2) / 1.0))
        
        else:  # 'face'
            # Whole face with soft edges
            mask = torch.exp(-((y**2 + x**2) / 2.0))
        
        # Normalize
        mask = mask / (mask.max() + 1e-8)
        
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
    
    def forward(self, x):
        """Apply spatial attention to features"""
        # x: (B, C, H, W)
        return x * self.mask


class ConfusionAwareAttention(nn.Module):
    """
    Learnable region weighting specialized for hard emotion pairs.
    Automatically learns to focus on discriminative regions for confusion classes.
    """
    
    def __init__(self, in_channels=128, num_classes=7):
        """
        Args:
            in_channels: Feature channels
            num_classes: Number of emotion classes
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Learn per-class region importance
        # Shape: (num_classes, num_regions)
        self.confusion_weights = nn.Parameter(
            torch.ones(num_classes, 3) / 3.0,  # 3 regions: mouth, eyes, face
            requires_grad=True
        )
        
        # Attention regularization
        self.register_buffer('confusion_pairs', torch.tensor([
            [3, 5],  # Fear ↔ Sad
            [5, 0],  # Sad ↔ Angry
            [0, 1],  # Angry ↔ Disgust
        ]))
    
    def forward(self, region_features, predicted_class):
        """
        Args:
            region_features: (B, num_regions, C) from RegionAttention
            predicted_class: (B,) - predicted emotion class
        
        Returns:
            weighted_features: (B, C) - fused with confusion awareness
        """
        B, num_regions, C = region_features.shape
        
        # Get class-specific attention weights
        # (B, num_regions)
        weights = self.confusion_weights[predicted_class]
        
        # Normalize weights
        weights = F.softmax(weights, dim=-1)  # (B, num_regions)
        
        # Weighted fusion: (B, C)
        weighted_feat = (region_features * weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_feat


class CombinedAttentionModule(nn.Module):
    """
    Complete attention module: Region Attention + Confusion-Aware Weighting
    """
    
    def __init__(self, in_channels=128, num_regions=3, num_classes=7):
        super().__init__()
        self.region_attention = RegionAttention(in_channels, num_regions)
        self.confusion_attention = ConfusionAwareAttention(in_channels, num_classes)
    
    def forward(self, x, predicted_class=None):
        """
        Args:
            x: (B, C, H, W) - input features
            predicted_class: (B,) - optional predicted class for confusion awareness
        
        Returns:
            attended_features: (B, C) - attended features
            attention_maps: (B, num_regions, H, W) - visualization
        """
        # Step 1: Generate region-wise attention
        region_features, attention_maps = self.region_attention(x)  # (B, K, C), (B, K, H, W)
        
        # Step 2: Apply confusion-aware weighting
        if predicted_class is not None:
            attended_feat = self.confusion_attention(region_features, predicted_class)
        else:
            # Default: equal weighting
            attended_feat = region_features.mean(dim=1)  # (B, C)
        
        return attended_feat, attention_maps
