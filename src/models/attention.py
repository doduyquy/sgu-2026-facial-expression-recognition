"""
Attention Mechanisms for FER
- Region Attention: Focus on mouth/eyes/eyebrows
- Confusion Pair Attention: Weight hard emotion pairs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Spatial Attention - focus on specific face regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x_att: (B, C, H, W) - spatially weighted
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(cat))
        return x * att


class RegionAttention(nn.Module):
    """
    Focus network on specific face regions:
    - Mouth: distinguish fear/sad/disgust/anger
    - Eyes: distinguish fear/surprise/sad
    - Eyebrows: distinguish anger/fear/sad
    
    Face regions (48x48):
    - Top region (eyes): rows 8-20
    - Mid region (nose/cheeks): rows 18-32  
    - Bottom region (mouth): rows 28-42
    """
    def __init__(self, feat_dim=128, num_regions=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_regions = num_regions
        
        # Spatial attention per region
        self.mouth_attention = SpatialAttention(kernel_size=5)  # Fine details
        self.eye_attention = SpatialAttention(kernel_size=5)
        self.eyebrow_attention = SpatialAttention(kernel_size=5)
        
        # Channel attention per region (learn which channels matter)
        self.mouth_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 16, feat_dim, 1),
            nn.Sigmoid()
        )
        self.eye_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 16, feat_dim, 1),
            nn.Sigmoid()
        )
        self.eyebrow_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 16, feat_dim, 1),
            nn.Sigmoid()
        )
        
        # Learn region weights
        self.region_weights = nn.Parameter(torch.ones(3) / 3.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - feature map from backbone (e.g., 128x6x6)
        Returns:
            x_enhanced: (B, C, H, W) - region-weighted features
        """
        B, C, H, W = x.shape
        
        # Define region boundaries (relative to 6x6 feature map from 48x48 input)
        # 48x48 input -> 6x6 feature map (8x downsampling)
        # Eye region: ~rows 8-20 → ~rows 1-2.5 in 6x6
        # Mouth region: ~rows 28-42 → ~rows 3.5-5 in 6x6
        
        # For simplicity with 6x6 feature map:
        # Top third (eyes): rows 0-2
        # Mid third (nose): rows 2-4
        # Bottom third (mouth): rows 4-6
        
        h_split = H // 3
        
        # Extract regions
        eye_region = x[:, :, :h_split, :]  # Top third
        mid_region = x[:, :, h_split:2*h_split, :]  # Middle third
        mouth_region = x[:, :, 2*h_split:, :]  # Bottom third
        
        # Apply spatial + channel attention per region
        eye_att = eye_region * self.eye_se(eye_region)
        eye_att = self.eye_attention(eye_att)
        
        mid_att = mid_region  # Middle region less critical for emotion
        
        mouth_att = mouth_region * self.mouth_se(mouth_region)
        mouth_att = self.mouth_attention(mouth_att)
        
        # Combine regions with learned weights
        weights = F.softmax(self.region_weights, dim=0)
        
        # Reconstruct feature map with attended regions
        x_enhanced = x.clone()
        x_enhanced[:, :, :h_split, :] = eye_att * weights[0]
        x_enhanced[:, :, h_split:2*h_split, :] = mid_att * weights[1]
        x_enhanced[:, :, 2*h_split:, :] = mouth_att * weights[2]
        
        # Normalize
        x_enhanced = x_enhanced / (weights.sum() + 1e-6)
        
        return x_enhanced


class ConfusionAwareAttention(nn.Module):
    """
    Attention that weights hard confusion pairs higher.
    Pairs: (fear, sad), (sad, anger), (anger, disgust)
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        
        # Define confusion pairs and their weights
        # Emotion indices: 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
        self.register_buffer('confusion_matrix', torch.zeros(num_classes, num_classes))
        
        # High confusion pairs (bidirectional)
        confusion_pairs = [
            (2, 4, 2.0),  # fear <-> sad: highest weight
            (4, 0, 1.8),  # sad <-> anger
            (0, 1, 1.6),  # anger <-> disgust
            (0, 2, 1.7),  # anger <-> fear
            (1, 2, 1.5),  # disgust <-> fear
            (4, 6, 1.4),  # sad <-> neutral
            (2, 6, 1.3),  # fear <-> neutral
        ]
        
        for i, j, weight in confusion_pairs:
            self.confusion_matrix[i, j] = weight
            self.confusion_matrix[j, i] = weight
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,) - ground truth labels
        Returns:
            attention_weights: (B,) - per-sample weight for confusion pairs
        """
        B = logits.shape[0]
        
        # Get predicted class
        preds = torch.argmax(logits, dim=1)
        
        # Check if prediction is a hard confusion pair
        weights = torch.ones(B, device=logits.device)
        
        for b in range(B):
            pred_c = preds[b].item()
            true_c = labels[b].item()
            
            confusion_weight = self.confusion_matrix[true_c, pred_c].item()
            if confusion_weight > 0:
                # This is a hard pair - weight it higher
                weights[b] = confusion_weight
        
        return weights


if __name__ == "__main__":
    # Test RegionAttention
    print("Testing RegionAttention...")
    region_att = RegionAttention(feat_dim=128, num_regions=3)
    x = torch.randn(2, 128, 6, 6)
    y = region_att(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"
    print("✓ RegionAttention passed!")
    
    # Test ConfusionAwareAttention
    print("\nTesting ConfusionAwareAttention...")
    conf_att = ConfusionAwareAttention(num_classes=7)
    logits = torch.randn(4, 7)
    labels = torch.tensor([2, 4, 0, 1])  # fear, sad, anger, disgust
    weights = conf_att(logits, labels)
    print(f"Attention weights: {weights}")
    print("✓ ConfusionAwareAttention passed!")
