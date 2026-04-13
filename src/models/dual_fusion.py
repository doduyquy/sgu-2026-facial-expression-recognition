import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import VGGFusionSpatialCNN
from .resnet import ResNet50

class ResNet50SpatialCNN(nn.Module):
    """
    Wrapper for ResNet50 that extracts 3x3 spatial features (9 tokens).
    Outputs: [B, 9, 1024]
    """
    def __init__(self, config, channels=1):
        super().__init__()
        self.resnet = ResNet50(config, channels)
        # We need Stage 4 features (6x6) and pool them to 3x3
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # Manually run the resnet until layer4 to get spatial features
        x = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.pool(x)

        x = self.resnet.layer2(x) # Architecture starts from layer2
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) # [B, 1024, 6, 6]
        
        x = self.pool(x)           # [B, 1024, 3, 3]
        x = torch.flatten(x, 2)    # [B, 1024, 9]
        x = x.transpose(1, 2)     # [B, 9, 1024]
        return x

class LocalAttentionModule(nn.Module):
    """
    Local Attention: Dense -> Conv1D -> Dense
    Improved implementation with proper transpositions.
    """
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size//2)
        self.dense2 = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [B, 9, 512]
        # 1. Dense (feature-wise reflection)
        out = self.gelu(self.dense1(x))
        
        # 2. Conv1D (local spatial/context correlation)
        out = out.transpose(1, 2)      # [B, 512, 9]
        out = self.gelu(self.conv1d(out))
        out = out.transpose(1, 2)      # [B, 9, 512]
        
        # 3. Dense (final refinement)
        out = self.dense2(out)
        return out

class HybridAttentionFusionBlock(nn.Module):
    def __init__(self, vgg_dim=512, res_dim=1024, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Projections to align dimensions
        self.proj_vgg = nn.Linear(vgg_dim, embed_dim)
        self.proj_res = nn.Linear(res_dim, embed_dim)
        
        # Concat + MLP path (The "Concat features" -> "MLP" in diagram)
        self.mlp_fusion = nn.Sequential(
            nn.Linear(vgg_dim + res_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Attention branches
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.local_attn = LocalAttentionModule(embed_dim, kernel_size=3)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vgg_feat, res_feat):
        # vgg_feat: [B, 9, 512], res_feat: [B, 9, 1024]
        
        # 1. Concat + MLP path
        concat_feat = torch.cat([vgg_feat, res_feat], dim=-1) # [B, 9, 1536]
        m_feat = self.mlp_fusion(concat_feat)                 # [B, 9, 512] (MLP in diagram)
        
        # 2. Key/Query/Value logic from diagram
        # Key: VGG, Query: ResNet
        k = self.proj_vgg(vgg_feat) # [B, 9, 512]
        q = self.proj_res(res_feat) # [B, 9, 512]
        
        # Value: Mean of [K, Q, M] as interpreted from the diagram's "Mean" box
        v = (k + q + m_feat) / 3.0
        
        # 3. Self Attention (Q, K, V)
        # Diagram shows "Self attention" taking filtered signals
        attn_out, _ = self.self_attn(q, k, v)
        
        # 4. Local Attention Branch
        l_attn = self.local_attn(attn_out)
        
        # 5. Residual connection (Residual add in diagram)
        out = self.norm1(attn_out + l_attn)
        out = self.norm2(out + v) 
        
        return out

class VGGResNetAttentionFusion(nn.Module):
    """
    Hybrid Model: VGG + ResNet50 + Multi-Attention Fusion.
    Supports Transfer Learning: load pretrained backbone weights and freeze/unfreeze.
    """
    def __init__(self, config, channels=1):
        super().__init__()
        self.vgg_backbone = VGGFusionSpatialCNN(config, channels)
        self.res_backbone = ResNet50SpatialCNN(config, channels)
        
        self.embed_dim = config['model'].get('embed_dim', 512)
        self.num_heads = config['model'].get('num_heads', 8)
        self.dropout = config['model'].get('transformer_dropout', 0.1)
        
        # Transfer Learning config
        model_cfg = config.get('model', {})
        self.freeze_epochs = model_cfg.get('freeze_backbone_epochs', 0)
        self.is_frozen = False
        
        self.fusion_block = HybridAttentionFusionBlock(
            vgg_dim=512, 
            res_dim=1024, 
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            dropout=self.dropout
        )
        
        # Final MLP head (7 emotions)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 9, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, config['data']['num_classes'])
        )

    def load_pretrained_backbones(self, vgg_ckpt_path, resnet_ckpt_path, device='cpu'):
        """
        Load pretrained weights từ các checkpoint VGG (69.2%) và ResNet (68.5%) đã train riêng.
        Chỉ load các layer backbone (b1-b4, conv1, layer2-4), bỏ qua classifier và attention nội bộ.
        """
        # ── Load VGG weights ──
        vgg_ckpt = torch.load(vgg_ckpt_path, map_location=device)
        vgg_state = vgg_ckpt['model_state_dict']
        
        # Chỉ lấy backbone layers (b1, b2, b3, b4, fusion_pool)
        # Bỏ qua: sa3, sa4 (Spatial Attention), classifier, aux_classifier, conv_proj (chưa train)
        vgg_filtered = {}
        for k, v in vgg_state.items():
            if k.startswith(('b1.', 'b2.', 'b3.', 'b4.', 'fusion_pool.')):
                vgg_filtered[k] = v
        
        missing, unexpected = self.vgg_backbone.load_state_dict(vgg_filtered, strict=False)
        print(f"[Transfer] VGG loaded: {len(vgg_filtered)} params | skipped: {len(missing)} missing")
        
        # ── Load ResNet weights ──
        res_ckpt = torch.load(resnet_ckpt_path, map_location=device)
        res_state = res_ckpt['model_state_dict']
        
        # Chỉ lấy backbone layers (conv1, bn1, layer2-4)
        # Bỏ qua: fc, avgpool, arcface_head
        res_filtered = {}
        for k, v in res_state.items():
            if k.startswith(('conv1.', 'bn1.', 'layer2.', 'layer3.', 'layer4.')):
                res_filtered[k] = v
        
        missing, unexpected = self.res_backbone.resnet.load_state_dict(res_filtered, strict=False)
        print(f"[Transfer] ResNet loaded: {len(res_filtered)} params | skipped: {len(missing)} missing")

    def freeze_backbones(self):
        """Đóng băng cả VGG và ResNet, chỉ cho train Fusion Head."""
        for param in self.vgg_backbone.parameters():
            param.requires_grad = False
        for param in self.res_backbone.parameters():
            param.requires_grad = False
        self.is_frozen = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Freeze] Backbones frozen. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_backbones(self):
        """Mở băng toàn bộ model để fine-tune."""
        for param in self.vgg_backbone.parameters():
            param.requires_grad = True
        for param in self.res_backbone.parameters():
            param.requires_grad = True
        self.is_frozen = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Unfreeze] All params trainable: {trainable:,}")

    def check_unfreeze(self, epoch):
        """Gọi bởi Trainer mỗi epoch để kiểm tra có nên mở băng không."""
        if self.is_frozen and self.freeze_epochs > 0 and epoch >= self.freeze_epochs:
            print(f"\n{'='*50}")
            print(f"[Phase 2] Epoch {epoch+1}: Unfreezing backbones for fine-tuning!")
            print(f"{'='*50}\n")
            self.unfreeze_backbones()
            return True  # Signal to trainer to rebuild optimizer
        return False

    def forward(self, x):
        # 1. Extract features from both backbones
        vgg_feat = self.vgg_backbone(x) # [B, 9, 512]
        res_feat = self.res_backbone(x) # [B, 9, 1024]
        
        # 2. Fuse features via Multi-Attention
        fused_feat = self.fusion_block(vgg_feat, res_feat) # [B, 9, 512]
        
        # 3. Final Classification
        out = torch.flatten(fused_feat, 1) # [B, 9 * 512]
        logits = self.classifier(out)
        
        return logits

