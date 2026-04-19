import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_T_Weights
from .vgg import VGGFusionSpatialCNN

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# =====================================================================
# 1. Swin Transformer Feature Extractor
# =====================================================================
class SwinFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng không gian từ Swin Transformer.
    Sử dụng Swin-Tiny (weights mặc định của ImageNet).
    Output: [B, 9, 768] để khớp số lượng 9 tokens tương tự ResNet cũ.
    """
    def __init__(self, channels=1):
        super().__init__()
        # Khởi tạo Swin-Tiny pre-trained
        self.swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        self.swin_dim = 768 # Output dimension của swin_v2_t stage cuối
        
        # Swin Transformer trong torchvision nhận ảnh 3 kênh. Nếu ảnh FER là 1 kênh:
        if channels == 1:
            # Sửa lại Patch Merging / Conv2d đầu ra ở lớp đầu tiên
            old_conv = self.swin.features[0][0] # Đây là Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
            new_conv = nn.Conv2d(1, old_conv.out_channels, 
                                 kernel_size=old_conv.kernel_size, 
                                 stride=old_conv.stride, 
                                 padding=old_conv.padding, 
                                 bias=(old_conv.bias is not None))
            # (Tùy chọn) Copy trọng số: tính trung bình 3 kênh RGB tạo thành kênh Grayscale
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
            self.swin.features[0][0] = new_conv

        # Gom feature thành không gian 3x3 (9 tokens cục bộ) tương tự ResNet cũ
        # Chú ý: Swin features module xuất ra dạng tensor 1D nhúng, torchvision swin trả về NCHW
        # Nên dùng pooling cho an toàn nếu đầu vào không phải 224x224
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # Resize lên 224x224 (hoặc kích thước phù hợp) vì Swin cần ảnh đủ lớn để chia Window Size (8x8)
        # Nếu input là 48x48 thì quá nhỏ so với Swin chuẩn. Ta upsample trước khi đưa vào.
        if x.shape[-1] < 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            
        # Trích xuất qua mạng Swin
        x = self.swin.features(x) # [B, 7, 7, 768] (NHWC)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2) # chuyển đổi sang thành [B, 768, 7, 7] (NCHW)
        
        # Adaptive pool về 3x3 để tạo 9 vùng tokens giống với thiết kế trước đây
        x = self.pool(x)          # [B, 768, 3, 3]
        x = torch.flatten(x, 2)   # [B, 768, 9]
        x = x.transpose(1, 2)     # [B, 9, 768]
        return x

# =====================================================================
# 2. Facial Region Dictionary
# =====================================================================
class FacialRegionDictionary(nn.Module):
    REGION_NAMES = [
        "forehead", "left_eye", "right_eye", 
        "nose", "mouth", "chin"
    ]

    def __init__(self, num_regions=6, embed_dim=512):
        super().__init__()
        self.num_regions = num_regions
        self.token_embed = nn.Embedding(num_regions, embed_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        self.register_buffer('region_ids', torch.arange(num_regions, dtype=torch.long))

    def forward(self, batch_size):
        tokens = self.token_embed(self.region_ids)  # [K, D]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]

# =====================================================================
# 3. Semantic-Visual Alignment (Cross-Attention)
# =====================================================================
class SemanticVisualAlignment(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        # Cross-Attention: region tokens query visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(dropout if dropout > 0. else 0.)

    def forward(self, region_tokens, visual_features):
        attn_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual_features,
            value=visual_features
        )
        region_enriched = self.norm1(region_tokens + self.drop_path(attn_out))
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + self.drop_path(ffn_out))

        return region_enriched, attn_weights

# =====================================================================
# 4. Swin Region-Aligned FER Model (MAIN)
# =====================================================================
class SwinRegionAlignedFER(nn.Module):
    def __init__(self, config, channels=1):
        super().__init__()
        model_cfg = config.get('model', {})
        self.embed_dim = model_cfg.get('embed_dim', 512)
        self.num_heads = model_cfg.get('num_heads', 4)
        self.num_regions = model_cfg.get('num_regions', 6)
        self.num_layers = model_cfg.get('num_encoder_layers', 2)
        self.dropout_rate = model_cfg.get('transformer_dropout', 0.1)
        num_classes = config['data']['num_classes']

        # ===== 1. Dual Backbone (VGG + Swin Transformer) =====
        self.vgg_backbone = VGGFusionSpatialCNN(config, channels)
        self.swin_backbone = SwinFeatureExtractor(channels)
        
        self.is_frozen = False
        self.freeze_epochs = model_cfg.get('freeze_backbone_epochs', 0)

        # Swin_V2_T trả về 768-D. Project về 512-D để đồng bộ với VGG.
        self.proj_swin = nn.Linear(768, self.embed_dim)

        # ===== 2. Facial Region Dictionary =====
        self.region_dict = FacialRegionDictionary(
            num_regions=self.num_regions,
            embed_dim=self.embed_dim
        )

        # ===== 3. Semantic-Visual Alignment =====
        self.alignment = SemanticVisualAlignment(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )

        # ===== 4. Hyper-visual Representation =====
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

        # ===== 5. Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 2,
            dropout=self.dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_regions, self.embed_dim) * 0.02
        )

        # ===== 6. Classification Head =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def freeze_backbones(self):
        for param in self.vgg_backbone.parameters(): param.requires_grad = False
        for param in self.swin_backbone.parameters(): param.requires_grad = False
        self.is_frozen = True
        print("[SwinRegionAligned] Backbones FROZEN.")

    def unfreeze_backbones(self):
        for param in self.parameters(): param.requires_grad = True
        self.is_frozen = False
        print("[SwinRegionAligned] All parameters UNFROZEN.")

    def forward(self, x):
        B = x.shape[0]

        # ── 1. Feature Extraction ──
        vgg_feat = self.vgg_backbone(x)          # [B, 9, 512]
        
        swin_feat = self.swin_backbone(x)        # [B, 9, 768]
        swin_feat = self.proj_swin(swin_feat)    # [B, 9, 512]

        visual_features = torch.cat([vgg_feat, swin_feat], dim=1)  # [B, 18, 512]

        # ── 2. Region Tokens ──
        region_tokens = self.region_dict(B)      # [B, 6, 512]

        # ── 3. Cross-Attention ──
        phi_sem, _ = self.alignment(region_tokens, visual_features)  # [B, 6, 512]

        # ── 4. Hyper-visual ──
        phi_visual = visual_features.mean(dim=1, keepdim=True)  # [B, 1, 512]
        phi_visual = self.visual_proj(phi_visual)               # [B, 1, 512]
        hyper_visual = phi_sem + phi_visual                     # [B, 6, 512]

        # ── 5. Transformer ──
        hyper_visual = hyper_visual + self.pos_embed            # [B, 6, 512]
        encoded = self.transformer_encoder(hyper_visual)        # [B, 6, 512]

        # ── 6. Classification ──
        pooled = encoded.mean(dim=1)             # [B, 512]
        logits = self.classifier(pooled)         # [B, num_classes]

        return logits

if __name__ == "__main__":
    print("=== Testing SwinRegionAlignedFER ===")
    config = {
        'data': {'num_classes': 7, 'channels': 1},
        'model': {
            'embed_dim': 512,
            'num_heads': 4,
            'num_regions': 6,
            'num_encoder_layers': 2,
            'transformer_dropout': 0.1,
        }
    }
    # Test tensor có kích thước 48x48 như trong FER
    dummy = torch.randn(2, 1, 48, 48)

    model = SwinRegionAlignedFER(config, channels=1)
    out = model(dummy)
    
    print(f"Output shape: {out.shape}")  # Phải là [2, 7]
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("Test Passed!")
