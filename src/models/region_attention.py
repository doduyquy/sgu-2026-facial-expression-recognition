import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import VGGFusionSpatialCNN
from .resnet import ResNet50

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
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
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ResNet50FeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng không gian từ ResNet50 dưới dạng lưới 3x3 (9 tokens).
    Không sử dụng CBAM hay bất kỳ Attention nội bộ nào.
    Output: [B, 9, 1024]
    """
    def __init__(self, config, channels=1):
        super().__init__()
        self.resnet = ResNet50(config, channels)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # Stem
        x = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.pool(x)        # [B, 64, 24, 24]

        # Stages (ResNet-35 bắt đầu từ layer2)
        x = self.resnet.layer2(x)      # [B, 256, 24, 24]
        x = self.resnet.layer3(x)      # [B, 512, 12, 12]
        x = self.resnet.layer4(x)      # [B, 1024, 6, 6]

        x = self.pool(x)              # [B, 1024, 3, 3]
        x = torch.flatten(x, 2)       # [B, 1024, 9]
        x = x.transpose(1, 2)         # [B, 9, 1024]
        return x


# =====================================================================
# 1. Facial Region Dictionary
# =====================================================================
class FacialRegionDictionary(nn.Module):
    # Danh sách tên các vùng khuôn mặt (giống Dictionary box trong sơ đồ)
    REGION_NAMES = [
        "forehead",    # 0: Trán, lông mày - nơi thể hiện nhíu mày (angry/sad)
        "left_eye",    # 1: Mắt trái - nheo mắt, mở to (surprise/fear)
        "right_eye",   # 2: Mắt phải - đối xứng với mắt trái
        "nose",        # 3: Mũi - nhăn mũi (disgust)
        "mouth",       # 4: Miệng - cười, mếu, há miệng (happy/surprise/sad)
        "chin",        # 5: Cằm, đường viền hàm - căng cơ hàm (angry)
    ]

    def __init__(self, num_regions=6, embed_dim=512):
        super().__init__()
        self.num_regions = num_regions
        
        # Tokenize: mỗi vùng → 1 vector embedding (giống t_1, t_2, ..., t_C trong sơ đồ)
        self.token_embed = nn.Embedding(num_regions, embed_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        
        # Lưu index cố định: [0, 1, 2, 3, 4, 5]
        self.register_buffer(
            'region_ids', 
            torch.arange(num_regions, dtype=torch.long)
        )
        
        print(f"--> Facial Region Dictionary: {self.REGION_NAMES}")

    def forward(self, batch_size):
        # Tokenize: index → embedding vectors
        # region_ids: [K] → token_embed: [K, D] → expand: [B, K, D]
        tokens = self.token_embed(self.region_ids)  # [K, D]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]


# =====================================================================
# 2. Semantic-Visual Alignment (Cross-Attention)
# =====================================================================
class SemanticVisualAlignment(nn.Module):
   
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        # Cross-Attention: region tokens query visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network để tinh chỉnh
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
    
        # Cross-Attention: Q = regions, K = V = visual
        attn_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual_features,
            value=visual_features
        )
        # Residual + Norm + DropPath
        region_enriched = self.norm1(region_tokens + self.drop_path(attn_out))

        # FFN + Residual + Norm + DropPath
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + self.drop_path(ffn_out))

        return region_enriched, attn_weights


# =====================================================================
# 3. Model chính: RegionAlignedFER
# =====================================================================
class RegionAlignedFER(nn.Module):

    def __init__(self, config, channels=1):
        super().__init__()
        model_cfg = config.get('model', {})
        self.embed_dim = model_cfg.get('embed_dim', 512)
        self.num_heads = model_cfg.get('num_heads', 4)
        self.num_regions = model_cfg.get('num_regions', 6)
        self.num_layers = model_cfg.get('num_encoder_layers', 2)
        self.dropout_rate = model_cfg.get('transformer_dropout', 0.1)
        num_classes = config['data']['num_classes']

        # ===== 1. Dual Backbone (Feature Extractors) =====
        self.vgg_backbone = VGGFusionSpatialCNN(config, channels)
        self.res_backbone = ResNet50FeatureExtractor(config, channels)
        
        # Transfer Learning state
        self.is_frozen = False
        self.freeze_epochs = model_cfg.get('freeze_backbone_epochs', 0)

        # Project ResNet 1024-d → 512-d để đồng bộ với VGG
        self.proj_res = nn.Linear(1024, self.embed_dim)

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
        # Pool visual features → single vector, rồi broadcast cộng vào Φ_sem
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate) # Add dropout here
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
            encoder_layer,
            num_layers=self.num_layers
        )

        # Positional Encoding cho region tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_regions, self.embed_dim) * 0.02
        )

        # ===== 6. Classification Head =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.5), # Keep this high
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def load_pretrained_backbones(self, vgg_ckpt_path, resnet_ckpt_path, device='cpu'):
        """Load pretrained weights into VGG and ResNet components.
        Tự động bỏ qua các weight bị lệch shape (ví dụ: sa4 kernel 3x3 vs 7x7).
        """
        # ── Load VGG ──
        vgg_ckpt = torch.load(vgg_ckpt_path, map_location=device)
        vgg_state = vgg_ckpt['model_state_dict']
        vgg_prefixes = ('b1.', 'b2.', 'b3.', 'b4.', 'fusion_pool.', 'sa3.', 'sa4.')
        vgg_filtered = {k: v for k, v in vgg_state.items() if k.startswith(vgg_prefixes)}
        
        # Lọc theo shape: chỉ nạp weight có kích thước khớp với model hiện tại
        model_state = self.vgg_backbone.state_dict()
        vgg_compatible = {}
        vgg_skipped = []
        for k, v in vgg_filtered.items():
            if k in model_state and model_state[k].shape == v.shape:
                vgg_compatible[k] = v
            else:
                vgg_skipped.append(k)
        
        self.vgg_backbone.load_state_dict(vgg_compatible, strict=False)
        print(f"[RegionAligned] VGG loaded: {len(vgg_compatible)} weights")
        if vgg_skipped:
            print(f"[RegionAligned] VGG skipped (shape mismatch): {vgg_skipped}")

        # ── Load ResNet ──
        res_ckpt = torch.load(resnet_ckpt_path, map_location=device)
        res_state = res_ckpt['model_state_dict']
        res_prefixes = ('conv1.', 'bn1.', 'layer2.', 'layer3.', 'layer4.')
        res_filtered = {k: v for k, v in res_state.items() if k.startswith(res_prefixes)}
        
        # Lọc theo shape cho ResNet
        res_model_state = self.res_backbone.resnet.state_dict()
        res_compatible = {}
        res_skipped = []
        for k, v in res_filtered.items():
            if k in res_model_state and res_model_state[k].shape == v.shape:
                res_compatible[k] = v
            else:
                res_skipped.append(k)
        
        self.res_backbone.resnet.load_state_dict(res_compatible, strict=False)
        print(f"[RegionAligned] ResNet loaded: {len(res_compatible)} weights")
        if res_skipped:
            print(f"[RegionAligned] ResNet skipped (shape mismatch): {res_skipped}")

    def freeze_backbones(self):
        """Freeze both backbones for Phase 1."""
        for param in self.vgg_backbone.parameters(): param.requires_grad = False
        for param in self.res_backbone.parameters(): param.requires_grad = False
        self.is_frozen = True
        print("[RegionAligned] Backbones FROZEN.")

    def unfreeze_backbones(self):
        """Unfreeze everything for Phase 2."""
        for param in self.parameters(): param.requires_grad = True
        self.is_frozen = False
        print("[RegionAligned] All parameters UNFROZEN.")

    def check_unfreeze(self, epoch):
        if self.is_frozen and self.freeze_epochs > 0 and epoch >= self.freeze_epochs:
            self.unfreeze_backbones()
            return True
        return False

    def forward(self, x):
        B = x.shape[0]

        # ── 1. Feature Extraction ──
        vgg_feat = self.vgg_backbone(x)          # [B, 9, 512]
        res_feat = self.res_backbone(x)          # [B, 9, 1024]
        res_feat = self.proj_res(res_feat)       # [B, 9, 512]

        # Φ_visual: nối đặc trưng từ cả hai backbone
        visual_features = torch.cat([vgg_feat, res_feat], dim=1)  # [B, 18, 512]

        # ── 2. Region Tokens ──
        region_tokens = self.region_dict(B)      # [B, 6, 512]

        # ── 3. Semantic-Visual Alignment (Cross-Attention) ──
        # region Q "soi" vào visual K,V
        phi_sem, attn_weights = self.alignment(
            region_tokens, visual_features
        )                                        # [B, 6, 512], [B, 6, 18]

        # ── 4. Hyper-visual Representation ──
        # Pool toàn bộ visual features → 1 vector, broadcast cộng vào Φ_sem
        phi_visual = visual_features.mean(dim=1, keepdim=True)  # [B, 1, 512]
        phi_visual = self.visual_proj(phi_visual)               # [B, 1, 512]
        hyper_visual = phi_sem + phi_visual                     # [B, 6, 512]

        # ── 5. Transformer Encoder ──
        hyper_visual = hyper_visual + self.pos_embed            # [B, 6, 512]
        encoded = self.transformer_encoder(hyper_visual)        # [B, 6, 512]

        # ── 6. Classification ──
        pooled = encoded.mean(dim=1)             # [B, 512]
        logits = self.classifier(pooled)         # [B, num_classes]

        return logits


# =====================================================================
# Testing
# =====================================================================
if __name__ == "__main__":
    print("=== Testing RegionAlignedFER ===")
    config = {
        'data': {'num_classes': 7, 'channels': 1},
        'model': {
            'embed_dim': 512,
            'num_heads': 4,
            'num_regions': 6,
            'num_encoder_layers': 2,
            'transformer_dropout': 0.1,
            'use_aux': False,
            'attention_type': None,
            'dropout_dense': 0.5,
            'dropout_block': 0.3,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.randn(2, 1, 48, 48).to(device)

    model = RegionAlignedFER(config, channels=1).to(device)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: [2, 7]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    assert out.shape == (2, 7), f"Expected (2, 7), got {out.shape}"
    print("\nTest Passed!")
