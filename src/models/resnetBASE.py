import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import CBAM, SpatialAttention,ChannelAttention
# Input:  (B, 1, 48, 48)
# conv1: 3x3, stride=1, pad=1           -> (B, 64, 48, 48)
# pool:  2x2, stride=2                  -> (B, 64, 24, 24)
# layer2: ConvBlock(s=1) + 2 IDs        -> (B, 256, 24, 24)
# layer3: ConvBlock(s=2) + 3 IDs        -> (B, 512, 12, 12)
# layer4: ConvBlock(s=2) + 3 IDs        -> (B, 1024, 6, 6)
# avgpool: AdaptiveAvgPool2d((1,1))     -> (B, 1024, 1, 1)
# flatten                               -> (B, 1024)
# fc                                    -> (B, num_classes)
#Hout = ((Hin + 2*pad - kernel_size) // stride) + 1
class IdentityBlock(nn.Module):
    def __init__(self, channels, filters):
        super(IdentityBlock,self). __init__()
        F1,F2,F3=filters
        self.conv1=nn.Conv2d(channels,F1,kernel_size=1)
        self.bn1=nn.BatchNorm2d(F1)

        self.conv2=nn.Conv2d(F1,F2,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(F2)
        
        self.conv3=nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3=nn.BatchNorm2d(F3)
        self.attn=nn.Identity()

        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        shorcut=x
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.bn3(self.conv3(x))
        x=self.attn(x)
        x=x+shorcut # cong phan du
        x=self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=2):
        super().__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f3),
        )
        self.attn = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.attn(x)
        x = self.relu(x + shortcut)
        return x


# class Resnet35(nn.Module):

#     def __init__(self, config, channels=1):
#         super().__init__()
        
#         # Lấy thông tin từ config
#         self.num_classes = config.get('data', {}).get('num_classes', 7) if config else 7
        
#         # 1. Stem
#         # Input: (B, channels, 48, 48)
#         self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # 2. Stages
#         # Mỗi block bottleneck sẽ có dạng [f1, f2, f3]
#         # Stage 2: 3 blocks -> Output spatial: 24x24
#         self.layer2 = nn.Sequential(
#             ConvBlock(64, [64, 64, 256], stride=1),
#             IdentityBlock(256, [64, 64, 256]),
#             IdentityBlock(256, [64, 64, 256])
#         )

#         # Stage 3: 4 blocks -> Output spatial: 12x12
#         self.layer3 = nn.Sequential(
#             ConvBlock(256, [128, 128, 512]),
#             IdentityBlock(512, [128, 128, 512]),
#             IdentityBlock(512, [128, 128, 512]),
#             IdentityBlock(512, [128, 128, 512])
#         )

#         # Stage 4: 4 blocks -> Output spatial: 6x6
#         self.layer4 = nn.Sequential(
#             ConvBlock(512, [256, 256, 1024]),
#             IdentityBlock(1024, [256, 256, 1024]),
#             IdentityBlock(1024, [256, 256, 1024]),
#             IdentityBlock(1024, [256, 256, 1024])
#         )
#         self.dropout = nn.Dropout(0.3)

#         # 3. Head for Classification (Trường hợp dùng độc lập)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(1024, self.num_classes)
#         # self.classifier = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d((1, 1)),          
#         #     nn.Flatten(),                        
#         #     nn.Dropout(0.3),
#         #     nn.Linear(1024, 512),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.4),
#         #     nn.Linear(512, self.num_classes)
#         # )

#     def forward(self, x):
#         """input: (B, C, 48, 48)
#        stage1: (B, 64, 24, 24)
#        stage2: (B, 256, 24, 24)
#        stage3: (B, 512, 12, 12)
#        stage4: (B, 1024, 6, 6)
#        sau quá trình trích xuất đặc trưng ta avgpool để giảm kích thước về (B, 1024, 1, 1)
#        sau đó flatten để giảm kích thước về (B, 1024)
#        cuối cùng ta đưa vào fc để phân loại"""
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)

#         # ResNet Blocks
#         x = self.layer2(x)  # -> (B, 256, 24, 24)
#         x = self.layer3(x)  # -> (B, 512, 12, 12)
#         x = self.layer4(x)  # -> (B, 1024, 6, 6)

#         x = self.avgpool(x) # -> (B, 1024, 1, 1)
#         x = torch.flatten(x, 1) # -> (B, 1024)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

#     # def forward(self, x): # thử tính toán nặng hơn 
#     #     x = self.relu(self.bn1(self.conv1(x)))
#     #     x = self.pool(x)
#     #     x = self.layer2(x)  # -> (B, 256, 24, 24)
#     #     x = self.layer3(x)  # -> (B, 512, 12, 12)
#     #     x = self.layer4(x)  # -> (B, 1024, 6, 6)
#     #     return self.classifier(x)
#     def extract_region_features(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = self.layer2(x) #[B,256,24,24]
#         x = self.layer3(x) #[B,512,12,12]
#         x = self.layer4(x)  # [B, 1024, 6, 6]
#         x = torch.flatten(x, 2)               # [B, 1024, 36]
#         x = x.transpose(1, 2)                 # [B, 36, 1024]
#         return x

class Resnet35(nn.Module):

    def __init__(self, config, channels=1):
        super().__init__()
        
        # Lấy thông tin từ config
        self.num_classes = config.get('data', {}).get('num_classes', 7) if config else 7
        
        # 1. Stem
        # Input: (B, channels, 48, 48)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Lấy setting Attention
        model_cfg = config.get('model', {}) if config else {}
        self.attention_type = model_cfg.get('attention_type', 'cbam')

        def get_attn(in_channels, kernel_size=7):
            if self.attention_type == 'cbam':
                return CBAM(in_channels, kernel_size=kernel_size)
            elif self.attention_type == 'spatial':
                return SpatialAttention(kernel_size=kernel_size)
            elif self.attention_type == 'channel':
                return ChannelAttention(in_channels)
            return nn.Identity()

        # 2. Stages
        # Mỗi block bottleneck sẽ có dạng [f1, f2, f3]
        # Stage 2: 3 blocks -> Output spatial: 24x24
        self.layer2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256])
        )
        for i in range(3): self.layer2[i].attn = get_attn(256, kernel_size=7)

        # Stage 3: 4 blocks -> Output spatial: 12x12, kernel=7
        self.layer3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512])
        )
        for i in range(4): self.layer3[i].attn = get_attn(512, kernel_size=7)

        # Stage 4: 4 blocks -> Output spatial: 6x6, kernel=3
        self.layer4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024])
        )
        for i in range(4): self.layer4[i].attn = get_attn(1024, kernel_size=3)

        # 3. Head for Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),          
        #     nn.Flatten(),                        
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, self.num_classes)
        # )

    def forward_features(self, x):
        """Return final feature map before classifier: [B, 1024, 6, 6]."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # ResNet Blocks
        x = self.layer2(x)  # -> (B, 256, 24, 24)
        x = self.layer3(x)  # -> (B, 512, 12, 12)
        x = self.layer4(x)  # -> (B, 1024, 6, 6)

        return x

    def forward(self, x):
        """input: (B, C, 48, 48)
       stage1: (B, 64, 24, 24)
       stage2: (B, 256, 24, 24)
       stage3: (B, 512, 12, 12)
       stage4: (B, 1024, 6, 6)
       sau quá trình trích xuất đặc trưng ta avgpool để giảm kích thước về (B, 1024, 1, 1)
       sau đó flatten để giảm kích thước về (B, 1024)
       cuối cùng ta đưa vào fc để phân loại"""
        x = self.forward_features(x)

        x = self.classifier(x)
        return x

    # def forward(self, x): # thử tính toán nặng hơn 
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.pool(x)
    #     x = self.layer2(x)  # -> (B, 256, 24, 24)
    #     x = self.layer3(x)  # -> (B, 512, 12, 12)
    #     x = self.layer4(x)  # -> (B, 1024, 6, 6)
    #     return self.classifier(x)
    def extract_region_features(self, x):
        x = self.forward_features(x)  # [B, 1024, 6, 6]
        x = torch.flatten(x, 2)               # [B, 1024, 36]
        x = x.transpose(1, 2)                 # [B, 36, 1024]
        return x
class TransformerBlock(nn.Module):
    def __init__(self, dim=512, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViTHead(nn.Module):
    def __init__(self, in_dim=1024, embed_dim=512, depth=2, num_heads=4, num_classes=7, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 36, embed_dim) * 0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=2.0,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.4),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 1024, 6, 6]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # [B, 36, 1024]
        x = self.proj(x)                   # [B, 36, 512]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.blocks(x)                 # [B, 36, 512]
        x = x.mean(dim=1)                  # [B, 512]
        out = self.classifier(x)
        return out

class ResNet35_CBAM_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get('model', {}) if config else {}
        self.backbone = Resnet35(config, channels=1)
        self.is_frozen = False
        self.freeze_epochs = model_cfg.get('freeze_backbone_epochs', 0)

        self.vit_head = TinyViTHead(
            in_dim=1024,
            embed_dim=model_cfg.get('vit_embed_dim', 512),
            depth=model_cfg.get('vit_depth', 2),
            num_heads=model_cfg.get('vit_num_heads', 4),
            num_classes=config['data']['num_classes'],
            dropout=model_cfg.get('vit_dropout', 0.1)
        )

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        out = self.vit_head(feat)
        return out

    def load_pretrained_backbone(self, ckpt_path, device='cpu'):
        """Load pretrained weights into the CNN backbone only (shape-safe)."""
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_state = ckpt.get('model_state_dict', ckpt)

        model_state = self.backbone.state_dict()
        compatible = {}
        skipped = []

        for k, v in saved_state.items():
            candidate_keys = [k]
            if k.startswith('resnet35.'):
                candidate_keys.append(k.replace('resnet35.', '', 1))
            if k.startswith('backbone.'):
                candidate_keys.append(k.replace('backbone.', '', 1))

            mapped_key = None
            for ckey in candidate_keys:
                if ckey in model_state and model_state[ckey].shape == v.shape:
                    mapped_key = ckey
                    break

            if mapped_key is not None:
                compatible[mapped_key] = v
            else:
                skipped.append(k)

        self.backbone.load_state_dict(compatible, strict=False)
        print(f"[ResNet35_CBAM_ViT] Backbone loaded: {len(compatible)} weights")
        if skipped:
            print(f"[ResNet35_CBAM_ViT] Skipped: {len(skipped)} keys")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.is_frozen = True
        print("[ResNet35_CBAM_ViT] Backbone FROZEN.")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        self.is_frozen = False
        print("[ResNet35_CBAM_ViT] All parameters UNFROZEN.")

    def check_unfreeze(self, epoch):
        if self.is_frozen and self.freeze_epochs > 0 and epoch >= self.freeze_epochs:
            self.unfreeze_backbone()
            return True
        return False

    def get_param_groups(self, backbone_lr, head_lr):
        backbone_params = list(self.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]

        print(
            f"[ResNet35_CBAM_ViT] Param groups: backbone={len(backbone_params)} tensors (lr={backbone_lr}), "
            f"head={len(head_params)} tensors (lr={head_lr})"
        )

        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]



class Dictionary(nn.Module):
    """
    Facial Region Dictionary: tạo K learnable tokens đại diện cho K vùng khuôn mặt.
    Mỗi token sẽ học cách "query" vào visual features qua Cross-Attention.
    Với Spatial Region Mask, mỗi token bị ép chỉ nhìn vào vùng tương ứng trên feature map 6×6:
        Row 0 = forehead, Row 1 = eyebrows, Row 2 = eyes,
        Row 3 = nose, Row 4 = mouth, Row 5 = chin
    """
    REGION_NAMES = [
        "forehead",    # 0: Row 0 — Trán
        "eyebrows",    # 1: Row 1 — Lông mày
        "eyes",        # 2: Row 2 — Mắt
        "nose",        # 3: Row 3 — Mũi
        "mouth",       # 4: Row 4 — Miệng
        "chin",        # 5: Row 5 — Cằm
    ]

    def __init__(self, num_regions=6, emb_dim=1024):
        super().__init__()
        self.num_regions = num_regions
        self.emb_dim = emb_dim

        # K learnable embedding tokens
        self.token_embedding = nn.Embedding(num_regions, emb_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Dynamic dictionary: điều kiện hóa token theo ngữ cảnh ảnh.
        self.context_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.region_gate = nn.Linear(emb_dim, num_regions)

        # Buffer cố định index → đảm bảo luôn đúng device
        self.register_buffer(
            'region_ids',
            torch.arange(num_regions, dtype=torch.long)
        )

        print(f"--> Facial Region Dictionary: {self.REGION_NAMES[:num_regions]}")

    def forward(self, batch_size, global_context=None):
        # region_ids: [K] → token_embedding: [K, D] → expand: [B, K, D]
        tokens = self.token_embedding(self.region_ids)  # [K, D]
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]

        if global_context is None:
            return tokens

        # global_context: [B, D] -> dynamic_bias: [B, 1, D]
        dynamic_bias = self.context_proj(global_context).unsqueeze(1)
        # gate theo từng region: [B, K, 1]
        region_gate = torch.sigmoid(self.region_gate(global_context)).unsqueeze(-1)

        # Token cuối = token nền + phần điều kiện hóa theo ảnh.
        return tokens + region_gate * dynamic_bias


class CNNDictionary(nn.Module):
    """
    ResNet35 (backbone) → visual tokens → Cross-Attention với Dictionary → Classifier.

    V2 Fixes (4 thủ phạm):
    1. BỎ hard Spatial Region Mask → cross-attention tự do attend toàn bộ 36 tokens
    2. Soft Spatial Grounding → region tokens khởi tạo từ visual features thật (pool theo hàng)
    3. BỎ 2-stage attention (cross_attn + attn_pool) → chỉ cross_attn + mean pool
    4. Visual Shortcut + Gated Fusion → classifier nhận cả visual global lẫn region features

    Flow:
        img → ResNet35 → [B, 36, 1024] visual tokens
        Soft grounding: reshape 6×6 → pool theo hàng → [B, 6, 1024] spatial prior
        Region tokens = dictionary + spatial_prior + region_pos
        Cross-Attention(Q=region, K=V=visual) → [B, 6, 1024] enriched
        Mean pool → [B, 1024] region_pooled
        Visual shortcut: visual.mean → [B, 1024] visual_global
        Gated Fusion: gate * region_pooled + (1-gate) * visual_global → [B, 1024]
        Classifier → [B, num_classes]
    """
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get('model', {})
        self.embed_dim = model_cfg.get('embed_dim', 1024)
        self.num_heads = model_cfg.get('num_heads', 4)
        self.num_regions = model_cfg.get('num_regions', 6)
        num_classes = config['data']['num_classes']

        # 1. CNN Backbone
        self.resnet35 = Resnet35(config, channels=1)

        # Transfer Learning state
        self.is_frozen = False
        self.freeze_epochs = model_cfg.get('freeze_backbone_epochs', 0)

        # 2. Facial Region Dictionary
        self.dic_region = Dictionary(
            num_regions=self.num_regions,
            emb_dim=self.embed_dim
        )

        # 3. Positional Encoding cho visual tokens (6×6 = 36 vị trí)
        self.visual_pos = nn.Parameter(torch.randn(1, 36, self.embed_dim) * 0.02)

        # Positional Encoding cho region tokens
        self.region_pos = nn.Parameter(torch.randn(1, self.num_regions, self.embed_dim) * 0.02)

        # 4. [FIX #2] Soft Spatial Grounding
        #    Pool visual features theo hàng → project → cộng vào region tokens
        #    Thay vì hard mask, region tokens "biết" vị trí qua actual visual content
        self.spatial_grounding = nn.Linear(self.embed_dim, self.embed_dim)

        # 5. [FIX #1] Cross-Attention: KHÔNG dùng mask → tự do attend toàn bộ 36 tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            dropout=0.2
        )

        # LayerNorm + Residual
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # FFN sau cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # [FIX #3] Bỏ Attention Pooling → dùng mean pool đơn giản
        # (không cần attn_pool_query, attn_pool, norm_pool)

        # 6. [FIX #4] Visual Shortcut: bypass cross-attention, đưa visual signal thẳng tới classifier
        self.visual_shortcut = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU()
        )

        # Gated Fusion: học cách blend region features + visual shortcut
        # gate ≈ 1 → tin region head, gate ≈ 0 → tin visual trực tiếp
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid()
        )

        # 7. Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        print(f"--> [CNNDictionary V2] Free cross-attention + soft grounding + visual shortcut")

    def forward(self, x):
        """
        x: [B, 1, 48, 48]
        return: [B, num_classes]
        """
        B = x.shape[0]

        # 1. Extract visual features từ ResNet35
        visual = self.resnet35.extract_region_features(x)  # [B, 36, 1024]
        visual = visual + self.visual_pos                   # + positional encoding

        # Context toàn ảnh để điều kiện hóa dictionary tokens.
        visual_context = visual.mean(dim=1)                 # [B, D]

        # 2. [FIX #2] Soft Spatial Grounding: pool visual features theo hàng
        #    visual [B, 36, D] → reshape [B, 6, 6, D] → mean over cols → [B, 6, D]
        #    Region "forehead" nhận thông tin thật từ row 0, "mouth" từ row 4, ...
        #    Nhưng cross-attention vẫn TỰ DO nhìn toàn bộ 36 tokens
        visual_grid = visual.reshape(B, 6, 6, self.embed_dim)
        spatial_prior = visual_grid.mean(dim=2)              # [B, 6, D]
        spatial_prior = self.spatial_grounding(spatial_prior) # [B, 6, D]

        # Region tokens = learnable dictionary + spatial grounding + positional encoding
        region_tokens = self.dic_region(B, global_context=visual_context) + spatial_prior + self.region_pos  # [B, K, D]

        # 3. [FIX #1] Cross-Attention: Q=region, K=V=visual — KHÔNG mask
        attn_out, self.attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual,
            value=visual
        )  # attn_out: [B, K, D], attn_weights: [B, K, 36]

        # Residual + LayerNorm
        region_enriched = self.norm1(region_tokens + attn_out)  # [B, K, D]

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + ffn_out)  # [B, K, D]

        # 4. [FIX #3] Mean pool thay vì attention pool → tránh loãng signal
        region_pooled = region_enriched.mean(dim=1)             # [B, D]

        # 5. [FIX #4] Visual Shortcut: global pool → bypass cross-attention
        visual_global = visual.mean(dim=1)                      # [B, D]
        visual_global = self.visual_shortcut(visual_global)     # [B, D]

        # Gated Fusion: model tự học blend ratio
        gate = self.fusion_gate(
            torch.cat([region_pooled, visual_global], dim=-1)
        )  # [B, D]
        fused = gate * region_pooled + (1 - gate) * visual_global  # [B, D]

        # 6. Classify
        logits = self.classifier(fused)                         # [B, num_classes]
        return logits

    # ── Transfer Learning ──

    def load_pretrained_backbone(self, ckpt_path, device='cpu'):
        """Load pretrained ResNet35 checkpoint vào backbone.
        Tự bỏ qua weight lệch shape (classifier cũ, attention kernel khác, ...).
        """
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_state = ckpt['model_state_dict']

        # State dict hiện tại của resnet35 backbone
        model_state = self.resnet35.state_dict()

        compatible = {}
        skipped = []
        for k, v in saved_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)

        self.resnet35.load_state_dict(compatible, strict=False)
        print(f"[CNNDictionary] Backbone loaded: {len(compatible)} weights")
        if skipped:
            print(f"[CNNDictionary] Skipped (shape mismatch / not in backbone): {skipped}")

    def freeze_backbone(self):
        """Freeze ResNet35 backbone — chỉ train Dictionary + Cross-Attention + Classifier."""
        for param in self.resnet35.parameters():
            param.requires_grad = False
        self.is_frozen = True
        print("[CNNDictionary] Backbone FROZEN.")

    def unfreeze_backbone(self):
        """Unfreeze toàn bộ model cho fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.is_frozen = False
        print("[CNNDictionary] All parameters UNFROZEN.")

    def check_unfreeze(self, epoch):
        """Tự động unfreeze backbone khi đến epoch chỉ định."""
        if self.is_frozen and self.freeze_epochs > 0 and epoch >= self.freeze_epochs:
            self.unfreeze_backbone()
            return True  # Signal trainer rebuild optimizer
        return False

    def get_param_groups(self, backbone_lr, head_lr):
        """Chia parameters thành 2 nhóm LR khác nhau.
        - backbone (ResNet35): LR thấp, chỉ tinh chỉnh nhẹ
        - head (Dictionary, Cross-Attention, Classifier): LR cao, cần học mạnh
        """
        backbone_params = list(self.resnet35.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        
        print(f"[CNNDictionary] Param groups: backbone={len(backbone_params)} tensors (lr={backbone_lr}), "
              f"head={len(head_params)} tensors (lr={head_lr})")
        
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]
