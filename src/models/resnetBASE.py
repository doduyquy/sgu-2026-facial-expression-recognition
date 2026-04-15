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

    def forward(self, x):
        """input: (B, C, 48, 48)
       stage1: (B, 64, 24, 24)
       stage2: (B, 256, 24, 24)
       stage3: (B, 512, 12, 12)
       stage4: (B, 1024, 6, 6)
       sau quá trình trích xuất đặc trưng ta avgpool để giảm kích thước về (B, 1024, 1, 1)
       sau đó flatten để giảm kích thước về (B, 1024)
       cuối cùng ta đưa vào fc để phân loại"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # ResNet Blocks
        x = self.layer2(x)  # -> (B, 256, 24, 24)
        x = self.layer3(x)  # -> (B, 512, 12, 12)
        x = self.layer4(x)  # -> (B, 1024, 6, 6)

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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer2(x) #[B,256,24,24]
        x = self.layer3(x) #[B,512,12,12]
        x = self.layer4(x)  # [B, 1024, 6, 6]
        x = torch.flatten(x, 2)               # [B, 1024, 36]
        x = x.transpose(1, 2)                 # [B, 36, 1024]
        return x

class Dictionary(nn.Module):
    """
    Facial Region Dictionary: tạo K learnable tokens đại diện cho K vùng khuôn mặt.
    Mỗi token sẽ học cách "query" vào visual features qua Cross-Attention.
    """
    REGION_NAMES = [
        "forehead",    # 0: Trán, lông mày
        "left_eye",    # 1: Mắt trái
        "right_eye",   # 2: Mắt phải
        "nose",        # 3: Mũi
        "mouth",       # 4: Miệng
        "chin",        # 5: Cằm
    ]

    def __init__(self, num_regions=6, emb_dim=1024):
        super().__init__()
        self.num_regions = num_regions

        # K learnable embedding tokens
        self.token_embedding = nn.Embedding(num_regions, emb_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # Buffer cố định index → đảm bảo luôn đúng device
        self.register_buffer(
            'region_ids',
            torch.arange(num_regions, dtype=torch.long)
        )

        print(f"--> Facial Region Dictionary: {self.REGION_NAMES[:num_regions]}")

    def forward(self, batch_size):
        # region_ids: [K] → token_embedding: [K, D] → expand: [B, K, D]
        tokens = self.token_embedding(self.region_ids)  # [K, D]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]


class CNNDictionary(nn.Module):
    """
    ResNet35 (backbone) → flatten visual tokens → Cross-Attention với Dictionary → Classifier.
    Không dùng Transformer Encoder, chỉ 1 lớp Cross-Attention để test heatmap.

    Flow:
        img → ResNet35 extract → [B, 36, 1024] visual tokens
        Dictionary → [B, K, 1024] region tokens
        Cross-Attention(Q=region, K=V=visual) → [B, K, 1024] enriched regions
        mean pool → [B, 1024] → classifier → [B, num_classes]
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

        # Positional Encoding cho region tokens (tăng cường identity cho dictionary)
        self.region_pos = nn.Parameter(torch.randn(1, self.num_regions, self.embed_dim) * 0.02)

        # 4. Cross-Attention: region tokens (Q) soi vào visual features (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        # LayerNorm + Residual cho cross-attention output
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # FFN sau cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # 5. Attention Pooling: học trọng số mỗi region thay vì mean cào bằng
        #    Dùng 1 learnable query vector "hỏi" K regions → weighted sum
        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=1,
            batch_first=True
        )
        self.norm_pool = nn.LayerNorm(self.embed_dim)

        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 1, 48, 48]
        return: [B, num_classes]
        """
        B = x.shape[0]

        # 1. Extract visual features từ ResNet35
        visual = self.resnet35.extract_region_features(x)  # [B, 36, 1024]
        visual = visual + self.visual_pos                   # + positional encoding

        # 2. Lấy region dictionary tokens + positional encoding
        region_tokens = self.dic_region(B)                  # [B, K, 1024]
        region_tokens = region_tokens + self.region_pos     # + region PE

        # 3. Cross-Attention: Q=region, K=V=visual
        #    Region tokens "hỏi" visual features: mỗi vùng mặt chú ý vào đâu?
        attn_out, self.attn_weights = self.cross_attn(
            query=region_tokens,
            key=visual,
            value=visual
        )  # attn_out: [B, K, 1024], attn_weights: [B, K, 36]

        # Residual + LayerNorm
        region_enriched = self.norm1(region_tokens + attn_out)  # [B, K, 1024]

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(region_enriched)
        region_enriched = self.norm2(region_enriched + ffn_out)  # [B, K, 1024]

        # 4. Attention Pooling: 1 query vector hỏi 6 regions → weighted sum
        #    Model tự học: "emotion này, region nào quan trọng nhất?"
        pool_query = self.attn_pool_query.expand(B, -1, -1)  # [B, 1, D]
        pooled, self.pool_weights = self.attn_pool(
            query=pool_query,
            key=region_enriched,
            value=region_enriched
        )  # pooled: [B, 1, D], pool_weights: [B, 1, K]
        pooled = self.norm_pool(pooled.squeeze(1))           # [B, D]

        # 5. Classify
        logits = self.classifier(pooled)                    # [B, num_classes]
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