import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import VGGFusionSpatialCNN
from .resnet import ResNet50

try:
    from transformers import CLIPTokenizer, CLIPTextModel
except ImportError:
    CLIPTokenizer, CLIPTextModel = None, None

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

def get_2d_sincos_pos_embed(embed_dim, grid_size=3):
    """
    Sinh mã hóa vị trí (Positional Encoding) dạng 2D Sin-Cos cố định.
    Rất lý tưởng cho Vision Transformer lưới 2 chiều (ví dụ 3x3 grid).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')  # tọa độ (w, h)
    
    # Kéo phẳng grid [grid_size * grid_size, 2]
    grid = torch.stack(grid, dim=0).reshape(2, -1)
    
    # Tạo omega. Chia embed_dim ra làm 2 mảng (mỗi mảng embed_dim // 2)
    # Vì mỗi mảng lại có 1 sin, 1 cos nên omega sẽ là embed_dim // 4
    emb_dim_half = embed_dim // 2
    omega = torch.arange(emb_dim_half // 2, dtype=torch.float32)
    omega = omega / (emb_dim_half / 2.0)
    omega = 1.0 / (10000**omega)  # shape: [embed_dim // 4]
    
    # Tính toán cho trục y
    out_y = torch.einsum('m,d->md', grid[0], omega)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # [9, 256]
    
    # Tính toán cho trục x
    out_x = torch.einsum('m,d->md', grid[1], omega)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # [9, 256]
    
    # Ghép trục x và y để full 512 embedding => [9, 512]
    emb = torch.cat([emb_y, emb_x], dim=1)
    return emb

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

class CLIPFacialRegionDictionary(nn.Module):
    def __init__(self, num_regions=6, embed_dim=512, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.num_regions = num_regions
        
        prompts = [
            "a photo of a person's forehead conveying eyebrow movement", 
            "a photo of a person's left eye", 
            "a photo of a person's right eye", 
            "a photo of a person's nose", 
            "a photo of a person's mouth and lips", 
            "a photo of a person's chin and jawline"
        ][:num_regions]
        
        if CLIPTokenizer is None or CLIPTextModel is None:
            raise ImportError("Please install transformers to use CLIPFacialRegionDictionary: `pip install transformers`")
            
        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        text_model = CLIPTextModel.from_pretrained(clip_model_name)
        
        print(f"--> Initializing CLIP Text Embeddings from: {clip_model_name}")
        with torch.no_grad():
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            outputs = text_model(**inputs)
            text_features = outputs.pooler_output  # [num_regions, clip_dim]
            
        self.token_embed = nn.Parameter(text_features, requires_grad=True)
        
        self.proj = nn.Identity()
        if text_features.shape[1] != embed_dim:
            self.proj = nn.Linear(text_features.shape[1], embed_dim)

    def forward(self, batch_size):
        tokens = self.proj(self.token_embed) # [K, D]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1) # [B, K, D]


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


class VisualPatchRegionAlignment(nn.Module):
    """
    Cross-Attention ngược chiều so với RegionAlignment gốc:
    Q = visual patch tokens [B, 18, D]
    K/V = region tokens [B, 6, D]
    Output giữ số token ảnh [B, 18, D] để đưa qua shifted axis-window encoder.
    """
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(dropout if dropout > 0. else 0.)

    def forward(self, visual_features, region_tokens):
        attn_out, attn_weights = self.cross_attn(
            query=visual_features,
            key=region_tokens,
            value=region_tokens
        )
        visual_enriched = self.norm1(visual_features + self.drop_path(attn_out))
        ffn_out = self.ffn(visual_enriched)
        visual_enriched = self.norm2(visual_enriched + self.drop_path(ffn_out))
        return visual_enriched, attn_weights


# =====================================================================
# 3. Sub-Graph Fusion (Upper/Lower Face Division)
# =====================================================================
class SubGraphFusion(nn.Module):
    """
    SubGraph Fusion cho Facial Regions.
    Chia 6 vùng thành 2 graph nhỏ hoàn toàn độc lập trong Self-Attention:
    - Upper-face (Indices 0,1,2,3): Trán, Mắt trái, Mắt phải, Mũi
    - Lower-face (Indices 4,5): Miệng, Cằm
    """
    def __init__(self, embed_dim, num_heads, dropout=0.3):
        super().__init__()
        # Self Attention cho từng đồ thị con
        self.upper_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.lower_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_upper = nn.LayerNorm(embed_dim)
        self.norm_lower = nn.LayerNorm(embed_dim)
        
        # Feed-Forward chung sau khi nối
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(dropout if dropout > 0. else 0.)
        
    def forward(self, x):
        # x shape: [B, 6, D]
        upper_nodes = x[:, :4, :] # [B, 4, D]
        lower_nodes = x[:, 4:, :] # [B, 2, D]
        
        # Self Attention trong nội bộ cụm (Tránh mắt làm nhiễu miệng và ngược lại)
        upper_out, _ = self.upper_attn(upper_nodes, upper_nodes, upper_nodes)
        lower_out, _ = self.lower_attn(lower_nodes, lower_nodes, lower_nodes)
        
        # Residual + Norm
        upper_fused = self.norm_upper(upper_nodes + self.drop_path(upper_out))
        lower_fused = self.norm_lower(lower_nodes + self.drop_path(lower_out))
        
        # Nối lại
        fused = torch.cat([upper_fused, lower_fused], dim=1) # [B, 6, D]
        
        # FFN kết hợp
        ffn_out = self.ffn(fused)
        out = self.norm_out(fused + self.drop_path(ffn_out))
        
        return out


class ShiftedAxisWindowBlock(nn.Module):
    """
    Shifted axis-window encoder nhẹ cho 18 visual tokens.
    Xem VGG 3x3 và ResNet 3x3 như một grid 3x6:
    [VGG row | ResNet row], attention theo hàng/cột rồi shifted attention.
    """
    def __init__(self, embed_dim, num_heads, grid_size=(3, 6), dropout=0.1):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.row_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.shift_row_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.shift_col_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.norm5 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.drop_path = DropPath(dropout if dropout > 0. else 0.)

    def _axis_attention(self, x, row_attn, col_attn):
        B, N, D = x.shape
        grid = x.view(B, self.grid_h, self.grid_w, D)

        row_tokens = grid.reshape(B * self.grid_h, self.grid_w, D)
        row_out, _ = row_attn(row_tokens, row_tokens, row_tokens)
        row_out = row_out.view(B, self.grid_h, self.grid_w, D).reshape(B, N, D)

        grid = x.view(B, self.grid_h, self.grid_w, D).transpose(1, 2).contiguous()
        col_tokens = grid.reshape(B * self.grid_w, self.grid_h, D)
        col_out, _ = col_attn(col_tokens, col_tokens, col_tokens)
        col_out = col_out.view(B, self.grid_w, self.grid_h, D).transpose(1, 2).contiguous().reshape(B, N, D)
        return row_out, col_out

    def forward(self, x):
        B, N, D = x.shape
        expected_tokens = self.grid_h * self.grid_w
        if N != expected_tokens:
            raise ValueError(f"ShiftedAxisWindowBlock expects {expected_tokens} tokens arranged as a {self.grid_h}x{self.grid_w} grid.")

        row_out, col_out = self._axis_attention(x, self.row_attn, self.col_attn)
        x = self.norm1(x + self.drop_path(row_out))
        x = self.norm2(x + self.drop_path(col_out))

        shift_h = max(self.grid_h // 2, 1)
        shift_w = max(self.grid_w // 2, 1)
        shifted = x.view(B, self.grid_h, self.grid_w, D)
        shifted = torch.roll(shifted, shifts=(-shift_h, -shift_w), dims=(1, 2)).reshape(B, N, D)

        shift_row_out, shift_col_out = self._axis_attention(shifted, self.shift_row_attn, self.shift_col_attn)
        shifted = self.norm3(shifted + self.drop_path(shift_row_out))
        shifted = self.norm4(shifted + self.drop_path(shift_col_out))

        x = shifted.view(B, self.grid_h, self.grid_w, D)
        x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2)).reshape(B, N, D)

        ffn_out = self.ffn(x)
        x = self.norm5(x + self.drop_path(ffn_out))
        return x


class ShiftedAxisWindowEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, grid_size=(3, 6), dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(*[
            ShiftedAxisWindowBlock(embed_dim=embed_dim, num_heads=num_heads, grid_size=grid_size, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        return self.layers(x)


# =====================================================================
# 4. Model chính: RegionAlignedFER
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

        # ===== 2. Region Tokens =====
        self.cross_attention_direction = model_cfg.get('cross_attention_direction', 'region_query')
        self.use_clip_dictionary = model_cfg.get('use_clip_dictionary', False)
        if self.use_clip_dictionary:
            clip_model_name = model_cfg.get('clip_model_name', "openai/clip-vit-base-patch32")
            self.region_dict = CLIPFacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim,
                clip_model_name=clip_model_name
            )
        else:
            self.region_dict = FacialRegionDictionary(
                num_regions=self.num_regions,
                embed_dim=self.embed_dim
            )

        # ===== 3. Semantic-Visual Alignment =====
        if self.cross_attention_direction == 'visual_query':
            self.alignment = VisualPatchRegionAlignment(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate
            )
            print("--> Cross-attention direction: visual patch tokens query region tokens")
        else:
            self.alignment = SemanticVisualAlignment(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate
            )
            print("--> Cross-attention direction: region tokens query visual patch tokens")
        
        # Positional Encoding Cố Định 2D (Sin-Cos) cho lưới 3x3 tokens
        # Sinh 1 ma trận tọa độ, xài chung hệ quy chiếu cho cả VGG và ResNet
        sincos = get_2d_sincos_pos_embed(self.embed_dim, grid_size=3).unsqueeze(0)  # [1, 9, 512]
        self.register_buffer('grid_pos_embed', sincos)

        # Type Embeddings để phân biệt VGG và ResNet khi nối lại
        self.vgg_type_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.res_type_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        visual_pos_embed = torch.cat([sincos, sincos], dim=1)  # [1, 18, 512]
        self.register_buffer('visual_pos_embed', visual_pos_embed)

        # ===== 4. Hyper-visual Representation =====
        # Pool visual features → single vector, rồi broadcast cộng vào Φ_sem
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(self.dropout_rate) # Add dropout here
        )

        # ===== 5. Transformer / SubGraph Encoder =====
        self.fusion_type = model_cfg.get('fusion_type', 'transformer')
        if self.cross_attention_direction == 'visual_query' and self.fusion_type == 'subgraph':
            raise ValueError("fusion_type='subgraph' expects 6 region tokens, but visual_query produces 18 visual tokens.")
        
        if self.fusion_type == 'subgraph':
            self.transformer_encoder = nn.Sequential(*[
                SubGraphFusion(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout_rate)
                for _ in range(self.num_layers)
            ])
            print("--> Loaded SubGraph Fusion Architecture (Upper/Lower Face decoupled)")
        elif self.fusion_type == 'swin':
            swin_grid_size = (3, 6) if self.cross_attention_direction == 'visual_query' else (2, 3)
            self.transformer_encoder = ShiftedAxisWindowEncoder(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                grid_size=swin_grid_size,
                dropout=self.dropout_rate
            )
            print(f"--> Loaded Shifted Axis-window Architecture ({swin_grid_size[0]}x{swin_grid_size[1]} visual grid)")
        else:
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
            print("--> Loaded Standard Transformer Architecture")

        # Positional Encoding cho region tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_regions, self.embed_dim) * 0.02
        )
        self.visual_residual_norm = nn.LayerNorm(self.embed_dim)

        # ===== 6. Classification Head =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.3), # Lowered from 0.5
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2), # Lowered from 0.3
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

        # Áp dụng chung ma trận vị trí tĩnh (Sin-Cos) cộng thêm Type Embedding cho từng backbone
        vgg_feat = vgg_feat + self.grid_pos_embed + self.vgg_type_embed
        res_feat = res_feat + self.grid_pos_embed + self.res_type_embed

        # Φ_visual: nối đặc trưng từ cả hai backbone
        visual_features = torch.cat([vgg_feat, res_feat], dim=1)  # [B, 18, 512]

        # ── 2. Region Tokens ──
        region_tokens = self.region_dict(B)      # [B, 6, 512]

        # ── 3. Semantic-Visual Alignment (Cross-Attention) ──
        if self.cross_attention_direction == 'visual_query':
            # Q = visual patch tokens, K/V = region tokens
            phi_sem, attn_weights = self.alignment(
                visual_features, region_tokens
            )                                    # [B, 18, 512], [B, 18, 6]
        else:
            # Q = region tokens, K/V = visual patch tokens
            phi_sem, attn_weights = self.alignment(
                region_tokens, visual_features
            )                                    # [B, 6, 512], [B, 6, 18]

        # ── 4. Hyper-visual Representation ──
        # Pool toàn bộ visual features → 1 vector, broadcast cộng vào Φ_sem
        phi_visual = visual_features.mean(dim=1, keepdim=True)  # [B, 1, 512]
        phi_visual = self.visual_proj(phi_visual)               # [B, 1, 512]
        hyper_visual = phi_sem + phi_visual                     # [B, 6/18, 512]

        # ── 5. Transformer/Swin Encoder ──
        if self.cross_attention_direction == 'visual_query':
            hyper_visual = hyper_visual + self.visual_pos_embed # [B, 18, 512]
        else:
            hyper_visual = hyper_visual + self.pos_embed        # [B, 6, 512]
        encoded = self.transformer_encoder(hyper_visual)        # [B, 6/18, 512]
        if self.cross_attention_direction == 'visual_query':
            encoded = self.visual_residual_norm(encoded + visual_features)

        # ── 6. Classification ──
        pooled = encoded.mean(dim=1)             # [B, 512]
        logits = self.classifier(pooled)         # [B, num_classes]

        # ── 7. Orthogonal Loss (Tránh việc tập trung trùng vùng) ──
        if self.cross_attention_direction == 'visual_query':
            region_attn = attn_weights.transpose(1, 2)  # [B, 6, 18]
        else:
            region_attn = attn_weights                 # [B, 6, 18]

        attn_norm = F.normalize(region_attn, p=2, dim=-1)
        sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        mask = torch.eye(sim.size(1), device=sim.device).bool()
        off_diag_sim = sim[:, ~mask]
        ortho_loss = off_diag_sim.mean()

        if self.training:
            return logits, ortho_loss
            
        if hasattr(self, 'return_attn') and self.return_attn:
            if self.cross_attention_direction == 'visual_query':
                return logits, attn_weights.transpose(1, 2)
            return logits, attn_weights
            
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
    
    if isinstance(out, tuple):
        logits, ortho_loss = out
        print(f"Logits shape: {logits.shape}")  # Expected: [2, 7]
        print(f"Orthogonal Loss: {ortho_loss.item():.4f}")
        assert logits.shape == (2, 7)
    else:
        print(f"Output shape: {out.shape}")  # Expected: [2, 7]
        assert out.shape == (2, 7)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nTest Passed!")
