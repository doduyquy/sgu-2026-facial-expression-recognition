import torch
import torch.nn as nn
import math
from .vgg import VGGFusionSpatialCNN
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop1(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.drop2(ff_out)       
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.shape[1]
        return x + self.pe[:, :T, :]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, max_len=10, dropout=0.1):
        super().__init__()
        # self.position_encoding = SinusoidalPositionalEncoding(embed_dim, max_len) # V1 style
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, dropout=dropout) 
            for _ in range(num_layers) 
        ])

    def forward(self, x):
        # x = self.position_encoding(x) # Chuyển việc quản lý PE ra ngoài để V2 dùng Learned PE
        for layer in self.layers:
            x = layer(x)
        return x

class VGGFusionTransformer(nn.Module):
    def __init__(self, config, channels=1):
        super().__init__()
        # Load hyperparameters from config
        self.embed_dim = config['model'].get('embed_dim', 512)
        self.num_heads = config['model'].get('num_heads', 8)
        self.num_layers = config['model'].get('num_layers', 4)
        self.dropout = config['model'].get('transformer_dropout', 0.1)
        
        self.vgg = VGGFusionSpatialCNN(config, channels)
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            num_layers=self.num_layers, 
            max_len=10,
            dropout=self.dropout
        )
        self.fc = nn.Linear(self.embed_dim, config['data']['num_classes'])

    def forward(self, x):
        x = self.vgg(x)          # [B, 9, 512]
        
        # Thêm Positional Encoding Sinusoidal (bản cũ)
        # Giả sử ta vẫn dùng Sinusoidal cho V1, ta khởi tạo PE trong __init__ và cộng ở đây
        if not hasattr(self, 'pe_layer'):
            self.pe_layer = SinusoidalPositionalEncoding(self.embed_dim, max_len=10).to(x.device)
        
        x = self.pe_layer(x)
        x = self.transformer(x)  # [B, 9, 512]
        
        # Pooling: Lấy trung bình cộng của 9 tokens để về 1 vector đặc trưng 512 chiều
        x = x.mean(dim=1)        # [B, 512]
        
        x = self.fc(x)           # [B, num_classes]
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10):
        super().__init__()
        # Ma trận PE có thể học được
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim))

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]

class VGGFusionTransformerV2(nn.Module):
    """
    Bản V2: dùng CLS Token + Learned Positional Embedding (Kiến trúc ViT chuẩn)
    """
    def __init__(self, config, channels=1):
        super().__init__()
        self.embed_dim = config['model'].get('embed_dim', 512)
        self.num_heads = config['model'].get('num_heads', 8)
        self.num_layers = config['model'].get('num_layers', 2)
        self.dropout = config['model'].get('transformer_dropout', 0.1)
        
        self.vgg = VGGFusionSpatialCNN(config, channels)
        
        # Token đại diện [CLS]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Learned PE cho 1 (CLS) + 9 (Patches) = 10 tokens
        self.pos_embed = LearnedPositionalEncoding(self.embed_dim, max_len=10)
        
        self.transformer = TransformerEncoder(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            num_layers=self.num_layers, 
            max_len=10,
            dropout=self.dropout
        )
        
        self.fc = nn.Linear(self.embed_dim, config['data']['num_classes'])

    def forward(self, x):
        x = self.vgg(x)          # [B, 9, 512]
        B = x.shape[0]
        
        # 1. Thêm CLS Token vào đầu chuỗi
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 512]
        x = torch.cat((cls_tokens, x), dim=1)        # [B, 10, 512]
        
        # 2. Cộng Learned Positional Encoding
        x = self.pos_embed(x)                        # [B, 10, 512]
        
        # 3. Qua Transformer
        x = self.transformer(x)                      # [B, 10, 512]
        
        # 4. Chỉ lấy output của CLS Token (vị trí index 0) để phân loại
        x = x[:, 0]                                  # [B, 512]
        
        x = self.fc(x)                               # [B, num_classes]
        return x

if __name__ == "__main__":
    print("=== Testing VGGFusionTransformer V1 & V2 ===")
    config = {
        'data': {
            'num_classes': 7,
            'channels': 1,
            'image_size': 48
        },
        'model': {
            'embed_dim': 512,
            'dropout_dense': 0.5,
            'dropout_block': 0.3,
            'use_aux': False,
            'transformer_dropout': 0.1,
            'num_layers': 2
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_input = torch.randn(2, 1, 48, 48).to(device)

    print("\n--- Testing V1 (Global Average Pooling) ---")
    model_v1 = VGGFusionTransformer(config, channels=1).to(device)
    output_v1 = model_v1(dummy_input)
    print(f"V1 Output shape: {output_v1.shape}")
    
    print("\n--- Testing V2 (CLS Token + Learned PE) ---")
    model_v2 = VGGFusionTransformerV2(config, channels=1).to(device)
    output_v2 = model_v2(dummy_input)
    print(f"V2 Output shape: {output_v2.shape}")
    
    assert output_v1.shape == (2, 7)
    assert output_v2.shape == (2, 7)
    print("\nAll Tests Passed!")

