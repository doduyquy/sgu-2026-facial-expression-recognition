import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAwareAdapter(nn.Module):
    """
    Expression-aware Adapter (EAA) phiên bản dành cho CNN.
    Sử dụng 1x1 Convolution để tinh chỉnh feature map mà không làm thay đổi kích thước không gian.
    Cấu trúc: BatchNorm -> 1x1 Conv (Down) -> ReLU -> 1x1 Conv (Up) + Residual
    """
    def __init__(self, in_channels, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)

class InstanceEnhancedClassifier(nn.Module):
    """
    Instance-enhanced Expression Classifier (IEC) cho CNN.
    Sử dụng learnable prototypes và tinh chỉnh chúng dựa trên vector đặc trưng toàn cục.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Learnable Prototypes (Textual Embeddings)
        self.textual_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.normal_(self.textual_embeddings, std=0.01)

        # Gamma Net: Học trọng số hòa trộn từ visual instance
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        """
        v: Visual Embedding sau khi Global Pooling [B, D]
        """
        v_norm = F.normalize(v, p=2, dim=-1) # [B, D]
        w_norm = F.normalize(self.textual_embeddings, p=2, dim=-1) # [K, D]

        gamma = self.gamma_net(v).unsqueeze(-1) # [B, 1, 1]

        # Mở rộng để tính toán
        w_expanded = w_norm.unsqueeze(0).expand(v.size(0), -1, -1) # [B, K, D]
        v_expanded = v_norm.unsqueeze(1).expand(-1, self.num_classes, -1) # [B, K, D]

        # Tinh chỉnh Prototypes (Slerp approximation)
        w_tilde = (1 - gamma) * w_expanded + gamma * v_expanded 
        w_tilde = F.normalize(w_tilde, p=2, dim=-1) # [B, K, D]

        # Tính Similarity
        logits = torch.bmm(v_norm.unsqueeze(1), w_tilde.transpose(1, 2)) # [B, 1, K]
        
        return logits.squeeze(1) # [B, K]

class VGGEA_CNN(nn.Module):
    """
    Mô hình VGG + EA (EAA & IEC) - KHÔNG DÙNG TRANSFORMER.
    Tối ưu cho ảnh độ phân giải thấp (48x48) và dữ liệu FER2013.
    """
    def __init__(self, config, channels=1):
        super().__init__()
        self.num_classes = config['data']['num_classes']
        self.dropout_block = config['model'].get('dropout_block', 0.3)
        self.embed_dim = 512 # Kênh cuối cùng của VGG

        # --- Backbone VGG ---
        # Block 1: 48x48 -> 24x24
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 2: 24x24 -> 12x12
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 3: 12x12 -> 6x6
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_block)
        )

        # Block 4: 6x6 -> 3x3
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(self.dropout_block)
        )

        # --- Emotion-Aware Adapter (EAA) ---
        # Áp dụng trực tiếp trên feature map cuối cùng
        self.eaa = CNNAwareAdapter(512, bottleneck_dim=128)

        # --- Global Pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- Instance-enhanced Classifier (IEC) ---
        self.iec = InstanceEnhancedClassifier(self.embed_dim, self.num_classes)

    def forward(self, x):
        # 1. Feature Extraction
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x) # [B, 512, 3, 3]

        # 2. Expression-aware Adaptation (EAA)
        x = self.eaa(x)

        # 3. Global Pooling
        v = self.gap(x) # [B, 512, 1, 1]
        v = torch.flatten(v, 1) # [B, 512]

        # 4. IEC Classification
        logits = self.iec(v)

        return logits

if __name__ == "__main__":
    print("=== Testing VGGEA_CNN (No Transformer) ===")
    config = {
        'data': {
            'num_classes': 7
        },
        'model': {
            'dropout_block': 0.3
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGGEA_CNN(config, channels=1).to(device)
    
    dummy_input = torch.randn(2, 1, 48, 48).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 7)
    print("Forward Pass Successful!")
