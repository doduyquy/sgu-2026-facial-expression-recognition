import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientB0(nn.Module):
    def __init__(self, config=None, pretrained=True):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: [B,1,48,48]

        # 1. grayscale → 3 channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # 2. resize → 224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. feature map
        local_feat = self.features(x)         # [B,1280,7,7]

        # 4. pooling
        global_feat = self.pool(local_feat)             # [B,1280,1,1]

        # 5. flatten
        global_feat = torch.flatten(global_feat, 1)      # [B,1280]
        local_feat = local_feat.flatten(2).transpose(1, 2)   # [B, 49, 1280]
        return global_feat, local_feat
        #output global_feat: [B,1280], local_feat: [B,49,1280]

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, adj):
        # x: [B, N, C_in] n là số nodes
        # adj: [B, N, N]: mức độ kết nối giữa node i và j

        # [B, N, N] @ [B, N, C] → [B, N, C]
        support = torch.bmm(adj, x)  # [B, N, C_in]

        # linear transform
        output = self.linear(support)  # [B, N, C_out]
        return output

class MotifGNN(nn.Module):
    def __init__(self, in_channels=1280, hidden_dim=512, num_classes=7):
        super().__init__()
        
        self.gcn1 = GCNLayer(in_channels, hidden_dim) #hoc patern cuc bo [mouth → chỉ biết miệng]
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim) #hoc nhung vung xa [mouth → biết miệng + mắt]
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def precompute_spatial_A(self, H, W, sigma=2.0): #xác định node nào gần nhau trên mặt thì nên kết nối mạnh hơn
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        coords = torch.stack([y.flatten(), x.flatten()], dim=1).float() # [9, 2]
        dist_sq = torch.cdist(coords, coords, p=2).pow(2)
        A = torch.exp(-dist_sq / (2 * sigma**2))
        A = A / A.sum(dim=-1, keepdim=True)
        return A
        
    def forward(self, local_feat):
        # local_feat: [B, 9, 1280]
        
        # 1. Tính Feature-based Adjacency (Các vùng có đặc trưng nhân diện giống nhau sẽ liên kết)
        import torch.nn.functional as F
        nodes_norm = F.normalize(local_feat, p=2, dim=-1)
        logits = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2)) / 0.1
        A_feat = F.softmax(logits, dim=-1)
        
        # 2. Kết hợp Spatial Adjacency (cố định vị trí trên mặt) và Feature Adjacency
        A = 0.5 * self.A_spatial.unsqueeze(0) + 0.5 * A_feat
        
        # 3. Truyền qua GCN
        h = self.gcn1(local_feat, A) # [B, 9, 512]
        h = F.relu(h)
        h = self.gcn2(h, A) # [B, 9, 512]
        h = F.relu(h)
        
        # 4. Global Pooling: Gộp 9 vùng mặt thành 1 vector duy nhất cho toàn ảnh
        graph_repr = h.mean(dim=1) # [B, 512]
        
        # 5. Phân loại
        logits = self.classifier(graph_repr)
        return logits

class Hybrid_CNN_GNN(nn.Module):
    def __init__(self, channels=1, num_classes=7):
        super().__init__()
        # Backbone trích xuất bộ tính năng (trả về global và local)
        self.backbone = EfficientB0(channels=channels)
        
        # GNN học sự tương tác biểu cảm giữa các vùng mặt
        # Do EfficientNet-B0 xuất ra tensor 1280 kênh, nên in_channels = 1280
        self.gnn = MotifGNN(in_channels=1280, hidden_dim=512, num_classes=num_classes)
        
    def forward(self, x):
        # Input ảnh FER2013: [B, 1, 48, 48]
        global_feat, local_feat = self.backbone(x)
        
        # Đưa thông tin phân mảnh (9 vùng cục bộ trên mặt) vào GNN để phân tích chuyên sâu
        logits = self.gnn(local_feat) # [B, num_classes]
        return logits
