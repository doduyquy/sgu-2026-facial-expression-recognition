import torch
import torch.nn as nn
import torch.nn.functional as F
from .region_attention import CLIPFacialRegionDictionary


def build_graph_feature_map(images: torch.Tensor) -> torch.Tensor:
    """
    Chuyển ảnh thô thành feature map 3 kênh [x, y, intensity]. Trobg đó xy là tọa độ, còn intensity là độ x/255
    
    Args:
        images: [B, 1, H, W], intensity in [0, 1]
    Returns:
        feat_map: [B, 3, H, W]
            channel 0 = normalized row coordinate (x)
            channel 1 = normalized col coordinate (y)
            channel 2 = pixel intensity
    """
    B, C, H, W = images.shape # B là batch size, C là số kênh, H là chiều cao, W là chiều rộng
    device = images.device

    xs = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W) # Tạo ra ma trận tọa độ x
    ys = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W) # Tạo ra ma trận tọa độ y

    xs = xs.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]  
    ys = ys.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]

    feat_map = torch.cat([xs, ys, images], dim=1)          # [B, 3, H, W] # Ghép 3 kênh lại với nhau
    return feat_map


#Graph --> SubGraph
def extract_patch_nodes(feat_map: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Trích xuất các node từ feature map bằng cửa sổ trượt.
    Mỗi patch = 1 node trên đồ thị.

    Args:
        feat_map: [B, 3, H, W]
        window_size: kích thước cửa sổ (ví dụ: 5)
        stride:      bước nhảy (ví dụ: 2)
    Returns:
        nodes: [B, T, patch_dim]  (T = số patch, patch_dim = ws*ws*3)
    """
    B, C, H, W = feat_map.shape
    ws = window_size

    # F.unfold: [B, C*ws*ws, T] # T là số patch
    unfolded = F.unfold(feat_map, kernel_size=ws, stride=stride)
    T = unfolded.shape[-1]

    # [B, T, C*ws*ws]
    nodes = unfolded.transpose(1, 2).contiguous()
    return nodes  # [B, T, ws*ws*3]

def precompute_spatial_A(H_img: int, W_img: int, ws: int, stride: int, sigma: float = 2.0, device='cpu') -> torch.Tensor:
    """
    Tính trước ma trận kề không gian cố định dựa trên Gaussian Kernel.
    """
    H_grid = (H_img - ws) // stride + 1
    W_grid = (W_img - ws) // stride + 1
    N = H_grid * W_grid
    
    # Tạo toạ độ lưới cho từng patch
    y, x = torch.meshgrid(torch.arange(H_grid, device=device), torch.arange(W_grid, device=device), indexing='ij')
    coords = torch.stack([y.flatten(), x.flatten()], dim=1).float() # [N, 2]
    
    # Tính bình phương khoảng cách Euclidean và áp dụng Gaussian Kernel
    dist_sq = torch.cdist(coords, coords, p=2).pow(2)
    A_spatial = torch.exp(-dist_sq / (2 * sigma**2))
    
    # Row-normalize để thành ma trận xác suất chuyển trạng thái (stochastic)
    A_spatial = A_spatial / A_spatial.sum(dim=-1, keepdim=True)
    return A_spatial # [N, N]

#graph layer

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer nhận ma trận A từ ngoài truyền vào.
    H' = Norm(H + ReLU(A * H * W))
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, nodes: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        nodes: [B, T, in_dim]
        A: [B, T, T] (hoặc [T, T] nếu xài chung broadcast được)
        Returns: [B, T, out_dim]
        """
        # --- Kết tập thông tin từ hàng xóm + Linear ---
        agg = torch.matmul(A, nodes)                           # [B, T, in_dim]
        out = self.linear(agg)                                 # [B, T, out_dim]
        
        # Thêm residual connection (Bảo toàn đặc trưng gốc của node)
        out = self.norm(nodes + F.relu(out))
        return out


class SubGraphPooling(nn.Module):
    """
    Semantic-Guided Hierarchical Gom cụm (Option C).
    Sử dụng CLIP Text Embeddings làm Mỏ Neo (Anchors) để kéo Patch vào đúng cụm ngữ nghĩa mặt người (Mắt, Mũi, Miệng...).
    """
    def __init__(self, in_dim: int, num_subgraphs: int, tau: float = 0.05, align_loss_weight: float = 0.1):
        super().__init__()
        self.num_subgraphs = num_subgraphs
        self.tau = tau
        self.align_loss_weight = align_loss_weight
        
        # Mỏ neo ngôn ngữ từ CLIP
        self.region_dict = CLIPFacialRegionDictionary(num_regions=num_subgraphs, embed_dim=in_dim)

    def forward(self, nodes: torch.Tensor, A: torch.Tensor):
        B, T, D = nodes.shape
        
        # [B, K, D] - Lấy Semantic Embeddings (Vector Chữ)
        keys = self.region_dict(B) 
        
        # Tính ma trận gán cụm S bằng Semantic Cosine Similarity
        nodes_norm = F.normalize(nodes, p=2, dim=-1)
        keys_norm  = F.normalize(keys, p=2, dim=-1)
        
        # S: [B, T, K] - Soft-assignment dựa trên Semantic Alignment
        logits = torch.bmm(nodes_norm, keys_norm.transpose(1, 2)) / self.tau
        S = F.softmax(logits, dim=-1)

        # Pool Đặc trưng (Feature Pooling dựa trên S): [B, K, D]
        subgraphs_sum = torch.bmm(S.transpose(1, 2), nodes)
        cluster_sizes = S.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8
        subgraphs = subgraphs_sum / cluster_sizes

        # Pool Topology (S^T A S) -> [B, K, K]
        pooled_A = torch.bmm(S.transpose(1, 2), torch.bmm(A, S))
        pooled_A = F.normalize(pooled_A, p=1, dim=-1)

        # Mất mát 1: Entropy Loss (khuyến khích cụm sắc nét)
        eps = 1e-8
        entropy = -(S * (S + eps).log()).sum(dim=-1).mean()
        
        # Mất mát 2: Semantic Alignment Loss (Option C)
        # Ép vector trung bình của cụm hình ảnh phải hội tụ về đúng vector chữ của CLIP
        subgraphs_norm = F.normalize(subgraphs, p=2, dim=-1) # [B, K, D]
        # Khoảng cách Cosine chỉ lấy cặp thuận (Cụm K so với Text K)
        mean_sim = (subgraphs_norm * keys_norm).sum(dim=-1).mean()
        align_loss = 1.0 - mean_sim
        
        # Tổng hợp Loss
        pool_loss = entropy + self.align_loss_weight * align_loss

        return subgraphs, pooled_A, pool_loss


#MotifGNN


class MotifGNN(nn.Module):
    """
    Pipeline: Pixel → Graph Feature Map → Patch Nodes → GCN → SubGraph Pooling → Classifier

    Tương thích hoàn toàn với Trainer:
        - training: trả về (logits, pool_loss)  [scalar aux loss]
        - eval:     trả về logits
    """

    def __init__(self, config, channels: int = 1):
        super().__init__()
        model_cfg  = config.get('model', {})
        data_cfg   = config.get('data',  {})

        self.image_size    = data_cfg.get('image_size', 48)
        self.num_classes   = data_cfg.get('num_classes', 7)
        self.channels      = channels

        # Graph hyper-params
        self.window_size   = model_cfg.get('window_size', 5)
        self.stride        = model_cfg.get('stride', 2)
        self.hidden_dim    = model_cfg.get('hidden_dim', 128)
        self.num_subgraphs = model_cfg.get('num_subgraphs', 6)  # 6 vùng mặt
        self.dropout_rate  = model_cfg.get('dropout', 0.3)
        self.pool_loss_weight = model_cfg.get('pool_loss_weight', 0.01)
        self.alpha         = model_cfg.get('alpha', 0.5)
        self.tau           = model_cfg.get('tau', 0.05)
        sigma = model_cfg.get('spatial_sigma', 2.0)

        # Precompute Spatial Adjacency Matrix
        self.register_buffer(
            'A_spatial',
            precompute_spatial_A(self.image_size, self.image_size, self.window_size, self.stride, sigma)
        )

        # patch_dim = window_size * window_size * 3  (x, y, intensity)
        in_channels = 3  # luôn là 3 vì ta tạo feat_map [x, y, intensity]
        patch_dim   = self.window_size * self.window_size * in_channels  # 5*5*3 = 75

        # ── Input projection: patch_dim → hidden_dim ──
        self.input_proj = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True)
        )

        # ── GCN Layers ──
        self.gcn1 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.gcn2 = GCNLayer(self.hidden_dim, self.hidden_dim)

        # ── Semantic-Guided SubGraph Pooling ──
        align_weight = model_cfg.get('align_loss_weight', 0.1)
        self.subgraph_pool = SubGraphPooling(self.hidden_dim, self.num_subgraphs, tau=self.tau, align_loss_weight=align_weight)

        # ── Inter-subgraph GCN (thay thế Self-Attention) ──
        self.subgraph_gcn = GCNLayer(self.hidden_dim, self.hidden_dim)

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

        print(f"--> MotifGNN | window={self.window_size} stride={self.stride} "
              f"hidden={self.hidden_dim} subgraphs={self.num_subgraphs}")

    # ----------------------------------------------------------
    # Step 1: Pixel → Graph nodes
    # ----------------------------------------------------------
    def _pixel_to_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, channels, H, W]
        Returns nodes: [B, T, patch_dim]
        """
        feat_map = build_graph_feature_map(x)               # [B, 3, H, W]
        nodes    = extract_patch_nodes(feat_map, self.window_size, self.stride)  # [B, T, 75]
        
        # Tách tọa độ và cường độ để chuẩn bừa độc lập
        # 0:25 (x), 25:50 (y), 50:75 (intensity)
        ws2 = self.window_size * self.window_size
        coords = nodes[:, :, :2*ws2]
        intens = nodes[:, :, 2*ws2:]
        
        # Chuẩn hóa cường độ (L2 nhắm vào tương phản patch)
        intens = F.normalize(intens, p=2, dim=-1)
        
        # Ghép lại - tọa độ giữ nguyên [0, 1] để GCN biết vị trí tương đối
        nodes = torch.cat([coords, intens], dim=-1)
        return nodes

    # ----------------------------------------------------------
    # Step 2: GCN encoding
    # ----------------------------------------------------------
    def _encode_graph(self, nodes: torch.Tensor):
        """
        nodes: [B, T, patch_dim]
        Returns: 
           h: [B, T, hidden_dim]
           A: [B, T, T] — ma trận kề cấp pixel (layer cuối của block này)
        """
        h = self.input_proj(nodes)   # [B, T, hidden_dim]
        
        # Lớp GCN 1
        nodes_norm = F.normalize(h, p=2, dim=-1)
        A_feat1 = F.softmax(torch.bmm(nodes_norm, nodes_norm.transpose(1, 2)) / self.tau, dim=-1)
        A1 = self.alpha * self.A_spatial.unsqueeze(0) + (1 - self.alpha) * A_feat1
        h = self.gcn1(h, A1)             # [B, T, hidden_dim]
        
        # Lớp GCN 2
        nodes_norm2 = F.normalize(h, p=2, dim=-1)
        A_feat2 = F.softmax(torch.bmm(nodes_norm2, nodes_norm2.transpose(1, 2)) / self.tau, dim=-1)
        A2 = self.alpha * self.A_spatial.unsqueeze(0) + (1 - self.alpha) * A_feat2
        h = self.gcn2(h, A2)             # [B, T, hidden_dim]
        
        return h, A2

    # ----------------------------------------------------------
    # Step 3: SubGraph pooling
    # ----------------------------------------------------------
    def _pool_subgraphs(self, h: torch.Tensor, A: torch.Tensor):
        """
        h: [B, T, hidden_dim]
        A: [B, T, T]
        Returns:
            subgraphs: [B, num_subgraphs, hidden_dim]
            pooled_A: [B, num_subgraphs, num_subgraphs]
            pool_loss: scalar
        """
        return self.subgraph_pool(h, A)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, channels, H, W]
        Returns (training):
            (logits [B, C], pool_loss scalar)
        Returns (eval):
            logits [B, C]
        """
        # ── 1. Pixel → Nodes ──
        nodes = self._pixel_to_nodes(x)            # [B, T, patch_dim]

        # ── 2. GCN ──
        h, A = self._encode_graph(nodes)           # [B, T, hidden_dim], [B, T, T]

        # ── 3. SubGraph Pooling (Hierarchical Graph) ──
        subgraphs, pooled_A, pool_loss = self._pool_subgraphs(h, A)  # [B, K, D], [B, K, K]

        # ── 4. Inter-SubGraph GCN ──
        subgraphs = self.subgraph_gcn(subgraphs, pooled_A) # [B, K, D]

        # ── 5. Global pooling over subgraphs ──
        graph_repr = subgraphs.mean(dim=1)         # [B, hidden_dim]

        # ── 6. Classification ──
        logits = self.classifier(graph_repr)       # [B, num_classes]

        if self.training:
            return logits, pool_loss * self.pool_loss_weight

        return logits


# =========================================================
# Test nhanh (python -m src.models.GNN)
# =========================================================
if __name__ == "__main__":
    print("=== Testing MotifGNN ===")
    config = {
        'data': {'num_classes': 7, 'image_size': 48, 'channels': 1},
        'model': {
            'window_size': 5,
            'stride': 2,
            'hidden_dim': 128,
            'num_subgraphs': 6,
            'num_heads': 4,
            'dropout': 0.3,
            'pool_loss_weight': 0.01,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MotifGNN(config, channels=1).to(device)

    # Training mode
    model.train()
    dummy = torch.randn(4, 1, 48, 48).to(device)
    logits, aux = model(dummy)
    print(f"[train] logits: {logits.shape}, pool_loss: {aux.item():.4f}")
    assert logits.shape == (4, 7)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"[eval]  output: {out.shape}")
    assert out.shape == (4, 7)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Test Passed!")
