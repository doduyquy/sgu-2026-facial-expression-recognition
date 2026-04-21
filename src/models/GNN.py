import torch
import torch.nn as nn
import torch.nn.functional as F


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

#graph layer

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer đơn giản:
        H' = A_hat * H * W
    A_hat được tính bằng Cosine Similarity giữa các node (soft adjacency).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        nodes: [B, T, in_dim]
        Returns: [B, T, out_dim]
        """
        # --- Xây dựng adjacency mềm bằng cosine similarity ---
        nodes_norm = F.normalize(nodes, p=2, dim=-1)           # [B, T, D] # Chuẩn hóa vector
        A = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # [B, T, T] # Tính ma trận tương đồng
        
        # Thêm nhiệt độ tau để làm sắc nét sự tập trung (tránh hiện tượng mọi node bị trộn đều thành 1)
        tau = 0.05
        A = F.softmax(A / tau, dim=-1)                         # Row-stochastic # Softmax để tính trọng số

        # --- Kết tập thông tin từ hàng xóm + Linear ---
        agg = torch.bmm(A, nodes)                              # [B, T, in_dim] # Kết tập thông tin
        out = self.linear(agg)                                 # [B, T, out_dim] # Linear để tăng chiều dữ liệu
        
        # Thêm residual connection (Bảo toàn đặc trưng gốc của node)
        out = self.norm(nodes + F.relu(out))
        return out


class SubGraphPooling(nn.Module):
    """
    Gom K node → S subgraph (tương tự DiffPool nhưng nhẹ hơn).
    Học ma trận gán S = softmax(MLP(nodes)) để assign mỗi node
    về 1 trong num_subgraphs cluster đại diện.

    Đồng thời trả về entropy regularization loss:
        L_ent = mean( sum_k{ -S_ik * log(S_ik) } )
    để buộc assignment sắc nét hơn.
    """
    def __init__(self, in_dim: int, num_subgraphs: int):
        super().__init__()
        self.num_subgraphs = num_subgraphs
        self.assign_net = nn.Sequential(
            nn.BatchNorm1d(in_dim), # Ổn định đặc trưng trước khi phân cụm
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_subgraphs)
        )

    def forward(self, nodes: torch.Tensor):
        """
        nodes: [B, T, D]
        Returns:
            subgraphs: [B, num_subgraphs, D]  — feature của mỗi subgraph
            pool_loss: scalar tensor          — entropy regularization
        """
        B, T, D = nodes.shape
        # Batch norm yêu cầu [B, D, T]
        nodes_bn = nodes.transpose(1, 2)
        nodes_bn = self.assign_net[0](nodes_bn)
        nodes_bn = nodes_bn.transpose(1, 2)
        
        # Ma trận gán [B, T, K]
        S = self.assign_net[1:](nodes_bn)
        S = F.softmax(S, dim=-1)

        # Pool: [B, K, T] x [B, T, D] = [B, K, D]
        subgraphs_sum = torch.bmm(S.transpose(1, 2), nodes)
        
        # Chia cho kích thước của subgraph (MEAN pooling) để chuẩn hoá độ lớn feature
        cluster_sizes = S.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8 # [B, K, 1]
        subgraphs = subgraphs_sum / cluster_sizes

        # Entropy loss: khuyến khích assignment dứt khoát
        eps = 1e-8
        entropy = -(S * (S + eps).log()).sum(dim=-1)  # [B, T]
        pool_loss = entropy.mean()

        return subgraphs, pool_loss


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

        # ── SubGraph Pooling ──
        self.subgraph_pool = SubGraphPooling(self.hidden_dim, self.num_subgraphs)

        # ── Inter-subgraph Self-Attention ──
        self.subgraph_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=model_cfg.get('num_heads', 4),
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

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
    def _encode_graph(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        nodes: [B, T, patch_dim]
        Returns: [B, T, hidden_dim]
        """
        h = self.input_proj(nodes)   # [B, T, hidden_dim]
        h = self.gcn1(h)             # [B, T, hidden_dim]
        h = self.gcn2(h)             # [B, T, hidden_dim]
        return h

    # ----------------------------------------------------------
    # Step 3: SubGraph pooling
    # ----------------------------------------------------------
    def _pool_subgraphs(self, h: torch.Tensor):
        """
        h: [B, T, hidden_dim]
        Returns:
            subgraphs: [B, num_subgraphs, hidden_dim]
            pool_loss: scalar
        """
        return self.subgraph_pool(h)

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
        h = self._encode_graph(nodes)              # [B, T, hidden_dim]

        # ── 3. SubGraph Pooling ──
        subgraphs, pool_loss = self._pool_subgraphs(h)  # [B, S, hidden_dim]

        # ── 4. Inter-SubGraph Self-Attention ──
        sg_attn, _ = self.subgraph_attn(subgraphs, subgraphs, subgraphs)
        subgraphs = self.attn_norm(subgraphs + sg_attn)  # residual

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
