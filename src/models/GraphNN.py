import torch
import torch.nn as nn
import torch.nn.functional as F


def build_graph_feature_map(images: torch.Tensor) -> torch.Tensor:
    """
    Chuyển ảnh thô thành handcrafted feature map.
    Không dùng CNN; các kênh thêm vào là đặc trưng xử lý ảnh cố định:
        [x, y, intensity, grad_x, grad_y, grad_mag, local_mean, local_std]
    
    Args:
        images: [B, 1, H, W], thường đã Normalize(mean=0.5, std=0.5)
    Returns:
        feat_map: [B, 8, H, W]
    """
    B, C, H, W = images.shape # B là batch size, C là số kênh, H là chiều cao, W là chiều rộng
    device = images.device
    dtype = images.dtype

    # DataLoader đang normalize ảnh bằng mean=0.5, std=0.5 nên đưa về [0, 1].
    # Nếu sau này input đã ở [0, 1], clamp vẫn giữ đặc trưng ổn định.
    intensity = (images[:, :1] * 0.5 + 0.5).clamp(0.0, 1.0)

    xs = torch.linspace(0, 1, H, device=device, dtype=dtype).view(H, 1).expand(H, W) # Tạo ra ma trận tọa độ x
    ys = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, W).expand(H, W) # Tạo ra ma trận tọa độ y

    xs = xs.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]  
    ys = ys.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)  # [B, 1, H, W]

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(intensity, sobel_x, padding=1)
    grad_y = F.conv2d(intensity, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

    local_mean = F.avg_pool2d(intensity, kernel_size=5, stride=1, padding=2)
    local_sq_mean = F.avg_pool2d(intensity.pow(2), kernel_size=5, stride=1, padding=2)
    local_std = torch.sqrt((local_sq_mean - local_mean.pow(2)).clamp_min(0.0) + 1e-8)

    feat_map = torch.cat(
        [xs, ys, intensity, grad_x, grad_y, grad_mag, local_mean, local_std],
        dim=1
    )          # [B, 8, H, W] # Ghép handcrafted feature lại với nhau
    return feat_map


#Graph --> SubGraph
def extract_patch_nodes(feat_map: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Trích xuất các node từ feature map bằng cửa sổ trượt.
    Mỗi patch = 1 node trên đồ thị.

    Args:
        feat_map: [B, C, H, W]
        window_size: kích thước cửa sổ (ví dụ: 5)
        stride:      bước nhảy (ví dụ: 2)
    Returns:
        nodes: [B, T, patch_dim]  (T = số patch, patch_dim = ws*ws*C)
    """
    B, C, H, W = feat_map.shape
    ws = window_size

    # F.unfold: [B, C*ws*ws, T] # T là số patch
    unfolded = F.unfold(feat_map, kernel_size=ws, stride=stride)
    T = unfolded.shape[-1]

    # [B, T, C*ws*ws]
    nodes = unfolded.transpose(1, 2).contiguous()
    return nodes  # [B, T, ws*ws*C]


def precompute_spatial_A(
    H_img: int,
    W_img: int,
    ws: int,
    stride: int,
    sigma: float = 2.0,
    device='cpu'
) -> torch.Tensor:
    """
    Tính trước adjacency không gian cố định giữa các patch.
    Patch gần nhau trên lưới ảnh sẽ có trọng số kết nối lớn hơn.
    """
    H_grid = (H_img - ws) // stride + 1
    W_grid = (W_img - ws) // stride + 1

    y, x = torch.meshgrid(
        torch.arange(H_grid, device=device),
        torch.arange(W_grid, device=device),
        indexing='ij'
    )
    coords = torch.stack([y.flatten(), x.flatten()], dim=1).float()  # [T, 2]

    dist_sq = torch.cdist(coords, coords, p=2).pow(2)
    A_spatial = torch.exp(-dist_sq / (2 * sigma**2))
    A_spatial = A_spatial / (A_spatial.sum(dim=-1, keepdim=True) + 1e-8)
    return A_spatial  # [T, T]


def precompute_region_prior(
    H_img: int,
    W_img: int,
    ws: int,
    stride: int,
    num_regions: int = 6,
    sigma_y: float = 0.16,
    sigma_x: float = 0.18,
    device='cpu'
) -> torch.Tensor:
    """
    Tạo prior mềm cho 6 vùng mặt theo tọa độ patch:
        0: forehead, 1: left_eye, 2: right_eye, 3: nose, 4: mouth, 5: chin
    Đây không phải landmark detector; nó chỉ là bias hình học nhẹ để SubGraphPooling
    có xu hướng học các vùng mặt ổn định và dễ visualize hơn.
    """
    H_grid = (H_img - ws) // stride + 1
    W_grid = (W_img - ws) // stride + 1

    y, x = torch.meshgrid(
        torch.linspace(0, 1, H_grid, device=device),
        torch.linspace(0, 1, W_grid, device=device),
        indexing='ij'
    )
    coords = torch.stack([y.flatten(), x.flatten()], dim=-1)  # [T, 2]

    if num_regions == 6:
        anchors = torch.tensor(
            [
                [0.18, 0.50],  # forehead
                [0.38, 0.33],  # left eye
                [0.38, 0.67],  # right eye
                [0.55, 0.50],  # nose
                [0.73, 0.50],  # mouth
                [0.88, 0.50],  # chin
            ],
            device=device,
            dtype=coords.dtype
        )
    else:
        # Fallback nếu thử nghiệm num_subgraphs khác 6.
        anchors = torch.linspace(0.15, 0.85, num_regions, device=device).unsqueeze(1)
        anchors = torch.cat([anchors, torch.full_like(anchors, 0.5)], dim=1)

    dy = (coords[:, None, 0] - anchors[None, :, 0]) / sigma_y
    dx = (coords[:, None, 1] - anchors[None, :, 1]) / sigma_x
    prior = torch.exp(-0.5 * (dy.pow(2) + dx.pow(2)))  # [T, K]
    prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
    return prior

#graph layer

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer:
        H' = A_hat * H * W
    A_hat được truyền từ ngoài vào để hỗ trợ hybrid adjacency.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)
        

    def forward(self, nodes: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        nodes: [B, T, in_dim]
        A:     [B, T, T]
        Returns: [B, T, out_dim]
        """
        # --- Kết tập thông tin từ hàng xóm + Linear ---
        agg = torch.bmm(A, nodes)                              # [B, T, in_dim] # Kết tập thông tin
        out = self.linear(agg)                                 # [B, T, out_dim] # Linear để tăng chiều dữ liệu
        
        # Thêm residual connection (Bảo toàn đặc trưng gốc của node)
        out = self.norm(nodes + F.relu(out))
        return out


class SubGraphPooling(nn.Module):
    """
    Gom T node → K vùng mặt bằng region-guided soft assignment.
    MLP học assignment từ feature node, còn region_prior đóng vai trò bias
    để các cụm bám theo trán, mắt trái/phải, mũi, miệng, cằm.

    pool_loss gồm:
        - entropy: khuyến khích assignment sắc nét
        - prior_ce: giữ assignment không trôi quá xa khỏi prior vùng mặt
        - balance: tránh mọi patch đổ vào 1-2 subgraph
    """
    def __init__(
        self,
        in_dim: int,
        num_subgraphs: int,
        prior_strength: float = 1.0,
        prior_loss_weight: float = 0.05,
        balance_loss_weight: float = 0.1
    ):
        super().__init__()
        self.num_subgraphs = num_subgraphs
        self.prior_strength = prior_strength
        self.prior_loss_weight = prior_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.assign_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_subgraphs)
        )

    def forward(self, nodes: torch.Tensor, region_prior: torch.Tensor = None):
        """
        nodes: [B, T, D]
        region_prior: [T, K], optional
        Returns:
            subgraphs: [B, num_subgraphs, D]  — feature của mỗi subgraph
            pool_loss: scalar tensor          — regularization cho assignment
            S: [B, T, K]
        """
        # Ma trận gán [B, T, K]
        logits = self.assign_net(nodes)
        if region_prior is not None:
            prior = region_prior.unsqueeze(0).to(device=nodes.device, dtype=nodes.dtype)
            logits = logits + self.prior_strength * (prior + 1e-8).log()
        else:
            prior = None

        S = F.softmax(logits, dim=-1)

        # Pool: [B, K, T] x [B, T, D] = [B, K, D]
        subgraphs_sum = torch.bmm(S.transpose(1, 2), nodes)
        
        # Chia cho kích thước của subgraph (MEAN pooling) để chuẩn hoá độ lớn feature
        cluster_sizes = S.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8 # [B, K, 1]
        subgraphs = subgraphs_sum / cluster_sizes

        # Entropy loss: khuyến khích assignment dứt khoát
        eps = 1e-8
        entropy = -(S * (S + eps).log()).sum(dim=-1)  # [B, T]
        pool_loss = entropy.mean()

        # Region prior loss: giữ các subgraph bám theo vùng mặt kỳ vọng.
        if prior is not None:
            prior_ce = -(prior * (S + eps).log()).sum(dim=-1).mean()
            pool_loss = pool_loss + self.prior_loss_weight * prior_ce

        # Balance loss: tránh collapse vào một vài vùng.
        mean_assign = S.mean(dim=1)  # [B, K]
        target = torch.full_like(mean_assign, 1.0 / self.num_subgraphs)
        balance = F.mse_loss(mean_assign, target)
        pool_loss = pool_loss + self.balance_loss_weight * balance

        return subgraphs, pool_loss, S


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
        alpha_init = float(model_cfg.get('alpha', 0.5))
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        self.tau           = model_cfg.get('tau', 0.05)
        self.spatial_sigma = model_cfg.get('spatial_sigma', 2.0)
        self.region_prior_strength = model_cfg.get('region_prior_strength', 1.0)
        self.region_prior_loss_weight = model_cfg.get('region_prior_loss_weight', 0.05)
        self.region_balance_loss_weight = model_cfg.get('region_balance_loss_weight', 0.1)

        self.register_buffer(
            'A_spatial',
            precompute_spatial_A(
                self.image_size,
                self.image_size,
                self.window_size,
                self.stride,
                self.spatial_sigma
            )
        )
        self.register_buffer(
            'region_prior',
            precompute_region_prior(
                self.image_size,
                self.image_size,
                self.window_size,
                self.stride,
                self.num_subgraphs
            )
        )

        # patch_dim = window_size * window_size * 8
        # [x, y, intensity, grad_x, grad_y, grad_mag, local_mean, local_std]
        self.node_feature_channels = 8
        patch_dim = self.window_size * self.window_size * self.node_feature_channels

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
        self.subgraph_pool = SubGraphPooling(
            self.hidden_dim,
            self.num_subgraphs,
            prior_strength=self.region_prior_strength,
            prior_loss_weight=self.region_prior_loss_weight,
            balance_loss_weight=self.region_balance_loss_weight
        )

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
              f"hidden={self.hidden_dim} subgraphs={self.num_subgraphs} "
              f"alpha_init={self.alpha_value:.3f} tau={self.tau} sigma={self.spatial_sigma} "
              f"region_prior={self.region_prior_strength}")

    @property
    def alpha_value(self) -> float:
        """
        Trọng số hiện tại của A_spatial trong hybrid adjacency.
        alpha gần 1: tin spatial nhiều hơn; alpha gần 0: tin feature/cosine nhiều hơn.
        """
        return torch.sigmoid(self.alpha_logit).detach().item()

    # ----------------------------------------------------------
    # Step 1: Pixel → Graph nodes
    # ----------------------------------------------------------
    def _pixel_to_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, channels, H, W]
        Returns nodes: [B, T, patch_dim]
        """
        # Nếu grayscale → chuẩn hóa về [0,1] nếu cần (giả sử đã được DataLoader norm)
        feat_map = build_graph_feature_map(x)               # [B, 8, H, W]
        nodes    = extract_patch_nodes(feat_map, self.window_size, self.stride)  # [B, T, patch_dim]

        # F.unfold sắp theo channel-major:
        # [x_patch, y_patch, intensity_patch, grad_x_patch, ...]
        # Giữ tọa độ ở [0, 1], chỉ normalize phần appearance/texture.
        ws2 = self.window_size * self.window_size
        coords = nodes[:, :, :2 * ws2]
        appearance = nodes[:, :, 2 * ws2:]
        appearance = F.normalize(appearance, p=2, dim=-1)
        nodes = torch.cat([coords, appearance], dim=-1)
        return nodes

    # ----------------------------------------------------------
    # Step 2: GCN encoding
    # ----------------------------------------------------------
    def _encode_graph(self, nodes: torch.Tensor, return_adjacency: bool = False):
        """
        nodes: [B, T, patch_dim]
        Returns: [B, T, hidden_dim]
        """
        h = self.input_proj(nodes)   # [B, T, hidden_dim]
        A1, A1_cos = self._build_hybrid_A(h, return_cos=True)
        h = self.gcn1(h, A1)         # [B, T, hidden_dim]

        A2, A2_cos = self._build_hybrid_A(h, return_cos=True)
        h = self.gcn2(h, A2)         # [B, T, hidden_dim]

        if return_adjacency:
            return h, {
                "A1_hybrid": A1,
                "A1_cos": A1_cos,
                "A2_hybrid": A2,
                "A2_cos": A2_cos,
            }
        return h

    def _build_hybrid_A(self, nodes: torch.Tensor, return_cos: bool = False):
        """
        A_hybrid = alpha * A_spatial + (1 - alpha) * A_cos
        A_spatial: cố định theo vị trí patch.
        A_cos: học động theo feature hiện tại của node.
        """
        nodes_norm = F.normalize(nodes, p=2, dim=-1)
        A_cos = torch.bmm(nodes_norm, nodes_norm.transpose(1, 2))
        A_cos = F.softmax(A_cos / self.tau, dim=-1)

        alpha = torch.sigmoid(self.alpha_logit).to(device=nodes.device, dtype=nodes.dtype)
        A_spatial = self.A_spatial.unsqueeze(0).expand(nodes.size(0), -1, -1)
        A = alpha * A_spatial + (1.0 - alpha) * A_cos
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        if return_cos:
            return A, A_cos
        return A

    # ----------------------------------------------------------
    # Step 3: SubGraph pooling
    # ----------------------------------------------------------
    def _pool_subgraphs(self, h: torch.Tensor):
        """
        h: [B, T, hidden_dim]
        Returns:
            subgraphs: [B, num_subgraphs, hidden_dim]
            pool_loss: scalar
            S: [B, T, num_subgraphs]
        """
        return self.subgraph_pool(h, self.region_prior)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor, return_assignments: bool = False):
        """
        Args:
            x: [B, channels, H, W]
            return_assignments: If True, returns (logits, assignments)
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
        subgraphs, pool_loss, S = self._pool_subgraphs(h)  # [B, S, hidden_dim], S: [B, T, K]

        # ── 4. Inter-SubGraph Self-Attention ──
        sg_attn, _ = self.subgraph_attn(subgraphs, subgraphs, subgraphs)
        subgraphs = self.attn_norm(subgraphs + sg_attn)  # residual

        # ── 5. Global pooling over subgraphs ──
        graph_repr = subgraphs.mean(dim=1)         # [B, hidden_dim]

        # ── 6. Classification ──
        logits = self.classifier(graph_repr)       # [B, num_classes]

        if return_assignments:
            return logits, S

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
