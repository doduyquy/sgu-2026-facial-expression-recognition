import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pixel_nodes(images: torch.Tensor) -> torch.Tensor:
    """
    Convert image pixels to graph nodes.

    Moi pixel la 1 node voi feature co ban:
        [row_coord, col_coord, intensity]

    Args:
        images: [B, 1, H, W], thuong da Normalize(mean=0.5, std=0.5)
    Returns:
        nodes: [B, H*W, 3]
    """
    B, _, H, W = images.shape
    device = images.device
    dtype = images.dtype

    intensity = (images[:, :1] * 0.5 + 0.5).clamp(0.0, 1.0)

    rows = torch.linspace(0, 1, H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    cols = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, W).expand(H, W)
    rows = rows.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    cols = cols.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

    feat_map = torch.cat([rows, cols, intensity], dim=1)
    return feat_map.flatten(2).transpose(1, 2).contiguous()


def precompute_8neighbor_graph(H: int, W: int):
    """
    Build fixed 8-neighbor pixel graph.

    A[i, j] > 0 nghia la node i nhan message tu hang xom j.
    neighbor_index[i] gom toi da 8 hang xom cua node i, -1 la padding bien anh.
    """
    N = H * W
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    A = torch.zeros(N, N, dtype=torch.float32)
    neighbor_index = torch.full((N, 8), -1, dtype=torch.long)

    for r in range(H):
        for c in range(W):
            center = r * W + c
            valid = []
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    valid.append(nr * W + nc)

            if valid:
                neighbor_index[center, :len(valid)] = torch.tensor(valid, dtype=torch.long)
                A[center, valid] = 1.0 / len(valid)

    return A, neighbor_index


def extract_delta_motifs_from_image(
    images: torch.Tensor,
    neighbor_index: torch.Tensor,
) -> dict:
    """
    Extract delta-pattern motif cho tung pixel.

    Motif cua 1 pixel gom:
        - center_intensity: [1]
        - neighbor_intensity: [8]
        - delta: neighbor_intensity - center_intensity [8]

    Args:
        images: [B, 1, H, W], thuong da Normalize(mean=0.5, std=0.5)
        neighbor_index: [N, 8], -1 la padding bien anh
    Returns:
        dict:
            motif_vector: [B, N, 17]
            center_intensity: [B, N, 1]
            neighbor_intensity: [B, N, 8]
            delta: [B, N, 8]
            valid_mask: [N, 8]
            pixel_index: [N]
            row_col: [N, 2]
    """
    B, _, H, W = images.shape
    N = H * W
    K = neighbor_index.size(1)
    device = images.device

    intensity = (images[:, :1] * 0.5 + 0.5).clamp(0.0, 1.0).flatten(2).squeeze(1)

    valid = neighbor_index.ge(0).to(device=device)
    safe_index = neighbor_index.clamp_min(0).to(device=device)
    neighbors = intensity[:, safe_index]  # [B, N, 8]
    centers = intensity.unsqueeze(-1).expand(-1, -1, K)

    neighbor_intensity = neighbors * valid.unsqueeze(0).to(dtype=intensity.dtype)
    delta = (neighbors - centers) * valid.unsqueeze(0).to(dtype=intensity.dtype)
    center_intensity = intensity.unsqueeze(-1)
    motif_vector = torch.cat([center_intensity, neighbor_intensity, delta], dim=-1)

    pixel_index = torch.arange(N, device=device)
    row_col = torch.stack([pixel_index // W, pixel_index % W], dim=-1)

    return {
        "motif_vector": motif_vector,
        "center_intensity": center_intensity,
        "neighbor_intensity": neighbor_intensity,
        "delta": delta,
        "valid_mask": valid,
        "pixel_index": pixel_index,
        "row_col": row_col,
    }


class PixelDeltaLayer(nn.Module):
    """
    Message passing theo dung y tuong:
        moi edge center -> neighbor dung delta = neighbor - center

    Message input cua tung hang xom:
        [center_feature, neighbor_feature, neighbor_feature - center_feature]
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.agg_proj = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, nodes: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [B, N, D]
            neighbor_index: [N, 8], -1 tai padding bien anh
        Returns:
            out: [B, N, out_dim]
        """
        B, N, D = nodes.shape
        K = neighbor_index.size(1)

        valid = neighbor_index.ge(0).to(device=nodes.device)
        safe_index = neighbor_index.clamp_min(0).to(device=nodes.device)

        batch_offset = (torch.arange(B, device=nodes.device) * N).view(B, 1, 1)
        flat_index = (safe_index.view(1, N, K) + batch_offset).reshape(-1)

        flat_nodes = nodes.reshape(B * N, D)
        neighbors = flat_nodes.index_select(0, flat_index).view(B, N, K, D)
        centers = nodes.unsqueeze(2).expand(-1, -1, K, -1)
        delta = neighbors - centers

        msg_input = torch.cat([centers, neighbors, delta], dim=-1)
        messages = self.message_net(msg_input)
        messages = messages * valid.view(1, N, K, 1).to(dtype=messages.dtype)

        valid_f = valid.view(1, N, K, 1).to(dtype=messages.dtype)
        degree = valid.sum(dim=-1).clamp_min(1).view(1, N, 1).to(dtype=messages.dtype)

        msg_mean = messages.sum(dim=2) / degree

        masked_for_max = messages.masked_fill(valid_f.eq(0), -torch.finfo(messages.dtype).max)
        msg_max = masked_for_max.max(dim=2).values
        msg_max = torch.where(torch.isfinite(msg_max), msg_max, torch.zeros_like(msg_max))

        msg_second_moment = (messages.pow(2) * valid_f).sum(dim=2) / degree
        msg_var = msg_second_moment - msg_mean.pow(2)
        msg_std = torch.sqrt(msg_var.clamp_min(0.0) + 1e-8)

        agg = self.agg_proj(torch.cat([msg_mean, msg_max, msg_std], dim=-1))

        return self.norm(self.residual(nodes) + agg)


class PixelGNN(nn.Module):
    """
    Pixel-level GNN cho FER:
        Pixel nodes -> 8-neighbor delta message passing -> global pooling -> classifier

    Khong gom patch/subgraph trong model nay. Window/patch pooling da bo theo yeu cau:
    moi pixel van la node goc va layer hoc truc tiep tu delta giua pixel trung tam
    voi 8 pixel hang xom.
    """
    def __init__(self, config, channels: int = 1):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        self.image_size = data_cfg.get("image_size", 48)
        self.num_classes = data_cfg.get("num_classes", 7)
        self.channels = channels
        self.hidden_dim = model_cfg.get("hidden_dim", 128)
        self.num_layers = model_cfg.get("num_layers", 3)
        self.dropout_rate = model_cfg.get("dropout", 0.3)

        A, neighbor_index = precompute_8neighbor_graph(self.image_size, self.image_size)
        self.register_buffer("A_8neighbor", A)
        self.register_buffer("neighbor_index", neighbor_index)

        # Layer dau tinh delta truc tiep tren [row, col, intensity].
        # Cac layer sau tinh delta tren embedding da hoc.
        layer_dims = [3] + [self.hidden_dim] * max(self.num_layers - 1, 0)
        self.layers = nn.ModuleList(
            PixelDeltaLayer(in_dim, self.hidden_dim, self.dropout_rate)
            for in_dim in layer_dims
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        print(
            f"--> PixelGNN | pixel_nodes={self.image_size * self.image_size} "
            f"8-neighbor hidden={self.hidden_dim} layers={self.num_layers}"
        )

    def _pixel_to_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return build_pixel_nodes(x)

    def _encode_graph(self, nodes: torch.Tensor) -> torch.Tensor:
        h = nodes
        for layer in self.layers:
            h = layer(h, self.neighbor_index)
        return h

    def forward(
        self,
        x: torch.Tensor,
        return_nodes: bool = False,
        return_adjacency: bool = False,
        return_delta_motifs: bool = False,
    ):
        """
        Args:
            x: [B, channels, H, W]
            return_nodes: tra them embedding cua tung pixel de visualize motif.
            return_adjacency: tra them ma tran ke 8-neighbor co dinh.
            return_delta_motifs: tra motif local cua toan bo 2304 pixel.
        """
        nodes = self._pixel_to_nodes(x)
        h = self._encode_graph(nodes)

        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        graph_repr = torch.cat([mean_pool, max_pool], dim=-1)
        logits = self.classifier(graph_repr)

        extras = {}
        if return_nodes:
            extras["pixel_embeddings"] = h
        if return_adjacency:
            extras["A_8neighbor"] = self.A_8neighbor
        if return_delta_motifs:
            motifs = extract_delta_motifs_from_image(x, self.neighbor_index)
            motifs["score"] = h.norm(dim=-1)
            extras["delta_motifs"] = motifs

        if extras:
            return logits, extras
        return logits


if __name__ == "__main__":
    print("=== Testing PixelGNN ===")
    config = {
        "data": {"num_classes": 7, "image_size": 48, "channels": 1},
        "model": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.3,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelGNN(config, channels=1).to(device)

    dummy = torch.randn(4, 1, 48, 48).to(device)
    logits = model(dummy)
    print(f"logits: {logits.shape}")
    assert logits.shape == (4, 7)

    logits, extras = model(dummy, return_nodes=True, return_adjacency=True, return_delta_motifs=True)
    print(f"pixel_embeddings: {extras['pixel_embeddings'].shape}")
    print(f"A_8neighbor: {extras['A_8neighbor'].shape}")
    print(f"delta_motifs: {extras['delta_motifs']['motif_vector'].shape}")
    print("Test Passed!")
