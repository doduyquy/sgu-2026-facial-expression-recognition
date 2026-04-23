import numpy as np


class GraphVectorizer:
    """
    Vectorize node_features của 1 graph thành 1 vector graph-level cố định.

    Baseline:
        graph_vec = concat(mean, std, max) trên axis nodes
        node_features shape [N, d] → graph_vec shape [3*d]

    Hỗ trợ cả 2 input types:
        transform()             — nhận PixelGraph cũ (node_features là np.ndarray)
        transform_from_tensor() — nhận torch.Tensor [N, d] từ PixelGraphSample mới
    """

    def __init__(self, use_mean: bool = True, use_std: bool = True, use_max: bool = True):
        self.use_mean = use_mean
        self.use_std = use_std
        self.use_max = use_max

        if not (use_mean or use_std or use_max):
            raise ValueError("Phải bật ít nhất 1 kiểu pooling.")

    def transform(self, graph) -> np.ndarray:
        """
        Nhận PixelGraph cũ (node_features là np.ndarray [N, d]).
        Trả về np.ndarray [D]. Giữ để backward compat.
        """
        x = graph.node_features  # [N, d]  np.ndarray
        parts = []
        if self.use_mean:
            parts.append(x.mean(axis=0))
        if self.use_std:
            parts.append(x.std(axis=0))
        if self.use_max:
            parts.append(x.max(axis=0))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def transform_from_tensor(self, node_features) -> "torch.Tensor":
        """
        Nhận node_features là torch.Tensor [N, d] từ PixelGraphSample.
        Trả về torch.FloatTensor [D].

        Dùng trong GraphVectorDatasetFromRepo (pipeline graph repository mới).
        """
        import torch
        x = node_features  # Tensor [N, d]
        parts = []
        if self.use_mean:
            parts.append(x.mean(dim=0))
        if self.use_std:
            parts.append(x.std(dim=0))
        if self.use_max:
            parts.append(x.max(dim=0).values)
        return torch.cat(parts, dim=0).float()

    def infer_output_dim(self, node_feature_dim: int) -> int:
        n_parts = int(self.use_mean) + int(self.use_std) + int(self.use_max)
        return n_parts * node_feature_dim
