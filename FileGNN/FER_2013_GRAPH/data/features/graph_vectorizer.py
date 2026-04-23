import numpy as np


class GraphVectorizer:
    """
    Vectorize 1 PixelGraph thành 1 vector graph-level cố định.

    Baseline hiện tại:
        graph_vec = concat(
            mean(node_features),
            std(node_features),
            max(node_features)
        )

    Nếu node_features có shape [N, d]
    -> graph_vec có shape [3*d]
    """

    def __init__(self, use_mean: bool = True, use_std: bool = True, use_max: bool = True):
        self.use_mean = use_mean
        self.use_std = use_std
        self.use_max = use_max

        if not (use_mean or use_std or use_max):
            raise ValueError("Phải bật ít nhất 1 kiểu pooling.")

    def transform(self, graph) -> np.ndarray:
        """
        graph: PixelGraph
        return: np.ndarray shape [D]
        """
        x = graph.node_features  # [N, d]

        parts = []

        if self.use_mean:
            parts.append(x.mean(axis=0))

        if self.use_std:
            parts.append(x.std(axis=0))

        if self.use_max:
            parts.append(x.max(axis=0))

        graph_vec = np.concatenate(parts, axis=0).astype(np.float32)
        return graph_vec

    def infer_output_dim(self, node_feature_dim: int) -> int:
        n_parts = int(self.use_mean) + int(self.use_std) + int(self.use_max)
        return n_parts * node_feature_dim