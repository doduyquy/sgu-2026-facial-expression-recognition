from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Cấu hình cho bước graph hóa ảnh FER-2013.
    Được khởi tạo từ YAML config dict (config['graph']).
    """

    image_size: int = 48
    connectivity: int = 8
    normalize_pixels: bool = True

    # Baseline node features:
    #   - intensity, x_norm, y_norm
    # Mở rộng sau: gx, gy, grad_mag, contrast
    node_features: List[str] = field(default_factory=lambda: [
        "intensity",
        "x_norm",
        "y_norm",
    ])

    edge_features: List[str] = field(default_factory=lambda: [
        "dx",
        "dy",
        "dist",
        "delta_intensity",
        "intensity_similarity",
    ])

    intensity_similarity_alpha: float = 1.0
    save_image_in_graph: bool = False

    @staticmethod
    def from_config(config: dict) -> "GraphConfig":
        """
        Khởi tạo GraphConfig từ YAML config dict.
        Dùng config['graph'] section.
        """
        g = config.get("graph", {})
        return GraphConfig(
            image_size=config.get("data", {}).get("image_size", 48),
            connectivity=g.get("connectivity", 8),
            normalize_pixels=g.get("normalize_pixels", True),
            node_features=g.get("node_features", ["intensity", "x_norm", "y_norm"]),
            edge_features=g.get("edge_features", ["dx", "dy", "dist", "delta_intensity", "intensity_similarity"]),
            intensity_similarity_alpha=g.get("intensity_similarity_alpha", 1.0),
            save_image_in_graph=g.get("save_image_in_graph", False),
        )
