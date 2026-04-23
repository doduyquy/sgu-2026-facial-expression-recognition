from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Cấu hình cho bước graph hóa ảnh FER-2013.
    """

    image_size: int = 48
    connectivity: int = 8
    normalize_pixels: bool = True

    # Baseline node features:
    #   - intensity
    #   - x_norm
    #   - y_norm
    #
    # Có thể mở rộng thêm:
    #   - gx
    #   - gy
    #   - grad_mag
    #   - contrast
    node_features: List[str] = field(default_factory=lambda: [
        "intensity",
        "x_norm",
        "y_norm",
    ])

    # Baseline edge features
    edge_features: List[str] = field(default_factory=lambda: [
        "dx",
        "dy",
        "dist",
        "delta_intensity",
        "intensity_similarity",
    ])

    intensity_similarity_alpha: float = 1.0

    # Có lưu ảnh vào graph cache hay không
    save_image_in_graph: bool = False