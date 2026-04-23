from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class PixelGraph:
    """
    Graph biểu diễn cho 1 ảnh FER-2013.

    Representation chính:
        - node_features: [N, d]
        - edge_index:    [2, M]
        - edge_attr:     [M, e]
    """
    graph_id: int
    label: int
    split: str
    usage: str
    height: int
    width: int
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    image: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)