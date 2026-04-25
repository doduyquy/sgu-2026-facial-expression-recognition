"""Serializable dataclasses for emotion-specific prototype motif banks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray]


@dataclass
class MotifPrototype:
    """One class-discriminative prototype subgraph descriptor."""

    motif_id: int
    class_id: int
    prototype: ArrayLike
    intra_score: float
    inter_score: float
    discriminative_score: float
    support: int
    exemplar: Optional[dict] = None


@dataclass
class MotifBank:
    """Per-emotion motif bank, friendly to torch.save / torch.load."""

    motifs: Dict[int, List[MotifPrototype]]
    descriptor_dim: int
    num_classes: int
    emotion_names: List[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
