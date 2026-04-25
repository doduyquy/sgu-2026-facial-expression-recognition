"""Serializable dataclasses for pixel-preserving motifs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class PixelMotifPrototype:
    """Prototype motif with descriptor centroid and pixel-level exemplars."""

    motif_id: int
    class_id: int
    prototype: torch.Tensor
    intra_score: float
    inter_score: float
    discriminative_score: float
    support: int
    exemplars: List[dict] = field(default_factory=list)


@dataclass
class PixelMotifBank:
    """Emotion-specific pixel-preserving motif bank."""

    motifs: Dict[int, List[PixelMotifPrototype]]
    descriptor_dim: int
    num_classes: int
    emotion_names: List[str]
    config: dict = field(default_factory=dict)
